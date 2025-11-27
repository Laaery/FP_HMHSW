#!/usr/bin/env python
# -*- coding:utf-8 -*-
# @Author: LL
# @File：noise_injection.py
"""
Inject noise into the phase vector to improve model robustness.
"""
from typing import Union, List, Literal
import os
import pandas as pd
import numpy as np
import itertools


def flip_fingerprint(
    X: pd.DataFrame,
    y: pd.Series,
    false_negative_generation: Union[int, List[int], None] = None,
    false_positive_generation: Union[int, List[int], None] = None,
    max_per_flip_type: int = -1,
    global_candidate_k: int = 3
) -> tuple[pd.DataFrame, pd.Series]:
    """
    Augment binary fingerprint data with perturbation metadata, keeping only 'Index' from metadata.

    Output DataFrame includes:
        - 'Index': preserved from original (same for all aug samples from same source)
        - fingerprint features (from column 5 onward)
        - 'perturbation_type': 'original', 'false_negative', or 'false_positive'
        - 'flip_count': number of bits flipped (NaN for original)

    Args:
        X: DataFrame with first 4 columns as metadata (including 'Index'), rest as binary features.
        y: Class labels.
        false_negative_generation: int or List[int], number(s) of 1→0 flips.
        false_positive_generation: int or List[int], number(s) of 0→1 flips.
        max_per_flip_type: Max number of augmented samples per flip order (-1 for all).
        global_candidate_k: Number of top global tracers to use for small classes.

    Returns:
        X_new: DataFrame with Index, features, perturbation_type, flip_count.
        y_new: Labels for all samples.
    """
    # === 1. Extract Index and features ===
    index_col = X.columns[0]  # 'Index'
    feature_cols = X.columns[1:].tolist()

    # Extract data
    indices = X[index_col].reset_index(drop=True)
    X_feat = X[feature_cols].reset_index(drop=True)
    y = y.reset_index(drop=True)

    # Filter classes by size
    class_counts = y.value_counts()
    mask_valid = y.isin(class_counts.index)
    indices = indices[mask_valid].reset_index(drop=True)
    X_feat = X_feat[mask_valid].reset_index(drop=True)
    y = y[mask_valid].reset_index(drop=True)

    # Normalize generation args
    fn_nums = (
        [false_negative_generation]
        if isinstance(false_negative_generation, int)
        else false_negative_generation or []
    )
    fp_nums = (
        [false_positive_generation]
        if isinstance(false_positive_generation, int)
        else false_positive_generation or []
    )

    new_rows = []
    new_labels = []

    # Global frequent features for FP candidates
    global_freq = X_feat.mean(axis=0).nlargest(global_candidate_k)
    global_candidate_positions = [X_feat.columns.get_loc(col) for col in global_freq.index]

    # Group by class
    grouped_indices = y.groupby(y).groups

    # === 2. Process each sample ===
    for i in range(len(X_feat)):
        row = X_feat.iloc[i]
        orig_index = indices.iloc[i]
        label = y.iloc[i]

        # -----------------------------
        # 1. False Negatives: 1 → 0
        # -----------------------------
        if fn_nums:
            ones_idx = np.where(row.values == 1)[0]
            for k in fn_nums:
                if len(ones_idx) < k:
                    continue
                combos = list(itertools.combinations(ones_idx, k))
                selected = combos if max_per_flip_type == -1 else combos[:max_per_flip_type]
                for flip_indices in selected:
                    new_feat = row.copy()
                    new_feat.iloc[list(flip_indices)] = 0
                    new_row = {
                        'Index': orig_index,
                        **new_feat.to_dict(),
                        'perturbation_type': 'false_negative',
                        'flip_count': k
                    }
                    new_rows.append(new_row)
                    new_labels.append(label)

        # -----------------------------
        # 2. False Positives: 0 → 1
        # -----------------------------
        if fp_nums:
            zeros_idx = np.where(row.values == 0)[0]
            if len(zeros_idx) == 0:
                continue

            # Select candidate positions
            if label in grouped_indices:
                same_class_mask = [idx != i and y[idx] == label for idx in range(len(y))]
                if sum(same_class_mask) > 0:
                    freq_in_class = X_feat[same_class_mask].mean(axis=0)
                    candidate_scores = freq_in_class.iloc[zeros_idx]
                    top_k = min(global_candidate_k, len(candidate_scores))
                    top_candidates = candidate_scores.nlargest(top_k).index
                    candidate_positions = [X_feat.columns.get_loc(col) for col in top_candidates]
                else:
                    candidate_positions = global_candidate_positions
            else:
                candidate_positions = global_candidate_positions

            valid_fp_positions = [pos for pos in candidate_positions if pos in zeros_idx]

            for k in fp_nums:
                if len(valid_fp_positions) < k:
                    continue
                combos = list(itertools.combinations(valid_fp_positions, k))
                selected = combos if max_per_flip_type == -1 else combos[:max_per_flip_type]
                for flip_indices in selected:
                    new_feat = row.copy()
                    new_feat.iloc[list(flip_indices)] = 1
                    new_row = {
                        'Index': orig_index,
                        **new_feat.to_dict(),
                        'perturbation_type': 'false_positive',
                        'flip_count': k
                    }
                    new_rows.append(new_row)
                    new_labels.append(label)

    # === 3. Create augmented data ===
    if new_rows:
        X_aug = pd.DataFrame(new_rows)
        y_aug = pd.Series(new_labels, name=y.name)
    else:
        X_aug = pd.DataFrame(columns=['Index'] + feature_cols + ['perturbation_type', 'flip_count'])
        y_aug = pd.Series([], dtype=y.dtype, name=y.name)

    # === 4. Original data with minimal metadata ===
    X_orig = pd.DataFrame({
        'Index': indices,
        **X_feat.to_dict('series'),
        'perturbation_type': 'original',
        'flip_count': np.nan
    })

    # === 5. Combine ===
    X_new = pd.concat([X_orig, X_aug], ignore_index=True)
    y_new = pd.concat([y, y_aug], ignore_index=True)

    return X_new, y_new


def mingle_and_flip_fingerprint(
    X: pd.DataFrame,
    y: pd.Series,
    post_mingle_mode: Literal['none', 'fn', 'fp', 'both'] = 'both',
    flip_nums: Union[int, List[int]] = 1,
    n_augments: int = 1,
    global_candidate_k: int = 3
) -> tuple[pd.DataFrame, pd.Series]:
    """
    Perform intra-class sample mingling and optional perturbation, preserving only 'Index'.

    Steps:
    1. Use first column as 'Index', features from column 5 onward.
    2. Filter classes with fewer than `min_class_size` samples.
    3. For classes with >=2 samples: mingle (logical OR), then apply FN/FP.
    4. For classes with 1 sample: skip mingle, directly apply FN/FP.
    5. All augmented samples inherit the original sample's 'Index'.
    6. Return only 'Index', features, perturbation_type, flip_count.

    Args:
        X: DataFrame with first column as 'Index', next 3 as metadata (ignored), features from col 5.
        y: Class labels.
        min_class_size: Minimum number of samples per class to retain.
        post_mingle_mode: Perturbation to apply after mingling or on single:
            - 'none': only return mingled/single samples
            - 'fn': apply false negative (1→0) flips
            - 'fp': apply false positive (0→1) flips
            - 'both': generate both
        flip_nums: Number(s) of bits to flip.
        n_augments: Max number of augmentations per flip order (-1 for all combinations).
        global_candidate_k: Number of top global frequent features for FP candidate selection.

    Returns:
        X_new: DataFrame with columns [Index, features..., perturbation_type, flip_count]
        y_new: Labels for all samples.
    """
    if isinstance(flip_nums, int):
        flip_nums = [flip_nums]

    # === 1. Extract Index and features ===
    index_col = X.columns[0]
    feature_cols = X.columns[1:].tolist()

    indices = X[index_col].reset_index(drop=True)
    X_feat = X[feature_cols].reset_index(drop=True)
    y = y.reset_index(drop=True)

    # Filter classes by size
    class_counts = y.value_counts()
    mask_valid = y.isin(class_counts.index)
    indices = indices[mask_valid].reset_index(drop=True)
    X_feat = X_feat[mask_valid].reset_index(drop=True)
    y = y[mask_valid].reset_index(drop=True)

    # Global candidates for FP generation
    global_freq = X_feat.mean(axis=0).nlargest(global_candidate_k)
    global_candidate_positions = [X_feat.columns.get_loc(col) for col in global_freq.index]

    # Storage for results
    all_rows = []  # List of dicts
    all_labels = []

    # Add original samples
    for i in range(len(X_feat)):
        all_rows.append({
            'Index': indices.iloc[i],
            **X_feat.iloc[i].to_dict(),
            'perturbation_type': 'original',
            'flip_count': np.nan
        })
        all_labels.append(y.iloc[i])

    # Group by class
    grouped = pd.Series(range(len(y)), index=y).groupby(level=0)

    for label in class_counts.index:
        class_mask = y == label
        class_indices = np.where(class_mask)[0]

        if len(class_indices) >= 2:
            # --- Major class: mingle pairs ---
            for i_idx, j_idx in itertools.combinations(class_indices, 2):
                x1 = X_feat.iloc[i_idx].values
                x2 = X_feat.iloc[j_idx].values
                mingled = np.clip(x1 + x2, 0, 1)  # Logical OR
                orig_index = indices.iloc[i_idx]  # Inherit Index from first sample (or use any)

                # Create mingled feature dict
                row_dict = {
                    'Index': orig_index,
                    **{col: val for col, val in zip(feature_cols, mingled)},
                }

                # 1. Mingle only
                if post_mingle_mode in ['none', 'both']:
                    all_rows.append({
                        **row_dict,
                        'perturbation_type': 'mingle_only',
                        'flip_count': 0
                    })
                    all_labels.append(label)

                # 2. FN on mingled
                if post_mingle_mode in ['fn', 'both']:
                    ones_idx = np.where(mingled == 1)[0]
                    for k in flip_nums:
                        if len(ones_idx) < k:
                            continue
                        combos = list(itertools.combinations(ones_idx, k))
                        selected = combos if n_augments == -1 else combos[:n_augments]
                        for flip_idx in selected:
                            new_vals = mingled.copy()
                            new_vals[list(flip_idx)] = 0
                            all_rows.append({
                                'Index': orig_index,
                                **{col: val for col, val in zip(feature_cols, new_vals)},
                                'perturbation_type': 'fn_after_mingle',
                                'flip_count': k
                            })
                            all_labels.append(label)

                # 3. FP on mingled
                if post_mingle_mode in ['fp', 'both']:
                    zeros_idx = np.where(mingled == 0)[0]
                    valid_fp_pos = [p for p in global_candidate_positions if p in zeros_idx]
                    for k in flip_nums:
                        if len(valid_fp_pos) < k:
                            continue
                        combos = list(itertools.combinations(valid_fp_pos, k))
                        selected = combos if n_augments == -1 else combos[:n_augments]
                        for flip_idx in selected:
                            new_vals = mingled.copy()
                            new_vals[list(flip_idx)] = 1
                            all_rows.append({
                                'Index': orig_index,
                                **{col: val for col, val in zip(feature_cols, new_vals)},
                                'perturbation_type': 'fp_after_mingle',
                                'flip_count': k
                            })
                            all_labels.append(label)

        else:
            # --- Minor class: single sample ---
            i_idx = class_indices[0]
            row = X_feat.iloc[i_idx]
            orig_index = indices.iloc[i_idx]
            base_dict = {
                'Index': orig_index,
                **row.to_dict()
            }

            # Unperturbed single
            if post_mingle_mode == 'none':
                all_rows.append({
                    **base_dict,
                    'perturbation_type': 'single_no_mingle',
                    'flip_count': 0
                })
                all_labels.append(y.iloc[i_idx])

            # FN on single
            if post_mingle_mode in ['fn', 'both']:
                ones_idx = np.where(row.values == 1)[0]
                for k in flip_nums:
                    if len(ones_idx) < k:
                        continue
                    combos = list(itertools.combinations(ones_idx, k))
                    selected = combos if n_augments == -1 else combos[:n_augments]
                    for flip_idx in selected:
                        new_vals = row.copy()
                        new_vals.iloc[list(flip_idx)] = 0
                        all_rows.append({
                            'Index': orig_index,
                            **new_vals.to_dict(),
                            'perturbation_type': 'fn_single_direct',
                            'flip_count': k
                        })
                        all_labels.append(y.iloc[i_idx])

            # FP on single
            if post_mingle_mode in ['fp', 'both']:
                zeros_idx = np.where(row.values == 0)[0]
                valid_fp_pos = [p for p in global_candidate_positions if p in zeros_idx]
                for k in flip_nums:
                    if len(valid_fp_pos) < k:
                        continue
                    combos = list(itertools.combinations(valid_fp_pos, k))
                    selected = combos if n_augments == -1 else combos[:n_augments]
                    for flip_idx in selected:
                        new_vals = row.copy()
                        new_vals.iloc[list(flip_idx)] = 1
                        all_rows.append({
                            'Index': orig_index,
                            **new_vals.to_dict(),
                            'perturbation_type': 'fp_single_direct',
                            'flip_count': k
                        })
                        all_labels.append(y.iloc[i_idx])

    # === 4. Build final DataFrame ===
    X_new = pd.DataFrame(all_rows)
    y_new = pd.Series(all_labels, name=y.name)

    # Ensure column order
    col_order = ['Index'] + feature_cols + ['perturbation_type', 'flip_count']
    X_new = X_new[col_order]

    return X_new, y_new


if __name__ == "__main__":
    df = pd.read_csv("../data/phase_vector.csv")
    X = df.drop(columns=['Source'])
    y = df['Source']

    # False negatives and positives with different flip counts
    X_new, y_new = flip_fingerprint(
        X,
        y,
        false_negative_generation=[1, 2],
        false_positive_generation=[1, 2],
        max_per_flip_type=-1,
        global_candidate_k=3
    )
    print("Original shape:", X.shape, y.shape)
    print("New shape:", X_new.shape, y_new.shape)
    print("New labels:", y_new.value_counts())

    X_new['Source'] = y_new.values
    X_new.to_csv("../data/phase_vector_fn_fp_1_2.csv", index=False)

    # Mingle and flip
    X_new, y_new = mingle_and_flip_fingerprint(
        X=X,
        y=y,
        post_mingle_mode='both',  # or 'fn', 'fp', 'none'
        flip_nums=[1, 2],
        n_augments=-1,
        global_candidate_k=3
    )
    print("After mingle and flip:")
    print("Original shape:", X.shape, y.shape)
    print("New shape:", X_new.shape, y_new.shape)
    print("New labels:", y_new.value_counts())
    X_new['Source'] = y_new.values
    X_new.to_csv("../data/phase_vector_mingle_fn_fp_1_2.csv", index=False)

    df1 = pd.read_csv("../data/phase_vector_fn_fp_1_2.csv")
    df2 = pd.read_csv("../data/phase_vector_mingle_fn_fp_1_2.csv")

    df_combined = pd.concat([df1, df2], ignore_index=True)
    df_unique = df_combined.drop_duplicates()

    output_path = "../data/phase_vector_fn_fp_mingle_fn_fp_1_2.csv"
    df_unique.to_csv(output_path, index=False)
    print(f"Combined and unique data saved to {output_path}")
