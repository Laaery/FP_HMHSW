#!/usr/bin/env python
# -*- coding:utf-8 -*-
# @Author: LL
# @File：fp_selection_rf.py
"""
Training base models for robust fingerprint selection.
"""
import joblib
import pandas as pd
import numpy as np
from joblib import delayed, Parallel
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score, accuracy_score, f1_score, classification_report, balanced_accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from collections import Counter, defaultdict
from scipy import stats
from tqdm import tqdm
from warnings import filterwarnings
import os

filterwarnings('ignore')

def proportional_stratified_split(subset, label_column, random_state=42):
    """
    Stratified split of a subset of data into train, validation, and test sets.
    """
    np.random.seed(random_state)

    train_idx, val_idx, test_idx = [], [], []
    all_labels = list(subset[label_column].unique())

    label_to_samples = defaultdict(list)
    for idx, row in subset.iterrows():
        label_to_samples[row[label_column]].append(idx)

    for label in all_labels:
        samples = label_to_samples[label]
        if len(samples) == 1:
            # Replicate sample if only one available
            train_idx.append(samples[0])
            val_idx.append(samples[0])
            test_idx.append(samples[0])
        elif len(samples) == 2:
            # Replicate sample if only two available
            chosen = np.random.choice(samples, 2, replace=False)
            train_idx.append(chosen[0])
            val_idx.append(chosen[1])
            test_idx.append(chosen[1])
        else:
            # At least 3 samples, split into train/val/test
            chosen = np.random.choice(samples, 3, replace=False)
            train_idx.append(chosen[0])
            val_idx.append(chosen[1])
            test_idx.append(chosen[2])

        # Remove allocated samples from the pool
        label_to_samples[label] = [s for s in samples if s not in train_idx + val_idx + test_idx]

    allocated_samples = set(train_idx + val_idx + test_idx)

    # Assign remaining samples to train/val/test according to the Index
    for idx in subset['Index'].unique():
        local_df = subset[(subset['Index'] == idx) & (~subset.index.isin(allocated_samples))]
        if len(local_df) == 0:
            continue

        local_indices = local_df.index.tolist()
        local_labels = local_df[label_column].tolist()
        n = len(local_indices)

        if n == 1:
            # One for training
            train_idx.append(local_indices[0])

        elif n == 2:
            # Two for training and validation
            train_idx.append(local_indices[0])
            val_idx.append(local_indices[1])

        elif n == 3:
            train_idx.append(local_indices[0])
            val_idx.append(local_indices[1])
            test_idx.append(local_indices[2])

        else:
            try:
                train_part, temp_indices, train_labels, temp_labels = train_test_split(
                    local_indices, local_labels,
                    test_size=0.4, random_state=random_state,
                    stratify=local_labels
                )
                val_part, test_part, _, _ = train_test_split(
                    temp_indices, temp_labels,
                    test_size=0.5, random_state=random_state,
                    stratify=temp_labels
                )
            except ValueError:
                train_part, temp = train_test_split(local_indices, test_size=0.4, random_state=random_state)
                val_part, test_part = train_test_split(temp, test_size=0.5, random_state=random_state)

            train_idx.extend(train_part)
            val_idx.extend(val_part)
            test_idx.extend(test_part)

    return train_idx, val_idx, test_idx


def final_verification(subset, train_idx, val_idx, test_idx, label_column):
    """
    Inspect final splits to ensure all labels are covered.
    """
    print("=== Final verification ===")
    all_labels = set(subset[label_column].unique())

    def check_and_fix(split_name, split_idx):
        present_labels = set(subset.loc[split_idx, label_column].unique()) if split_idx else set()
        missing_labels = all_labels - present_labels
        if missing_labels:
            print(f"{split_name} missed: {missing_labels}")
            for label in missing_labels:
                candidates = subset[subset[label_column] == label].index.tolist()
                for c in candidates:
                    if c not in train_idx and c not in val_idx and c not in test_idx:
                        split_idx.append(c)
                        break
        else:
            print(f"{split_name}: all labels covered ({len(present_labels)}/{len(all_labels)})")

    check_and_fix("Train", train_idx)
    check_and_fix("Val", val_idx)
    check_and_fix("Test", test_idx)

    train_labels = set(subset.loc[train_idx, label_column].unique())
    val_labels = set(subset.loc[val_idx, label_column].unique())
    test_labels = set(subset.loc[test_idx, label_column].unique())

    if all_labels <= train_labels and all_labels <= val_labels and all_labels <= test_labels:
        print("\n✓ Perfect split! All labels covered in train, val, and test sets.")
        return True
    else:
        print("\n✗ Still missing labels after fix.")
        return False


def RFE_Customized(X_train, y_train, X_val, y_val, feature_cols, random_seed=42, repeat_idx=0):
    """
    Recursive Feature Elimination (RFE) guided by RandomForest feature importance.

    Parameters:
    -----------
    X_train : pd.DataFrame or np.ndarray
        Training feature matrix.
    y_train : pd.Series or np.ndarray
        Training labels.
    X_val : pd.DataFrame or np.ndarray
        Validation feature matrix.
    y_val : pd.Series or np.ndarray
        Validation labels.
    feature_cols : list
        List of feature names (if using DataFrame) or column indices.
    random_seed : int, default=42
        Random seed for reproducibility in RandomForest.
    repeat_idx : int, default=0
        Index of the current repetition (for tracking in loops).

    Returns:
    --------
    performance_history : list of dict
        History of performance metrics at each feature count.
    feature_elimination_order : list
        List of features ordered by elimination (last = most important).
    """
    current_features = feature_cols.copy()
    performance_history = []
    feature_elimination_order = []

    while len(current_features) >= 1:
        print(f"Number of features: {len(current_features)}")

        # Train RandomForest model
        model = RandomForestClassifier(n_estimators=100, random_state=random_seed, class_weight='balanced', max_depth=25, min_samples_leaf=5, n_jobs=1)
        model.fit(X_train[current_features], y_train)

        # Predict on training set
        y_train_pred = model.predict(X_train[current_features])
        train_acc = accuracy_score(y_train, y_train_pred)
        train_acc_balanced = balanced_accuracy_score(y_train, y_train_pred)

        # Predict on validation set
        y_val_pred = model.predict(X_val[current_features])
        val_acc = accuracy_score(y_val, y_val_pred)
        val_acc_balanced = balanced_accuracy_score(y_val, y_val_pred)

        try:
            y_val_pred_proba = model.predict_proba(X_val[current_features])
            val_auc = roc_auc_score(y_val, y_val_pred_proba, multi_class='ovr', average='macro')
            y_train_pred_proba = model.predict_proba(X_train[current_features])
            train_auc = roc_auc_score(y_train, y_train_pred_proba, multi_class='ovr', average='macro')
        except ValueError as e:
            print(f"Warning: Cannot compute AUC (e.g., single class in pred): {e}")
            val_auc = 0.0
            train_auc = 0.0

        try:
            train_f1_macro = f1_score(y_train, y_train_pred, average='macro')
            train_f1_weighted = f1_score(y_train, y_train_pred, average='weighted')
            val_f1_macro = f1_score(y_val, y_val_pred, average='macro')
            val_f1_weighted = f1_score(y_val, y_val_pred, average='weighted')
        except Exception as e:
            print(f"Warning: Error computing F1-score: {e}")
            train_f1_macro = 0.0
            train_f1_weighted = 0.0
            val_f1_macro = 0.0
            val_f1_weighted = 0.0

        # Record performance
        performance_history.append({
            'repeat': repeat_idx + 1,
            'seed': random_seed,
            'n_features': len(current_features),
            'train_acc': train_acc,
            'train_acc_balanced': train_acc_balanced,
            'train_auc': train_auc,
            'train_f1_macro': train_f1_macro,
            'train_f1_weighted': train_f1_weighted,
            'val_acc': val_acc,
            'val_acc_balanced': val_acc_balanced,
            'val_auc': val_auc,
            'val_f1_macro': val_f1_macro,
            'val_f1_weighted': val_f1_weighted
        })
        if len(current_features) == 1:
            break

        # Get feature importance and remove the least important
        importance = model.feature_importances_
        feat_imp = pd.Series(importance, index=current_features)
        worst_feature = feat_imp.idxmin()
        feature_elimination_order.append(worst_feature)
        current_features.remove(worst_feature)

    return performance_history, feature_elimination_order


def _run_single_repeat(
    scenario_name,
    config,
    repeat_idx,
    random_seed,
    df,
    X_full,
    y_full,
    feature_cols,
    output_dir,
    train_with_robust_fingerprint,
    robust_features=None
):
    """
        Run a single repetition of the RFE process for a given scenario.
    """
    mask = pd.Series(False, index=df.index)
    for pt, fc in config['include']:
        if np.isnan(fc):
            condition = (df['perturbation_type'] == pt) & (df['flip_count'].isna())
        else:
            condition = (df['perturbation_type'] == pt) & (df['flip_count'] == fc)
        mask |= condition

    subset = df[mask].copy()
    try:
        train_idx, val_idx, test_idx = proportional_stratified_split(
            subset, label_column='Source', random_state=random_seed
        )
        if not final_verification(subset, train_idx, val_idx, test_idx, label_column='Source'):
            return None
    except Exception as e:
        print(f"[{scenario_name}-seed{random_seed}] Split failed: {e}")
        return None

    X_train = X_full.loc[train_idx]
    X_val = X_full.loc[val_idx]
    X_test = X_full.loc[test_idx]
    y_train = y_full.loc[train_idx]
    y_val = y_full.loc[val_idx]
    y_test = y_full.loc[test_idx]

    split_dir = f"../output/{output_dir}/data_splits"
    os.makedirs(split_dir, exist_ok=True)
    joblib.dump({
        'train_idx': train_idx,
        'val_idx': val_idx,
        'test_idx': test_idx
    }, f"{split_dir}/{scenario_name}_seed_{random_seed}.pkl")

    if not train_with_robust_fingerprint:
        # RFE
        performance_history, elimination_order = RFE_Customized(
            X_train, y_train, X_val, y_val, feature_cols,
            random_seed=42,
            repeat_idx=repeat_idx
        )
        perf_df = pd.DataFrame(performance_history)
        perf_df.to_csv(
            f"../output/{output_dir}/performance_history_{scenario_name}_repeat_{repeat_idx+1}_seed_{random_seed}.csv",
            index=False
        )
        best_idx = perf_df['val_acc_balanced'].idxmax()
        best_n = int(perf_df.loc[best_idx, 'n_features'])
        best_features = feature_cols.copy()
        for i in range(len(feature_cols) - best_n):
            if i < len(elimination_order):
                best_features.remove(elimination_order[i])
        best_row = perf_df.loc[best_idx]
    else:
        best_features = robust_features
        best_n = len(best_features)
        model = RandomForestClassifier(
            n_estimators=100, random_state=42, class_weight='balanced',
            max_depth=25, min_samples_leaf=5, n_jobs=1
        )
        model.fit(X_train[best_features], y_train)

        y_train_pred = model.predict(X_train[best_features])
        y_val_pred = model.predict(X_val[best_features])
        y_train_proba = model.predict_proba(X_train[best_features])
        y_val_proba = model.predict_proba(X_val[best_features])

        best_row = {
            'train_acc_balanced': balanced_accuracy_score(y_train, y_train_pred),
            'train_f1_macro': f1_score(y_train, y_train_pred, average='macro'),
            'train_f1_weighted': f1_score(y_train, y_train_pred, average='weighted'),
            'train_auc': roc_auc_score(y_train, y_train_proba, multi_class='ovr', average='macro'),
            'val_acc_balanced': balanced_accuracy_score(y_val, y_val_pred),
            'val_f1_macro': f1_score(y_val, y_val_pred, average='macro'),
            'val_f1_weighted': f1_score(y_val, y_val_pred, average='weighted'),
            'val_auc': roc_auc_score(y_val, y_val_proba, multi_class='ovr', average='macro'),
        }

    # Final model on test set
    final_model = RandomForestClassifier(
        n_estimators=100, random_state=42, class_weight='balanced',
        max_depth=25, min_samples_leaf=5, n_jobs=1
    )
    final_model.fit(X_train[best_features], y_train)

    try:
        y_test_pred = final_model.predict(X_test[best_features])
        y_test_proba = final_model.predict_proba(X_test[best_features])
        test_metrics = {
            'test_auc': roc_auc_score(y_test, y_test_proba, multi_class='ovr', average='macro'),
            'test_f1_macro': f1_score(y_test, y_test_pred, average='macro'),
            'test_f1_weighted': f1_score(y_test, y_test_pred, average='weighted'),
            'test_acc': accuracy_score(y_test, y_test_pred),
            'test_acc_balanced': balanced_accuracy_score(y_test, y_test_pred)
        }
    except Exception as e:
        test_metrics = {
            'test_auc': 0.0,
            'test_f1_macro': 0.0,
            'test_f1_weighted': 0.0,
            'test_acc': 0.0,
            'test_acc_balanced': 0.0
        }

    result = {
        'scenario': scenario_name,
        'repeat': repeat_idx + 1,
        'seed': random_seed,
        'best_n_features': best_n,
        'best_features': ', '.join(best_features),
        'best_train_acc_balanced': best_row['train_acc_balanced'],
        'best_train_f1_macro': best_row['train_f1_macro'],
        'best_train_f1_weighted': best_row['train_f1_weighted'],
        'best_train_auc': best_row['train_auc'],
        'best_val_acc_balanced': best_row['val_acc_balanced'],
        'best_val_f1_macro': best_row['val_f1_macro'],
        'best_val_f1_weighted': best_row['val_f1_weighted'],
        'best_val_auc': best_row['val_auc'],
        **test_metrics
    }
    return result


def run_scenario_experiments_full_parallel(
    df,
    X_full,
    y_full,
    feature_cols,
    scenarios,
    output_dir,
    random_seeds=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
    train_with_robust_fingerprint=False,
    robust_fingerprint_path=None,
    n_jobs=48
):
    """
    Runs the full experiment for each scenario in parallel.
    """
    os.makedirs(f"../output/{output_dir}", exist_ok=True)
    os.makedirs(f"../output/{output_dir}/data_splits", exist_ok=True)

    robust_features = None
    if train_with_robust_fingerprint:
        robust_df = pd.read_csv(robust_fingerprint_path)
        col = 'feature' if 'feature' in robust_df.columns else robust_df.columns[0]
        robust_features = robust_df[col].dropna().tolist()
        robust_features = [f for f in robust_features if f in feature_cols]
        if not robust_features:
            raise ValueError("No valid robust features!")

    tasks = []
    for name, config in scenarios.items():
        for repeat_idx, seed in enumerate(random_seeds):
            tasks.append((name, config, repeat_idx, seed))

    print(f"Total tasks: {len(tasks)} (scenarios × repeats)")
    print(f"Running with n_jobs={n_jobs} (max 48 cores)")

    results = Parallel(
        n_jobs=n_jobs,
        backend='loky',
        verbose=10,
        batch_size=1
    )(
        delayed(_run_single_repeat)(
            name, config, repeat_idx, seed,
            df, X_full, y_full, feature_cols,
            output_dir,
            train_with_robust_fingerprint,
            robust_features
        )
        for (name, config, repeat_idx, seed) in tasks
    )

    valid_results = [r for r in results if r is not None]
    print(f"Completed {len(valid_results)} / {len(tasks)} tasks successfully.")

    if not valid_results:
        raise RuntimeError("No valid results returned!")

    df_all = pd.DataFrame(valid_results)
    df_all.to_csv(f"../output/{output_dir}/all_repeat_results.csv", index=False)


    summary_records = []
    for scenario in df_all['scenario'].unique():
        sub = df_all[df_all['scenario'] == scenario]
        record = {
            'scenario': scenario,
            'n_repeats': len(sub),
            'avg_best_n_features': sub['best_n_features'].mean(),
        }
        metrics = [
            'test_auc', 'test_f1_macro', 'test_f1_weighted', 'test_acc_balanced',
            'best_train_auc', 'best_train_f1_macro', 'best_train_f1_weighted', 'best_train_acc_balanced',
            'best_val_auc', 'best_val_f1_macro', 'best_val_f1_weighted', 'best_val_acc_balanced'
        ]
        for m in metrics:
            record[f'avg_{m}'] = sub[m].mean()
            record[f'std_{m}'] = sub[m].std()
        summary_records.append(record)

    summary_df = pd.DataFrame(summary_records)
    summary_df.to_csv(f"../output/{output_dir}/summary_results.csv", index=False)

    for name, group in df_all.groupby('scenario'):
        group.to_csv(f"../output/{output_dir}/repeat_detailed_results_{name}.csv", index=False)

    return summary_df, df_all


if __name__ == '__main__':
    # -----------------------------
    # 1. Load data
    # -----------------------------
    DATA_PATH = '../data/phase_vector_fn_fp_mingle_fn_fp_1_2.csv'
    print(f"Loading data from {DATA_PATH} ...")
    df = pd.read_csv(DATA_PATH)

    # Define columns
    metadata_cols = ['Index', 'perturbation_type', 'flip_count', 'Source']
    feature_cols = [col for col in df.columns if col not in metadata_cols]
    X_full = df[feature_cols].copy()
    y_full = df['Source'].copy()
    # Encode labels
    le = LabelEncoder()
    y_full = pd.Series(le.fit_transform(y_full), index=y_full.index)
    print("labels:", dict(zip(le.classes_, le.transform(le.classes_))))

    print(f"Features: {len(feature_cols)}")
    print(f"Total samples: {len(df)}, Unique Index: {df['Index'].nunique()}")

    scenarios = {
        'S1': {
            'include': [
                ('original', np.nan),
                ('false_negative', 1.0),
                ('false_negative', 2.0)
            ]
        },
        'S2': {
            'include': [
                ('original', np.nan),
                ('false_positive', 1.0),
                ('false_positive', 2.0)
            ]
        },
        'S3': {
            'include': [
                ('original', np.nan),
                ('false_negative', 1.0),
                ('false_positive', 1.0)
            ]
        },
        'S4': {
            'include': [
                ('original', np.nan),
                ('false_negative', 1.0),
                ('false_negative', 2.0),
                ('false_positive', 1.0),
                ('false_positive', 2.0)
            ]
        },
        'S5': {
            'include': [
                ('original', np.nan),
                ('mingle_only', 0)
            ]
        },
        'S6': {
            'include': [
                ('original', np.nan),
                ('mingle_only', 0),
                ('fn_after_mingle', 1.0),
                ('fp_after_mingle', 1.0),
                ('fn_single_direct', 1.0),
                ('fp_single_direct', 1.0)
            ]
        },
        'S7': {
            'include': [
                ('original', np.nan),
                ('mingle_only', 0),
                ('fn_after_mingle', 1.0),
                ('fn_after_mingle', 2.0),
                ('fp_after_mingle', 1.0),
                ('fp_after_mingle', 2.0),
                ('fn_single_direct', 1.0),
                ('fn_single_direct', 2.0),
                ('fp_single_direct', 1.0),
                ('fp_single_direct', 2.0)
            ]
        }
    }

    # -----------------------------
    # 3. Train and evaluate models on each scenario
    # -----------------------------
    input_dir = 'fn_fp_mingle_fn_fp_bacc_balanced_250909'
    output_dir = 'fn_fp_mingle_fn_fp_bacc_balanced_250930'
    results = []
    all_optimal_features = []
    # Two mode: train with robust fingerprint or not
    summary, all_results = run_scenario_experiments_full_parallel(
        df=df,
        X_full=X_full,
        y_full=y_full,
        feature_cols=feature_cols,
        scenarios=scenarios,
        output_dir=output_dir,
        random_seeds=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
        train_with_robust_fingerprint=False,
        n_jobs=48
    )
