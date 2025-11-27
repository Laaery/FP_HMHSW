#!/usr/bin/env python
# -*- coding:utf-8 -*-
# @Author: LL
# @File：calculate_iid.py
import os
import re

import joblib
import pandas as pd
import numpy as np

# ==========Configuration==========
SPLIT_DIR = r'..\output\fn_fp_mingle_fn_fp_bacc_balanced\data_splits'
DATA_PATH = r'..\data\phase_vector_fn_fp_mingle_fn_fp_1_2.csv'
OUTPUT_CSV = r'..\output\fn_fp_mingle_fn_fp_bacc_balanced\iid\feature_source_distribution_shift_by_scenario.csv'
OUTPUT_DIST_CSV = r'..\output\fn_fp_mingle_fn_fp_bacc_balanced\iid\source_distribution_by_scenario.csv'
os.makedirs(os.path.dirname(OUTPUT_CSV), exist_ok=True)
# ========== Scenarios setting==========
SCENARIOS = {
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

# ========== Loading ==========
df = pd.read_csv(DATA_PATH)
print(f"Shape of data: {df.shape}")

df['flip_count'] = pd.to_numeric(df['flip_count'], errors='coerce')

meta_cols = ['perturbation_type', 'flip_count', 'Source', 'Index']
feature_cols = [col for col in df.columns if col not in meta_cols]
print(f" Number of features: {len(feature_cols)} ")


# ========== Define scenario ==========
def get_scenario_mask(df, scenario_config):
    mask = pd.Series(False, index=df.index)
    for (ptype, fcount) in scenario_config['include']:
        if pd.isna(fcount):
            condition = (df['perturbation_type'] == ptype) & (df['flip_count'].isna())
        else:
            condition = (df['perturbation_type'] == ptype) & (np.isclose(df['flip_count'], fcount, atol=1e-6))
        mask = mask | condition
    return mask


# ========== Define metrics ==========
def binary_psi(p1, p2, eps=1e-8):
    p1 = max(min(p1, 1 - eps), eps)
    p2 = max(min(p2, 1 - eps), eps)
    return (p2 - p1) * np.log(p2 / p1)


def discrete_tvd(p1, p2):
    return 0.5 * np.sum(np.abs(p1 - p2))


def discrete_psi(p1, p2, eps=1e-8):
    p1 = np.clip(p1, eps, 1 - eps)
    p2 = np.clip(p2, eps, 1 - eps)
    return np.sum((p2 - p1) * np.log(p2 / p1))


all_records = []
distribution_records = []

split_files_new = sorted([f for f in os.listdir(SPLIT_DIR) if f.startswith('S') and '_seed_' in f])


for split_file in split_files_new:

    match = re.match(r'(S\d+)_seed_\d+\.pkl', split_file)
    if not match:
        continue

    scenario = match.group(1)
    if scenario not in SCENARIOS:
        print(f"Unknown: {scenario}")
        continue

    split_path = os.path.join(SPLIT_DIR, split_file)
    split_data = joblib.load(split_path)

    train_idx = split_data['train_idx']
    val_idx = split_data['val_idx']
    test_idx = split_data['test_idx']

    df_train_scenario = df.iloc[train_idx].copy()
    df_val_scenario = df.iloc[val_idx].copy()
    df_test_scenario = df.iloc[test_idx].copy()

    df_train_orig = df_train_scenario[
        (df_train_scenario['perturbation_type'] == 'original') &
        (df_train_scenario['flip_count'].isna())
        ].copy()
    df_val_orig = df_val_scenario[
        (df_val_scenario['perturbation_type'] == 'original') &
        (df_val_scenario['flip_count'].isna())
        ].copy()
    df_test_orig = df_test_scenario[
        (df_test_scenario['perturbation_type'] == 'original') &
        (df_test_scenario['flip_count'].isna())
        ].copy()

    print(f" train original: {len(df_train_orig)} ")
    print(f" test original: {len(df_test_orig)} ")
    print(f" train scenario: {len(df_test_scenario)} ")
    print(f" test scenario: {len(df_test_scenario)} ")

    for feat in feature_cols:
        p_train = df_train_orig[feat].mean()
        p_val = df_val_orig[feat].mean()
        p_test = df_test_orig[feat].mean()
        print(p_train, p_val, p_test)

        train_val_tvd = abs(p_train - p_val)
        train_val_psi = binary_psi(p_train, p_val)
        train_test_tvd = abs(p_train - p_test)
        train_test_psi = binary_psi(p_train, p_test)
        val_test_tvd = abs(p_val - p_test)
        val_test_psi = binary_psi(p_val, p_test)

        all_records.append({
            'split_file': split_file,
            'scenario': 'original',
            'feature': feat,
            'train_val_TVD': train_val_tvd,
            'train_val_PSI': train_val_psi,
            'train_test_TVD': train_test_tvd,
            'train_test_PSI': train_test_psi,
            'val_test_TVD': val_test_tvd,
            'val_test_PSI': val_test_psi,
            'type': 'feature'
        })

    # --- source ---
    all_categories = pd.concat([df_train_orig['Source'], df_val_orig['Source'],
                                df_test_orig['Source']]).unique()

    train_counts = df_train_orig['Source'].value_counts().reindex(all_categories, fill_value=0)
    val_counts = df_val_orig['Source'].value_counts().reindex(all_categories, fill_value=0)
    test_counts = df_test_orig['Source'].value_counts().reindex(all_categories, fill_value=0)

    p_train_source = train_counts.values / len(df_train_orig)
    p_val_source = val_counts.values / len(df_val_orig)
    p_test_source = test_counts.values / len(df_test_orig)

    train_test_tvd_source = discrete_tvd(p_train_source, p_test_source)
    train_val_tvd_source = discrete_tvd(p_train_source, p_val_source)
    val_test_tvd_source = discrete_tvd(p_val_source, p_test_source)

    train_test_psi_source = discrete_psi(p_train_source, p_test_source)
    train_val_psi_source = discrete_psi(p_train_source, p_val_source)
    val_test_psi_source = discrete_psi(p_val_source, p_test_source)


    all_records.append({
        'split_file': split_file,
        'scenario': 'original',
        'feature': 'Source',
        'train_val_TVD': train_val_tvd_source,
        'train_val_PSI': train_val_psi_source,
        'train_test_TVD': train_test_tvd_source,
        'train_test_PSI': train_test_psi_source,
        'val_test_TVD': val_test_tvd_source,
        'val_test_PSI': val_test_psi_source,
        'type': 'source'
    })
    for cat in all_categories:
        train_ct = train_counts[cat]
        val_ct = val_counts[cat]
        test_ct = test_counts[cat]

        distribution_records.append({
            'split_file': split_file,
            'scenario': 'original',
            'type': 'Source',
            'class': cat,
            'train_ratio': train_ct / len(df_train_orig),
            'val_ratio': val_ct / len(df_val_orig),
            'test_ratio': test_ct / len(df_test_orig)
        })


    if len(df_train_orig) == 0:
        print(" No data")
        continue

    # --- feature ---
    for feat in feature_cols:
        p_train = df_train_scenario[feat].mean()
        p_val = df_val_scenario[feat].mean()
        p_test = df_test_scenario[feat].mean()

        train_val_tvd = abs(p_train - p_val)
        train_val_psi = binary_psi(p_train, p_val)
        train_test_tvd = abs(p_train - p_test)
        train_test_psi = binary_psi(p_train, p_test)
        val_test_tvd = abs(p_val - p_test)
        val_test_psi = binary_psi(p_val, p_test)


        all_records.append({
            'split_file': split_file,
            'scenario': scenario,
            'feature': feat,
            'train_val_TVD': train_val_tvd,
            'train_val_PSI': train_val_psi,
            'train_test_TVD': train_test_tvd,
            'train_test_PSI': train_test_psi,
            'val_test_TVD': val_test_tvd,
            'val_test_PSI': val_test_psi,
            'type': 'feature'
        })

    # --- source ---
    all_categories = pd.concat([df_train_scenario['Source'], df_val_scenario['Source'],
                                df_test_scenario['Source']]).unique()

    train_counts = df_train_scenario['Source'].value_counts().reindex(all_categories, fill_value=0)
    val_counts = df_val_scenario['Source'].value_counts().reindex(all_categories, fill_value=0)
    test_counts = df_test_scenario['Source'].value_counts().reindex(all_categories, fill_value=0)

    p_train_source = train_counts.values / len(df_train_scenario)
    p_val_source = val_counts.values / len(df_val_scenario)
    p_test_source = test_counts.values / len(df_test_scenario)

    train_test_tvd_source = discrete_tvd(p_train_source, p_test_source)
    train_val_tvd_source = discrete_tvd(p_train_source, p_val_source)
    val_test_tvd_source = discrete_tvd(p_val_source, p_test_source)

    train_test_psi_source = discrete_psi(p_train_source, p_test_source)
    train_val_psi_source = discrete_psi(p_train_source, p_val_source)
    val_test_psi_source = discrete_psi(p_val_source, p_test_source)

    all_records.append({
        'split_file': split_file,
        'scenario': scenario,
        'feature': 'Source',
        'train_val_TVD': train_val_tvd_source,
        'train_val_PSI': train_val_psi_source,
        'train_test_TVD': train_test_tvd_source,
        'train_test_PSI': train_test_psi_source,
        'val_test_TVD': val_test_tvd_source,
        'val_test_PSI': val_test_psi_source,
        'type': 'source'
    })
    for cat in all_categories:
        train_ct = train_counts[cat]
        val_ct = val_counts[cat]
        test_ct = test_counts[cat]

        distribution_records.append({
            'split_file': split_file,
            'scenario': scenario,
            'type': 'Source',
            'class': cat,
            'train_ratio': train_ct / len(df_train_scenario),
            'val_ratio': val_ct / len(df_val_scenario),
            'test_ratio': test_ct / len(df_test_scenario)
        })

# --- output ---
if len(all_records) > 0:
    df_output = pd.DataFrame(all_records)
    df_output.to_csv(OUTPUT_CSV, index=False)
    df_dist = pd.DataFrame(distribution_records)
    df_dist.to_csv(OUTPUT_DIST_CSV, index=False)
    print(f"\nSaved to: {OUTPUT_CSV}")
    print(f" Number of scenarios: {df_output['scenario'].nunique()}")
    print(f" Number of features: {df_output[df_output['type'] == 'feature']['feature'].nunique()}")
    print(f" Number of splits: {df_output['split_file'].nunique()}")

