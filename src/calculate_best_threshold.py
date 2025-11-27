#!/usr/bin/env python
# -*- coding:utf-8 -*-
# @Author: LL
# @File：calculate_best_threshold.py
import numpy as np
import pandas as pd
import os
from joblib import Parallel, delayed
from sklearn.model_selection import train_test_split
from tqdm import tqdm
from collections import defaultdict
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import balanced_accuracy_score, f1_score, roc_auc_score

# -------------------------------
# 1. Configuration
# -------------------------------
DATA_FILE = '../data/phase_vector_fn_fp_mingle_fn_fp_1_2.csv'
ROBUST_FEATURE_CSV = "../output/fn_fp_mingle_fn_fp_bacc_balanced/robust_feature_set_wss.csv"
OUTPUT_DIR = "../output/best_threshold_calculation"
TEMP_DIR = os.path.join(OUTPUT_DIR, "temp")
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(TEMP_DIR, exist_ok=True)

DETAILED_LOG_FILE = os.path.join(OUTPUT_DIR, "model_performance_detailed_log.csv")
SUMMARY_FILE = os.path.join(OUTPUT_DIR, "summary_with_train_val_test.csv")

LABEL_COLUMN = 'Source'
META_COLS = ['scenario', 'perturbation_type', 'flip_count', 'Index']
RANDOM_SEEDS = list(range(10))
THRESHOLDS = np.arange(0.00, 1.01, 0.01)
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


def proportional_stratified_split(subset, label_column, random_state=42):
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


def build_scenario_subset(df_data, scenario_config):
    mask = pd.Series(False, index=df_data.index)
    for pt, fc in scenario_config['include']:
        if pd.isna(fc):
            condition = (df_data['perturbation_type'] == pt) & (df_data['flip_count'].isna())
        else:
            condition = (df_data['perturbation_type'] == pt) & (df_data['flip_count'] == fc)
        mask |= condition
    return df_data[mask].copy()


# -------------------------------
# 3. Helper functions
# -------------------------------
def calculate_metrics(y_true, y_pred, y_proba, classes):
    bal_acc = balanced_accuracy_score(y_true, y_pred)
    f1_macro = f1_score(y_true, y_pred, average='macro')
    macro_auc = roc_auc_score(y_true, y_proba, multi_class='ovr', average='macro')
    return bal_acc, f1_macro, macro_auc


# -------------------------------
# 4. Main function
# -------------------------------
def evaluate_threshold_seed(threshold, seed, X_full, y_full, df_data, feature_scores, classes):
    selected_features = [f for f, s in feature_scores.items() if s >= threshold]
    n_features = len(selected_features)

    if n_features == 0:
        return {
            'threshold': threshold, 'seed': seed, 'n_features': 0,
            'train_bal_acc': 0.0, 'train_f1': 0.0, 'train_auc': 0.0,
            'val_bal_acc': 0.0, 'val_f1': 0.0, 'val_auc': 0.0,
            'test_bal_acc': 0.0, 'test_f1': 0.0, 'test_auc': 0.0,
            'selected_features': '[]'
        }

    X_sel = X_full[selected_features].copy()

    try:
        train_idx, val_idx, test_idx = proportional_stratified_split(
            df_data, label_column=LABEL_COLUMN, random_state=seed
        )

        X_train = X_sel.loc[train_idx]
        X_val = X_sel.loc[val_idx]
        X_test = X_sel.loc[test_idx]
        y_train = y_full.loc[train_idx]
        y_val = y_full.loc[val_idx]
        y_test = y_full.loc[test_idx]

        if len(X_train) == 0 or len(y_test) == 0:
            m_train = m_val = m_test = (0.0, 0.0, 0.0)
        else:
            model = RandomForestClassifier(
                n_estimators=100,
                random_state=42,
                class_weight='balanced',
                max_depth=25,
                min_samples_leaf=5,
                n_jobs=1
            )
            model.fit(X_train, y_train)

            y_train_pred = model.predict(X_train)
            y_val_pred = model.predict(X_val)
            y_test_pred = model.predict(X_test)

            y_train_proba = model.predict_proba(X_train)
            y_val_proba = model.predict_proba(X_val)
            y_test_proba = model.predict_proba(X_test)

            m_train = calculate_metrics(y_train, y_train_pred, y_train_proba, classes)
            m_val = calculate_metrics(y_val, y_val_pred, y_val_proba, classes)
            m_test = calculate_metrics(y_test, y_test_pred, y_test_proba, classes)

        return {
            'threshold': threshold,
            'seed': seed,
            'n_features': n_features,
            'train_bal_acc': m_train[0], 'train_f1': m_train[1], 'train_auc': m_train[2],
            'val_bal_acc': m_val[0], 'val_f1': m_val[1], 'val_auc': m_val[2],
            'test_bal_acc': m_test[0], 'test_f1': m_test[1], 'test_auc': m_test[2],
            'selected_features': ','.join(selected_features)
        }

    except Exception as e:
        print(f"Error at threshold={threshold:.2f}, seed={seed}: {e}")
        return {
            'threshold': threshold, 'seed': seed, 'n_features': n_features,
            'train_bal_acc': np.nan, 'train_f1': np.nan, 'train_auc': np.nan,
            'val_bal_acc': np.nan, 'val_f1': np.nan, 'val_auc': np.nan,
            'test_bal_acc': np.nan, 'test_f1': np.nan, 'test_auc': np.nan,
            'selected_features': ','.join(selected_features)
        }



def main():
    print("Starting Threshold Sweep for Each Scenario")
    print("=" * 80)

    try:
        df_data_all = pd.read_csv(DATA_FILE)
        print(f"Loaded full data: {df_data_all.shape}")
    except Exception as e:
        print(f"Failed to load data.csv: {e}")
        return

    for scenario_name, config in SCENARIOS.items():
        print(f"\n{'=' * 50}")
        print(f"Processing Scenario: {scenario_name}")
        print(f"{'=' * 50}")

        # 1. Construct scenario subset
        df_scenario = build_scenario_subset(df_data_all, config)
        if len(df_scenario) == 0:
            print(f"No data matched for {scenario_name}. Skipping.")
            continue
        print(f"Built subset: {len(df_scenario)} samples")

        # 2. Load robust feature scores
        try:
            robust_df = pd.read_csv(ROBUST_FEATURE_CSV)
            if 'feature' not in robust_df.columns or 'weighted_stability_score' not in robust_df.columns:
                print(f"Invalid columns in {ROBUST_FEATURE_CSV}, skipping...")
                continue
            feature_scores = dict(zip(robust_df['feature'], robust_df['weighted_stability_score']))
            print(f"Loaded {len(feature_scores)} features from {ROBUST_FEATURE_CSV}")
        except Exception as e:
            print(f"Cannot load {ROBUST_FEATURE_CSV}: {e}. Skipping scenario.")
            continue

        # 3. Filter features based on threshold
        feature_cols = [c for c in df_scenario.columns if c not in META_COLS + [LABEL_COLUMN]]
        feature_scores = {f: s for f, s in feature_scores.items() if f in feature_cols}
        if not feature_scores:
            print("No common features between CSV and data. Skipping.")
            continue

        # 4. Data preprocessing
        X_full = df_scenario[feature_cols].astype(float)
        y_full = df_scenario[LABEL_COLUMN]
        classes = np.unique(y_full)

        # 5. setup output
        output_dir = os.path.join(OUTPUT_DIR, scenario_name)
        temp_dir = os.path.join(output_dir, "temp")
        os.makedirs(output_dir, exist_ok=True)
        os.makedirs(temp_dir, exist_ok=True)

        detailed_log = os.path.join(output_dir, "model_performance_detailed_log.csv")
        summary_file = os.path.join(output_dir, "summary_with_train_val_test.csv")

        detailed_columns = [
            'threshold', 'seed', 'n_features',
            'train_bal_acc', 'train_f1', 'train_auc',
            'val_bal_acc', 'val_f1', 'val_auc',
            'test_bal_acc', 'test_f1', 'test_auc',
            'selected_features'
        ]

        summary_columns = [
            'threshold', 'seed', 'n_features',
            'train_bal_acc', 'train_f1', 'train_auc',
            'val_bal_acc', 'val_f1', 'val_auc',
            'test_bal_acc', 'test_f1', 'test_auc'
        ]

        pd.DataFrame(columns=detailed_columns).to_csv(detailed_log, index=False)
        pd.DataFrame(columns=summary_columns).to_csv(summary_file, index=False)
        print(f"Output path: {output_dir}")

        # 6. Run threshold sweep
        for threshold in tqdm(THRESHOLDS, desc=f"{scenario_name} - Threshold"):
            tasks = [(threshold, seed) for seed in RANDOM_SEEDS]
            results = Parallel(n_jobs=-1)(
                delayed(evaluate_threshold_seed)(
                    th, seed, X_full, y_full, df_scenario, feature_scores, classes
                )
                for th, seed in tasks
            )

            df_chunk = pd.DataFrame(results)

            df_chunk.to_csv(os.path.join(temp_dir, f"threshold_{threshold:.2f}.csv"), index=False)

            if os.path.exists(detailed_log) and os.path.getsize(detailed_log) > 0:
                df_chunk.to_csv(detailed_log, mode='a', header=False, index=False)
            else:
                df_chunk.to_csv(detailed_log, mode='w', header=True, index=False)

            summary_row = df_chunk.mean(numeric_only=True).round(4)
            summary_row['threshold'] = threshold
            summary_row = summary_row.to_frame().T

            if os.path.exists(summary_file) and os.path.getsize(summary_file) > 0:
                summary_row.to_csv(summary_file, mode='a', header=False, index=False)
            else:
                summary_row.to_csv(summary_file, mode='w', header=True, index=False)

        print(f"Completed: {scenario_name}")

    print(f"\nAll scenarios completed! Results saved to: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
