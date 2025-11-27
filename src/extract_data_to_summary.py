#!/usr/bin/env python
# -*- coding:utf-8 -*-
# @Author: LL
# @File：extract_data_to_summary.py
import os
import pandas as pd
import numpy as np
from pathlib import Path

# ================== Configuration ==================
per_experiment_dir = '../output/hpo_exp/per_experiment'
output_summary_path = '../output/hpo_exp/summary_results_new.csv'
detailed_output_path = '../output/hpo_exp/detailed_results_with_median_representatives.csv'

models = [
    'LogisticRegression',
    'AnnoyKNN',
    'BernoulliNB',
    'MLP',
    'RandomForest',
    'XGBoost'
]

metrics_to_extract = {
    'train_acc_balanced': ('avg_train_acc', 'std_train_acc'),
    'train_f1_macro': ('avg_train_f1', 'std_train_f1'),
    'train_auc': ('avg_train_auc', 'std_train_auc'),
    'test_acc_balanced': ('avg_test_acc', 'std_test_acc'),
    'test_f1_macro': ('avg_test_f1', 'std_test_f1'),
    'test_auc': ('avg_test_auc', 'std_test_auc'),
    'test_acc_balanced_clean': ('avg_test_acc_clean', 'std_test_acc_clean'),
    'test_f1_macro_clean': ('avg_test_f1_clean', 'std_test_f1_clean'),
    'test_auc_clean': ('avg_test_auc_clean', 'std_test_auc_clean')
}

n_seeds = 10
# ==================================================

results = []
detailed_records = []

for model in models:
    print(f"Processing {model}...")

    metric_values = {key: [] for key in metrics_to_extract.keys()}

    seed_scores = {}

    for seed in range(n_seeds):
        file_path = Path(per_experiment_dir) / f"{model}_seed_{seed}.csv"
        if not file_path.exists():
            print(f"  Warning: {file_path} not found.")
            continue

        try:
            df_seed = pd.read_csv(file_path)
            record = {'model': model, 'seed': seed}

            for _, row in df_seed.iterrows():
                metric_name = row['metric']
                value = row['value']

                for key in metric_values.keys():
                    if key == metric_name:
                        metric_values[key].append(value)
                        if key == 'test_acc_balanced':
                            record['test_acc_balanced'] = value
                        elif key == 'test_f1_macro':
                            record['test_f1_macro'] = value
                        elif key == 'test_auc':
                            record['test_macro_auc'] = value
                        break

            if 'test_acc_balanced' in record and 'test_f1_macro' in record and 'test_macro_auc' in record:
                avg_score = np.mean([record['test_acc_balanced'], record['test_f1_macro'], record['test_macro_auc']])
                record['test_avg_score'] = avg_score
                seed_scores[seed] = {
                    'acc': record['test_acc_balanced'],
                    'f1': record['test_f1_macro'],
                    'auc': record['test_macro_auc'],
                    'avg_score': avg_score
                }
            else:
                record['test_avg_score'] = np.nan
                print(f" Missing metrics for {model}_seed_{seed}")

            detailed_records.append(record)

        except Exception as e:
            print(f" Error reading {file_path}: {e}")

    if sum(len(v) for v in metric_values.values()) == 0:
        print(f" No data loaded for {model}")
        continue

    row_data = {'model': model, 'n_repeats': len(metric_values['test_acc_balanced'])}

    for key, (avg_col, std_col) in metrics_to_extract.items():
        values = metric_values[key]
        row_data[avg_col] = np.mean(values) if values else np.nan
        row_data[std_col] = np.std(values, ddof=1) if len(values) > 1 else 0.0

    results.append(row_data)

# ================== Find median representatives ==================
median_representatives = []

for model in models:
    if model not in [r['model'] for r in results]:
        continue

    scores = []
    for rec in detailed_records:
        if rec['model'] == model and 'test_avg_score' in rec and pd.notna(rec['test_avg_score']):
            scores.append((rec['seed'], rec['test_avg_score']))

    if len(scores) == 0:
        continue

    avg_scores = [s[1] for s in scores]
    median_score = np.median(avg_scores)

    sorted_by_dist = sorted(scores, key=lambda x: abs(x[1] - median_score))
    selected_seeds = [item[0] for item in sorted_by_dist[:2]]

    for seed in selected_seeds:
        median_representatives.append({'model': model, 'representative_seed': seed})

detailed_df = pd.DataFrame(detailed_records)
detailed_df['is_median_representative'] = False

for _, row in pd.DataFrame(median_representatives).iterrows():
    model = row['model']
    seed = row['representative_seed']
    match_idx = (detailed_df['model'] == model) & (detailed_df['seed'] == seed)
    detailed_df.loc[match_idx, 'is_median_representative'] = True

Path(detailed_output_path).parent.mkdir(parents=True, exist_ok=True)

detailed_df.to_csv(detailed_output_path, index=False)
print(f"\n Detailed results with median representatives saved to: {detailed_output_path}")

print("\nMedian representative seeds:")
print(pd.DataFrame(median_representatives))

result_df = pd.DataFrame(results)
result_df.to_csv(output_summary_path, index=False)
print(f"\nSummary saved to: {output_summary_path}")
print(result_df[['model', 'avg_test_acc', 'avg_test_f1', 'avg_test_auc']].round(4))