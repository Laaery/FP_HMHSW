#!/usr/bin/env python
# -*- coding:utf-8 -*-
# @Author: LL
# @File：calculate_wss.py

"""
Calculate the Weighted stability score (WSS) to determine the optimal number of fingerprints.
"""
import os
import pandas as pd
import numpy as np
from collections import defaultdict
import matplotlib.pyplot as plt
import seaborn as sns
import scienceplots
from kneed import KneeLocator
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize
from matplotlib.gridspec import GridSpec
from matplotlib.patches import Rectangle
from mpl_toolkits.axes_grid1.inset_locator import inset_axes

plt.style.use(['science', 'no-latex'])
plt.rcParams['font.family'] = 'Arial'

# -----------------------------
# Configuration
# -----------------------------
INPUT_DIR = '../output/fn_fp_mingle_fn_fp_bacc_balanced_250930'  # OR mingle_fn_fp
OUTPUT_DIR = '../output/fn_fp_mingle_fn_fp_bacc_balanced_250930'  # OR mingle_fn_fp
DATA_FILE = '../data/phase_vector_fn_fp_mingle_fn_fp_1_2.csv'  # OR phase_vector_mingle_fn_fp_1_2.csv
THRESHOLD = 0.99  # Threshold for stability selection

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


def load_data(data_file):
    if not os.path.exists(data_file):
        raise FileNotFoundError(f"Data file not found: {data_file}")
    return pd.read_csv(data_file)


def count_scenario_samples(df, scenarios):
    scenario_sample_sizes = {}
    for name, config in scenarios.items():
        mask = pd.Series(False, index=df.index)
        for pt, fc in config['include']:
            if pd.isna(fc):
                condition = (df['perturbation_type'] == pt) & (df['flip_count'].isna())
            else:
                condition = (df['perturbation_type'] == pt) & (df['flip_count'] == fc)
            mask |= condition
        count = mask.sum()
        scenario_sample_sizes[name] = count
        print(f"   {name}: {count} samples")
    return scenario_sample_sizes


def load_all_results(input_dir, scenarios):
    all_results = []
    scenario_names = list(scenarios.keys())
    for name in scenario_names:
        file_path = os.path.join(input_dir, f'repeat_detailed_results_{name}.csv')
        if not os.path.exists(file_path):
            print(f"{file_path} not found, skipping scenario '{name}'")
            continue
        df = pd.read_csv(file_path)
        df['scenario'] = name
        all_results.append(df)
    return pd.concat(all_results, ignore_index=True) if all_results else pd.DataFrame()


def compute_scenario_weights(scenario_sample_sizes):
    total_samples = sum(scenario_sample_sizes.values())
    if total_samples == 0:
        raise ValueError("Cannot compute weights: total sample size is zero.")
    return {name: size / total_samples for name, size in scenario_sample_sizes.items()}


def collect_feature_stability_scores(results_df, scenario_sample_sizes):
    scenario_feature_counts = defaultdict(lambda: defaultdict(int))
    scenario_repeats = defaultdict(int)

    for _, row in results_df.iterrows():
        scenario = row['scenario']
        features_str = row['best_features']
        try:
            features = [f.strip() for f in str(features_str).split(', ') if f.strip()]
        except Exception as e:
            print(f"Cannot parse features for scenario '{scenario}': {features_str}. Error: {e}")
            continue
        for feat in features:
            scenario_feature_counts[scenario][feat] += 1
        scenario_repeats[scenario] += 1

    weights = compute_scenario_weights(scenario_sample_sizes)

    feature_scores = defaultdict(float)
    feature_details = defaultdict(dict)

    for scenario_name in scenario_feature_counts:
        n_repeats = scenario_repeats[scenario_name]
        weight = weights[scenario_name]

        for feature, count in scenario_feature_counts[scenario_name].items():
            frequency = count / n_repeats
            score_contribution = frequency * weight
            feature_scores[feature] += score_contribution

            feature_details[feature][scenario_name] = {
                'frequency': frequency,
                'raw_count': count,
                'n_repeats': n_repeats,
                'weight': weight
            }

    return feature_scores, feature_details


def select_robust_feature_set(feature_scores, threshold=0.8, output_dir=None):
    """
    Selects a set of features based on their weighted stability scores.

    Parameters:
    -----------
    feature_scores : dict
        A dictionary where keys are feature names and values are their weighted stability scores.
    threshold : float or str
        The threshold for selecting features.

    Returns:
    --------
    final_features : list
    summary_df : pd.DataFrame
    """
    if not feature_scores:
        return [], pd.DataFrame()

    all_scores = np.array(list(feature_scores.values()))
    sorted_scores = np.sort(all_scores)
    n_features = len(sorted_scores)

    print(f"Input threshold: {threshold}")

    final_features = [feat for feat, score in feature_scores.items() if score >= threshold]
    scores = [score for feat, score in feature_scores.items() if score >= threshold]

    # 创建结果 DataFrame
    summary_df = pd.DataFrame({
        'feature': final_features,
        'weighted_stability_score': scores
    })


    return final_features, summary_df



def plot_stability_heatmap(feature_details, output_dir):
    """Plots a heatmap of feature stability scores."""
    scenarios = list(SCENARIOS.keys())
    short_labels = [f'S{i + 1}' for i in range(len(scenarios))]
    label_mapping = dict(zip(scenarios, short_labels))

    feature_total_scores = {}
    for feat, sc_dict in feature_details.items():
        total_score = sum(d['frequency'] * d['weight'] for d in sc_dict.values())
        feature_total_scores[feat] = total_score
    order_df = pd.read_csv("..\output\dim_reduction\phase_vector_chi_square_results.csv")
    all_features = order_df['Feature'].tolist()

    data = []
    weighted_data = []
    for feature in all_features:
        row = []
        weighted_row = []
        for scenario in scenarios:
            scenario_data = feature_details[feature].get(scenario, {})
            freq = scenario_data.get('frequency', 0)
            weight = scenario_data.get('weight', 1.0)
            row.append(freq)
            weighted_row.append(freq * weight)
        data.append(row)
        weighted_data.append(weighted_row)

    if not data:
        print("No data to plot.")
        return

    data_df = pd.DataFrame(data, index=range(len(all_features)), columns=scenarios)
    data_df.columns = [label_mapping[col] for col in data_df.columns]
    avg_freq_per_fingerprint = data_df.mean(axis=1).values

    weighted_data_df = pd.DataFrame(weighted_data, index=range(len(all_features)), columns=scenarios)
    weighted_data_df.columns = [label_mapping[col] for col in weighted_data_df.columns]
    weighted_avg_freq_per_fingerprint = weighted_data_df.sum(axis=1).values

    fig = plt.figure(figsize=(9, 8))


    gs = GridSpec(2, 2, figure=fig, width_ratios=[4, 1], wspace=0.1, height_ratios=[1, 4], hspace=0.1)

    # # --- Figure 1: Average Frequency per Fingerprint ---
    ax1 = fig.add_subplot(gs[2])
    cbar_ax = fig.add_subplot(gs[1])

    # Configuration
    cell_height = 1.0
    cell_width = 0.35
    col_spacing = 0.2
    row_spacing = 0
    side_margin = 0.1

    total_width = data_df.shape[1] * cell_width + (data_df.shape[1] - 1) * col_spacing
    start_x = (4 - total_width) / 2

    x_positions = start_x + np.cumsum([0] + [cell_width + col_spacing] * (data_df.shape[1] - 1))
    y_positions = np.arange(data_df.shape[0]) * (cell_height + row_spacing)

    norm = Normalize(vmin=data_df.values.min(), vmax=data_df.values.max())
    cmap = plt.get_cmap('RdYlBu_r')

    for (i, j), val in np.ndenumerate(data_df.values):
        rect = Rectangle(
            (x_positions[j], y_positions[i]),
            width=cell_width,
            height=cell_height,
            facecolor=cmap(norm(val)),
            edgecolor='none'
        )
        ax1.add_patch(rect)

    ax1.set_xlim(0, 4)
    ax1.set_ylim(y_positions[-1] + cell_height, 0)

    for x in x_positions[1:]:
        ax1.axvline(x - col_spacing / 2, color='white', linewidth=2)

    ax1.axvline(start_x - col_spacing / 2, color='white', linewidth=2)
    ax1.axvline(x_positions[-1] + cell_width + col_spacing / 2, color='white', linewidth=2)

    ax1.set_xticks(x_positions + cell_width / 2)
    ax1.set_xticklabels(data_df.columns)
    ax1.set_yticks(y_positions + cell_height / 2)
    ax1.set_yticklabels(data_df.index)

    sm = ScalarMappable(norm=norm, cmap=cmap)
    sm.set_array([])
    cbar = plt.colorbar(sm, cax=cbar_ax)
    cbar.ax.tick_params(labelsize=8)

    y_ticks_pos = list(range(0, len(all_features), 20))
    ax1.set_yticks(y_ticks_pos)
    ax1.set_yticklabels([str(pos) for pos in y_ticks_pos], rotation=0)
    ax1.set_ylabel('Fingerprint Index')
    ax1.set_xlabel('Scenarios')
    for _, spine in ax1.spines.items():
        spine.set_visible(True)
        spine.set_linewidth(1.5)

    # --- Figure 2: Weighted Stability Score ---
    ax2 = fig.add_subplot(gs[3])
    fingerprint_indices = range(len(all_features))

    ax2.plot(weighted_avg_freq_per_fingerprint, fingerprint_indices,
             color='darkred', linewidth=1.2, label='Avg WSS')

    ax2.fill_betweenx(
        fingerprint_indices, 0, weighted_avg_freq_per_fingerprint,
        color='lightcoral', alpha=0.4
    )


    ax2.set_ylim(ax1.get_ylim())
    ax2.set_yticks(y_ticks_pos)
    ax2.set_yticklabels([str(pos) for pos in y_ticks_pos])
    ax2.set_xlabel('Weighted Stability Score')
    ax2.set_ylabel('')
    ax2.grid(True, axis='y', alpha=0.3, linestyle='--')

    for spine in ax2.spines.values():
        spine.set_linewidth(1.5)

    ax3 = fig.add_subplot(gs[0], sharex=ax1)

    n_features = len(data_df)
    stacked_data = pd.DataFrame({
        '0.8–1.0': ((data_df >= 0.8) & (data_df <= 1.0)).sum(axis=0) / n_features,
        '0.6–0.8': ((data_df >= 0.6) & (data_df < 0.8)).sum(axis=0) / n_features,
        '0.4–0.6': ((data_df >= 0.4) & (data_df < 0.6)).sum(axis=0) / n_features,
        '0.2–0.4': ((data_df >= 0.2) & (data_df < 0.4)).sum(axis=0) / n_features,
        '0.0–0.2': (data_df < 0.2).sum(axis=0) / n_features
    })

    x_pos = ax1.get_xticks()

    n_scenarios = len(x_pos)
    top3_flags = [None] * n_scenarios

    for i in range(n_scenarios):
        col_data = stacked_data.iloc[i].copy()
        valid_data = col_data[col_data > 0]
        sorted_labels = valid_data.sort_values(ascending=False).index.tolist()
        top3_flags[i] = set(sorted_labels[:3])

    bottom = [0] * len(x_pos)
    colors = ['#A50026', '#FDAE61', '#FFFFBF', '#ABD9E9', '#2C7BB6']


    for label, color in zip(stacked_data.columns, colors):
        heights = stacked_data[label]
        ax3.bar(
            x_pos,
            heights,
            bottom=bottom,
            width=cell_width,
            color=color,
            edgecolor='white',
            align='center',
            label=label
        )

        for i, (x, y) in enumerate(zip(x_pos, heights)):
            if y > 0 and label in top3_flags[i]:
                text_y = bottom[i] + y / 2
                percentage = y * 100
                ax3.text(
                    x, text_y, f'{percentage:.1f}%',
                    ha='center', va='center', fontsize=7, color='black', fontdict={'weight': 'bold'}
                )

        bottom += heights

    ax3.set_xlim(ax1.get_xlim())
    ax3.set_ylim(0, 1)
    ax3.set_yticks([0.0, 0.5, 1.0])
    ax3.set_yticklabels(['0%', '50%', '100%'])
    ax3.set_ylabel('Percentage of Fingerprints')
    ax3.tick_params(axis='x', which='both', length=0)
    legend = ax3.legend(
        title='Frequency Range',
        loc='lower center',
        bbox_to_anchor=(0.5, 1.02),
        ncol=5,
        fontsize=10,
        title_fontsize=11,
        frameon=False,
        handlelength=1.2,
        columnspacing=1.0
    )

    for spine in ax3.spines.values():
        spine.set_linewidth(1.5)

    ax_legend = fig.add_subplot(gs[0, 1])
    ax_legend.axis('off')

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'stability_heatmap_with_trend.png'), dpi=300, bbox_inches='tight')
    plt.close()
    print("Heatmap: stability_heatmap_with_trend.png (with Weighted Stability Score)")




def main():
    print("Select Robust Fingerprint Set Across Scenarios ")
    print("=" * 60)

    try:
        df_data = load_data(DATA_FILE)
        print("Complete data loaded successfully.")
        scenario_sample_sizes = count_scenario_samples(df_data, SCENARIOS)
        print(f"Number of scenarios: {len(SCENARIOS)}")
    except Exception as e:
        print(f"Failed to load data or count scenarios: {e}")
        return

    results_df = load_all_results(INPUT_DIR, SCENARIOS)
    if results_df.empty:
        print("File not found or no valid results loaded.")
        return

    print(f" {len(results_df)} results loaded!")

    feature_scores, feature_details = collect_feature_stability_scores(results_df, scenario_sample_sizes)

    if not feature_scores:
        print("No features found in the results.")
        return

    print(f"{len(feature_scores)} features found across all scenarios.")

    robust_features, summary_df = select_robust_feature_set(feature_scores, threshold=THRESHOLD, output_dir=OUTPUT_DIR)

    print(f"\n Select {len(robust_features)} robust features across all scenarios for Weighted Stability Score >= {THRESHOLD}")
    for i, feat in enumerate(robust_features, 1):
        score = feature_scores[feat]
        print(f"   {i:2d}. {feat:<30} (Weighted Stability Score: {score:.4f})")
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    summary_df.to_csv(os.path.join(OUTPUT_DIR, f'robust_feature_set_{THRESHOLD}.csv'), index=False)

    detail_data = []
    for feature in feature_scores.keys():
        row = {'feature': feature, 'total_weighted_score': feature_scores[feature]}
        for scenario in SCENARIOS.keys():
            details = feature_details[feature].get(scenario, {})
            row[f'{scenario}_freq'] = details.get('frequency', 0)
            row[f'{scenario}_count'] = details.get('raw_count', 0)
        detail_data.append(row)

    detail_df = pd.DataFrame(detail_data)
    detail_df.to_csv(os.path.join(OUTPUT_DIR, 'feature_stability_detailed.csv'), index=False)

    plot_stability_heatmap(feature_details, OUTPUT_DIR)

    weights = compute_scenario_weights(scenario_sample_sizes)
    print("\n Scenario Sample Sizes and Weights:")
    for name in SCENARIOS.keys():
        size = scenario_sample_sizes.get(name, 0)
        weight = weights.get(name, 0)
        print(f"   {name:<20} : {size:3d} samples -> weight = {weight:.3f}")

    print(f"\nSaved results:")
    print("   - robust_feature_set.csv        : Robust feature set summary")
    print("   - feature_stability_detailed.csv: Detailed feature stability analysis")
    print("   - stability_heatmap.png         : Feature selection frequency heatmap")
    print("   - robust_feature_set.txt        : Text summary of robust features")
    print("\n Feature selection completed successfully!")


if __name__ == '__main__':
    main()
