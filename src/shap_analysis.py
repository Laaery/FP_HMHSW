#!/usr/bin/env python
# -*- coding:utf-8 -*-
# @Author: LL
# @File：shap_analysis.py
import os
import pandas as pd
import numpy as np
import shap
from matplotlib.patches import Patch
import matplotlib.pyplot as plt
import joblib
import scienceplots
import re
import matplotlib as mpl


plt.rcParams['font.family'] = 'Arial'
mpl.rcParams['mathtext.default'] = 'regular'


# ============ CONFIGURATION =============
data_path = '../data/phase_vector_fn_fp_mingle_fn_fp_1_2.csv'
model_path = '../output/hpo_exp/models/model_MLP_seed_5.pkl'
data_split = '../output/hpo_exp/data_splits/split_seed_5.pkl'
output_plot_path = '../output/hpo_exp/shap/shap_by_source.png'
feature_names = '../output/hpo_exp/feature_names.pkl'
label_encoder_path = "../output/hpo_exp/label_encoder.pkl"
# =============================================

def plot_avg_shap(avg_shap_matrix, title, output_path):
    fig, axes = plt.subplots(
        n_classes, 1,
        figsize=(16, 10),
        sharex=True,
        gridspec_kw={'hspace': 0}
    )

    if n_classes == 1:
        axes = [axes]
    elif isinstance(axes, np.ndarray):
        axes = axes.flatten()

    all_vals = avg_shap_matrix.flatten()
    y_max = np.max(np.abs(all_vals)) * 1

    group_size = 8
    n_groups = int(np.ceil(n_features / group_size))

    facecolor = '#DCDCDC'
    alpha = 0.8
    edgecolor_group = 'lightgrey'
    linewidth_group = 0.8
    for group_idx in range(n_groups):
        start_x = group_idx * group_size
        end_x = min(start_x + group_size, n_features)

        for ax in axes:
            if group_idx % 2 == 0:
                ax.axvspan(
                    start_x, end_x,
                    facecolor=facecolor,
                    edgecolor='none',
                    alpha=alpha,
                    zorder=0
                )
        for ax in axes:
            ax.axvline(
                x=start_x,
                color=edgecolor_group,
                linewidth=linewidth_group,
                alpha=0.85,
                zorder=1,
                linestyle='--',
                dash_capstyle='round'
            )
        if group_idx == n_groups - 1:
            for ax in axes:
                ax.axvline(
                    x=end_x,
                    color=edgecolor_group,
                    linewidth=linewidth_group,
                    alpha=0.85,
                    zorder=1,
                    linestyle='--',
                    dash_capstyle='round'
                )

    bar_width = 0.75
    x_centers = np.arange(n_features) + bar_width / 2


    label_encoder = joblib.load(label_encoder_path)
    source_mapping = {i: label for i, label in enumerate(label_encoder.classes_)}

    for cls_idx, ax in enumerate(axes):
        values = avg_shap_matrix[cls_idx]

        pos_mask = values > 0
        neg_mask = values < 0
        pos_vals = values * pos_mask
        neg_vals = values * neg_mask

        ax.bar(
            x_centers, pos_vals, width=bar_width,
            color='#f91592', edgecolor='black', linewidth=0.5, alpha=0.5, align='center'
        )
        ax.bar(
            x_centers, neg_vals, width=bar_width,
            color='#00BFFF', edgecolor='black', linewidth=0.5, alpha=0.5, align='center'
        )

        ax.set_ylim(-y_max, y_max)
        ax.set_yscale('symlog', linthresh=1e-5)
        ax.axhline(y=0, color='black', linewidth=0.8, alpha=0.7)

        cls_num = classes[cls_idx]
        cls_label = source_mapping[cls_num]

        ax.set_ylabel(
            f"{cls_label}",
            fontsize=7,
            rotation=0,
            labelpad=100,
            ha='center',
            va='center'
        )
        ax.set_yticks([])
        ax.spines['left'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)
        ax.spines['bottom'].set_visible(False)
        ax.tick_params(axis='both', which='both', top=False, right=False, left=False, bottom=False)

    axes[-1].set_xlabel("Feature Name", fontsize=10, labelpad=10)

    axes[-1].set_xticks(x_centers)
    x_chem = [chem_formula_to_latex(feature) for feature in feature_cols]
    axes[-1].set_xticklabels(x_chem, fontsize=5, rotation=90, ha='center', fontname='Arial')

    legend_elements = [
        Patch(facecolor='#f91592', label='SHAP Value > 0', alpha=0.5),
        Patch(facecolor='#00BFFF', label='SHAP Value < 0', alpha=0.5)
    ]
    fig.legend(handles=legend_elements, loc='upper right', fontsize=10)

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.xlim(0, n_features - 1)
    plt.suptitle(title, fontsize=12)
    plt.savefig(output_path, dpi=450)
    plt.close()
    print(f"Saved to {output_path}")




def chem_formula_to_latex(formula: str) -> str:
    segments = formula.split('·')
    latex_segments = []

    for seg_idx, seg in enumerate(segments):
        m = re.match(r'^\d+(\.\d+)?', seg)
        prefix = ''
        body = seg
        if m:
            prefix = m.group(0)
            body = seg[m.end():]

        def replace_numbers(m):
            return f"_{{{m.group()}}}"
        if seg_idx == 0:
            body_latex = re.sub(r'\d+(\.\d+)?', replace_numbers, body)
        else:
            m2 = re.match(r'^\d+(\.\d+)?', body)
            if m2:
                first_num = m2.group(0)
                rest = body[m2.end():]
                rest_latex = re.sub(r'\d+(\.\d+)?', replace_numbers, rest)
                body_latex = first_num + rest_latex
            else:
                body_latex = re.sub(r'\d+(\.\d+)?', replace_numbers, body)

        latex_segments.append(prefix + body_latex)

    return '$' + '·'.join(latex_segments) + '$'


def export_and_plot_feature_ranking(
    avg_shap,
    shap_array=None,
    feature_cols=None,
    classes=None,
    source_mapping=None,
    output_plot_path="output.png",
    top_k=20
):
    """
    Export and plot feature ranking based on SHAP values.
    """
    all_ranked_features = []
    if source_mapping == None:
        label_encoder = joblib.load(label_encoder_path)
        source_mapping = {i: label for i, label in enumerate(label_encoder.classes_)}

    for cls_idx, cls_num in enumerate(classes):
        cls_name = source_mapping[cls_num]
        shap_vals = avg_shap[cls_idx]  # (n_features,)

        sorted_idx = np.argsort(shap_vals)[::-1]
        ranked_features = [(feature_cols[i], shap_vals[i]) for i in sorted_idx]

        for fname, sval in ranked_features:
            all_ranked_features.append({
                'Source': cls_name,
                'Source_Class': cls_num,
                'Feature': fname,
                'SHAP_Value': sval,
                'Rank': len([x for x in all_ranked_features if x['Source'] == cls_name]) + 1
            })

    df_all = pd.DataFrame(all_ranked_features)
    df_all = df_all.sort_values(['Source', 'SHAP_Value'], ascending=[True, False])
    df_all['Rank'] = df_all.groupby('Source').cumcount() + 1
    csv_path = output_plot_path.replace(".png", "_all_features_ranked.csv")
    df_all.to_csv(csv_path, index=False, float_format="%.8f")

    if shap_array is not None:
        global_importance = np.mean(np.abs(shap_array), axis=(0,1))
    else:
        global_importance = np.mean(np.abs(avg_shap), axis=0)

    sorted_idx = np.argsort(global_importance)[::-1]
    top_k_idx = sorted_idx[:top_k]
    top_k_features = [feature_cols[i] for i in top_k_idx]
    top_k_importance = global_importance[top_k_idx]

    top_k_features_latex = [chem_formula_to_latex(f) for f in top_k_features]

    plt.figure(figsize=(12, 6))
    bars = plt.bar(
        range(top_k),
        top_k_importance,
        color='skyblue',
        edgecolor='black',
        linewidth=0.6,
        alpha=0.8
    )

    for i, bar in enumerate(bars):
        height = bar.get_height()
        plt.text(
            bar.get_x() + bar.get_width() / 2,
            height + 0.01 * max(top_k_importance),
            f'{height:.3f}',
            ha='center',
            va='bottom',
            fontsize=8,
            rotation=90
        )

    plt.xticks(range(top_k), top_k_features_latex, rotation=90, fontsize=10, fontname='Arial')
    plt.ylabel("Global Feature Importance\n(Mean |SHAP|)", fontsize=11)
    plt.title("Top Feature Importance Across All Classes", fontsize=13, pad=20)
    plt.ylim(0, max(top_k_importance) * 1.15)
    plt.grid(axis='y', alpha=0.3, linestyle='--', linewidth=0.5)
    plt.tight_layout()

    global_plot_path = output_plot_path.replace(".png", "_global_feature_importance.png")
    plt.savefig(global_plot_path, dpi=450, bbox_inches='tight')
    plt.close()

    return df_all, top_k_features, top_k_importance


# 1. Load model
model = joblib.load(model_path)

# 2. Load data
df = pd.read_csv(data_path)
feature_cols = joblib.load(feature_names)
X_full = df[feature_cols].values
y_full = df['Source'].values
split_data = joblib.load(data_split)
test_idx = split_data['test_idx']
X_test = X_full[test_idx]
y_test = y_full[test_idx]

n_samples, n_features = X_test.shape

if hasattr(model, 'classes_'):
    classes = model.classes_
    n_classes = len(classes)
    print(f" Number of classes: {n_classes}")
else:
    raise ValueError("Model does not have 'classes_' attribute")


# 3. Calculate SHAP values
explainer = shap.SamplingExplainer(model.predict_proba, X_test)
shap_values = explainer(X_test)
print(" SHAP values calculated")
shap_array = shap_values.values

if shap_array.ndim == 3:
    shap_array = shap_array.transpose(2, 0, 1)
elif shap_array.ndim == 2 and n_classes == 2:
    shap_array = shap_array[None, ...]
    shap_array = np.concatenate([shap_array, -shap_array], axis=0)
else:
    shap_array = shap_array[None, ...]


# 4. Calculate average SHAP values for each feature when feature value is 0 or 1
avg_shap_when1 = np.zeros((n_classes, n_features))
avg_shap_when0 = np.zeros((n_classes, n_features))

for cls_idx in range(n_classes):
    shap_per_sample = shap_array[cls_idx]  # (n_samples, n_features)

    for j in range(n_features):
        mask1 = X_test[:, j] == 1
        mask0 = X_test[:, j] == 0
        if mask1.sum() > 0:
            avg_shap_when1[cls_idx, j] = shap_per_sample[mask1, j].mean()
        if mask0.sum() > 0:
            avg_shap_when0[cls_idx, j] = shap_per_sample[mask0, j].mean()

records = []
for cls_idx, cls_label in enumerate(classes):
    for j, feat_name in enumerate(feature_cols):
        records.append({
            'class': cls_label,
            'feature': feat_name,
            'avg_shap_when_1': avg_shap_when1[cls_idx, j],
            'avg_shap_when_0': avg_shap_when0[cls_idx, j]
        })

df_shap_summary = pd.DataFrame(records)

output_path = output_plot_path.replace(".png", "_shap_summary.csv")
df_shap_summary.to_csv(output_path, index=False)

plot_avg_shap(avg_shap_when1, "Mean SHAP (Feature=1)", output_plot_path.replace(".png", "_when1.png"))
plot_avg_shap(avg_shap_when0, "Mean SHAP (Feature=0)", output_plot_path.replace(".png", "_when0.png"))