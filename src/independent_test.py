#!/usr/bin/env python
# -*- coding:utf-8 -*-
# @Author: LL
# @File：independent_test.py
"""
Perform Chi-Square test for independence between binary features and categorical target variable.
"""
import pandas as pd
import numpy as np
from scipy.stats import chi2_contingency
from statsmodels.stats.multitest import multipletests
import matplotlib.pyplot as plt
import scienceplots

plt.style.use(['science', 'no-latex'])
plt.rcParams['font.family'] = 'Arial'

df = pd.read_csv("../data/phase_vector.csv")
metadata_cols = ['Index', 'Source']
binary_features = [col for col in df.columns if col not in metadata_cols]
target_col = 'Source'

if target_col not in df.columns:
    raise ValueError(f"Column '{target_col}' not found in data.")

X = df[binary_features]
y = df[target_col]

# -------------------------------
# 2. Chi-Square test for independence
# -------------------------------
results = []

for feat in binary_features:
    ct = pd.crosstab(y, X[feat])
    try:
        chi2_stat, p_val, dof, expected = chi2_contingency(ct)
    except:
        p_val = np.nan
        chi2_stat = np.nan
        dof = np.nan


    n = ct.sum().sum()
    phi2 = chi2_stat / n
    r, k = ct.shape
    phi2_corr = max(0, phi2 - (k - 1) * (r - 1) / (n - 1))
    r_corr = r - (r - 1)**2 / (n - 1)
    k_corr = k - (k - 1)**2 / (n - 1)
    cramers_v = np.sqrt(phi2_corr / min(k_corr - 1, r_corr - 1)) if min(k_corr - 1, r_corr - 1) > 0 else 0

    results.append({
        'Feature': feat,
        'Chi2': chi2_stat,
        'p-value': p_val,
        'Cramers_V': cramers_v
    })


results_df = pd.DataFrame(results)

# -------------------------------
# 3. Robust FDR correction
# -------------------------------

_, pvals_adj, _, _ = multipletests(results_df['p-value'], method='fdr_bh')  # FDR adjustment
results_df['p-adj'] = pvals_adj
results_df['significant'] = pvals_adj < 0.05

# -------------------------------
# 4. Summary
# -------------------------------
print("Top significant features associated with 'Source':")
print(results_df[results_df['significant']])

print(f"\ntotal {len(binary_features)} , {results_df['significant'].sum()} features are significantly associated with '{target_col}' after FDR correction.")

results_df.insert(0, 'Fingerprint_index', range(len(results_df)))
results_df.to_csv("../output/dim_reduction/phase_vector_chi_square_results.csv", index=False)

# -------------------------------
# 5. Visualization
# -------------------------------
plt.figure(figsize=(10, 6))
scatter = plt.scatter(results_df['Cramers_V'], -np.log10(results_df['p-adj']),
                     c=results_df['significant'], cmap='RdYlGn_r', alpha=0.8, edgecolors='k', linewidth=0.3)
plt.xlabel("Cramér's V (Effect Size)", fontsize=12)
plt.ylabel(r'$-\log_{10}(adjusted\ p\text{-}value)$', fontsize=12)

plt.axhline(-np.log10(0.05), color='gray', linestyle='--', alpha=0.7, label='p-adj = 0.05')
plt.axvline(0.1, color='blue', linestyle='--', alpha=0.5, label="Cramér's V > 0.1 (moderate)")

plt.legend(title="Significant", loc='upper right')
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig("../output/dim_reduction/association_binary_features_source.png", dpi=300, bbox_inches='tight')

results_df = results_df.copy()
results_df['p-adj'] = results_df['p-adj'].clip(lower=1e-200)

results_df['neg_log_padj'] = -np.log10(results_df['p-adj'])

v_bins = [0, 0.1, 0.2, 0.4, 0.6, 0.8, 1.0]
v_labels = ['V: 0.0–0.1', 'V: 0.1–0.2', 'V: 0.2–0.4', 'V: 0.4–0.6', 'V: 0.6–0.8', 'V: 0.8–1']
results_df['v_group'] = pd.cut(results_df['Cramers_V'], bins=v_bins, labels=v_labels, include_lowest=True)
v_colors = ['lightgray', 'skyblue', 'gold', 'orange', '#F08080', 'red']
color_map = dict(zip(v_labels, v_colors))

plt.figure(figsize=(7.2, 4.8))

# Plot each group with different colors
for label, color in color_map.items():
    subset = results_df[results_df['v_group'] == label]
    plt.scatter(subset['Fingerprint_index'], subset['neg_log_padj'],
                c=color, label=label, s=25, alpha=1, edgecolors='none')
    # Vertical lines
    for _, row in subset.iterrows():
        plt.plot([row['Fingerprint_index'], row['Fingerprint_index']], [0, row['neg_log_padj']],
                 color=color, alpha=0.5, linewidth=0.5)


plt.xlabel('Fingerprint Index', fontsize=12)
plt.ylabel('-log10(Adjusted p-value)', fontsize=12)
plt.axhline(-np.log10(0.05), color='gray', linestyle='--', alpha=0.7, linewidth=1, label='FDR = 0.05')

plt.xticks(fontsize=12)
plt.yticks(fontsize=12)
for spine in plt.gca().spines.values():
    spine.set_linewidth(1.5)
ax = plt.gca()
ax.tick_params(axis='both', which='both', top=False, right=False)
ax.tick_params(axis='both', which='major', length=5, width=1.5, direction='out')
ax.tick_params(axis='both', which='minor', length=3, width=0.5, direction='out')

plt.tight_layout()
plt.savefig("../output/dim_reduction/feature_significance_source.png", dpi=300, bbox_inches='tight')

# -------------------------------
# 6. Feature Distribution by Cramér's V Strength
# -------------------------------
v_bins = [0, 0.1, 0.2, 0.4, 0.6, 0.8, 1.0]
v_labels = ['0.0 – 0.1', '0.1 – 0.2', '0.2 – 0.4', '0.4 – 0.6', '0.6 – 0.8', '0.8 – 1']

results_df['v_group'] = pd.cut(results_df['Cramers_V'], bins=v_bins, labels=v_labels, include_lowest=True)

v_counts = results_df['v_group'].value_counts().reindex(v_labels, fill_value=0)
v_props = v_counts / len(results_df) * 100

plt.figure(figsize=(4, 4))
colors = ['lightgray', 'skyblue', 'gold', 'orange', '#F08080', 'red']

# Plot pie chart
wedges, texts, autotexts = plt.pie(v_counts, labels=v_labels, autopct='%1.1f%%',
                                   startangle=90, colors=colors, pctdistance=1.8,
                                   wedgeprops=dict(width=0.5, edgecolor='grey', linewidth=0.5,
                                                   alpha=0.8))

centre_circle = plt.Circle((0,0), 0.5, fc='white')
fig = plt.gcf()
fig.gca().add_artist(centre_circle)

plt.tight_layout()
plt.savefig("../output/dim_reduction/feature_cramers_v_distribution.png", dpi=300, bbox_inches='tight', transparent=True)
