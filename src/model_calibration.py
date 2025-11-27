#!/usr/bin/env python
# -*- coding:utf-8 -*-
# @Author: LL
# @File：model_calibration.py
import random

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.calibration import calibration_curve, CalibratedClassifierCV
from sklearn.metrics import brier_score_loss, log_loss
from sklearn.isotonic import IsotonicRegression
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from model.ANN import AnnoyKNNClassifier
from sklearn.preprocessing import label_binarize, LabelEncoder
import joblib
import os
import scienceplots

plt.style.use(['science', 'no-latex'])
plt.rcParams['font.family'] = 'Arial'

DATA_PATH = '../data/phase_vector_fn_fp_mingle_fn_fp_1_2.csv'
MODEL_DIR = '../output/hpo_exp/models/'
FEATURE_NAMES = '../output/hpo_exp/feature_names.pkl'
DATA_SPLIT_DIR = '../output/hpo_exp/data_splits/'
OUTPUT_DIR = '../output/hpo_exp/calibration_results'
LABEL_COLUMN = 'Source'
MEDIAN_REP_CSV = '../output/hpo_exp/detailed_results_clean_with_median_representatives.csv'

def compute_ece(y_true, y_prob, n_bins=10):
    """Compute Expected Calibration Error (ECE) for multiclass"""
    y_pred = np.argmax(y_prob, axis=1)
    confidences = np.max(y_prob, axis=1)

    bin_boundaries = np.linspace(0, 1, n_bins + 1)
    bin_lowers = bin_boundaries[:-1]
    bin_uppers = bin_boundaries[1:]

    ece = 0.0
    for i in range(n_bins):
        in_bin = (confidences > bin_lowers[i]) & (confidences <= bin_uppers[i])
        prop_in_bin = in_bin.mean()
        if prop_in_bin > 0:
            accuracy_in_bin = (y_true[in_bin] == y_pred[in_bin]).mean()
            avg_confidence_in_bin = confidences[in_bin].mean()
            ece += np.abs(avg_confidence_in_bin - accuracy_in_bin) * prop_in_bin
    return ece


def compute_brier_multiclass(y_true, y_prob):
    """Compute multi-class Brier Score: mean squared error between one-hot labels and predicted probabilities"""
    n_classes = y_prob.shape[1]
    y_true_onehot = np.eye(n_classes)[y_true]  # (n_samples, n_classes)
    return np.mean(np.sum((y_prob - y_true_onehot) ** 2, axis=1))


def plot_multi_model_calibration_curves(
        results_dict,
        n_bins=10,
        title="Multi-Model Aggregated Calibration Curve (Macro OvR)",
        save_path=None
):
    """
    Plot aggregated calibration curves for multiple models on the same plot.

    results_dict: dict of {'Model Name': {'y_true': ..., 'y_prob': ..., 'ECE': ..., 'Brier': ...}}
    """
    plt.figure(figsize=(6, 5))

    for idx, (model_name, result) in enumerate(results_dict.items()):
        y_true = result['y_true']
        y_prob = result['y_prob']

        n_classes = y_prob.shape[1]
        y_true_bin = label_binarize(y_true, classes=np.arange(n_classes))
        all_y_true = np.concatenate([y_true_bin[:, i] for i in range(n_classes)])
        all_y_prob = np.concatenate([y_prob[:, i] for i in range(n_classes)])

        prob_true, prob_pred = calibration_curve(all_y_true, all_y_prob, n_bins=n_bins, strategy="uniform")

        ece = result.get('ECE', compute_ece(y_true, y_prob))
        brier = result.get('Brier', compute_brier_multiclass(y_true, y_prob))
        plt.plot(
            prob_pred, prob_true, 'o-',
            label=f"{model_name} (ECE={ece:.4f}, Brier={brier:.4f})",
            alpha=0.8, markersize=8, linewidth=1.5
        )

    plt.plot([0, 1], [0, 1], "k:", label="Perfect Calibration", linewidth=2)

    plt.xlim(0, 1)
    plt.ylim(0, 1)
    plt.xlabel("Mean Predicted Probability", fontsize=12)
    plt.ylabel("Fraction of Positives", fontsize=12)
    plt.legend(fontsize=9, loc='lower right', ncol=1)
    plt.tight_layout()
    ax = plt.gca()
    for spine in ax.spines.values():
        spine.set_linewidth(2)
    ax.tick_params(axis='both', which='both', top=False, right=False)
    ax.tick_params(axis='both', which='major', length=5, width=1.5, direction='out')

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()


def plot_per_model_two_seeds_subplots(median_results_dict, n_bins=10, save_path=None):
    model_types = list(median_results_dict.keys())
    n_models = len(model_types)
    ncols = 3
    nrows = (n_models + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(4.5 * ncols, 4.5 * nrows))
    if nrows == 1:
        axes = axes.reshape(1, -1)
    axes = axes.flatten()

    cmap = plt.cm.Set1
    for idx, model_name in enumerate(model_types):
        ax = axes[idx]
        seed_results = median_results_dict[model_name]

        for i, res in enumerate(seed_results):
            y_true = res['y_true']
            y_prob = res['y_prob']

            n_classes = y_prob.shape[1]
            y_true_bin = label_binarize(y_true, classes=np.arange(n_classes))
            all_y_true = np.concatenate([y_true_bin[:, j] for j in range(n_classes)])
            all_y_prob = np.concatenate([y_prob[:, j] for j in range(n_classes)])

            prob_true, prob_pred = calibration_curve(all_y_true, all_y_prob, n_bins=n_bins, strategy="uniform")

            ece = compute_ece(y_true, y_prob)

            ax.plot(prob_pred, prob_true, 'o-', label=f"Seed {res['seed']} (ECE={ece:.4f})",
                    color=cmap(i), markersize=6, linewidth=1.5)

        ax.plot([0, 1], [0, 1], "k:", label="Perfect Calibration", linewidth=1.5)
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.set_title(f"{model_name}", fontsize=12, fontweight='bold')
        ax.set_xlabel("Mean Predicted Probability", fontsize=12, fontweight='bold')
        ax.set_ylabel("Fraction of Positives", fontsize=12, fontweight='bold')
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.2)
        for spine in ax.spines.values():
            spine.set_linewidth(1.5)
        ax.tick_params(axis='both', which='both', top=False, right=False)
        ax.tick_params(axis='both', which='major', length=5, width=1, direction='out')
        ax.tick_params(axis='both', which='minor', length=3, width=0.5, direction='out')

    for idx in range(n_models, len(axes)):
        fig.delaxes(axes[idx])

    plt.tight_layout()
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()


def plot_calibration_comparison_isotonic(
        model,
        X_calib, y_calib,
        X_test_final, y_test_final,
        model_name,
        seed,
        n_bins=10,
        save_path=None
):

    y_prob_calib = model.predict_proba(X_calib)
    y_prob_test_raw = model.predict_proba(X_test_final)
    n_classes = y_prob_test_raw.shape[1]

    ir_models = []
    for i in range(n_classes):
        ir = IsotonicRegression(out_of_bounds='clip')
        y_binary = (y_calib == i).astype(int)
        ir.fit(y_prob_calib[:, i], y_binary)
        ir_models.append(ir)

    y_prob_test_calibrated = np.array([
        ir_models[i].predict(y_prob_test_raw[:, i]) for i in range(n_classes)
    ]).T

    ece_raw = compute_ece(y_test_final, y_prob_test_raw)
    ece_cal = compute_ece(y_test_final, y_prob_test_calibrated)
    brier_raw = compute_brier_multiclass(y_test_final, y_prob_test_raw)
    brier_cal = compute_brier_multiclass(y_test_final, y_prob_test_calibrated)

    y_true_bin = label_binarize(y_test_final, classes=np.arange(n_classes))
    y_raw_flat = np.concatenate([y_prob_test_raw[:, i] for i in range(n_classes)])
    y_cal_flat = np.concatenate([y_prob_test_calibrated[:, i] for i in range(n_classes)])
    y_true_flat = np.concatenate([y_true_bin[:, i] for i in range(n_classes)])

    prob_true_raw, prob_pred_raw = calibration_curve(
        y_true_flat, y_raw_flat, n_bins=n_bins, strategy="uniform"
    )
    prob_true_cal, prob_pred_cal = calibration_curve(
        y_true_flat, y_cal_flat, n_bins=n_bins, strategy="uniform"
    )

    plt.figure(figsize=(6, 5))
    plt.plot(prob_pred_raw, prob_true_raw, 'o-', label=f'Before (ECE={ece_raw:.4f}, Brier={brier_raw:.4f})',
             color='tab:red', markersize=6, linewidth=1.8, alpha=0.8)
    plt.plot(prob_pred_cal, prob_true_cal, 's-', label=f'After (ECE={ece_cal:.4f}, Brier={brier_raw:.4f})',
             color='tab:blue', markersize=6, linewidth=1.8, alpha=0.8)
    plt.plot([0, 1], [0, 1], 'k--', label='Perfect Calibration', linewidth=1.5)

    plt.xlim(0, 1)
    plt.ylim(0, 1)
    plt.xlabel("Mean Predicted Probability", fontsize=12)
    plt.ylabel("Fraction of Positives", fontsize=12)
    plt.title(f"{model_name} (seed {seed})\nReliability Curves Before and After Isotonic Calibration", fontsize=12)
    plt.legend(fontsize=9)
    plt.grid(True, alpha=0.3)
    ax = plt.gca()
    for spine in ax.spines.values():
        spine.set_linewidth(1.5)
    ax.tick_params(axis='both', which='both', top=False, right=False)
    ax.tick_params(axis='both', which='major', length=5, width=1, direction='out')
    ax.tick_params(axis='both', which='minor', length=3, width=0.5, direction='out')

    plt.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()

    return y_prob_test_calibrated

if __name__ == "__main__":
    print("Loading full dataset...")
    df = pd.read_csv(DATA_PATH)

    feature_cols = joblib.load(FEATURE_NAMES)
    X_full = df[feature_cols].values
    y_full_raw = df['Source'].values

    le = LabelEncoder()
    y_full = le.fit_transform(y_full_raw)
    n_classes = len(le.classes_)
    print(f"Number of classes: {n_classes}")

    model_types = [
        'LogisticRegression',
        'AnnoyKNN',
        'BernoulliNB',
        'MLP',
        'RandomForest',
        'XGBoost'
    ]

    all_results = []

    print("Loading median representative seeds...")
    rep_df = pd.read_csv(MEDIAN_REP_CSV)
    median_rep_seeds = {}
    for model_type in model_types:
        seeds = rep_df[(rep_df['model'] == model_type) & (rep_df['is_median_representative'])]['seed'].tolist()
        median_rep_seeds[model_type] = seeds
        print(f"{model_type}: {sorted(seeds)}")

    # ========== Reliability Curve Comparison ==========
    median_results_dict = {}

    for model_type in model_types:
        seeds_to_use = median_rep_seeds[model_type]
        model_list = []
        for seed in seeds_to_use:
            split_data = joblib.load(os.path.join(DATA_SPLIT_DIR, f"split_seed_{seed}.pkl"))
            test_idx = split_data['test_idx']
            y_test = y_full[test_idx]
            X_test = X_full[test_idx]

            model_path = os.path.join(MODEL_DIR, f"model_{model_type}_seed_{seed}")
            if model_type == 'AnnoyKNN':
                model = AnnoyKNNClassifier.load(model_path)
            else:
                model = joblib.load(model_path + ".pkl")
            y_prob = model.predict_proba(X_test)

            model_list.append({
                'seed': seed,
                'y_true': y_test,
                'y_prob': y_prob
            })
        median_results_dict[model_type] = model_list

    plot_per_model_two_seeds_subplots(
        median_results_dict,
        save_path=f"{OUTPUT_DIR}/per_model_calibration_medians_6subplots.png"
    )
