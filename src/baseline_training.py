#!/usr/bin/env python
# -*- coding:utf-8 -*-
# @Author: LL
# @File：baseline_training.py
import pandas as pd
import numpy as np
import joblib
import os
import sys
from sklearn.model_selection import KFold, StratifiedKFold
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score, precision_score, recall_score, \
    roc_auc_score
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import BernoulliNB
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import label_binarize, LabelEncoder
from xgboost import XGBClassifier
import optuna
import warnings

warnings.filterwarnings("ignore")

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from model.ANN import AnnoyKNNClassifier


def suggest_params(trial, param_space):
    params = {}
    for param, (value_range, dist_type) in param_space.items():
        if dist_type == 'float-log':
            params[param] = trial.suggest_float(param, value_range[0], value_range[1], log=True)
        elif dist_type == 'float':
            params[param] = trial.suggest_float(param, value_range[0], value_range[1])
        elif dist_type == 'int':
            params[param] = trial.suggest_int(param, value_range[0], value_range[1])
        elif dist_type == 'categorical':
            params[param] = trial.suggest_categorical(param, value_range)
        else:
            raise ValueError(f"Unknown distribution: {dist_type}")
    return params


models_with_search = {
    'LogisticRegression': {
        'estimator': LogisticRegression,
        'init_params': {
            'max_iter': 100, 'random_state': 42, 'n_jobs': -1,
            'class_weight': 'balanced', 'solver': 'saga'
        },
        'param_space': {
            'C': ((1e-4, 1e2), 'float-log'),
            'penalty': (['l1', 'l2', 'elasticnet'], 'categorical'),
            'l1_ratio': ((0.1, 0.9), 'float'),
        }
    },
    'AnnoyKNN': {
        'estimator': AnnoyKNNClassifier,
        'init_params': {'n_jobs': 1},
        'param_space': {
            'n_neighbors': ((3, 50), 'int'),
            'n_trees': ((10, 50), 'int'),
            'metric': (['euclidean', 'manhattan', 'cosine', 'hamming'], 'categorical'),
            'search_k': ((3, 500), 'int')
        }
    },
    'BernoulliNB': {
        'estimator': BernoulliNB,
        'init_params': {'binarize': None},
        'param_space': {
            'alpha': ((1e-4, 1e2), 'float-log'),
        }
    },
    'MLP': {
        'estimator': MLPClassifier,
        'init_params': {
            'solver': 'adam', 'max_iter': 400, 'shuffle': True,
            'random_state': 42, 'tol': 1e-4, 'early_stopping': True,
            'validation_fraction': 0.1, 'n_iter_no_change': 10, 'verbose': False
        },
        'param_space': {
            'hidden_layer_sizes': ([16, 32, 64, 128, 256], 'categorical'),
            'activation': (['relu', 'tanh', 'logistic'], 'categorical'),
            'learning_rate_init': ((1e-5, 1e-1), 'float-log'),
            'batch_size': ([32, 64, 128, 256], 'categorical'),
            'alpha': ((1e-4, 1e-1), 'float-log'),
        }
    },
    'RandomForest': {
        'estimator': RandomForestClassifier,
        'init_params': {'random_state': 42, 'n_jobs': -1},
        'param_space': {
            'n_estimators': ((50, 300), 'int'),
            'max_depth': ((3, 50), 'int'),
            'min_samples_split': ((2, 20), 'int'),
            'min_samples_leaf': ((1, 10), 'int'),
            'max_features': (['sqrt', 'log2', None], 'categorical'),
            'bootstrap': ([True, False], 'categorical'),
            'criterion': (['gini', 'entropy'], 'categorical'),
            'class_weight': (['balanced', 'balanced_subsample'], 'categorical')
        }
    },
    'XGBoost': {
        'estimator': XGBClassifier,
        'init_params': {'eval_metric': 'mlogloss', 'random_state': 42, 'n_jobs': -1},
        'param_space': {
            'n_estimators': ((50, 300), 'int'),
            'max_depth': ((3, 50), 'int'),
            'learning_rate': ((1e-3, 5e-1), 'float-log'),
            'subsample': ((0.6, 1.0), 'float'),
            'colsample_bytree': ((0.6, 1.0), 'float'),
            'gamma': ((0.0, 5.0), 'float'),
            'reg_alpha': ((1e-4, 10.0), 'float-log'),
            'reg_lambda': ((1e-4, 10.0), 'float-log'),
            'min_child_weight': ((1, 10), 'int')
        }
    }
}


def auc_macro_clean(y_true, y_pred_proba, model_classes, multi_class='ovr', average='macro'):
    """
    Safe macro AUC computation.

    Parameters:
    - y_true: true labels (encoded, e.g., integers)
    - y_pred_proba: output of model.predict_proba(), shape (n_samples, n_model_classes)
    - model_classes: model.classes_ (the full list of classes seen during training)
    """
    y_true = np.asarray(y_true)
    y_pred_proba = np.asarray(y_pred_proba)
    model_classes = np.asarray(model_classes)

    # Ensure y_pred_proba has correct number of columns
    if y_pred_proba.shape[1] != len(model_classes):
        raise ValueError(
            f"y_pred_proba has {y_pred_proba.shape[1]} columns, "
            f"but model_classes has {len(model_classes)} classes."
        )

    # Check that all y_true are in model_classes
    if not set(y_true).issubset(set(model_classes)):
        print("y_true contains labels not seen during training.")
        return np.nan

    n_classes = len(model_classes)

    if n_classes == 2:
        # Binary classification
        return roc_auc_score(y_true, y_pred_proba[:, 1])
    else:
        # Multi-class: binarize using FULL model_classes
        y_true_bin = label_binarize(y_true, classes=model_classes)
        return roc_auc_score(
            y_true_bin,
            y_pred_proba,
            multi_class=multi_class,
            average=average
        )


def main(csv_path, output_csv):
    df = pd.read_csv(csv_path)

    meta_cols = ['Index', 'perturbation_type', 'flip_count']
    target_col = 'Source'
    feature_cols = [col for col in df.columns if col not in meta_cols + [target_col]]

    df_orig = df[df['perturbation_type'] == 'original']

    value_counts = df_orig[target_col].value_counts()
    rare_classes = value_counts[value_counts <= 2].index

    if len(rare_classes) > 0:
        print(f"Merging {len(rare_classes)} rare classes (≤2 samples) into 'Rare'.")
        df_orig[target_col] = df_orig[target_col].replace(rare_classes, 'Rare')
    else:
        print("No rare classes found (all classes have >2 samples).")

    # Label encode
    le = LabelEncoder()
    df_orig[target_col] = le.fit_transform(df_orig[target_col])

    all_results = []


    for seed in range(10):
        print(f"\nProcessing seed {seed}...")

        split_file = f"../output/hpo_exp_6/data_splits/split_seed_{seed}.pkl"
        if not os.path.exists(split_file):
            print(f"Split file {split_file} not found. Skipping seed {seed}.")
            continue

        split = joblib.load(split_file)
        test_clean_idx = split['test_clean_idx']

        total_indices = set(df_orig.index)
        test_indices = set(test_clean_idx)
        if not test_indices.issubset(total_indices):
            raise ValueError(f"Test indices in seed {seed} not in original data!")

        train_indices = list(total_indices - test_indices)

        X_train = df_orig.loc[train_indices, feature_cols].values
        y_train = df_orig.loc[train_indices, target_col].values
        X_test = df_orig.loc[test_clean_idx, feature_cols].values
        y_test = df_orig.loc[test_clean_idx, target_col].values

        train_classes = set(y_train)
        test_mask = np.array([y in train_classes for y in y_test])
        if not np.any(test_mask):
            print(f"Seed {seed}: No common classes in test set. Skipping.")
            continue
        X_test_clean = X_test[test_mask]
        y_test_clean = y_test[test_mask]

        unique_train = np.unique(y_train)
        is_consecutive = np.array_equal(unique_train, np.arange(len(unique_train)))

        if is_consecutive:
            y_train_fit = y_train
            y_test_fit = y_test_clean
        else:
            label_map = {label: idx for idx, label in enumerate(unique_train)}
            y_train_fit = np.array([label_map[y] for y in y_train], dtype=int)
            y_test_fit = np.array([label_map[y] for y in y_test_clean], dtype=int)

        X_test_for_eval = X_test_clean

        if len(y_train) == 0:
            print(f"Seed {seed}: Empty training set. Skipping.")
            continue
        if len(np.unique(y_train)) < 2:
            print(f"Seed {seed}: Only one class in training set after merging. Skipping.")
            continue

        for model_name, config in models_with_search.items():
            print(f"  Tuning {model_name}...")

            EstimatorClass = config['estimator']
            init_params = config['init_params']
            param_space = config['param_space']

            def objective(trial):
                params = suggest_params(trial, param_space)
                if model_name == 'LogisticRegression':
                    if params['penalty'] != 'elasticnet':
                        params.pop('l1_ratio', None)

                model = EstimatorClass(**init_params, **params)

                cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
                scores = []
                for train_idx, val_idx in cv.split(X_train, y_train):
                    X_tr, X_val = X_train[train_idx], X_train[val_idx]
                    y_tr, y_val = y_train[train_idx], y_train[val_idx]
                    try:
                        model.fit(X_tr, y_tr)
                        y_pred = model.predict(X_val)
                        score = balanced_accuracy_score(y_val, y_pred)
                        scores.append(score)
                    except Exception:
                        return -1.0
                return np.mean(scores) if scores else -1.0

            study = optuna.create_study(direction='maximize')
            study.optimize(objective, n_trials=10, show_progress_bar=False)

            best_params = study.best_params
            if model_name == 'LogisticRegression' and best_params.get('penalty') != 'elasticnet':
                best_params.pop('l1_ratio', None)

            final_model = EstimatorClass(**init_params, **best_params)
            final_model.fit(X_train, y_train_fit)
            y_pred = final_model.predict(X_test_for_eval)
            y_proba = final_model.predict_proba(X_test_for_eval)

            acc = accuracy_score(y_test_fit, y_pred)
            bal_acc = balanced_accuracy_score(y_test_fit, y_pred)
            f1 = f1_score(y_test_fit, y_pred, average='macro')
            auc_macro = auc_macro_clean(y_test_fit, y_proba, multi_class='ovr', average='macro', model_classes=final_model.classes_)


            result = {
                'seed': seed,
                'model': model_name,
                'accuracy': acc,
                'balanced_accuracy': bal_acc,
                'f1_macro': f1,
                'auc_macro': auc_macro,
                'best_params': str(best_params)
            }
            all_results.append(result)


        if all_results:
            results_df = pd.DataFrame(all_results)
            results_df.to_csv(output_csv, index=False)
            print(f"\n✅ All results saved to {output_csv}")
        else:
            print("No results to save.")



if __name__ == "__main__":
    csv_path = "../data/phase_vector_fn_fp_mingle_fn_fp_1_2.csv"
    output_csv = "../output/baseline/baseline_results.csv"
    os.makedirs(os.path.dirname(output_csv), exist_ok=True)
    main(csv_path, output_csv)
