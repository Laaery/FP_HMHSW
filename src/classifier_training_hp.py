#!/usr/bin/env python
# -*- coding:utf-8 -*-
# @Author: LL
# @File：classifier_training_hp.py
"""
Hyperparameter tuning for classifiers.
"""
import os
import sys
import joblib
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.naive_bayes import BernoulliNB
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import LabelEncoder, label_binarize
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    f1_score,
    roc_auc_score
)
from joblib import Parallel, delayed, parallel_backend
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.utils import compute_class_weight
from sklearn.base import clone
from tqdm import tqdm
from xgboost import XGBClassifier
import optuna

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from model.ANN import AnnoyKNNClassifier



def clean_feature_names(features):
    cleaned = []
    for f in features:
        f_str = str(f)
        f_str = f_str.replace('[', '_').replace(']', '_')
        f_str = f_str.replace('<', '_lt_').replace('>', '_gt_')
        cleaned.append(f_str)
    return cleaned


def auc_macro_clean(y_true_encoded, y_pred_proba, multi_class='ovr', average='macro'):
    y_true_encoded = np.asarray(y_true_encoded)
    y_pred_proba = np.asarray(y_pred_proba)

    present_classes = np.unique(y_true_encoded)
    y_score_present = y_pred_proba[:, present_classes]

    y_true_bin = label_binarize(y_true_encoded, classes=present_classes)

    if len(present_classes) == 2:
        y_score_input = y_score_present[:, 1]
        auc = roc_auc_score(y_true_encoded, y_score_input, average=average)
    else:
        auc = roc_auc_score(
            y_true_bin,
            y_score_present,
            multi_class=multi_class,
            average=average
        )

    return auc


def get_sample_weight(y):
    classes = np.unique(y)
    class_weights = compute_class_weight('balanced', classes=classes, y=y)
    return np.array([class_weights[yi] for yi in y])


def evaluate_single_fold(fold_info, X_train, y_train, estimator, params, use_gpu, model_name):
    fold, train_local_idx, val_local_idx = fold_info
    X_fold_train = X_train.iloc[train_local_idx]
    X_fold_val = X_train.iloc[val_local_idx]
    y_fold_train = y_train.iloc[train_local_idx]
    y_fold_val = y_train.iloc[val_local_idx]

    model = clone(estimator)
    model.set_params(**params)
    if use_gpu and model_name.lower().startswith('xgb'):
        model.set_params(device='cuda')

    if model_name == 'XGBoost':
        fold_sample_weights = get_sample_weight(y_fold_train)
        model.fit(X_fold_train, y_fold_train, sample_weight=fold_sample_weights)
    else:
        model.fit(X_fold_train, y_fold_train)

    y_fold_pred = model.predict(X_fold_val)
    y_fold_proba = model.predict_proba(X_fold_val)

    bal_acc = balanced_accuracy_score(y_fold_val, y_fold_pred)
    macro_f1 = f1_score(y_fold_val, y_fold_pred, average='macro')
    macro_auc = roc_auc_score(y_fold_val, y_fold_proba, average='macro', multi_class='ovr')

    total_score = bal_acc + macro_f1 + macro_auc
    return fold, bal_acc, macro_f1, macro_auc, total_score


def train_on_all_scenarios(
        df,
        X_full,
        y_full,
        feature_cols,
        robust_fingerprint_path,
        models_with_search,
        output_dir,
        random_seeds=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
        hpo_trials=30,
        hpo_timeout=None,
        n_jobs=-1,
        use_gpu=False
):
    """
    Train classifiers on all scenarios.
    """
    print("Starting high-performance joint training with auto-scenario & parallel execution...")

    # ==================== 1. Load robust features ====================
    if not os.path.exists(robust_fingerprint_path):
        raise FileNotFoundError(f"Robust fingerprint not found: {robust_fingerprint_path}")
    robust_df = pd.read_csv(robust_fingerprint_path)
    selected_features = robust_df['feature'].dropna().tolist() if 'feature' in robust_df.columns \
        else robust_df.iloc[:, 0].dropna().tolist()
    selected_features = [f for f in selected_features if f in feature_cols]

    selected_features = clean_feature_names(selected_features)
    # Save with joblib
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    joblib.dump(selected_features, os.path.join(output_dir, 'feature_names.pkl'))
    X_full.columns = clean_feature_names(X_full.columns)

    if not selected_features:
        raise ValueError("No valid robust features loaded.")

    print(f"Using {len(selected_features)} robust features.")

    # ==================== 2. Generate data splits for all random seeds ====================
    print("Pre-generating data splits for all random seeds...")
    splits_save_dir = os.path.join(output_dir, "data_splits")
    os.makedirs(splits_save_dir, exist_ok=True)

    pre_splits = {}
    test_samples_wo_noises = {}
    for seed in random_seeds:
        train_idx, test_idx = train_test_split(df.index, test_size=0.2, stratify=df['Index'], random_state=seed)
        pre_splits[seed] = (train_idx, test_idx)
        # Find clean samples in the test set
        test_mask_in_df = df.index.isin(test_idx)
        is_original = df['perturbation_type'] == 'original'
        flip_empty = df['flip_count'].isna()
        is_clean = is_original & flip_empty
        test_clean_mask = test_mask_in_df & is_clean
        test_idx_clean = df.index[test_clean_mask].tolist()

        test_samples_wo_noises[seed] = test_idx_clean

        pkl_path = os.path.join(splits_save_dir, f"split_seed_{seed}.pkl")
        joblib.dump({
            'train_idx': train_idx,
            'test_idx': test_idx,
            'test_clean_idx': test_idx_clean
        }, pkl_path)
        print(f"Data split for seed={seed} saved as PKL: {pkl_path}")
        print(f"  Found {len(test_idx_clean)} clean samples in test set")

    # ==================== 3. Simultaneously train all models on all scenarios ====================
    os.makedirs(output_dir, exist_ok=True)

    all_results = []

    def run_one_experiment(model_name, config, seed):
        estimator = config['estimator']
        param_space = config['param_space']
        train_idx, test_idx = pre_splits[seed]

        X_train = X_full.loc[train_idx][selected_features]
        X_test = X_full.loc[test_idx][selected_features]
        y_train = y_full.loc[train_idx]
        y_test = y_full.loc[test_idx]

        test_idx_clean = test_samples_wo_noises[seed]
        X_test_clean = X_full.loc[test_idx_clean][selected_features]
        y_test_clean = y_full.loc[test_idx_clean]

        # Calculate sample weights
        sample_weights = get_sample_weight(y_train)

        hpo_log_dir = os.path.join(output_dir, "hpo_logs")
        os.makedirs(hpo_log_dir, exist_ok=True)
        hpo_log_file = os.path.join(hpo_log_dir, f"hpo_{model_name}_seed_{seed}.csv")

        hpo_trials_log = []

        def hpo_objective(trial):
            params = {}
            for pname, (space, ptype) in param_space.items():
                if ptype == 'categorical':
                    params[pname] = trial.suggest_categorical(pname, space)
                elif ptype == 'int':
                    low, high = space
                    params[pname] = trial.suggest_int(pname, low, high)
                elif ptype == 'float':
                    low, high = space
                    params[pname] = trial.suggest_float(pname, low, high)
                elif ptype == 'float-log':
                    low, high = space
                    params[pname] = trial.suggest_float(pname, low, high, log=True)

            if use_gpu and model_name.lower().startswith('xgb'):
                params['device'] = 'cuda'

            # --- 5 fold CV ---
            skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
            stratify_col = df.loc[train_idx, 'Index']

            fold_infos = [
                (fold, train_local_idx, val_local_idx)
                for fold, (train_local_idx, val_local_idx) in enumerate(skf.split(X_train, stratify_col))
            ]

            cv_results = Parallel(n_jobs=5, prefer="threads")(
                delayed(evaluate_single_fold)(
                    info,
                    X_train, y_train, estimator, params, use_gpu, model_name
                ) for info in fold_infos
            )

            cv_results.sort(key=lambda x: x[0])

            cv_bal_accs = [res[1] for res in cv_results]
            cv_macro_f1s = [res[2] for res in cv_results]
            cv_macro_aucs = [res[3] for res in cv_results]
            cv_total_scores = [res[4] for res in cv_results]

            # Mean CV scores
            mean_total_score = np.mean(cv_total_scores)

            # --- Log HPO trial ---
            trial_log = {
                'trial_number': trial.number,
                **{f'param_{k}': v for k, v in params.items()},
                'cv_bal_accs': str(cv_bal_accs),
                'cv_macro_f1s': str(cv_macro_f1s),
                'cv_macro_aucs': str(cv_macro_aucs),
                'cv_total_scores': str(cv_total_scores),
                'mean_bal_acc': np.mean(cv_bal_accs),
                'mean_macro_f1': np.mean(cv_macro_f1s),
                'mean_macro_auc': np.mean(cv_macro_aucs),
                'mean_total_score': mean_total_score,
                'std_total_score': np.std(cv_total_scores)
            }
            hpo_trials_log.append(trial_log)

            return mean_total_score

        sampler = optuna.samplers.TPESampler(seed=seed)
        study = optuna.create_study(direction='maximize', sampler=sampler)
        study.optimize(hpo_objective, n_trials=hpo_trials, timeout=hpo_timeout, show_progress_bar=False)

        best_params = study.best_params
        best_val_balanced_acc = study.best_value

        if hpo_trials_log:
            hpo_df = pd.DataFrame(hpo_trials_log)
            hpo_df.to_csv(hpo_log_file, index=False)
            print(f"HPO log saved: {hpo_log_file}")
        else:
            print(f"No HPO trials logged for {model_name}-seed{seed}")

        # --- Training final model ---
        final_model = clone(estimator)
        final_model.set_params(**best_params)
        if use_gpu and model_name.lower().startswith('xgb'):
            final_model.set_params(device='cuda')

        if model_name == 'XGBoost':
            sample_weights_full = get_sample_weight(y_train)
            final_model.fit(X_train, y_train, sample_weight=sample_weights_full)
        else:
            final_model.fit(X_train, y_train)

        y_train_pred = final_model.predict(X_train)
        y_train_pred_proba = final_model.predict_proba(X_train)

        y_test_pred = final_model.predict(X_test)
        y_test_pred_proba = final_model.predict_proba(X_test)

        y_test_pred_clean = final_model.predict(X_test_clean)
        y_test_pred_proba_clean = final_model.predict_proba(X_test_clean)

        metrics = {
            'test_acc_balanced': balanced_accuracy_score(y_test, y_test_pred),
            'test_f1_macro': f1_score(y_test, y_test_pred, average='macro'),
            'test_f1_weighted': f1_score(y_test, y_test_pred, average='weighted'),
            'test_auc': roc_auc_score(y_test, y_test_pred_proba, multi_class='ovr', average='macro'),
            'test_acc_balanced_clean': balanced_accuracy_score(y_test_clean, y_test_pred_clean),
            'test_f1_macro_clean': f1_score(y_test_clean, y_test_pred_clean, average='macro'),
            'test_f1_weighted_clean': f1_score(y_test_clean, y_test_pred_clean, average='weighted'),
            'test_auc_clean': auc_macro_clean(y_test_clean, y_test_pred_proba_clean, multi_class='ovr',
                                              average='macro'),
            'train_acc_balanced': balanced_accuracy_score(y_train, y_train_pred),
            'train_f1_macro': f1_score(y_train, y_train_pred, average='macro'),
            'train_f1_weighted': f1_score(y_train, y_train_pred, average='weighted'),
            'train_auc': roc_auc_score(y_train, y_train_pred_proba, multi_class='ovr', average='macro')
        }

        per_exp_dir = os.path.join(output_dir, "per_experiment")
        os.makedirs(per_exp_dir, exist_ok=True)
        result_file = os.path.join(per_exp_dir, f"{model_name}_seed_{seed}.csv")

        result_data = {
            'metric': list(metrics.keys()),
            'value': list(metrics.values())
        }
        result_df = pd.DataFrame(result_data)
        result_df['model'] = model_name
        result_df['seed'] = seed
        result_df['n_features'] = len(selected_features)
        result_df['best_params'] = str(best_params)
        result_df['features'] = ', '.join(selected_features)

        result_df.to_csv(result_file, index=False)
        print(f"Detailed result saved: {result_file}")

        model_save_dir = os.path.join(output_dir, "models")
        os.makedirs(model_save_dir, exist_ok=True)

        model_filename = f"model_{model_name}_seed_{seed}.pkl"
        model_path = os.path.join(model_save_dir, model_filename)
        if model_name == 'AnnoyKNN':
            os.makedirs(os.path.join(model_save_dir, f"model_{model_name}_seed_{seed}"), exist_ok=True)
            final_model.save(os.path.join(model_save_dir, f"model_{model_name}_seed_{seed}"))
        else:
            joblib.dump(final_model, model_path)
        print(f"Model saved: {model_path}")

        summary_result = {
            'model': model_name,
            'repeat': seed,
            'seed': seed,
            'n_features': len(selected_features),
            'best_params': str(best_params),
            **{k: v for k, v in metrics.items()}
        }

        return summary_result

    # --- Parallel execution ---
    print(f"Starting parallel execution on {n_jobs} cores...")
    args_list = [
        (model_name, config, seed)
        for model_name, config in models_with_search.items()
        for seed in random_seeds
    ]

    with parallel_backend("loky"):
        results = Parallel(n_jobs=n_jobs, verbose=0)(
            delayed(run_one_experiment)(name, config, seed)
            for name, config, seed in tqdm(args_list, desc="Running Experiments", unit="exp")
        )

    # ==================== 4. Save results ====================
    results_df = pd.DataFrame(results)
    results_df.to_csv(os.path.join(output_dir, "detailed_results.csv"), index=False)

    summary = results_df.groupby('model').agg(
        avg_train_acc=('train_acc_balanced', 'mean'),
        std_train_acc=('train_acc_balanced', 'std'),
        avg_train_f1=('train_f1_macro', 'mean'),
        std_train_f1=('train_f1_macro', 'std'),
        avg_train_auc=('train_auc', 'mean'),
        std_train_auc=('train_auc', 'std'),
        avg_test_acc=('test_acc_balanced', 'mean'),
        std_test_acc=('test_acc_balanced', 'std'),
        avg_test_f1=('test_f1_macro', 'mean'),
        std_test_f1=('test_f1_macro', 'std'),
        avg_test_auc=('test_auc', 'mean'),
        std_test_auc=('test_auc', 'std'),
        n_repeats=('seed', 'count')
    ).round(4).reset_index()

    summary.to_csv(os.path.join(output_dir, "summary_results.csv"), index=False)
    print(f"All done. Results saved to {output_dir}")

    return summary.to_dict('records'), results


if __name__ == '__main__':
    DATA_PATH = '../data/phase_vector_fn_fp_mingle_fn_fp_1_2.csv'
    OUTPUT_DIR = '../output/hpo_exp'
    os.makedirs(OUTPUT_DIR, exist_ok=True)
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
    # Save label joblib
    joblib.dump(le, os.path.join(OUTPUT_DIR, 'label_encoder.pkl'))

    print(f"Features: {len(feature_cols)}")
    print(f"Total samples: {len(df)}, Unique Index: {df['Index'].nunique()}")

    # # Define model and hyperparameter search space
    models_with_search = {
        # 1. Logistic Regression
        'LogisticRegression': {
            'estimator': LogisticRegression(max_iter=100, random_state=42, n_jobs=1, class_weight='balanced',
                                            solver='saga'),
            'param_space': {
                'C': ((1e-4, 1e2), 'float-log'),
                'penalty': (['l1', 'l2', 'elasticnet'], 'categorical'),
                'l1_ratio': ((0.1, 0.9), 'float'),
            }
        },
        'AnnoyKNN': {
            'estimator': AnnoyKNNClassifier(n_jobs=1),
            'param_space': {
                'n_neighbors': ((3, 50), 'int'),
                'n_trees': ((10, 50), 'int'),
                'metric': (['euclidean', 'manhattan', 'cosine', 'hamming'], 'categorical'),
                'search_k': ((3, 500), 'int')
            }
        },
        'BernoulliNB': {
            'estimator': BernoulliNB(binarize=None),
            'param_space': {
                'alpha': ((1e-4, 1e2), 'float-log'),
            }
        },
        'MLP': {
            'estimator': MLPClassifier(
                solver='adam',
                max_iter=400,
                shuffle=True,
                random_state=42,
                tol=1e-4,
                early_stopping=True,
                validation_fraction=0.1,
                n_iter_no_change=10,
                verbose=False
            ),
            'param_space': {
                'hidden_layer_sizes': ([16, 32, 64, 128, 256], 'categorical'),
                'activation': (['relu', 'tanh', 'logistic'], 'categorical'),
                'learning_rate_init': ((1e-5, 1e-1), 'float-log'),
                'batch_size': ([32, 64, 128, 256], 'categorical'),
                'alpha': ((1e-4, 1e-1), 'float-log'),
            }
        },

        'RandomForest': {
            'estimator': RandomForestClassifier(random_state=42, n_jobs=1),
            'param_space': {
                'n_estimators': ((50, 300), 'int'),
                'max_depth': ((3, 50), 'int'),
                'min_samples_split': ((2, 20), 'int'),
                'min_samples_leaf': ((1, 10), 'int'),
                'max_features': (['sqrt', 'log2', None], 'categorical'),
                'bootstrap': (([True, False]), 'categorical'),
                'criterion': (['gini', 'entropy'], 'categorical'),
                'class_weight': (['balanced', 'balanced_subsample'], 'categorical')
            }
        },

        'XGBoost': {
            'estimator': XGBClassifier(eval_metric='mlogloss', random_state=42, n_jobs=1),
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

    summary, results = train_on_all_scenarios(
        df=df,
        X_full=X_full,
        y_full=y_full,
        feature_cols=feature_cols,
        robust_fingerprint_path="../output/fn_fp_mingle_fn_fp_bacc_balanced/robust_feature_set_0.99.csv",
        models_with_search=models_with_search,
        output_dir=OUTPUT_DIR,
        random_seeds=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
        hpo_trials=100,
        n_jobs=10,
        use_gpu=False
    )
