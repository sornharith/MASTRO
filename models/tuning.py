"""
Hyperparameter tuning functions for the multi-agent dropout prediction system
"""
import numpy as np
import optuna
import gc
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.metrics import roc_auc_score
from catboost import CatBoostClassifier
from utils.logger import log


def tune_catboost_with_optuna(X, y, groups, use_gpu, n_trials=100):
    """
    Uses Optuna to find the best hyperparameters for CatBoost.
    Uses StratifiedGroupKFold to ensure balanced and grouped splits.
    """
    # Create a groups array based on student_id for the split
    # Note: This assumes 'student_id' is in the index or a column of the original df

    if X.empty or (X.nunique().sum() == len(X.columns)):
        log(f"--> WARNING: All features for a training split are constant. Returning dummy model.")
        return CatBoostClassifier(iterations=1, verbose=False)

    def objective(trial):
        params = {
            'iterations': 1500,
            'depth': trial.suggest_int('depth', 3, 10),
            'learning_rate': trial.suggest_float('learning_rate', 0.005, 0.2, log=True),
            'l2_leaf_reg': trial.suggest_float('l2_leaf_reg', 1.0, 9.0, log=True),
            'bootstrap_type': trial.suggest_categorical('bootstrap_type', ['Bernoulli', 'MVS']),
            'subsample': trial.suggest_float('subsample', 0.8, 1.0),
            'random_state': 42,
            'task_type': "CPU",
            'verbose': False
        }
        
        # FIX: Use StratifiedGroupKFold for robust splitting
        cv = StratifiedGroupKFold(n_splits=5)
        aucs = []
        for train_idx, val_idx in cv.split(X, y, groups):
            X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
            y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]

            if y_train.nunique() < 2:
                continue
            if X_train.empty or (X_train.nunique().sum() == len(X_train.columns)):
                continue

            model = CatBoostClassifier(**params)
            model.fit(X_train, y_train, eval_set=[(X_val, y_val)], early_stopping_rounds=150, verbose=False)
            preds = model.predict_proba(X_val)[:, 1]
            aucs.append(roc_auc_score(y_val, preds))

        if not aucs:
             raise optuna.exceptions.TrialPruned("All folds skipped due to single-class targets.")
        return np.mean(aucs)

    storage_name = "sqlite:///catboost_tuning.db"
    study_name = "catboost-hyperparameter-study"
    study = optuna.create_study(
        direction='maximize', storage=storage_name,
        study_name=study_name, load_if_exists=True
    )
    try:
        study.optimize(objective, n_trials=n_trials, n_jobs=-1)
    except optuna.exceptions.TrialPruned as e:
        log(f"--> WARNING: Optuna study pruned for an agent. {e}. Using dummy model.")
        return CatBoostClassifier(iterations=1, verbose=False)

    final_params = study.best_params
    final_params.update({
        'iterations': 2000, 'random_state': 42,
        'task_type': "GPU" if use_gpu else "CPU",
        'devices': "0" if use_gpu else None,
        'auto_class_weights': "Balanced",
    })
    best_model = CatBoostClassifier(**final_params)
    best_model.fit(X, y, verbose=False)
    
    del study
    gc.collect()
    
    return best_model


def get_cluster_stats(X_train, y_train, available_cols):
    """
    Get cluster statistics for course difficulty agent
    """
    from sklearn.cluster import KMeans
    import pandas as pd
    
    # ───────── Course-Difficulty agent with real cluster stats ─────────
    if available_cols:
        # fit K-Means
        km_ts = KMeans(n_clusters=6, random_state=42, n_init=10).fit(X_train[available_cols])

        # compute dropout prevalence per cluster (correct grouping)
        labels = km_ts.labels_
        prev_ts = (pd.DataFrame({"label": labels,
                                "dropout": y_train.values})
                .groupby("label")["dropout"]
                .mean())

        # pretty dict for prompt
        cluster_stats = {int(k): round(float(v), 3) for k, v in prev_ts.items()}

        def risk_diff_ts(row):
            lbl = km_ts.predict(row[available_cols].to_frame().T)[0]
            return float(prev_ts.get(lbl, 0.5))      # fallback 0.5 if unseen

        def rat_diff_ts(row):
            return f"Cluster dropout μ={risk_diff_ts(row):.2f}"
    else:
        cluster_stats = {}
        def risk_diff_ts(row): return 0.5
        def rat_diff_ts(row): return "Clustering unavailable."
    
    return cluster_stats, risk_diff_ts, rat_diff_ts