"""
CatBoost agent implementation for the multi-agent dropout prediction system
"""
import pandas as pd
import numpy as np
import joblib
import os
from utils.logger import log
from models.tuning import tune_catboost_with_optuna


class CatAgentTS:
    """
    CatBoost agent for time series data with preprocessing and model persistence
    """
    def __init__(self, X_train, y_train, groups, cols, name, dataset_name,n_students, retune=False):
        self.cols = [c for c in cols if c in X_train.columns]
        self.name = name
        self.groups = groups
        self.model_path = f"model_save/{dataset_name}/{n_students}_students/{name}_agent_model_{dataset_name}.joblib" # model_save/{f_out}/stacker_model_{f_out}_{n_students}
        self.model = None

        if not self.cols:
            log(f"Warning: No columns for {self.name}, creating a dummy agent.")
            return

        if X_train[self.cols].std().sum() < 1e-9:
            log(f"--> WARNING: All features for agent '{self.name}' are constant. Skipping training.")
            return

        if os.path.exists(self.model_path) and not retune:
            log(f"Loading pre-trained model for {self.name} from {self.model_path}")
            self.model = joblib.load(self.model_path)
        else:
            log(f"No pre-trained model found or retuning forced. Training new model for {self.name}...")
            
            # FIX: Clean up any old database file BEFORE starting
            if os.path.exists("catboost_tuning.db"):
                os.remove("catboost_tuning.db")

            use_gpu = True  # This would be determined by torch.cuda.is_available() in main
            self.model = tune_catboost_with_optuna(X_train[self.cols], y_train, self.groups, use_gpu, n_trials=100)
            log(f"Saving trained model to {self.name}_agent_model_{dataset_name}.joblib")
            joblib.dump(self.model, self.model_path)

    def risk(self, row):
        if self.model is None:
            return 0.5
        return self.model.predict_proba(row[self.cols].to_frame().T)[0, 1]

    def rat(self, row):
        if self.model is None:
            return f"Analysis unavailable for {self.name} due to lack of dynamic data."

        # Import SHAP here to avoid issues if it's not available
        try:
            import shap
            expl = shap.TreeExplainer(self.model, model_output="raw")
            sv = expl.shap_values(row[self.cols].to_frame().T)[0]
            j = int(np.argmax(np.abs(sv)))
            return friendly_ts(self.cols[j], row[self.cols[j]], sv[j])
        except ImportError:
            return f"Analysis unavailable for {self.name} due to missing SHAP library."


def friendly_ts(f, val, sv):
    dir_ = "increase" if sv > 0 else "decrease"
    sign = "+" if sv > 0 else "−"
    FEAT_TS = {
        "clicks_mean": "average daily clicks",
        "clicks_trend": "click trend (increasing/decreasing)",
        "assessment_score_mean": "average assessment score",
        "days_since_last_activity": "days since last VLE activity",
        "engagement_consistency": "consistency of engagement",
        "clicks_recent_vs_early": "recent vs early activity ratio"
    }
    desc = FEAT_TS.get(f, f)
    return f"{desc}={val:.3f} tends to **{dir_}** dropout risk ({sign}{abs(sv):.1%})."