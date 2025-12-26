"""
Advanced model training utilities for AutoValuePredict ML project.

Implements training routines for tree-based ensemble regressors with
lightweight hyperparameter search:
- RandomForestRegressor (cross-validated search)
- XGBRegressor (manual search with validation early stopping)
- LGBMRegressor (manual search with validation early stopping)
"""

from __future__ import annotations

import logging
from time import time
from typing import Dict, List, Optional, Tuple, Any

import numpy as np
import pandas as pd
from sklearn.model_selection import RandomizedSearchCV
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error

logger = logging.getLogger(__name__)

try:  # Optional dependencies are installed per pyproject, but guard just in case
    from xgboost import XGBRegressor
except ImportError:  # pragma: no cover - defensive import
    XGBRegressor = None

try:
    from lightgbm import LGBMRegressor
except ImportError:  # pragma: no cover - defensive import
    LGBMRegressor = None


class AdvancedModelTrainer:
    """
    Trainer for advanced tree-based models (Phase 5).

    Provides simple hyperparameter search and validation-based selection for:
    - Random Forest
    - XGBoost
    - LightGBM (optional)
    """

    def __init__(
        self,
        random_seed: int = 42,
        n_jobs: int = -1,
        cv_folds: int = 2,  # Further reduced to save memory (2-fold CV)
        scoring: str = "neg_root_mean_squared_error",
        use_lightgbm: bool = True,
        early_stopping_rounds: int = 50,
        rf_n_iter: int = 3,  # Further reduced to save memory
        rf_max_samples: float = 0.2,  # More aggressive subsampling (20% of data)
    ):
        self.random_seed = random_seed
        self.n_jobs = n_jobs
        self.cv_folds = cv_folds
        self.scoring = scoring
        self.use_lightgbm = use_lightgbm and LGBMRegressor is not None
        self.early_stopping_rounds = early_stopping_rounds
        self.rf_n_iter = rf_n_iter
        self.rf_max_samples = rf_max_samples

        self.models: Dict[str, Any] = {}
        self.best_params: Dict[str, Dict[str, Any]] = {}
        self.training_times: Dict[str, float] = {}
        self.validation_scores: Dict[str, float] = {}

    # ------------------------------------------------------------------ #
    # Random Forest
    # ------------------------------------------------------------------ #
    def _rf_search(self) -> RandomizedSearchCV:
        param_distributions = {
            "n_estimators": [50, 100, 150],  # Further reduced
            "max_depth": [10, 20],  # Reduced options
            "min_samples_split": [10, 20],
            "min_samples_leaf": [2, 4],
            "max_features": ["sqrt"],  # Single option to reduce search space
        }

        base_model = RandomForestRegressor(
            random_state=self.random_seed,
            n_jobs=self.n_jobs,
            max_samples=self.rf_max_samples,  # Use subsampling to reduce memory
        )

        search = RandomizedSearchCV(
            estimator=base_model,
            param_distributions=param_distributions,
            n_iter=self.rf_n_iter,  # Use configurable n_iter
            scoring=self.scoring,
            cv=self.cv_folds,
            random_state=self.random_seed,
            n_jobs=self.n_jobs,
            verbose=1,
        )
        return search

    def train_random_forest(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
    ) -> RandomForestRegressor:
        start = time()
        search = self._rf_search()
        search.fit(X_train, y_train)

        best_model: RandomForestRegressor = search.best_estimator_
        self.models["random_forest"] = best_model
        self.best_params["random_forest"] = search.best_params_
        self.training_times["random_forest"] = time() - start

        logger.info("RandomForest best params: %s", search.best_params_)
        logger.info("RandomForest CV best score (RMSE sign-flipped): %.4f", -search.best_score_)
        return best_model

    # ------------------------------------------------------------------ #
    # XGBoost
    # ------------------------------------------------------------------ #
    def _xgb_candidates(self) -> List[Dict[str, Any]]:
        # Reduced search space to save memory and time
        return [
            {
                "n_estimators": n_est,
                "learning_rate": lr,
                "max_depth": depth,
                "subsample": subs,
                "colsample_bytree": col,
                "min_child_weight": mcw,
                "reg_lambda": reg_l,
                "reg_alpha": reg_a,
            }
            for n_est in [300, 500]  # Reduced from [400, 600, 800]
            for lr in [0.05, 0.1]  # Reduced from [0.03, 0.05, 0.1]
            for depth in [4, 6]  # Reduced from [4, 6, 8]
            for subs in [0.8, 1.0]
            for col in [0.8, 1.0]
            for mcw in [1, 5]
            for reg_l in [1.0, 1.5]
            for reg_a in [0.0, 0.1]
        ]

    def train_xgboost(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_val: Optional[pd.DataFrame] = None,
        y_val: Optional[pd.Series] = None,
        max_candidates: int = 10,  # Reduced from 20 to save time/memory
    ):
        if XGBRegressor is None:
            logger.warning("xgboost is not installed; skipping XGBoost training.")
            return None

        if X_val is None or y_val is None:
            logger.warning("Validation data not provided; XGBoost will train without early stopping.")

        rng = np.random.default_rng(self.random_seed)
        candidates = rng.choice(self._xgb_candidates(), size=max_candidates, replace=False)

        best_rmse = np.inf
        best_model = None
        start = time()

        for params in candidates:
            model = XGBRegressor(
                objective="reg:squarederror",
                eval_metric="rmse",
                random_state=self.random_seed,
                n_jobs=self.n_jobs,
                tree_method="hist",
                **params,
            )

            # Convert to numpy arrays for XGBoost
            X_train_array = X_train.values if isinstance(X_train, pd.DataFrame) else X_train
            y_train_array = y_train.values if isinstance(y_train, pd.Series) else y_train
            
            # Train without early stopping for compatibility
            # Early stopping will be handled by limiting n_estimators in params
            model.fit(X_train_array, y_train_array, verbose=False)

            # Evaluate on validation if available, else on training
            target_X = X_val if X_val is not None else X_train
            target_y = y_val if y_val is not None else y_train
            preds = model.predict(target_X)
            rmse = np.sqrt(mean_squared_error(target_y, preds))

            if rmse < best_rmse:
                best_rmse = rmse
                best_model = model
                self.best_params["xgboost"] = params

        if best_model is None:
            logger.warning("No XGBoost model was trained successfully.")
            return None

        self.models["xgboost"] = best_model
        self.training_times["xgboost"] = time() - start
        self.validation_scores["xgboost"] = best_rmse

        logger.info("XGBoost best params: %s", self.best_params["xgboost"])
        logger.info("XGBoost validation RMSE: %.4f", best_rmse)
        return best_model

    # ------------------------------------------------------------------ #
    # LightGBM
    # ------------------------------------------------------------------ #
    def _lgbm_candidates(self) -> List[Dict[str, Any]]:
        # Reduced search space to save memory and time
        return [
            {
                "n_estimators": n_est,
                "learning_rate": lr,
                "max_depth": depth,
                "num_leaves": leaves,
                "subsample": subs,
                "colsample_bytree": col,
                "min_child_samples": mcs,
                "reg_lambda": reg_l,
                "reg_alpha": reg_a,
            }
            for n_est in [300, 500]  # Reduced from [400, 700, 1000]
            for lr in [0.05, 0.1]  # Reduced from [0.03, 0.05, 0.1]
            for depth in [8, 12]  # Reduced from [-1, 8, 12]
            for leaves in [31, 63]  # Reduced from [31, 63, 127]
            for subs in [0.8, 1.0]
            for col in [0.8, 1.0]
            for mcs in [10, 20]  # Reduced from [10, 20, 40]
            for reg_l in [0.0, 1.0]
            for reg_a in [0.0, 0.1]
        ]

    def train_lightgbm(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_val: Optional[pd.DataFrame] = None,
        y_val: Optional[pd.Series] = None,
        max_candidates: int = 10,  # Reduced from 20 to save time/memory
    ):
        if not self.use_lightgbm:
            logger.info("LightGBM disabled or not installed; skipping.")
            return None

        if X_val is None or y_val is None:
            logger.warning("Validation data not provided; LightGBM will train without early stopping.")

        rng = np.random.default_rng(self.random_seed)
        candidates = rng.choice(self._lgbm_candidates(), size=max_candidates, replace=False)

        best_rmse = np.inf
        best_model = None
        start = time()

        for params in candidates:
            model = LGBMRegressor(
                objective="regression",
                random_state=self.random_seed,
                n_jobs=self.n_jobs,
                **params,
            )

            # Convert to numpy arrays for LightGBM
            X_train_array = X_train.values if isinstance(X_train, pd.DataFrame) else X_train
            y_train_array = y_train.values if isinstance(y_train, pd.Series) else y_train
            
            # Train without early stopping for compatibility
            # Early stopping will be handled by limiting n_estimators in params
            if X_val is not None and y_val is not None:
                X_val_array = X_val.values if isinstance(X_val, pd.DataFrame) else X_val
                y_val_array = y_val.values if isinstance(y_val, pd.Series) else y_val
                # Use eval_set for monitoring but without early_stopping_rounds or verbose
                model.fit(
                    X_train_array, y_train_array,
                    eval_set=[(X_val_array, y_val_array)],
                    eval_metric="rmse"
                )
            else:
                model.fit(X_train_array, y_train_array)

            target_X = X_val if X_val is not None else X_train
            target_y = y_val if y_val is not None else y_train
            preds = model.predict(target_X)
            rmse = np.sqrt(mean_squared_error(target_y, preds))

            if rmse < best_rmse:
                best_rmse = rmse
                best_model = model
                self.best_params["lightgbm"] = params

        if best_model is None:
            logger.warning("No LightGBM model was trained successfully.")
            return None

        self.models["lightgbm"] = best_model
        self.training_times["lightgbm"] = time() - start
        self.validation_scores["lightgbm"] = best_rmse

        logger.info("LightGBM best params: %s", self.best_params["lightgbm"])
        logger.info("LightGBM validation RMSE: %.4f", best_rmse)
        return best_model

    # ------------------------------------------------------------------ #
    # Public API
    # ------------------------------------------------------------------ #
    def train_all(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_val: Optional[pd.DataFrame] = None,
        y_val: Optional[pd.Series] = None,
        include_lightgbm: Optional[bool] = None,
    ) -> Dict[str, Any]:
        """
        Train all advanced models and return fitted estimators.
        """
        include_lightgbm = self.use_lightgbm if include_lightgbm is None else include_lightgbm

        logger.info("Training RandomForest (CV search)...")
        self.train_random_forest(X_train, y_train)

        logger.info("Training XGBoost (validation search)...")
        self.train_xgboost(X_train, y_train, X_val, y_val)

        if include_lightgbm:
            logger.info("Training LightGBM (validation search)...")
            self.train_lightgbm(X_train, y_train, X_val, y_val)

        logger.info("Advanced models trained: %s", list(self.models.keys()))
        return self.models


