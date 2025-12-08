"""
Baseline model implementations for AutoValuePredict ML project.

This module provides simple baseline models for regression tasks:
- Mean/Median baseline
- Linear Regression
- Ridge Regression
- Lasso Regression
- Decision Tree
"""

import numpy as np
import pandas as pd
from typing import Dict, Optional, Any
import logging
from time import time

from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.tree import DecisionTreeRegressor

logger = logging.getLogger(__name__)


class MeanBaseline:
    """
    Simple mean baseline model.
    
    Predicts the mean of training target values for all predictions.
    """
    
    def __init__(self):
        """Initialize the mean baseline."""
        self.mean_value: Optional[float] = None
        self.training_time: float = 0.0
    
    def fit(self, X: pd.DataFrame, y: pd.Series):
        """
        Fit the model by calculating the mean of target values.
        
        Args:
            X: Feature DataFrame (not used, but kept for sklearn compatibility)
            y: Target series
            
        Returns:
            Self
        """
        start_time = time()
        self.mean_value = float(y.mean())
        self.training_time = time() - start_time
        
        logger.info(f"Mean baseline fitted. Mean value: {self.mean_value:.2f}")
        return self
    
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """
        Predict using the mean value.
        
        Args:
            X: Feature DataFrame (not used, but kept for sklearn compatibility)
            
        Returns:
            Array of predictions (all equal to mean)
        """
        if self.mean_value is None:
            raise ValueError("Model must be fitted before prediction.")
        
        return np.full(len(X), self.mean_value)
    
    def get_params(self, deep: bool = True) -> Dict[str, Any]:
        """Get model parameters (for sklearn compatibility)."""
        return {}
    
    def set_params(self, **params) -> 'MeanBaseline':
        """Set model parameters (for sklearn compatibility)."""
        return self


class MedianBaseline:
    """
    Simple median baseline model.
    
    Predicts the median of training target values for all predictions.
    """
    
    def __init__(self):
        """Initialize the median baseline."""
        self.median_value: Optional[float] = None
        self.training_time: float = 0.0
    
    def fit(self, X: pd.DataFrame, y: pd.Series):
        """
        Fit the model by calculating the median of target values.
        
        Args:
            X: Feature DataFrame (not used, but kept for sklearn compatibility)
            y: Target series
            
        Returns:
            Self
        """
        start_time = time()
        self.median_value = float(y.median())
        self.training_time = time() - start_time
        
        logger.info(f"Median baseline fitted. Median value: {self.median_value:.2f}")
        return self
    
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """
        Predict using the median value.
        
        Args:
            X: Feature DataFrame (not used, but kept for sklearn compatibility)
            
        Returns:
            Array of predictions (all equal to median)
        """
        if self.median_value is None:
            raise ValueError("Model must be fitted before prediction.")
        
        return np.full(len(X), self.median_value)
    
    def get_params(self, deep: bool = True) -> Dict[str, Any]:
        """Get model parameters (for sklearn compatibility)."""
        return {}
    
    def set_params(self, **params) -> 'MedianBaseline':
        """Set model parameters (for sklearn compatibility)."""
        return self


class BaselineModelTrainer:
    """
    Trainer for baseline models.
    
    Provides a convenient interface to train and evaluate multiple baseline models.
    """
    
    def __init__(self, random_seed: int = 42):
        """
        Initialize the baseline trainer.
        
        Args:
            random_seed: Random seed for reproducibility
        """
        self.random_seed = random_seed
        self.models: Dict[str, Any] = {}
        self.training_times: Dict[str, float] = {}
    
    def train_all_baselines(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_val: Optional[pd.DataFrame] = None,
        y_val: Optional[pd.Series] = None
    ) -> Dict[str, Any]:
        """
        Train all baseline models.
        
        Args:
            X_train: Training features
            y_train: Training target
            X_val: Optional validation features
            y_val: Optional validation target
            
        Returns:
            Dictionary mapping model names to trained models
        """
        logger.info("Training all baseline models...")
        
        # 1. Mean Baseline
        logger.info("Training Mean Baseline...")
        mean_model = MeanBaseline()
        mean_model.fit(X_train, y_train)
        self.models['mean_baseline'] = mean_model
        self.training_times['mean_baseline'] = mean_model.training_time
        
        # 2. Median Baseline
        logger.info("Training Median Baseline...")
        median_model = MedianBaseline()
        median_model.fit(X_train, y_train)
        self.models['median_baseline'] = median_model
        self.training_times['median_baseline'] = median_model.training_time
        
        # 3. Linear Regression
        logger.info("Training Linear Regression...")
        start_time = time()
        linear_model = LinearRegression()
        linear_model.fit(X_train, y_train)
        self.models['linear_regression'] = linear_model
        self.training_times['linear_regression'] = time() - start_time
        
        # 4. Ridge Regression
        logger.info("Training Ridge Regression...")
        start_time = time()
        ridge_model = Ridge(alpha=1.0, random_state=self.random_seed)
        ridge_model.fit(X_train, y_train)
        self.models['ridge_regression'] = ridge_model
        self.training_times['ridge_regression'] = time() - start_time
        
        # 5. Lasso Regression
        logger.info("Training Lasso Regression...")
        start_time = time()
        lasso_model = Lasso(alpha=1.0, random_state=self.random_seed)
        lasso_model.fit(X_train, y_train)
        self.models['lasso_regression'] = lasso_model
        self.training_times['lasso_regression'] = time() - start_time
        
        # 6. Decision Tree
        logger.info("Training Decision Tree...")
        start_time = time()
        tree_model = DecisionTreeRegressor(
            max_depth=10,
            min_samples_split=20,
            min_samples_leaf=10,
            random_state=self.random_seed
        )
        tree_model.fit(X_train, y_train)
        self.models['decision_tree'] = tree_model
        self.training_times['decision_tree'] = time() - start_time
        
        logger.info("All baseline models trained successfully!")
        logger.info(f"Training times: {self.training_times}")
        
        return self.models
    
    def predict_all(
        self,
        X: pd.DataFrame,
        models: Optional[Dict[str, Any]] = None
    ) -> Dict[str, np.ndarray]:
        """
        Generate predictions for all models.
        
        Args:
            X: Feature DataFrame
            models: Optional dictionary of models (uses self.models if not provided)
            
        Returns:
            Dictionary mapping model names to predictions
        """
        if models is None:
            models = self.models
        
        if not models:
            raise ValueError("No models available. Train models first.")
        
        predictions = {}
        for model_name, model in models.items():
            predictions[model_name] = model.predict(X)
        
        return predictions
    
    def get_model_info(self) -> pd.DataFrame:
        """
        Get information about all trained models.
        
        Returns:
            DataFrame with model information
        """
        if not self.models:
            return pd.DataFrame()
        
        info_data = {
            'model_name': [],
            'training_time': [],
            'model_type': []
        }
        
        for name, model in self.models.items():
            info_data['model_name'].append(name)
            info_data['training_time'].append(self.training_times.get(name, 0.0))
            info_data['model_type'].append(type(model).__name__)
        
        return pd.DataFrame(info_data)

