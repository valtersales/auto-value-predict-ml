"""
Feature engineering pipeline module for AutoValuePredict ML project.

This module provides a complete pipeline for feature engineering that can be
saved and loaded for inference.
"""

import pandas as pd
import numpy as np
from typing import Dict, Optional, List
from pathlib import Path
import joblib
import logging

from .engineering import FeatureEngineeringPipeline
from .selectors import FeatureSelector

logger = logging.getLogger(__name__)


class FeaturePipeline:
    """
    Complete feature engineering pipeline with persistence.
    
    This class combines feature engineering and feature selection,
    and provides methods to save/load the pipeline for inference.
    """
    
    def __init__(
        self,
        use_feature_selection: bool = False,
        feature_selection_method: str = 'correlation',
        use_advanced_features: bool = False,
        target_col: str = 'price',
        random_seed: int = 42
    ):
        """
        Initialize the feature pipeline.
        
        Args:
            use_feature_selection: Whether to apply feature selection
            feature_selection_method: Method for feature selection
            use_advanced_features: Whether to use advanced features (Phase 3.1.1 - Optional)
            target_col: Name of target column
            random_seed: Random seed for reproducibility
        """
        self.use_feature_selection = use_feature_selection
        self.feature_selection_method = feature_selection_method
        self.use_advanced_features = use_advanced_features
        self.target_col = target_col
        self.random_seed = random_seed
        
        # Initialize components
        self.feature_engineering = FeatureEngineeringPipeline(
            use_advanced_features=use_advanced_features,
            target_col=target_col,
            random_seed=random_seed
        )
        
        self.feature_selector = FeatureSelector(
            method=feature_selection_method
        ) if use_feature_selection else None
        
        self.is_fitted = False
        self.feature_names_: List[str] = []
    
    def fit(self, X: pd.DataFrame, y: Optional[pd.Series] = None):
        """
        Fit the pipeline on training data.
        
        Args:
            X: Training DataFrame
            y: Target series (optional)
            
        Returns:
            Self
        """
        logger.info("Fitting feature pipeline...")
        
        # Fit feature engineering
        X_transformed = self.feature_engineering.fit_transform(X, y)
        
        # Fit feature selection if enabled
        if self.feature_selector and y is not None:
            if isinstance(y, pd.Series):
                y_series = y
            else:
                y_series = pd.Series(y) if not isinstance(y, pd.Series) else y
            
            X_transformed = self.feature_selector.fit_transform(X_transformed, y_series)
            logger.info(f"Feature selection applied: {len(self.feature_selector.selected_features)} features selected")
        
        # Final safety check: ensure all columns are numeric
        numeric_cols = X_transformed.select_dtypes(include=[np.number]).columns.tolist()
        non_numeric_cols = [col for col in X_transformed.columns if col not in numeric_cols]
        
        if non_numeric_cols:
            logger.warning(f"Removing non-numeric columns from training data: {non_numeric_cols}")
            X_transformed = X_transformed[numeric_cols]
        
        # Store feature names
        self.feature_names_ = X_transformed.columns.tolist()
        self.is_fitted = True
        
        logger.info(f"Feature pipeline fitted: {len(self.feature_names_)} features")
        
        return self
    
    def transform(self, X: pd.DataFrame, y: Optional[pd.Series] = None) -> pd.DataFrame:
        """
        Transform data using fitted pipeline.
        
        Args:
            X: Input DataFrame
            y: Target series (optional, only needed for some transformations)
            
        Returns:
            Transformed DataFrame
        """
        if not self.is_fitted:
            raise ValueError("Pipeline must be fitted before transform. Call fit() first.")
        
        logger.info("Transforming features...")
        
        # Apply feature engineering
        X_transformed = self.feature_engineering.transform(X, y)
        
        # Apply feature selection if enabled
        if self.feature_selector:
            X_transformed = self.feature_selector.transform(X_transformed)
        
        # Ensure same column order as training
        missing_cols = set(self.feature_names_) - set(X_transformed.columns)
        extra_cols = set(X_transformed.columns) - set(self.feature_names_)
        
        if missing_cols:
            logger.warning(f"Missing features in transform: {missing_cols}")
            # Add missing columns with zeros
            for col in missing_cols:
                X_transformed[col] = 0
        
        if extra_cols:
            logger.warning(f"Extra features in transform: {extra_cols}")
        
        # Reorder columns to match training
        X_transformed = X_transformed[[col for col in self.feature_names_ if col in X_transformed.columns]]
        
        # Final safety check: ensure all columns are numeric
        numeric_cols = X_transformed.select_dtypes(include=[np.number]).columns.tolist()
        non_numeric_cols = [col for col in X_transformed.columns if col not in numeric_cols]
        
        if non_numeric_cols:
            logger.warning(f"Removing non-numeric columns from final output: {non_numeric_cols}")
            # Only keep columns that are both numeric AND in feature_names_
            # This maintains immutability: feature_names_ is never modified after fit()
            valid_numeric_cols = [col for col in self.feature_names_ if col in numeric_cols]
            X_transformed = X_transformed[valid_numeric_cols]
        
        return X_transformed
    
    def fit_transform(self, X: pd.DataFrame, y: Optional[pd.Series] = None) -> pd.DataFrame:
        """Fit and transform in one step."""
        return self.fit(X, y).transform(X, y)
    
    def save(self, filepath: str):
        """
        Save the fitted pipeline to disk.
        
        Args:
            filepath: Path to save the pipeline
        """
        if not self.is_fitted:
            raise ValueError("Pipeline must be fitted before saving. Call fit() first.")
        
        filepath = Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)
        
        joblib.dump(self, filepath)
        logger.info(f"Feature pipeline saved to {filepath}")
    
    @classmethod
    def load(cls, filepath: str) -> 'FeaturePipeline':
        """
        Load a fitted pipeline from disk.
        
        Args:
            filepath: Path to the saved pipeline
            
        Returns:
            Loaded FeaturePipeline instance
        """
        filepath = Path(filepath)
        if not filepath.exists():
            raise FileNotFoundError(f"Pipeline file not found: {filepath}")
        
        pipeline = joblib.load(filepath)
        logger.info(f"Feature pipeline loaded from {filepath}")
        
        return pipeline
    
    def get_feature_names(self) -> List[str]:
        """
        Get list of feature names after transformation.
        
        Returns:
            List of feature names
        """
        return self.feature_names_.copy()
    
    def get_feature_importance_report(self) -> Optional[pd.DataFrame]:
        """
        Get feature importance report if feature selection is enabled.
        
        Returns:
            DataFrame with feature importance or None
        """
        if self.feature_selector:
            return self.feature_selector.get_feature_importance_report()
        return None
