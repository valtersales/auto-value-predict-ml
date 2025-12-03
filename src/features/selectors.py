"""
Feature selection module for AutoValuePredict ML project.

This module provides utilities for feature selection, including
correlation-based selection, mutual information, and feature importance.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Set
from sklearn.feature_selection import (
    SelectKBest,
    f_regression,
    mutual_info_regression,
    SelectFromModel
)
from sklearn.ensemble import RandomForestRegressor
import logging

logger = logging.getLogger(__name__)


class FeatureSelector:
    """
    Feature selector for regression tasks.
    
    Provides multiple selection methods:
    - Correlation-based selection
    - Mutual information
    - Feature importance from models
    """
    
    def __init__(
        self,
        method: str = 'correlation',
        k: Optional[int] = None,
        correlation_threshold: float = 0.01,
        remove_highly_correlated: bool = True,
        correlation_cutoff: float = 0.95
    ):
        """
        Initialize the feature selector.
        
        Args:
            method: Selection method ('correlation', 'mutual_info', 'importance', 'all')
            k: Number of features to select (None = auto)
            correlation_threshold: Minimum correlation with target
            remove_highly_correlated: Whether to remove highly correlated features
            correlation_cutoff: Threshold for removing highly correlated features
        """
        self.method = method
        self.k = k
        self.correlation_threshold = correlation_threshold
        self.remove_highly_correlated = remove_highly_correlated
        self.correlation_cutoff = correlation_cutoff
        
        self.selected_features: List[str] = []
        self.feature_scores: Dict[str, float] = {}
    
    def fit(self, X: pd.DataFrame, y: pd.Series):
        """
        Fit the feature selector.
        
        Args:
            X: Feature DataFrame
            y: Target series
            
        Returns:
            Self
        """
        logger.info(f"Fitting feature selector using method: {self.method}")
        
        numeric_features = X.select_dtypes(include=[np.number]).columns.tolist()
        
        if self.method == 'correlation':
            self._select_by_correlation(X[numeric_features], y)
        elif self.method == 'mutual_info':
            self._select_by_mutual_info(X[numeric_features], y)
        elif self.method == 'importance':
            self._select_by_importance(X[numeric_features], y)
        elif self.method == 'all':
            # Combine all methods
            self._select_by_correlation(X[numeric_features], y)
            mi_features = self._select_by_mutual_info(X[numeric_features], y)
            imp_features = self._select_by_importance(X[numeric_features], y)
            # Union of all selected features
            all_features = set(self.selected_features) | set(mi_features) | set(imp_features)
            self.selected_features = list(all_features)
        else:
            raise ValueError(f"Unknown method: {self.method}")
        
        # Remove highly correlated features
        if self.remove_highly_correlated:
            self._remove_correlated_features(X[self.selected_features])
        
        logger.info(f"Selected {len(self.selected_features)} features")
        
        return self
    
    def _select_by_correlation(self, X: pd.DataFrame, y: pd.Series) -> List[str]:
        """Select features based on correlation with target."""
        correlations = X.corrwith(y).abs().sort_values(ascending=False)
        self.feature_scores.update(correlations.to_dict())
        
        if self.k:
            selected = correlations.head(self.k).index.tolist()
        else:
            selected = correlations[correlations >= self.correlation_threshold].index.tolist()
        
        self.selected_features = selected
        return selected
    
    def _select_by_mutual_info(self, X: pd.DataFrame, y: pd.Series) -> List[str]:
        """Select features based on mutual information."""
        mi_scores = mutual_info_regression(X, y, random_state=42)
        mi_df = pd.DataFrame({
            'feature': X.columns,
            'score': mi_scores
        }).sort_values('score', ascending=False)
        
        self.feature_scores.update(dict(zip(mi_df['feature'], mi_df['score'])))
        
        if self.k:
            selected = mi_df.head(self.k)['feature'].tolist()
        else:
            # Select features with score > median
            threshold = mi_df['score'].median()
            selected = mi_df[mi_df['score'] >= threshold]['feature'].tolist()
        
        if not self.selected_features:
            self.selected_features = selected
        
        return selected
    
    def _select_by_importance(self, X: pd.DataFrame, y: pd.Series) -> List[str]:
        """Select features based on model importance."""
        # Use Random Forest for feature importance
        rf = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
        rf.fit(X, y)
        
        importance_df = pd.DataFrame({
            'feature': X.columns,
            'importance': rf.feature_importances_
        }).sort_values('importance', ascending=False)
        
        self.feature_scores.update(dict(zip(importance_df['feature'], importance_df['importance'])))
        
        if self.k:
            selected = importance_df.head(self.k)['feature'].tolist()
        else:
            # Select features with importance > mean
            threshold = importance_df['importance'].mean()
            selected = importance_df[importance_df['importance'] >= threshold]['feature'].tolist()
        
        if not self.selected_features:
            self.selected_features = selected
        
        return selected
    
    def _remove_correlated_features(self, X: pd.DataFrame):
        """Remove highly correlated features."""
        corr_matrix = X.corr().abs()
        
        # Find pairs of highly correlated features
        upper_triangle = corr_matrix.where(
            np.triu(np.ones(corr_matrix.shape), k=1).astype(bool)
        )
        
        to_remove = set()
        for col in upper_triangle.columns:
            if col in self.selected_features:
                correlated = upper_triangle.index[upper_triangle[col] > self.correlation_cutoff].tolist()
                # Keep the feature with higher score, remove others
                for corr_feature in correlated:
                    if corr_feature in self.selected_features:
                        score_col = self.feature_scores.get(col, 0)
                        score_corr = self.feature_scores.get(corr_feature, 0)
                        if score_corr > score_col:
                            to_remove.add(col)
                        else:
                            to_remove.add(corr_feature)
        
        self.selected_features = [
            f for f in self.selected_features if f not in to_remove
        ]
        
        if to_remove:
            logger.info(f"Removed {len(to_remove)} highly correlated features")
    
    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Select features from DataFrame.
        
        Args:
            X: Input DataFrame
            
        Returns:
            DataFrame with selected features only
        """
        if not self.selected_features:
            raise ValueError("Selector must be fitted before transform. Call fit() first.")
        
        # Get available features (some might not exist after transformations)
        available_features = [f for f in self.selected_features if f in X.columns]
        
        if len(available_features) < len(self.selected_features):
            missing = set(self.selected_features) - set(available_features)
            logger.warning(f"Some selected features not found: {missing}")
        
        return X[available_features]
    
    def fit_transform(self, X: pd.DataFrame, y: pd.Series) -> pd.DataFrame:
        """Fit and transform in one step."""
        return self.fit(X, y).transform(X)
    
    def get_feature_importance_report(self) -> pd.DataFrame:
        """
        Get feature importance report.
        
        Returns:
            DataFrame with features and their scores
        """
        if not self.feature_scores:
            return pd.DataFrame()
        
        report = pd.DataFrame({
            'feature': list(self.feature_scores.keys()),
            'score': list(self.feature_scores.values())
        }).sort_values('score', ascending=False)
        
        report['selected'] = report['feature'].isin(self.selected_features)
        
        return report

