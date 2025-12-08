"""
Model evaluation module for AutoValuePredict ML project.

This module provides comprehensive evaluation metrics and visualization
tools for regression models.
"""

import numpy as np
import pandas as pd
from typing import Dict, Optional, Tuple
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import logging

from sklearn.metrics import (
    mean_squared_error,
    mean_absolute_error,
    r2_score
)

logger = logging.getLogger(__name__)


class ModelEvaluator:
    """
    Comprehensive model evaluator for regression tasks.
    
    Provides metrics:
    - RMSE (Root Mean Squared Error)
    - MAE (Mean Absolute Error)
    - MAPE (Mean Absolute Percentage Error)
    - R² Score (Coefficient of Determination)
    
    Also provides residual analysis and visualizations.
    """
    
    def __init__(self):
        """Initialize the evaluator."""
        self.metrics: Dict[str, float] = {}
        self.residuals: Optional[np.ndarray] = None
        self.predictions: Optional[np.ndarray] = None
        self.actuals: Optional[np.ndarray] = None
    
    def evaluate(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        store_residuals: bool = True
    ) -> Dict[str, float]:
        """
        Calculate all evaluation metrics.
        
        Args:
            y_true: True target values
            y_pred: Predicted values
            store_residuals: Whether to store residuals for analysis
            
        Returns:
            Dictionary with all metrics
        """
        # Store for residual analysis
        if store_residuals:
            self.actuals = np.array(y_true)
            self.predictions = np.array(y_pred)
            self.residuals = self.actuals - self.predictions
        
        # Calculate metrics
        self.metrics = {
            'rmse': self._rmse(y_true, y_pred),
            'mae': self._mae(y_true, y_pred),
            'mape': self._mape(y_true, y_pred),
            'r2': self._r2(y_true, y_pred)
        }
        
        logger.info(f"Evaluation metrics: RMSE={self.metrics['rmse']:.2f}, "
                   f"MAE={self.metrics['mae']:.2f}, "
                   f"MAPE={self.metrics['mape']:.2%}, "
                   f"R²={self.metrics['r2']:.4f}")
        
        return self.metrics
    
    def _rmse(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """Calculate Root Mean Squared Error."""
        return np.sqrt(mean_squared_error(y_true, y_pred))
    
    def _mae(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """Calculate Mean Absolute Error."""
        return mean_absolute_error(y_true, y_pred)
    
    def _mape(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """
        Calculate Mean Absolute Percentage Error.
        
        Handles division by zero by using a small epsilon.
        """
        y_true = np.array(y_true)
        y_pred = np.array(y_pred)
        
        # Avoid division by zero
        mask = y_true != 0
        if not mask.any():
            logger.warning("All true values are zero. MAPE cannot be calculated.")
            return np.inf
        
        percentage_errors = np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])
        return np.mean(percentage_errors)
    
    def _r2(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """Calculate R² Score."""
        return r2_score(y_true, y_pred)
    
    def plot_residuals(
        self,
        figsize: Tuple[int, int] = (15, 5),
        save_path: Optional[str] = None
    ):
        """
        Create residual analysis plots.
        
        Creates three subplots:
        1. Residuals vs Predicted values
        2. Q-Q plot for residual distribution
        3. Residual distribution histogram
        
        Args:
            figsize: Figure size (width, height)
            save_path: Optional path to save the figure
        """
        if self.residuals is None:
            raise ValueError("No residuals stored. Call evaluate() with store_residuals=True first.")
        
        fig, axes = plt.subplots(1, 3, figsize=figsize)
        
        # 1. Residuals vs Predicted
        axes[0].scatter(self.predictions, self.residuals, alpha=0.5)
        axes[0].axhline(y=0, color='r', linestyle='--')
        axes[0].set_xlabel('Predicted Values')
        axes[0].set_ylabel('Residuals')
        axes[0].set_title('Residuals vs Predicted Values')
        axes[0].grid(True, alpha=0.3)
        
        # 2. Q-Q plot
        stats.probplot(self.residuals, dist="norm", plot=axes[1])
        axes[1].set_title('Q-Q Plot (Residuals)')
        axes[1].grid(True, alpha=0.3)
        
        # 3. Residual distribution
        axes[2].hist(self.residuals, bins=50, edgecolor='black', alpha=0.7)
        axes[2].axvline(x=0, color='r', linestyle='--')
        axes[2].set_xlabel('Residuals')
        axes[2].set_ylabel('Frequency')
        axes[2].set_title('Residual Distribution')
        axes[2].grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Residual plots saved to {save_path}")
        
        return fig
    
    def plot_predictions_vs_actuals(
        self,
        figsize: Tuple[int, int] = (8, 8),
        save_path: Optional[str] = None
    ):
        """
        Plot predicted vs actual values.
        
        Args:
            figsize: Figure size (width, height)
            save_path: Optional path to save the figure
        """
        if self.predictions is None or self.actuals is None:
            raise ValueError("No predictions/actuals stored. Call evaluate() first.")
        
        fig, ax = plt.subplots(figsize=figsize)
        
        # Scatter plot
        ax.scatter(self.actuals, self.predictions, alpha=0.5)
        
        # Perfect prediction line
        min_val = min(self.actuals.min(), self.predictions.min())
        max_val = max(self.actuals.max(), self.predictions.max())
        ax.plot([min_val, max_val], [min_val, max_val], 'r--', lw=2, label='Perfect Prediction')
        
        ax.set_xlabel('Actual Values')
        ax.set_ylabel('Predicted Values')
        ax.set_title('Predicted vs Actual Values')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Predictions vs actuals plot saved to {save_path}")
        
        return fig
    
    def get_residual_statistics(self) -> Dict[str, float]:
        """
        Get statistical summary of residuals.
        
        Returns:
            Dictionary with residual statistics
        """
        if self.residuals is None:
            raise ValueError("No residuals stored. Call evaluate() with store_residuals=True first.")
        
        return {
            'mean': float(np.mean(self.residuals)),
            'std': float(np.std(self.residuals)),
            'min': float(np.min(self.residuals)),
            'max': float(np.max(self.residuals)),
            'median': float(np.median(self.residuals)),
            'skewness': float(stats.skew(self.residuals)),
            'kurtosis': float(stats.kurtosis(self.residuals))
        }
    
    def get_summary_report(self) -> pd.DataFrame:
        """
        Get a summary report with all metrics and residual statistics.
        
        Returns:
            DataFrame with evaluation summary
        """
        report_data = {
            'Metric': [],
            'Value': []
        }
        
        # Add evaluation metrics
        for metric_name, value in self.metrics.items():
            report_data['Metric'].append(metric_name.upper())
            if metric_name == 'mape':
                report_data['Value'].append(f"{value:.2%}")
            elif metric_name == 'r2':
                report_data['Value'].append(f"{value:.4f}")
            else:
                report_data['Value'].append(f"{value:.2f}")
        
        # Add residual statistics
        if self.residuals is not None:
            residual_stats = self.get_residual_statistics()
            for stat_name, value in residual_stats.items():
                report_data['Metric'].append(f"Residual {stat_name.capitalize()}")
                report_data['Value'].append(f"{value:.2f}")
        
        return pd.DataFrame(report_data)


def compare_models(
    model_results: Dict[str, Dict[str, float]],
    save_path: Optional[str] = None
) -> pd.DataFrame:
    """
    Compare multiple models' performance metrics.
    
    Args:
        model_results: Dictionary mapping model names to their metrics
        save_path: Optional path to save comparison plot
        
    Returns:
        DataFrame with comparison results
    """
    comparison_df = pd.DataFrame(model_results).T
    
    # Create comparison plot
    if save_path:
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        metrics_to_plot = ['rmse', 'mae', 'mape', 'r2']
        metric_titles = ['RMSE', 'MAE', 'MAPE', 'R² Score']
        
        for idx, (metric, title) in enumerate(zip(metrics_to_plot, metric_titles)):
            ax = axes[idx // 2, idx % 2]
            comparison_df[metric].plot(kind='bar', ax=ax, color='steelblue')
            ax.set_title(f'{title} Comparison')
            ax.set_ylabel(title)
            ax.set_xlabel('Model')
            ax.tick_params(axis='x', rotation=45)
            ax.grid(True, alpha=0.3, axis='y')
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        logger.info(f"Model comparison plot saved to {save_path}")
    
    return comparison_df

