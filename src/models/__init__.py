"""
Models module for AutoValuePredict ML project.

This module contains model implementations, evaluation tools, and training utilities.
"""

from .baseline import (
    MeanBaseline,
    MedianBaseline,
    BaselineModelTrainer
)

from .evaluator import (
    ModelEvaluator,
    compare_models
)

__all__ = [
    'MeanBaseline',
    'MedianBaseline',
    'BaselineModelTrainer',
    'ModelEvaluator',
    'compare_models'
]

