"""
API module for AutoValuePredict ML project.

This module provides REST API endpoints for car price prediction.
"""

from .main import app
from .predictor import CarPricePredictor
from .schemas import (
    CarInput,
    PredictionResponse,
    BatchPredictionRequest,
    BatchPredictionResponse,
    ModelInfoResponse,
    HealthResponse
)

__all__ = [
    'app',
    'CarPricePredictor',
    'CarInput',
    'PredictionResponse',
    'BatchPredictionRequest',
    'BatchPredictionResponse',
    'ModelInfoResponse',
    'HealthResponse'
]
