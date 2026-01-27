"""
Error handling module for AutoValuePredict ML API.

This module provides custom exception classes and error handlers for the API.
"""

from typing import Optional, Dict, Any
from fastapi import HTTPException, Request, status
from fastapi.responses import JSONResponse
import logging

logger = logging.getLogger(__name__)


class ModelNotFoundError(Exception):
    """Raised when a model cannot be found."""
    pass


class ModelLoadError(Exception):
    """Raised when a model fails to load."""
    pass


class FeatureTransformationError(Exception):
    """Raised when feature transformation fails."""
    pass


class PredictionError(Exception):
    """Raised when prediction fails."""
    pass


class ValidationError(Exception):
    """Raised when input validation fails."""
    pass


async def model_not_found_handler(request: Request, exc: ModelNotFoundError) -> JSONResponse:
    """Handle ModelNotFoundError exceptions."""
    logger.error(f"Model not found: {str(exc)}")
    return JSONResponse(
        status_code=status.HTTP_404_NOT_FOUND,
        content={
            "error": "ModelNotFoundError",
            "message": str(exc),
            "detail": "The requested model could not be found. Please check the model name and version."
        }
    )


async def model_load_error_handler(request: Request, exc: ModelLoadError) -> JSONResponse:
    """Handle ModelLoadError exceptions."""
    logger.error(f"Model load error: {str(exc)}")
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content={
            "error": "ModelLoadError",
            "message": str(exc),
            "detail": "Failed to load the model. Please contact the administrator."
        }
    )


async def feature_transformation_error_handler(request: Request, exc: FeatureTransformationError) -> JSONResponse:
    """Handle FeatureTransformationError exceptions."""
    logger.error(f"Feature transformation error: {str(exc)}")
    return JSONResponse(
        status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
        content={
            "error": "FeatureTransformationError",
            "message": str(exc),
            "detail": "Failed to transform input features. Please check your input data."
        }
    )


async def prediction_error_handler(request: Request, exc: PredictionError) -> JSONResponse:
    """Handle PredictionError exceptions."""
    logger.error(f"Prediction error: {str(exc)}")
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content={
            "error": "PredictionError",
            "message": str(exc),
            "detail": "Failed to generate prediction. Please try again or contact support."
        }
    )


async def validation_error_handler(request: Request, exc: ValidationError) -> JSONResponse:
    """Handle ValidationError exceptions."""
    logger.error(f"Validation error: {str(exc)}")
    return JSONResponse(
        status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
        content={
            "error": "ValidationError",
            "message": str(exc),
            "detail": "Input validation failed. Please check your input data."
        }
    )


def create_validation_error_response(message: str, errors: Optional[Dict[str, Any]] = None) -> JSONResponse:
    """
    Create a standardized validation error response.
    
    Args:
        message: Error message
        errors: Optional dictionary with field-specific errors
        
    Returns:
        JSONResponse with validation error details
    """
    content = {
        "error": "ValidationError",
        "message": message
    }
    if errors:
        content["errors"] = errors
    
    return JSONResponse(
        status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
        content=content
    )
