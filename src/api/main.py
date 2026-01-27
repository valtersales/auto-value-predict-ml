"""
FastAPI application for AutoValuePredict ML API.

This module provides REST API endpoints for car price prediction.
"""

import os
import logging
from pathlib import Path
from typing import Optional
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from .schemas import (
    CarInput,
    PredictionResponse,
    BatchPredictionRequest,
    BatchPredictionResponse,
    ModelInfoResponse,
    HealthResponse
)
from .predictor import CarPricePredictor
from .errors import (
    ModelNotFoundError,
    ModelLoadError,
    FeatureTransformationError,
    PredictionError,
    ValidationError,
    model_not_found_handler,
    model_load_error_handler,
    feature_transformation_error_handler,
    prediction_error_handler,
    validation_error_handler
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Global predictor instance
predictor: Optional[CarPricePredictor] = None


def load_model(
    model_name: Optional[str] = None,
    model_version: Optional[str] = None
) -> CarPricePredictor:
    """
    Load model and feature pipeline.
    
    Args:
        model_name: Name of the model to load (default: from env or 'lightgbm')
        model_version: Version of the model to load (default: latest)
        
    Returns:
        CarPricePredictor instance
        
    Raises:
        ModelLoadError: If model loading fails
    """
    try:
        from src.models.persistence import ModelPersistence
        
        # Get model name from environment or use default
        if model_name is None:
            model_name = os.getenv('MODEL_NAME', 'lightgbm')
        
        logger.info(f"Loading model: {model_name} (version: {model_version or 'latest'})")
        
        # Initialize persistence manager
        persistence = ModelPersistence()
        
        # Load model
        model, metadata, feature_pipeline = persistence.load_model(
            model_name=model_name,
            version=model_version,
            load_pipeline=True
        )
        
        # Create predictor
        predictor_instance = CarPricePredictor(model, feature_pipeline, metadata)
        
        logger.info(f"Model loaded successfully: {metadata.model_name} v{metadata.version}")
        return predictor_instance
        
    except FileNotFoundError as e:
        logger.error(f"Model file not found: {str(e)}")
        raise ModelLoadError(f"Model not found: {str(e)}")
    except Exception as e:
        logger.error(f"Failed to load model: {str(e)}")
        raise ModelLoadError(f"Failed to load model: {str(e)}")


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifespan context manager for startup and shutdown."""
    # Startup
    global predictor
    try:
        logger.info("Starting API application...")
        predictor = load_model()
        logger.info("API application started successfully")
    except Exception as e:
        logger.error(f"Failed to start API: {str(e)}")
        logger.warning("API will start without model. Some endpoints may not work.")
        predictor = None
    
    yield
    
    # Shutdown
    logger.info("Shutting down API application...")
    predictor = None


# Create FastAPI app
app = FastAPI(
    title="AutoValuePredict ML API",
    description="Machine Learning API for predicting used car prices in Brazil",
    version="1.0.0",
    lifespan=lifespan
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, specify allowed origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Register exception handlers
app.add_exception_handler(ModelNotFoundError, model_not_found_handler)
app.add_exception_handler(ModelLoadError, model_load_error_handler)
app.add_exception_handler(FeatureTransformationError, feature_transformation_error_handler)
app.add_exception_handler(PredictionError, prediction_error_handler)
app.add_exception_handler(ValidationError, validation_error_handler)


@app.get("/", tags=["Root"])
async def root():
    """Root endpoint."""
    return {
        "message": "AutoValuePredict ML API",
        "version": "1.0.0",
        "docs": "/docs",
        "health": "/health"
    }


@app.get("/health", response_model=HealthResponse, tags=["Health"])
async def health_check():
    """
    Health check endpoint.
    
    Returns the current health status of the API and model.
    """
    global predictor
    
    if predictor is None:
        return HealthResponse(
            status="unhealthy",
            model_loaded=False,
            model_name=None,
            model_version=None
        )
    
    model_info = predictor.get_model_info()
    return HealthResponse(
        status="healthy",
        model_loaded=True,
        model_name=model_info['model_name'],
        model_version=model_info['version']
    )


@app.get("/model/info", response_model=ModelInfoResponse, tags=["Model"])
async def get_model_info():
    """
    Get information about the loaded model.
    
    Returns model metadata including performance metrics and hyperparameters.
    """
    global predictor
    
    if predictor is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Model is not loaded"
        )
    
    model_info = predictor.get_model_info()
    return ModelInfoResponse(**model_info)


@app.post("/predict", response_model=PredictionResponse, tags=["Prediction"])
async def predict(car: CarInput):
    """
    Predict price for a single car.
    
    Args:
        car: Car input data
        
    Returns:
        Prediction response with predicted price
    """
    global predictor
    
    if predictor is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Model is not loaded"
        )
    
    try:
        prediction = predictor.predict_single(car)
        model_info = predictor.get_model_info()
        
        return PredictionResponse(
            prediction=prediction,
            model_name=model_info['model_name'],
            model_version=model_info['version']
        )
    except (FeatureTransformationError, PredictionError) as e:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=str(e)
        )
    except Exception as e:
        logger.error(f"Unexpected error in prediction: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Internal server error: {str(e)}"
        )


@app.post("/predict/batch", response_model=BatchPredictionResponse, tags=["Prediction"])
async def predict_batch(request: BatchPredictionRequest):
    """
    Predict prices for multiple cars (batch prediction).
    
    Args:
        request: Batch prediction request with list of cars
        
    Returns:
        Batch prediction response with list of predicted prices
    """
    global predictor
    
    if predictor is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Model is not loaded"
        )
    
    try:
        predictions = predictor.predict_batch(request.cars)
        model_info = predictor.get_model_info()
        
        return BatchPredictionResponse(
            predictions=predictions,
            model_name=model_info['model_name'],
            model_version=model_info['version'],
            count=len(predictions)
        )
    except (FeatureTransformationError, PredictionError) as e:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=str(e)
        )
    except Exception as e:
        logger.error(f"Unexpected error in batch prediction: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Internal server error: {str(e)}"
        )


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
