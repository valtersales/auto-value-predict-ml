"""
Pydantic schemas for AutoValuePredict ML API.

This module defines request and response models for API endpoints.
"""

from typing import Optional, List
from pydantic import BaseModel, Field, validator
from datetime import datetime


class CarInput(BaseModel):
    """Input schema for single car prediction."""
    
    brand: str = Field(..., description="Car brand (e.g., 'Fiat', 'Volkswagen')")
    model: str = Field(..., description="Car model (e.g., 'Uno', 'Gol')")
    year: int = Field(..., ge=1985, le=2023, description="Manufacturing year")
    km: float = Field(..., ge=0, le=500000, description="Mileage in kilometers")
    state: str = Field(..., description="Brazilian state code (e.g., 'SP', 'RJ')")
    city: str = Field(..., description="City name")
    fuel_type: str = Field(..., description="Fuel type: 'Flex', 'Gasolina', 'Diesel', 'Elétrico', 'Híbrido', 'GNV'")
    transmission: str = Field(..., description="Transmission type: 'Manual', 'Automático', 'Automatizado', 'CVT'")
    engine_size: float = Field(..., ge=0.7, le=7.0, description="Engine size in liters")
    color: str = Field(..., description="Car color")
    doors: int = Field(..., ge=2, le=5, description="Number of doors")
    condition: str = Field(..., description="Condition: 'Regular', 'Bom', 'Ótimo', 'Excelente'")
    age_years: int = Field(..., ge=0, le=40, description="Vehicle age in years")
    year_of_reference: Optional[int] = Field(None, ge=2020, le=2024, description="Reference year for pricing")
    month_of_reference: Optional[str] = Field(None, description="Reference month (e.g., '2024-01')")
    
    @validator('fuel_type')
    def validate_fuel_type(cls, v):
        valid_types = {'Flex', 'Gasolina', 'Diesel', 'Elétrico', 'Híbrido', 'GNV'}
        if v not in valid_types:
            raise ValueError(f"fuel_type must be one of {valid_types}")
        return v
    
    @validator('transmission')
    def validate_transmission(cls, v):
        valid_types = {'Manual', 'Automático', 'Automatizado', 'CVT'}
        if v not in valid_types:
            raise ValueError(f"transmission must be one of {valid_types}")
        return v
    
    @validator('condition')
    def validate_condition(cls, v):
        valid_conditions = {'Regular', 'Bom', 'Ótimo', 'Excelente'}
        if v not in valid_conditions:
            raise ValueError(f"condition must be one of {valid_conditions}")
        return v
    
    @validator('state')
    def validate_state(cls, v):
        valid_states = {
            'AC', 'AL', 'AP', 'AM', 'BA', 'CE', 'DF', 'ES', 'GO',
            'MA', 'MT', 'MS', 'MG', 'PA', 'PB', 'PR', 'PE', 'PI',
            'RJ', 'RN', 'RS', 'RO', 'RR', 'SC', 'SP', 'SE', 'TO'
        }
        if v.upper() not in valid_states:
            raise ValueError(f"state must be a valid Brazilian state code (e.g., 'SP', 'RJ')")
        return v.upper()
    
    @validator('doors')
    def validate_doors(cls, v):
        if v not in {2, 3, 4, 5}:
            raise ValueError("doors must be 2, 3, 4, or 5")
        return v
    
    class Config:
        schema_extra = {
            "example": {
                "brand": "Fiat",
                "model": "Uno",
                "year": 2020,
                "km": 50000.0,
                "state": "SP",
                "city": "São Paulo",
                "fuel_type": "Flex",
                "transmission": "Manual",
                "engine_size": 1.0,
                "color": "Branco",
                "doors": 4,
                "condition": "Bom",
                "age_years": 4,
                "year_of_reference": 2024,
                "month_of_reference": "2024-01"
            }
        }


class PredictionResponse(BaseModel):
    """Response schema for single prediction."""
    
    prediction: float = Field(..., description="Predicted price in Brazilian Reais")
    model_name: str = Field(..., description="Model name used for prediction")
    model_version: str = Field(..., description="Model version used for prediction")
    timestamp: datetime = Field(default_factory=datetime.now, description="Prediction timestamp")
    
    class Config:
        schema_extra = {
            "example": {
                "prediction": 45000.50,
                "model_name": "lightgbm",
                "model_version": "v1.0.0",
                "timestamp": "2024-01-15T10:30:00"
            }
        }


class BatchPredictionRequest(BaseModel):
    """Request schema for batch predictions."""
    
    cars: List[CarInput] = Field(..., min_items=1, max_items=100, description="List of cars to predict")
    
    class Config:
        schema_extra = {
            "example": {
                "cars": [
                    {
                        "brand": "Fiat",
                        "model": "Uno",
                        "year": 2020,
                        "km": 50000.0,
                        "state": "SP",
                        "city": "São Paulo",
                        "fuel_type": "Flex",
                        "transmission": "Manual",
                        "engine_size": 1.0,
                        "color": "Branco",
                        "doors": 4,
                        "condition": "Bom",
                        "age_years": 4,
                        "year_of_reference": 2024,
                        "month_of_reference": "2024-01"
                    }
                ]
            }
        }


class BatchPredictionResponse(BaseModel):
    """Response schema for batch predictions."""
    
    predictions: List[float] = Field(..., description="List of predicted prices")
    model_name: str = Field(..., description="Model name used for prediction")
    model_version: str = Field(..., description="Model version used for prediction")
    timestamp: datetime = Field(default_factory=datetime.now, description="Prediction timestamp")
    count: int = Field(..., description="Number of predictions")
    
    class Config:
        schema_extra = {
            "example": {
                "predictions": [45000.50, 35000.00],
                "model_name": "lightgbm",
                "model_version": "v1.0.0",
                "timestamp": "2024-01-15T10:30:00",
                "count": 2
            }
        }


class ModelInfoResponse(BaseModel):
    """Response schema for model information."""
    
    model_name: str = Field(..., description="Model name")
    model_type: str = Field(..., description="Model type (e.g., 'lightgbm', 'xgboost')")
    version: str = Field(..., description="Model version")
    training_date: str = Field(..., description="Training date (ISO format)")
    performance_metrics: dict = Field(..., description="Performance metrics")
    feature_count: int = Field(..., description="Number of features")
    hyperparameters: dict = Field(..., description="Model hyperparameters")
    
    class Config:
        schema_extra = {
            "example": {
                "model_name": "lightgbm",
                "model_type": "lightgbm",
                "version": "v1.0.0",
                "training_date": "2024-01-10T12:00:00",
                "performance_metrics": {
                    "rmse": 5000.0,
                    "mae": 4000.0,
                    "mape": 0.15,
                    "r2": 0.85
                },
                "feature_count": 50,
                "hyperparameters": {
                    "n_estimators": 100,
                    "max_depth": 10
                }
            }
        }


class HealthResponse(BaseModel):
    """Response schema for health check."""
    
    status: str = Field(..., description="Service status")
    timestamp: datetime = Field(default_factory=datetime.now, description="Check timestamp")
    model_loaded: bool = Field(..., description="Whether model is loaded")
    model_name: Optional[str] = Field(None, description="Loaded model name")
    model_version: Optional[str] = Field(None, description="Loaded model version")
    
    class Config:
        schema_extra = {
            "example": {
                "status": "healthy",
                "timestamp": "2024-01-15T10:30:00",
                "model_loaded": True,
                "model_name": "lightgbm",
                "model_version": "v1.0.0"
            }
        }
