"""
Tests for AutoValuePredict ML API.

This module contains tests for all API endpoints.
"""

import pytest
from fastapi.testclient import TestClient
from unittest.mock import Mock, patch, MagicMock
import pandas as pd
import numpy as np
from datetime import datetime

from src.api.main import app, predictor, load_model
from src.api.schemas import CarInput
from src.api.predictor import CarPricePredictor
from src.api.errors import ModelLoadError, FeatureTransformationError, PredictionError


# Test client
client = TestClient(app)


# Mock data
def create_mock_car_input():
    """Create a mock CarInput for testing."""
    return CarInput(
        brand="Fiat",
        model="Uno",
        year=2020,
        km=50000.0,
        state="SP",
        city="São Paulo",
        fuel_type="Flex",
        transmission="Manual",
        engine_size=1.0,
        color="Branco",
        doors=4,
        condition="Bom",
        age_years=4,
        year_of_reference=2024,
        month_of_reference="2024-01"
    )


def create_mock_metadata():
    """Create mock model metadata."""
    mock_metadata = Mock()
    mock_metadata.model_name = "lightgbm"
    mock_metadata.model_type = "lightgbm"
    mock_metadata.version = "v1.0.0"
    mock_metadata.training_date = "2024-01-10T12:00:00"
    mock_metadata.performance_metrics = {
        "rmse": 5000.0,
        "mae": 4000.0,
        "mape": 0.15,
        "r2": 0.85
    }
    mock_metadata.feature_list = ["feature_1", "feature_2", "feature_3"]
    mock_metadata.hyperparameters = {
        "n_estimators": 100,
        "max_depth": 10
    }
    return mock_metadata


def create_mock_predictor():
    """Create a mock CarPricePredictor."""
    mock_model = Mock()
    mock_model.predict = Mock(return_value=np.array([45000.0]))
    
    mock_pipeline = Mock()
    mock_pipeline.transform = Mock(return_value=pd.DataFrame({
        "feature_1": [1.0],
        "feature_2": [2.0],
        "feature_3": [3.0]
    }))
    
    mock_metadata = create_mock_metadata()
    
    return CarPricePredictor(mock_model, mock_pipeline, mock_metadata)


class TestRootEndpoint:
    """Tests for root endpoint."""
    
    def test_root(self):
        """Test root endpoint."""
        response = client.get("/")
        assert response.status_code == 200
        data = response.json()
        assert "message" in data
        assert "version" in data
        assert "docs" in data


class TestHealthEndpoint:
    """Tests for health endpoint."""
    
    @patch('src.api.main.predictor', None)
    def test_health_no_model(self):
        """Test health check when model is not loaded."""
        response = client.get("/health")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "unhealthy"
        assert data["model_loaded"] is False
    
    @patch('src.api.main.predictor', create_mock_predictor())
    def test_health_with_model(self):
        """Test health check when model is loaded."""
        response = client.get("/health")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "healthy"
        assert data["model_loaded"] is True
        assert data["model_name"] == "lightgbm"
        assert data["model_version"] == "v1.0.0"


class TestModelInfoEndpoint:
    """Tests for model info endpoint."""
    
    @patch('src.api.main.predictor', None)
    def test_model_info_no_model(self):
        """Test model info when model is not loaded."""
        response = client.get("/model/info")
        assert response.status_code == 503
    
    @patch('src.api.main.predictor', create_mock_predictor())
    def test_model_info_with_model(self):
        """Test model info when model is loaded."""
        response = client.get("/model/info")
        assert response.status_code == 200
        data = response.json()
        assert data["model_name"] == "lightgbm"
        assert data["model_type"] == "lightgbm"
        assert data["version"] == "v1.0.0"
        assert "performance_metrics" in data
        assert "hyperparameters" in data


class TestPredictEndpoint:
    """Tests for single prediction endpoint."""
    
    @patch('src.api.main.predictor', None)
    def test_predict_no_model(self):
        """Test prediction when model is not loaded."""
        car = create_mock_car_input()
        response = client.post("/predict", json=car.dict())
        assert response.status_code == 503
    
    @patch('src.api.main.predictor', create_mock_predictor())
    def test_predict_success(self):
        """Test successful prediction."""
        car = create_mock_car_input()
        response = client.post("/predict", json=car.dict())
        assert response.status_code == 200
        data = response.json()
        assert "prediction" in data
        assert data["prediction"] == 45000.0
        assert data["model_name"] == "lightgbm"
        assert data["model_version"] == "v1.0.0"
    
    @patch('src.api.main.predictor', create_mock_predictor())
    def test_predict_invalid_input(self):
        """Test prediction with invalid input."""
        invalid_car = {
            "brand": "Fiat",
            "year": 2020,
            # Missing required fields
        }
        response = client.post("/predict", json=invalid_car)
        assert response.status_code == 422
    
    @patch('src.api.main.predictor', create_mock_predictor())
    def test_predict_invalid_fuel_type(self):
        """Test prediction with invalid fuel type."""
        car_dict = create_mock_car_input().dict()
        car_dict["fuel_type"] = "Invalid"
        response = client.post("/predict", json=car_dict)
        assert response.status_code == 422
    
    @patch('src.api.main.predictor', create_mock_predictor())
    def test_predict_invalid_state(self):
        """Test prediction with invalid state."""
        car_dict = create_mock_car_input().dict()
        car_dict["state"] = "XX"
        response = client.post("/predict", json=car_dict)
        assert response.status_code == 422


class TestBatchPredictEndpoint:
    """Tests for batch prediction endpoint."""
    
    @patch('src.api.main.predictor', None)
    def test_batch_predict_no_model(self):
        """Test batch prediction when model is not loaded."""
        cars = [create_mock_car_input()]
        response = client.post("/predict/batch", json={"cars": [car.dict() for car in cars]})
        assert response.status_code == 503
    
    @patch('src.api.main.predictor', create_mock_predictor())
    def test_batch_predict_success(self):
        """Test successful batch prediction."""
        mock_predictor = create_mock_predictor()
        mock_predictor.model.predict = Mock(return_value=np.array([45000.0, 35000.0]))
        mock_predictor.feature_pipeline.transform = Mock(return_value=pd.DataFrame({
            "feature_1": [1.0, 1.5],
            "feature_2": [2.0, 2.5],
            "feature_3": [3.0, 3.5]
        }))
        
        with patch('src.api.main.predictor', mock_predictor):
            cars = [create_mock_car_input(), create_mock_car_input()]
            response = client.post("/predict/batch", json={"cars": [car.dict() for car in cars]})
            assert response.status_code == 200
            data = response.json()
            assert "predictions" in data
            assert len(data["predictions"]) == 2
            assert data["count"] == 2
    
    @patch('src.api.main.predictor', create_mock_predictor())
    def test_batch_predict_empty_list(self):
        """Test batch prediction with empty list."""
        response = client.post("/predict/batch", json={"cars": []})
        assert response.status_code == 422
    
    @patch('src.api.main.predictor', create_mock_predictor())
    def test_batch_predict_too_many_cars(self):
        """Test batch prediction with too many cars."""
        cars = [create_mock_car_input()] * 101
        response = client.post("/predict/batch", json={"cars": [car.dict() for car in cars]})
        assert response.status_code == 422


class TestCarInputValidation:
    """Tests for CarInput validation."""
    
    def test_valid_car_input(self):
        """Test valid car input."""
        car = create_mock_car_input()
        assert car.brand == "Fiat"
        assert car.year == 2020
    
    def test_invalid_year(self):
        """Test invalid year."""
        with pytest.raises(Exception):
            car_dict = create_mock_car_input().dict()
            car_dict["year"] = 1900
            CarInput(**car_dict)
    
    def test_invalid_km(self):
        """Test invalid km."""
        with pytest.raises(Exception):
            car_dict = create_mock_car_input().dict()
            car_dict["km"] = -1000
            CarInput(**car_dict)
    
    def test_invalid_doors(self):
        """Test invalid doors."""
        with pytest.raises(Exception):
            car_dict = create_mock_car_input().dict()
            car_dict["doors"] = 6
            CarInput(**car_dict)


class TestCarPricePredictor:
    """Tests for CarPricePredictor class."""
    
    def test_predict_single(self):
        """Test single prediction."""
        predictor = create_mock_predictor()
        car = create_mock_car_input()
        
        prediction = predictor.predict_single(car)
        assert prediction == 45000.0
        predictor.model.predict.assert_called_once()
    
    def test_predict_batch(self):
        """Test batch prediction."""
        predictor = create_mock_predictor()
        predictor.model.predict = Mock(return_value=np.array([45000.0, 35000.0]))
        predictor.feature_pipeline.transform = Mock(return_value=pd.DataFrame({
            "feature_1": [1.0, 1.5],
            "feature_2": [2.0, 2.5],
            "feature_3": [3.0, 3.5]
        }))
        
        cars = [create_mock_car_input(), create_mock_car_input()]
        predictions = predictor.predict_batch(cars)
        
        assert len(predictions) == 2
        assert predictions[0] == 45000.0
        assert predictions[1] == 35000.0
    
    def test_get_model_info(self):
        """Test getting model info."""
        predictor = create_mock_predictor()
        info = predictor.get_model_info()
        
        assert info["model_name"] == "lightgbm"
        assert info["version"] == "v1.0.0"
        assert "performance_metrics" in info


class TestErrorHandling:
    """Tests for error handling."""
    
    def test_feature_transformation_error(self):
        """Test feature transformation error."""
        predictor = create_mock_predictor()
        predictor.feature_pipeline = None
        
        car = create_mock_car_input()
        
        with pytest.raises(FeatureTransformationError):
            predictor.predict_single(car)
    
    def test_prediction_error(self):
        """Test prediction error."""
        predictor = create_mock_predictor()
        predictor.model.predict = Mock(side_effect=Exception("Model error"))
        
        car = create_mock_car_input()
        
        with pytest.raises(PredictionError):
            predictor.predict_single(car)
