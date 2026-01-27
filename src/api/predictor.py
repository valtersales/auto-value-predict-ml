"""
Prediction module for AutoValuePredict ML API.

This module provides functions for making predictions using loaded models.
"""

import pandas as pd
import numpy as np
from typing import List, Dict, Any, Optional, Tuple
import logging

from .schemas import CarInput
from .errors import FeatureTransformationError, PredictionError

logger = logging.getLogger(__name__)


class CarPricePredictor:
    """Predictor for car prices using loaded model and feature pipeline."""
    
    def __init__(self, model: Any, feature_pipeline: Any, metadata: Any):
        """
        Initialize the predictor.
        
        Args:
            model: Loaded ML model
            feature_pipeline: Loaded feature engineering pipeline
            metadata: Model metadata
        """
        self.model = model
        self.feature_pipeline = feature_pipeline
        self.metadata = metadata
        logger.info(f"Initialized predictor with model: {metadata.model_name} v{metadata.version}")
    
    def _car_input_to_dataframe(self, car: CarInput) -> pd.DataFrame:
        """
        Convert CarInput to DataFrame.
        
        Args:
            car: CarInput object
            
        Returns:
            DataFrame with single row
        """
        data = {
            'brand': [car.brand],
            'model': [car.model],
            'year': [car.year],
            'km': [car.km],
            'state': [car.state],
            'city': [car.city],
            'fuel_type': [car.fuel_type],
            'transmission': [car.transmission],
            'engine_size': [car.engine_size],
            'color': [car.color],
            'doors': [car.doors],
            'condition': [car.condition],
            'age_years': [car.age_years],
            'year_of_reference': [car.year_of_reference or 2024],
            'month_of_reference': [car.month_of_reference or '2024-01']
        }
        
        return pd.DataFrame(data)
    
    def _cars_to_dataframe(self, cars: List[CarInput]) -> pd.DataFrame:
        """
        Convert list of CarInput to DataFrame.
        
        Args:
            cars: List of CarInput objects
            
        Returns:
            DataFrame with multiple rows
        """
        data = {
            'brand': [car.brand for car in cars],
            'model': [car.model for car in cars],
            'year': [car.year for car in cars],
            'km': [car.km for car in cars],
            'state': [car.state for car in cars],
            'city': [car.city for car in cars],
            'fuel_type': [car.fuel_type for car in cars],
            'transmission': [car.transmission for car in cars],
            'engine_size': [car.engine_size for car in cars],
            'color': [car.color for car in cars],
            'doors': [car.doors for car in cars],
            'condition': [car.condition for car in cars],
            'age_years': [car.age_years for car in cars],
            'year_of_reference': [car.year_of_reference or 2024 for car in cars],
            'month_of_reference': [car.month_of_reference or '2024-01' for car in cars]
        }
        
        return pd.DataFrame(data)
    
    def _transform_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Transform features using the feature pipeline.
        
        Args:
            df: Input DataFrame
            
        Returns:
            Transformed DataFrame
            
        Raises:
            FeatureTransformationError: If transformation fails
        """
        try:
            if self.feature_pipeline is None:
                raise FeatureTransformationError("Feature pipeline is not loaded")
            
            # Transform features (no target column needed for inference)
            transformed_df = self.feature_pipeline.transform(df)
            
            # Ensure feature order matches training
            expected_features = self.metadata.feature_list
            if set(transformed_df.columns) != set(expected_features):
                missing = set(expected_features) - set(transformed_df.columns)
                extra = set(transformed_df.columns) - set(expected_features)
                if missing:
                    logger.warning(f"Missing features: {missing}")
                if extra:
                    logger.warning(f"Extra features: {extra}")
            
            # Reorder columns to match expected order
            available_features = [f for f in expected_features if f in transformed_df.columns]
            transformed_df = transformed_df[available_features]
            
            # Add missing features with zeros
            for feature in expected_features:
                if feature not in transformed_df.columns:
                    transformed_df[feature] = 0.0
            
            # Ensure correct order
            transformed_df = transformed_df[expected_features]
            
            return transformed_df
            
        except Exception as e:
            logger.error(f"Feature transformation failed: {str(e)}")
            raise FeatureTransformationError(f"Failed to transform features: {str(e)}")
    
    def predict_single(self, car: CarInput) -> float:
        """
        Predict price for a single car.
        
        Args:
            car: CarInput object
            
        Returns:
            Predicted price
            
        Raises:
            PredictionError: If prediction fails
        """
        try:
            # Convert to DataFrame
            df = self._car_input_to_dataframe(car)
            
            # Transform features
            transformed_df = self._transform_features(df)
            
            # Make prediction
            prediction = self.model.predict(transformed_df)
            
            # Handle single prediction (model might return array)
            if isinstance(prediction, np.ndarray):
                prediction = float(prediction[0])
            else:
                prediction = float(prediction)
            
            logger.info(f"Prediction made: R$ {prediction:,.2f}")
            return prediction
            
        except FeatureTransformationError:
            raise
        except Exception as e:
            logger.error(f"Prediction failed: {str(e)}")
            raise PredictionError(f"Failed to make prediction: {str(e)}")
    
    def predict_batch(self, cars: List[CarInput]) -> List[float]:
        """
        Predict prices for multiple cars.
        
        Args:
            cars: List of CarInput objects
            
        Returns:
            List of predicted prices
            
        Raises:
            PredictionError: If prediction fails
        """
        try:
            if not cars:
                raise PredictionError("Empty list of cars provided")
            
            # Convert to DataFrame
            df = self._cars_to_dataframe(cars)
            
            # Transform features
            transformed_df = self._transform_features(df)
            
            # Make predictions
            predictions = self.model.predict(transformed_df)
            
            # Convert to list of floats
            if isinstance(predictions, np.ndarray):
                predictions = [float(p) for p in predictions]
            else:
                predictions = [float(predictions)]
            
            logger.info(f"Batch prediction made: {len(predictions)} predictions")
            return predictions
            
        except FeatureTransformationError:
            raise
        except Exception as e:
            logger.error(f"Batch prediction failed: {str(e)}")
            raise PredictionError(f"Failed to make batch prediction: {str(e)}")
    
    def get_model_info(self) -> Dict[str, Any]:
        """
        Get information about the loaded model.
        
        Returns:
            Dictionary with model information
        """
        return {
            'model_name': self.metadata.model_name,
            'model_type': self.metadata.model_type,
            'version': self.metadata.version,
            'training_date': self.metadata.training_date,
            'performance_metrics': self.metadata.performance_metrics,
            'feature_count': len(self.metadata.feature_list),
            'hyperparameters': self.metadata.hyperparameters
        }
