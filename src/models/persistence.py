"""
Model persistence and versioning module for AutoValuePredict ML project.

This module provides:
- Model saving with comprehensive metadata
- Model versioning system
- Model registry management
- Model loading and validation
"""

import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, Optional, List, Tuple
import joblib
import pandas as pd
import numpy as np

logger = logging.getLogger(__name__)


class ModelMetadata:
    """Model metadata container."""
    
    def __init__(
        self,
        model_name: str,
        model_type: str,
        version: str,
        training_date: str,
        performance_metrics: Dict[str, float],
        feature_list: List[str],
        hyperparameters: Dict[str, Any],
        training_info: Optional[Dict[str, Any]] = None
    ):
        """
        Initialize model metadata.
        
        Args:
            model_name: Name of the model
            model_type: Type of model (e.g., 'lightgbm', 'xgboost')
            version: Model version (e.g., 'v1.0.0')
            training_date: Date when model was trained (ISO format)
            performance_metrics: Dictionary of performance metrics
            feature_list: List of feature names used by the model
            hyperparameters: Dictionary of model hyperparameters
            training_info: Additional training information
        """
        self.model_name = model_name
        self.model_type = model_type
        self.version = version
        self.training_date = training_date
        self.performance_metrics = performance_metrics
        self.feature_list = feature_list
        self.hyperparameters = hyperparameters
        self.training_info = training_info or {}
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert metadata to dictionary."""
        return {
            'model_name': self.model_name,
            'model_type': self.model_type,
            'version': self.version,
            'training_date': self.training_date,
            'performance_metrics': self.performance_metrics,
            'feature_list': self.feature_list,
            'hyperparameters': self.hyperparameters,
            'training_info': self.training_info
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'ModelMetadata':
        """Create metadata from dictionary."""
        return cls(
            model_name=data['model_name'],
            model_type=data['model_type'],
            version=data['version'],
            training_date=data['training_date'],
            performance_metrics=data['performance_metrics'],
            feature_list=data['feature_list'],
            hyperparameters=data['hyperparameters'],
            training_info=data.get('training_info', {})
        )


class ModelPersistence:
    """Model persistence and versioning manager."""
    
    def __init__(self, models_dir: Optional[Path] = None):
        """
        Initialize model persistence manager.
        
        Args:
            models_dir: Base directory for models (default: project_root/models)
        """
        if models_dir is None:
            # Default to project_root/models
            project_root = Path(__file__).parent.parent.parent
            models_dir = project_root / "models"
        
        self.models_dir = Path(models_dir)
        self.models_dir.mkdir(parents=True, exist_ok=True)
        
        # Registry file
        self.registry_file = self.models_dir / "model_registry.json"
        self._registry = self._load_registry()
    
    def _load_registry(self) -> Dict[str, Any]:
        """Load model registry from file."""
        if self.registry_file.exists():
            try:
                with open(self.registry_file, 'r') as f:
                    return json.load(f)
            except Exception as e:
                logger.warning(f"Could not load registry: {e}. Starting with empty registry.")
                return {'models': [], 'versions': {}}
        return {'models': [], 'versions': {}}
    
    def _save_registry(self):
        """Save model registry to file."""
        try:
            with open(self.registry_file, 'w') as f:
                json.dump(self._registry, f, indent=2)
        except Exception as e:
            logger.error(f"Could not save registry: {e}")
    
    def _get_next_version(self, model_name: str) -> str:
        """
        Get next version for a model.
        
        Args:
            model_name: Name of the model
            
        Returns:
            Next version string (e.g., 'v1.0.0', 'v1.0.1')
        """
        if model_name not in self._registry['versions']:
            return 'v1.0.0'
        
        versions = self._registry['versions'][model_name]
        if not versions:
            return 'v1.0.0'
        
        # Get latest version
        latest = max(versions, key=lambda v: self._parse_version(v))
        major, minor, patch = self._parse_version(latest)
        
        # Increment patch version
        return f'v{major}.{minor}.{patch + 1}'
    
    def _parse_version(self, version: str) -> Tuple[int, int, int]:
        """
        Parse version string to tuple.
        
        Args:
            version: Version string (e.g., 'v1.0.0')
            
        Returns:
            Tuple of (major, minor, patch)
        """
        version = version.lstrip('v')
        parts = version.split('.')
        return (int(parts[0]), int(parts[1]), int(parts[2]))
    
    def save_model(
        self,
        model: Any,
        model_name: str,
        model_type: str,
        performance_metrics: Dict[str, float],
        feature_list: List[str],
        hyperparameters: Dict[str, Any],
        feature_pipeline: Optional[Any] = None,
        version: Optional[str] = None,
        training_info: Optional[Dict[str, Any]] = None,
        base_dir: Optional[str] = None
    ) -> Tuple[Path, ModelMetadata]:
        """
        Save model with metadata and versioning.
        
        Args:
            model: Trained model object
            model_name: Name of the model
            model_type: Type of model (e.g., 'lightgbm')
            performance_metrics: Dictionary of performance metrics
            feature_list: List of feature names
            hyperparameters: Dictionary of hyperparameters
            feature_pipeline: Optional feature pipeline to save
            version: Model version (auto-generated if not provided)
            training_info: Additional training information
            base_dir: Base directory for saving (default: models_dir/model_name)
            
        Returns:
            Tuple of (model_directory, metadata)
        """
        # Generate version if not provided
        if version is None:
            version = self._get_next_version(model_name)
        
        # Create model directory
        if base_dir is None:
            model_dir = self.models_dir / model_name / version
        else:
            model_dir = Path(base_dir) / model_name / version
        
        model_dir.mkdir(parents=True, exist_ok=True)
        
        # Save model
        model_path = model_dir / f"{model_name}.joblib"
        joblib.dump(model, model_path)
        logger.info(f"Saved model to {model_path}")
        
        # Save feature pipeline if provided
        pipeline_path = None
        if feature_pipeline is not None:
            pipeline_path = model_dir / "feature_pipeline.joblib"
            joblib.dump(feature_pipeline, pipeline_path)
            logger.info(f"Saved feature pipeline to {pipeline_path}")
        
        # Create metadata
        metadata = ModelMetadata(
            model_name=model_name,
            model_type=model_type,
            version=version,
            training_date=datetime.now().isoformat(),
            performance_metrics=performance_metrics,
            feature_list=feature_list,
            hyperparameters=hyperparameters,
            training_info=training_info or {}
        )
        
        # Save metadata
        metadata_path = model_dir / "metadata.json"
        with open(metadata_path, 'w') as f:
            json.dump(metadata.to_dict(), f, indent=2)
        logger.info(f"Saved metadata to {metadata_path}")
        
        # Update registry
        registry_entry = {
            'model_name': model_name,
            'version': version,
            'path': str(model_dir),
            'training_date': metadata.training_date,
            'performance_metrics': performance_metrics,
            'is_production': False  # Can be set manually later
        }
        
        self._registry['models'].append(registry_entry)
        
        if model_name not in self._registry['versions']:
            self._registry['versions'][model_name] = []
        self._registry['versions'][model_name].append(version)
        
        self._save_registry()
        
        return model_dir, metadata
    
    def load_model(
        self,
        model_name: str,
        version: Optional[str] = None,
        load_pipeline: bool = True
    ) -> Tuple[Any, ModelMetadata, Optional[Any]]:
        """
        Load model with metadata.
        
        Args:
            model_name: Name of the model
            version: Model version (loads latest if not provided)
            load_pipeline: Whether to load feature pipeline
            
        Returns:
            Tuple of (model, metadata, feature_pipeline)
        """
        # Get version
        if version is None:
            if model_name not in self._registry['versions']:
                raise ValueError(f"Model '{model_name}' not found in registry")
            versions = self._registry['versions'][model_name]
            if not versions:
                raise ValueError(f"No versions found for model '{model_name}'")
            version = max(versions, key=lambda v: self._parse_version(v))
            logger.info(f"Loading latest version: {version}")
        
        # Find model directory
        model_dir = None
        for entry in self._registry['models']:
            if entry['model_name'] == model_name and entry['version'] == version:
                model_dir = Path(entry['path'])
                break
        
        if model_dir is None:
            # Try direct path
            model_dir = self.models_dir / model_name / version
        
        if not model_dir.exists():
            raise FileNotFoundError(f"Model directory not found: {model_dir}")
        
        # Load model
        model_path = model_dir / f"{model_name}.joblib"
        if not model_path.exists():
            raise FileNotFoundError(f"Model file not found: {model_path}")
        
        model = joblib.load(model_path)
        logger.info(f"Loaded model from {model_path}")
        
        # Load metadata
        metadata_path = model_dir / "metadata.json"
        if not metadata_path.exists():
            raise FileNotFoundError(f"Metadata file not found: {metadata_path}")
        
        with open(metadata_path, 'r') as f:
            metadata_dict = json.load(f)
        metadata = ModelMetadata.from_dict(metadata_dict)
        
        # Load feature pipeline if requested
        pipeline = None
        if load_pipeline:
            pipeline_path = model_dir / "feature_pipeline.joblib"
            if pipeline_path.exists():
                pipeline = joblib.load(pipeline_path)
                logger.info(f"Loaded feature pipeline from {pipeline_path}")
        
        return model, metadata, pipeline
    
    def list_models(self) -> List[Dict[str, Any]]:
        """
        List all models in registry.
        
        Returns:
            List of model entries
        """
        return self._registry['models'].copy()
    
    def get_model_versions(self, model_name: str) -> List[str]:
        """
        Get all versions for a model.
        
        Args:
            model_name: Name of the model
            
        Returns:
            List of version strings
        """
        return self._registry['versions'].get(model_name, []).copy()
    
    def set_production_model(self, model_name: str, version: str):
        """
        Mark a model version as production.
        
        Args:
            model_name: Name of the model
            version: Model version
        """
        for entry in self._registry['models']:
            if entry['model_name'] == model_name and entry['version'] == version:
                # Unset other production models
                for other_entry in self._registry['models']:
                    if other_entry['model_name'] == model_name:
                        other_entry['is_production'] = False
                
                # Set this one as production
                entry['is_production'] = True
                self._save_registry()
                logger.info(f"Set {model_name} {version} as production model")
                return
        
        raise ValueError(f"Model {model_name} version {version} not found")
    
    def get_production_model(self, model_name: str) -> Optional[str]:
        """
        Get production version for a model.
        
        Args:
            model_name: Name of the model
            
        Returns:
            Version string or None
        """
        for entry in self._registry['models']:
            if entry['model_name'] == model_name and entry.get('is_production', False):
                return entry['version']
        return None


def validate_model(
    model: Any,
    metadata: ModelMetadata,
    X_test: Optional[pd.DataFrame] = None,
    y_test: Optional[pd.Series] = None
) -> Dict[str, Any]:
    """
    Validate a loaded model.
    
    Args:
        model: Loaded model
        metadata: Model metadata
        X_test: Optional test features for validation
        y_test: Optional test target for validation
        
    Returns:
        Dictionary with validation results
    """
    validation_results = {
        'model_name': metadata.model_name,
        'version': metadata.version,
        'model_type': metadata.model_type,
        'has_predict_method': hasattr(model, 'predict'),
        'feature_count': len(metadata.feature_list),
        'validation_passed': True,
        'errors': []
    }
    
    # Check if model has predict method
    if not validation_results['has_predict_method']:
        validation_results['validation_passed'] = False
        validation_results['errors'].append("Model does not have 'predict' method")
    
    # Test prediction if test data provided
    if X_test is not None:
        try:
            # Check feature compatibility
            if len(X_test.columns) != len(metadata.feature_list):
                validation_results['validation_passed'] = False
                validation_results['errors'].append(
                    f"Feature count mismatch: expected {len(metadata.feature_list)}, "
                    f"got {len(X_test.columns)}"
                )
            
            # Try prediction
            predictions = model.predict(X_test)
            validation_results['prediction_test_passed'] = True
            validation_results['prediction_shape'] = predictions.shape
            
            # If y_test provided, calculate metrics
            if y_test is not None:
                from src.models import ModelEvaluator
                evaluator = ModelEvaluator()
                metrics = evaluator.evaluate(y_test, predictions, store_residuals=False)
                validation_results['test_metrics'] = metrics
        except Exception as e:
            validation_results['validation_passed'] = False
            validation_results['prediction_test_passed'] = False
            validation_results['errors'].append(f"Prediction test failed: {str(e)}")
    
    return validation_results
