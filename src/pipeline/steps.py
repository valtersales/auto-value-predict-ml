"""
Pipeline steps implementation for AutoValuePredict ML project.

This module contains concrete implementations of pipeline steps.
Steps are organized by phase and can be added incrementally.
"""

import pandas as pd
import logging
from typing import Dict, Any, List
from pathlib import Path

# Use absolute imports - PYTHONPATH should be set by the calling script
try:
    # Try relative imports first (when used as a package)
    from .base import PipelineStep
    from ..data.loader import DatasetLoader
    from ..data.cleaner import DataCleaner
    from ..data.validator import DataValidator
    from ..data.splitter import DataSplitter
except (ImportError, ValueError):
    # Fallback to absolute imports (when PYTHONPATH includes src/)
    from src.pipeline.base import PipelineStep
    from src.data.loader import DatasetLoader
    from src.data.cleaner import DataCleaner
    from src.data.validator import DataValidator
    from src.data.splitter import DataSplitter

logger = logging.getLogger(__name__)


class LoadDataStep(PipelineStep):
    """Step 1: Load raw/enriched datasets."""
    
    def __init__(self, data_dir: str = None, combine_datasets: bool = True):
        """
        Initialize the load data step.
        
        Args:
            data_dir: Directory containing data files
            combine_datasets: Whether to combine multiple datasets
        """
        super().__init__(name="load_data")
        self.data_dir = data_dir
        self.combine_datasets = combine_datasets
    
    def validate(self, context: Dict[str, Any]) -> bool:
        """Validate that data directory exists."""
        if self.data_dir:
            data_path = Path(self.data_dir)
            if not data_path.exists():
                logger.error(f"Data directory not found: {data_path}")
                return False
        return True
    
    def execute(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Load datasets."""
        loader = DatasetLoader(data_dir=self.data_dir)
        
        if self.combine_datasets:
            df_cars, df_2022 = loader.load_all()
            df_combined = pd.concat([df_cars, df_2022], ignore_index=True)
            logger.info(f"Loaded and combined datasets: {len(df_combined):,} rows")
            context['data'] = df_combined
        else:
            df_cars = loader.load_fipe_cars()
            logger.info(f"Loaded fipe_cars dataset: {len(df_cars):,} rows")
            context['data'] = df_cars
        
        context['artifacts']['loader'] = loader
        return context


class ValidateDataStep(PipelineStep):
    """Step 2: Validate data quality."""
    
    def __init__(self, validate_before_cleaning: bool = True):
        """
        Initialize the validate data step.
        
        Args:
            validate_before_cleaning: Whether this is validation before cleaning
        """
        name = "validate_data_before" if validate_before_cleaning else "validate_data_after"
        super().__init__(name=name)
        self.validate_before_cleaning = validate_before_cleaning
    
    def get_dependencies(self) -> List[str]:
        """This step depends on load_data."""
        return ['load_data']
    
    def validate(self, context: Dict[str, Any]) -> bool:
        """Validate that data exists in context."""
        if 'data' not in context:
            logger.error("No data found in context. Run load_data step first.")
            return False
        return True
    
    def execute(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Validate data."""
        validator = DataValidator()
        df = context['data']
        
        validation_results = validator.validate(df)
        
        # Store validation results
        context['artifacts'][f'{self.name}_results'] = validation_results
        context['artifacts']['validator'] = validator
        
        # Log results
        if validation_results['is_valid']:
            logger.info("✅ Data validation passed!")
        else:
            logger.warning("⚠️ Data validation found issues:")
            report = validator.get_validation_report(validation_results)
            logger.info(report)
        
        return context


class CleanDataStep(PipelineStep):
    """Step 3: Clean and preprocess data."""
    
    def __init__(
        self,
        remove_duplicates: bool = True,
        handle_outliers: bool = True,
        handle_missing: bool = True,
        standardize_text: bool = True,
        outlier_method: str = 'combined',
        random_seed: int = 42
    ):
        """
        Initialize the clean data step.
        
        Args:
            remove_duplicates: Whether to remove duplicates
            handle_outliers: Whether to handle outliers
            handle_missing: Whether to handle missing values
            standardize_text: Whether to standardize text fields
            outlier_method: Method for outlier detection
            random_seed: Random seed for reproducibility
        """
        super().__init__(name="clean_data")
        self.remove_duplicates = remove_duplicates
        self.handle_outliers = handle_outliers
        self.handle_missing = handle_missing
        self.standardize_text = standardize_text
        self.outlier_method = outlier_method
        self.random_seed = random_seed
    
    def get_dependencies(self) -> List[str]:
        """This step depends on load_data."""
        return ['load_data']
    
    def validate(self, context: Dict[str, Any]) -> bool:
        """Validate that data exists in context."""
        if 'data' not in context:
            logger.error("No data found in context. Run load_data step first.")
            return False
        return True
    
    def execute(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Clean data."""
        df = context['data']
        original_len = len(df)
        
        cleaner = DataCleaner(
            remove_duplicates=self.remove_duplicates,
            handle_outliers=self.handle_outliers,
            handle_missing=self.handle_missing,
            standardize_text=self.standardize_text,
            outlier_method=self.outlier_method,
            random_seed=self.random_seed
        )
        
        df_cleaned = cleaner.fit(df).clean(df)
        
        logger.info(
            f"Cleaned data: {len(df_cleaned):,} rows "
            f"({len(df_cleaned)/original_len*100:.1f}% of original)"
        )
        
        context['data'] = df_cleaned
        context['artifacts']['cleaner'] = cleaner
        return context


class SplitDataStep(PipelineStep):
    """Step 4: Split data into train/validation/test sets."""
    
    def __init__(
        self,
        train_size: float = 0.7,
        val_size: float = 0.15,
        test_size: float = 0.15,
        stratify_by_price: bool = False,
        time_based_split: bool = False,
        save_splits: bool = True,
        prefix: str = "fipe_cleaned",
        random_seed: int = 42
    ):
        """
        Initialize the split data step.
        
        Args:
            train_size: Proportion for training set
            val_size: Proportion for validation set
            test_size: Proportion for test set
            stratify_by_price: Whether to stratify by price ranges
            time_based_split: Whether to use time-based split
            save_splits: Whether to save splits to disk
            prefix: Prefix for output filenames
            random_seed: Random seed for reproducibility
        """
        super().__init__(name="split_data")
        self.train_size = train_size
        self.val_size = val_size
        self.test_size = test_size
        self.stratify_by_price = stratify_by_price
        self.time_based_split = time_based_split
        self.save_splits = save_splits
        self.prefix = prefix
        self.random_seed = random_seed
    
    def get_dependencies(self) -> list[str]:
        """This step depends on clean_data."""
        return ['clean_data']
    
    def validate(self, context: Dict[str, Any]) -> bool:
        """Validate that cleaned data exists in context."""
        if 'data' not in context:
            logger.error("No data found in context. Run clean_data step first.")
            return False
        return True
    
    def execute(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Split data."""
        df = context['data']
        
        splitter = DataSplitter(
            train_size=self.train_size,
            val_size=self.val_size,
            test_size=self.test_size,
            stratify_by_price=self.stratify_by_price,
            time_based_split=self.time_based_split,
            random_seed=self.random_seed
        )
        
        train_df, val_df, test_df = splitter.split(df)
        
        # Store splits in context
        context['train_data'] = train_df
        context['val_data'] = val_df
        context['test_data'] = test_df
        
        # Save splits if requested
        if self.save_splits:
            output_dir = context.get('output_dir')
            file_paths = splitter.save_splits(
                train_df, val_df, test_df,
                output_dir=output_dir,
                prefix=self.prefix
            )
            context['artifacts']['split_files'] = file_paths
        
        context['artifacts']['splitter'] = splitter
        return context


# Placeholder steps for future phases
# These will be implemented as phases are completed

class FeatureEngineeringStep(PipelineStep):
    """Step 5: Feature engineering (Phase 3 - To be implemented)."""
    
    def __init__(self):
        super().__init__(name="feature_engineering", enabled=False)
    
    def get_dependencies(self) -> List[str]:
        return ['split_data']
    
    def validate(self, context: Dict[str, Any]) -> bool:
        if 'train_data' not in context:
            logger.error("No train_data found. Run split_data step first.")
            return False
        return True
    
    def execute(self, context: Dict[str, Any]) -> Dict[str, Any]:
        logger.warning("FeatureEngineeringStep not yet implemented. Skipping.")
        return context


class TrainBaselineModelsStep(PipelineStep):
    """Step 6: Train baseline models (Phase 4 - To be implemented)."""
    
    def __init__(self):
        super().__init__(name="train_baseline_models", enabled=False)
    
    def get_dependencies(self) -> List[str]:
        return ['feature_engineering']
    
    def validate(self, context: Dict[str, Any]) -> bool:
        return True
    
    def execute(self, context: Dict[str, Any]) -> Dict[str, Any]:
        logger.warning("TrainBaselineModelsStep not yet implemented. Skipping.")
        return context


class TrainAdvancedModelsStep(PipelineStep):
    """Step 7: Train advanced models (Phase 5 - To be implemented)."""
    
    def __init__(self):
        super().__init__(name="train_advanced_models", enabled=False)
    
    def get_dependencies(self) -> List[str]:
        return ['train_baseline_models']
    
    def validate(self, context: Dict[str, Any]) -> bool:
        return True
    
    def execute(self, context: Dict[str, Any]) -> Dict[str, Any]:
        logger.warning("TrainAdvancedModelsStep not yet implemented. Skipping.")
        return context

