"""
Pipeline steps implementation for AutoValuePredict ML project.

This module contains concrete implementations of pipeline steps.
Steps are organized by phase and can be added incrementally.
"""

import pandas as pd
import numpy as np
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
    from ..features.pipeline import FeaturePipeline
except (ImportError, ValueError):
    # Fallback to absolute imports (when PYTHONPATH includes src/)
    from src.pipeline.base import PipelineStep
    from src.data.loader import DatasetLoader
    from src.data.cleaner import DataCleaner
    from src.data.validator import DataValidator
    from src.data.splitter import DataSplitter
    from src.features.pipeline import FeaturePipeline

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
    """Step 5: Feature engineering (Phase 3)."""
    
    def __init__(
        self,
        use_feature_selection: bool = False,
        feature_selection_method: str = 'correlation',
        target_col: str = 'price',
        save_pipeline: bool = True,
        random_seed: int = 42
    ):
        """
        Initialize the feature engineering step.
        
        Args:
            use_feature_selection: Whether to apply feature selection
            feature_selection_method: Method for feature selection
            target_col: Name of target column
            save_pipeline: Whether to save the fitted pipeline
            random_seed: Random seed for reproducibility
        """
        super().__init__(name="feature_engineering", enabled=True)
        self.use_feature_selection = use_feature_selection
        self.feature_selection_method = feature_selection_method
        self.target_col = target_col
        self.save_pipeline = save_pipeline
        self.random_seed = random_seed
    
    def get_dependencies(self) -> List[str]:
        return ['split_data']
    
    def validate(self, context: Dict[str, Any]) -> bool:
        if 'train_data' not in context:
            logger.error("No train_data found. Run split_data step first.")
            return False
        if self.target_col not in context['train_data'].columns:
            logger.error(f"Target column '{self.target_col}' not found in train_data.")
            return False
        return True
    
    def execute(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Execute feature engineering on train/val/test splits."""
        train_df = context['train_data'].copy()
        val_df = context['val_data'].copy()
        test_df = context['test_data'].copy()
        
        logger.info("Starting feature engineering...")
        logger.info(f"Train: {len(train_df):,} rows")
        logger.info(f"Validation: {len(val_df):,} rows")
        logger.info(f"Test: {len(test_df):,} rows")
        
        # Initialize feature pipeline
        feature_pipeline = FeaturePipeline(
            use_feature_selection=self.use_feature_selection,
            feature_selection_method=self.feature_selection_method,
            target_col=self.target_col,
            random_seed=self.random_seed
        )
        
        # Prepare target
        y_train = train_df[self.target_col].copy()
        X_train = train_df.drop(columns=[self.target_col])
        
        # Fit pipeline on training data
        logger.info("Fitting feature pipeline on training data...")
        X_train_transformed = feature_pipeline.fit_transform(X_train, y_train)
        
        # Transform validation and test sets
        logger.info("Transforming validation set...")
        y_val = val_df[self.target_col].copy()
        X_val = val_df.drop(columns=[self.target_col])
        X_val_transformed = feature_pipeline.transform(X_val, y_val)
        
        logger.info("Transforming test set...")
        y_test = test_df[self.target_col].copy()
        X_test = test_df.drop(columns=[self.target_col])
        X_test_transformed = feature_pipeline.transform(X_test, y_test)
        
        # Store transformed data
        context['X_train'] = X_train_transformed
        context['y_train'] = y_train
        context['X_val'] = X_val_transformed
        context['y_val'] = y_val
        context['X_test'] = X_test_transformed
        context['y_test'] = y_test
        
        # Store feature pipeline
        context['artifacts']['feature_pipeline'] = feature_pipeline
        
        # Save pipeline if requested
        if self.save_pipeline:
            # Get output_dir from context (set by pipeline) or use default
            output_dir = context.get('output_dir')
            if output_dir is None:
                project_root = Path(__file__).parent.parent.parent
                output_dir = project_root / "data" / "processed"
            
            output_dir = Path(output_dir)
            output_dir.mkdir(parents=True, exist_ok=True)
            
            pipeline_path = output_dir / "feature_pipeline.joblib"
            feature_pipeline.save(str(pipeline_path))
            context['artifacts']['feature_pipeline_path'] = str(pipeline_path)
            logger.info(f"Feature pipeline saved to {pipeline_path}")
        
        # Log feature information
        logger.info(f"Feature engineering completed:")
        logger.info(f"  Original features: {len(X_train.columns)}")
        logger.info(f"  Engineered features: {len(X_train_transformed.columns)}")
        logger.info(f"  Feature names: {list(X_train_transformed.columns)[:10]}...")
        
        # Get feature importance report if available
        if self.use_feature_selection:
            importance_report = feature_pipeline.get_feature_importance_report()
            if importance_report is not None:
                logger.info("\nTop 10 features by importance:")
                top_features = importance_report.head(10)
                for _, row in top_features.iterrows():
                    logger.info(f"  {row['feature']}: {row['score']:.4f}")
        
        return context


class TrainBaselineModelsStep(PipelineStep):
    """Step 6: Train baseline models (Phase 4)."""
    
    def __init__(
        self,
        random_seed: int = 42,
        save_results: bool = True
    ):
        """
        Initialize the train baseline models step.
        
        Args:
            random_seed: Random seed for reproducibility
            save_results: Whether to save model results
        """
        super().__init__(name="train_baseline_models", enabled=True)
        self.random_seed = random_seed
        self.save_results = save_results
    
    def get_dependencies(self) -> List[str]:
        return ['feature_engineering']
    
    def validate(self, context: Dict[str, Any]) -> bool:
        """Validate that feature engineering has been completed."""
        required_keys = ['X_train', 'y_train', 'X_val', 'y_val']
        for key in required_keys:
            if key not in context:
                logger.error(f"Missing required key '{key}' in context. Run feature_engineering step first.")
                return False
        return True
    
    def execute(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Train and evaluate baseline models."""
        from src.models import BaselineModelTrainer, ModelEvaluator, compare_models
        
        X_train = context['X_train']
        y_train = context['y_train']
        X_val = context['X_val']
        y_val = context['y_val']
        
        logger.info("Training baseline models...")
        logger.info(f"Training set: {len(X_train):,} samples, {X_train.shape[1]} features")
        logger.info(f"Validation set: {len(X_val):,} samples")
        
        # Initialize trainer
        trainer = BaselineModelTrainer(random_seed=self.random_seed)
        
        # Train all models
        models = trainer.train_all_baselines(X_train, y_train, X_val, y_val)
        
        # Generate predictions
        logger.info("Generating predictions on validation set...")
        predictions = trainer.predict_all(X_val)
        
        # Evaluate all models
        logger.info("Evaluating models...")
        evaluators = {}
        metrics = {}
        
        for model_name, y_pred in predictions.items():
            evaluator = ModelEvaluator()
            model_metrics = evaluator.evaluate(y_val, y_pred, store_residuals=True)
            evaluators[model_name] = evaluator
            metrics[model_name] = model_metrics
            
            logger.info(f"\n{model_name.upper()} Metrics:")
            logger.info(f"  RMSE: {model_metrics['rmse']:.2f}")
            logger.info(f"  MAE: {model_metrics['mae']:.2f}")
            logger.info(f"  MAPE: {model_metrics['mape']:.2%}")
            logger.info(f"  R²: {model_metrics['r2']:.4f}")
        
        # Create comparison DataFrame
        comparison_df = compare_models(metrics)
        
        # Store results in context
        context['baseline_models'] = models
        context['baseline_metrics'] = metrics
        context['baseline_evaluators'] = evaluators
        context['baseline_comparison'] = comparison_df
        context['baseline_trainer'] = trainer
        
        # Save results if requested
        # Models and results always go to models/ directory, separate from data/processed
        if self.save_results:
            project_root = Path(__file__).parent.parent.parent
            output_dir = project_root / "models" / "baseline_results"
            output_dir.mkdir(parents=True, exist_ok=True)
            
            # Save comparison metrics
            comparison_path = output_dir / "baseline_models_comparison.csv"
            comparison_df.to_csv(comparison_path)
            logger.info(f"Saved comparison metrics to {comparison_path}")
            
            # Save comparison plot
            try:
                plot_path = output_dir / "baseline_models_comparison.png"
                compare_models(metrics, save_path=str(plot_path))
            except Exception as e:
                logger.warning(f"Could not save comparison plot: {e}")
            
            # Save individual model reports
            for model_name, evaluator in evaluators.items():
                try:
                    report = evaluator.get_summary_report()
                    report_path = output_dir / f"{model_name}_report.csv"
                    report.to_csv(report_path, index=False)
                    
                    # Save residual plots
                    residual_plot_path = output_dir / f"{model_name}_residuals.png"
                    evaluator.plot_residuals(save_path=str(residual_plot_path))
                    
                    # Save predictions vs actuals plot
                    pred_plot_path = output_dir / f"{model_name}_predictions_vs_actuals.png"
                    evaluator.plot_predictions_vs_actuals(save_path=str(pred_plot_path))
                except Exception as e:
                    logger.warning(f"Could not save plots for {model_name}: {e}")
            
            context['artifacts']['baseline_results_dir'] = str(output_dir)
        
        logger.info("Baseline models training completed!")
        return context


class TrainAdvancedModelsStep(PipelineStep):
    """Step 7: Train advanced models (Phase 5)."""
    
    def __init__(
        self,
        random_seed: int = 42,
        save_results: bool = True,
        save_models: bool = True,
        include_lightgbm: bool = True
    ):
        super().__init__(name="train_advanced_models", enabled=True)
        self.random_seed = random_seed
        self.save_results = save_results
        self.save_models = save_models
        self.include_lightgbm = include_lightgbm
    
    def get_dependencies(self) -> List[str]:
        # Requires engineered features; baseline step is optional for this phase
        return ['feature_engineering']
    
    def validate(self, context: Dict[str, Any]) -> bool:
        required_keys = ['X_train', 'y_train', 'X_val', 'y_val']
        for key in required_keys:
            if key not in context:
                logger.error(f"Missing required key '{key}' in context. Run feature_engineering step first.")
                return False
        return True
    
    def execute(self, context: Dict[str, Any]) -> Dict[str, Any]:
        from pathlib import Path
        import joblib
        from src.models import AdvancedModelTrainer, ModelEvaluator, compare_models
        
        X_train = context['X_train']
        y_train = context['y_train']
        X_val = context['X_val']
        y_val = context['y_val']
        
        logger.info("Training advanced models (RF, XGBoost, LightGBM)...")
        trainer = AdvancedModelTrainer(
            random_seed=self.random_seed,
            use_lightgbm=self.include_lightgbm
        )
        
        models = trainer.train_all(
            X_train=X_train,
            y_train=y_train,
            X_val=X_val,
            y_val=y_val,
            include_lightgbm=self.include_lightgbm
        )
        
        # Evaluate on validation set
        metrics = {}
        evaluators = {}
        for model_name, model in models.items():
            evaluator = ModelEvaluator()
            preds = model.predict(X_val)
            model_metrics = evaluator.evaluate(y_val, preds, store_residuals=True)
            metrics[model_name] = model_metrics
            evaluators[model_name] = evaluator
            
            logger.info(f"\n{model_name.upper()} Metrics:")
            logger.info(f"  RMSE: {model_metrics['rmse']:.2f}")
            logger.info(f"  MAE: {model_metrics['mae']:.2f}")
            logger.info(f"  MAPE: {model_metrics['mape']:.2%}")
            logger.info(f"  R²: {model_metrics['r2']:.4f}")
        
        comparison_df = compare_models(metrics)
        
        # Persist artifacts
        if self.save_results or self.save_models:
            project_root = Path(__file__).parent.parent.parent
            output_dir = project_root / "models" / "advanced_results"
            output_dir.mkdir(parents=True, exist_ok=True)
            
            if self.save_results:
                comparison_path = output_dir / "advanced_models_comparison.csv"
                comparison_df.to_csv(comparison_path)
                logger.info(f"Saved comparison metrics to {comparison_path}")
                
                # Save individual reports and plots
                for model_name, evaluator in evaluators.items():
                    try:
                        report = evaluator.get_summary_report()
                        report_path = output_dir / f"{model_name}_report.csv"
                        report.to_csv(report_path, index=False)
                        
                        residual_plot_path = output_dir / f"{model_name}_residuals.png"
                        evaluator.plot_residuals(save_path=str(residual_plot_path))
                        
                        pred_plot_path = output_dir / f"{model_name}_predictions_vs_actuals.png"
                        evaluator.plot_predictions_vs_actuals(save_path=str(pred_plot_path))
                    except Exception as e:
                        logger.warning(f"Could not save plots for {model_name}: {e}")
            
            if self.save_models:
                for model_name, model in models.items():
                    model_path = output_dir / f"{model_name}.joblib"
                    try:
                        joblib.dump(model, model_path)
                        logger.info(f"Saved model '{model_name}' to {model_path}")
                    except Exception as e:
                        logger.warning(f"Could not save model {model_name}: {e}")
        
        # Store in context
        context['advanced_models'] = models
        context['advanced_metrics'] = metrics
        context['advanced_evaluators'] = evaluators
        context['advanced_comparison'] = comparison_df
        context['advanced_trainer'] = trainer
        
        logger.info("Advanced models training completed!")
        return context


class EvaluateTestSetStep(PipelineStep):
    """Step 8: Evaluate trained models on test set (Phase 6.3)."""
    
    def __init__(
        self,
        save_results: bool = True,
        model_name: str = 'lightgbm'  # Evaluate best model by default
    ):
        super().__init__(name="evaluate_test_set", enabled=True)
        self.save_results = save_results
        self.model_name = model_name
    
    def get_dependencies(self) -> List[str]:
        return ['train_advanced_models', 'feature_engineering']
    
    def validate(self, context: Dict[str, Any]) -> bool:
        required_keys = ['X_test', 'y_test', 'advanced_models']
        for key in required_keys:
            if key not in context:
                logger.error(f"Missing required key '{key}' in context. Run train_advanced_models step first.")
                return False
        
        # Check if model exists
        if self.model_name not in context['advanced_models']:
            logger.warning(f"Model '{self.model_name}' not found. Available models: {list(context['advanced_models'].keys())}")
            # Use first available model
            self.model_name = list(context['advanced_models'].keys())[0]
            logger.info(f"Using model '{self.model_name}' instead")
        
        return True
    
    def execute(self, context: Dict[str, Any]) -> Dict[str, Any]:
        from src.models import ModelEvaluator, compare_models
        import joblib
        
        X_test = context['X_test']
        y_test = context['y_test']
        models = context['advanced_models']
        val_metrics = context.get('advanced_metrics', {})
        
        logger.info("=" * 80)
        logger.info("Evaluating models on TEST SET")
        logger.info("=" * 80)
        logger.info(f"Test set size: {len(X_test):,} samples")
        
        test_results = {}
        test_metrics = {}
        
        # Evaluate all models or just the specified one
        models_to_evaluate = {self.model_name: models[self.model_name]} if self.model_name in models else models
        
        for model_name, model in models_to_evaluate.items():
            logger.info(f"\nEvaluating {model_name.upper()}...")
            
            # Generate predictions
            y_pred = model.predict(X_test)
            
            # Evaluate
            evaluator = ModelEvaluator()
            metrics = evaluator.evaluate(y_test, y_pred, store_residuals=True)
            
            test_results[model_name] = {
                'metrics': metrics,
                'evaluator': evaluator,
                'predictions': y_pred
            }
            test_metrics[model_name] = metrics
            
            logger.info(f"  RMSE: {metrics['rmse']:.2f}")
            logger.info(f"  MAE: {metrics['mae']:.2f}")
            logger.info(f"  MAPE: {metrics['mape']:.2%}")
            logger.info(f"  R²: {metrics['r2']:.4f}")
        
        # Create test comparison DataFrame
        test_comparison_df = compare_models(test_metrics)
        
        # Compare validation vs test if validation metrics available
        comparison_data = []
        if val_metrics:
            logger.info("\n" + "=" * 80)
            logger.info("VALIDATION vs TEST COMPARISON")
            logger.info("=" * 80)
            
            for model_name in test_metrics.keys():
                if model_name in val_metrics:
                    val = val_metrics[model_name]
                    test = test_metrics[model_name]
                    
                    logger.info(f"\n{model_name.upper()}:")
                    logger.info(f"  RMSE: Val={val['rmse']:.2f}, Test={test['rmse']:.2f}, "
                               f"Diff={test['rmse'] - val['rmse']:.2f} "
                               f"({((test['rmse'] - val['rmse']) / val['rmse']) * 100:.2f}%)")
                    logger.info(f"  MAPE: Val={val['mape']:.2%}, Test={test['mape']:.2%}, "
                               f"Diff={test['mape'] - val['mape']:.4f} "
                               f"({((test['mape'] - val['mape']) / val['mape']) * 100:.2f}%)")
                    logger.info(f"  R²:   Val={val['r2']:.4f}, Test={test['r2']:.4f}, "
                               f"Diff={test['r2'] - val['r2']:.4f} "
                               f"({((test['r2'] - val['r2']) / val['r2']) * 100:.2f}%)")
                    
                    comparison_data.append({
                        'model': model_name,
                        'metric': 'RMSE',
                        'validation': val['rmse'],
                        'test': test['rmse'],
                        'difference': test['rmse'] - val['rmse'],
                        'pct_change': ((test['rmse'] - val['rmse']) / val['rmse']) * 100
                    })
        
        # Save results if requested
        if self.save_results:
            project_root = Path(__file__).parent.parent.parent
            output_dir = project_root / "models" / "test_results"
            output_dir.mkdir(parents=True, exist_ok=True)
            
            # Save test comparison
            test_comparison_path = output_dir / "test_set_comparison.csv"
            test_comparison_df.to_csv(test_comparison_path)
            logger.info(f"\nSaved test comparison to {test_comparison_path}")
            
            # Save individual test reports and plots
            for model_name, result in test_results.items():
                evaluator = result['evaluator']
                try:
                    report = evaluator.get_summary_report()
                    report_path = output_dir / f"{model_name}_test_report.csv"
                    report.to_csv(report_path, index=False)
                    
                    residual_plot_path = output_dir / f"{model_name}_test_residuals.png"
                    evaluator.plot_residuals(save_path=str(residual_plot_path))
                    
                    pred_plot_path = output_dir / f"{model_name}_test_predictions_vs_actuals.png"
                    evaluator.plot_predictions_vs_actuals(save_path=str(pred_plot_path))
                except Exception as e:
                    logger.warning(f"Could not save plots for {model_name}: {e}")
            
            # Save validation vs test comparison
            if comparison_data:
                import pandas as pd
                comparison_df = pd.DataFrame(comparison_data)
                comparison_path = output_dir / "validation_vs_test_comparison.csv"
                comparison_df.to_csv(comparison_path, index=False)
                logger.info(f"Saved validation vs test comparison to {comparison_path}")
        
        # Store in context
        context['test_metrics'] = test_metrics
        context['test_results'] = test_results
        context['test_comparison'] = test_comparison_df
        
        logger.info("\nTest set evaluation completed!")
        return context


class AnalyzeSegmentsAndErrorsStep(PipelineStep):
    """Step 9: Analyze model performance by segments and errors (Phase 6.3)."""
    
    def __init__(
        self,
        save_results: bool = True,
        model_name: str = 'lightgbm',
        n_worst_predictions: int = 50
    ):
        super().__init__(name="analyze_segments_and_errors", enabled=True)
        self.save_results = save_results
        self.model_name = model_name
        self.n_worst_predictions = n_worst_predictions
    
    def get_dependencies(self) -> List[str]:
        return ['evaluate_test_set', 'feature_engineering']
    
    def validate(self, context: Dict[str, Any]) -> bool:
        required_keys = ['X_test', 'y_test', 'test_data', 'advanced_models']
        for key in required_keys:
            if key not in context:
                logger.error(f"Missing required key '{key}' in context.")
                return False
        
        # Check if model exists
        if self.model_name not in context['advanced_models']:
            logger.warning(f"Model '{self.model_name}' not found. Available models: {list(context['advanced_models'].keys())}")
            self.model_name = list(context['advanced_models'].keys())[0]
            logger.info(f"Using model '{self.model_name}' instead")
        
        return True
    
    def execute(self, context: Dict[str, Any]) -> Dict[str, Any]:
        import numpy as np
        import pandas as pd
        from src.models import ModelEvaluator
        
        X_test = context['X_test']
        y_test = context['y_test']
        test_df = context['test_data']
        model = context['advanced_models'][self.model_name]
        
        logger.info("=" * 80)
        logger.info("SEGMENT AND ERROR ANALYSIS")
        logger.info("=" * 80)
        
        # Generate predictions
        logger.info(f"Generating predictions using {self.model_name}...")
        y_pred = model.predict(X_test)
        
        # Create segments
        test_df_segmented = self._create_segments(test_df.copy())
        
        # Segment analysis
        segment_results = self._analyze_by_segments(
            test_df_segmented, y_test, y_pred
        )
        
        # Error analysis
        error_results = self._analyze_errors(
            test_df_segmented, y_test, y_pred
        )
        
        # Save results if requested
        if self.save_results:
            project_root = Path(__file__).parent.parent.parent
            output_dir = project_root / "models" / "analysis_results"
            output_dir.mkdir(parents=True, exist_ok=True)
            
            # Save segment analysis
            segment_results_path = output_dir / "segment_analysis.csv"
            segment_results.to_csv(segment_results_path, index=False)
            logger.info(f"Saved segment analysis to {segment_results_path}")
            
            # Save error analysis
            worst_path = output_dir / "worst_predictions.csv"
            worst_predictions = error_results.nlargest(self.n_worst_predictions, 'absolute_error')
            worst_predictions.to_csv(worst_path, index=False)
            logger.info(f"Saved worst {self.n_worst_predictions} predictions to {worst_path}")
        
        # Store in context
        context['segment_analysis'] = segment_results
        context['error_analysis'] = error_results
        
        logger.info("\nSegment and error analysis completed!")
        return context
    
    def _create_segments(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create segment columns for analysis."""
        # Price segments
        df['price_segment'] = pd.cut(
            df['price'],
            bins=[0, 30000, 50000, 70000, 100000, 150000, 200000, float('inf')],
            labels=['<30k', '30-50k', '50-70k', '70-100k', '100-150k', '150-200k', '>200k']
        )
        
        # Age segments
        if 'age_years' in df.columns:
            df['age_segment'] = pd.cut(
                df['age_years'],
                bins=[0, 2, 5, 10, 15, float('inf')],
                labels=['0-2', '3-5', '6-10', '11-15', '>15']
            )
        elif 'year' in df.columns:
            current_year = 2024
            df['age_years'] = current_year - df['year']
            df['age_segment'] = pd.cut(
                df['age_years'],
                bins=[0, 2, 5, 10, 15, float('inf')],
                labels=['0-2', '3-5', '6-10', '11-15', '>15']
            )
        
        # Brand segments
        if 'brand' in df.columns:
            top_brands = df['brand'].value_counts().head(10).index.tolist()
            df['brand_segment'] = df['brand'].apply(
                lambda x: x if x in top_brands else 'Other'
            )
        
        # Region segments
        if 'state' in df.columns:
            region_mapping = {
                'AC': 'Norte', 'AP': 'Norte', 'AM': 'Norte', 'PA': 'Norte', 'RO': 'Norte', 'RR': 'Norte', 'TO': 'Norte',
                'AL': 'Nordeste', 'BA': 'Nordeste', 'CE': 'Nordeste', 'MA': 'Nordeste', 'PB': 'Nordeste',
                'PE': 'Nordeste', 'PI': 'Nordeste', 'RN': 'Nordeste', 'SE': 'Nordeste',
                'ES': 'Sudeste', 'MG': 'Sudeste', 'RJ': 'Sudeste', 'SP': 'Sudeste',
                'PR': 'Sul', 'RS': 'Sul', 'SC': 'Sul',
                'DF': 'Centro-Oeste', 'GO': 'Centro-Oeste', 'MS': 'Centro-Oeste', 'MT': 'Centro-Oeste'
            }
            df['region'] = df['state'].map(region_mapping).fillna('Unknown')
        
        return df
    
    def _analyze_by_segments(
        self,
        test_df: pd.DataFrame,
        y_test: pd.Series,
        y_pred: np.ndarray
    ) -> pd.DataFrame:
        """Analyze model performance by different segments."""
        from src.models import ModelEvaluator
        
        segment_results = []
        
        # Price segments
        if 'price_segment' in test_df.columns:
            for segment in test_df['price_segment'].cat.categories:
                mask = test_df['price_segment'] == segment
                if mask.sum() > 0:
                    y_true_seg = y_test[mask]
                    y_pred_seg = y_pred[mask]
                    
                    evaluator = ModelEvaluator()
                    metrics = evaluator.evaluate(y_true_seg, y_pred_seg, store_residuals=False)
                    
                    segment_results.append({
                        'segment_type': 'price',
                        'segment_value': str(segment),
                        'n_samples': mask.sum(),
                        'rmse': metrics['rmse'],
                        'mae': metrics['mae'],
                        'mape': metrics['mape'],
                        'r2': metrics['r2']
                    })
        
        # Age segments
        if 'age_segment' in test_df.columns:
            for segment in test_df['age_segment'].cat.categories:
                mask = test_df['age_segment'] == segment
                if mask.sum() > 0:
                    y_true_seg = y_test[mask]
                    y_pred_seg = y_pred[mask]
                    
                    evaluator = ModelEvaluator()
                    metrics = evaluator.evaluate(y_true_seg, y_pred_seg, store_residuals=False)
                    
                    segment_results.append({
                        'segment_type': 'age',
                        'segment_value': str(segment),
                        'n_samples': mask.sum(),
                        'rmse': metrics['rmse'],
                        'mae': metrics['mae'],
                        'mape': metrics['mape'],
                        'r2': metrics['r2']
                    })
        
        # Brand segments
        if 'brand_segment' in test_df.columns:
            for segment in test_df['brand_segment'].unique():
                mask = test_df['brand_segment'] == segment
                if mask.sum() > 0:
                    y_true_seg = y_test[mask]
                    y_pred_seg = y_pred[mask]
                    
                    evaluator = ModelEvaluator()
                    metrics = evaluator.evaluate(y_true_seg, y_pred_seg, store_residuals=False)
                    
                    segment_results.append({
                        'segment_type': 'brand',
                        'segment_value': str(segment),
                        'n_samples': mask.sum(),
                        'rmse': metrics['rmse'],
                        'mae': metrics['mae'],
                        'mape': metrics['mape'],
                        'r2': metrics['r2']
                    })
        
        # Region segments
        if 'region' in test_df.columns:
            for segment in test_df['region'].unique():
                mask = test_df['region'] == segment
                if mask.sum() > 0:
                    y_true_seg = y_test[mask]
                    y_pred_seg = y_pred[mask]
                    
                    evaluator = ModelEvaluator()
                    metrics = evaluator.evaluate(y_true_seg, y_pred_seg, store_residuals=False)
                    
                    segment_results.append({
                        'segment_type': 'region',
                        'segment_value': str(segment),
                        'n_samples': mask.sum(),
                        'rmse': metrics['rmse'],
                        'mae': metrics['mae'],
                        'mape': metrics['mape'],
                        'r2': metrics['r2']
                    })
        
        return pd.DataFrame(segment_results)
    
    def _analyze_errors(
        self,
        test_df: pd.DataFrame,
        y_test: pd.Series,
        y_pred: np.ndarray
    ) -> pd.DataFrame:
        """Analyze prediction errors."""
        errors = y_test.values - y_pred
        absolute_errors = np.abs(errors)
        percentage_errors = (absolute_errors / y_test.values) * 100
        
        error_df = test_df.copy()
        error_df['actual_price'] = y_test.values
        error_df['predicted_price'] = y_pred
        error_df['error'] = errors
        error_df['absolute_error'] = absolute_errors
        error_df['percentage_error'] = percentage_errors
        
        return error_df


class SaveModelWithVersioningStep(PipelineStep):
    """Step 10: Save best model with versioning and metadata (Phase 7)."""
    
    def __init__(
        self,
        model_name: str = 'lightgbm',
        save_pipeline: bool = True
    ):
        super().__init__(name="save_model_with_versioning", enabled=True)
        self.model_name = model_name
        self.save_pipeline = save_pipeline
    
    def get_dependencies(self) -> List[str]:
        return ['train_advanced_models', 'evaluate_test_set', 'feature_engineering']
    
    def validate(self, context: Dict[str, Any]) -> bool:
        required_keys = ['advanced_models', 'advanced_metrics']
        for key in required_keys:
            if key not in context:
                logger.error(f"Missing required key '{key}' in context.")
                return False
        
        # Check if model exists
        if self.model_name not in context['advanced_models']:
            logger.warning(f"Model '{self.model_name}' not found. Available models: {list(context['advanced_models'].keys())}")
            self.model_name = list(context['advanced_models'].keys())[0]
            logger.info(f"Using model '{self.model_name}' instead")
        
        return True
    
    def execute(self, context: Dict[str, Any]) -> Dict[str, Any]:
        from src.models import ModelPersistence
        
        model = context['advanced_models'][self.model_name]
        metrics = context['advanced_metrics'][self.model_name]
        
        # Get test metrics if available
        test_metrics = context.get('test_metrics', {}).get(self.model_name, metrics)
        
        # Use test metrics if available, otherwise use validation metrics
        performance_metrics = {
            'rmse': test_metrics.get('rmse', metrics['rmse']),
            'mae': test_metrics.get('mae', metrics['mae']),
            'mape': test_metrics.get('mape', metrics['mape']),
            'r2': test_metrics.get('r2', metrics['r2'])
        }
        
        logger.info("=" * 80)
        logger.info("SAVING MODEL WITH VERSIONING")
        logger.info("=" * 80)
        logger.info(f"Model: {self.model_name}")
        logger.info(f"Performance metrics: {performance_metrics}")
        
        # Extract hyperparameters
        hyperparameters = {}
        if hasattr(model, 'get_params'):
            hyperparameters = model.get_params()
        
        # Get feature list from feature pipeline
        feature_list = []
        feature_pipeline = None
        
        if 'artifacts' in context and 'feature_pipeline' in context['artifacts']:
            feature_pipeline = context['artifacts']['feature_pipeline']
            if hasattr(feature_pipeline, 'get_feature_names'):
                feature_list = feature_pipeline.get_feature_names()
        
        # Get training info
        training_info = {
            'train_size': len(context.get('X_train', [])),
            'val_size': len(context.get('X_val', [])),
            'test_size': len(context.get('X_test', [])),
            'n_features': len(feature_list) if feature_list else 0
        }
        
        # Initialize persistence manager
        project_root = Path(__file__).parent.parent.parent
        persistence = ModelPersistence(models_dir=project_root / "models")
        
        # Save model
        model_dir, metadata = persistence.save_model(
            model=model,
            model_name=self.model_name,
            model_type=self.model_name,
            performance_metrics=performance_metrics,
            feature_list=feature_list,
            hyperparameters=hyperparameters,
            feature_pipeline=feature_pipeline if self.save_pipeline else None,
            training_info=training_info
        )
        
        logger.info(f"\nModel saved successfully!")
        logger.info(f"  Directory: {model_dir}")
        logger.info(f"  Version: {metadata.version}")
        logger.info(f"  Training date: {metadata.training_date}")
        
        # Store in context
        context['saved_model_dir'] = model_dir
        context['saved_model_metadata'] = metadata
        context['model_persistence'] = persistence
        
        logger.info("\nModel versioning completed!")
        return context

