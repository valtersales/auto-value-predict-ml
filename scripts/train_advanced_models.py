"""
Script to train and evaluate advanced models (Random Forest, XGBoost, LightGBM).

This script:
1. Loads processed train/validation/test splits
2. Applies feature engineering pipeline
3. Trains advanced models (Random Forest, XGBoost, LightGBM)
4. Evaluates models on validation set
5. Generates comparison reports and visualizations
6. Saves trained models
"""

import sys
from pathlib import Path
import pandas as pd
import numpy as np
import logging
from typing import Dict, Any

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.models import AdvancedModelTrainer, ModelEvaluator, compare_models
from src.features.pipeline import FeaturePipeline

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def load_data_splits(data_dir: Path) -> Dict[str, pd.DataFrame]:
    """
    Load train, validation, and test splits.
    
    Args:
        data_dir: Directory containing split files
        
    Returns:
        Dictionary with 'train', 'val', 'test' DataFrames
    """
    logger.info(f"Loading data splits from {data_dir}")
    
    train_path = data_dir / "fipe_cleaned_train.csv"
    val_path = data_dir / "fipe_cleaned_val.csv"
    test_path = data_dir / "fipe_cleaned_test.csv"
    
    if not all(p.exists() for p in [train_path, val_path, test_path]):
        raise FileNotFoundError(
            f"Data splits not found. Expected files:\n"
            f"  - {train_path}\n"
            f"  - {val_path}\n"
            f"  - {test_path}\n"
            "Please run the preprocessing pipeline first."
        )
    
    train_df = pd.read_csv(train_path)
    val_df = pd.read_csv(val_path)
    test_df = pd.read_csv(test_path)
    
    logger.info(f"Loaded splits: Train={len(train_df):,}, Val={len(val_df):,}, Test={len(test_df):,}")
    
    return {
        'train': train_df,
        'val': val_df,
        'test': test_df
    }


def prepare_features(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
    target_col: str = 'price',
    use_feature_selection: bool = False,
    use_advanced_features: bool = False,
    random_seed: int = 42
) -> Dict[str, Any]:
    """
    Prepare features using the feature engineering pipeline.
    
    Args:
        train_df: Training DataFrame
        val_df: Validation DataFrame
        test_df: Test DataFrame
        target_col: Name of target column
        use_feature_selection: Whether to use feature selection
        use_advanced_features: Whether to use advanced features
        random_seed: Random seed
        
    Returns:
        Dictionary with X_train, y_train, X_val, y_val, X_test, y_test, and feature_pipeline
    """
    logger.info("Preparing features using feature engineering pipeline...")
    
    # Initialize feature pipeline
    feature_pipeline = FeaturePipeline(
        use_feature_selection=use_feature_selection,
        feature_selection_method='correlation',
        use_advanced_features=use_advanced_features,
        target_col=target_col,
        random_seed=random_seed
    )
    
    # Prepare targets
    y_train = train_df[target_col].copy()
    y_val = val_df[target_col].copy()
    y_test = test_df[target_col].copy()
    
    # Prepare features (drop target)
    X_train = train_df.drop(columns=[target_col])
    X_val = val_df.drop(columns=[target_col])
    X_test = test_df.drop(columns=[target_col])
    
    # Fit pipeline on training data
    logger.info("Fitting feature pipeline on training data...")
    X_train_transformed = feature_pipeline.fit_transform(X_train, y_train)
    
    # Transform validation and test sets
    logger.info("Transforming validation set...")
    X_val_transformed = feature_pipeline.transform(X_val, y_val)
    
    logger.info("Transforming test set...")
    X_test_transformed = feature_pipeline.transform(X_test, y_test)
    
    logger.info(f"Feature engineering complete. Features: {X_train_transformed.shape[1]}")
    
    return {
        'X_train': X_train_transformed,
        'y_train': y_train,
        'X_val': X_val_transformed,
        'y_val': y_val,
        'X_test': X_test_transformed,
        'y_test': y_test,
        'feature_pipeline': feature_pipeline
    }


def train_and_evaluate_advanced_models(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_val: pd.DataFrame,
    y_val: pd.Series,
    random_seed: int = 42,
    include_lightgbm: bool = True,
    output_dir: Path = None
) -> Dict[str, Any]:
    """
    Train and evaluate all advanced models.
    
    Args:
        X_train: Training features
        y_train: Training target
        X_val: Validation features
        y_val: Validation target
        random_seed: Random seed
        include_lightgbm: Whether to include LightGBM training
        output_dir: Output directory for saving results
        
    Returns:
        Dictionary with models, metrics, and evaluators
    """
    logger.info("Training advanced models...")
    logger.info(f"Training set: {len(X_train):,} samples, {X_train.shape[1]} features")
    logger.info(f"Validation set: {len(X_val):,} samples")
    
    # Initialize trainer
    trainer = AdvancedModelTrainer(
        random_seed=random_seed,
        use_lightgbm=include_lightgbm
    )
    
    # Train all models
    models = trainer.train_all(
        X_train=X_train,
        y_train=y_train,
        X_val=X_val,
        y_val=y_val,
        include_lightgbm=include_lightgbm
    )
    
    # Evaluate all models
    logger.info("Evaluating models on validation set...")
    evaluators = {}
    metrics = {}
    
    for model_name, model in models.items():
        evaluator = ModelEvaluator()
        preds = model.predict(X_val)
        model_metrics = evaluator.evaluate(y_val, preds, store_residuals=True)
        evaluators[model_name] = evaluator
        metrics[model_name] = model_metrics
        
        logger.info(f"\n{model_name.upper()} Metrics:")
        logger.info(f"  RMSE: {model_metrics['rmse']:.2f}")
        logger.info(f"  MAE: {model_metrics['mae']:.2f}")
        logger.info(f"  MAPE: {model_metrics['mape']:.2%}")
        logger.info(f"  R²: {model_metrics['r2']:.4f}")
    
    # Create comparison DataFrame
    comparison_df = compare_models(metrics)
    
    # Save results if output directory provided
    if output_dir:
        import joblib
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Save comparison metrics
        comparison_path = output_dir / "advanced_models_comparison.csv"
        comparison_df.to_csv(comparison_path)
        logger.info(f"Saved comparison metrics to {comparison_path}")
        
        # Save comparison plot
        try:
            plot_path = output_dir / "advanced_models_comparison.png"
            compare_models(metrics, save_path=str(plot_path))
        except Exception as e:
            logger.warning(f"Could not save comparison plot: {e}")
        
        # Save individual model reports and plots
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
        
        # Save trained models
        logger.info("Saving trained models...")
        for model_name, model in models.items():
            model_path = output_dir / f"{model_name}.joblib"
            try:
                joblib.dump(model, model_path)
                logger.info(f"Saved model '{model_name}' to {model_path}")
            except Exception as e:
                logger.warning(f"Could not save model {model_name}: {e}")
    
    return {
        'models': models,
        'metrics': metrics,
        'evaluators': evaluators,
        'comparison_df': comparison_df,
        'trainer': trainer
    }


def main():
    """Main execution function."""
    # Configuration
    data_dir = project_root / "data" / "processed"
    # Models and results always go to models/ directory, separate from data/processed
    output_dir = project_root / "models" / "advanced_results"
    target_col = 'price'
    random_seed = 42
    use_feature_selection = False  # Can enable for optimization
    use_advanced_features = False  # Can enable for better performance
    include_lightgbm = True  # Include LightGBM training
    
    logger.info("=" * 80)
    logger.info("Advanced Models Training and Evaluation")
    logger.info("=" * 80)
    
    try:
        # Load data splits
        splits = load_data_splits(data_dir)
        
        # Prepare features
        feature_data = prepare_features(
            splits['train'],
            splits['val'],
            splits['test'],
            target_col=target_col,
            use_feature_selection=use_feature_selection,
            use_advanced_features=use_advanced_features,
            random_seed=random_seed
        )
        
        # Train and evaluate advanced models
        results = train_and_evaluate_advanced_models(
            feature_data['X_train'],
            feature_data['y_train'],
            feature_data['X_val'],
            feature_data['y_val'],
            random_seed=random_seed,
            include_lightgbm=include_lightgbm,
            output_dir=output_dir
        )
        
        # Print summary
        logger.info("\n" + "=" * 80)
        logger.info("ADVANCED MODELS SUMMARY")
        logger.info("=" * 80)
        logger.info("\nModel Comparison:")
        print(results['comparison_df'].to_string())
        
        logger.info("\n" + "=" * 80)
        logger.info("Training completed successfully!")
        logger.info(f"Results saved to: {output_dir}")
        logger.info("=" * 80)
        
    except Exception as e:
        logger.error(f"Error during training: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()

