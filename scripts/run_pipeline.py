#!/usr/bin/env python3
"""
Main pipeline execution script for AutoValuePredict ML project.

This script orchestrates the entire ML pipeline, executing steps incrementally
as phases are completed.

Usage:
    # Run full pipeline (up to implemented phases)
    python scripts/run_pipeline.py

    # Run specific step
    python scripts/run_pipeline.py --step split_data

    # Start from specific step
    python scripts/run_pipeline.py --start-from clean_data

    # Skip specific steps
    python scripts/run_pipeline.py --skip validate_data_before
"""

import sys
import argparse
import logging
from pathlib import Path

# Add project root to path to enable absolute imports
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(project_root / "src"))

from src.pipeline import MLPipeline
from src.pipeline.steps import (
    LoadDataStep,
    ValidateDataStep,
    CleanDataStep,
    SplitDataStep,
    FeatureEngineeringStep,
    TrainBaselineModelsStep,
    TrainAdvancedModelsStep
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def create_pipeline() -> MLPipeline:
    """
    Create and configure the ML pipeline with all available steps.
    
    Returns:
        Configured MLPipeline instance
    """
    pipeline = MLPipeline(
        name="AutoValuePredict ML Pipeline",
        save_state=True
    )
    
    # Phase 2: Data Preprocessing & Cleaning
    pipeline.add_step(LoadDataStep(combine_datasets=True))
    pipeline.add_step(ValidateDataStep(validate_before_cleaning=True))
    pipeline.add_step(CleanDataStep(
        remove_duplicates=True,
        handle_outliers=True,
        handle_missing=True,
        standardize_text=True,
        outlier_method='combined',
        random_seed=42
    ))
    pipeline.add_step(ValidateDataStep(validate_before_cleaning=False))
    pipeline.add_step(SplitDataStep(
        train_size=0.7,
        val_size=0.15,
        test_size=0.15,
        stratify_by_price=False,
        time_based_split=False,
        save_splits=True,
        prefix="fipe_cleaned",
        random_seed=42
    ))
    
    # Phase 3: Feature Engineering
    pipeline.add_step(FeatureEngineeringStep(
        use_feature_selection=False,  # Can enable later for optimization
        feature_selection_method='correlation',
        target_col='price',
        save_pipeline=True,
        random_seed=42
    ))
    
    # Phase 4: Baseline Models (placeholder - to be implemented)
    pipeline.add_step(TrainBaselineModelsStep())
    
    # Phase 5: Advanced Models (placeholder - to be implemented)
    pipeline.add_step(TrainAdvancedModelsStep())
    
    return pipeline


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description='Run AutoValuePredict ML Pipeline'
    )
    parser.add_argument(
        '--step',
        type=str,
        help='Execute only a specific step'
    )
    parser.add_argument(
        '--start-from',
        type=str,
        help='Start execution from a specific step'
    )
    parser.add_argument(
        '--stop-at',
        type=str,
        help='Stop execution at a specific step'
    )
    parser.add_argument(
        '--skip',
        nargs='+',
        help='Skip specific steps'
    )
    parser.add_argument(
        '--status',
        action='store_true',
        help='Show pipeline status and exit'
    )
    parser.add_argument(
        '--list-steps',
        action='store_true',
        help='List all available steps and exit'
    )
    
    args = parser.parse_args()
    
    # Create pipeline
    pipeline = create_pipeline()
    
    # List steps if requested
    if args.list_steps:
        print("\nAvailable Pipeline Steps:")
        print("=" * 60)
        for i, step in enumerate(pipeline.steps, 1):
            status = "✅ Implemented" if step.enabled else "⏳ Placeholder"
            print(f"{i}. {step.name:30s} {status}")
        print("=" * 60)
        return
    
    # Show status if requested
    if args.status:
        status = pipeline.get_status()
        print("\nPipeline Status:")
        print("=" * 60)
        print(f"Name: {status['name']}")
        print(f"Total Steps: {status['total_steps']}")
        print(f"Executed Steps: {status['executed_steps']}")
        print("\nStep Details:")
        for step_info in status['steps']:
            executed = "✅" if step_info['executed'] else "⏳"
            enabled = "enabled" if step_info['enabled'] else "disabled"
            print(f"  {executed} {step_info['name']:30s} ({enabled})")
        print("=" * 60)
        return
    
    # Execute pipeline
    try:
        context = pipeline.execute(
            start_from=args.start_from,
            stop_at=args.stop_at,
            skip_steps=args.skip
        )
        
        logger.info("\n" + "=" * 60)
        logger.info("Pipeline execution summary:")
        logger.info("=" * 60)
        logger.info(f"Steps executed: {len(context['metadata']['steps_executed'])}")
        logger.info(f"Steps: {', '.join(context['metadata']['steps_executed'])}")
        
        if 'train_data' in context:
            logger.info(f"\nData splits created:")
            logger.info(f"  Train: {len(context['train_data']):,} rows")
            logger.info(f"  Validation: {len(context['val_data']):,} rows")
            logger.info(f"  Test: {len(context['test_data']):,} rows")
        
        logger.info("\n✅ Pipeline completed successfully!")
        
    except Exception as e:
        logger.error(f"\n❌ Pipeline failed with error: {e}")
        raise


if __name__ == "__main__":
    main()

