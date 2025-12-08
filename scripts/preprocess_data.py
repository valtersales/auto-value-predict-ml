#!/usr/bin/env python3
"""
Data preprocessing script for AutoValuePredict ML project.

This script demonstrates the Phase 2 preprocessing pipeline using the
modular pipeline system. For the full pipeline, use run_pipeline.py.

Usage:
    python scripts/preprocess_data.py
"""

import sys
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
    SplitDataStep
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def main():
    """
    Main preprocessing pipeline (Phase 2 only).
    
    This demonstrates how to run just the preprocessing steps.
    For the full pipeline, use scripts/run_pipeline.py
    """
    logger.info("=" * 60)
    logger.info("Phase 2: Data Preprocessing Pipeline")
    logger.info("=" * 60)
    
    # Create pipeline with Phase 2 steps only
    pipeline = MLPipeline(
        name="Phase 2: Data Preprocessing",
        save_state=True
    )
    
    # Add Phase 2 steps
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
    
    # Execute pipeline
    context = pipeline.execute()
    
    logger.info("\n" + "=" * 60)
    logger.info("Phase 2 preprocessing completed successfully!")
    logger.info("=" * 60)
    
    if 'split_files' in context['artifacts']:
        logger.info("\nOutput files:")
        for split_name, file_path in context['artifacts']['split_files'].items():
            logger.info(f"  {split_name}: {file_path}")
    
    return context


if __name__ == "__main__":
    main()

