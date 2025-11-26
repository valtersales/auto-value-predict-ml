"""
Data splitting module for AutoValuePredict ML project.

This module provides utilities for splitting datasets into train/validation/test sets,
with support for stratified splits and time-based splits.
"""

import pandas as pd
import numpy as np
from typing import Tuple, Optional, Dict
from pathlib import Path
import logging

logger = logging.getLogger(__name__)


class DataSplitter:
    """
    Data splitter for FIPE car datasets.
    
    Provides methods for:
    - Train/validation/test split
    - Stratified split by price ranges
    - Time-based split (by year_of_reference)
    """
    
    def __init__(
        self,
        train_size: float = 0.7,
        val_size: float = 0.15,
        test_size: float = 0.15,
        stratify_by_price: bool = False,
        time_based_split: bool = False,
        random_seed: int = 42
    ):
        """
        Initialize the DataSplitter.
        
        Args:
            train_size: Proportion of data for training (default: 0.7)
            val_size: Proportion of data for validation (default: 0.15)
            test_size: Proportion of data for testing (default: 0.15)
            stratify_by_price: Whether to stratify by price ranges (default: False)
            time_based_split: Whether to use time-based split (default: False)
            random_seed: Random seed for reproducibility (default: 42)
            
        Raises:
            ValueError: If train_size + val_size + test_size != 1.0
        """
        if abs(train_size + val_size + test_size - 1.0) > 1e-6:
            raise ValueError(
                f"train_size ({train_size}) + val_size ({val_size}) + "
                f"test_size ({test_size}) must equal 1.0"
            )
        
        self.train_size = train_size
        self.val_size = val_size
        self.test_size = test_size
        self.stratify_by_price = stratify_by_price
        self.time_based_split = time_based_split
        self.random_seed = random_seed
        
        # Price bins for stratification
        self.price_bins = [
            0, 20000, 40000, 60000, 80000, 100000,
            150000, 200000, 300000, 500000, float('inf')
        ]
    
    def split(
        self,
        df: pd.DataFrame,
        target_col: str = 'price'
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """
        Split the dataset into train, validation, and test sets.
        
        Args:
            df: DataFrame to split
            target_col: Name of the target column (for stratification)
            
        Returns:
            Tuple of (train_df, val_df, test_df)
        """
        if self.time_based_split:
            return self._time_based_split(df)
        elif self.stratify_by_price:
            return self._stratified_split(df, target_col)
        else:
            return self._random_split(df)
    
    def _random_split(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """Random split without stratification."""
        np.random.seed(self.random_seed)
        
        # Shuffle the dataframe
        df_shuffled = df.sample(frac=1, random_state=self.random_seed).reset_index(drop=True)
        
        n = len(df_shuffled)
        n_train = int(n * self.train_size)
        n_val = int(n * self.val_size)
        
        train_df = df_shuffled.iloc[:n_train].reset_index(drop=True)
        val_df = df_shuffled.iloc[n_train:n_train + n_val].reset_index(drop=True)
        test_df = df_shuffled.iloc[n_train + n_val:].reset_index(drop=True)
        
        logger.info(
            f"Random split: Train={len(train_df):,} ({len(train_df)/n*100:.1f}%), "
            f"Val={len(val_df):,} ({len(val_df)/n*100:.1f}%), "
            f"Test={len(test_df):,} ({len(test_df)/n*100:.1f}%)"
        )
        
        return train_df, val_df, test_df
    
    def _stratified_split(
        self,
        df: pd.DataFrame,
        target_col: str = 'price'
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """
        Stratified split by price ranges.
        
        Args:
            df: DataFrame to split
            target_col: Name of the target column
            
        Returns:
            Tuple of (train_df, val_df, test_df)
        """
        if target_col not in df.columns:
            logger.warning(
                f"Target column '{target_col}' not found. Falling back to random split."
            )
            return self._random_split(df)
        
        # Create price bins
        df_copy = df.copy()
        df_copy['price_bin'] = pd.cut(
            df_copy[target_col],
            bins=self.price_bins,
            labels=False,
            include_lowest=True
        )
        
        # Remove any rows with NaN bins (shouldn't happen, but just in case)
        df_copy = df_copy.dropna(subset=['price_bin'])
        
        # Split each bin proportionally
        train_dfs = []
        val_dfs = []
        test_dfs = []
        
        np.random.seed(self.random_seed)
        
        for bin_id in df_copy['price_bin'].unique():
            bin_df = df_copy[df_copy['price_bin'] == bin_id].copy()
            bin_df = bin_df.sample(frac=1, random_state=self.random_seed).reset_index(drop=True)
            
            n_bin = len(bin_df)
            n_train_bin = int(n_bin * self.train_size)
            n_val_bin = int(n_bin * self.val_size)
            
            train_dfs.append(bin_df.iloc[:n_train_bin])
            val_dfs.append(bin_df.iloc[n_train_bin:n_train_bin + n_val_bin])
            test_dfs.append(bin_df.iloc[n_train_bin + n_val_bin:])
        
        train_df = pd.concat(train_dfs, ignore_index=True).drop(columns=['price_bin'])
        val_df = pd.concat(val_dfs, ignore_index=True).drop(columns=['price_bin'])
        test_df = pd.concat(test_dfs, ignore_index=True).drop(columns=['price_bin'])
        
        # Shuffle each split
        train_df = train_df.sample(frac=1, random_state=self.random_seed).reset_index(drop=True)
        val_df = val_df.sample(frac=1, random_state=self.random_seed).reset_index(drop=True)
        test_df = test_df.sample(frac=1, random_state=self.random_seed).reset_index(drop=True)
        
        n = len(df)
        logger.info(
            f"Stratified split (by price): Train={len(train_df):,} ({len(train_df)/n*100:.1f}%), "
            f"Val={len(val_df):,} ({len(val_df)/n*100:.1f}%), "
            f"Test={len(test_df):,} ({len(test_df)/n*100:.1f}%)"
        )
        
        return train_df, val_df, test_df
    
    def _time_based_split(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """
        Time-based split by year_of_reference.
        
        Uses chronological ordering to ensure no data leakage.
        
        Args:
            df: DataFrame to split
            
        Returns:
            Tuple of (train_df, val_df, test_df)
        """
        if 'year_of_reference' not in df.columns:
            logger.warning(
                "Column 'year_of_reference' not found. Falling back to random split."
            )
            return self._random_split(df)
        
        # Sort by year_of_reference
        df_sorted = df.sort_values('year_of_reference').reset_index(drop=True)
        
        n = len(df_sorted)
        n_train = int(n * self.train_size)
        n_val = int(n * self.val_size)
        
        train_df = df_sorted.iloc[:n_train].reset_index(drop=True)
        val_df = df_sorted.iloc[n_train:n_train + n_val].reset_index(drop=True)
        test_df = df_sorted.iloc[n_train + n_val:].reset_index(drop=True)
        
        logger.info(
            f"Time-based split: Train={len(train_df):,} ({len(train_df)/n*100:.1f}%), "
            f"Val={len(val_df):,} ({len(val_df)/n*100:.1f}%), "
            f"Test={len(test_df):,} ({len(test_df)/n*100:.1f}%)"
        )
        
        if len(train_df) > 0:
            logger.info(
                f"  Train years: {train_df['year_of_reference'].min()} - "
                f"{train_df['year_of_reference'].max()}"
            )
        if len(val_df) > 0:
            logger.info(
                f"  Val years: {val_df['year_of_reference'].min()} - "
                f"{val_df['year_of_reference'].max()}"
            )
        if len(test_df) > 0:
            logger.info(
                f"  Test years: {test_df['year_of_reference'].min()} - "
                f"{test_df['year_of_reference'].max()}"
            )
        
        return train_df, val_df, test_df
    
    def save_splits(
        self,
        train_df: pd.DataFrame,
        val_df: pd.DataFrame,
        test_df: pd.DataFrame,
        output_dir: Optional[str] = None,
        prefix: str = "fipe"
    ) -> Dict[str, Path]:
        """
        Save train, validation, and test splits to CSV files.
        
        Args:
            train_df: Training DataFrame
            val_df: Validation DataFrame
            test_df: Test DataFrame
            output_dir: Output directory (default: data/processed)
            prefix: Prefix for output filenames (default: "fipe")
            
        Returns:
            Dictionary mapping split names to file paths
        """
        if output_dir is None:
            project_root = Path(__file__).parent.parent.parent
            output_dir = project_root / "data" / "processed"
        else:
            output_dir = Path(output_dir)
        
        output_dir.mkdir(parents=True, exist_ok=True)
        
        file_paths = {}
        
        # Save train split
        train_path = output_dir / f"{prefix}_train.csv"
        train_df.to_csv(train_path, index=False)
        file_paths['train'] = train_path
        logger.info(f"Saved train split to {train_path} ({len(train_df):,} rows)")
        
        # Save validation split
        val_path = output_dir / f"{prefix}_val.csv"
        val_df.to_csv(val_path, index=False)
        file_paths['val'] = val_path
        logger.info(f"Saved validation split to {val_path} ({len(val_df):,} rows)")
        
        # Save test split
        test_path = output_dir / f"{prefix}_test.csv"
        test_df.to_csv(test_path, index=False)
        file_paths['test'] = test_path
        logger.info(f"Saved test split to {test_path} ({len(test_df):,} rows)")
        
        return file_paths

