"""
Dataset loader module for AutoValuePredict ML project.

This module provides utilities to load enriched FIPE datasets.
"""

import pandas as pd
from pathlib import Path
from typing import Tuple, Optional


class DatasetLoader:
    """
    Loader for enriched FIPE datasets.
    
    This class provides methods to load the enriched datasets used in the
    AutoValuePredict ML project. It handles path resolution and basic
    data loading operations.
    """
    
    def __init__(self, data_dir: Optional[str] = None):
        """
        Initialize the DatasetLoader.
        
        Args:
            data_dir: Path to the data directory. If None, uses default
                     'data/processed' relative to project root.
        """
        if data_dir is None:
            # Assume we're in project root or adjust path accordingly
            project_root = Path(__file__).parent.parent.parent
            self.data_dir = project_root / "data" / "processed"
        else:
            self.data_dir = Path(data_dir)
        
        if not self.data_dir.exists():
            raise FileNotFoundError(
                f"Data directory not found: {self.data_dir}. "
                "Please ensure the data/processed directory exists."
            )
    
    def load_fipe_cars(self) -> pd.DataFrame:
        """
        Load the enriched FIPE cars dataset (historical data).
        
        Returns:
            DataFrame containing the enriched historical FIPE data.
            
        Raises:
            FileNotFoundError: If the file doesn't exist.
        """
        file_path = self.data_dir / "fipe_cars_enriched.csv"
        
        if not file_path.exists():
            raise FileNotFoundError(
                f"File not found: {file_path}. "
                "Please ensure fipe_cars_enriched.csv exists in data/processed/"
            )
        
        print(f"Loading {file_path}...")
        df = pd.read_csv(file_path)
        print(f"Loaded {len(df):,} rows and {len(df.columns)} columns")
        
        return df
    
    def load_fipe_2022(self) -> pd.DataFrame:
        """
        Load the enriched FIPE 2022 dataset.
        
        Returns:
            DataFrame containing the enriched 2022 FIPE data.
            
        Raises:
            FileNotFoundError: If the file doesn't exist.
        """
        file_path = self.data_dir / "fipe_2022_enriched.csv"
        
        if not file_path.exists():
            raise FileNotFoundError(
                f"File not found: {file_path}. "
                "Please ensure fipe_2022_enriched.csv exists in data/processed/"
            )
        
        print(f"Loading {file_path}...")
        df = pd.read_csv(file_path)
        print(f"Loaded {len(df):,} rows and {len(df.columns)} columns")
        
        return df
    
    def load_all(self) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Load both enriched datasets.
        
        Returns:
            Tuple of (fipe_cars_df, fipe_2022_df)
        """
        df_cars = self.load_fipe_cars()
        df_2022 = self.load_fipe_2022()
        
        return df_cars, df_2022

