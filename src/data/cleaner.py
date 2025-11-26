"""
Data cleaning module for AutoValuePredict ML project.

This module provides utilities for cleaning and preprocessing FIPE datasets,
including duplicate removal, outlier handling, missing value treatment,
data type corrections, and text standardization.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple, Union
from scipy import stats
import logging

logger = logging.getLogger(__name__)


class DataCleaner:
    """
    Data cleaner for FIPE car datasets.
    
    Provides methods for:
    - Removing duplicates
    - Handling outliers (IQR, Z-score, domain-based)
    - Handling missing values
    - Correcting data types
    - Standardizing text fields
    """
    
    def __init__(
        self,
        remove_duplicates: bool = True,
        handle_outliers: bool = True,
        handle_missing: bool = True,
        standardize_text: bool = True,
        outlier_method: str = "iqr",
        z_score_threshold: float = 3.0,
        iqr_factor: float = 1.5,
        random_seed: int = 42
    ):
        """
        Initialize the DataCleaner.
        
        Args:
            remove_duplicates: Whether to remove duplicate rows
            handle_outliers: Whether to handle outliers
            handle_missing: Whether to handle missing values
            standardize_text: Whether to standardize text fields
            outlier_method: Method for outlier detection ('iqr', 'zscore', 'domain', 'combined')
            z_score_threshold: Threshold for Z-score method (default: 3.0)
            iqr_factor: Factor for IQR method (default: 1.5)
            random_seed: Random seed for reproducibility
        """
        self.remove_duplicates = remove_duplicates
        self.handle_outliers = handle_outliers
        self.handle_missing = handle_missing
        self.standardize_text = standardize_text
        self.outlier_method = outlier_method
        self.z_score_threshold = z_score_threshold
        self.iqr_factor = iqr_factor
        self.random_seed = random_seed
        
        # Domain-based limits for outliers
        self.domain_limits = {
            'price': (1000, 2000000),  # R$ 1,000 to R$ 2,000,000
            'km': (0, 500000),  # 0 to 500,000 km
            'year': (1985, 2023),  # Valid year range
            'age_years': (0, 40),  # 0 to 40 years old
            'engine_size': (0.7, 7.0),  # 0.7L to 7.0L
            'doors': (2, 5),  # 2 to 5 doors
        }
        
        # Statistics for outlier detection (will be set during fit)
        self.outlier_stats: Dict[str, Dict] = {}
        
    def fit(self, df: pd.DataFrame) -> 'DataCleaner':
        """
        Fit the cleaner on the dataset to compute statistics for outlier detection.
        
        Args:
            df: Input DataFrame
            
        Returns:
            Self for method chaining
        """
        if self.handle_outliers:
            numeric_cols = df.select_dtypes(include=[np.number]).columns
            for col in numeric_cols:
                if col in ['price', 'km', 'year', 'age_years', 'engine_size']:
                    self.outlier_stats[col] = {
                        'mean': df[col].mean(),
                        'std': df[col].std(),
                        'q25': df[col].quantile(0.25),
                        'q75': df[col].quantile(0.75),
                        'iqr': df[col].quantile(0.75) - df[col].quantile(0.25),
                    }
        
        return self
    
    def clean(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Clean the dataset by applying all configured cleaning operations.
        
        Args:
            df: Input DataFrame to clean
            
        Returns:
            Cleaned DataFrame
        """
        df = df.copy()
        original_len = len(df)
        
        logger.info(f"Starting data cleaning on {original_len:,} rows")
        
        # Fit statistics if needed
        if self.handle_outliers and not self.outlier_stats:
            self.fit(df)
        
        # 1. Remove duplicates
        if self.remove_duplicates:
            df = self._remove_duplicates(df)
        
        # 2. Handle missing values
        if self.handle_missing:
            df = self._handle_missing_values(df)
        
        # 3. Correct data types
        df = self._correct_data_types(df)
        
        # 4. Fix age calculation inconsistencies
        df = self._fix_age_calculation(df)
        
        # 5. Handle outliers
        if self.handle_outliers:
            df = self._handle_outliers(df)
        
        # 6. Standardize text fields
        if self.standardize_text:
            df = self._standardize_text_fields(df)
        
        # 7. Final validation - remove rows that violate business rules
        df = self._remove_invalid_rows(df)
        
        final_len = len(df)
        removed = original_len - final_len
        logger.info(
            f"Data cleaning completed: {removed:,} rows removed "
            f"({removed/original_len*100:.2f}%), "
            f"{final_len:,} rows remaining"
        )
        
        return df
    
    def _remove_duplicates(self, df: pd.DataFrame) -> pd.DataFrame:
        """Remove duplicate rows."""
        before = len(df)
        df = df.drop_duplicates()
        after = len(df)
        removed = before - after
        
        if removed > 0:
            logger.info(f"Removed {removed:,} duplicate rows")
        
        return df
    
    def _handle_missing_values(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Handle missing values in the dataset.
        
        Strategy:
        - For numeric columns: Fill with median (robust to outliers)
        - For categorical columns: Fill with mode
        - For age_years: Recalculate from year and year_of_reference
        """
        # Handle age_years first (recalculate if negative or missing)
        if 'age_years' in df.columns:
            mask = (df['age_years'] < 0) | df['age_years'].isna()
            if mask.any():
                df.loc[mask, 'age_years'] = (
                    df.loc[mask, 'year_of_reference'] - df.loc[mask, 'year']
                )
        
        # Handle other numeric columns
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        for col in numeric_cols:
            if col != 'age_years' and df[col].isna().any():
                median_val = df[col].median()
                df[col].fillna(median_val, inplace=True)
                logger.info(f"Filled {df[col].isna().sum()} missing values in {col} with median: {median_val}")
        
        # Handle categorical columns
        categorical_cols = df.select_dtypes(include=['object']).columns
        for col in categorical_cols:
            if df[col].isna().any():
                mode_val = df[col].mode()[0] if len(df[col].mode()) > 0 else 'Unknown'
                df[col].fillna(mode_val, inplace=True)
                logger.info(f"Filled missing values in {col} with mode: {mode_val}")
        
        return df
    
    def _correct_data_types(self, df: pd.DataFrame) -> pd.DataFrame:
        """Correct data types according to expected schema."""
        # Ensure integer columns are int64
        int_cols = ['year', 'doors', 'age_years', 'year_of_reference']
        for col in int_cols:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce').astype('Int64')
                df[col] = df[col].fillna(0).astype('int64')
        
        # Ensure float columns are float64
        float_cols = ['price', 'km', 'engine_size']
        for col in float_cols:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce').astype('float64')
        
        return df
    
    def _fix_age_calculation(self, df: pd.DataFrame) -> pd.DataFrame:
        """Fix age calculation inconsistencies."""
        if 'age_years' in df.columns and 'year_of_reference' in df.columns and 'year' in df.columns:
            # Recalculate age_years from year and year_of_reference
            calculated_age = df['year_of_reference'] - df['year']
            
            # Update age_years where it's inconsistent or negative
            mask = (df['age_years'] != calculated_age) | (df['age_years'] < 0)
            if mask.any():
                df.loc[mask, 'age_years'] = calculated_age[mask]
                logger.info(f"Fixed {mask.sum():,} age calculation inconsistencies")
        
        return df
    
    def _handle_outliers(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Handle outliers using the specified method.
        
        Methods:
        - 'iqr': Interquartile Range method
        - 'zscore': Z-score method
        - 'domain': Domain knowledge-based limits
        - 'combined': Use all methods (most conservative)
        """
        outlier_mask = pd.Series(False, index=df.index)
        
        numeric_cols = ['price', 'km', 'year', 'age_years', 'engine_size']
        
        for col in numeric_cols:
            if col not in df.columns:
                continue
            
            col_outliers = pd.Series(False, index=df.index)
            
            if self.outlier_method in ['iqr', 'combined']:
                if col in self.outlier_stats:
                    q25 = self.outlier_stats[col]['q25']
                    q75 = self.outlier_stats[col]['q75']
                    iqr = self.outlier_stats[col]['iqr']
                    
                    lower_bound = q25 - self.iqr_factor * iqr
                    upper_bound = q75 + self.iqr_factor * iqr
                    
                    iqr_outliers = (df[col] < lower_bound) | (df[col] > upper_bound)
                    col_outliers = col_outliers | iqr_outliers
            
            if self.outlier_method in ['zscore', 'combined']:
                if col in self.outlier_stats:
                    mean = self.outlier_stats[col]['mean']
                    std = self.outlier_stats[col]['std']
                    
                    if std > 0:
                        z_scores = np.abs((df[col] - mean) / std)
                        zscore_outliers = z_scores > self.z_score_threshold
                        col_outliers = col_outliers | zscore_outliers
            
            if self.outlier_method in ['domain', 'combined']:
                if col in self.domain_limits:
                    lower, upper = self.domain_limits[col]
                    domain_outliers = (df[col] < lower) | (df[col] > upper)
                    col_outliers = col_outliers | domain_outliers
            
            outlier_mask = outlier_mask | col_outliers
        
        # Remove outliers
        if outlier_mask.any():
            n_outliers = outlier_mask.sum()
            logger.info(f"Removed {n_outliers:,} outlier rows ({n_outliers/len(df)*100:.2f}%)")
            df = df[~outlier_mask]
        
        return df
    
    def _standardize_text_fields(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Standardize text fields (brand, model, city names).
        
        Operations:
        - Strip whitespace
        - Title case for brand and model
        - Remove extra spaces
        - Handle common variations
        """
        # Standardize brand names
        if 'brand' in df.columns:
            df['brand'] = df['brand'].astype(str).str.strip()
            df['brand'] = df['brand'].str.title()
            df['brand'] = df['brand'].str.replace(r'\s+', ' ', regex=True)
        
        # Standardize model names
        if 'model' in df.columns:
            df['model'] = df['model'].astype(str).str.strip()
            df['model'] = df['model'].str.title()
            df['model'] = df['model'].str.replace(r'\s+', ' ', regex=True)
        
        # Standardize city names
        if 'city' in df.columns:
            df['city'] = df['city'].astype(str).str.strip()
            df['city'] = df['city'].str.title()
            df['city'] = df['city'].str.replace(r'\s+', ' ', regex=True)
        
        # Standardize state names
        if 'state' in df.columns:
            df['state'] = df['state'].astype(str).str.strip().str.upper()
        
        # Standardize fuel_type
        if 'fuel_type' in df.columns:
            df['fuel_type'] = df['fuel_type'].astype(str).str.strip().str.title()
        
        # Standardize transmission
        if 'transmission' in df.columns:
            df['transmission'] = df['transmission'].astype(str).str.strip().str.title()
        
        # Standardize color
        if 'color' in df.columns:
            df['color'] = df['color'].astype(str).str.strip().str.title()
        
        # Standardize condition
        if 'condition' in df.columns:
            df['condition'] = df['condition'].astype(str).str.strip().str.title()
        
        return df
    
    def _remove_invalid_rows(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Remove rows that violate basic business rules.
        
        Rules:
        - Price must be positive
        - KM must be non-negative
        - Year must be within valid range
        - Age must be non-negative
        - Year must be <= year_of_reference
        """
        before = len(df)
        
        # Price must be positive
        if 'price' in df.columns:
            df = df[df['price'] > 0]
        
        # KM must be non-negative
        if 'km' in df.columns:
            df = df[df['km'] >= 0]
        
        # Year must be within valid range
        if 'year' in df.columns:
            df = df[(df['year'] >= 1985) & (df['year'] <= 2023)]
        
        # Age must be non-negative
        if 'age_years' in df.columns:
            df = df[df['age_years'] >= 0]
        
        # Year must be <= year_of_reference
        if 'year' in df.columns and 'year_of_reference' in df.columns:
            df = df[df['year'] <= df['year_of_reference']]
        
        # Engine size must be positive and reasonable
        if 'engine_size' in df.columns:
            df = df[(df['engine_size'] > 0) & (df['engine_size'] <= 7.0)]
        
        # Doors must be valid (2, 3, 4, or 5)
        if 'doors' in df.columns:
            df = df[df['doors'].isin([2, 3, 4, 5])]
        
        after = len(df)
        removed = before - after
        
        if removed > 0:
            logger.info(f"Removed {removed:,} rows violating business rules")
        
        return df

