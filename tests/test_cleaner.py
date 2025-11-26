"""
Unit tests for the data cleaner module.
"""

import pytest
import pandas as pd
import numpy as np
from src.data.cleaner import DataCleaner


@pytest.fixture
def sample_data():
    """Create sample data for testing."""
    return pd.DataFrame({
        'brand': ['Toyota', 'Honda', 'Toyota', 'Ford', 'Toyota'],
        'model': ['Corolla', 'Civic', 'Corolla', 'Fiesta', 'Corolla'],
        'year': [2015, 2018, 2015, 2020, 2015],
        'price': [50000.0, 70000.0, 50000.0, 60000.0, 50000.0],
        'km': [50000.0, 30000.0, 50000.0, 20000.0, 50000.0],
        'state': ['SP', 'RJ', 'SP', 'MG', 'SP'],
        'city': ['São Paulo', 'Rio de Janeiro', 'São Paulo', 'Belo Horizonte', 'São Paulo'],
        'fuel_type': ['Flex', 'Gasolina', 'Flex', 'Flex', 'Flex'],
        'transmission': ['Manual', 'Automático', 'Manual', 'Manual', 'Manual'],
        'engine_size': [1.8, 2.0, 1.8, 1.6, 1.8],
        'color': ['Branco', 'Prata', 'Branco', 'Preto', 'Branco'],
        'doors': [4, 4, 4, 4, 4],
        'condition': ['Bom', 'Ótimo', 'Bom', 'Excelente', 'Bom'],
        'age_years': [8, 5, 8, 3, 8],
        'year_of_reference': [2023, 2023, 2023, 2023, 2023],
        'month_of_reference': ['2023-01', '2023-01', '2023-01', '2023-01', '2023-01']
    })


@pytest.fixture
def cleaner():
    """Create a DataCleaner instance for testing."""
    return DataCleaner(random_seed=42)


def test_remove_duplicates(cleaner, sample_data):
    """Test duplicate removal."""
    # Add explicit duplicates
    df_with_duplicates = pd.concat([sample_data, sample_data.iloc[[0]]], ignore_index=True)
    
    cleaned = cleaner.clean(df_with_duplicates)
    
    assert len(cleaned) == len(sample_data)
    assert cleaned.duplicated().sum() == 0


def test_handle_missing_values(cleaner, sample_data):
    """Test missing value handling."""
    # Add missing values
    df_with_missing = sample_data.copy()
    df_with_missing.loc[0, 'price'] = np.nan
    df_with_missing.loc[1, 'brand'] = np.nan
    
    cleaned = cleaner.clean(df_with_missing)
    
    assert cleaned['price'].isna().sum() == 0
    assert cleaned['brand'].isna().sum() == 0


def test_fix_age_calculation(cleaner, sample_data):
    """Test age calculation fix."""
    # Create inconsistent age
    df_with_bad_age = sample_data.copy()
    df_with_bad_age.loc[0, 'age_years'] = -1
    df_with_bad_age.loc[1, 'age_years'] = 999
    
    cleaned = cleaner.clean(df_with_bad_age)
    
    # Age should be recalculated correctly
    assert cleaned.loc[0, 'age_years'] == cleaned.loc[0, 'year_of_reference'] - cleaned.loc[0, 'year']
    assert cleaned.loc[1, 'age_years'] == cleaned.loc[1, 'year_of_reference'] - cleaned.loc[1, 'year']


def test_handle_outliers_iqr(cleaner, sample_data):
    """Test outlier handling with IQR method."""
    # Add outliers
    df_with_outliers = sample_data.copy()
    df_with_outliers.loc[0, 'price'] = 5000000.0  # Extreme outlier
    df_with_outliers.loc[1, 'km'] = 1000000.0  # Extreme outlier
    
    cleaner.outlier_method = 'iqr'
    cleaned = cleaner.fit(df_with_outliers).clean(df_with_outliers)
    
    # Outliers should be removed
    assert cleaned['price'].max() < 5000000.0
    assert cleaned['km'].max() < 1000000.0


def test_handle_outliers_domain(cleaner, sample_data):
    """Test outlier handling with domain limits."""
    # Add outliers outside domain limits
    df_with_outliers = sample_data.copy()
    df_with_outliers.loc[0, 'price'] = 5000000.0  # Outside domain limit
    df_with_outliers.loc[1, 'km'] = -1000.0  # Negative km
    
    cleaner.outlier_method = 'domain'
    cleaned = cleaner.clean(df_with_outliers)
    
    # Outliers should be removed
    assert cleaned['price'].max() <= cleaner.domain_limits['price'][1]
    assert cleaned['km'].min() >= cleaner.domain_limits['km'][0]


def test_standardize_text_fields(cleaner, sample_data):
    """Test text field standardization."""
    # Add unstandardized text
    df_unstandardized = sample_data.copy()
    df_unstandardized.loc[0, 'brand'] = '  toyota  '
    df_unstandardized.loc[1, 'model'] = '  civic  '
    df_unstandardized.loc[2, 'city'] = 'são  paulo'
    
    cleaned = cleaner.clean(df_unstandardized)
    
    # Text should be standardized
    assert cleaned.loc[0, 'brand'] == 'Toyota'
    assert cleaned.loc[1, 'model'] == 'Civic'
    assert cleaned.loc[2, 'city'] == 'São Paulo'


def test_remove_invalid_rows(cleaner, sample_data):
    """Test removal of invalid rows."""
    # Add invalid rows
    df_with_invalid = sample_data.copy()
    df_with_invalid.loc[0, 'price'] = -1000.0  # Negative price
    df_with_invalid.loc[1, 'km'] = -5000.0  # Negative km
    df_with_invalid.loc[2, 'year'] = 2030  # Future year
    df_with_invalid.loc[3, 'doors'] = 6  # Invalid doors
    
    cleaned = cleaner.clean(df_with_invalid)
    
    # Invalid rows should be removed
    assert cleaned['price'].min() > 0
    assert cleaned['km'].min() >= 0
    assert cleaned['year'].max() <= 2023
    assert cleaned['doors'].isin([2, 3, 4, 5]).all()


def test_correct_data_types(cleaner, sample_data):
    """Test data type corrections."""
    # Mess up data types
    df_wrong_types = sample_data.copy()
    df_wrong_types['year'] = df_wrong_types['year'].astype(str)
    df_wrong_types['price'] = df_wrong_types['price'].astype(str)
    
    cleaned = cleaner.clean(df_wrong_types)
    
    # Types should be corrected
    assert pd.api.types.is_integer_dtype(cleaned['year'])
    assert pd.api.types.is_float_dtype(cleaned['price'])


def test_cleaner_configuration(cleaner):
    """Test cleaner configuration options."""
    # Test with different configurations
    cleaner_no_outliers = DataCleaner(handle_outliers=False)
    assert cleaner_no_outliers.handle_outliers == False
    
    cleaner_no_text = DataCleaner(standardize_text=False)
    assert cleaner_no_text.standardize_text == False


def test_fit_method(cleaner, sample_data):
    """Test fit method for computing statistics."""
    cleaner.fit(sample_data)
    
    assert 'price' in cleaner.outlier_stats
    assert 'mean' in cleaner.outlier_stats['price']
    assert 'std' in cleaner.outlier_stats['price']
    assert 'q25' in cleaner.outlier_stats['price']
    assert 'q75' in cleaner.outlier_stats['price']

