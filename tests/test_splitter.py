"""
Unit tests for the data splitter module.
"""

import pytest
import pandas as pd
import numpy as np
from pathlib import Path
from src.data.splitter import DataSplitter


@pytest.fixture
def sample_data():
    """Create sample data for testing."""
    np.random.seed(42)
    n_samples = 1000
    
    return pd.DataFrame({
        'brand': np.random.choice(['Toyota', 'Honda', 'Ford'], n_samples),
        'model': np.random.choice(['Corolla', 'Civic', 'Fiesta'], n_samples),
        'year': np.random.randint(2010, 2023, n_samples),
        'price': np.random.uniform(20000, 150000, n_samples),
        'km': np.random.uniform(0, 200000, n_samples),
        'state': np.random.choice(['SP', 'RJ', 'MG'], n_samples),
        'city': np.random.choice(['São Paulo', 'Rio', 'BH'], n_samples),
        'fuel_type': np.random.choice(['Flex', 'Gasolina'], n_samples),
        'transmission': np.random.choice(['Manual', 'Automático'], n_samples),
        'engine_size': np.random.uniform(1.0, 2.5, n_samples),
        'color': np.random.choice(['Branco', 'Prata'], n_samples),
        'doors': np.random.choice([2, 4, 5], n_samples),
        'condition': np.random.choice(['Bom', 'Ótimo'], n_samples),
        'age_years': np.random.randint(0, 15, n_samples),
        'year_of_reference': np.random.choice([2021, 2022, 2023], n_samples),
        'month_of_reference': ['2023-01'] * n_samples
    })


@pytest.fixture
def splitter():
    """Create a DataSplitter instance."""
    return DataSplitter(random_seed=42)


def test_random_split(splitter, sample_data):
    """Test random split."""
    train, val, test = splitter.split(sample_data)
    
    # Check sizes
    total = len(train) + len(val) + len(test)
    assert total == len(sample_data)
    assert abs(len(train) / len(sample_data) - splitter.train_size) < 0.01
    assert abs(len(val) / len(sample_data) - splitter.val_size) < 0.01
    assert abs(len(test) / len(sample_data) - splitter.test_size) < 0.01
    
    # Check no overlap
    train_indices = set(train.index)
    val_indices = set(val.index)
    test_indices = set(test.index)
    
    assert len(train_indices & val_indices) == 0
    assert len(train_indices & test_indices) == 0
    assert len(val_indices & test_indices) == 0


def test_stratified_split(splitter, sample_data):
    """Test stratified split by price."""
    splitter.stratify_by_price = True
    train, val, test = splitter.split(sample_data)
    
    # Check sizes
    total = len(train) + len(val) + len(test)
    assert total == len(sample_data)
    
    # Check price distribution is similar across splits
    train_price_mean = train['price'].mean()
    val_price_mean = val['price'].mean()
    test_price_mean = test['price'].mean()
    
    # Means should be relatively close (within 20%)
    price_means = [train_price_mean, val_price_mean, test_price_mean]
    assert max(price_means) / min(price_means) < 1.5


def test_time_based_split(splitter, sample_data):
    """Test time-based split."""
    splitter.time_based_split = True
    train, val, test = splitter.split(sample_data)
    
    # Check sizes
    total = len(train) + len(val) + len(test)
    assert total == len(sample_data)
    
    # Check chronological ordering
    if len(train) > 0 and len(val) > 0:
        assert train['year_of_reference'].max() <= val['year_of_reference'].min()
    if len(val) > 0 and len(test) > 0:
        assert val['year_of_reference'].max() <= test['year_of_reference'].min()


def test_splitter_initialization():
    """Test splitter initialization."""
    # Valid initialization
    splitter = DataSplitter(train_size=0.8, val_size=0.1, test_size=0.1)
    assert splitter.train_size == 0.8
    assert splitter.val_size == 0.1
    assert splitter.test_size == 0.1
    
    # Invalid initialization (sum != 1.0)
    with pytest.raises(ValueError):
        DataSplitter(train_size=0.8, val_size=0.1, test_size=0.2)


def test_save_splits(splitter, sample_data, tmp_path):
    """Test saving splits to files."""
    train, val, test = splitter.split(sample_data)
    
    file_paths = splitter.save_splits(train, val, test, output_dir=str(tmp_path), prefix="test")
    
    # Check files were created
    assert Path(file_paths['train']).exists()
    assert Path(file_paths['val']).exists()
    assert Path(file_paths['test']).exists()
    
    # Check file contents
    train_loaded = pd.read_csv(file_paths['train'])
    assert len(train_loaded) == len(train)
    
    val_loaded = pd.read_csv(file_paths['val'])
    assert len(val_loaded) == len(val)
    
    test_loaded = pd.read_csv(file_paths['test'])
    assert len(test_loaded) == len(test)


def test_split_reproducibility(splitter, sample_data):
    """Test that splits are reproducible with same seed."""
    train1, val1, test1 = splitter.split(sample_data)
    train2, val2, test2 = splitter.split(sample_data)
    
    # Check that splits are identical
    pd.testing.assert_frame_equal(train1, train2)
    pd.testing.assert_frame_equal(val1, val2)
    pd.testing.assert_frame_equal(test1, test2)


def test_stratified_split_fallback(splitter, sample_data):
    """Test stratified split falls back to random if target column missing."""
    splitter.stratify_by_price = True
    df_no_price = sample_data.drop(columns=['price'])
    
    # Should not raise error, should fall back to random split
    train, val, test = splitter.split(df_no_price, target_col='price')
    
    total = len(train) + len(val) + len(test)
    assert total == len(df_no_price)


def test_time_based_split_fallback(splitter, sample_data):
    """Test time-based split falls back to random if year_of_reference missing."""
    splitter.time_based_split = True
    df_no_year = sample_data.drop(columns=['year_of_reference'])
    
    # Should not raise error, should fall back to random split
    train, val, test = splitter.split(df_no_year)
    
    total = len(train) + len(val) + len(test)
    assert total == len(df_no_year)

