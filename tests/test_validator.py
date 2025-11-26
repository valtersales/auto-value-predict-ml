"""
Unit tests for the data validator module.
"""

import pytest
import pandas as pd
import numpy as np
from src.data.validator import DataValidator


@pytest.fixture
def valid_data():
    """Create valid sample data."""
    return pd.DataFrame({
        'brand': ['Toyota', 'Honda', 'Ford'],
        'model': ['Corolla', 'Civic', 'Fiesta'],
        'year': [2015, 2018, 2020],
        'price': [50000.0, 70000.0, 60000.0],
        'km': [50000.0, 30000.0, 20000.0],
        'state': ['SP', 'RJ', 'MG'],
        'city': ['São Paulo', 'Rio de Janeiro', 'Belo Horizonte'],
        'fuel_type': ['Flex', 'Gasolina', 'Flex'],
        'transmission': ['Manual', 'Automático', 'Manual'],
        'engine_size': [1.8, 2.0, 1.6],
        'color': ['Branco', 'Prata', 'Preto'],
        'doors': [4, 4, 4],
        'condition': ['Bom', 'Ótimo', 'Excelente'],
        'age_years': [8, 5, 3],
        'year_of_reference': [2023, 2023, 2023],
        'month_of_reference': ['2023-01', '2023-01', '2023-01']
    })


@pytest.fixture
def validator():
    """Create a DataValidator instance."""
    return DataValidator()


def test_validate_schema_valid(validator, valid_data):
    """Test schema validation with valid data."""
    result = validator.validate_schema(valid_data)
    
    assert result['is_valid'] == True
    assert len(result['issues']) == 0


def test_validate_schema_missing_columns(validator, valid_data):
    """Test schema validation with missing columns."""
    df_missing = valid_data.drop(columns=['brand', 'model'])
    
    result = validator.validate_schema(df_missing)
    
    assert result['is_valid'] == False
    assert len(result['issues']) > 0
    assert any(issue['type'] == 'missing_columns' for issue in result['issues'])


def test_validate_schema_type_mismatch(validator, valid_data):
    """Test schema validation with type mismatches."""
    df_wrong_types = valid_data.copy()
    df_wrong_types['year'] = df_wrong_types['year'].astype(str)
    df_wrong_types['price'] = df_wrong_types['price'].astype(str)
    
    result = validator.validate_schema(df_wrong_types)
    
    assert result['is_valid'] == False
    assert any(issue['type'] == 'type_mismatch' for issue in result['issues'])


def test_validate_ranges_valid(validator, valid_data):
    """Test range validation with valid data."""
    result = validator.validate_ranges(valid_data)
    
    assert result['is_valid'] == True
    assert len(result['issues']) == 0


def test_validate_ranges_out_of_range(validator, valid_data):
    """Test range validation with out-of-range values."""
    df_out_of_range = valid_data.copy()
    df_out_of_range.loc[0, 'price'] = 5000000.0  # Too high
    df_out_of_range.loc[1, 'km'] = -1000.0  # Negative
    df_out_of_range.loc[2, 'year'] = 2030  # Future year
    
    result = validator.validate_ranges(df_out_of_range)
    
    assert result['is_valid'] == False
    assert len(result['issues']) > 0


def test_validate_categoricals_valid(validator, valid_data):
    """Test categorical validation with valid data."""
    result = validator.validate_categoricals(valid_data)
    
    assert result['is_valid'] == True
    assert len(result['issues']) == 0


def test_validate_categoricals_invalid(validator, valid_data):
    """Test categorical validation with invalid values."""
    df_invalid_cat = valid_data.copy()
    df_invalid_cat.loc[0, 'fuel_type'] = 'InvalidFuel'
    df_invalid_cat.loc[1, 'transmission'] = 'InvalidTransmission'
    df_invalid_cat.loc[2, 'state'] = 'XX'  # Invalid state code
    
    result = validator.validate_categoricals(df_invalid_cat)
    
    assert result['is_valid'] == False
    assert len(result['issues']) > 0


def test_validate_business_rules_valid(validator, valid_data):
    """Test business rules validation with valid data."""
    result = validator.validate_business_rules(valid_data)
    
    assert result['is_valid'] == True
    assert len(result['violations']) == 0


def test_validate_business_rules_violations(validator, valid_data):
    """Test business rules validation with violations."""
    df_violations = valid_data.copy()
    df_violations.loc[0, 'price'] = -1000.0  # Negative price
    df_violations.loc[1, 'km'] = -5000.0  # Negative km
    df_violations.loc[2, 'year'] = 2030  # Year > year_of_reference
    df_violations.loc[2, 'age_years'] = -1  # Negative age
    
    result = validator.validate_business_rules(df_violations)
    
    assert result['is_valid'] == False
    assert len(result['violations']) > 0


def test_validate_business_rules_age_consistency(validator, valid_data):
    """Test age calculation consistency check."""
    df_inconsistent_age = valid_data.copy()
    df_inconsistent_age.loc[0, 'age_years'] = 999  # Inconsistent with year calculation
    
    result = validator.validate_business_rules(df_inconsistent_age)
    
    # Should detect inconsistency
    assert any(
        v['rule'] == 'Age calculation consistency'
        for v in result['violations']
    )


def test_validate_comprehensive(validator, valid_data):
    """Test comprehensive validation."""
    result = validator.validate(valid_data)
    
    assert 'is_valid' in result
    assert 'schema_validation' in result
    assert 'range_validation' in result
    assert 'categorical_validation' in result
    assert 'business_rules_validation' in result
    assert 'summary' in result


def test_get_validation_report(validator, valid_data):
    """Test validation report generation."""
    result = validator.validate(valid_data)
    report = validator.get_validation_report(result)
    
    assert isinstance(report, str)
    assert 'Data Validation Report' in report
    assert 'Total Rows' in report


def test_validator_constants(validator):
    """Test validator constants are properly set."""
    assert len(validator.expected_schema) > 0
    assert len(validator.expected_ranges) > 0
    assert len(validator.valid_categoricals) > 0
    assert len(validator.valid_states) == 27  # 26 states + DF

