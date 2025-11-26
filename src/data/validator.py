"""
Data validation module for AutoValuePredict ML project.

This module provides utilities for validating FIPE datasets, including
schema validation, range checks, categorical value validation, and
business rule validation.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple, Set
import logging

logger = logging.getLogger(__name__)


class DataValidator:
    """
    Data validator for FIPE car datasets.
    
    Provides methods for:
    - Schema validation
    - Range checks
    - Categorical value validation
    - Business rule validation
    """
    
    def __init__(self):
        """Initialize the DataValidator."""
        # Expected schema
        self.expected_schema = {
            'brand': 'object',
            'model': 'object',
            'year': 'int64',
            'price': 'float64',
            'km': 'float64',
            'state': 'object',
            'city': 'object',
            'fuel_type': 'object',
            'transmission': 'object',
            'engine_size': 'float64',
            'color': 'object',
            'doors': 'int64',
            'condition': 'object',
            'age_years': 'int64',
            'year_of_reference': 'int64',
            'month_of_reference': 'object'
        }
        
        # Expected ranges
        self.expected_ranges = {
            'year': (1985, 2023),
            'price': (1000, 2000000),
            'km': (0, 500000),
            'age_years': (0, 40),
            'engine_size': (0.7, 7.0),
            'doors': (2, 5),
        }
        
        # Valid categorical values
        self.valid_categoricals = {
            'fuel_type': {'Flex', 'Gasolina', 'Diesel', 'Elétrico', 'Híbrido', 'GNV'},
            'transmission': {'Manual', 'Automático', 'Automatizado', 'CVT'},
            'condition': {'Regular', 'Bom', 'Ótimo', 'Excelente'},
            'doors': {2, 3, 4, 5},
        }
        
        # Brazilian states (UF codes)
        self.valid_states = {
            'AC', 'AL', 'AP', 'AM', 'BA', 'CE', 'DF', 'ES', 'GO',
            'MA', 'MT', 'MS', 'MG', 'PA', 'PB', 'PR', 'PE', 'PI',
            'RJ', 'RN', 'RS', 'RO', 'RR', 'SC', 'SP', 'SE', 'TO'
        }
    
    def validate(self, df: pd.DataFrame) -> Dict[str, any]:
        """
        Perform comprehensive validation on the dataset.
        
        Args:
            df: DataFrame to validate
            
        Returns:
            Dictionary with validation results
        """
        results = {
            'is_valid': True,
            'schema_validation': self.validate_schema(df),
            'range_validation': self.validate_ranges(df),
            'categorical_validation': self.validate_categoricals(df),
            'business_rules_validation': self.validate_business_rules(df),
            'summary': {}
        }
        
        # Overall validation status
        all_validations = [
            results['schema_validation']['is_valid'],
            results['range_validation']['is_valid'],
            results['categorical_validation']['is_valid'],
            results['business_rules_validation']['is_valid']
        ]
        
        results['is_valid'] = all(all_validations)
        
        # Summary
        results['summary'] = {
            'total_rows': len(df),
            'total_issues': sum([
                len(results['schema_validation'].get('issues', [])),
                len(results['range_validation'].get('issues', [])),
                len(results['categorical_validation'].get('issues', [])),
                len(results['business_rules_validation'].get('violations', []))
            ]),
            'validation_passed': results['is_valid']
        }
        
        return results
    
    def validate_schema(self, df: pd.DataFrame) -> Dict[str, any]:
        """
        Validate that the DataFrame matches the expected schema.
        
        Args:
            df: DataFrame to validate
            
        Returns:
            Dictionary with validation results
        """
        issues = []
        
        # Check for missing columns
        missing_cols = set(self.expected_schema.keys()) - set(df.columns)
        if missing_cols:
            issues.append({
                'type': 'missing_columns',
                'columns': list(missing_cols),
                'severity': 'error'
            })
        
        # Check for unexpected columns
        unexpected_cols = set(df.columns) - set(self.expected_schema.keys())
        if unexpected_cols:
            issues.append({
                'type': 'unexpected_columns',
                'columns': list(unexpected_cols),
                'severity': 'warning'
            })
        
        # Check data types
        type_issues = []
        for col, expected_type in self.expected_schema.items():
            if col in df.columns:
                actual_type = str(df[col].dtype)
                if actual_type != expected_type:
                    type_issues.append({
                        'column': col,
                        'expected': expected_type,
                        'actual': actual_type
                    })
        
        if type_issues:
            issues.append({
                'type': 'type_mismatch',
                'issues': type_issues,
                'severity': 'error'
            })
        
        return {
            'is_valid': len(issues) == 0,
            'issues': issues
        }
    
    def validate_ranges(self, df: pd.DataFrame) -> Dict[str, any]:
        """
        Validate that numerical columns are within expected ranges.
        
        Args:
            df: DataFrame to validate
            
        Returns:
            Dictionary with validation results
        """
        issues = []
        
        for col, (min_val, max_val) in self.expected_ranges.items():
            if col not in df.columns:
                continue
            
            # Check for values outside range
            out_of_range = df[(df[col] < min_val) | (df[col] > max_val)]
            
            if len(out_of_range) > 0:
                issues.append({
                    'column': col,
                    'expected_range': (min_val, max_val),
                    'violations': len(out_of_range),
                    'percentage': (len(out_of_range) / len(df)) * 100,
                    'min_found': float(out_of_range[col].min()),
                    'max_found': float(out_of_range[col].max()),
                    'severity': 'warning' if len(out_of_range) / len(df) < 0.05 else 'error'
                })
        
        return {
            'is_valid': len(issues) == 0,
            'issues': issues
        }
    
    def validate_categoricals(self, df: pd.DataFrame) -> Dict[str, any]:
        """
        Validate that categorical columns contain only valid values.
        
        Args:
            df: DataFrame to validate
            
        Returns:
            Dictionary with validation results
        """
        issues = []
        
        for col, valid_values in self.valid_categoricals.items():
            if col not in df.columns:
                continue
            
            # Get unique values in the column
            unique_values = set(df[col].dropna().unique())
            
            # Check for invalid values
            invalid_values = unique_values - valid_values
            
            if invalid_values:
                invalid_count = df[col].isin(invalid_values).sum()
                issues.append({
                    'column': col,
                    'valid_values': list(valid_values),
                    'invalid_values': list(invalid_values),
                    'violations': invalid_count,
                    'percentage': (invalid_count / len(df)) * 100,
                    'severity': 'warning' if invalid_count / len(df) < 0.05 else 'error'
                })
        
        # Validate state codes
        if 'state' in df.columns:
            unique_states = set(df['state'].dropna().str.upper().unique())
            invalid_states = unique_states - self.valid_states
            
            if invalid_states:
                invalid_count = df['state'].str.upper().isin(invalid_states).sum()
                issues.append({
                    'column': 'state',
                    'valid_values': list(self.valid_states),
                    'invalid_values': list(invalid_states),
                    'violations': invalid_count,
                    'percentage': (invalid_count / len(df)) * 100,
                    'severity': 'warning' if invalid_count / len(df) < 0.05 else 'error'
                })
        
        return {
            'is_valid': len(issues) == 0,
            'issues': issues
        }
    
    def validate_business_rules(self, df: pd.DataFrame) -> Dict[str, any]:
        """
        Validate business rules and domain constraints.
        
        Rules:
        1. Price must be positive
        2. KM must be non-negative
        3. Year must be <= year_of_reference
        4. Age must be non-negative
        5. Age calculation consistency (age_years == year_of_reference - year)
        6. KM per year should be reasonable (< 50,000 km/year)
        7. Engine size must be positive
        8. Doors must be valid (2, 3, 4, or 5)
        
        Args:
            df: DataFrame to validate
            
        Returns:
            Dictionary with validation results
        """
        violations = []
        
        # Rule 1: Price must be positive
        if 'price' in df.columns:
            invalid_price = df[df['price'] <= 0]
            if len(invalid_price) > 0:
                violations.append({
                    'rule': 'Price > 0',
                    'violations': len(invalid_price),
                    'percentage': (len(invalid_price) / len(df)) * 100,
                    'description': 'Non-positive prices found',
                    'severity': 'error'
                })
        
        # Rule 2: KM must be non-negative
        if 'km' in df.columns:
            invalid_km = df[df['km'] < 0]
            if len(invalid_km) > 0:
                violations.append({
                    'rule': 'KM >= 0',
                    'violations': len(invalid_km),
                    'percentage': (len(invalid_km) / len(df)) * 100,
                    'description': 'Negative mileage found',
                    'severity': 'error'
                })
        
        # Rule 3: Year must be <= year_of_reference
        if 'year' in df.columns and 'year_of_reference' in df.columns:
            invalid_year_ref = df[df['year'] > df['year_of_reference']]
            if len(invalid_year_ref) > 0:
                violations.append({
                    'rule': 'Year <= year_of_reference',
                    'violations': len(invalid_year_ref),
                    'percentage': (len(invalid_year_ref) / len(df)) * 100,
                    'description': 'Vehicle year greater than reference year',
                    'severity': 'error'
                })
        
        # Rule 4: Age must be non-negative
        if 'age_years' in df.columns:
            invalid_age = df[df['age_years'] < 0]
            if len(invalid_age) > 0:
                violations.append({
                    'rule': 'Age >= 0',
                    'violations': len(invalid_age),
                    'percentage': (len(invalid_age) / len(df)) * 100,
                    'description': 'Negative age values found',
                    'severity': 'error'
                })
        
        # Rule 5: Age calculation consistency
        if all(col in df.columns for col in ['age_years', 'year_of_reference', 'year']):
            calculated_age = df['year_of_reference'] - df['year']
            invalid_age_calc = df[df['age_years'] != calculated_age]
            if len(invalid_age_calc) > 0:
                violations.append({
                    'rule': 'Age calculation consistency',
                    'violations': len(invalid_age_calc),
                    'percentage': (len(invalid_age_calc) / len(df)) * 100,
                    'description': 'Age calculation mismatch (age_years != year_of_reference - year)',
                    'severity': 'warning'
                })
        
        # Rule 6: KM per year should be reasonable (< 50,000 km/year)
        if all(col in df.columns for col in ['km', 'age_years']):
            # Avoid division by zero
            km_per_year = df['km'] / (df['age_years'] + 1)  # +1 to avoid division by zero
            invalid_km_rate = df[km_per_year > 50000]
            if len(invalid_km_rate) > 0:
                violations.append({
                    'rule': 'KM per year (< 50,000 km/year)',
                    'violations': len(invalid_km_rate),
                    'percentage': (len(invalid_km_rate) / len(df)) * 100,
                    'description': 'Unusually high mileage rate found',
                    'severity': 'warning'
                })
        
        # Rule 7: Engine size must be positive
        if 'engine_size' in df.columns:
            invalid_engine = df[df['engine_size'] <= 0]
            if len(invalid_engine) > 0:
                violations.append({
                    'rule': 'Engine size > 0',
                    'violations': len(invalid_engine),
                    'percentage': (len(invalid_engine) / len(df)) * 100,
                    'description': 'Non-positive engine sizes found',
                    'severity': 'error'
                })
        
        # Rule 8: Doors must be valid
        if 'doors' in df.columns:
            invalid_doors = df[~df['doors'].isin([2, 3, 4, 5])]
            if len(invalid_doors) > 0:
                violations.append({
                    'rule': 'Doors (2, 3, 4, or 5)',
                    'violations': len(invalid_doors),
                    'percentage': (len(invalid_doors) / len(df)) * 100,
                    'description': f'Invalid door counts: {sorted(invalid_doors["doors"].unique())}',
                    'severity': 'error'
                })
        
        return {
            'is_valid': len(violations) == 0,
            'violations': violations
        }
    
    def get_validation_report(self, validation_results: Dict[str, any]) -> str:
        """
        Generate a human-readable validation report.
        
        Args:
            validation_results: Results from validate() method
            
        Returns:
            Formatted validation report string
        """
        report = []
        report.append("=" * 60)
        report.append("Data Validation Report")
        report.append("=" * 60)
        report.append("")
        
        summary = validation_results['summary']
        report.append(f"Total Rows: {summary['total_rows']:,}")
        report.append(f"Total Issues: {summary['total_issues']}")
        report.append(f"Validation Status: {'✅ PASSED' if summary['validation_passed'] else '❌ FAILED'}")
        report.append("")
        
        # Schema validation
        schema = validation_results['schema_validation']
        report.append("Schema Validation:")
        if schema['is_valid']:
            report.append("  ✅ Schema is valid")
        else:
            report.append("  ❌ Schema validation failed:")
            for issue in schema['issues']:
                report.append(f"    - {issue['type']}: {issue}")
        report.append("")
        
        # Range validation
        ranges = validation_results['range_validation']
        report.append("Range Validation:")
        if ranges['is_valid']:
            report.append("  ✅ All ranges are valid")
        else:
            report.append("  ⚠️  Range validation issues:")
            for issue in ranges['issues']:
                report.append(
                    f"    - {issue['column']}: {issue['violations']:,} violations "
                    f"({issue['percentage']:.2f}%) - "
                    f"Range [{issue['min_found']:.2f}, {issue['max_found']:.2f}] "
                    f"outside expected [{issue['expected_range'][0]}, {issue['expected_range'][1]}]"
                )
        report.append("")
        
        # Categorical validation
        categoricals = validation_results['categorical_validation']
        report.append("Categorical Validation:")
        if categoricals['is_valid']:
            report.append("  ✅ All categorical values are valid")
        else:
            report.append("  ⚠️  Categorical validation issues:")
            for issue in categoricals['issues']:
                report.append(
                    f"    - {issue['column']}: {issue['violations']:,} violations "
                    f"({issue['percentage']:.2f}%) - "
                    f"Invalid values: {issue['invalid_values']}"
                )
        report.append("")
        
        # Business rules validation
        business = validation_results['business_rules_validation']
        report.append("Business Rules Validation:")
        if business['is_valid']:
            report.append("  ✅ All business rules are satisfied")
        else:
            report.append("  ⚠️  Business rule violations:")
            for violation in business['violations']:
                report.append(
                    f"    - {violation['rule']}: {violation['violations']:,} violations "
                    f"({violation['percentage']:.2f}%) - {violation['description']}"
                )
        report.append("")
        
        return "\n".join(report)

