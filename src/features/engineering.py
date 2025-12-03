"""
Feature engineering module for AutoValuePredict ML project.

This module provides utilities for creating and transforming features,
including temporal features, categorical encoding, numerical transformations,
and location features.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple, Set
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.base import BaseEstimator, TransformerMixin
import logging

logger = logging.getLogger(__name__)


class TemporalFeatureCreator(BaseEstimator, TransformerMixin):
    """
    Create temporal features from vehicle data.
    
    Features created:
    - age_years: Vehicle age (already exists, verified)
    - age_squared: Age squared for non-linear relationships
    """
    
    def __init__(self):
        """Initialize the temporal feature creator."""
        pass
    
    def fit(self, X: pd.DataFrame, y=None):
        """Fit the transformer (no-op for this transformer)."""
        return self
    
    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Create temporal features.
        
        Args:
            X: Input DataFrame with 'age_years' column
            
        Returns:
            DataFrame with added temporal features
        """
        X = X.copy()
        
        # Verify age_years exists
        if 'age_years' not in X.columns:
            if 'year' in X.columns and 'year_of_reference' in X.columns:
                X['age_years'] = X['year_of_reference'] - X['year']
                logger.info("Created age_years from year and year_of_reference")
            else:
                logger.warning("Cannot create age_years: missing required columns")
                return X
        
        # Create age squared for non-linear relationship
        X['age_squared'] = X['age_years'] ** 2
        
        return X


class CategoricalEncoder(BaseEstimator, TransformerMixin):
    """
    Encode categorical features.
    
    Uses:
    - One-hot encoding for low cardinality features (fuel_type, transmission, condition)
    - Target encoding for high cardinality features (brand, model, state)
    """
    
    def __init__(
        self,
        one_hot_features: Optional[List[str]] = None,
        target_encode_features: Optional[List[str]] = None,
        target_col: str = 'price'
    ):
        """
        Initialize the categorical encoder.
        
        Args:
            one_hot_features: List of features to one-hot encode
            target_encode_features: List of features to target encode
            target_col: Name of target column for target encoding
        """
        if one_hot_features is None:
            one_hot_features = ['fuel_type', 'transmission', 'condition']
        
        if target_encode_features is None:
            target_encode_features = ['brand', 'model', 'state']
        
        self.one_hot_features = one_hot_features
        self.target_encode_features = target_encode_features
        self.target_col = target_col
        
        # Will store encoding mappings during fit
        self.target_encodings: Dict[str, Dict] = {}
        self.feature_names_: List[str] = []
    
    def fit(self, X: pd.DataFrame, y: Optional[pd.Series] = None):
        """
        Fit the encoder on training data.
        
        Args:
            X: Input DataFrame
            y: Target series (optional, will use X[self.target_col] if not provided)
            
        Returns:
            Self
        """
        if y is None and self.target_col in X.columns:
            y = X[self.target_col]
        elif y is None:
            raise ValueError(
                f"Target column '{self.target_col}' not found in X and y not provided"
            )
        
        # Fit target encodings
        for feature in self.target_encode_features:
            if feature in X.columns:
                # Calculate mean target value per category
                encoding_map = X.groupby(feature)[y.name if hasattr(y, 'name') else self.target_col].mean().to_dict()
                self.target_encodings[feature] = encoding_map
                logger.info(
                    f"Fitted target encoding for {feature}: "
                    f"{len(encoding_map)} categories"
                )
        
        # Store feature names and unique values for one-hot encoding
        self.feature_names_ = []
        self.one_hot_categories: Dict[str, List] = {}
        for feature in self.one_hot_features:
            if feature in X.columns:
                unique_values = sorted(X[feature].unique())
                self.one_hot_categories[feature] = unique_values
                for value in unique_values:
                    self.feature_names_.append(f"{feature}_{value}")
        
        return self
    
    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Transform categorical features.
        
        Args:
            X: Input DataFrame
            
        Returns:
            DataFrame with encoded features
        """
        X = X.copy()
        
        # Apply target encoding
        for feature in self.target_encode_features:
            if feature in X.columns:
                encoding_map = self.target_encodings.get(feature, {})
                # Use mean of all encodings for unseen categories
                default_value = np.mean(list(encoding_map.values())) if encoding_map else 0.0
                X[f"{feature}_encoded"] = X[feature].map(encoding_map).fillna(default_value)
                # Drop original column
                X = X.drop(columns=[feature])
        
        # Apply one-hot encoding
        for feature in self.one_hot_features:
            if feature in X.columns:
                # Get categories from fit to ensure consistency
                categories = self.one_hot_categories.get(feature, [])
                
                # Create dummies
                dummies = pd.get_dummies(X[feature], prefix=feature, dtype=int)
                
                # Ensure all categories from fit are present (for consistency)
                if categories:
                    for cat in categories:
                        col_name = f"{feature}_{cat}"
                        if col_name not in dummies.columns:
                            dummies[col_name] = 0
                    
                    # Reorder columns to match fit order
                    expected_cols = [f"{feature}_{cat}" for cat in categories]
                    # Only include columns that exist
                    available_cols = [col for col in expected_cols if col in dummies.columns]
                    # Add any extra columns that appeared in transform but not in fit
                    extra_cols = [col for col in dummies.columns if col not in expected_cols]
                    dummies = dummies[available_cols + extra_cols]
                
                X = pd.concat([X, dummies], axis=1)
                # Drop original column
                X = X.drop(columns=[feature])
        
        return X


class NumericalTransformer(BaseEstimator, TransformerMixin):
    """
    Transform numerical features.
    
    Transformations:
    - Log transformation for skewed features (price, km)
    - Standardization/normalization
    """
    
    def __init__(
        self,
        log_transform_features: Optional[List[str]] = None,
        standardize_features: Optional[List[str]] = None,
        normalize_features: Optional[List[str]] = None,
        use_log: bool = True,
        use_standardization: bool = True
    ):
        """
        Initialize the numerical transformer.
        
        Args:
            log_transform_features: Features to apply log transformation
            standardize_features: Features to standardize (z-score)
            normalize_features: Features to normalize (min-max)
            use_log: Whether to apply log transformation
            use_standardization: Whether to apply standardization
        """
        if log_transform_features is None:
            log_transform_features = ['price', 'km']
        
        if standardize_features is None:
            # Will standardize all numerical features except target
            standardize_features = []
        
        self.log_transform_features = log_transform_features
        self.standardize_features = standardize_features
        self.normalize_features = normalize_features or []
        self.use_log = use_log
        self.use_standardization = use_standardization
        
        # Scalers
        self.scalers: Dict[str, StandardScaler] = {}
        self.normalizers: Dict[str, MinMaxScaler] = {}
    
    def fit(self, X: pd.DataFrame, y=None):
        """
        Fit the transformer on training data.
        
        Args:
            X: Input DataFrame
            y: Target (ignored)
            
        Returns:
            Self
        """
        # Fit standardizers
        if self.use_standardization:
            # If no specific features provided, standardize all numerical features
            if not self.standardize_features:
                numeric_cols = X.select_dtypes(include=[np.number]).columns.tolist()
                # Exclude target and already log-transformed features
                exclude = ['price', 'km'] + [f"{f}_log" for f in self.log_transform_features]
                self.standardize_features = [
                    col for col in numeric_cols 
                    if col not in exclude and not col.endswith('_log')
                ]
            
            for feature in self.standardize_features:
                if feature in X.columns:
                    scaler = StandardScaler()
                    scaler.fit(X[[feature]])
                    self.scalers[feature] = scaler
        
        # Fit normalizers
        for feature in self.normalize_features:
            if feature in X.columns:
                normalizer = MinMaxScaler()
                normalizer.fit(X[[feature]])
                self.normalizers[feature] = normalizer
        
        return self
    
    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Transform numerical features.
        
        Args:
            X: Input DataFrame
            
        Returns:
            DataFrame with transformed features
        """
        X = X.copy()
        
        # Apply log transformation
        if self.use_log:
            for feature in self.log_transform_features:
                if feature in X.columns:
                    # Add small epsilon to avoid log(0)
                    X[f"{feature}_log"] = np.log1p(X[feature])
                    logger.debug(f"Applied log transformation to {feature}")
        
        # Apply standardization
        for feature, scaler in self.scalers.items():
            if feature in X.columns:
                X[feature] = scaler.transform(X[[feature]])
        
        # Apply normalization
        for feature, normalizer in self.normalizers.items():
            if feature in X.columns:
                X[feature] = normalizer.transform(X[[feature]])
        
        return X


class LocationFeatureCreator(BaseEstimator, TransformerMixin):
    """
    Create location-based features.
    
    Features:
    - State encoding (one-hot or target encoding, handled by CategoricalEncoder)
    - Region encoding (Norte, Nordeste, Sul, Sudeste, Centro-Oeste)
    """
    
    # Brazilian regions mapping
    REGIONS = {
        'Norte': ['AC', 'AP', 'AM', 'PA', 'RO', 'RR', 'TO'],
        'Nordeste': ['AL', 'BA', 'CE', 'MA', 'PB', 'PE', 'PI', 'RN', 'SE'],
        'Centro-Oeste': ['DF', 'GO', 'MT', 'MS'],
        'Sudeste': ['ES', 'MG', 'RJ', 'SP'],
        'Sul': ['PR', 'RS', 'SC']
    }
    
    def __init__(self, create_region: bool = True):
        """
        Initialize the location feature creator.
        
        Args:
            create_region: Whether to create region feature
        """
        self.create_region = create_region
        self.region_map: Dict[str, str] = {}
    
    def fit(self, X: pd.DataFrame, y=None):
        """Fit the transformer (build region mapping)."""
        if self.create_region:
            # Build reverse mapping: state -> region
            for region, states in self.REGIONS.items():
                for state in states:
                    self.region_map[state] = region
        
        return self
    
    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Create location features.
        
        Args:
            X: Input DataFrame with 'state' column
            
        Returns:
            DataFrame with added location features
        """
        X = X.copy()
        
        if self.create_region and 'state' in X.columns:
            X['region'] = X['state'].map(self.region_map).fillna('Unknown')
            logger.debug("Created region feature from state")
        
        return X


class AdvancedFeatureCreator(BaseEstimator, TransformerMixin):
    """
    Create advanced features (Phase 3.1.1 - Optional Enhancements).
    
    Features created:
    - Depreciation rate calculation
    - Frequency encoding (brand/model frequency)
    - Interaction features (Brand × Year, Fuel × Transmission, Age × Condition, Km per year)
    - Binning (Price bins, Age bins, Mileage bins)
    - Advanced location features (City size category)
    """
    
    def __init__(
        self,
        create_depreciation: bool = True,
        create_frequency_encoding: bool = True,
        create_interactions: bool = True,
        create_binning: bool = True,
        create_city_size: bool = True,
        n_bins: int = 5
    ):
        """
        Initialize the advanced feature creator.
        
        Args:
            create_depreciation: Whether to create depreciation rate
            create_frequency_encoding: Whether to create frequency encodings
            create_interactions: Whether to create interaction features
            create_binning: Whether to create binned features
            create_city_size: Whether to create city size category
            n_bins: Number of bins for binning features
        """
        self.create_depreciation = create_depreciation
        self.create_frequency_encoding = create_frequency_encoding
        self.create_interactions = create_interactions
        self.create_binning = create_binning
        self.create_city_size = create_city_size
        self.n_bins = n_bins
        
        # Will store frequency mappings during fit
        self.brand_frequency: Dict[str, float] = {}
        self.model_frequency: Dict[str, float] = {}
        
        # Will store bin edges during fit
        self.price_bins: Optional[np.ndarray] = None
        self.age_bins: Optional[np.ndarray] = None
        self.km_bins: Optional[np.ndarray] = None
    
    def fit(self, X: pd.DataFrame, y: Optional[pd.Series] = None):
        """
        Fit the transformer on training data.
        
        Args:
            X: Training DataFrame
            y: Target series (optional, needed for price bins)
            
        Returns:
            Self
        """
        # Fit frequency encodings
        if self.create_frequency_encoding:
            if 'brand' in X.columns:
                self.brand_frequency = (X['brand'].value_counts() / len(X)).to_dict()
            
            if 'model' in X.columns:
                self.model_frequency = (X['model'].value_counts() / len(X)).to_dict()
        
        # Fit bin edges
        if self.create_binning:
            if y is not None and 'price' in X.columns:
                # Use target for price bins
                self.price_bins = pd.qcut(y, q=self.n_bins, duplicates='drop', retbins=True)[1]
            elif 'price' in X.columns:
                self.price_bins = pd.qcut(X['price'], q=self.n_bins, duplicates='drop', retbins=True)[1]
            
            if 'age_years' in X.columns:
                self.age_bins = pd.qcut(
                    X['age_years'], 
                    q=self.n_bins, 
                    duplicates='drop', 
                    retbins=True
                )[1]
            
            if 'km' in X.columns:
                self.km_bins = pd.qcut(
                    X['km'], 
                    q=self.n_bins, 
                    duplicates='drop', 
                    retbins=True
                )[1]
        
        return self
    
    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Create advanced features.
        
        Args:
            X: Input DataFrame
            
        Returns:
            DataFrame with added advanced features
        """
        X = X.copy()
        
        # 1. Depreciation rate calculation
        if self.create_depreciation and 'price' in X.columns and 'age_years' in X.columns:
            # Simple depreciation: price / (age_years + 1) to avoid division by zero
            X['depreciation_rate'] = X['price'] / (X['age_years'] + 1)
            logger.debug("Created depreciation_rate feature")
        
        # 2. Frequency encoding
        if self.create_frequency_encoding:
            if 'brand' in X.columns and self.brand_frequency:
                X['brand_frequency'] = X['brand'].map(self.brand_frequency).fillna(0)
            
            if 'model' in X.columns and self.model_frequency:
                X['model_frequency'] = X['model'].map(self.model_frequency).fillna(0)
        
        # 3. Interaction features
        if self.create_interactions:
            # Brand × Year
            if 'brand' in X.columns and 'year' in X.columns:
                # Create interaction as string then encode (will be handled by categorical encoder)
                X['brand_year_interaction'] = X['brand'].astype(str) + '_' + X['year'].astype(str)
            
            # Fuel × Transmission
            if 'fuel_type' in X.columns and 'transmission' in X.columns:
                X['fuel_transmission_interaction'] = (
                    X['fuel_type'].astype(str) + '_' + X['transmission'].astype(str)
                )
            
            # Age × Condition
            if 'age_years' in X.columns and 'condition' in X.columns:
                # Create bins for age first
                age_bins = pd.cut(X['age_years'], bins=5, labels=['Very_New', 'New', 'Medium', 'Old', 'Very_Old'])
                X['age_condition_interaction'] = age_bins.astype(str) + '_' + X['condition'].astype(str)
            
            # Km per year
            if 'km' in X.columns and 'age_years' in X.columns:
                X['km_per_year'] = X['km'] / (X['age_years'] + 1)  # Avoid division by zero
                logger.debug("Created km_per_year interaction feature")
        
        # 4. Binning
        if self.create_binning:
            if 'price' in X.columns and self.price_bins is not None:
                try:
                    X['price_bin'] = pd.cut(X['price'], bins=self.price_bins, include_lowest=True, duplicates='drop')
                except Exception as e:
                    logger.warning(f"Could not create price bins: {e}")
            
            if 'age_years' in X.columns and self.age_bins is not None:
                try:
                    X['age_bin'] = pd.cut(X['age_years'], bins=self.age_bins, include_lowest=True, duplicates='drop')
                except Exception as e:
                    logger.warning(f"Could not create age bins: {e}")
            
            if 'km' in X.columns and self.km_bins is not None:
                try:
                    X['km_bin'] = pd.cut(X['km'], bins=self.km_bins, include_lowest=True, duplicates='drop')
                except Exception as e:
                    logger.warning(f"Could not create km bins: {e}")
        
        # 5. Advanced location features - City size category
        if self.create_city_size and 'city' in X.columns:
            # Simple heuristic: count occurrences as proxy for city size
            city_counts = X['city'].value_counts()
            # Categorize: Large (>1000), Medium (100-1000), Small (<100)
            city_size_map = {}
            for city, count in city_counts.items():
                if count > 1000:
                    city_size_map[city] = 'Large'
                elif count > 100:
                    city_size_map[city] = 'Medium'
                else:
                    city_size_map[city] = 'Small'
            
            X['city_size'] = X['city'].map(city_size_map).fillna('Small')
            logger.debug("Created city_size feature")
        
        return X


class FeatureEngineeringPipeline:
    """
    Complete feature engineering pipeline combining all transformations.
    
    This class orchestrates all feature engineering steps in the correct order.
    """
    
    def __init__(
        self,
        use_temporal_features: bool = True,
        use_categorical_encoding: bool = True,
        use_numerical_transforms: bool = True,
        use_location_features: bool = True,
        use_advanced_features: bool = False,  # Optional enhancements
        target_col: str = 'price',
        random_seed: int = 42
    ):
        """
        Initialize the feature engineering pipeline.
        
        Args:
            use_temporal_features: Whether to create temporal features
            use_categorical_encoding: Whether to encode categorical features
            use_numerical_transforms: Whether to transform numerical features
            use_location_features: Whether to create location features
            use_advanced_features: Whether to create advanced features (Phase 3.1.1 - Optional)
            target_col: Name of target column
            random_seed: Random seed for reproducibility
        """
        self.use_temporal_features = use_temporal_features
        self.use_categorical_encoding = use_categorical_encoding
        self.use_numerical_transforms = use_numerical_transforms
        self.use_location_features = use_location_features
        self.use_advanced_features = use_advanced_features
        self.target_col = target_col
        self.random_seed = random_seed
        
        # Initialize transformers
        self.temporal_creator = TemporalFeatureCreator() if use_temporal_features else None
        self.location_creator = LocationFeatureCreator() if use_location_features else None
        self.advanced_creator = AdvancedFeatureCreator() if use_advanced_features else None
        self.categorical_encoder = CategoricalEncoder(target_col=target_col) if use_categorical_encoding else None
        self.numerical_transformer = NumericalTransformer() if use_numerical_transforms else None
        
        self.is_fitted = False
    
    def fit(self, X: pd.DataFrame, y: Optional[pd.Series] = None):
        """
        Fit all transformers on training data.
        
        Args:
            X: Training DataFrame
            y: Target series (optional)
            
        Returns:
            Self
        """
        logger.info("Fitting feature engineering pipeline...")
        
        # Prepare target
        if y is None and self.target_col in X.columns:
            y = X[self.target_col].copy()
            X_work = X.drop(columns=[self.target_col])
        elif y is None:
            X_work = X.copy()
        else:
            X_work = X.copy()
        
        # Step 1: Create temporal features
        if self.temporal_creator:
            X_work = self.temporal_creator.fit_transform(X_work)
        
        # Step 2: Create location features
        if self.location_creator:
            X_work = self.location_creator.fit_transform(X_work)
        
        # Step 2.5: Create advanced features (before encoding, so interactions can be encoded)
        if self.advanced_creator:
            X_work = self.advanced_creator.fit_transform(X_work, y)
        
        # Step 3: Fit categorical encoder (needs target)
        if self.categorical_encoder:
            if y is not None:
                X_with_target = X_work.copy()
                if isinstance(y, pd.Series):
                    X_with_target[self.target_col] = y.values
                else:
                    X_with_target[self.target_col] = y
                self.categorical_encoder.fit(X_with_target, y)
            else:
                logger.warning("Cannot fit categorical encoder: target not available")
        
        # Step 4: Fit numerical transformer
        if self.numerical_transformer:
            self.numerical_transformer.fit(X_work)
        
        self.is_fitted = True
        logger.info("Feature engineering pipeline fitted successfully")
        
        return self
    
    def transform(self, X: pd.DataFrame, y: Optional[pd.Series] = None) -> pd.DataFrame:
        """
        Transform data using fitted transformers.
        
        Args:
            X: Input DataFrame
            y: Target series (optional, only needed if not in X)
            
        Returns:
            Transformed DataFrame
        """
        if not self.is_fitted:
            raise ValueError("Pipeline must be fitted before transform. Call fit() first.")
        
        logger.info("Transforming features...")
        
        # Prepare target
        if y is None and self.target_col in X.columns:
            y = X[self.target_col].copy()
            X_work = X.drop(columns=[self.target_col])
        else:
            X_work = X.copy()
        
        # Step 1: Create temporal features
        if self.temporal_creator:
            X_work = self.temporal_creator.transform(X_work)
        
        # Step 2: Create location features
        if self.location_creator:
            X_work = self.location_creator.transform(X_work)
        
        # Step 2.5: Create advanced features
        if self.advanced_creator:
            X_work = self.advanced_creator.transform(X_work)
        
        # Step 3: Encode categorical features
        if self.categorical_encoder:
            X_work = self.categorical_encoder.transform(X_work)
        
        # Step 4: Transform numerical features
        if self.numerical_transformer:
            X_work = self.numerical_transformer.transform(X_work)
        
        logger.info(f"Feature engineering completed: {X_work.shape[1]} features")
        
        return X_work
    
    def fit_transform(self, X: pd.DataFrame, y: Optional[pd.Series] = None) -> pd.DataFrame:
        """
        Fit and transform in one step.
        
        Args:
            X: Input DataFrame
            y: Target series (optional)
            
        Returns:
            Transformed DataFrame
        """
        return self.fit(X, y).transform(X, y)

