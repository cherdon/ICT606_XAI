"""
Preprocessing utilities for wine quality data analysis.
"""
import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Union, Optional
from sklearn.preprocessing import RobustScaler, PowerTransformer


def bin_column(
    df: pd.DataFrame,
    column: str,
    bin_logic: Dict[str, Tuple[Optional[float], Optional[float]]],
    new_column_name: Optional[str] = None,
    drop_original: bool = False
) -> pd.DataFrame:
    """
    Bin a numeric column into categorical bins based on provided logic.
    
    Parameters
    ----------
    df : pd.DataFrame
        Input dataframe
    column : str
        Name of the column to bin
    bin_logic : Dict[str, Tuple[Optional[float], Optional[float]]]
        Dictionary where keys are bin labels and values are tuples of (min_val, max_val).
        Use None for unbounded ranges.
        Example: {'low': (None, 4), 'medium': (4, 6), 'high': (7, None)}
        - 'low': values < 4
        - 'medium': values >= 4 and < 6
        - 'high': values >= 7
    new_column_name : str, optional
        Name for the new binned column. If None, uses f"{column}_binned"
    drop_original : bool, default False
        Whether to drop the original column after binning
        
    Returns
    -------
    pd.DataFrame
        DataFrame with the new binned column added
        
    Examples
    --------
    >>> bin_logic = {'low': (None, 6), 'medium': (6, 7), 'high': (7, None)}
    >>> df = bin_column(df, 'quality', bin_logic, new_column_name='quality_category')
    """
    df = df.copy()
    
    if new_column_name is None:
        new_column_name = f"{column}_binned"
    
    # Initialize the new column with None (object dtype for string labels)
    df[new_column_name] = pd.Series([None] * len(df), dtype='object')
    
    # Apply binning logic
    for label, (min_val, max_val) in bin_logic.items():
        if min_val is None and max_val is not None:
            # Values less than max_val
            mask = df[column] < max_val
        elif min_val is not None and max_val is None:
            # Values greater than or equal to min_val
            mask = df[column] >= min_val
        elif min_val is not None and max_val is not None:
            # Values in range [min_val, max_val)
            mask = (df[column] >= min_val) & (df[column] < max_val)
        else:
            # Both None - match all remaining NaN
            mask = df[new_column_name].isna()
        
        df.loc[mask, new_column_name] = label
    
    if drop_original:
        df = df.drop(columns=[column])
    
    return df


def create_binary_column(
    df: pd.DataFrame,
    column: str,
    threshold: float,
    new_column_name: Optional[str] = None,
    drop_original: bool = False
) -> pd.DataFrame:
    """
    Create a binary column based on a threshold.
    
    Parameters
    ----------
    df : pd.DataFrame
        Input dataframe
    column : str
        Name of the column to convert
    threshold : float
        Values >= threshold will be 1, values < threshold will be 0
    new_column_name : str, optional
        Name for the new binary column. If None, uses f"{column}_binary"
    drop_original : bool, default False
        Whether to drop the original column
        
    Returns
    -------
    pd.DataFrame
        DataFrame with the new binary column added
    """
    df = df.copy()
    
    if new_column_name is None:
        new_column_name = f"{column}_binary"
    
    df[new_column_name] = (df[column] >= threshold).astype(int)
    
    if drop_original:
        df = df.drop(columns=[column])
    
    return df


def check_class_distribution(
    df: pd.DataFrame,
    target_column: str
) -> pd.DataFrame:
    """
    Check the distribution of classes in a target column.
    
    Parameters
    ----------
    df : pd.DataFrame
        Input dataframe
    target_column : str
        Name of the target column
        
    Returns
    -------
    pd.DataFrame
        DataFrame with class counts and percentages
    """
    counts = df[target_column].value_counts()
    percentages = df[target_column].value_counts(normalize=True) * 100
    
    distribution = pd.DataFrame({
        'count': counts,
        'percentage': percentages.round(2)
    })
    
    return distribution.sort_index()


def handle_missing_values(
    df: pd.DataFrame,
    strategy: str = 'drop',
    fill_value: Optional[Union[float, Dict[str, float]]] = None
) -> pd.DataFrame:
    """
    Handle missing values in the dataframe.
    
    Parameters
    ----------
    df : pd.DataFrame
        Input dataframe
    strategy : str, default 'drop'
        Strategy for handling missing values:
        - 'drop': Drop rows with any missing values
        - 'fill_mean': Fill with column mean (numeric columns only)
        - 'fill_median': Fill with column median (numeric columns only)
        - 'fill_value': Fill with specified value
    fill_value : float or dict, optional
        Value to fill when strategy='fill_value'. Can be a single value
        or a dict mapping column names to values.
        
    Returns
    -------
    pd.DataFrame
        DataFrame with missing values handled
    """
    df = df.copy()
    
    if strategy == 'drop':
        return df.dropna()
    elif strategy == 'fill_mean':
        return df.fillna(df.mean(numeric_only=True))
    elif strategy == 'fill_median':
        return df.fillna(df.median(numeric_only=True))
    elif strategy == 'fill_value':
        if fill_value is None:
            raise ValueError("fill_value must be provided when strategy='fill_value'")
        return df.fillna(fill_value)
    else:
        raise ValueError(f"Unknown strategy: {strategy}")


def remove_duplicates(df: pd.DataFrame, subset: Optional[List[str]] = None) -> pd.DataFrame:
    """
    Remove duplicate rows from the dataframe.
    
    Parameters
    ----------
    df : pd.DataFrame
        Input dataframe
    subset : list of str, optional
        Columns to consider for identifying duplicates.
        If None, uses all columns.
        
    Returns
    -------
    pd.DataFrame
        DataFrame with duplicates removed
    """
    return df.drop_duplicates(subset=subset)


def encode_categorical_columns(
    df: pd.DataFrame,
    columns: Optional[List[str]] = None,
    encoding: str = 'label'
) -> Tuple[pd.DataFrame, Dict[str, Dict]]:
    """
    Encode categorical columns for machine learning.
    
    Parameters
    ----------
    df : pd.DataFrame
        Input dataframe
    columns : list of str, optional
        Columns to encode. If None, encodes all object/category columns.
    encoding : str, default 'label'
        Encoding type:
        - 'label': Label encoding (0, 1, 2, ...)
        - 'onehot': One-hot encoding
        
    Returns
    -------
    Tuple[pd.DataFrame, Dict[str, Dict]]
        - Encoded DataFrame
        - Dictionary mapping column names to their encoding mappings
    """
    df = df.copy()
    
    if columns is None:
        columns = df.select_dtypes(include=['object', 'category']).columns.tolist()
    
    encodings = {}
    
    if encoding == 'label':
        for col in columns:
            unique_values = df[col].unique()
            mapping = {val: idx for idx, val in enumerate(sorted(unique_values))}
            df[col] = df[col].map(mapping)
            encodings[col] = mapping
    elif encoding == 'onehot':
        df = pd.get_dummies(df, columns=columns, drop_first=False)
        encodings = {col: 'onehot' for col in columns}
    else:
        raise ValueError(f"Unknown encoding: {encoding}")
    
    return df, encodings


def load_wine_data(filepath: str, wine_type: str = 'red') -> pd.DataFrame:
    """
    Load wine quality dataset.
    
    Parameters
    ----------
    filepath : str
        Path to the CSV file
    wine_type : str, default 'red'
        Type of wine ('red' or 'white') - used for logging purposes
        
    Returns
    -------
    pd.DataFrame
        Loaded wine data
    """
    # Wine quality datasets use semicolon separator
    df = pd.read_csv(filepath, sep=';')
    print(f"Loaded {wine_type} wine data: {df.shape[0]} samples, {df.shape[1]} features")
    return df


def remove_multicollinear_features(
    df: pd.DataFrame,
    target_column: str,
    threshold: float = 0.8,
    verbose: bool = True
) -> Tuple[pd.DataFrame, List[str]]:
    """
    Remove features with correlation above threshold, keeping the one
    more correlated with the target variable.
    
    Parameters
    ----------
    df : pd.DataFrame
        Input dataframe with features and target
    target_column : str
        Name of the target column
    threshold : float, default 0.8
        Correlation threshold above which to remove features
    verbose : bool, default True
        Whether to print information about removed features
        
    Returns
    -------
    Tuple[pd.DataFrame, List[str]]
        - DataFrame with multicollinear features removed
        - List of removed feature names
    """
    df = df.copy()
    
    # Get feature columns (exclude target)
    feature_cols = [col for col in df.columns if col != target_column]
    
    # Calculate correlation matrix for features only
    corr_matrix = df[feature_cols].corr().abs()
    
    # Calculate correlation with target (handle categorical targets)
    target_is_numeric = pd.api.types.is_numeric_dtype(df[target_column])
    
    if target_is_numeric:
        target_corr = df[feature_cols].corrwith(df[target_column]).abs()
    else:
        # For categorical targets, encode temporarily to compute correlation
        from sklearn.preprocessing import LabelEncoder
        le = LabelEncoder()
        target_encoded = le.fit_transform(df[target_column])
        target_series = pd.Series(target_encoded, index=df.index)
        target_corr = df[feature_cols].corrwith(target_series).abs()
    
    # Find highly correlated pairs
    removed_features = set()
    
    for i in range(len(feature_cols)):
        if feature_cols[i] in removed_features:
            continue
        for j in range(i + 1, len(feature_cols)):
            if feature_cols[j] in removed_features:
                continue
            
            if corr_matrix.iloc[i, j] > threshold:
                feat_i = feature_cols[i]
                feat_j = feature_cols[j]
                
                # Keep the feature more correlated with target
                if target_corr[feat_i] >= target_corr[feat_j]:
                    to_remove = feat_j
                    to_keep = feat_i
                else:
                    to_remove = feat_i
                    to_keep = feat_j
                
                removed_features.add(to_remove)
                
                if verbose:
                    print(f"  Removing '{to_remove}' (corr with '{to_keep}': {corr_matrix.iloc[i, j]:.3f})")
                    print(f"    → Kept '{to_keep}' (target corr: {target_corr[to_keep]:.3f})")
    
    # Remove the features
    removed_list = list(removed_features)
    df = df.drop(columns=removed_list)
    
    if verbose:
        print(f"  Total features removed: {len(removed_list)}")
        print(f"  Remaining features: {len(df.columns) - 1}")  # -1 for target
    
    return df, removed_list


def apply_yeo_johnson_transform(
    X_train: np.ndarray,
    X_test: np.ndarray,
    feature_names: List[str],
    verbose: bool = True
) -> Tuple[np.ndarray, np.ndarray, PowerTransformer]:
    """
    Apply Yeo-Johnson power transformation to handle skewness.
    Fits on training data and transforms both train and test.
    
    Parameters
    ----------
    X_train : np.ndarray
        Training feature matrix
    X_test : np.ndarray
        Test feature matrix
    feature_names : List[str]
        Names of features (for verbose output)
    verbose : bool, default True
        Whether to print transformation info
        
    Returns
    -------
    Tuple[np.ndarray, np.ndarray, PowerTransformer]
        - Transformed training data
        - Transformed test data
        - Fitted PowerTransformer object
    """
    transformer = PowerTransformer(method='yeo-johnson', standardize=False)
    
    X_train_transformed = transformer.fit_transform(X_train)
    X_test_transformed = transformer.transform(X_test)
    
    if verbose:
        print(f"  Applied Yeo-Johnson transformation to {len(feature_names)} features")
    
    return X_train_transformed, X_test_transformed, transformer


def apply_robust_scaling(
    X_train: np.ndarray,
    X_test: np.ndarray,
    verbose: bool = True
) -> Tuple[np.ndarray, np.ndarray, RobustScaler]:
    """
    Apply RobustScaler to features. Uses median and IQR, making it
    robust to outliers.
    Fits on training data and transforms both train and test.
    
    Parameters
    ----------
    X_train : np.ndarray
        Training feature matrix
    X_test : np.ndarray
        Test feature matrix
    verbose : bool, default True
        Whether to print scaling info
        
    Returns
    -------
    Tuple[np.ndarray, np.ndarray, RobustScaler]
        - Scaled training data
        - Scaled test data
        - Fitted RobustScaler object
    """
    scaler = RobustScaler()
    
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    if verbose:
        print(f"  Applied RobustScaler (median/IQR-based)")
    
    return X_train_scaled, X_test_scaled, scaler


def get_preprocessing_pipeline_info() -> str:
    """
    Return a string describing the preprocessing pipeline for documentation.
    """
    return """
    Preprocessing Pipeline:
    1. Binary target creation (quality >= 7 → Premium)
    2. Handle missing values and duplicates
    3. Remove multicollinear features (correlation > 0.8)
    4. Stratified train/test split (80/20)
    5. Yeo-Johnson transformation (fit on train)
    6. RobustScaler (fit on train)
    7. SMOTE oversampling (on training data only)
    """
