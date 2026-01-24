"""
Preprocessing utilities for wine quality data analysis.
"""
import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Union, Optional


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
