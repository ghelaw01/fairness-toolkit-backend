"""Data Anonymization and Pseudonymization"""

import pandas as pd
import hashlib
from typing import List


def anonymize_data(df: pd.DataFrame, sensitive_columns: List[str]) -> pd.DataFrame:
    """
    Anonymize sensitive columns by removing or generalizing them.
    
    Args:
        df: DataFrame to anonymize
        sensitive_columns: List of column names to anonymize
        
    Returns:
        Anonymized DataFrame
    """
    df_anon = df.copy()
    
    for col in sensitive_columns:
        if col in df_anon.columns:
            # For numeric columns, generalize to ranges
            if pd.api.types.is_numeric_dtype(df_anon[col]):
                df_anon[col] = pd.cut(df_anon[col], bins=5, labels=['Very Low', 'Low', 'Medium', 'High', 'Very High'])
            else:
                # For categorical, replace with generic labels
                df_anon[col] = 'REDACTED'
    
    return df_anon


def pseudonymize_data(df: pd.DataFrame, id_columns: List[str]) -> pd.DataFrame:
    """
    Pseudonymize identifier columns using hashing.
    
    Args:
        df: DataFrame to pseudonymize
        id_columns: List of identifier column names
        
    Returns:
        Pseudonymized DataFrame
    """
    df_pseudo = df.copy()
    
    for col in id_columns:
        if col in df_pseudo.columns:
            df_pseudo[col] = df_pseudo[col].apply(
                lambda x: hashlib.sha256(str(x).encode()).hexdigest()[:16]
            )
    
    return df_pseudo


def apply_differential_privacy(df: pd.DataFrame, epsilon: float = 1.0) -> pd.DataFrame:
    """
    Apply differential privacy by adding noise to numeric columns.
    
    Args:
        df: DataFrame
        epsilon: Privacy parameter (smaller = more private)
        
    Returns:
        DataFrame with noise added
    """
    import numpy as np
    
    df_private = df.copy()
    
    for col in df_private.select_dtypes(include=[np.number]).columns:
        sensitivity = df_private[col].std()
        noise = np.random.laplace(0, sensitivity / epsilon, size=len(df_private))
        df_private[col] = df_private[col] + noise
    
    return df_private
