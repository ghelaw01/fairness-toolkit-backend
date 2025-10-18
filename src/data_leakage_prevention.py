"""
Data Leakage Prevention Module

This module prevents common data leakage issues in fairness analysis by:
1. Identifying and blocking leaky features
2. Validating feature sets
3. Providing safe default features for known datasets
"""

from typing import List, Dict, Set, Tuple
import pandas as pd

# Known leaky features for COMPAS dataset
COMPAS_LEAKY_FEATURES = {
    # Direct target leakage
    'is_recid', 'r_case_number', 'r_charge_degree', 'r_days_from_arrest',
    'r_offense_date', 'r_charge_desc', 'r_jail_in', 'r_jail_out',
    'violent_recid', 'is_violent_recid', 'vr_case_number', 'vr_charge_degree',
    'vr_offense_date', 'vr_charge_desc', 'two_year_recid',
    
    # COMPAS's own predictions (circular reasoning)
    'decile_score', 'score_text', 'v_decile_score', 'v_score_text',
    
    # Post-prediction information
    'in_custody', 'out_custody', 'c_jail_in', 'c_jail_out',
    'c_days_from_compas', 'days_b_screening_arrest',
    
    # Identifiers (not predictive, just noise)
    'id', 'name', 'first', 'last', 'c_case_number',
    
    # Dates (can leak temporal information)
    'compas_screening_date', 'screening_date', 'dob',
    'c_offense_date', 'c_arrest_date', 'start', 'end',
    
    # Event flag (related to target)
    'event',
    
    # Assessment type (metadata)
    'type_of_assessment', 'v_type_of_assessment',
}

# Safe features for COMPAS dataset
COMPAS_SAFE_FEATURES = {
    # Demographics (legitimate predictors)
    'age', 'sex', 'race', 'age_cat',
    
    # Criminal history (legitimate predictors)
    'juv_fel_count', 'juv_misd_count', 'juv_other_count', 'priors_count',
    
    # Current charge information (legitimate)
    'c_charge_degree', 'c_charge_desc',
}

# Sensitive attributes for COMPAS
COMPAS_SENSITIVE_ATTRS = {'race', 'sex', 'age_cat', 'age'}

def detect_leaky_features(
    df: pd.DataFrame,
    target_column: str,
    feature_columns: List[str],
    dataset_name: str = "unknown"
) -> Tuple[List[str], List[str], Dict[str, str]]:
    """
    Detect potentially leaky features in the dataset.
    
    Returns:
        - leaky_features: List of features that are definitely leaky
        - suspicious_features: List of features that might be leaky
        - reasons: Dict mapping feature names to reasons why they're leaky
    """
    leaky_features = []
    suspicious_features = []
    reasons = {}
    
    # Check against known leaky features for COMPAS
    if 'compas' in dataset_name.lower() or 'recid' in target_column.lower():
        for feat in feature_columns:
            if feat.lower() in {f.lower() for f in COMPAS_LEAKY_FEATURES}:
                leaky_features.append(feat)
                reasons[feat] = "Known leaky feature for COMPAS dataset"
    
    # Generic leakage detection
    for feat in feature_columns:
        feat_lower = feat.lower()
        
        # Check for target-related names
        if target_column.lower() in feat_lower and feat != target_column:
            if feat not in leaky_features:
                suspicious_features.append(feat)
                reasons[feat] = f"Feature name contains target column name '{target_column}'"
        
        # Check for prediction-related names
        if any(keyword in feat_lower for keyword in ['score', 'prediction', 'pred', 'risk', 'label']):
            if feat not in leaky_features:
                suspicious_features.append(feat)
                reasons[feat] = "Feature name suggests it might be a prediction or score"
        
        # Check for high correlation with target
        if feat in df.columns and target_column in df.columns:
            try:
                if df[feat].dtype in ['int64', 'float64'] and df[target_column].dtype in ['int64', 'float64']:
                    corr = abs(df[feat].corr(df[target_column]))
                    if corr > 0.95:
                        if feat not in leaky_features and feat not in suspicious_features:
                            suspicious_features.append(feat)
                            reasons[feat] = f"Extremely high correlation with target ({corr:.3f})"
            except:
                pass
    
    return leaky_features, suspicious_features, reasons


def get_safe_features(
    df: pd.DataFrame,
    target_column: str,
    sensitive_attributes: List[str],
    dataset_name: str = "unknown",
    auto_select: bool = True
) -> Tuple[List[str], Dict[str, str]]:
    """
    Get safe features for modeling, avoiding data leakage.
    
    Args:
        df: DataFrame
        target_column: Name of target column
        sensitive_attributes: List of sensitive attribute names
        dataset_name: Name of dataset (for known dataset handling)
        auto_select: If True, automatically select safe features
    
    Returns:
        - safe_features: List of safe feature names
        - warnings: Dict of warnings about excluded features
    """
    warnings = {}
    
    # For COMPAS dataset, use known safe features
    if 'compas' in dataset_name.lower() or 'recid' in target_column.lower():
        available_safe = [f for f in COMPAS_SAFE_FEATURES if f in df.columns]
        # Remove sensitive attributes from features
        safe_features = [f for f in available_safe if f not in sensitive_attributes]
        
        # Add warning about excluded features
        all_numeric = df.select_dtypes(include=['number']).columns.tolist()
        excluded = set(all_numeric) - set(safe_features) - {target_column} - set(sensitive_attributes)
        if excluded:
            warnings['excluded_features'] = f"Excluded {len(excluded)} potentially leaky features: {', '.join(list(excluded)[:5])}{'...' if len(excluded) > 5 else ''}"
        
        return safe_features, warnings
    
    # Generic safe feature selection
    if auto_select:
        # Start with all numeric columns
        numeric_cols = df.select_dtypes(include=['number']).columns.tolist()
        
        # Remove target and sensitive attributes
        candidate_features = [c for c in numeric_cols if c != target_column and c not in sensitive_attributes]
        
        # Detect and remove leaky features
        leaky, suspicious, reasons = detect_leaky_features(df, target_column, candidate_features, dataset_name)
        
        safe_features = [f for f in candidate_features if f not in leaky and f not in suspicious]
        
        if leaky:
            warnings['leaky_features'] = f"Removed {len(leaky)} leaky features: {', '.join(leaky[:5])}{'...' if len(leaky) > 5 else ''}"
        if suspicious:
            warnings['suspicious_features'] = f"Removed {len(suspicious)} suspicious features: {', '.join(suspicious[:5])}{'...' if len(suspicious) > 5 else ''}"
        
        return safe_features, warnings
    
    return [], {}


def validate_feature_set(
    df: pd.DataFrame,
    target_column: str,
    feature_columns: List[str],
    sensitive_attributes: List[str],
    dataset_name: str = "unknown"
) -> Tuple[bool, List[str], Dict[str, str]]:
    """
    Validate that a feature set doesn't contain data leakage.
    
    Returns:
        - is_valid: True if feature set is valid
        - errors: List of error messages
        - warnings: Dict of warning messages
    """
    errors = []
    warnings = {}
    
    # Detect leaky features
    leaky, suspicious, reasons = detect_leaky_features(df, target_column, feature_columns, dataset_name)
    
    if leaky:
        errors.append(f"Data leakage detected! The following features must be removed: {', '.join(leaky)}")
        for feat in leaky:
            if feat in reasons:
                errors.append(f"  - {feat}: {reasons[feat]}")
    
    if suspicious:
        warnings['suspicious'] = f"Warning: {len(suspicious)} suspicious features detected"
        for feat in suspicious[:3]:  # Show first 3
            if feat in reasons:
                warnings[f'suspicious_{feat}'] = reasons[feat]
    
    # Check for minimum number of features
    valid_features = [f for f in feature_columns if f not in leaky]
    if len(valid_features) < 3:
        errors.append(f"Too few features after removing leaky ones ({len(valid_features)}). Need at least 3 features.")
    
    is_valid = len(errors) == 0
    
    return is_valid, errors, warnings

