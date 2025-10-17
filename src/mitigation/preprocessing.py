"""
Pre-processing Bias Mitigation Techniques

This module implements techniques that transform the dataset before model training
to reduce bias and improve fairness.
"""

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from typing import Dict, Tuple
import logging

logger = logging.getLogger(__name__)


class Reweighing:
    """Reweighing technique that adjusts instance weights to ensure fairness."""
    
    def __init__(self):
        self.weights_ = None
        self.group_weights_ = {}
        
    def fit(self, X: pd.DataFrame, y: pd.Series, sensitive_attr: str):
        """Calculate reweighing factors for each group."""
        sensitive_values = X[sensitive_attr]
        n = len(X)
        weights = np.ones(n)
        
        for s_val in sensitive_values.unique():
            for y_val in y.unique():
                mask = (sensitive_values == s_val) & (y == y_val)
                p_s = (sensitive_values == s_val).sum() / n
                p_y = (y == y_val).sum() / n
                expected = p_s * p_y
                observed = mask.sum() / n
                
                if observed > 0:
                    weight = expected / observed
                    weights[mask] = weight
                    self.group_weights_[(s_val, y_val)] = weight
        
        self.weights_ = weights
        return self
    
    def transform(self, X: pd.DataFrame, y: pd.Series, sensitive_attr: str):
        """Return the dataset with calculated weights."""
        if self.weights_ is None:
            raise ValueError("Must call fit() before transform()")
        return X, y, self.weights_
    
    def fit_transform(self, X: pd.DataFrame, y: pd.Series, sensitive_attr: str):
        """Fit and transform in one step."""
        self.fit(X, y, sensitive_attr)
        return self.transform(X, y, sensitive_attr)
    
    def get_info(self) -> Dict:
        """Get information about the reweighing."""
        return {
            "technique": "Reweighing",
            "description": "Adjusts instance weights to achieve statistical parity",
            "group_weights": {str(k): float(v) for k, v in self.group_weights_.items()},
            "impact": "Balances representation of different groups in the dataset",
            "plain_language": "This technique gives more importance to underrepresented groups in your data, helping the model treat all groups more fairly."
        }


class DisparateImpactRemover:
    """Disparate Impact Remover that transforms features to remove discrimination."""
    
    def __init__(self, repair_level: float = 1.0):
        self.repair_level = repair_level
        self.feature_medians_ = {}
        
    def fit(self, X: pd.DataFrame, sensitive_attr: str):
        """Learn the median values for each group."""
        self.sensitive_attr_ = sensitive_attr
        
        for col in X.columns:
            if col != sensitive_attr:
                self.feature_medians_[col] = {}
                for group in X[sensitive_attr].unique():
                    mask = X[sensitive_attr] == group
                    self.feature_medians_[col][group] = X.loc[mask, col].median()
        
        return self
    
    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Transform features to remove disparate impact."""
        X_transformed = X.copy()
        sensitive_attr = self.sensitive_attr_
        
        for col in X.columns:
            if col != sensitive_attr and col in self.feature_medians_:
                global_median = X[col].median()
                
                for group in X[sensitive_attr].unique():
                    mask = X[sensitive_attr] == group
                    group_median = self.feature_medians_[col][group]
                    repair_amount = self.repair_level * (global_median - group_median)
                    X_transformed.loc[mask, col] = X.loc[mask, col] + repair_amount
        
        return X_transformed
    
    def fit_transform(self, X: pd.DataFrame, sensitive_attr: str) -> pd.DataFrame:
        """Fit and transform in one step."""
        self.fit(X, sensitive_attr)
        return self.transform(X)
    
    def get_info(self) -> Dict:
        """Get information about the technique."""
        return {
            "technique": "Disparate Impact Remover",
            "description": "Transforms features to remove discrimination while preserving rank-ordering",
            "repair_level": self.repair_level,
            "impact": "Reduces correlation between features and protected attributes",
            "plain_language": f"This technique adjusts your data features to reduce bias by {int(self.repair_level*100)}%, while keeping the relative ordering of individuals intact."
        }


def apply_preprocessing_mitigation(
    X: pd.DataFrame,
    y: pd.Series,
    sensitive_attr: str,
    technique: str = 'reweighing',
    **kwargs
) -> Dict:
    """Apply a preprocessing bias mitigation technique."""
    logger.info(f"Applying preprocessing technique: {technique}")
    
    if technique == 'reweighing':
        mitigator = Reweighing()
        X_new, y_new, weights = mitigator.fit_transform(X, y, sensitive_attr)
        return {
            "X": X_new,
            "y": y_new,
            "weights": weights.tolist(),
            "info": mitigator.get_info(),
            "technique_type": "preprocessing"
        }
    
    elif technique == 'disparate_impact_removal':
        repair_level = kwargs.get('repair_level', 1.0)
        mitigator = DisparateImpactRemover(repair_level=repair_level)
        X_new = mitigator.fit_transform(X, sensitive_attr)
        return {
            "X": X_new,
            "y": y,
            "weights": None,
            "info": mitigator.get_info(),
            "technique_type": "preprocessing"
        }
    
    else:
        raise ValueError(f"Unknown preprocessing technique: {technique}")
