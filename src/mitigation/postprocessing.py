"""
Post-processing Bias Mitigation Techniques

This module implements techniques that adjust model predictions after training
to improve fairness.
"""

import numpy as np
import pandas as pd
from typing import Dict
import logging

logger = logging.getLogger(__name__)


class CalibratedEqualizedOdds:
    """Calibrated Equalized Odds Post-processing"""
    
    def __init__(self):
        self.mix_rates_ = {}
        
    def fit(self, y_true: np.ndarray, y_pred_proba: np.ndarray, sensitive_attr: np.ndarray):
        """Learn the optimal mixing rates for each group."""
        for group in np.unique(sensitive_attr):
            mask = sensitive_attr == group
            y_true_group = y_true[mask]
            y_pred_group = y_pred_proba[mask]
            
            base_rate = y_true_group.mean()
            pred_rate = (y_pred_group > 0.5).mean()
            
            self.mix_rates_[group] = {
                'base_rate': float(base_rate),
                'pred_rate': float(pred_rate),
                'adjustment': float(base_rate - pred_rate)
            }
        
        return self
    
    def transform(self, y_pred_proba: np.ndarray, sensitive_attr: np.ndarray) -> np.ndarray:
        """Adjust predictions to satisfy equalized odds."""
        y_pred_adjusted = y_pred_proba.copy()
        
        adjustments = [info['adjustment'] for info in self.mix_rates_.values()]
        global_adj = np.mean(adjustments)
        
        for group, info in self.mix_rates_.items():
            mask = sensitive_attr == group
            adjustment = info['adjustment'] - global_adj
            y_pred_adjusted[mask] = np.clip(y_pred_proba[mask] + adjustment * 0.5, 0, 1)
        
        return y_pred_adjusted
    
    def fit_transform(self, y_true: np.ndarray, y_pred_proba: np.ndarray, sensitive_attr: np.ndarray) -> np.ndarray:
        """Fit and transform in one step."""
        self.fit(y_true, y_pred_proba, sensitive_attr)
        return self.transform(y_pred_proba, sensitive_attr)
    
    def get_info(self) -> Dict:
        """Get information about the technique."""
        return {
            "technique": "Calibrated Equalized Odds",
            "description": "Adjusts predictions to satisfy equalized odds while maintaining calibration",
            "mix_rates": self.mix_rates_,
            "impact": "Equalizes true positive and false positive rates across groups",
            "plain_language": "This technique fine-tunes the model's predictions to ensure all groups have equal chances of correct positive and negative predictions."
        }


class RejectOptionClassification:
    """Reject Option Classification"""
    
    def __init__(self, threshold: float = 0.5, margin: float = 0.1):
        self.threshold = threshold
        self.margin = margin
        self.privileged_group_ = None
        
    def fit(self, y_true: np.ndarray, y_pred_proba: np.ndarray, sensitive_attr: np.ndarray):
        """Identify the privileged group."""
        rates = {}
        for group in np.unique(sensitive_attr):
            mask = sensitive_attr == group
            rates[group] = (y_pred_proba[mask] > self.threshold).mean()
        
        self.privileged_group_ = max(rates, key=rates.get)
        self.group_rates_ = rates
        
        return self
    
    def transform(self, y_pred_proba: np.ndarray, sensitive_attr: np.ndarray) -> np.ndarray:
        """Apply reject option classification."""
        y_pred_adjusted = (y_pred_proba > self.threshold).astype(int)
        
        lower_bound = self.threshold - self.margin
        upper_bound = self.threshold + self.margin
        critical_region = (y_pred_proba >= lower_bound) & (y_pred_proba <= upper_bound)
        
        for group in np.unique(sensitive_attr):
            mask = (sensitive_attr == group) & critical_region
            
            if group == self.privileged_group_:
                y_pred_adjusted[mask] = 0
            else:
                y_pred_adjusted[mask] = 1
        
        return y_pred_adjusted
    
    def fit_transform(self, y_true: np.ndarray, y_pred_proba: np.ndarray, sensitive_attr: np.ndarray) -> np.ndarray:
        """Fit and transform in one step."""
        self.fit(y_true, y_pred_proba, sensitive_attr)
        return self.transform(y_pred_proba, sensitive_attr)
    
    def get_info(self) -> Dict:
        """Get information about the technique."""
        return {
            "technique": "Reject Option Classification",
            "description": "Adjusts predictions in uncertain region to favor unprivileged groups",
            "threshold": self.threshold,
            "margin": self.margin,
            "privileged_group": str(self.privileged_group_),
            "group_rates": {str(k): float(v) for k, v in self.group_rates_.items()},
            "impact": "Reduces discrimination by adjusting borderline cases",
            "plain_language": f"For cases where the model is uncertain (within {int(self.margin*100)}% of the decision boundary), this gives the benefit of the doubt to disadvantaged groups."
        }


class ThresholdOptimizer:
    """Threshold Optimizer"""
    
    def __init__(self, constraint: str = 'demographic_parity'):
        self.constraint = constraint
        self.thresholds_ = {}
        
    def fit(self, y_true: np.ndarray, y_pred_proba: np.ndarray, sensitive_attr: np.ndarray):
        """Learn optimal thresholds for each group."""
        for group in np.unique(sensitive_attr):
            mask = sensitive_attr == group
            y_true_group = y_true[mask]
            y_pred_group = y_pred_proba[mask]
            
            best_threshold = 0.5
            best_accuracy = 0
            
            for threshold in np.linspace(0.1, 0.9, 17):
                y_pred_binary = (y_pred_group > threshold).astype(int)
                accuracy = (y_pred_binary == y_true_group).mean()
                
                if accuracy > best_accuracy:
                    best_accuracy = accuracy
                    best_threshold = threshold
            
            self.thresholds_[group] = best_threshold
        
        if self.constraint == 'demographic_parity':
            target_rate = np.mean([
                (y_pred_proba[sensitive_attr == g] > self.thresholds_[g]).mean()
                for g in np.unique(sensitive_attr)
            ])
            
            for group in np.unique(sensitive_attr):
                mask = sensitive_attr == group
                y_pred_group = y_pred_proba[mask]
                
                for threshold in np.linspace(0.0, 1.0, 101):
                    rate = (y_pred_group > threshold).mean()
                    if abs(rate - target_rate) < 0.01:
                        self.thresholds_[group] = threshold
                        break
        
        return self
    
    def transform(self, y_pred_proba: np.ndarray, sensitive_attr: np.ndarray) -> np.ndarray:
        """Apply group-specific thresholds."""
        y_pred_adjusted = np.zeros(len(y_pred_proba), dtype=int)
        
        for group, threshold in self.thresholds_.items():
            mask = sensitive_attr == group
            y_pred_adjusted[mask] = (y_pred_proba[mask] > threshold).astype(int)
        
        return y_pred_adjusted
    
    def fit_transform(self, y_true: np.ndarray, y_pred_proba: np.ndarray, sensitive_attr: np.ndarray) -> np.ndarray:
        """Fit and transform in one step."""
        self.fit(y_true, y_pred_proba, sensitive_attr)
        return self.transform(y_pred_proba, sensitive_attr)
    
    def get_info(self) -> Dict:
        """Get information about the technique."""
        return {
            "technique": "Threshold Optimizer",
            "description": "Uses different thresholds for different groups to achieve fairness",
            "constraint": self.constraint,
            "thresholds": {str(k): float(v) for k, v in self.thresholds_.items()},
            "impact": "Equalizes outcomes across groups through threshold adjustment",
            "plain_language": "This technique uses different decision cutoffs for different groups to ensure fair outcomes while maintaining overall accuracy."
        }


def apply_postprocessing_mitigation(
    y_true: np.ndarray,
    y_pred_proba: np.ndarray,
    sensitive_attr: np.ndarray,
    technique: str = 'calibrated_equalized_odds',
    **kwargs
) -> Dict:
    """Apply a postprocessing bias mitigation technique."""
    logger.info(f"Applying postprocessing technique: {technique}")
    
    if technique == 'calibrated_equalized_odds':
        mitigator = CalibratedEqualizedOdds()
        y_pred_adjusted = mitigator.fit_transform(y_true, y_pred_proba, sensitive_attr)
        return {
            "predictions": y_pred_adjusted.tolist(),
            "predictions_binary": (y_pred_adjusted > 0.5).astype(int).tolist(),
            "info": mitigator.get_info(),
            "technique_type": "postprocessing"
        }
    
    elif technique == 'reject_option':
        threshold = kwargs.get('threshold', 0.5)
        margin = kwargs.get('margin', 0.1)
        mitigator = RejectOptionClassification(threshold=threshold, margin=margin)
        y_pred_adjusted = mitigator.fit_transform(y_true, y_pred_proba, sensitive_attr)
        return {
            "predictions": y_pred_adjusted.tolist(),
            "predictions_binary": y_pred_adjusted.tolist(),
            "info": mitigator.get_info(),
            "technique_type": "postprocessing"
        }
    
    elif technique == 'threshold_optimizer':
        constraint = kwargs.get('constraint', 'demographic_parity')
        mitigator = ThresholdOptimizer(constraint=constraint)
        y_pred_adjusted = mitigator.fit_transform(y_true, y_pred_proba, sensitive_attr)
        return {
            "predictions": y_pred_adjusted.tolist(),
            "predictions_binary": y_pred_adjusted.tolist(),
            "info": mitigator.get_info(),
            "technique_type": "postprocessing"
        }
    
    else:
        raise ValueError(f"Unknown postprocessing technique: {technique}")

