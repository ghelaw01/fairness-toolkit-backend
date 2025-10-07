"""
AI Fairness Metrics Module

This module provides comprehensive fairness metrics for evaluating AI systems
in public policy contexts. It includes various fairness definitions and 
measurement tools commonly used in algorithmic fairness research.

Author: Manus AI
License: MIT
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Union
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score
import warnings

class FairnessMetrics:
    """
    A comprehensive class for computing fairness metrics across different groups.
    
    This class implements various fairness definitions including:
    - Demographic Parity
    - Equalized Odds
    - Equal Opportunity
    - Calibration
    - Individual Fairness measures
    """
    
    def __init__(self, y_true: np.ndarray, y_pred: np.ndarray, 
                 sensitive_features: np.ndarray, y_prob: Optional[np.ndarray] = None):
        """
        Initialize the FairnessMetrics class.
        
        Args:
            y_true: True binary labels (0 or 1)
            y_pred: Predicted binary labels (0 or 1)
            sensitive_features: Protected attribute values (e.g., race, gender)
            y_prob: Predicted probabilities (optional, for calibration metrics)
        """
        self.y_true = np.array(y_true)
        self.y_pred = np.array(y_pred)
        self.sensitive_features = np.array(sensitive_features)
        self.y_prob = np.array(y_prob) if y_prob is not None else None
        
        # Validate inputs
        self._validate_inputs()
        
        # Get unique groups
        self.groups = np.unique(self.sensitive_features)
        
    def _validate_inputs(self):
        """Validate input arrays for consistency."""
        if len(self.y_true) != len(self.y_pred):
            raise ValueError("y_true and y_pred must have the same length")
        
        if len(self.y_true) != len(self.sensitive_features):
            raise ValueError("y_true and sensitive_features must have the same length")
        
        if self.y_prob is not None and len(self.y_true) != len(self.y_prob):
            raise ValueError("y_true and y_prob must have the same length")
        
        # Check if labels are binary
        unique_true = np.unique(self.y_true)
        unique_pred = np.unique(self.y_pred)
        
        if not all(label in [0, 1] for label in unique_true):
            raise ValueError("y_true must contain only binary values (0, 1)")
        
        if not all(label in [0, 1] for label in unique_pred):
            raise ValueError("y_pred must contain only binary values (0, 1)")
    
    def demographic_parity(self) -> Dict[str, float]:
        """
        Calculate demographic parity (statistical parity) across groups.
        
        Demographic parity is satisfied when the probability of positive 
        prediction is the same across all groups.
        
        Returns:
            Dict containing positive prediction rates for each group and overall disparity
        """
        results = {}
        positive_rates = {}
        
        for group in self.groups:
            mask = self.sensitive_features == group
            group_pred = self.y_pred[mask]
            positive_rate = np.mean(group_pred)
            positive_rates[f"group_{group}"] = positive_rate
            results[f"positive_rate_group_{group}"] = positive_rate
        
        # Calculate disparity (difference between max and min rates)
        rates = list(positive_rates.values())
        results["demographic_parity_difference"] = max(rates) - min(rates)
        results["demographic_parity_ratio"] = min(rates) / max(rates) if max(rates) > 0 else 0
        
        return results
    
    def equalized_odds(self) -> Dict[str, float]:
        """
        Calculate equalized odds across groups.
        
        Equalized odds is satisfied when true positive rate and false positive rate
        are the same across all groups.
        
        Returns:
            Dict containing TPR, FPR for each group and disparity measures
        """
        results = {}
        tpr_by_group = {}
        fpr_by_group = {}
        
        for group in self.groups:
            mask = self.sensitive_features == group
            y_true_group = self.y_true[mask]
            y_pred_group = self.y_pred[mask]
            
            # Calculate confusion matrix
            tn, fp, fn, tp = confusion_matrix(y_true_group, y_pred_group).ravel()
            
            # True Positive Rate (Sensitivity/Recall)
            tpr = tp / (tp + fn) if (tp + fn) > 0 else 0
            # False Positive Rate
            fpr = fp / (fp + tn) if (fp + tn) > 0 else 0
            
            tpr_by_group[f"group_{group}"] = tpr
            fpr_by_group[f"group_{group}"] = fpr
            
            results[f"tpr_group_{group}"] = tpr
            results[f"fpr_group_{group}"] = fpr
        
        # Calculate disparities
        tpr_values = list(tpr_by_group.values())
        fpr_values = list(fpr_by_group.values())
        
        results["tpr_difference"] = max(tpr_values) - min(tpr_values)
        results["fpr_difference"] = max(fpr_values) - min(fpr_values)
        results["equalized_odds_difference"] = max(results["tpr_difference"], results["fpr_difference"])
        
        return results
    
    def equal_opportunity(self) -> Dict[str, float]:
        """
        Calculate equal opportunity across groups.
        
        Equal opportunity is satisfied when true positive rate is the same
        across all groups (focuses only on positive class).
        
        Returns:
            Dict containing TPR for each group and disparity measures
        """
        results = {}
        tpr_by_group = {}
        
        for group in self.groups:
            mask = self.sensitive_features == group
            y_true_group = self.y_true[mask]
            y_pred_group = self.y_pred[mask]
            
            # Calculate True Positive Rate
            tp = np.sum((y_true_group == 1) & (y_pred_group == 1))
            fn = np.sum((y_true_group == 1) & (y_pred_group == 0))
            
            tpr = tp / (tp + fn) if (tp + fn) > 0 else 0
            tpr_by_group[f"group_{group}"] = tpr
            results[f"tpr_group_{group}"] = tpr
        
        # Calculate disparity
        tpr_values = list(tpr_by_group.values())
        results["equal_opportunity_difference"] = max(tpr_values) - min(tpr_values)
        results["equal_opportunity_ratio"] = min(tpr_values) / max(tpr_values) if max(tpr_values) > 0 else 0
        
        return results
    
    def calibration_metrics(self) -> Dict[str, float]:
        """
        Calculate calibration metrics across groups.
        
        Calibration measures whether predicted probabilities match actual outcomes.
        Requires y_prob to be provided.
        
        Returns:
            Dict containing calibration metrics for each group
        """
        if self.y_prob is None:
            raise ValueError("y_prob must be provided for calibration metrics")
        
        results = {}
        
        for group in self.groups:
            mask = self.sensitive_features == group
            y_true_group = self.y_true[mask]
            y_prob_group = self.y_prob[mask]
            
            # Bin probabilities and calculate calibration
            n_bins = 10
            bin_boundaries = np.linspace(0, 1, n_bins + 1)
            bin_lowers = bin_boundaries[:-1]
            bin_uppers = bin_boundaries[1:]
            
            calibration_error = 0
            total_samples = 0
            
            for bin_lower, bin_upper in zip(bin_lowers, bin_uppers):
                in_bin = (y_prob_group > bin_lower) & (y_prob_group <= bin_upper)
                prop_in_bin = in_bin.mean()
                
                if prop_in_bin > 0:
                    accuracy_in_bin = y_true_group[in_bin].mean()
                    avg_confidence_in_bin = y_prob_group[in_bin].mean()
                    
                    calibration_error += np.abs(avg_confidence_in_bin - accuracy_in_bin) * prop_in_bin
                    total_samples += np.sum(in_bin)
            
            results[f"calibration_error_group_{group}"] = calibration_error
            results[f"samples_group_{group}"] = total_samples
        
        # Calculate overall calibration disparity
        calibration_errors = [v for k, v in results.items() if "calibration_error" in k]
        results["calibration_disparity"] = max(calibration_errors) - min(calibration_errors)
        
        return results
    
    def predictive_parity(self) -> Dict[str, float]:
        """
        Calculate predictive parity (precision parity) across groups.
        
        Predictive parity is satisfied when precision is the same across all groups.
        
        Returns:
            Dict containing precision for each group and disparity measures
        """
        results = {}
        precision_by_group = {}
        
        for group in self.groups:
            mask = self.sensitive_features == group
            y_true_group = self.y_true[mask]
            y_pred_group = self.y_pred[mask]
            
            # Calculate precision
            precision = precision_score(y_true_group, y_pred_group, zero_division=0)
            precision_by_group[f"group_{group}"] = precision
            results[f"precision_group_{group}"] = precision
        
        # Calculate disparity
        precision_values = list(precision_by_group.values())
        results["predictive_parity_difference"] = max(precision_values) - min(precision_values)
        results["predictive_parity_ratio"] = min(precision_values) / max(precision_values) if max(precision_values) > 0 else 0
        
        return results
    
    def overall_accuracy_equality(self) -> Dict[str, float]:
        """
        Calculate overall accuracy equality across groups.
        
        Returns:
            Dict containing accuracy for each group and disparity measures
        """
        results = {}
        accuracy_by_group = {}
        
        for group in self.groups:
            mask = self.sensitive_features == group
            y_true_group = self.y_true[mask]
            y_pred_group = self.y_pred[mask]
            
            accuracy = accuracy_score(y_true_group, y_pred_group)
            accuracy_by_group[f"group_{group}"] = accuracy
            results[f"accuracy_group_{group}"] = accuracy
        
        # Calculate disparity
        accuracy_values = list(accuracy_by_group.values())
        results["accuracy_difference"] = max(accuracy_values) - min(accuracy_values)
        results["accuracy_ratio"] = min(accuracy_values) / max(accuracy_values) if max(accuracy_values) > 0 else 0
        
        return results
    
    def treatment_equality(self) -> Dict[str, float]:
        """
        Calculate treatment equality across groups.
        
        Treatment equality is satisfied when the ratio of false negatives to 
        false positives is the same across groups.
        
        Returns:
            Dict containing FN/FP ratios for each group and disparity measures
        """
        results = {}
        fn_fp_ratios = {}
        
        for group in self.groups:
            mask = self.sensitive_features == group
            y_true_group = self.y_true[mask]
            y_pred_group = self.y_pred[mask]
            
            # Calculate confusion matrix
            tn, fp, fn, tp = confusion_matrix(y_true_group, y_pred_group).ravel()
            
            # Calculate FN/FP ratio
            fn_fp_ratio = fn / fp if fp > 0 else float('inf') if fn > 0 else 0
            fn_fp_ratios[f"group_{group}"] = fn_fp_ratio
            results[f"fn_fp_ratio_group_{group}"] = fn_fp_ratio
            results[f"fn_group_{group}"] = fn
            results[f"fp_group_{group}"] = fp
        
        # Calculate disparity (excluding infinite values)
        finite_ratios = [r for r in fn_fp_ratios.values() if np.isfinite(r)]
        if finite_ratios:
            results["treatment_equality_difference"] = max(finite_ratios) - min(finite_ratios)
        else:
            results["treatment_equality_difference"] = 0
        
        return results
    
    def compute_all_metrics(self) -> Dict[str, Dict[str, float]]:
        """
        Compute all available fairness metrics.
        
        Returns:
            Dict containing all fairness metrics organized by category
        """
        all_metrics = {}
        
        try:
            all_metrics["demographic_parity"] = self.demographic_parity()
        except Exception as e:
            warnings.warn(f"Could not compute demographic parity: {e}")
            all_metrics["demographic_parity"] = {}
        
        try:
            all_metrics["equalized_odds"] = self.equalized_odds()
        except Exception as e:
            warnings.warn(f"Could not compute equalized odds: {e}")
            all_metrics["equalized_odds"] = {}
        
        try:
            all_metrics["equal_opportunity"] = self.equal_opportunity()
        except Exception as e:
            warnings.warn(f"Could not compute equal opportunity: {e}")
            all_metrics["equal_opportunity"] = {}
        
        try:
            all_metrics["predictive_parity"] = self.predictive_parity()
        except Exception as e:
            warnings.warn(f"Could not compute predictive parity: {e}")
            all_metrics["predictive_parity"] = {}
        
        try:
            all_metrics["overall_accuracy"] = self.overall_accuracy_equality()
        except Exception as e:
            warnings.warn(f"Could not compute overall accuracy: {e}")
            all_metrics["overall_accuracy"] = {}
        
        try:
            all_metrics["treatment_equality"] = self.treatment_equality()
        except Exception as e:
            warnings.warn(f"Could not compute treatment equality: {e}")
            all_metrics["treatment_equality"] = {}
        
        if self.y_prob is not None:
            try:
                all_metrics["calibration"] = self.calibration_metrics()
            except Exception as e:
                warnings.warn(f"Could not compute calibration metrics: {e}")
                all_metrics["calibration"] = {}
        
        return all_metrics
    
    def fairness_summary(self) -> pd.DataFrame:
        """
        Generate a summary DataFrame of all fairness metrics.
        
        Returns:
            DataFrame with fairness metrics summary
        """
        all_metrics = self.compute_all_metrics()
        
        summary_data = []
        for metric_category, metrics in all_metrics.items():
            for metric_name, value in metrics.items():
                if "difference" in metric_name or "ratio" in metric_name or "disparity" in metric_name:
                    summary_data.append({
                        "Metric Category": metric_category,
                        "Metric Name": metric_name,
                        "Value": value,
                        "Interpretation": self._interpret_metric(metric_name, value)
                    })
        
        return pd.DataFrame(summary_data)
    
    def _interpret_metric(self, metric_name: str, value: float) -> str:
        """
        Provide interpretation for fairness metric values.
        
        Args:
            metric_name: Name of the metric
            value: Metric value
            
        Returns:
            String interpretation of the metric
        """
        if "difference" in metric_name or "disparity" in metric_name:
            if abs(value) < 0.05:
                return "Very Fair (< 5% difference)"
            elif abs(value) < 0.1:
                return "Moderately Fair (5-10% difference)"
            elif abs(value) < 0.2:
                return "Some Bias (10-20% difference)"
            else:
                return "Significant Bias (> 20% difference)"
        
        elif "ratio" in metric_name:
            if value >= 0.9:
                return "Very Fair (ratio ≥ 0.9)"
            elif value >= 0.8:
                return "Moderately Fair (ratio ≥ 0.8)"
            elif value >= 0.7:
                return "Some Bias (ratio ≥ 0.7)"
            else:
                return "Significant Bias (ratio < 0.7)"
        
        else:
            return f"Value: {value:.3f}"


def compute_fairness_metrics(y_true: np.ndarray, y_pred: np.ndarray, 
                           sensitive_features: np.ndarray, 
                           y_prob: Optional[np.ndarray] = None) -> Dict[str, Dict[str, float]]:
    """
    Convenience function to compute all fairness metrics.
    
    Args:
        y_true: True binary labels
        y_pred: Predicted binary labels  
        sensitive_features: Protected attribute values
        y_prob: Predicted probabilities (optional)
        
    Returns:
        Dict containing all fairness metrics
    """
    fm = FairnessMetrics(y_true, y_pred, sensitive_features, y_prob)
    return fm.compute_all_metrics()


def fairness_report(y_true: np.ndarray, y_pred: np.ndarray, 
                   sensitive_features: np.ndarray, 
                   y_prob: Optional[np.ndarray] = None) -> pd.DataFrame:
    """
    Generate a comprehensive fairness report.
    
    Args:
        y_true: True binary labels
        y_pred: Predicted binary labels
        sensitive_features: Protected attribute values
        y_prob: Predicted probabilities (optional)
        
    Returns:
        DataFrame with fairness metrics summary
    """
    fm = FairnessMetrics(y_true, y_pred, sensitive_features, y_prob)
    return fm.fairness_summary()

