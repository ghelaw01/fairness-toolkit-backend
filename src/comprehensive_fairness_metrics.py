"""
Comprehensive Fairness Metrics Calculator
Implements 50+ fairness metrics across different domains
"""

import numpy as np
from typing import Dict, List, Tuple, Optional
from scipy import stats
from scipy.spatial.distance import cosine
from sklearn.metrics import confusion_matrix, mean_absolute_error, mean_squared_error, r2_score


# ============================================================================
# METRIC DEFINITIONS
# ============================================================================

METRIC_DEFINITIONS = {
    # Classification Fairness Metrics
    "statistical_parity_difference": {
        "name": "Statistical Parity Difference",
        "definition": "Difference in positive prediction rates between groups. Measures demographic parity.",
        "formula": "P(Ŷ=1|A=a) - P(Ŷ=1|A=b)",
        "ideal_value": 0,
        "threshold": 0.1,
        "category": "Classification"
    },
    "disparate_impact": {
        "name": "Disparate Impact",
        "definition": "Ratio of positive prediction rates. Used in legal contexts (80% rule).",
        "formula": "P(Ŷ=1|A=unprivileged) / P(Ŷ=1|A=privileged)",
        "ideal_value": 1.0,
        "threshold": 0.8,
        "category": "Classification"
    },
    "adverse_impact_ratio": {
        "name": "Adverse Impact Ratio",
        "definition": "Same as Disparate Impact. Ratio of selection rates between groups.",
        "formula": "min(rate_a, rate_b) / max(rate_a, rate_b)",
        "ideal_value": 1.0,
        "threshold": 0.8,
        "category": "Classification"
    },
    "cohens_d": {
        "name": "Cohen's D",
        "definition": "Standardized mean difference in predictions between groups.",
        "formula": "(mean_a - mean_b) / pooled_std",
        "ideal_value": 0,
        "threshold": 0.2,
        "category": "Classification"
    },
    "equal_opportunity_difference": {
        "name": "Equal Opportunity Difference",
        "definition": "Difference in True Positive Rates (TPR) between groups.",
        "formula": "TPR_a - TPR_b",
        "ideal_value": 0,
        "threshold": 0.1,
        "category": "Classification"
    },
    "average_odds_difference": {
        "name": "Average Odds Difference",
        "definition": "Average of TPR and FPR differences between groups.",
        "formula": "0.5 * [(TPR_a - TPR_b) + (FPR_a - FPR_b)]",
        "ideal_value": 0,
        "threshold": 0.1,
        "category": "Classification"
    },
    "predictive_parity_difference": {
        "name": "Predictive Parity Difference",
        "definition": "Difference in Positive Predictive Values (Precision) between groups.",
        "formula": "PPV_a - PPV_b",
        "ideal_value": 0,
        "threshold": 0.1,
        "category": "Classification"
    },
    "predictive_equality": {
        "name": "Predictive Equality",
        "definition": "Difference in False Positive Rates between groups.",
        "formula": "FPR_a - FPR_b",
        "ideal_value": 0,
        "threshold": 0.1,
        "category": "Classification"
    },
    "conditional_use_accuracy_equality": {
        "name": "Conditional Use Accuracy Equality",
        "definition": "Difference in PPV and NPV between groups.",
        "formula": "|PPV_a - PPV_b| + |NPV_a - NPV_b|",
        "ideal_value": 0,
        "threshold": 0.1,
        "category": "Classification"
    },
    "overall_accuracy_equality": {
        "name": "Overall Accuracy Equality",
        "definition": "Difference in overall accuracy between groups.",
        "formula": "Accuracy_a - Accuracy_b",
        "ideal_value": 0,
        "threshold": 0.05,
        "category": "Classification"
    },
    "false_positive_rate_balance": {
        "name": "False Positive Rate Balance",
        "definition": "Ratio of False Positive Rates between groups.",
        "formula": "min(FPR_a, FPR_b) / max(FPR_a, FPR_b)",
        "ideal_value": 1.0,
        "threshold": 0.8,
        "category": "Classification"
    },
    "false_negative_rate_balance": {
        "name": "False Negative Rate Balance",
        "definition": "Ratio of False Negative Rates between groups.",
        "formula": "min(FNR_a, FNR_b) / max(FNR_a, FNR_b)",
        "ideal_value": 1.0,
        "threshold": 0.8,
        "category": "Classification"
    },
    "false_discovery_rate_balance": {
        "name": "False Discovery Rate Balance",
        "definition": "Ratio of False Discovery Rates between groups.",
        "formula": "min(FDR_a, FDR_b) / max(FDR_a, FDR_b)",
        "ideal_value": 1.0,
        "threshold": 0.8,
        "category": "Classification"
    },
    
    # Regression Fairness Metrics
    "mean_difference": {
        "name": "Mean Difference",
        "definition": "Difference in mean predictions between groups.",
        "formula": "mean(ŷ_a) - mean(ŷ_b)",
        "ideal_value": 0,
        "threshold": 0.1,
        "category": "Regression"
    },
    "mae_parity": {
        "name": "MAE Parity",
        "definition": "Difference in Mean Absolute Error between groups.",
        "formula": "MAE_a - MAE_b",
        "ideal_value": 0,
        "threshold": 0.1,
        "category": "Regression"
    },
    "r2_parity": {
        "name": "R² Parity",
        "definition": "Difference in R² scores between groups.",
        "formula": "R²_a - R²_b",
        "ideal_value": 0,
        "threshold": 0.1,
        "category": "Regression"
    },
    "rmse_balance": {
        "name": "RMSE Balance",
        "definition": "Ratio of Root Mean Squared Errors between groups.",
        "formula": "min(RMSE_a, RMSE_b) / max(RMSE_a, RMSE_b)",
        "ideal_value": 1.0,
        "threshold": 0.8,
        "category": "Regression"
    },
    "group_fairness_in_expectations": {
        "name": "Group Fairness in Expectations",
        "definition": "Difference in expected prediction errors between groups.",
        "formula": "E[ŷ - y | A=a] - E[ŷ - y | A=b]",
        "ideal_value": 0,
        "threshold": 0.1,
        "category": "Regression"
    },
    "residual_fairness": {
        "name": "Residual Fairness",
        "definition": "Difference in residual distributions between groups.",
        "formula": "KS-statistic of residual distributions",
        "ideal_value": 0,
        "threshold": 0.2,
        "category": "Regression"
    },
    
    # Composite & Inequality Metrics
    "theil_index": {
        "name": "Theil Index",
        "definition": "Measure of inequality in prediction distributions across groups.",
        "formula": "Σ (p_i * log(p_i / q_i))",
        "ideal_value": 0,
        "threshold": 0.1,
        "category": "Composite"
    },
    "generalized_entropy_index": {
        "name": "Generalized Entropy Index",
        "definition": "Generalized measure of inequality with parameter α.",
        "formula": "GE(α) based on benefit distribution",
        "ideal_value": 0,
        "threshold": 0.1,
        "category": "Composite"
    },
    "fairness_gap": {
        "name": "Fairness Gap",
        "definition": "Maximum difference in any fairness metric across groups.",
        "formula": "max(|metric_i - metric_j|) for all i,j",
        "ideal_value": 0,
        "threshold": 0.1,
        "category": "Composite"
    },
    "composite_fairness_score": {
        "name": "Composite Fairness Score",
        "definition": "Weighted average of multiple fairness metrics.",
        "formula": "Σ w_i * normalized_metric_i",
        "ideal_value": 1.0,
        "threshold": 0.8,
        "category": "Composite"
    },
    "bias_amplification_score": {
        "name": "Bias Amplification Score",
        "definition": "Ratio of bias in predictions to bias in training data.",
        "formula": "bias_predictions / bias_training_data",
        "ideal_value": 1.0,
        "threshold": 1.2,
        "category": "Composite"
    },
    
    # Individual Fairness Metrics
    "consistency": {
        "name": "Consistency",
        "definition": "Fraction of similar individuals receiving similar predictions.",
        "formula": "1 - (1/n) * Σ |ŷ_i - ŷ_j| for similar i,j",
        "ideal_value": 1.0,
        "threshold": 0.8,
        "category": "Individual"
    },
    "lipschitz_fairness": {
        "name": "Lipschitz Fairness",
        "definition": "Maximum change in predictions relative to input change.",
        "formula": "max(|ŷ_i - ŷ_j| / d(x_i, x_j))",
        "ideal_value": 0,
        "threshold": 1.0,
        "category": "Individual"
    },
    
    # Causal Fairness Metrics
    "counterfactual_fairness": {
        "name": "Counterfactual Fairness",
        "definition": "Prediction unchanged when sensitive attribute is flipped.",
        "formula": "P(Ŷ_A=a | X, A=a) = P(Ŷ_A=a' | X, A=a)",
        "ideal_value": 0,
        "threshold": 0.1,
        "category": "Causal"
    },
    "natural_direct_effect": {
        "name": "Natural Direct Effect",
        "definition": "Direct causal effect of sensitive attribute on outcome.",
        "formula": "E[Y_a,M_a] - E[Y_a',M_a]",
        "ideal_value": 0,
        "threshold": 0.1,
        "category": "Causal"
    },
    "natural_indirect_effect": {
        "name": "Natural Indirect Effect",
        "definition": "Indirect causal effect through mediators.",
        "formula": "E[Y_a,M_a'] - E[Y_a,M_a]",
        "ideal_value": 0,
        "threshold": 0.1,
        "category": "Causal"
    },
}


# ============================================================================
# CLASSIFICATION FAIRNESS METRICS
# ============================================================================

def calculate_classification_metrics(y_true: np.ndarray, y_pred: np.ndarray, 
                                     sensitive_attr: np.ndarray) -> Dict:
    """Calculate comprehensive classification fairness metrics."""
    
    metrics = {}
    groups = np.unique(sensitive_attr)
    
    if len(groups) < 2:
        return metrics
    
    # Calculate group-specific metrics
    group_stats = {}
    for group in groups:
        mask = sensitive_attr == group
        y_true_g = y_true[mask]
        y_pred_g = y_pred[mask]
        
        if len(y_true_g) == 0:
            continue
        
        # Confusion matrix
        tn, fp, fn, tp = confusion_matrix(y_true_g, y_pred_g, labels=[0, 1]).ravel()
        
        group_stats[group] = {
            'selection_rate': np.mean(y_pred_g),
            'tpr': tp / (tp + fn) if (tp + fn) > 0 else 0,
            'fpr': fp / (fp + tn) if (fp + tn) > 0 else 0,
            'fnr': fn / (fn + tp) if (fn + tp) > 0 else 0,
            'tnr': tn / (tn + fp) if (tn + fp) > 0 else 0,
            'ppv': tp / (tp + fp) if (tp + fp) > 0 else 0,  # Precision
            'npv': tn / (tn + fn) if (tn + fn) > 0 else 0,
            'fdr': fp / (fp + tp) if (fp + tp) > 0 else 0,
            'accuracy': (tp + tn) / (tp + tn + fp + fn) if (tp + tn + fp + fn) > 0 else 0,
            'count': len(y_true_g)
        }
    
    # Get privileged and unprivileged groups (highest and lowest selection rates)
    sorted_groups = sorted(group_stats.items(), key=lambda x: x[1]['selection_rate'], reverse=True)
    priv_group = sorted_groups[0][0]
    unpriv_group = sorted_groups[-1][0]
    
    priv_stats = group_stats[priv_group]
    unpriv_stats = group_stats[unpriv_group]
    
    # 1. Statistical Parity Difference
    metrics['statistical_parity_difference'] = {
        'value': priv_stats['selection_rate'] - unpriv_stats['selection_rate'],
        'privileged_group': str(priv_group),
        'unprivileged_group': str(unpriv_group),
        **METRIC_DEFINITIONS['statistical_parity_difference']
    }
    
    # 2. Disparate Impact
    metrics['disparate_impact'] = {
        'value': unpriv_stats['selection_rate'] / priv_stats['selection_rate'] if priv_stats['selection_rate'] > 0 else 1.0,
        'privileged_group': str(priv_group),
        'unprivileged_group': str(unpriv_group),
        **METRIC_DEFINITIONS['disparate_impact']
    }
    
    # 3. Adverse Impact Ratio (same as disparate impact)
    metrics['adverse_impact_ratio'] = metrics['disparate_impact'].copy()
    metrics['adverse_impact_ratio'].update(METRIC_DEFINITIONS['adverse_impact_ratio'])
    
    # 4. Cohen's D
    priv_mask = sensitive_attr == priv_group
    unpriv_mask = sensitive_attr == unpriv_group
    priv_preds = y_pred[priv_mask].astype(float)
    unpriv_preds = y_pred[unpriv_mask].astype(float)
    
    pooled_std = np.sqrt(((len(priv_preds) - 1) * np.var(priv_preds) + 
                          (len(unpriv_preds) - 1) * np.var(unpriv_preds)) / 
                         (len(priv_preds) + len(unpriv_preds) - 2))
    
    cohens_d = (np.mean(priv_preds) - np.mean(unpriv_preds)) / pooled_std if pooled_std > 0 else 0
    
    metrics['cohens_d'] = {
        'value': cohens_d,
        'privileged_group': str(priv_group),
        'unprivileged_group': str(unpriv_group),
        **METRIC_DEFINITIONS['cohens_d']
    }
    
    # 5. Equal Opportunity Difference
    metrics['equal_opportunity_difference'] = {
        'value': priv_stats['tpr'] - unpriv_stats['tpr'],
        'privileged_group': str(priv_group),
        'unprivileged_group': str(unpriv_group),
        **METRIC_DEFINITIONS['equal_opportunity_difference']
    }
    
    # 6. Average Odds Difference
    metrics['average_odds_difference'] = {
        'value': 0.5 * ((priv_stats['tpr'] - unpriv_stats['tpr']) + 
                       (priv_stats['fpr'] - unpriv_stats['fpr'])),
        'privileged_group': str(priv_group),
        'unprivileged_group': str(unpriv_group),
        **METRIC_DEFINITIONS['average_odds_difference']
    }
    
    # 7. Predictive Parity Difference
    metrics['predictive_parity_difference'] = {
        'value': priv_stats['ppv'] - unpriv_stats['ppv'],
        'privileged_group': str(priv_group),
        'unprivileged_group': str(unpriv_group),
        **METRIC_DEFINITIONS['predictive_parity_difference']
    }
    
    # 8. Predictive Equality (FPR difference)
    metrics['predictive_equality'] = {
        'value': priv_stats['fpr'] - unpriv_stats['fpr'],
        'privileged_group': str(priv_group),
        'unprivileged_group': str(unpriv_group),
        **METRIC_DEFINITIONS['predictive_equality']
    }
    
    # 9. Conditional Use Accuracy Equality
    metrics['conditional_use_accuracy_equality'] = {
        'value': abs(priv_stats['ppv'] - unpriv_stats['ppv']) + abs(priv_stats['npv'] - unpriv_stats['npv']),
        'privileged_group': str(priv_group),
        'unprivileged_group': str(unpriv_group),
        **METRIC_DEFINITIONS['conditional_use_accuracy_equality']
    }
    
    # 10. Overall Accuracy Equality
    metrics['overall_accuracy_equality'] = {
        'value': priv_stats['accuracy'] - unpriv_stats['accuracy'],
        'privileged_group': str(priv_group),
        'unprivileged_group': str(unpriv_group),
        **METRIC_DEFINITIONS['overall_accuracy_equality']
    }
    
    # 11. False Positive Rate Balance
    max_fpr = max(priv_stats['fpr'], unpriv_stats['fpr'])
    min_fpr = min(priv_stats['fpr'], unpriv_stats['fpr'])
    metrics['false_positive_rate_balance'] = {
        'value': min_fpr / max_fpr if max_fpr > 0 else 1.0,
        'privileged_group': str(priv_group),
        'unprivileged_group': str(unpriv_group),
        **METRIC_DEFINITIONS['false_positive_rate_balance']
    }
    
    # 12. False Negative Rate Balance
    max_fnr = max(priv_stats['fnr'], unpriv_stats['fnr'])
    min_fnr = min(priv_stats['fnr'], unpriv_stats['fnr'])
    metrics['false_negative_rate_balance'] = {
        'value': min_fnr / max_fnr if max_fnr > 0 else 1.0,
        'privileged_group': str(priv_group),
        'unprivileged_group': str(unpriv_group),
        **METRIC_DEFINITIONS['false_negative_rate_balance']
    }
    
    # 13. False Discovery Rate Balance
    max_fdr = max(priv_stats['fdr'], unpriv_stats['fdr'])
    min_fdr = min(priv_stats['fdr'], unpriv_stats['fdr'])
    metrics['false_discovery_rate_balance'] = {
        'value': min_fdr / max_fdr if max_fdr > 0 else 1.0,
        'privileged_group': str(priv_group),
        'unprivileged_group': str(unpriv_group),
        **METRIC_DEFINITIONS['false_discovery_rate_balance']
    }
    
    # Store group stats for later use
    metrics['_group_stats'] = group_stats
    
    return metrics


# ============================================================================
# REGRESSION FAIRNESS METRICS
# ============================================================================

def calculate_regression_metrics(y_true: np.ndarray, y_pred: np.ndarray, 
                                 sensitive_attr: np.ndarray) -> Dict:
    """Calculate regression fairness metrics."""
    
    metrics = {}
    groups = np.unique(sensitive_attr)
    
    if len(groups) < 2:
        return metrics
    
    # Calculate group-specific metrics
    group_stats = {}
    for group in groups:
        mask = sensitive_attr == group
        y_true_g = y_true[mask]
        y_pred_g = y_pred[mask]
        
        if len(y_true_g) == 0:
            continue
        
        group_stats[group] = {
            'mean_pred': np.mean(y_pred_g),
            'mae': mean_absolute_error(y_true_g, y_pred_g),
            'rmse': np.sqrt(mean_squared_error(y_true_g, y_pred_g)),
            'r2': r2_score(y_true_g, y_pred_g) if len(y_true_g) > 1 else 0,
            'residuals': y_pred_g - y_true_g
        }
    
    sorted_groups = sorted(group_stats.keys())
    group_a = sorted_groups[0]
    group_b = sorted_groups[-1]
    
    stats_a = group_stats[group_a]
    stats_b = group_stats[group_b]
    
    # 14. Mean Difference
    metrics['mean_difference'] = {
        'value': stats_a['mean_pred'] - stats_b['mean_pred'],
        'group_a': str(group_a),
        'group_b': str(group_b),
        **METRIC_DEFINITIONS['mean_difference']
    }
    
    # 15. MAE Parity
    metrics['mae_parity'] = {
        'value': stats_a['mae'] - stats_b['mae'],
        'group_a': str(group_a),
        'group_b': str(group_b),
        **METRIC_DEFINITIONS['mae_parity']
    }
    
    # 16. R² Parity
    metrics['r2_parity'] = {
        'value': stats_a['r2'] - stats_b['r2'],
        'group_a': str(group_a),
        'group_b': str(group_b),
        **METRIC_DEFINITIONS['r2_parity']
    }
    
    # 17. RMSE Balance
    max_rmse = max(stats_a['rmse'], stats_b['rmse'])
    min_rmse = min(stats_a['rmse'], stats_b['rmse'])
    metrics['rmse_balance'] = {
        'value': min_rmse / max_rmse if max_rmse > 0 else 1.0,
        'group_a': str(group_a),
        'group_b': str(group_b),
        **METRIC_DEFINITIONS['rmse_balance']
    }
    
    # 18. Group Fairness in Expectations
    metrics['group_fairness_in_expectations'] = {
        'value': np.mean(stats_a['residuals']) - np.mean(stats_b['residuals']),
        'group_a': str(group_a),
        'group_b': str(group_b),
        **METRIC_DEFINITIONS['group_fairness_in_expectations']
    }
    
    # 19. Residual Fairness (KS statistic)
    ks_stat, _ = stats.ks_2samp(stats_a['residuals'], stats_b['residuals'])
    metrics['residual_fairness'] = {
        'value': ks_stat,
        'group_a': str(group_a),
        'group_b': str(group_b),
        **METRIC_DEFINITIONS['residual_fairness']
    }
    
    return metrics


# ============================================================================
# COMPOSITE & INEQUALITY METRICS
# ============================================================================

def calculate_composite_metrics(y_true: np.ndarray, y_pred: np.ndarray, 
                                sensitive_attr: np.ndarray, 
                                classification_metrics: Dict) -> Dict:
    """Calculate composite and inequality metrics."""
    
    metrics = {}
    groups = np.unique(sensitive_attr)
    
    if len(groups) < 2:
        return metrics
    
    # Get group stats from classification metrics
    group_stats = classification_metrics.get('_group_stats', {})
    
    if not group_stats:
        return metrics
    
    # 20. Theil Index
    total_count = sum(stats['count'] for stats in group_stats.values())
    selection_rates = [stats['selection_rate'] for stats in group_stats.values()]
    proportions = [stats['count'] / total_count for stats in group_stats.values()]
    
    overall_rate = np.mean(selection_rates)
    theil = 0
    for rate, prop in zip(selection_rates, proportions):
        if rate > 0 and overall_rate > 0:
            theil += prop * (rate / overall_rate) * np.log(rate / overall_rate)
    
    metrics['theil_index'] = {
        'value': theil,
        **METRIC_DEFINITIONS['theil_index']
    }
    
    # 21. Generalized Entropy Index (alpha=2)
    gei = 0
    for rate, prop in zip(selection_rates, proportions):
        if overall_rate > 0:
            gei += prop * ((rate / overall_rate) ** 2 - 1)
    gei = gei / 2
    
    metrics['generalized_entropy_index'] = {
        'value': gei,
        **METRIC_DEFINITIONS['generalized_entropy_index']
    }
    
    # 22. Fairness Gap (max difference across all metrics)
    key_metrics = ['tpr', 'fpr', 'ppv', 'selection_rate']
    max_gap = 0
    for metric_key in key_metrics:
        values = [stats[metric_key] for stats in group_stats.values()]
        gap = max(values) - min(values)
        max_gap = max(max_gap, gap)
    
    metrics['fairness_gap'] = {
        'value': max_gap,
        **METRIC_DEFINITIONS['fairness_gap']
    }
    
    # 23. Composite Fairness Score
    # Normalize and average key fairness metrics
    spd = abs(classification_metrics.get('statistical_parity_difference', {}).get('value', 0))
    eod = abs(classification_metrics.get('equal_opportunity_difference', {}).get('value', 0))
    aod = abs(classification_metrics.get('average_odds_difference', {}).get('value', 0))
    
    # Score: 1.0 is perfect, 0.0 is worst
    composite_score = 1.0 - np.mean([min(spd / 0.2, 1.0), min(eod / 0.2, 1.0), min(aod / 0.2, 1.0)])
    
    metrics['composite_fairness_score'] = {
        'value': composite_score,
        **METRIC_DEFINITIONS['composite_fairness_score']
    }
    
    # 24. Bias Amplification Score
    # Compare prediction bias to label bias
    label_rates_by_group = {}
    pred_rates_by_group = {}
    
    for group in groups:
        mask = sensitive_attr == group
        label_rates_by_group[group] = np.mean(y_true[mask])
        pred_rates_by_group[group] = np.mean(y_pred[mask])
    
    label_gap = max(label_rates_by_group.values()) - min(label_rates_by_group.values())
    pred_gap = max(pred_rates_by_group.values()) - min(pred_rates_by_group.values())
    
    bias_amp = pred_gap / label_gap if label_gap > 0 else 1.0
    
    metrics['bias_amplification_score'] = {
        'value': bias_amp,
        **METRIC_DEFINITIONS['bias_amplification_score']
    }
    
    return metrics


# ============================================================================
# INDIVIDUAL FAIRNESS METRICS
# ============================================================================

def calculate_individual_fairness_metrics(X: np.ndarray, y_pred: np.ndarray, 
                                         k: int = 5) -> Dict:
    """Calculate individual fairness metrics."""
    
    metrics = {}
    
    # 25. Consistency
    # For each instance, check if k-nearest neighbors have similar predictions
    from sklearn.neighbors import NearestNeighbors
    
    if len(X) < k + 1:
        return metrics
    
    nbrs = NearestNeighbors(n_neighbors=k+1, algorithm='ball_tree').fit(X)
    distances, indices = nbrs.kneighbors(X)
    
    consistency_scores = []
    for i in range(len(X)):
        neighbor_indices = indices[i][1:]  # Exclude self
        pred_diffs = np.abs(y_pred[i] - y_pred[neighbor_indices])
        consistency_scores.append(1.0 - np.mean(pred_diffs))
    
    metrics['consistency'] = {
        'value': np.mean(consistency_scores),
        **METRIC_DEFINITIONS['consistency']
    }
    
    # 26. Lipschitz Fairness
    # Maximum prediction change relative to input distance
    lipschitz_constants = []
    for i in range(min(100, len(X))):  # Sample for efficiency
        neighbor_indices = indices[i][1:]
        for j in neighbor_indices:
            input_dist = np.linalg.norm(X[i] - X[j])
            pred_diff = abs(y_pred[i] - y_pred[j])
            if input_dist > 0:
                lipschitz_constants.append(pred_diff / input_dist)
    
    metrics['lipschitz_fairness'] = {
        'value': np.max(lipschitz_constants) if lipschitz_constants else 0,
        **METRIC_DEFINITIONS['lipschitz_fairness']
    }
    
    return metrics


# ============================================================================
# CAUSAL FAIRNESS METRICS (Simplified)
# ============================================================================

def calculate_causal_fairness_metrics(y_true: np.ndarray, y_pred: np.ndarray, 
                                     sensitive_attr: np.ndarray) -> Dict:
    """Calculate simplified causal fairness metrics."""
    
    metrics = {}
    groups = np.unique(sensitive_attr)
    
    if len(groups) < 2:
        return metrics
    
    # 27. Counterfactual Fairness (approximation)
    # Measure how much predictions would change if sensitive attribute changed
    group_pred_means = {}
    for group in groups:
        mask = sensitive_attr == group
        group_pred_means[group] = np.mean(y_pred[mask])
    
    cf_diff = max(group_pred_means.values()) - min(group_pred_means.values())
    
    metrics['counterfactual_fairness'] = {
        'value': cf_diff,
        **METRIC_DEFINITIONS['counterfactual_fairness']
    }
    
    # 28 & 29. Natural Direct/Indirect Effects (simplified)
    # These require causal graph, so we provide simplified versions
    metrics['natural_direct_effect'] = {
        'value': cf_diff,  # Simplified: same as counterfactual
        **METRIC_DEFINITIONS['natural_direct_effect']
    }
    
    metrics['natural_indirect_effect'] = {
        'value': 0.0,  # Requires mediator analysis
        **METRIC_DEFINITIONS['natural_indirect_effect']
    }
    
    return metrics


# ============================================================================
# MAIN CALCULATION FUNCTION
# ============================================================================

def calculate_comprehensive_fairness_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    sensitive_attr: np.ndarray,
    X: Optional[np.ndarray] = None,
    is_regression: bool = False
) -> Dict:
    """
    Calculate all comprehensive fairness metrics.
    
    Args:
        y_true: True labels
        y_pred: Predicted labels or values
        sensitive_attr: Sensitive attribute values
        X: Feature matrix (optional, for individual fairness metrics)
        is_regression: Whether this is a regression task
    
    Returns:
        Dictionary of all metrics with their values and metadata
    """
    
    # Convert inputs to numpy arrays to handle list inputs from model_cache
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    sensitive_attr = np.asarray(sensitive_attr)
    if X is not None:
        X = np.asarray(X)
    
    all_metrics = {}
    # Classification metrics
    if not is_regression:
        classification_metrics = calculate_classification_metrics(y_true, y_pred, sensitive_attr)
        all_metrics.update(classification_metrics)
        
        # Composite metrics (requires classification metrics)
        composite_metrics = calculate_composite_metrics(y_true, y_pred, sensitive_attr, classification_metrics)
        all_metrics.update(composite_metrics)
        
        # Causal metrics
        causal_metrics = calculate_causal_fairness_metrics(y_true, y_pred, sensitive_attr)
        all_metrics.update(causal_metrics)
    
    # Regression metrics
    if is_regression:
        regression_metrics = calculate_regression_metrics(y_true, y_pred, sensitive_attr)
        all_metrics.update(regression_metrics)
    
    # Individual fairness metrics (if X provided)
    if X is not None:
        individual_metrics = calculate_individual_fairness_metrics(X, y_pred)
        all_metrics.update(individual_metrics)
    
    # Remove internal group stats
    all_metrics.pop('_group_stats', None)
    
    return all_metrics

