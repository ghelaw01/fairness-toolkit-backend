"""
Bias Detection Module

This module provides comprehensive bias detection tools for AI systems
in public policy contexts. It includes statistical tests, visualization tools,
and automated bias detection methods.

Author: Manus AI
License: MIT
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Optional, Tuple, Union, Any
from scipy import stats
from sklearn.metrics import confusion_matrix
import warnings

class BiasDetector:
    """
    A comprehensive class for detecting various types of bias in AI systems.
    
    This class provides methods for:
    - Statistical bias testing
    - Intersectional bias analysis
    - Temporal bias detection
    - Data bias identification
    """
    
    def __init__(self, data: pd.DataFrame, target_column: str, 
                 sensitive_attributes: List[str], prediction_column: Optional[str] = None):
        """
        Initialize the BiasDetector.
        
        Args:
            data: Dataset containing features, target, and predictions
            target_column: Name of the target/outcome column
            sensitive_attributes: List of sensitive attribute column names
            prediction_column: Name of the prediction column (if available)
        """
        self.data = data.copy()
        self.target_column = target_column
        self.sensitive_attributes = sensitive_attributes
        self.prediction_column = prediction_column
        
        # Validate inputs
        self._validate_inputs()
        
        # Store unique values for each sensitive attribute
        self.attribute_values = {}
        for attr in self.sensitive_attributes:
            self.attribute_values[attr] = sorted(self.data[attr].unique())
    
    def _validate_inputs(self):
        """Validate input parameters."""
        if self.target_column not in self.data.columns:
            raise ValueError(f"Target column '{self.target_column}' not found in data")
        
        for attr in self.sensitive_attributes:
            if attr not in self.data.columns:
                raise ValueError(f"Sensitive attribute '{attr}' not found in data")
        
        if self.prediction_column and self.prediction_column not in self.data.columns:
            raise ValueError(f"Prediction column '{self.prediction_column}' not found in data")
    
    def detect_representation_bias(self) -> Dict[str, Any]:
        """
        Detect representation bias in the dataset.
        
        Representation bias occurs when certain groups are under- or over-represented
        in the dataset compared to the population.
        
        Returns:
            Dict containing representation analysis results
        """
        results = {
            'group_sizes': {},
            'group_proportions': {},
            'representation_ratios': {},
            'chi_square_tests': {}
        }
        
        for attr in self.sensitive_attributes:
            # Calculate group sizes and proportions
            value_counts = self.data[attr].value_counts()
            total_size = len(self.data)
            
            results['group_sizes'][attr] = value_counts.to_dict()
            results['group_proportions'][attr] = (value_counts / total_size).to_dict()
            
            # Calculate representation ratios (largest group / smallest group)
            proportions = value_counts / total_size
            max_prop = proportions.max()
            min_prop = proportions.min()
            results['representation_ratios'][attr] = max_prop / min_prop if min_prop > 0 else float('inf')
            
            # Chi-square test for uniform distribution
            expected_freq = total_size / len(value_counts)
            chi2_stat, chi2_p_value = stats.chisquare(value_counts.values, 
                                                     [expected_freq] * len(value_counts))
            
            results['chi_square_tests'][attr] = {
                'chi2_statistic': chi2_stat,
                'p_value': chi2_p_value,
                'significant': chi2_p_value < 0.05
            }
        
        return results
    
    def detect_outcome_bias(self) -> Dict[str, Any]:
        """
        Detect bias in outcomes across different groups.
        
        Returns:
            Dict containing outcome bias analysis results
        """
        results = {
            'outcome_rates': {},
            'statistical_tests': {},
            'effect_sizes': {}
        }
        
        for attr in self.sensitive_attributes:
            outcome_by_group = {}
            group_data = []
            
            for value in self.attribute_values[attr]:
                group_mask = self.data[attr] == value
                group_outcomes = self.data[group_mask][self.target_column]
                outcome_rate = group_outcomes.mean()
                
                outcome_by_group[value] = {
                    'outcome_rate': outcome_rate,
                    'sample_size': len(group_outcomes),
                    'positive_outcomes': group_outcomes.sum()
                }
                
                group_data.append(group_outcomes.values)
            
            results['outcome_rates'][attr] = outcome_by_group
            
            # Statistical tests
            if len(group_data) == 2:
                # Two-sample t-test for two groups
                stat, p_value = stats.ttest_ind(group_data[0], group_data[1])
                test_name = 't_test'
                
                # Cohen's d effect size
                pooled_std = np.sqrt(((len(group_data[0]) - 1) * np.var(group_data[0], ddof=1) + 
                                    (len(group_data[1]) - 1) * np.var(group_data[1], ddof=1)) / 
                                   (len(group_data[0]) + len(group_data[1]) - 2))
                cohens_d = (np.mean(group_data[0]) - np.mean(group_data[1])) / pooled_std
                effect_size = abs(cohens_d)
                
            else:
                # ANOVA for multiple groups
                stat, p_value = stats.f_oneway(*group_data)
                test_name = 'anova'
                
                # Eta-squared effect size
                ss_between = sum(len(group) * (np.mean(group) - np.mean(np.concatenate(group_data)))**2 
                               for group in group_data)
                ss_total = sum((x - np.mean(np.concatenate(group_data)))**2 
                             for group in group_data for x in group)
                effect_size = ss_between / ss_total if ss_total > 0 else 0
            
            results['statistical_tests'][attr] = {
                'test_type': test_name,
                'statistic': stat,
                'p_value': p_value,
                'significant': p_value < 0.05
            }
            
            results['effect_sizes'][attr] = effect_size
        
        return results
    
    def detect_prediction_bias(self) -> Dict[str, Any]:
        """
        Detect bias in model predictions across different groups.
        
        Requires prediction_column to be specified.
        
        Returns:
            Dict containing prediction bias analysis results
        """
        if not self.prediction_column:
            raise ValueError("Prediction column must be specified for prediction bias detection")
        
        results = {
            'prediction_rates': {},
            'accuracy_by_group': {},
            'confusion_matrices': {},
            'bias_metrics': {}
        }
        
        for attr in self.sensitive_attributes:
            prediction_by_group = {}
            accuracy_by_group = {}
            confusion_matrices = {}
            
            for value in self.attribute_values[attr]:
                group_mask = self.data[attr] == value
                group_data = self.data[group_mask]
                
                if len(group_data) == 0:
                    continue
                
                # Prediction rates
                prediction_rate = group_data[self.prediction_column].mean()
                prediction_by_group[value] = {
                    'prediction_rate': prediction_rate,
                    'sample_size': len(group_data)
                }
                
                # Accuracy
                correct_predictions = (group_data[self.target_column] == 
                                     group_data[self.prediction_column]).sum()
                accuracy = correct_predictions / len(group_data)
                accuracy_by_group[value] = accuracy
                
                # Confusion matrix
                cm = confusion_matrix(group_data[self.target_column], 
                                    group_data[self.prediction_column])
                confusion_matrices[value] = cm.tolist()
            
            results['prediction_rates'][attr] = prediction_by_group
            results['accuracy_by_group'][attr] = accuracy_by_group
            results['confusion_matrices'][attr] = confusion_matrices
            
            # Calculate bias metrics
            prediction_rates = [info['prediction_rate'] for info in prediction_by_group.values()]
            accuracy_rates = list(accuracy_by_group.values())
            
            results['bias_metrics'][attr] = {
                'prediction_rate_difference': max(prediction_rates) - min(prediction_rates),
                'accuracy_difference': max(accuracy_rates) - min(accuracy_rates),
                'prediction_rate_ratio': min(prediction_rates) / max(prediction_rates) if max(prediction_rates) > 0 else 0,
                'accuracy_ratio': min(accuracy_rates) / max(accuracy_rates) if max(accuracy_rates) > 0 else 0
            }
        
        return results
    
    def detect_intersectional_bias(self, max_combinations: int = 10) -> Dict[str, Any]:
        """
        Detect bias at intersections of multiple sensitive attributes.
        
        Args:
            max_combinations: Maximum number of intersectional groups to analyze
            
        Returns:
            Dict containing intersectional bias analysis results
        """
        if len(self.sensitive_attributes) < 2:
            warnings.warn("Intersectional analysis requires at least 2 sensitive attributes")
            return {}
        
        results = {
            'intersectional_groups': {},
            'outcome_disparities': {},
            'sample_sizes': {}
        }
        
        # Create intersectional groups
        intersectional_data = []
        
        for _, row in self.data.iterrows():
            intersection_key = tuple(row[attr] for attr in self.sensitive_attributes)
            intersectional_data.append({
                'intersection': intersection_key,
                'outcome': row[self.target_column],
                'prediction': row[self.prediction_column] if self.prediction_column else None
            })
        
        # Convert to DataFrame for easier analysis
        intersect_df = pd.DataFrame(intersectional_data)
        
        # Analyze each intersectional group
        intersection_groups = intersect_df.groupby('intersection')
        
        # Limit to most common intersections if too many
        group_sizes = intersection_groups.size().sort_values(ascending=False)
        top_intersections = group_sizes.head(max_combinations).index
        
        for intersection in top_intersections:
            group_data = intersection_groups.get_group(intersection)
            
            intersection_str = ' & '.join([f"{attr}={val}" 
                                         for attr, val in zip(self.sensitive_attributes, intersection)])
            
            outcome_rate = group_data['outcome'].mean()
            sample_size = len(group_data)
            
            results['intersectional_groups'][intersection_str] = {
                'outcome_rate': outcome_rate,
                'sample_size': sample_size
            }
            
            if self.prediction_column:
                prediction_rate = group_data['prediction'].mean()
                accuracy = (group_data['outcome'] == group_data['prediction']).mean()
                
                results['intersectional_groups'][intersection_str].update({
                    'prediction_rate': prediction_rate,
                    'accuracy': accuracy
                })
        
        # Calculate disparities
        outcome_rates = [info['outcome_rate'] for info in results['intersectional_groups'].values()]
        if outcome_rates:
            results['outcome_disparities'] = {
                'max_outcome_rate': max(outcome_rates),
                'min_outcome_rate': min(outcome_rates),
                'outcome_rate_difference': max(outcome_rates) - min(outcome_rates),
                'outcome_rate_ratio': min(outcome_rates) / max(outcome_rates) if max(outcome_rates) > 0 else 0
            }
        
        return results
    
    def detect_temporal_bias(self, time_column: str, time_periods: int = 5) -> Dict[str, Any]:
        """
        Detect bias that changes over time.
        
        Args:
            time_column: Name of the time/date column
            time_periods: Number of time periods to analyze
            
        Returns:
            Dict containing temporal bias analysis results
        """
        if time_column not in self.data.columns:
            raise ValueError(f"Time column '{time_column}' not found in data")
        
        # Convert time column to datetime if needed
        time_data = pd.to_datetime(self.data[time_column])
        
        # Create time periods
        time_quantiles = pd.qcut(time_data, q=time_periods, labels=False, duplicates='drop')
        
        results = {
            'time_periods': {},
            'temporal_trends': {},
            'bias_evolution': {}
        }
        
        # Analyze each time period
        for period in range(time_periods):
            period_mask = time_quantiles == period
            period_data = self.data[period_mask]
            
            if len(period_data) == 0:
                continue
            
            period_results = {
                'sample_size': len(period_data),
                'time_range': {
                    'start': time_data[period_mask].min(),
                    'end': time_data[period_mask].max()
                },
                'group_outcomes': {}
            }
            
            # Analyze outcomes by group in this time period
            for attr in self.sensitive_attributes:
                group_outcomes = {}
                for value in self.attribute_values[attr]:
                    group_mask = period_data[attr] == value
                    if group_mask.sum() > 0:
                        outcome_rate = period_data[group_mask][self.target_column].mean()
                        group_outcomes[value] = outcome_rate
                
                period_results['group_outcomes'][attr] = group_outcomes
            
            results['time_periods'][f'period_{period}'] = period_results
        
        # Calculate temporal trends
        for attr in self.sensitive_attributes:
            attr_trends = {}
            
            for value in self.attribute_values[attr]:
                trend_data = []
                for period in range(time_periods):
                    period_key = f'period_{period}'
                    if (period_key in results['time_periods'] and 
                        attr in results['time_periods'][period_key]['group_outcomes'] and
                        value in results['time_periods'][period_key]['group_outcomes'][attr]):
                        
                        outcome_rate = results['time_periods'][period_key]['group_outcomes'][attr][value]
                        trend_data.append(outcome_rate)
                
                if len(trend_data) > 1:
                    # Calculate trend (slope)
                    x = np.arange(len(trend_data))
                    slope, intercept, r_value, p_value, std_err = stats.linregress(x, trend_data)
                    
                    attr_trends[value] = {
                        'slope': slope,
                        'r_squared': r_value**2,
                        'p_value': p_value,
                        'trend_direction': 'increasing' if slope > 0 else 'decreasing',
                        'significant_trend': p_value < 0.05
                    }
            
            results['temporal_trends'][attr] = attr_trends
        
        return results
    
    def comprehensive_bias_report(self) -> Dict[str, Any]:
        """
        Generate a comprehensive bias detection report.
        
        Returns:
            Dict containing all bias detection results
        """
        report = {
            'representation_bias': self.detect_representation_bias(),
            'outcome_bias': self.detect_outcome_bias()
        }
        
        if self.prediction_column:
            report['prediction_bias'] = self.detect_prediction_bias()
        
        if len(self.sensitive_attributes) >= 2:
            report['intersectional_bias'] = self.detect_intersectional_bias()
        
        return report
    
    def plot_bias_summary(self, figsize: Tuple[int, int] = (15, 10)) -> plt.Figure:
        """
        Create a comprehensive bias summary visualization.
        
        Args:
            figsize: Figure size
            
        Returns:
            Matplotlib figure
        """
        fig, axes = plt.subplots(2, 2, figsize=figsize)
        fig.suptitle('Bias Detection Summary', fontsize=16, fontweight='bold')
        
        # 1. Representation bias
        repr_bias = self.detect_representation_bias()
        
        attr_names = []
        repr_ratios = []
        for attr, ratio in repr_bias['representation_ratios'].items():
            attr_names.append(attr)
            repr_ratios.append(ratio if ratio != float('inf') else 10)  # Cap at 10 for visualization
        
        axes[0, 0].bar(attr_names, repr_ratios, color='skyblue')
        axes[0, 0].set_title('Representation Bias\n(Higher = More Imbalanced)')
        axes[0, 0].set_ylabel('Representation Ratio')
        axes[0, 0].tick_params(axis='x', rotation=45)
        
        # 2. Outcome bias
        outcome_bias = self.detect_outcome_bias()
        
        effect_sizes = []
        for attr in self.sensitive_attributes:
            if attr in outcome_bias['effect_sizes']:
                effect_sizes.append(outcome_bias['effect_sizes'][attr])
            else:
                effect_sizes.append(0)
        
        axes[0, 1].bar(self.sensitive_attributes, effect_sizes, color='lightcoral')
        axes[0, 1].set_title('Outcome Bias\n(Effect Size)')
        axes[0, 1].set_ylabel('Effect Size')
        axes[0, 1].tick_params(axis='x', rotation=45)
        
        # 3. Group outcome rates
        if self.sensitive_attributes:
            attr = self.sensitive_attributes[0]  # Use first attribute
            outcome_rates = []
            group_labels = []
            
            for value in self.attribute_values[attr]:
                group_mask = self.data[attr] == value
                outcome_rate = self.data[group_mask][self.target_column].mean()
                outcome_rates.append(outcome_rate)
                group_labels.append(f"{attr}={value}")
            
            axes[1, 0].bar(group_labels, outcome_rates, color='lightgreen')
            axes[1, 0].set_title(f'Outcome Rates by {attr}')
            axes[1, 0].set_ylabel('Outcome Rate')
            axes[1, 0].tick_params(axis='x', rotation=45)
        
        # 4. Prediction bias (if available)
        if self.prediction_column:
            pred_bias = self.detect_prediction_bias()
            
            accuracy_diffs = []
            for attr in self.sensitive_attributes:
                if attr in pred_bias['bias_metrics']:
                    accuracy_diffs.append(pred_bias['bias_metrics'][attr]['accuracy_difference'])
                else:
                    accuracy_diffs.append(0)
            
            axes[1, 1].bar(self.sensitive_attributes, accuracy_diffs, color='orange')
            axes[1, 1].set_title('Prediction Bias\n(Accuracy Difference)')
            axes[1, 1].set_ylabel('Accuracy Difference')
            axes[1, 1].tick_params(axis='x', rotation=45)
        else:
            axes[1, 1].text(0.5, 0.5, 'No Predictions\nAvailable', 
                           ha='center', va='center', transform=axes[1, 1].transAxes,
                           fontsize=12)
            axes[1, 1].set_title('Prediction Bias')
        
        plt.tight_layout()
        return fig
    
    def export_bias_report(self, output_path: str) -> None:
        """
        Export a comprehensive bias report to a file.
        
        Args:
            output_path: Path to save the report
        """
        report = self.comprehensive_bias_report()
        
        report_content = """
# Comprehensive Bias Detection Report

## Executive Summary
This report analyzes potential biases in the dataset and model predictions across different demographic groups.

## Representation Bias Analysis
"""
        
        repr_bias = report['representation_bias']
        for attr, ratio in repr_bias['representation_ratios'].items():
            report_content += f"""
### {attr}
- Representation Ratio: {ratio:.2f}
- Group Sizes: {repr_bias['group_sizes'][attr]}
- Chi-square Test p-value: {repr_bias['chi_square_tests'][attr]['p_value']:.4f}
"""
        
        report_content += """
## Outcome Bias Analysis
"""
        
        outcome_bias = report['outcome_bias']
        for attr in self.sensitive_attributes:
            if attr in outcome_bias['outcome_rates']:
                report_content += f"""
### {attr}
- Effect Size: {outcome_bias['effect_sizes'][attr]:.4f}
- Statistical Test p-value: {outcome_bias['statistical_tests'][attr]['p_value']:.4f}
- Outcome Rates by Group:
"""
                for group, stats in outcome_bias['outcome_rates'][attr].items():
                    report_content += f"  - {group}: {stats['outcome_rate']:.3f} ({stats['sample_size']} samples)\n"
        
        if 'prediction_bias' in report:
            report_content += """
## Prediction Bias Analysis
"""
            pred_bias = report['prediction_bias']
            for attr in self.sensitive_attributes:
                if attr in pred_bias['bias_metrics']:
                    metrics = pred_bias['bias_metrics'][attr]
                    report_content += f"""
### {attr}
- Accuracy Difference: {metrics['accuracy_difference']:.4f}
- Prediction Rate Difference: {metrics['prediction_rate_difference']:.4f}
- Accuracy Ratio: {metrics['accuracy_ratio']:.4f}
"""
        
        # Save report
        with open(output_path, 'w') as f:
            f.write(report_content)


def detect_bias(data: pd.DataFrame, target_column: str, 
               sensitive_attributes: List[str], 
               prediction_column: Optional[str] = None) -> BiasDetector:
    """
    Convenience function to create a bias detector.
    
    Args:
        data: Dataset
        target_column: Name of the target column
        sensitive_attributes: List of sensitive attribute names
        prediction_column: Name of the prediction column (optional)
        
    Returns:
        BiasDetector instance
    """
    return BiasDetector(data, target_column, sensitive_attributes, prediction_column)

