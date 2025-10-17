"""
AI Explainability Module

This module provides comprehensive explainability tools for AI systems
in public policy contexts. It includes various interpretability methods
and visualization tools for understanding model decisions.

Author: Manus AI
License: MIT
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Optional, Tuple, Union, Any
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier, export_text
from sklearn.linear_model import LogisticRegression
from sklearn.inspection import permutation_importance
import shap
import warnings

class ModelExplainer:
    """
    A comprehensive class for explaining AI model predictions.
    
    This class provides various explainability methods including:
    - Feature importance analysis
    - SHAP (SHapley Additive exPlanations) values
    - Local explanations for individual predictions
    - Global model behavior analysis
    """
    
    def __init__(self, model: Any, X_train: np.ndarray, X_test: np.ndarray, 
                 feature_names: List[str], class_names: Optional[List[str]] = None):
        """
        Initialize the ModelExplainer.
        
        Args:
            model: Trained machine learning model
            X_train: Training features
            X_test: Test features
            feature_names: Names of the features
            class_names: Names of the classes (optional)
        """
        self.model = model
        self.X_train = X_train
        self.X_test = X_test
        self.feature_names = feature_names
        self.class_names = class_names or ["Class 0", "Class 1"]
        
        # Initialize SHAP explainer
        self.shap_explainer = None
        self.shap_values = None
        
        self._initialize_shap()
    
    def _initialize_shap(self):
        """Initialize SHAP explainer based on model type."""
        try:
            if isinstance(self.model, (RandomForestClassifier, DecisionTreeClassifier)):
                self.shap_explainer = shap.TreeExplainer(self.model)
            elif isinstance(self.model, LogisticRegression):
                self.shap_explainer = shap.LinearExplainer(self.model, self.X_train)
            else:
                # Use KernelExplainer as fallback (slower but works with any model)
                self.shap_explainer = shap.KernelExplainer(
                    self.model.predict_proba, 
                    shap.sample(self.X_train, 100)
                )
        except Exception as e:
            warnings.warn(f"Could not initialize SHAP explainer: {e}")
    
    def compute_feature_importance(self) -> Dict[str, np.ndarray]:
        """
        Compute feature importance using multiple methods.
        
        Returns:
            Dict containing different types of feature importance
        """
        importance_dict = {}
        
        # Model-specific feature importance
        if hasattr(self.model, 'feature_importances_'):
            importance_dict['model_importance'] = self.model.feature_importances_
        elif hasattr(self.model, 'coef_'):
            # For linear models, use absolute coefficients
            importance_dict['model_importance'] = np.abs(self.model.coef_[0])
        
        # Permutation importance
        try:
            perm_importance = permutation_importance(
                self.model, self.X_test, 
                self.model.predict(self.X_test), 
                n_repeats=10, random_state=42
            )
            importance_dict['permutation_importance'] = perm_importance.importances_mean
            importance_dict['permutation_importance_std'] = perm_importance.importances_std
        except Exception as e:
            warnings.warn(f"Could not compute permutation importance: {e}")
        
        return importance_dict
    
    def plot_feature_importance(self, importance_type: str = 'model_importance', 
                              top_k: int = 10, figsize: Tuple[int, int] = (10, 6)) -> plt.Figure:
        """
        Plot feature importance.
        
        Args:
            importance_type: Type of importance to plot
            top_k: Number of top features to show
            figsize: Figure size
            
        Returns:
            Matplotlib figure
        """
        importance_dict = self.compute_feature_importance()
        
        if importance_type not in importance_dict:
            raise ValueError(f"Importance type '{importance_type}' not available")
        
        importance = importance_dict[importance_type]
        
        # Get top k features
        top_indices = np.argsort(importance)[-top_k:]
        top_importance = importance[top_indices]
        top_features = [self.feature_names[i] for i in top_indices]
        
        # Create plot
        fig, ax = plt.subplots(figsize=figsize)
        bars = ax.barh(range(len(top_features)), top_importance)
        ax.set_yticks(range(len(top_features)))
        ax.set_yticklabels(top_features)
        ax.set_xlabel('Importance')
        ax.set_title(f'Top {top_k} Features - {importance_type.replace("_", " ").title()}')
        
        # Add value labels on bars
        for i, (bar, value) in enumerate(zip(bars, top_importance)):
            ax.text(value + 0.001, i, f'{value:.3f}', 
                   va='center', ha='left', fontsize=9)
        
        plt.tight_layout()
        return fig
    
    def compute_shap_values(self, X: Optional[np.ndarray] = None) -> np.ndarray:
        """
        Compute SHAP values for explanations.
        
        Args:
            X: Input features (uses X_test if None)
            
        Returns:
            SHAP values array
        """
        if self.shap_explainer is None:
            raise ValueError("SHAP explainer not initialized")
        
        if X is None:
            X = self.X_test
        
        try:
            shap_values = self.shap_explainer.shap_values(X)
            
            # For binary classification, some explainers return values for both classes
            if isinstance(shap_values, list) and len(shap_values) == 2:
                # Use positive class SHAP values
                shap_values = shap_values[1]
            
            self.shap_values = shap_values
            return shap_values
            
        except Exception as e:
            warnings.warn(f"Could not compute SHAP values: {e}")
            return np.zeros((X.shape[0], X.shape[1]))
    
    def plot_shap_summary(self, X: Optional[np.ndarray] = None, 
                         plot_type: str = 'dot', max_display: int = 10) -> plt.Figure:
        """
        Create SHAP summary plot.
        
        Args:
            X: Input features (uses X_test if None)
            plot_type: Type of plot ('dot', 'bar', 'violin')
            max_display: Maximum number of features to display
            
        Returns:
            Matplotlib figure
        """
        if X is None:
            X = self.X_test
        
        shap_values = self.compute_shap_values(X)
        
        # Create the plot
        plt.figure(figsize=(10, 6))
        
        if plot_type == 'bar':
            shap.summary_plot(shap_values, X, feature_names=self.feature_names,
                            plot_type='bar', max_display=max_display, show=False)
        else:
            shap.summary_plot(shap_values, X, feature_names=self.feature_names,
                            max_display=max_display, show=False)
        
        fig = plt.gcf()
        plt.tight_layout()
        return fig
    
    def explain_instance(self, instance_idx: int, X: Optional[np.ndarray] = None) -> Dict[str, Any]:
        """
        Explain a single prediction instance.
        
        Args:
            instance_idx: Index of the instance to explain
            X: Input features (uses X_test if None)
            
        Returns:
            Dict containing explanation information
        """
        if X is None:
            X = self.X_test
        
        instance = X[instance_idx:instance_idx+1]
        prediction = self.model.predict(instance)[0]
        
        if hasattr(self.model, 'predict_proba'):
            prediction_proba = self.model.predict_proba(instance)[0]
        else:
            prediction_proba = None
        
        # Get SHAP values for this instance
        shap_values = self.compute_shap_values(instance)
        instance_shap = shap_values[0] if len(shap_values) > 0 else np.zeros(len(self.feature_names))
        
        # Create feature contribution summary
        feature_contributions = []
        for i, (feature, value, shap_val) in enumerate(zip(
            self.feature_names, instance[0], instance_shap
        )):
            # Handle potential array values - extract scalar value
            if hasattr(shap_val, '__iter__') and not isinstance(shap_val, str):
                shap_value = float(shap_val.item()) if hasattr(shap_val, 'item') else float(shap_val[0])
            else:
                shap_value = float(shap_val)
                
            feature_contributions.append({
                'feature': feature,
                'value': float(value),
                'shap_value': shap_value,
                'contribution': 'Positive' if shap_value > 0 else 'Negative',
                'magnitude': abs(shap_value)
            })
        
        # Sort by magnitude
        feature_contributions.sort(key=lambda x: x['magnitude'], reverse=True)
        
        explanation = {
            'instance_index': instance_idx,
            'prediction': prediction,
            'prediction_proba': prediction_proba,
            'feature_values': dict(zip(self.feature_names, instance[0])),
            'shap_values': dict(zip(self.feature_names, instance_shap)),
            'feature_contributions': feature_contributions,
            'top_positive_features': [
                fc for fc in feature_contributions if fc['contribution'] == 'Positive'
            ][:5],
            'top_negative_features': [
                fc for fc in feature_contributions if fc['contribution'] == 'Negative'
            ][:5]
        }
        
        return explanation
    
    def plot_instance_explanation(self, instance_idx: int, X: Optional[np.ndarray] = None,
                                top_k: int = 10) -> plt.Figure:
        """
        Plot explanation for a single instance.
        
        Args:
            instance_idx: Index of the instance to explain
            X: Input features (uses X_test if None)
            top_k: Number of top features to show
            
        Returns:
            Matplotlib figure
        """
        explanation = self.explain_instance(instance_idx, X)
        
        # Get top k features by absolute SHAP value
        top_features = explanation['feature_contributions'][:top_k]
        
        features = [fc['feature'] for fc in top_features]
        shap_vals = [fc['shap_value'] for fc in top_features]
        colors = ['red' if val < 0 else 'blue' for val in shap_vals]
        
        # Create plot
        fig, ax = plt.subplots(figsize=(10, 6))
        bars = ax.barh(range(len(features)), shap_vals, color=colors, alpha=0.7)
        ax.set_yticks(range(len(features)))
        ax.set_yticklabels(features)
        ax.set_xlabel('SHAP Value (Impact on Prediction)')
        ax.set_title(f'Feature Contributions for Instance {instance_idx}\n'
                    f'Prediction: {explanation["prediction"]} '
                    f'(Prob: {explanation["prediction_proba"][1]:.3f})'
                    if explanation["prediction_proba"] is not None else
                    f'Prediction: {explanation["prediction"]}')
        
        # Add vertical line at x=0
        ax.axvline(x=0, color='black', linestyle='-', alpha=0.3)
        
        # Add value labels
        for i, (bar, val) in enumerate(zip(bars, shap_vals)):
            ax.text(val + (0.001 if val >= 0 else -0.001), i, f'{val:.3f}',
                   va='center', ha='left' if val >= 0 else 'right', fontsize=9)
        
        plt.tight_layout()
        return fig
    
    def global_feature_analysis(self) -> Dict[str, Any]:
        """
        Perform global analysis of feature behavior across the dataset.
        
        Returns:
            Dict containing global feature analysis
        """
        shap_values = self.compute_shap_values()
        
        analysis = {
            'feature_importance_ranking': {},
            'feature_impact_distribution': {},
            'feature_correlations': {},
            'most_influential_features': []
        }
        
        # Global feature importance from SHAP
        mean_abs_shap = np.mean(np.abs(shap_values), axis=0)
        feature_importance_ranking = sorted(
            zip(self.feature_names, mean_abs_shap),
            key=lambda x: x[1], reverse=True
        )
        
        analysis['feature_importance_ranking'] = {
            name: importance for name, importance in feature_importance_ranking
        }
        
        # Feature impact distribution
        for i, feature in enumerate(self.feature_names):
            feature_shap = shap_values[:, i]
            analysis['feature_impact_distribution'][feature] = {
                'mean': np.mean(feature_shap),
                'std': np.std(feature_shap),
                'min': np.min(feature_shap),
                'max': np.max(feature_shap),
                'positive_impact_rate': np.mean(feature_shap > 0)
            }
        
        # Most influential features (top 5)
        analysis['most_influential_features'] = feature_importance_ranking[:5]
        
        return analysis
    
    def generate_explanation_report(self, output_path: str) -> None:
        """
        Generate a comprehensive explanation report.
        
        Args:
            output_path: Path to save the report
        """
        # Compute various analyses
        feature_importance = self.compute_feature_importance()
        global_analysis = self.global_feature_analysis()
        
        # Generate sample instance explanations
        sample_indices = np.random.choice(len(self.X_test), size=min(5, len(self.X_test)), replace=False)
        sample_explanations = [self.explain_instance(idx) for idx in sample_indices]
        
        # Create report content
        report_content = f"""
# AI Model Explainability Report

## Model Overview
- Model Type: {type(self.model).__name__}
- Number of Features: {len(self.feature_names)}
- Number of Test Samples: {len(self.X_test)}

## Global Feature Importance

### Top 10 Most Important Features
"""
        
        # Add feature importance ranking
        for i, (feature, importance) in enumerate(global_analysis['feature_importance_ranking'].items()):
            if i >= 10:
                break
            report_content += f"{i+1}. {feature}: {importance:.4f}\n"
        
        report_content += """
## Feature Impact Analysis

### Feature Impact Distribution
"""
        
        # Add feature impact distribution
        for feature, stats in global_analysis['feature_impact_distribution'].items():
            report_content += f"""
**{feature}:**
- Mean Impact: {stats['mean']:.4f}
- Standard Deviation: {stats['std']:.4f}
- Positive Impact Rate: {stats['positive_impact_rate']:.2%}
"""
        
        report_content += """
## Sample Instance Explanations

"""
        
        # Add sample explanations
        for i, explanation in enumerate(sample_explanations):
            report_content += f"""
### Instance {explanation['instance_index']}
- Prediction: {explanation['prediction']}
- Prediction Probability: {explanation['prediction_proba'][1]:.3f if explanation['prediction_proba'] is not None else 'N/A'}

**Top Contributing Features:**
"""
            for j, fc in enumerate(explanation['top_positive_features'][:3]):
                report_content += f"{j+1}. {fc['feature']}: {fc['shap_value']:.4f} (value: {fc['value']:.3f})\n"
        
        # Save report
        with open(output_path, 'w') as f:
            f.write(report_content)


class DecisionTreeExplainer:
    """
    Specialized explainer for decision tree models.
    """
    
    def __init__(self, model: DecisionTreeClassifier, feature_names: List[str]):
        """
        Initialize DecisionTreeExplainer.
        
        Args:
            model: Trained decision tree model
            feature_names: Names of the features
        """
        self.model = model
        self.feature_names = feature_names
    
    def get_tree_rules(self, max_depth: Optional[int] = None) -> str:
        """
        Get human-readable tree rules.
        
        Args:
            max_depth: Maximum depth to display
            
        Returns:
            String representation of tree rules
        """
        return export_text(
            self.model, 
            feature_names=self.feature_names,
            max_depth=max_depth
        )
    
    def get_decision_path(self, instance: np.ndarray) -> Dict[str, Any]:
        """
        Get the decision path for a specific instance.
        
        Args:
            instance: Input instance
            
        Returns:
            Dict containing decision path information
        """
        # Get decision path
        leaf_id = self.model.decision_path(instance.reshape(1, -1))
        feature = self.model.tree_.feature
        threshold = self.model.tree_.threshold
        
        path_info = {
            'path': [],
            'final_prediction': self.model.predict(instance.reshape(1, -1))[0],
            'prediction_proba': self.model.predict_proba(instance.reshape(1, -1))[0]
        }
        
        # Extract path
        for node_id in leaf_id.indices:
            if feature[node_id] != -2:  # Not a leaf node
                feature_name = self.feature_names[feature[node_id]]
                threshold_val = threshold[node_id]
                feature_val = instance[feature[node_id]]
                
                if feature_val <= threshold_val:
                    decision = f"{feature_name} <= {threshold_val:.3f} (value: {feature_val:.3f})"
                else:
                    decision = f"{feature_name} > {threshold_val:.3f} (value: {feature_val:.3f})"
                
                path_info['path'].append(decision)
        
        return path_info


def explain_model(model: Any, X_train: np.ndarray, X_test: np.ndarray, 
                 feature_names: List[str], class_names: Optional[List[str]] = None) -> ModelExplainer:
    """
    Convenience function to create a model explainer.
    
    Args:
        model: Trained machine learning model
        X_train: Training features
        X_test: Test features
        feature_names: Names of the features
        class_names: Names of the classes (optional)
        
    Returns:
        ModelExplainer instance
    """
    return ModelExplainer(model, X_train, X_test, feature_names, class_names)




# ========== LIME EXPLANATIONS ==========

def generate_lime_explanation(model, X_train, X_test, instance_idx, feature_names, class_names=None, num_features=10):
    """
    Generate LIME (Local Interpretable Model-agnostic Explanations) for a specific instance.
    
    Args:
        model: Trained model
        X_train: Training data
        X_test: Test data  
        instance_idx: Index of instance to explain
        feature_names: List of feature names
        class_names: List of class names
        num_features: Number of top features to show
        
    Returns:
        Dictionary with LIME explanation
    """
    try:
        from lime import lime_tabular
        
        # Create LIME explainer
        explainer = lime_tabular.LimeTabularExplainer(
            X_train,
            feature_names=feature_names,
            class_names=class_names or ['0', '1'],
            mode='classification',
            discretize_continuous=True
        )
        
        # Get instance
        instance = X_test[instance_idx]
        
        # Generate explanation
        exp = explainer.explain_instance(
            instance,
            model.predict_proba,
            num_features=num_features
        )
        
        # Extract feature contributions
        feature_weights = exp.as_list()
        
        # Get prediction
        prediction = model.predict([instance])[0]
        prediction_proba = model.predict_proba([instance])[0]
        
        return {
            "instance_index": int(instance_idx),
            "prediction": int(prediction),
            "prediction_probability": {
                str(i): float(p) for i, p in enumerate(prediction_proba)
            },
            "feature_weights": [
                {
                    "feature": fw[0],
                    "weight": float(fw[1]),
                    "direction": "increases" if fw[1] > 0 else "decreases"
                }
                for fw in feature_weights
            ],
            "plain_language": f"This prediction was influenced most by: {', '.join([fw[0].split()[0] for fw in feature_weights[:3]])}",
            "interpretation": "Positive weights push toward class 1, negative weights push toward class 0"
        }
        
    except Exception as e:
        return {"error": f"Error generating LIME explanation: {str(e)}"}


# ========== COUNTERFACTUAL EXPLANATIONS ==========

def generate_counterfactual_explanation(model, instance, X_train, feature_names, desired_class=None, max_changes=5):
    """
    Generate counterfactual explanations showing minimal changes needed for different outcome.
    
    Args:
        model: Trained model
        instance: Instance to explain (1D array)
        X_train: Training data for reference
        feature_names: List of feature names
        desired_class: Target class (if None, flip current prediction)
        max_changes: Maximum number of features to change
        
    Returns:
        Dictionary with counterfactual explanation
    """
    try:
        from scipy.optimize import differential_evolution
        
        # Get current prediction
        current_pred = model.predict([instance])[0]
        target_class = desired_class if desired_class is not None else (1 - current_pred)
        
        # Define bounds for each feature (min/max from training data)
        bounds = [(X_train[:, i].min(), X_train[:, i].max()) for i in range(X_train.shape[1])]
        
        # Objective function: minimize distance while achieving target class
        def objective(x):
            # Prediction score for target class
            pred_proba = model.predict_proba([x])[0][target_class]
            
            # Distance from original instance (L1 norm)
            distance = np.sum(np.abs(x - instance))
            
            # Number of changed features
            num_changes = np.sum(np.abs(x - instance) > 1e-6)
            
            # Penalize if prediction doesn't match target
            class_penalty = 0 if pred_proba > 0.5 else 1000
            
            # Penalize too many changes
            change_penalty = max(0, num_changes - max_changes) * 100
            
            return distance + class_penalty + change_penalty
        
        # Optimize
        result = differential_evolution(
            objective,
            bounds,
            maxiter=100,
            seed=42,
            atol=0.01,
            tol=0.01
        )
        
        counterfactual = result.x
        cf_pred = model.predict([counterfactual])[0]
        cf_proba = model.predict_proba([counterfactual])[0]
        
        # Find changed features
        changes = []
        for i, (orig, cf, name) in enumerate(zip(instance, counterfactual, feature_names)):
            if abs(orig - cf) > 1e-6:
                changes.append({
                    "feature": name,
                    "original_value": float(orig),
                    "counterfactual_value": float(cf),
                    "change": float(cf - orig),
                    "change_percentage": float((cf - orig) / (orig + 1e-10) * 100) if orig != 0 else float('inf')
                })
        
        # Sort by magnitude of change
        changes.sort(key=lambda x: abs(x["change"]), reverse=True)
        
        return {
            "original_prediction": int(current_pred),
            "counterfactual_prediction": int(cf_pred),
            "counterfactual_probability": {
                str(i): float(p) for i, p in enumerate(cf_proba)
            },
            "num_changes": len(changes),
            "changes": changes[:max_changes],
            "plain_language": f"To change the prediction from {current_pred} to {target_class}, you would need to change {len(changes)} features: " + 
                            ", ".join([f"{c['feature']}" for c in changes[:3]]),
            "actionable": len(changes) <= max_changes,
            "interpretation": "These are the minimal changes needed to achieve a different outcome"
        }
        
    except Exception as e:
        return {"error": f"Error generating counterfactual: {str(e)}"}


def generate_multiple_counterfactuals(model, instance, X_train, feature_names, num_counterfactuals=3):
    """
    Generate multiple diverse counterfactual explanations.
    
    Args:
        model: Trained model
        instance: Instance to explain
        X_train: Training data
        feature_names: List of feature names
        num_counterfactuals: Number of counterfactuals to generate
        
    Returns:
        List of counterfactual explanations
    """
    counterfactuals = []
    
    for i in range(num_counterfactuals):
        max_changes = 3 + i  # Vary the number of allowed changes
        cf = generate_counterfactual_explanation(
            model, instance, X_train, feature_names, max_changes=max_changes
        )
        if "error" not in cf:
            counterfactuals.append(cf)
    
    return {
        "counterfactuals": counterfactuals,
        "summary": f"Generated {len(counterfactuals)} alternative scenarios",
        "plain_language": "Here are different ways to change the outcome, from simplest to more complex"
    }

