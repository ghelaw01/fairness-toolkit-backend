"""
Enhanced Explainability API Endpoints
Provides comprehensive explainability features including SHAP values,
individual predictions, group-specific analysis, and fairness-aware insights.
"""

import io
import base64
import numpy as np
import pandas as pd
from flask import Blueprint, request, jsonify
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt

from ..explainability import ModelExplainer

explainability_bp = Blueprint("explainability", __name__)

# Shared cache with main fairness API
from .fairness_api import model_cache, analysis_data


def _convert_numpy_types(obj):
    """Recursively convert numpy types to Python native types for JSON serialization."""
    if isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, np.bool_):
        return bool(obj)
    elif isinstance(obj, dict):
        return {key: _convert_numpy_types(value) for key, value in obj.items()}
    elif isinstance(obj, list):
        return [_convert_numpy_types(item) for item in obj]
    else:
        return obj


def _fig_to_base64(fig):
    """Convert matplotlib figure to base64 encoded string."""
    buf = io.BytesIO()
    fig.savefig(buf, format='png', bbox_inches='tight', dpi=100)
    buf.seek(0)
    img_base64 = base64.b64encode(buf.read()).decode('utf-8')
    plt.close(fig)
    return f"data:image/png;base64,{img_base64}"


@explainability_bp.route("/shap/summary", methods=["GET"])
def get_shap_summary():
    """
    Generate SHAP summary plot showing global feature importance.
    Returns base64 encoded image.
    """
    try:
        if "model" not in model_cache or "X_test" not in model_cache:
            return jsonify({"error": "No trained model available. Please run analysis first."}), 400
        
        model = model_cache["model"]
        X_test = np.array(model_cache["X_test"])
        X_columns = model_cache["X_columns"]
        
        # Create explainer
        explainer = ModelExplainer(
            model=model,
            X_train=X_test,  # Use test data as reference (already available)
            X_test=X_test,
            feature_names=X_columns
        )
        
        # Compute SHAP values
        shap_values = explainer.compute_shap_values(X_test)
        
        # Create summary plot
        fig = explainer.plot_shap_summary(X_test, plot_type='dot', max_display=10)
        img_base64 = _fig_to_base64(fig)
        
        # Also compute mean absolute SHAP values for ranking
        mean_abs_shap = np.abs(shap_values).mean(axis=0)
        shap_importance = [
            {"feature": feat, "importance": float(val)}
            for feat, val in sorted(zip(X_columns, mean_abs_shap), key=lambda x: x[1], reverse=True)
        ]
        
        return jsonify({
            "summary_plot": img_base64,
            "shap_importance": shap_importance[:10],
            "message": "SHAP summary computed successfully"
        })
        
    except Exception as e:
        import traceback
        return jsonify({
            "error": f"Error computing SHAP summary: {str(e)}",
            "traceback": traceback.format_exc()
        }), 500


@explainability_bp.route("/shap/individual", methods=["POST"])
def explain_individual_prediction():
    """
    Explain a single prediction using SHAP values.
    Expects JSON: {"instance_idx": 0}
    """
    try:
        if "model" not in model_cache or "X_test" not in model_cache:
            return jsonify({"error": "No trained model available. Please run analysis first."}), 400
        
        data = request.get_json() or {}
        instance_idx = data.get("instance_idx", 0)
        
        model = model_cache["model"]
        X_test = np.array(model_cache["X_test"])
        X_columns = model_cache["X_columns"]
        y_test = model_cache.get("y_test", [])
        y_pred = model_cache.get("y_pred", [])
        
        if instance_idx < 0 or instance_idx >= len(X_test):
            return jsonify({"error": f"Invalid instance index. Must be between 0 and {len(X_test)-1}"}), 400
        
        # Create explainer
        explainer = ModelExplainer(
            model=model,
            X_train=X_test,
            X_test=X_test,
            feature_names=X_columns
        )
        
        # Get explanation for this instance
        explanation = explainer.explain_instance(instance_idx, X_test)
        
        # Add actual vs predicted
        if instance_idx < len(y_test):
            explanation["actual_label"] = int(y_test[instance_idx])
        if instance_idx < len(y_pred):
            explanation["predicted_label"] = int(y_pred[instance_idx])
        
        # Create waterfall plot
        try:
            fig = explainer.plot_instance_explanation(instance_idx, X_test, plot_type='waterfall')
            explanation["waterfall_plot"] = _fig_to_base64(fig)
        except Exception as e:
            explanation["waterfall_plot"] = None
            explanation["plot_error"] = str(e)
        
        # Convert numpy types
        explanation = _convert_numpy_types(explanation)
        
        return jsonify({
            "explanation": explanation,
            "message": "Individual explanation computed successfully"
        })
        
    except Exception as e:
        import traceback
        return jsonify({
            "error": f"Error explaining individual prediction: {str(e)}",
            "traceback": traceback.format_exc()
        }), 500


@explainability_bp.route("/shap/group_comparison", methods=["POST"])
def compare_groups_shap():
    """
    Compare SHAP values across different demographic groups.
    Expects JSON: {"sensitive_attribute": "race"}
    """
    try:
        if "model" not in model_cache or "X_test" not in model_cache:
            return jsonify({"error": "No trained model available. Please run analysis first."}), 400
        
        data = request.get_json() or {}
        sensitive_attr = data.get("sensitive_attribute")
        
        if not sensitive_attr:
            return jsonify({"error": "sensitive_attribute is required"}), 400
        
        model = model_cache["model"]
        X_test = np.array(model_cache["X_test"])
        X_columns = model_cache["X_columns"]
        sensitive_test = model_cache.get("sensitive_test", {})
        
        if sensitive_attr not in sensitive_test:
            return jsonify({
                "error": f"Sensitive attribute '{sensitive_attr}' not found. Available: {list(sensitive_test.keys())}"
            }), 400
        
        sens_values = np.array(sensitive_test[sensitive_attr])
        unique_groups = np.unique(sens_values)
        
        # Create explainer
        explainer = ModelExplainer(
            model=model,
            X_train=X_test,
            X_test=X_test,
            feature_names=X_columns
        )
        
        # Compute SHAP values for all test instances
        shap_values = explainer.compute_shap_values(X_test)
        
        # Compute mean SHAP values per group
        group_shap = {}
        for group in unique_groups:
            mask = sens_values == group
            group_shap_values = shap_values[mask]
            mean_shap = np.abs(group_shap_values).mean(axis=0)
            
            group_shap[str(group)] = {
                "feature_importance": [
                    {"feature": feat, "mean_abs_shap": float(val)}
                    for feat, val in zip(X_columns, mean_shap)
                ],
                "top_features": sorted(
                    [{"feature": feat, "mean_abs_shap": float(val)} for feat, val in zip(X_columns, mean_shap)],
                    key=lambda x: x["mean_abs_shap"],
                    reverse=True
                )[:10],
                "sample_count": int(mask.sum())
            }
        
        # Identify features with largest disparities
        feature_disparities = []
        for i, feat in enumerate(X_columns):
            group_values = [group_shap[str(g)]["feature_importance"][i]["mean_abs_shap"] for g in unique_groups]
            disparity = max(group_values) - min(group_values)
            feature_disparities.append({
                "feature": feat,
                "disparity": float(disparity),
                "group_values": {str(g): float(v) for g, v in zip(unique_groups, group_values)}
            })
        
        feature_disparities.sort(key=lambda x: x["disparity"], reverse=True)
        
        return jsonify({
            "sensitive_attribute": sensitive_attr,
            "groups": list(map(str, unique_groups)),
            "group_shap_analysis": group_shap,
            "feature_disparities": feature_disparities[:10],
            "fairness_insights": {
                "most_disparate_features": [fd["feature"] for fd in feature_disparities[:5]],
                "interpretation": "Features with high disparity may contribute to unfair predictions across groups"
            },
            "message": "Group comparison completed successfully"
        })
        
    except Exception as e:
        import traceback
        return jsonify({
            "error": f"Error comparing groups: {str(e)}",
            "traceback": traceback.format_exc()
        }), 500


@explainability_bp.route("/fairness_aware_features", methods=["GET"])
def get_fairness_aware_features():
    """
    Identify features that contribute most to fairness issues.
    Analyzes correlation between features and prediction disparities.
    """
    try:
        if "model" not in model_cache or "X_test" not in model_cache:
            return jsonify({"error": "No trained model available. Please run analysis first."}), 400
        
        if "results" not in analysis_data:
            return jsonify({"error": "No analysis results available."}), 400
        
        model = model_cache["model"]
        X_test = np.array(model_cache["X_test"])
        X_columns = model_cache["X_columns"]
        y_pred = np.array(model_cache.get("y_pred", []))
        sensitive_test = model_cache.get("sensitive_test", {})
        
        # Create explainer
        explainer = ModelExplainer(
            model=model,
            X_train=X_test,
            X_test=X_test,
            feature_names=X_columns
        )
        
        # Compute SHAP values
        shap_values = explainer.compute_shap_values(X_test)
        
        # Analyze each sensitive attribute
        fairness_aware_analysis = {}
        
        for sens_attr, sens_values in sensitive_test.items():
            sens_array = np.array(sens_values)
            unique_groups = np.unique(sens_array)
            
            # For each feature, compute how differently it affects different groups
            feature_bias_scores = []
            
            for i, feat in enumerate(X_columns):
                feature_shap = shap_values[:, i]
                
                # Compute mean SHAP per group
                group_mean_shap = {}
                for group in unique_groups:
                    mask = sens_array == group
                    group_mean_shap[str(group)] = float(feature_shap[mask].mean())
                
                # Disparity in SHAP values across groups
                shap_disparity = max(group_mean_shap.values()) - min(group_mean_shap.values())
                
                # Correlation with prediction disparity
                # (features that vary a lot across groups and have high SHAP are problematic)
                feature_values = X_test[:, i]
                group_feature_means = {}
                for group in unique_groups:
                    mask = sens_array == group
                    group_feature_means[str(group)] = float(feature_values[mask].mean())
                
                feature_disparity = max(group_feature_means.values()) - min(group_feature_means.values())
                
                # Bias score: combination of SHAP disparity and feature disparity
                bias_score = abs(shap_disparity) * (1 + abs(feature_disparity))
                
                feature_bias_scores.append({
                    "feature": feat,
                    "bias_score": float(bias_score),
                    "shap_disparity": float(abs(shap_disparity)),
                    "feature_disparity": float(abs(feature_disparity)),
                    "group_shap_values": group_mean_shap,
                    "group_feature_means": group_feature_means
                })
            
            # Sort by bias score
            feature_bias_scores.sort(key=lambda x: x["bias_score"], reverse=True)
            
            fairness_aware_analysis[sens_attr] = {
                "top_biased_features": feature_bias_scores[:10],
                "interpretation": f"Features that contribute most to prediction disparities across {sens_attr} groups",
                "recommendation": "Consider removing or transforming these features, or use fairness-aware training"
            }
        
        return jsonify({
            "fairness_aware_analysis": fairness_aware_analysis,
            "message": "Fairness-aware feature analysis completed successfully"
        })
        
    except Exception as e:
        import traceback
        return jsonify({
            "error": f"Error analyzing fairness-aware features: {str(e)}",
            "traceback": traceback.format_exc()
        }), 500


@explainability_bp.route("/test_instances", methods=["GET"])
def get_test_instances():
    """
    Get list of test instances for selection in UI.
    Returns sample of instances with their predictions and sensitive attributes.
    """
    try:
        if "X_test" not in model_cache:
            return jsonify({"error": "No test data available. Please run analysis first."}), 400
        
        X_test = model_cache["X_test"]
        y_test = model_cache.get("y_test", [])
        y_pred = model_cache.get("y_pred", [])
        y_pred_proba = model_cache.get("y_pred_proba", [])
        sensitive_test = model_cache.get("sensitive_test", {})
        X_columns = model_cache.get("X_columns", [])
        
        # Get up to 100 instances
        num_instances = min(len(X_test), 100)
        
        instances = []
        for i in range(num_instances):
            instance = {
                "index": i,
                "actual": int(y_test[i]) if i < len(y_test) else None,
                "predicted": int(y_pred[i]) if i < len(y_pred) else None,
                "probability": float(y_pred_proba[i]) if i < len(y_pred_proba) else None,
                "sensitive_attributes": {
                    attr: str(vals[i]) if i < len(vals) else None
                    for attr, vals in sensitive_test.items()
                },
                "features": {
                    feat: float(X_test[i][j]) if j < len(X_test[i]) else None
                    for j, feat in enumerate(X_columns)
                } if i < len(X_test) else {}
            }
            instances.append(instance)
        
        return jsonify({
            "instances": instances,
            "total_count": len(X_test),
            "message": f"Retrieved {num_instances} test instances"
        })
        
    except Exception as e:
        import traceback
        return jsonify({
            "error": f"Error retrieving test instances: {str(e)}",
            "traceback": traceback.format_exc()
        }), 500

