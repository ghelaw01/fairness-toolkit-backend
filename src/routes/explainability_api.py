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

# SHAP cache to avoid recomputation
shap_cache = {}


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
    OPTIMIZED: Uses sampling and caching for faster computation.
    """
    try:
        if "model" not in model_cache or "X_test" not in model_cache:
            return jsonify({"error": "No trained model available. Please run analysis first."}), 400
        
        model = model_cache["model"]
        X_test = np.array(model_cache["X_test"])
        X_columns = model_cache["X_columns"]
        
        # Check if SHAP values are already cached
        cache_key = "shap_values_full"
        if cache_key in shap_cache:
            shap_values = shap_cache[cache_key]
            X_sample = shap_cache.get("X_sample", X_test)
        else:
            # OPTIMIZATION: Sample data for faster computation
            # Use up to 20 instances (or all if less) - ultra-reduced for free tier memory constraints
            max_samples = min(20, len(X_test))
            if len(X_test) > max_samples:
                # Random sample for diversity
                np.random.seed(42)
                sample_indices = np.random.choice(len(X_test), max_samples, replace=False)
                X_sample = X_test[sample_indices]
            else:
                X_sample = X_test
            
            # Create explainer
            explainer = ModelExplainer(
                model=model,
                X_train=X_sample,  # Use sample as reference
                X_test=X_sample,
                feature_names=X_columns
            )
            
            # Compute SHAP values
            shap_values = explainer.compute_shap_values(X_sample)
            
            # Cache the results
            shap_cache[cache_key] = shap_values
            shap_cache["X_sample"] = X_sample
            shap_cache["explainer"] = explainer
            if len(X_test) > max_samples:
                shap_cache["sample_indices"] = sample_indices
        
        # Create summary plot
        fig = plt.figure(figsize=(10, 6))
        import shap
        shap.summary_plot(shap_values, X_sample, feature_names=X_columns, 
                         max_display=10, show=False)
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
            "samples_used": len(X_sample),
            "total_samples": len(X_test),
            "cached": cache_key in shap_cache and cache_key != "shap_values_full",
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
    OPTIMIZED: Uses cached explainer if available.
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
        
        # Validate dimensions
        n_features_model = model.n_features_in_ if hasattr(model, 'n_features_in_') else len(X_columns)
        n_features_data = X_test.shape[1] if len(X_test.shape) > 1 else len(X_test[0])
        
        if n_features_data != n_features_model:
            return jsonify({
                "error": f"Dimension mismatch: X_test has {n_features_data} features, but model expects {n_features_model}. This may be due to data preprocessing issues.",
                "details": {
                    "X_test_shape": X_test.shape if hasattr(X_test, 'shape') else [len(X_test), len(X_test[0]) if X_test else 0],
                    "X_columns_count": len(X_columns),
                    "model_features": n_features_model
                }
            }), 500
        
        # OPTIMIZATION: Use cached SHAP values if available
        cache_key = "shap_values_full"
        if cache_key in shap_cache:
            shap_values = shap_cache[cache_key]
            X_sample = shap_cache.get("X_sample", X_test)
            sample_indices = shap_cache.get("sample_indices", range(len(X_sample)))
            
            # Find the instance in the sample
            if instance_idx in sample_indices:
                sample_idx = list(sample_indices).index(instance_idx)
                instance_shap = shap_values[sample_idx]
                instance_data = X_sample[sample_idx]
            else:
                # Instance not in sample, use first instance as fallback
                instance_shap = shap_values[0]
                instance_data = X_sample[0]
                instance_idx = sample_indices[0] if len(sample_indices) > 0 else 0
        else:
            # No cached SHAP values, compute for this instance only
            explainer = shap.TreeExplainer(model)
            instance_shap = explainer.shap_values(X_test[instance_idx:instance_idx+1])
            if isinstance(instance_shap, list):
                instance_shap = instance_shap[1][0]  # Binary classification
            else:
                instance_shap = instance_shap[0]
            instance_data = X_test[instance_idx]
        
        # Build explanation dict
        explanation = {
            "instance_idx": int(instance_idx),
            "shap_values": {feat: float(val) for feat, val in zip(X_columns, instance_shap)},
            "feature_values": {feat: float(val) for feat, val in zip(X_columns, instance_data)}
        }
        
        # Add actual vs predicted
        if instance_idx < len(y_test):
            explanation["actual_label"] = int(y_test[instance_idx])
        if instance_idx < len(y_pred):
            explanation["predicted_label"] = int(y_pred[instance_idx])
        
        # Create waterfall plot using SHAP directly
        try:
            import shap
            fig = plt.figure(figsize=(10, 6))
            # Create explanation object for waterfall
            base_value = model.predict_proba(X_test)[:, 1].mean()  # Average prediction
            shap_exp = shap.Explanation(
                values=instance_shap,
                base_values=base_value,
                data=instance_data,
                feature_names=X_columns
            )
            shap.waterfall_plot(shap_exp, show=False)
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
        
        # OPTIMIZATION: Use cached SHAP values if available
        cache_key = "shap_values_full"
        if cache_key in shap_cache:
            shap_values = shap_cache[cache_key]
            X_sample = shap_cache.get("X_sample", X_test)
            # Map back to full test set if we used sampling
            if len(X_sample) < len(X_test):
                # Use sample indices to map sensitive attributes
                sample_indices = shap_cache.get("sample_indices", range(len(X_sample)))
                sens_values = sens_values[sample_indices]
        else:
            # Sample for faster computation - ultra-reduced for free tier memory
            max_samples = min(20, len(X_test))
            if len(X_test) > max_samples:
                np.random.seed(42)
                sample_indices = np.random.choice(len(X_test), max_samples, replace=False)
                X_sample = X_test[sample_indices]
                sens_values = sens_values[sample_indices]
                shap_cache["sample_indices"] = sample_indices
            else:
                X_sample = X_test
                sample_indices = range(len(X_test))
            
            # Create explainer
            explainer = ModelExplainer(
                model=model,
                X_train=X_sample,
                X_test=X_sample,
                feature_names=X_columns
            )
            
            # Compute SHAP values
            shap_values = explainer.compute_shap_values(X_sample)
            shap_cache[cache_key] = shap_values
            shap_cache["X_sample"] = X_sample
            shap_cache["explainer"] = explainer
        
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
        
        # OPTIMIZATION: Use cached SHAP values if available
        cache_key = "shap_values_full"
        if cache_key in shap_cache:
            shap_values = shap_cache[cache_key]
            X_sample = shap_cache.get("X_sample", X_test)
            # If we used sampling, adjust sensitive attributes
            if len(X_sample) < len(X_test):
                sample_indices = shap_cache.get("sample_indices", range(len(X_sample)))
                sensitive_test_sampled = {k: np.array(v)[sample_indices] for k, v in sensitive_test.items()}
                sensitive_test = sensitive_test_sampled
                X_test = X_sample
        else:
            # Sample for faster computation - ultra-reduced for free tier memory
            max_samples = min(20, len(X_test))
            if len(X_test) > max_samples:
                np.random.seed(42)
                sample_indices = np.random.choice(len(X_test), max_samples, replace=False)
                X_sample = X_test[sample_indices]
                sensitive_test = {k: np.array(v)[sample_indices] for k, v in sensitive_test.items()}
                X_test = X_sample
                shap_cache["sample_indices"] = sample_indices
            else:
                X_sample = X_test
            
            # Create explainer
            explainer = ModelExplainer(
                model=model,
                X_train=X_sample,
                X_test=X_sample,
                feature_names=X_columns
            )
            
            # Compute SHAP values
            shap_values = explainer.compute_shap_values(X_sample)
            shap_cache[cache_key] = shap_values
            shap_cache["X_sample"] = X_sample
            shap_cache["explainer"] = explainer
        
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



@explainability_bp.route("/clear_cache", methods=["POST"])
def clear_shap_cache():
    """
    Clear SHAP cache to force recomputation.
    Useful when model or data changes.
    """
    try:
        global shap_cache
        cache_size = len(shap_cache)
        shap_cache.clear()
        
        return jsonify({
            "message": f"SHAP cache cleared ({cache_size} items removed)",
            "cache_cleared": True
        })
        
    except Exception as e:
        return jsonify({
            "error": f"Error clearing cache: {str(e)}"
        }), 500


@explainability_bp.route("/cache_status", methods=["GET"])
def get_cache_status():
    """
    Get current cache status and statistics.
    """
    try:
        cache_info = {
            "cached": "shap_values_full" in shap_cache,
            "cache_size": len(shap_cache),
            "has_explainer": "explainer" in shap_cache,
            "has_shap_values": "shap_values_full" in shap_cache,
            "has_sample": "X_sample" in shap_cache,
            "sample_size": len(shap_cache.get("X_sample", [])) if "X_sample" in shap_cache else 0
        }
        
        return jsonify({
            "cache_status": cache_info,
            "message": "Cache status retrieved successfully"
        })
        
    except Exception as e:
        return jsonify({
            "error": f"Error getting cache status: {str(e)}"
        }), 500

