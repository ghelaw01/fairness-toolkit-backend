"""
Fairness Analysis API Routes (portable, no external toolkit deps)
"""

from __future__ import annotations

import os
import io
import traceback
from typing import Dict, List, Tuple, Optional

import numpy as np
import pandas as pd
from flask import Blueprint, request, jsonify
from werkzeug.utils import secure_filename

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.inspection import permutation_importance

fairness_bp = Blueprint("fairness", __name__)
analysis_data: Dict[str, object] = {}
model_cache: Dict[str, object] = {}

# ---------- helpers ----------

def _df_info(df: pd.DataFrame, filename: str = "") -> Dict:
    return {
        "filename": filename or "(in-memory)",
        "shape": [int(df.shape[0]), int(df.shape[1])],
        "columns": list(map(str, df.columns)),
        "dtypes": {str(c): str(t) for c, t in df.dtypes.items()},
        "missing_values": {str(c): int(v) for c, v in df.isna().sum().items()},
        "sample_data": df.head(5).fillna("").to_dict("records"),
    }

def _find_compas_csv() -> Tuple[Optional[str], List[str]]:
    tried = []
    fname = "compas-scores-two-years.csv"
    here = os.path.abspath(os.path.dirname(__file__))
    for _ in range(8):
        for rel in ["data", os.path.join("..", "data")]:
            p = os.path.abspath(os.path.join(here, rel, fname))
            tried.append(p)
            if os.path.exists(p):
                return p, tried
        here = os.path.abspath(os.path.join(here, ".."))
    return None, tried

def _ensure_binary_target(y: pd.Series) -> Tuple[np.ndarray, Optional[LabelEncoder]]:
    if y.dtype.kind in ("i", "u", "b", "f"):
        uniq = sorted(pd.unique(y.dropna()))
        if len(uniq) > 2:
            raise ValueError("Target must be binary")
        if uniq != [0, 1]:
            mapping = {uniq[0]: 0, uniq[-1]: 1}
            return y.map(mapping).to_numpy(), None
        return y.to_numpy(), None
    le = LabelEncoder()
    y_enc = le.fit_transform(y.astype(str))
    if len(set(y_enc)) != 2:
        raise ValueError("Target must be binary")
    return y_enc, le

def _one_hot_frame(frame: pd.DataFrame) -> pd.DataFrame:
    cats = frame.select_dtypes(include=["object", "category"]).columns.tolist()
    return pd.get_dummies(frame, columns=cats, drop_first=True) if cats else frame.copy()

def _spread(d: Dict[str, float]) -> float:
    vals = list(d.values())
    return float(max(vals) - min(vals)) if vals else 0.0

def _group_rates(y_true: np.ndarray, y_pred: np.ndarray, sensitive: np.ndarray) -> Dict:
    y_true = np.asarray(y_true).astype(int)
    y_pred = np.asarray(y_pred).astype(int)
    s = pd.Series(np.asarray(sensitive))

    selection_rate, tpr, fpr = {}, {}, {}
    for g in s.unique():
        idx = s == g
        yt, yp = y_true[idx], y_pred[idx]
        if len(yt) == 0:
            continue
        tn, fp, fn, tp = confusion_matrix(yt, yp, labels=[0, 1]).ravel()
        selection_rate[str(g)] = float((yp == 1).mean())
        tpr[str(g)] = float(tp / (tp + fn)) if (tp + fn) > 0 else 0.0
        fpr[str(g)] = float(fp / (fp + tn)) if (fp + tn) > 0 else 0.0

    disparities = {
        "selection_rate_diff": _spread(selection_rate),
        "tpr_diff": _spread(tpr),
        "fpr_diff": _spread(fpr),
    }
    summary = [
        {"Metric Name": "Selection Rate Disparity (max-min)", "Value": disparities["selection_rate_diff"],
         "Interpretation": "Gap in predicted positives across groups (smaller is fairer)."},
        {"Metric Name": "TPR Disparity (max-min)", "Value": disparities["tpr_diff"],
         "Interpretation": "Gap in true positive rates (smaller is fairer)."},
        {"Metric Name": "FPR Disparity (max-min)", "Value": disparities["fpr_diff"],
         "Interpretation": "Gap in false positive rates (smaller is fairer)."},
    ]
    return {"groups": {"selection_rate": selection_rate, "tpr": tpr, "fpr": fpr},
            "disparities": disparities, "summary": summary}

def _bias_label(gap: float, low=0.15, high=0.30) -> str:
    return "Low Bias" if gap < low else ("Moderate Bias" if gap < high else "High Bias")

# ---------- endpoints ----------

@fairness_bp.route("/health", methods=["GET"])
def health_check():
    return jsonify({"status": "healthy", "message": "AI Fairness Toolkit API is running"})

@fairness_bp.route("/upload", methods=["POST"])
def upload_data():
    try:
        if "file" not in request.files:
            return jsonify({"error": "No file field 'file' in request"}), 400
        file = request.files["file"]
        if not file or file.filename.strip() == "":
            return jsonify({"error": "No file selected"}), 400
        
        filename = secure_filename(file.filename)
        file_ext = filename.lower().split('.')[-1] if '.' in filename else ''
        
        # Support multiple file formats
        supported_formats = ['csv', 'xlsx', 'xls', 'json', 'tsv', 'txt']
        if file_ext not in supported_formats:
            return jsonify({"error": f"Unsupported file format. Supported formats: {', '.join(supported_formats)}"}), 400

        # Read file based on format
        df = None
        if file_ext == 'csv':
            stream = io.StringIO(file.stream.read().decode("utf-8", errors="ignore"))
            df = pd.read_csv(stream)
        elif file_ext in ['xlsx', 'xls']:
            try:
                df = pd.read_excel(file.stream, engine='openpyxl' if file_ext == 'xlsx' else None)
            except ImportError:
                return jsonify({"error": "Excel support not installed. Please install openpyxl: pip install openpyxl"}), 400
        elif file_ext == 'json':
            stream = io.StringIO(file.stream.read().decode("utf-8", errors="ignore"))
            df = pd.read_json(stream)
        elif file_ext in ['tsv', 'txt']:
            stream = io.StringIO(file.stream.read().decode("utf-8", errors="ignore"))
            df = pd.read_csv(stream, sep='\t')
        
        if df is None or df.empty:
            return jsonify({"error": "Failed to read file or file is empty"}), 400
        
        if len(df.columns) == 0:
            return jsonify({"error": "No columns found in uploaded file"}), 400
        
        analysis_data["raw_data"] = df
        analysis_data["data_info"] = _df_info(df, filename)
        return jsonify({"message": "Data uploaded successfully", "data_info": analysis_data["data_info"]})
    except Exception as e:
        return jsonify({"error": f"Error processing file: {e}", "traceback": traceback.format_exc()}), 500


@fairness_bp.route("/recommend_model", methods=["POST"])
def recommend_model_endpoint():
    """Recommend best model based on dataset characteristics"""
    try:
        if "raw_data" not in analysis_data:
            return jsonify({"error": "No data uploaded. Please upload data first."}), 400
        
        p = request.get_json(silent=True) or {}
        target = p.get("target_column", "")
        sens_attrs: List[str] = p.get("sensitive_attributes", [])
        
        df: pd.DataFrame = analysis_data["raw_data"].copy()
        
        if not target or target not in df.columns:
            return jsonify({"error": "Valid target_column required"}), 400
        
        # Import recommendation system
        from .fairness_api_recommend import analyze_dataset_characteristics, recommend_model
        
        # Analyze dataset
        characteristics = analyze_dataset_characteristics(df, target, sens_attrs)
        
        # Get recommendations
        result = recommend_model(characteristics)
        
        return jsonify(result)
    except Exception as e:
        return jsonify({"error": f"Error recommending model: {e}", "traceback": traceback.format_exc()}), 500


@fairness_bp.route("/analyze", methods=["POST"])
def run_fairness_analysis():
    try:
        if "raw_data" not in analysis_data:
            return jsonify({"error": "No data uploaded. Please upload data first."}), 400

        p = request.get_json(silent=True) or {}
        target = p.get("target_column", "")
        sens_attrs: List[str] = p.get("sensitive_attributes", [])
        feats: List[str] = p.get("feature_columns", [])
        model_type = (p.get("model_type") or "random_forest").lower()

        df: pd.DataFrame = analysis_data["raw_data"].copy()
        if not target or target not in df.columns:
            return jsonify({"error": "Valid target_column required"}), 400
        if not sens_attrs:
            return jsonify({"error": "At least one sensitive attribute must be specified"}), 400
        for a in sens_attrs:
            if a not in df.columns:
                return jsonify({"error": f"Sensitive attribute '{a}' not found"}), 400

        if not feats:
            numeric = df.select_dtypes(include=[np.number]).columns.tolist()
            feats = [c for c in numeric if c != target and c not in sens_attrs]
        missing = [c for c in feats if c not in df.columns]
        if missing:
            return jsonify({"error": f"Feature columns not found: {missing}"}), 400

        # Select columns and handle missing values more carefully
        cols = [target] + sens_attrs + feats
        work = df[cols].copy()
        
        # Drop rows where target or sensitive attributes are missing
        work = work.dropna(subset=[target] + sens_attrs)
        
        # For features, fill missing values with median for numeric, mode for categorical
        for col in feats:
            if work[col].dtype in [np.float64, np.int64]:
                work[col] = work[col].fillna(work[col].median())
            else:
                work[col] = work[col].fillna(work[col].mode()[0] if not work[col].mode().empty else 'missing')
        
        if work.empty or len(work) < 10:
            return jsonify({"error": "Insufficient data after preprocessing (need at least 10 rows)"}), 400

        y_raw = work[target]
        y, _ = _ensure_binary_target(y_raw)

        X = _one_hot_frame(work[feats].copy())
        X_columns = list(X.columns)
        
        # Fill any remaining NaN values after one-hot encoding
        X = X.fillna(0)

        X_train, X_test, y_train, y_test, idx_train, idx_test = train_test_split(
            X, y, work.index, test_size=0.2, random_state=42, stratify=y
        )

        # Models that benefit from scaling
        models_needing_scaling = ["logistic_regression", "svm", "knn", "naive_bayes"]
        
        scaler = None
        X_train_used, X_test_used = X_train, X_test
        if model_type in models_needing_scaling:
            scaler = StandardScaler()
            X_train_used = scaler.fit_transform(X_train)
            X_test_used = scaler.transform(X_test)

        # Train model based on type
        if model_type == "logistic_regression":
            model = LogisticRegression(max_iter=1000, random_state=42)
            model.fit(X_train_used, y_train)
        elif model_type == "random_forest":
            model = RandomForestClassifier(n_estimators=200, random_state=42, n_jobs=-1)
            model.fit(X_train_used, y_train)
        elif model_type == "gradient_boosting":
            from sklearn.ensemble import GradientBoostingClassifier
            model = GradientBoostingClassifier(n_estimators=100, random_state=42)
            model.fit(X_train_used, y_train)
        elif model_type == "svm":
            from sklearn.svm import SVC
            model = SVC(kernel='rbf', probability=True, random_state=42)
            model.fit(X_train_used, y_train)
        elif model_type == "decision_tree":
            from sklearn.tree import DecisionTreeClassifier
            model = DecisionTreeClassifier(max_depth=10, random_state=42)
            model.fit(X_train_used, y_train)
        elif model_type == "naive_bayes":
            from sklearn.naive_bayes import GaussianNB
            model = GaussianNB()
            model.fit(X_train_used, y_train)
        elif model_type == "knn":
            from sklearn.neighbors import KNeighborsClassifier
            model = KNeighborsClassifier(n_neighbors=5, n_jobs=-1)
            model.fit(X_train_used, y_train)
        elif model_type == "xgboost":
            try:
                from xgboost import XGBClassifier
                model = XGBClassifier(n_estimators=100, random_state=42, eval_metric='logloss', use_label_encoder=False)
                model.fit(X_train_used, y_train)
            except ImportError:
                return jsonify({"error": "XGBoost not installed. Please install with: pip install xgboost"}), 400
        else:
            return jsonify({"error": f"Unsupported model type: {model_type}"}), 400

        y_pred = model.predict(X_test_used)
        y_pred_proba = model.predict_proba(X_test_used)[:, 1] if hasattr(model, 'predict_proba') else y_pred
        acc = float(accuracy_score(y_test, y_pred))

        sens_test = {a: work.loc[idx_test, a].values for a in sens_attrs}
        sens_all = {a: work[a].values for a in sens_attrs}

        # Fairness (test)
        fairness_results = {a: _group_rates(y_test, y_pred, sens_test[a]) for a in sens_attrs}

        # Bias details (representation/outcome/prediction + TPR/FPR)
        bias_details = {}
        y_all_series = pd.Series(y, index=work.index)

        for a in sens_attrs:
            s_all = pd.Series(sens_all[a])
            counts = s_all.value_counts(dropna=False)
            props = (counts / float(len(s_all)))

            outcome_rates = {str(k): float(v) for k, v in y_all_series.groupby(work[a]).mean().to_dict().items()}

            s_test = pd.Series(sens_test[a])
            pred_rates, tpr_g, fpr_g = {}, {}, {}
            for g in s_test.unique():
                m = (s_test == g).values
                if not m.any():
                    continue
                yt, yp = y_test[m], y_pred[m]
                pred_rates[str(g)] = float((yp == 1).mean())
                tn, fp, fn, tp = confusion_matrix(yt, yp, labels=[0, 1]).ravel()
                tpr_g[str(g)] = float(tp / (tp + fn)) if (tp + fn) > 0 else 0.0
                fpr_g[str(g)] = float(fp / (fp + tn)) if (fp + tn) > 0 else 0.0

            # Array shapes many UIs expect
            representation_data = [
                {"group": str(k), "count": int(counts[k]), "proportion": float(props[k])}
                for k in counts.index
            ]
            outcome_bias_data = [{"group": k, "rate": v} for k, v in sorted(outcome_rates.items())]
            prediction_bias_data = [{"group": k, "rate": v} for k, v in sorted(pred_rates.items())]
            tpr_data = [{"group": k, "rate": v} for k, v in sorted(tpr_g.items())]
            fpr_data = [{"group": k, "rate": v} for k, v in sorted(fpr_g.items())]

            bias_details[a] = {
                "representation_distribution": {
                    "counts": {str(k): int(v) for k, v in counts.to_dict().items()},
                    "proportions": {str(k): float(v) for k, v in props.to_dict().items()},
                },
                "representation_data": representation_data,
                "outcome_rates": outcome_rates,
                "outcome_bias_data": outcome_bias_data,
                "prediction_rates": pred_rates,
                "prediction_bias_data": prediction_bias_data,
                "tpr_by_group": tpr_g,
                "tpr_data": tpr_data,
                "fpr_by_group": fpr_g,
                "fpr_data": fpr_data,
            }

        # Bias cards + mirror arrays for convenience
        bias_cards = {}
        for a, fr in fairness_results.items():
            cards = {
                "representation": _bias_label(fr["disparities"]["selection_rate_diff"]),
                "outcome": _bias_label(fr["disparities"]["tpr_diff"]),
                "prediction": _bias_label(fr["disparities"]["fpr_diff"]),
                # mirror arrays:
                "representation_data": bias_details[a]["representation_data"],
                "outcome_bias_data": bias_details[a]["outcome_bias_data"],
                "prediction_bias_data": bias_details[a]["prediction_bias_data"],
                "tpr_data": bias_details[a]["tpr_data"],
                "fpr_data": bias_details[a]["fpr_data"],
            }
            bias_cards[a] = cards

        # Feature importance + aliases (labels/values/scores/data)
        feat_imp: Dict[str, object] = {"type": "none", "features": [], "importances": []}
        try:
            if hasattr(model, "feature_importances_"):
                imp = [float(v) for v in model.feature_importances_]
                feat_imp = {"type": "model_importances", "features": X_columns, "importances": imp}
            elif hasattr(model, "coef_"):
                imp = [float(v) for v in np.abs(model.coef_[0])]
                feat_imp = {"type": "abs_logit_coefficients", "features": X_columns, "importances": imp}
            if not feat_imp["importances"] or sum(feat_imp["importances"]) == 0.0:
                pi = permutation_importance(model, X_test_used, y_test, n_repeats=5, random_state=42)
                feat_imp = {"type": "permutation_importance", "features": X_columns,
                            "importances": [float(v) for v in pi.importances_mean]}
        except Exception as e:
            feat_imp = {"type": "none", "error": f"Feature importance failed: {e}", "features": [], "importances": []}

        feat_imp["labels"] = feat_imp.get("features", [])
        feat_imp["values"] = feat_imp.get("importances", [])
        feat_imp["scores"] = feat_imp.get("importances", [])
        feat_imp["data"] = [{"feature": f, "importance": v} for f, v in zip(feat_imp["labels"], feat_imp["values"])]

        top_features = sorted(
            [(f, w) for f, w in zip(feat_imp.get("labels", []), feat_imp.get("values", []))],
            key=lambda x: x[1], reverse=True
        )[:10]

        results = {
            "model_performance": {
                "accuracy": acc, "model_type": model_type, "feature_columns": feats,
                "target_column": target, "sensitive_attributes": sens_attrs,
            },
            "fairness_analysis": fairness_results,
            "bias_detection": bias_cards,
            "bias_detection_details": bias_details,
            "explainability": {
                "feature_importance": feat_imp,
                "global_analysis": {"top_features": top_features},
                "sample_explanations": [],
            },
            "data_summary": {
                "total_samples": int(work.shape[0]),
                "train_samples": int(X_train.shape[0]),
                "test_samples": int(X_test.shape[0]),
                "feature_count": int(len(feats)),
            },
        }

        analysis_data["results"] = results
        analysis_data["df"] = work  # Store for preprocessing mitigation
        analysis_data["target_column"] = target
        analysis_data["sensitive_attrs"] = sens_attrs
        
        model_cache["model"] = model
        model_cache["scaler"] = scaler
        model_cache["X_columns"] = X_columns
        model_cache["feature_columns_original"] = feats
        model_cache["y_test"] = y_test.tolist()
        model_cache["y_pred"] = y_pred.tolist()
        model_cache["y_pred_proba"] = y_pred_proba.tolist()
        model_cache["X_test"] = X_test.values.tolist()
        model_cache["sensitive_test"] = {a: sens_test[a].tolist() for a in sens_attrs}

        return jsonify({"message": "Analysis completed successfully", "results": results})
    except Exception as e:
        return jsonify({"error": f"Error running analysis: {e}", "traceback": traceback.format_exc()}), 500

@fairness_bp.route("/results", methods=["GET"])
def get_results():
    if "results" not in analysis_data:
        return jsonify({"error": "No analysis results available. Please run analysis first."}), 400
    return jsonify(analysis_data["results"])

@fairness_bp.route("/predict", methods=["POST"])
def predict_instance():
    try:
        if "model" not in model_cache:
            return jsonify({"error": "No trained model available. Please run analysis first."}), 400
        payload = request.get_json(silent=True) or {}
        if not payload:
            return jsonify({"error": "No instance data provided"}), 400

        model = model_cache["model"]
        scaler: Optional[StandardScaler] = model_cache.get("scaler")
        orig_feats: List[str] = model_cache["feature_columns_original"]
        X_cols: List[str] = model_cache["X_columns"]

        row = {f: payload.get(f, np.nan) for f in orig_feats}
        inst_enc = _one_hot_frame(pd.DataFrame([row])).reindex(columns=X_cols, fill_value=0)
        X_used = scaler.transform(inst_enc) if scaler is not None else inst_enc
        pred = int(model.predict(X_used)[0])
        prob = model.predict_proba(X_used)[0].tolist() if hasattr(model, "predict_proba") else None
        return jsonify({"prediction": pred, "probability": prob, "features_used": orig_feats})
    except Exception as e:
        return jsonify({"error": f"Error making prediction: {e}", "traceback": traceback.format_exc()}), 500

@fairness_bp.route("/datasets/compas", methods=["GET"])
def load_compas_dataset():
    try:
        path, tried = _find_compas_csv()
        if not path:
            return jsonify({"error": "COMPAS dataset not found", "tried_paths": tried}), 404
        df = pd.read_csv(path)
        analysis_data["raw_data"] = df
        analysis_data["data_info"] = _df_info(df, os.path.basename(path))
        print(f"Loaded COMPAS dataset with {df.shape[0]} records and {df.shape[1]} columns")
        return jsonify({
            "message": "COMPAS dataset loaded successfully",
            "data_info": analysis_data["data_info"],
            "suggested_config": {
                "target_column": "two_year_recid",
                "sensitive_attributes": ["race", "sex"],
                "feature_columns": ["age", "priors_count", "decile_score", "juv_fel_count", "juv_misd_count", "juv_other_count"],
            },
        })
    except Exception as e:
        return jsonify({"error": f"Error loading COMPAS dataset: {e}", "traceback": traceback.format_exc()}), 500

@fairness_bp.route("/export/report", methods=["GET"])
def export_report():
    try:
        if "results" not in analysis_data:
            return jsonify({"error": "No analysis results available"}), 400
        
        r = analysis_data["results"]
        
        def interpret_disparity(value: float) -> str:
            """Interpret disparity value"""
            if value <= 0.05:
                return "✅ Very Low (Excellent)"
            elif value <= 0.10:
                return "✅ Low (Good)"
            elif value <= 0.20:
                return "⚠️ Moderate (Investigate)"
            else:
                return "❌ High (Action Needed)"
        
        md = [
            "# AI Fairness Analysis Report",
            "",
            f"**Generated:** {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}",
            "",
            "---",
            "",
            "## Executive Summary",
            "",
            f"This report presents a comprehensive fairness analysis of a **{r['model_performance']['model_type']}** model trained to predict **{r['model_performance']['target_column']}**. The analysis examines potential disparities across demographic groups using industry-standard fairness metrics.",
            "",
            f"**Model Accuracy:** {r['model_performance']['accuracy']:.1%}",
            f"**Dataset Size:** {r['data_summary']['total_samples']:,} samples",
            f"**Sensitive Attributes Analyzed:** {', '.join(r['fairness_analysis'].keys())}",
            "",
            "---",
            "",
            "## 1. Model Performance",
            "",
            f"- **Model Type:** {r['model_performance']['model_type']}",
            f"- **Overall Accuracy:** {r['model_performance']['accuracy']:.3f} ({r['model_performance']['accuracy']:.1%})",
            f"- **Target Variable:** {r['model_performance']['target_column']}",
            f"- **Features Used:** {len(r['model_performance']['feature_columns'])} features",
            "",
            "### Dataset Split",
            "",
            f"- **Total Samples:** {r['data_summary']['total_samples']:,}",
            f"- **Training Set:** {r['data_summary']['train_samples']:,} samples ({r['data_summary']['train_samples']/r['data_summary']['total_samples']:.1%})",
            f"- **Test Set:** {r['data_summary']['test_samples']:,} samples ({r['data_summary']['test_samples']/r['data_summary']['total_samples']:.1%})",
            "",
            "---",
            "",
            "## 2. Fairness Metrics Explained",
            "",
            "This analysis uses three key fairness metrics to detect disparities across demographic groups:",
            "",
            "### 2.1 Selection Rate Disparity (max–min)",
            "",
            "**What it measures:** For each demographic group, we calculate the selection rate—the fraction of individuals the model predicts as positive (e.g., high risk, approved, flagged). The disparity is the difference between the highest and lowest selection rates across all groups.",
            "",
            "**Formula:** `max(selection_rate) - min(selection_rate)` across all groups",
            "",
            "**Interpretation:**",
            "- **0.00** = Perfect demographic parity (all groups receive positive predictions at the same rate)",
            "- **Larger values** = Bigger gaps in how often different groups are predicted positive",
            "",
            "**Fairness Notion:** Demographic Parity",
            "",
            "**Example:** If selection rates by race are:",
            "- Group A: 0.72 (72% predicted positive)",
            "- Group B: 0.58 (58% predicted positive)",
            "- Group C: 0.41 (41% predicted positive)",
            "",
            "Then Selection Rate Disparity = 0.72 − 0.41 = **0.31**",
            "",
            "*(As a ratio check: 0.41 / 0.72 = 0.57, which would fail the 80% rule often used in HR)*",
            "",
            "### 2.2 True Positive Rate (TPR) Disparity (max–min)",
            "",
            "**What it measures:** TPR per group represents the fraction of truly positive individuals that the model correctly identifies as positive. The disparity is the difference between the highest and lowest TPR across groups.",
            "",
            "**Formula:** `max(TPR) - min(TPR)` across all groups, where TPR = True Positives / (True Positives + False Negatives)",
            "",
            "**Interpretation:**",
            "- **0.00** = Every group has the same chance of being correctly identified when truly positive",
            "- **Larger values** = Some groups receive more correct positive predictions than others",
            "",
            "**Fairness Notion:** Equal Opportunity (parity of TPR)",
            "",
            "**Impact:** High TPR disparity means one group is less likely to receive help or benefits when they truly deserve them.",
            "",
            "### 2.3 False Positive Rate (FPR) Disparity (max–min)",
            "",
            "**What it measures:** FPR per group represents the fraction of truly negative individuals that are incorrectly predicted as positive. The disparity is the difference between the highest and lowest FPR across groups.",
            "",
            "**Formula:** `max(FPR) - min(FPR)` across all groups, where FPR = False Positives / (False Positives + True Negatives)",
            "",
            "**Interpretation:**",
            "- **0.00** = Every group suffers false positives at the same rate",
            "- **Larger values** = Some groups receive more false alarms than others",
            "",
            "**Fairness Notion:** Part of Equalized Odds (parity of both TPR & FPR)",
            "",
            "**Impact:** High FPR disparity means one group is more likely to be wrongly flagged or denied when they shouldn't be.",
            "",
            "### Disparity Thresholds",
            "",
            "The following thresholds are used to interpret disparity values:",
            "",
            "| Disparity Level | Range | Interpretation | Action |",
            "|----------------|-------|----------------|--------|",
            "| ✅ Very Low | ≤ 0.05 | Excellent fairness | Monitor regularly |",
            "| ✅ Low | 0.05 - 0.10 | Good fairness | Continue monitoring |",
            "| ⚠️ Moderate | 0.10 - 0.20 | Investigate causes | Review and analyze |",
            "| ❌ High | ≥ 0.20 | Action needed | Immediate mitigation |",
            "",
            "*Note: These are rules of thumb. Appropriate thresholds depend on domain, risk, and sample sizes.*",
            "",
            "---",
            "",
            "## 3. Fairness Analysis Results",
            "",
        ]
        
        # Add detailed results for each sensitive attribute
        for attr, fdata in r["fairness_analysis"].items():
            md.extend([
                f"### 3.{list(r['fairness_analysis'].keys()).index(attr) + 1} {attr.title()} Analysis",
                "",
                f"**Demographic Groups:** {', '.join(fdata.get('groups', {}).keys())}",
                "",
                "#### Fairness Metrics Summary",
                "",
                "| Metric | Value | Assessment | Interpretation |",
                "|--------|-------|------------|----------------|"
            ])
            
            for item in fdata.get("summary", []):
                metric_name = item['Metric Name']
                value = item['Value']
                assessment = interpret_disparity(value)
                interpretation = item['Interpretation']
                md.append(f"| {metric_name} | {value:.3f} | {assessment} | {interpretation} |")
            
            md.extend(["", "#### Group-Level Breakdown", ""])
            
            # Add group-level details if available
            if 'groups' in fdata:
                md.append("| Group | Selection Rate | TPR | FPR |")
                md.append("|-------|---------------|-----|-----|")
                for group, metrics in fdata['groups'].items():
                    sel_rate = metrics.get('selection_rate', 0)
                    tpr = metrics.get('tpr', 0)
                    fpr = metrics.get('fpr', 0)
                    md.append(f"| {group} | {sel_rate:.3f} | {tpr:.3f} | {fpr:.3f} |")
            
            md.extend(["", ""])
        
        md.extend([
            "---",
            "",
            "## 4. Interpretation Guide",
            "",
            "### How to Act on These Metrics",
            "",
            "#### If Selection Rate Disparity is High:",
            "",
            "**Problem:** One group is selected (predicted positive) far more or less frequently than others.",
            "",
            "**Potential Actions:**",
            "- Check if your decision threshold causes disparate impact",
            "- Consider calibrated thresholds per group",
            "- Apply reweighting or resampling techniques",
            "- Review feature importance for biased predictors",
            "- Ensure training data is representative of all groups",
            "",
            "**Example:** If Black applicants have a 41% approval rate while White applicants have 72%, investigate whether this reflects true differences in qualifications or model bias.",
            "",
            "#### If TPR Disparity is High:",
            "",
            "**Problem:** One group is less likely to be correctly identified when they truly are positive (e.g., missing qualified candidates, failing to identify those who need help).",
            "",
            "**Potential Actions:**",
            "- Examine features that are informative for underperforming groups",
            "- Check sampling quality and label accuracy for affected groups",
            "- Consider group-specific threshold tuning",
            "- Increase representation of affected groups in training data",
            "- Use fairness-aware training algorithms (e.g., equalized odds post-processing)",
            "",
            "**Example:** If the model correctly identifies 85% of qualified White candidates but only 65% of qualified Black candidates, you're systematically missing opportunities for the latter group.",
            "",
            "#### If FPR Disparity is High:",
            "",
            "**Problem:** One group receives more false alarms (incorrectly flagged as positive when they're actually negative).",
            "",
            "**Potential Actions:**",
            "- Review calibration across groups",
            "- Apply cost-sensitive training with group-aware costs",
            "- Use fairness constraints that balance FPR across groups",
            "- Investigate whether certain features are unreliable for specific groups",
            "- Consider threshold adjustments to reduce false positives for affected groups",
            "",
            "**Example:** If 15% of innocent Black defendants are incorrectly flagged as high-risk while only 5% of White defendants are, this creates unfair burden on the Black group.",
            "",
            "---",
            "",
            "## 5. Recommendations",
            "",
        ])
        
        # Generate specific recommendations based on results
        recommendations = []
        rec_num = 1
        
        for attr, fdata in r["fairness_analysis"].items():
            for item in fdata.get("summary", []):
                value = item['Value']
                metric = item['Metric Name']
                
                if value >= 0.20:
                    recommendations.append(f"{rec_num}. **{attr.title()} - {metric}:** HIGH disparity detected ({value:.3f}). Immediate action required. Review model predictions for {attr} groups and consider bias mitigation techniques.")
                    rec_num += 1
                elif value >= 0.10:
                    recommendations.append(f"{rec_num}. **{attr.title()} - {metric}:** MODERATE disparity detected ({value:.3f}). Investigate root causes and monitor closely.")
                    rec_num += 1
        
        if not recommendations:
            recommendations.append("1. **Overall:** All fairness metrics show low disparity. Continue monitoring regularly to ensure fairness is maintained over time.")
        
        recommendations.extend([
            f"{rec_num}. **Continuous Monitoring:** Establish regular fairness audits (quarterly recommended) to detect emerging disparities.",
            f"{rec_num + 1}. **Stakeholder Engagement:** Involve affected communities and domain experts in reviewing these results and proposed interventions.",
            f"{rec_num + 2}. **Documentation:** Maintain detailed records of fairness assessments and any mitigation actions taken.",
            f"{rec_num + 3}. **Model Updates:** When retraining or updating the model, repeat this fairness analysis to ensure improvements don't introduce new biases."
        ])
        
        md.extend(recommendations)
        md.extend([
            "",
            "---",
            "",
            "## 6. Methodology",
            "",
            "### Data Processing",
            "",
            "- Missing values in features were imputed using median (numeric) and mode (categorical) strategies",
            "- Categorical features were one-hot encoded",
            "- Data was split 80/20 for training and testing",
            "",
            "### Model Training",
            "",
            f"- **Algorithm:** {r['model_performance']['model_type']}",
            "- **Training:** Stratified sampling to preserve class distribution",
            "- **Evaluation:** Metrics computed on held-out test set",
            "",
            "### Fairness Metrics Computation",
            "",
            "For each sensitive attribute and demographic group:",
            "",
            "1. **Selection Rate:** Fraction of individuals predicted positive",
            "2. **True Positive Rate (TPR):** Sensitivity = TP / (TP + FN)",
            "3. **False Positive Rate (FPR):** FP / (FP + TN)",
            "4. **Disparity:** Maximum value minus minimum value across all groups",
            "",
            "### Limitations",
            "",
            "- Analysis assumes binary classification (positive/negative outcomes)",
            "- Fairness metrics may conflict (optimizing one may worsen another)",
            "- Statistical significance depends on sample sizes per group",
            "- Intersectional fairness (e.g., Black women) not explicitly analyzed",
            "",
            "---",
            "",
            "## 7. References & Standards",
            "",
            "- **Demographic Parity:** Dwork et al. (2012), \"Fairness Through Awareness\"",
            "- **Equal Opportunity:** Hardt et al. (2016), \"Equality of Opportunity in Supervised Learning\"",
            "- **Equalized Odds:** Hardt et al. (2016)",
            "- **80% Rule:** EEOC Uniform Guidelines on Employee Selection Procedures",
            "",
            "---",
            "",
            "## Appendix: Feature Importance",
            "",
        ])
        
        # Add feature importance if available
        if 'feature_importance' in r:
            md.append("Top 10 Most Important Features:")
            md.append("")
            md.append("| Rank | Feature | Importance |")
            md.append("|------|---------|------------|")
            
            sorted_features = sorted(r['feature_importance'].items(), key=lambda x: x[1], reverse=True)[:10]
            for idx, (feat, imp) in enumerate(sorted_features, 1):
                md.append(f"| {idx} | {feat} | {imp:.4f} |")
        
        md.extend([
            "",
            "---",
            "",
            "## Technical Details",
            "",
            f"- **Report Generated:** {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}",
            "- **Analysis Framework:** scikit-learn with custom fairness metrics",
            "- **Python Version:** 3.11+",
            "",
            "---",
            "",
            "*This report was generated automatically by the AI Fairness & Explainability Toolkit.*",
            "*For questions or concerns about this analysis, please consult with domain experts and fairness specialists.*"
        ])
        
        return jsonify({"report": "\n".join(md), "format": "markdown"})
    except Exception as e:
        return jsonify({"error": f"Error generating report: {e}", "traceback": traceback.format_exc()}), 500



# ========== BIAS MITIGATION ENDPOINTS ==========

@fairness_bp.route("/mitigate/recommend", methods=["POST"])
def get_mitigation_recommendations_endpoint():
    """Get intelligent recommendations for bias mitigation techniques."""
    try:
        from ..mitigation import get_mitigation_recommendations
        
        data = request.get_json()
        fairness_metrics = data.get("fairness_metrics", {})
        data_info = data.get("data_info", {})
        use_case = data.get("use_case", "general")
        
        recommendations = get_mitigation_recommendations(fairness_metrics, data_info, use_case)
        
        return jsonify(recommendations)
    except Exception as e:
        return jsonify({"error": f"Error generating recommendations: {e}", "traceback": traceback.format_exc()}), 500


@fairness_bp.route("/mitigate/apply", methods=["POST"])
def apply_mitigation():
    """Apply a bias mitigation technique to the data."""
    try:
        from ..mitigation import apply_preprocessing_mitigation, apply_postprocessing_mitigation
        
        data = request.get_json()
        technique = data.get("technique")
        technique_type = data.get("technique_type")
        
        if technique_type == "preprocessing":
            # Get data from analysis_data
            if "df" not in analysis_data or "target_column" not in analysis_data:
                return jsonify({"error": "No data loaded. Please run analysis first."}), 400
            
            df = analysis_data["df"]
            target_column = analysis_data["target_column"]
            sensitive_attr = data.get("sensitive_attr")
            
            if not sensitive_attr:
                return jsonify({"error": "sensitive_attr is required for preprocessing"}), 400
            
            # Prepare data
            X = df.drop(columns=[target_column])
            y = df[target_column]
            
            # Apply mitigation
            result = apply_preprocessing_mitigation(
                X, y, sensitive_attr, technique,
                **data.get("parameters", {})
            )
            
            # Store mitigated data
            analysis_data["mitigated_X"] = result["X"]
            analysis_data["mitigated_y"] = result["y"]
            analysis_data["mitigation_weights"] = result.get("weights")
            analysis_data["mitigation_info"] = result["info"]
            
            return jsonify({
                "success": True,
                "info": result["info"],
                "message": "Preprocessing mitigation applied successfully. You can now retrain your model with the mitigated data."
            })
        
        elif technique_type == "postprocessing":
            # Get predictions from model_cache
            if "y_test" not in model_cache or "y_pred_proba" not in model_cache:
                return jsonify({"error": "No model predictions available. Please run analysis first."}), 400
            
            y_true = np.array(model_cache["y_test"])
            y_pred_proba = np.array(model_cache["y_pred_proba"])
            
            # Get sensitive attribute from request or use first stored one
            sensitive_attr_name = data.get("sensitive_attr")
            if not sensitive_attr_name:
                return jsonify({"error": "sensitive_attr is required for postprocessing"}), 400
            
            # Get the sensitive attribute values from the stored test data
            if "sensitive_test" not in model_cache or sensitive_attr_name not in model_cache["sensitive_test"]:
                return jsonify({"error": f"Sensitive attribute '{sensitive_attr_name}' not found in test data. Please run analysis first."}), 400
            
            sensitive_attr = np.array(model_cache["sensitive_test"][sensitive_attr_name])
            
            # Apply mitigation
            result = apply_postprocessing_mitigation(
                y_true, y_pred_proba, sensitive_attr, technique,
                **data.get("parameters", {})
            )
            
            # Store mitigated predictions
            model_cache["mitigated_predictions"] = result["predictions"]
            model_cache["mitigation_info"] = result["info"]
            
            return jsonify({
                "success": True,
                "info": result["info"],
                "predictions": result["predictions"],
                "message": "Postprocessing mitigation applied successfully."
            })
        
        else:
            return jsonify({"error": f"Unknown technique_type: {technique_type}"}), 400
            
    except Exception as e:
        return jsonify({"error": f"Error applying mitigation: {e}", "traceback": traceback.format_exc()}), 500


@fairness_bp.route("/mitigate/compare", methods=["POST"])
def compare_mitigation_results():
    """Compare fairness metrics before and after mitigation."""
    try:
        from ..fairness_metrics import (
            demographic_parity,
            equal_opportunity,
            equalized_odds,
            predictive_parity
        )
        
        data = request.get_json()
        
        # Get original metrics
        if "y_test" not in model_cache or "y_pred" not in model_cache:
            return jsonify({"error": "No original predictions available"}), 400
        
        y_true = np.array(model_cache["y_test"])
        y_pred_original = np.array(model_cache["y_pred"])
        
        # Get mitigated predictions
        if "mitigated_predictions" not in model_cache:
            return jsonify({"error": "No mitigated predictions available. Apply mitigation first."}), 400
        
        y_pred_mitigated = np.array(model_cache["mitigated_predictions"])
        if len(y_pred_mitigated) != len(y_true):
            y_pred_mitigated = (y_pred_mitigated > 0.5).astype(int)
        
        sensitive_attr_data = data.get("sensitive_attr_data")
        if not sensitive_attr_data:
            return jsonify({"error": "sensitive_attr_data is required"}), 400
        
        sensitive_attr = np.array(sensitive_attr_data)
        
        # Calculate metrics for both
        def calc_metrics(y_pred):
            return {
                "demographic_parity": demographic_parity(y_true, y_pred, sensitive_attr),
                "equal_opportunity": equal_opportunity(y_true, y_pred, sensitive_attr),
                "equalized_odds": equalized_odds(y_true, y_pred, sensitive_attr),
                "predictive_parity": predictive_parity(y_true, y_pred, sensitive_attr),
                "accuracy": float(accuracy_score(y_true, y_pred))
            }
        
        original_metrics = calc_metrics(y_pred_original)
        mitigated_metrics = calc_metrics(y_pred_mitigated)
        
        # Calculate improvements
        improvements = {
            "demographic_parity_improvement": abs(original_metrics["demographic_parity"]["difference"]) - abs(mitigated_metrics["demographic_parity"]["difference"]),
            "equal_opportunity_improvement": abs(original_metrics["equal_opportunity"]["difference"]) - abs(mitigated_metrics["equal_opportunity"]["difference"]),
            "accuracy_change": mitigated_metrics["accuracy"] - original_metrics["accuracy"]
        }
        
        return jsonify({
            "original": original_metrics,
            "mitigated": mitigated_metrics,
            "improvements": improvements,
            "summary": {
                "fairness_improved": improvements["demographic_parity_improvement"] > 0,
                "accuracy_maintained": improvements["accuracy_change"] > -0.05,
                "recommendation": "Mitigation successful!" if improvements["demographic_parity_improvement"] > 0 and improvements["accuracy_change"] > -0.05 else "Consider trying a different technique"
            }
        })
        
    except Exception as e:
        return jsonify({"error": f"Error comparing results: {e}", "traceback": traceback.format_exc()}), 500





# ========== ENHANCED EXPLAINABILITY ENDPOINTS ==========

@fairness_bp.route("/explain/lime", methods=["POST"])
def explain_with_lime():
    """Generate LIME explanation for a specific instance."""
    try:
        from ..explainability import generate_lime_explanation
        
        data = request.get_json()
        instance_idx = data.get("instance_idx", 0)
        num_features = data.get("num_features", 10)
        
        if "model" not in model_cache or "X_test" not in model_cache:
            return jsonify({"error": "No model available. Please run analysis first."}), 400
        
        model = model_cache["model"]
        X_train = model_cache.get("X_train")
        X_test = model_cache["X_test"]
        feature_names = model_cache.get("feature_names", [])
        
        explanation = generate_lime_explanation(
            model, X_train, X_test, instance_idx, feature_names, num_features=num_features
        )
        
        return jsonify(explanation)
        
    except Exception as e:
        return jsonify({"error": f"Error generating LIME explanation: {e}", "traceback": traceback.format_exc()}), 500


@fairness_bp.route("/explain/counterfactual", methods=["POST"])
def explain_with_counterfactual():
    """Generate counterfactual explanation for a specific instance."""
    try:
        from ..explainability import generate_counterfactual_explanation
        
        data = request.get_json()
        instance_idx = data.get("instance_idx", 0)
        max_changes = data.get("max_changes", 5)
        
        if "model" not in model_cache or "X_test" not in model_cache:
            return jsonify({"error": "No model available. Please run analysis first."}), 400
        
        model = model_cache["model"]
        X_train = model_cache.get("X_train")
        X_test = model_cache["X_test"]
        feature_names = model_cache.get("feature_names", [])
        
        instance = X_test[instance_idx]
        
        explanation = generate_counterfactual_explanation(
            model, instance, X_train, feature_names, max_changes=max_changes
        )
        
        return jsonify(explanation)
        
    except Exception as e:
        return jsonify({"error": f"Error generating counterfactual: {e}", "traceback": traceback.format_exc()}), 500


@fairness_bp.route("/explain/counterfactuals/multiple", methods=["POST"])
def explain_with_multiple_counterfactuals():
    """Generate multiple counterfactual explanations."""
    try:
        from ..explainability import generate_multiple_counterfactuals
        
        data = request.get_json()
        instance_idx = data.get("instance_idx", 0)
        num_counterfactuals = data.get("num_counterfactuals", 3)
        
        if "model" not in model_cache or "X_test" not in model_cache:
            return jsonify({"error": "No model available. Please run analysis first."}), 400
        
        model = model_cache["model"]
        X_train = model_cache.get("X_train")
        X_test = model_cache["X_test"]
        feature_names = model_cache.get("feature_names", [])
        
        instance = X_test[instance_idx]
        
        explanations = generate_multiple_counterfactuals(
            model, instance, X_train, feature_names, num_counterfactuals=num_counterfactuals
        )
        
        return jsonify(explanations)
        
    except Exception as e:
        return jsonify({"error": f"Error generating counterfactuals: {e}", "traceback": traceback.format_exc()}), 500





# ========== REPORTING ENDPOINTS ==========

@fairness_bp.route("/report/generate", methods=["POST"])
def generate_report():
    """Generate a comprehensive fairness report."""
    try:
        from ..reporting import generate_fairness_pdf_report
        
        data = request.get_json()
        
        # Compile all analysis results
        analysis_results = {
            "dataset_name": analysis_data.get("filename", "Unknown"),
            "target_column": analysis_data.get("target_column", "Unknown"),
            "sensitive_attr": data.get("sensitive_attr", "Unknown"),
            "fairness_metrics": data.get("fairness_metrics", {}),
            "mitigation_recommendations": data.get("mitigation_recommendations", {})
        }
        
        report_content = generate_fairness_pdf_report(analysis_results)
        
        return jsonify({
            "report": report_content,
            "format": "markdown",
            "message": "Report generated successfully"
        })
        
    except Exception as e:
        return jsonify({"error": f"Error generating report: {e}", "traceback": traceback.format_exc()}), 500


@fairness_bp.route("/export/<format>", methods=["POST"])
def export_data(format):
    """Export analysis results in specified format (csv, excel, json)."""
    try:
        from ..reporting import export_to_csv, export_to_excel, export_to_json
        from flask import send_file
        
        data = request.get_json()
        
        if format == 'csv':
            content = export_to_csv(data)
            return send_file(
                io.BytesIO(content),
                mimetype='text/csv',
                as_attachment=True,
                download_name='fairness_results.csv'
            )
        elif format == 'excel':
            content = export_to_excel(data)
            return send_file(
                io.BytesIO(content),
                mimetype='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet',
                as_attachment=True,
                download_name='fairness_results.xlsx'
            )
        elif format == 'json':
            content = export_to_json(data)
            return send_file(
                io.BytesIO(content),
                mimetype='application/json',
                as_attachment=True,
                download_name='fairness_results.json'
            )
        else:
            return jsonify({"error": f"Unsupported format: {format}"}), 400
            
    except Exception as e:
        return jsonify({"error": f"Error exporting data: {e}", "traceback": traceback.format_exc()}), 500

