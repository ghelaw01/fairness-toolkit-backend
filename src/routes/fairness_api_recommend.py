"""
Model Recommendation System
Analyzes dataset characteristics and recommends the best ML model
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler


def analyze_dataset_characteristics(df: pd.DataFrame, target_col: str, sensitive_attrs: List[str]) -> Dict:
    """Analyze dataset to determine best model type"""
    
    # Exclude target and sensitive attributes from features
    feature_cols = [c for c in df.columns if c != target_col and c not in sensitive_attrs]
    
    n_samples = len(df)
    n_features = len(feature_cols)
    
    # Check feature types
    numeric_features = df[feature_cols].select_dtypes(include=[np.number]).columns.tolist()
    categorical_features = [c for c in feature_cols if c not in numeric_features]
    
    n_numeric = len(numeric_features)
    n_categorical = len(categorical_features)
    
    # Check target balance
    target_counts = df[target_col].value_counts()
    class_balance = target_counts.min() / target_counts.max() if len(target_counts) > 1 else 1.0
    
    # Check for missing values
    missing_ratio = df[feature_cols].isna().sum().sum() / (n_samples * n_features) if n_features > 0 else 0
    
    # Check feature correlations (for numeric features only)
    if n_numeric > 1:
        corr_matrix = df[numeric_features].corr().abs()
        # Get upper triangle of correlation matrix
        upper_triangle = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
        high_corr_features = [column for column in upper_triangle.columns if any(upper_triangle[column] > 0.8)]
        feature_correlation_level = "high" if len(high_corr_features) > 0 else "low"
    else:
        feature_correlation_level = "low"
    
    # Determine dataset complexity
    if n_samples < 1000:
        dataset_size = "small"
    elif n_samples < 10000:
        dataset_size = "medium"
    else:
        dataset_size = "large"
    
    if n_features < 10:
        feature_complexity = "low"
    elif n_features < 50:
        feature_complexity = "medium"
    else:
        feature_complexity = "high"
    
    return {
        "n_samples": n_samples,
        "n_features": n_features,
        "n_numeric": n_numeric,
        "n_categorical": n_categorical,
        "dataset_size": dataset_size,
        "feature_complexity": feature_complexity,
        "class_balance": float(class_balance),
        "missing_ratio": float(missing_ratio),
        "feature_correlation": feature_correlation_level
    }


def recommend_model(characteristics: Dict) -> Dict:
    """Recommend best model based on dataset characteristics"""
    
    n_samples = characteristics["n_samples"]
    n_features = characteristics["n_features"]
    dataset_size = characteristics["dataset_size"]
    feature_complexity = characteristics["feature_complexity"]
    class_balance = characteristics["class_balance"]
    n_categorical = characteristics["n_categorical"]
    feature_correlation = characteristics["feature_correlation"]
    
    recommendations = []
    
    # Rule-based recommendation system
    
    # 1. Random Forest - Good general purpose, handles non-linearity well
    rf_score = 70
    rf_reasons = ["Robust to overfitting", "Handles mixed feature types well"]
    
    if dataset_size in ["medium", "large"]:
        rf_score += 10
        rf_reasons.append("Works well with medium to large datasets")
    
    if feature_complexity in ["medium", "high"]:
        rf_score += 10
        rf_reasons.append("Handles complex feature interactions")
    
    if n_categorical > 0:
        rf_score += 5
        rf_reasons.append("Naturally handles categorical features")
    
    recommendations.append({
        "model": "random_forest",
        "name": "Random Forest",
        "score": rf_score,
        "reasons": rf_reasons,
        "pros": ["No feature scaling needed", "Feature importance available", "Robust to outliers"],
        "cons": ["Can be slow on very large datasets", "Less interpretable than simpler models"]
    })
    
    # 2. Logistic Regression - Good for linear relationships, interpretable
    lr_score = 60
    lr_reasons = ["Highly interpretable", "Fast training"]
    
    if feature_complexity == "low":
        lr_score += 15
        lr_reasons.append("Works well with simpler feature spaces")
    
    if feature_correlation == "low":
        lr_score += 10
        lr_reasons.append("Features are not highly correlated")
    
    if class_balance > 0.3:
        lr_score += 5
        lr_reasons.append("Classes are reasonably balanced")
    
    recommendations.append({
        "model": "logistic_regression",
        "name": "Logistic Regression",
        "score": lr_score,
        "reasons": lr_reasons,
        "pros": ["Very interpretable", "Fast training and prediction", "Works well with linear relationships"],
        "cons": ["Assumes linear relationships", "May underfit complex data"]
    })
    
    # 3. Gradient Boosting - Excellent performance, handles complexity
    gb_score = 75
    gb_reasons = ["Excellent predictive performance", "Handles non-linear relationships"]
    
    if dataset_size in ["medium", "large"]:
        gb_score += 10
        gb_reasons.append("Performs well on medium to large datasets")
    
    if feature_complexity in ["medium", "high"]:
        gb_score += 10
        gb_reasons.append("Excels with complex feature interactions")
    
    if class_balance < 0.3:
        gb_score += 5
        gb_reasons.append("Handles imbalanced classes well")
    
    recommendations.append({
        "model": "gradient_boosting",
        "name": "Gradient Boosting",
        "score": gb_score,
        "reasons": gb_reasons,
        "pros": ["State-of-the-art performance", "Handles missing values", "Feature importance available"],
        "cons": ["Longer training time", "Requires careful tuning", "Can overfit if not regularized"]
    })
    
    # 4. Decision Tree - Simple, interpretable
    dt_score = 50
    dt_reasons = ["Very interpretable", "Fast training"]
    
    if dataset_size == "small":
        dt_score += 10
        dt_reasons.append("Works well with small datasets")
    
    if feature_complexity == "low":
        dt_score += 10
        dt_reasons.append("Good for simple decision boundaries")
    
    recommendations.append({
        "model": "decision_tree",
        "name": "Decision Tree",
        "score": dt_score,
        "reasons": dt_reasons,
        "pros": ["Highly interpretable", "No feature scaling needed", "Handles non-linear relationships"],
        "cons": ["Prone to overfitting", "Unstable (small data changes affect tree structure)"]
    })
    
    # 5. SVM - Good for high-dimensional data
    svm_score = 55
    svm_reasons = ["Effective in high dimensions"]
    
    if n_features > n_samples * 0.5:
        svm_score += 15
        svm_reasons.append("Excellent for high-dimensional data")
    
    if dataset_size == "small":
        svm_score += 10
        svm_reasons.append("Works well with smaller datasets")
    
    if class_balance > 0.4:
        svm_score += 5
        svm_reasons.append("Performs well with balanced classes")
    
    recommendations.append({
        "model": "svm",
        "name": "Support Vector Machine",
        "score": svm_score,
        "reasons": svm_reasons,
        "pros": ["Effective in high dimensions", "Memory efficient", "Versatile (different kernels)"],
        "cons": ["Slow on large datasets", "Requires feature scaling", "Less interpretable"]
    })
    
    # 6. Naive Bayes - Fast, works well with text/categorical
    nb_score = 45
    nb_reasons = ["Very fast training and prediction"]
    
    if n_categorical > n_features * 0.5:
        nb_score += 15
        nb_reasons.append("Works well with many categorical features")
    
    if dataset_size == "large":
        nb_score += 10
        nb_reasons.append("Scales well to large datasets")
    
    recommendations.append({
        "model": "naive_bayes",
        "name": "Naive Bayes",
        "score": nb_score,
        "reasons": nb_reasons,
        "pros": ["Very fast", "Works well with high dimensions", "Good for text classification"],
        "cons": ["Assumes feature independence", "May underperform on complex relationships"]
    })
    
    # 7. KNN - Simple, non-parametric
    knn_score = 40
    knn_reasons = ["Non-parametric", "No training phase"]
    
    if dataset_size == "small":
        knn_score += 15
        knn_reasons.append("Works well with small datasets")
    
    if feature_complexity == "low":
        knn_score += 10
        knn_reasons.append("Good for simple patterns")
    
    recommendations.append({
        "model": "knn",
        "name": "K-Nearest Neighbors",
        "score": knn_score,
        "reasons": knn_reasons,
        "pros": ["Simple and intuitive", "No training required", "Naturally handles multi-class"],
        "cons": ["Slow prediction on large datasets", "Sensitive to feature scaling", "Curse of dimensionality"]
    })
    
    # Sort by score
    recommendations.sort(key=lambda x: x["score"], reverse=True)
    
    # Add rank
    for idx, rec in enumerate(recommendations, 1):
        rec["rank"] = idx
        if idx == 1:
            rec["recommended"] = True
        else:
            rec["recommended"] = False
    
    return {
        "recommendations": recommendations,
        "top_choice": recommendations[0],
        "characteristics": characteristics
    }
