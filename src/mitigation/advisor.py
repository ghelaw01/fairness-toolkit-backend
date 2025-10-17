"""
Mitigation Advisor

This module provides intelligent recommendations for bias mitigation techniques.
"""

import numpy as np
from typing import Dict, List


class MitigationAdvisor:
    """Intelligent advisor for selecting appropriate bias mitigation techniques."""
    
    def __init__(self):
        self.recommendations_ = []
        
    def analyze_and_recommend(self, fairness_metrics: Dict, data_info: Dict, use_case: str = 'general') -> List[Dict]:
        """Analyze fairness issues and recommend mitigation techniques."""
        recommendations = []
        
        # Extract fairness metric differences with robust fallbacks
        dp_diff = abs(fairness_metrics.get('demographic_parity', {}).get('difference', 0))
        eo_diff = abs(fairness_metrics.get('equal_opportunity', {}).get('difference', 0))
        eqo_diff = abs(fairness_metrics.get('equalized_odds', {}).get('difference', 0))
        
        # If no specific metrics, check if any fairness issues exist at all
        has_fairness_issue = dp_diff > 0.05 or eo_diff > 0.05 or eqo_diff > 0.05
        
        # Always provide at least one recommendation if there's any fairness concern
        if dp_diff > 0.15 or (has_fairness_issue and dp_diff > 0.08):
            recommendations.append({
                "technique": "reweighing",
                "type": "preprocessing",
                "priority": "high" if dp_diff > 0.15 else "medium",
                "reason": "Demographic parity violation detected" if dp_diff > 0.15 else "Moderate group imbalance detected",
                "expected_impact": "High - directly addresses group representation imbalance",
                "trade_offs": "May slightly reduce overall accuracy",
                "plain_language": f"Your data shows {'significant' if dp_diff > 0.15 else 'moderate'} imbalance between groups ({dp_diff:.2%} difference). Reweighing gives more importance to underrepresented groups.",
                "when_to_use": "Best when groups have different representation in your data",
                "difficulty": "Easy"
            })
        
        if dp_diff > 0.1 or (has_fairness_issue and dp_diff > 0.05):
            recommendations.append({
                "technique": "threshold_optimizer",
                "type": "postprocessing",
                "priority": "high" if dp_diff > 0.1 else "medium",
                "reason": "Can quickly balance outcomes without retraining",
                "expected_impact": "High - directly equalizes prediction rates",
                "trade_offs": "Uses different thresholds for different groups",
                "plain_language": f"This adjusts the decision cutoff for each group to ensure fair outcomes. Current difference: {dp_diff:.2%}",
                "when_to_use": "When you need quick results without retraining the model",
                "difficulty": "Easy"
            })
        
        if eo_diff > 0.1 or (has_fairness_issue and eo_diff > 0.05):
            recommendations.append({
                "technique": "calibrated_equalized_odds",
                "type": "postprocessing",
                "priority": "high" if eo_diff > 0.15 else "medium",
                "reason": "Equal opportunity violation detected",
                "expected_impact": "High - ensures equal true positive rates across groups",
                "trade_offs": "May affect overall model calibration",
                "plain_language": f"This ensures all groups have equal chances of getting positive outcomes when they deserve them. Current difference: {eo_diff:.2%}",
                "when_to_use": "When you want to ensure equal benefit for qualified individuals across all groups",
                "difficulty": "Medium"
            })
        
        # If still no recommendations, provide general advice
        if not recommendations:
            recommendations.append({
                "technique": "threshold_optimizer",
                "type": "postprocessing",
                "priority": "medium",
                "reason": "General fairness improvement recommended",
                "expected_impact": "Moderate - can help balance outcomes",
                "trade_offs": "Uses different thresholds for different groups",
                "plain_language": "While your fairness metrics look reasonable, you can still improve fairness by optimizing decision thresholds for each group.",
                "when_to_use": "As a proactive measure to ensure fairness",
                "difficulty": "Easy"
            })
            
            recommendations.append({
                "technique": "reweighing",
                "type": "preprocessing",
                "priority": "low",
                "reason": "Preventive measure for data balance",
                "expected_impact": "Low to Moderate - ensures balanced representation",
                "trade_offs": "May slightly reduce overall accuracy",
                "plain_language": "Reweighing can help ensure all groups are properly represented in your training data.",
                "when_to_use": "As a preventive measure during data preparation",
                "difficulty": "Easy"
            })
        
        # Sort by priority
        priority_order = {"high": 0, "medium": 1, "low": 2}
        recommendations.sort(key=lambda x: priority_order[x["priority"]])
        
        self.recommendations_ = recommendations
        return recommendations
    
    def get_summary(self) -> Dict:
        """Get a summary of all recommendations."""
        if not self.recommendations_:
            return {"total_recommendations": 0, "message": "No recommendations yet"}
        
        high_priority = sum(1 for r in self.recommendations_ if r['priority'] == 'high')
        
        return {
            "total_recommendations": len(self.recommendations_),
            "high_priority": high_priority,
            "top_recommendation": self.recommendations_[0] if self.recommendations_ else None
        }


def get_mitigation_recommendations(fairness_metrics: Dict, data_info: Dict = None, use_case: str = 'general') -> Dict:
    """Get mitigation recommendations."""
    advisor = MitigationAdvisor()
    recommendations = advisor.analyze_and_recommend(fairness_metrics, data_info or {}, use_case)
    summary = advisor.get_summary()
    
    return {
        "recommendations": recommendations,
        "summary": summary,
        "next_steps": [
            "1. Review the recommended techniques",
            "2. Start with the highest priority recommendation",
            "3. Apply the technique to your data",
            "4. Re-run fairness analysis to measure improvement"
        ]
    }

