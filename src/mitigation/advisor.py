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
        
        dp_diff = abs(fairness_metrics.get('demographic_parity', {}).get('difference', 0))
        eo_diff = abs(fairness_metrics.get('equal_opportunity', {}).get('difference', 0))
        
        if dp_diff > 0.15:
            recommendations.append({
                "technique": "reweighing",
                "type": "preprocessing",
                "priority": "high",
                "reason": "Significant demographic parity violation detected",
                "expected_impact": "High - directly addresses group representation imbalance",
                "trade_offs": "May slightly reduce overall accuracy",
                "plain_language": "Your data shows significant imbalance between groups. Reweighing gives more importance to underrepresented groups.",
                "when_to_use": "Best when groups have different representation in your data",
                "difficulty": "Easy"
            })
        
        if dp_diff > 0.1:
            recommendations.append({
                "technique": "threshold_optimizer",
                "type": "postprocessing",
                "priority": "high",
                "reason": "Can quickly balance outcomes without retraining",
                "expected_impact": "High - directly equalizes prediction rates",
                "trade_offs": "Uses different thresholds for different groups",
                "plain_language": "This adjusts the decision cutoff for each group to ensure fair outcomes.",
                "when_to_use": "When you need quick results without retraining the model",
                "difficulty": "Easy"
            })
        
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
