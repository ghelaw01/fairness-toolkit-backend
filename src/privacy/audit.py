"""Audit Logging for Governance and Accountability"""

import json
import logging
from datetime import datetime
from typing import Dict
import os


class AuditLogger:
    """Audit logger for tracking all fairness analyses."""
    
    def __init__(self, log_file: str = "audit_log.jsonl"):
        self.log_file = log_file
        self.logger = logging.getLogger("fairness_audit")
        
    def log_event(self, event_type: str, details: Dict):
        """Log an audit event."""
        event = {
            "timestamp": datetime.now().isoformat(),
            "event_type": event_type,
            "details": details
        }
        
        # Log to file
        with open(self.log_file, 'a') as f:
            f.write(json.dumps(event) + '\n')
        
        return event
    
    def log_analysis(self, analysis_type: str, dataset_info: Dict, results: Dict):
        """Log a fairness analysis."""
        return self.log_event("fairness_analysis", {
            "analysis_type": analysis_type,
            "dataset": dataset_info,
            "results_summary": {
                "metrics_computed": list(results.get("fairness_metrics", {}).keys()),
                "bias_detected": any(
                    abs(m.get("difference", 0)) > 0.1 
                    for m in results.get("fairness_metrics", {}).values() 
                    if isinstance(m, dict)
                )
            }
        })
    
    def log_mitigation(self, technique: str, parameters: Dict, results: Dict):
        """Log a bias mitigation application."""
        return self.log_event("bias_mitigation", {
            "technique": technique,
            "parameters": parameters,
            "improvement": results.get("improvement", {})
        })
    
    def get_audit_trail(self, limit: int = 100) -> list:
        """Retrieve recent audit events."""
        if not os.path.exists(self.log_file):
            return []
        
        events = []
        with open(self.log_file, 'r') as f:
            for line in f:
                events.append(json.loads(line))
        
        return events[-limit:]


def log_analysis(analysis_type: str, dataset_info: Dict, results: Dict):
    """Convenience function to log an analysis."""
    logger = AuditLogger()
    return logger.log_analysis(analysis_type, dataset_info, results)
