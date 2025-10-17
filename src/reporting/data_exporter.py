"""Data Exporter - Export analysis results in various formats"""

import pandas as pd
import json
from typing import Dict
import io


def export_to_csv(data: Dict, filename: str = "fairness_results.csv") -> bytes:
    """Export fairness metrics to CSV."""
    
    # Convert metrics to DataFrame
    rows = []
    fairness_metrics = data.get("fairness_metrics", {})
    
    for metric_name, metric_data in fairness_metrics.items():
        if isinstance(metric_data, dict):
            row = {"metric": metric_name}
            row.update(metric_data)
            rows.append(row)
    
    df = pd.DataFrame(rows)
    
    # Convert to CSV
    buffer = io.StringIO()
    df.to_csv(buffer, index=False)
    return buffer.getvalue().encode('utf-8')


def export_to_excel(data: Dict, filename: str = "fairness_results.xlsx") -> bytes:
    """Export fairness metrics to Excel."""
    
    buffer = io.BytesIO()
    
    with pd.ExcelWriter(buffer, engine='openpyxl') as writer:
        # Fairness metrics sheet
        fairness_metrics = data.get("fairness_metrics", {})
        rows = []
        for metric_name, metric_data in fairness_metrics.items():
            if isinstance(metric_data, dict):
                row = {"metric": metric_name}
                row.update(metric_data)
                rows.append(row)
        
        df_metrics = pd.DataFrame(rows)
        df_metrics.to_excel(writer, sheet_name='Fairness Metrics', index=False)
        
        # Recommendations sheet
        recs = data.get("mitigation_recommendations", {}).get("recommendations", [])
        if recs:
            df_recs = pd.DataFrame(recs)
            df_recs.to_excel(writer, sheet_name='Recommendations', index=False)
    
    return buffer.getvalue()


def export_to_json(data: Dict) -> bytes:
    """Export all results to JSON."""
    return json.dumps(data, indent=2).encode('utf-8')
