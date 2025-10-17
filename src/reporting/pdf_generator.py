"""PDF Report Generator - Streamlined"""

from datetime import datetime
from typing import Dict
import io


def generate_fairness_pdf_report(analysis_results: Dict) -> str:
    """Generate markdown report (can be converted to PDF)."""
    
    report = []
    report.append("# AI Fairness Analysis Report\n")
    report.append(f"**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
    
    # Executive Summary
    report.append("## Executive Summary\n\n")
    fairness_metrics = analysis_results.get("fairness_metrics", {})
    dp_diff = abs(fairness_metrics.get("demographic_parity", {}).get("difference", 0))
    
    if dp_diff > 0.2:
        report.append("⚠️ **HIGH RISK**: Significant fairness concerns detected.\n\n")
    elif dp_diff > 0.1:
        report.append("⚡ **MODERATE RISK**: Some fairness issues identified.\n\n")
    else:
        report.append("✓ **LOW RISK**: Fairness metrics within acceptable range.\n\n")
    
    # Metrics
    report.append("## Fairness Metrics\n\n")
    for metric_name, metric_data in fairness_metrics.items():
        if isinstance(metric_data, dict) and 'difference' in metric_data:
            diff = metric_data['difference']
            status = "✓ Fair" if abs(diff) < 0.1 else "⚠️ Needs Attention"
            report.append(f"- **{metric_name.replace('_', ' ').title()}**: {diff:.3f} - {status}\n")
    
    report.append("\n")
    
    # Recommendations
    report.append("## Recommendations\n\n")
    recs = analysis_results.get("mitigation_recommendations", {}).get("recommendations", [])
    for i, rec in enumerate(recs[:3], 1):
        report.append(f"{i}. **{rec.get('technique', '').replace('_', ' ').title()}**\n")
        report.append(f"   - {rec.get('plain_language', '')}\n")
    
    return "".join(report)


def generate_stakeholder_report(analysis_results: Dict, stakeholder_type: str = 'executive') -> str:
    """Generate stakeholder-specific report."""
    return generate_fairness_pdf_report(analysis_results)
