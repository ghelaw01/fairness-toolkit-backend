"""PDF Report Generator - Comprehensive Stakeholder Reports"""

from datetime import datetime
from typing import Dict, Optional
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


def generate_stakeholder_report(analysis_results: Dict, stakeholder_type: str = 'executive', 
                                 before_metrics: Optional[Dict] = None, 
                                 after_metrics: Optional[Dict] = None) -> str:
    """Generate stakeholder-specific report with before/after comparison."""
    
    if stakeholder_type == 'executive':
        return _generate_executive_report(analysis_results, before_metrics, after_metrics)
    elif stakeholder_type == 'technical':
        return _generate_technical_report(analysis_results, before_metrics, after_metrics)
    elif stakeholder_type == 'compliance':
        return _generate_compliance_report(analysis_results, before_metrics, after_metrics)
    else:
        return generate_fairness_pdf_report(analysis_results)


def _generate_executive_report(analysis_results: Dict, before_metrics: Optional[Dict], after_metrics: Optional[Dict]) -> str:
    """Generate executive summary report."""
    
    report = []
    report.append("# Executive Summary: AI Fairness Analysis\n\n")
    report.append(f"**Report Date:** {datetime.now().strftime('%B %d, %Y')}\n")
    report.append(f"**Dataset:** {analysis_results.get('dataset_name', 'N/A')}\n")
    report.append(f"**Target Variable:** {analysis_results.get('target_column', 'N/A')}\n")
    report.append(f"**Protected Attribute:** {analysis_results.get('sensitive_attr', 'N/A')}\n\n")
    
    report.append("---\n\n")
    
    # Key Findings
    report.append("## 🎯 Key Findings\n\n")
    
    model_perf = analysis_results.get('model_performance', {})
    accuracy = model_perf.get('accuracy', 0)
    report.append(f"**Model Accuracy:** {accuracy*100:.1f}%\n\n")
    
    # Fairness Assessment
    fairness_analysis = analysis_results.get('fairness_analysis', {})
    if fairness_analysis:
        first_attr = list(fairness_analysis.keys())[0] if fairness_analysis else None
        if first_attr:
            disparities = fairness_analysis[first_attr].get('disparities', {})
            sr_diff = abs(disparities.get('selection_rate_diff', 0))
            tpr_diff = abs(disparities.get('tpr_diff', 0))
            fpr_diff = abs(disparities.get('fpr_diff', 0))
            
            # Risk Level
            max_disparity = max(sr_diff, tpr_diff, fpr_diff)
            if max_disparity > 0.2:
                risk_level = "🔴 **HIGH RISK**"
                risk_desc = "Significant fairness concerns require immediate attention"
            elif max_disparity > 0.1:
                risk_level = "🟡 **MODERATE RISK**"
                risk_desc = "Some fairness issues identified, mitigation recommended"
            else:
                risk_level = "🟢 **LOW RISK**"
                risk_desc = "Fairness metrics within acceptable range"
            
            report.append(f"**Fairness Risk Level:** {risk_level}\n\n")
            report.append(f"*{risk_desc}*\n\n")
    
    # Before/After Comparison (if mitigation was applied)
    if before_metrics and after_metrics:
        report.append("## 📊 Mitigation Impact\n\n")
        report.append("**Bias Reduction Results:**\n\n")
        
        # Use comprehensive metrics from summary
        before_summary = before_metrics.get('summary', [])
        after_summary = after_metrics.get('summary', [])
        
        if before_summary and after_summary:
            report.append("| Metric | Category | Before | After | Change |\n")
            report.append("|--------|----------|--------|-------|--------|\n")
            
            # Show key metrics first, then others
            key_metrics = ["Statistical Parity Difference", "Disparate Impact", "Equal Opportunity Difference", 
                          "Average Odds Difference", "Predictive Parity Difference"]
            
            # Sort: key metrics first, then alphabetically
            all_metrics = sorted(before_summary, key=lambda x: (
                0 if x['Metric Name'] in key_metrics else 1,
                key_metrics.index(x['Metric Name']) if x['Metric Name'] in key_metrics else 0,
                x['Metric Name']
            ))
            
            for before_item in all_metrics[:15]:  # Show top 15 metrics in executive report
                metric_name = before_item['Metric Name']
                before_val = before_item['Value']
                category = before_item.get('Category', 'Unknown')
                
                # Find corresponding after metric
                after_item = next((m for m in after_summary if m['Metric Name'] == metric_name), None)
                if after_item:
                    after_val = after_item['Value']
                    change = after_val - before_val
                    change_indicator = "✓" if abs(after_val) < abs(before_val) else "✗"
                    
                    report.append(f"| {metric_name} | {category} | {before_val:.4f} | {after_val:.4f} | {change_indicator} {change:+.4f} |\n")
        else:
            # Fallback to old format if comprehensive metrics not available
            before_disp = before_metrics.get('disparities', {})
            after_disp = after_metrics.get('disparities', {})
            
            metrics = [
                ('Selection Rate Disparity', 'selection_rate_diff'),
                ('True Positive Rate Disparity', 'tpr_diff'),
                ('False Positive Rate Disparity', 'fpr_diff')
            ]
            
            report.append("| Metric | Before | After | Improvement |\n")
            report.append("|--------|--------|-------|-------------|\n")
            
            for metric_name, metric_key in metrics:
                before_val = abs(before_disp.get(metric_key, 0))
                after_val = abs(after_disp.get(metric_key, 0))
                improvement = before_val - after_val
                improvement_pct = (improvement / before_val * 100) if before_val > 0 else 0
                
                report.append(f"| {metric_name} | {before_val*100:.1f}% | {after_val*100:.1f}% | ")
            report.append(f"↓ {improvement*100:.1f}% ({improvement_pct:.0f}% reduction) |\n")
        
        report.append("\n")
    
    # Recommendations
    report.append("## 💡 Recommendations\n\n")
    
    if before_metrics and after_metrics:
        report.append("✅ **Mitigation Applied Successfully**\n\n")
        report.append("Next steps:\n")
        report.append("1. Monitor model performance in production\n")
        report.append("2. Conduct regular fairness audits\n")
        report.append("3. Document mitigation decisions for compliance\n\n")
    else:
        recs = analysis_results.get('mitigation_recommendations', [])
        if isinstance(recs, list) and recs:
            report.append("**Recommended Actions:**\n\n")
            for i, rec in enumerate(recs[:3], 1):
                technique = rec.get('technique', '').replace('_', ' ').title()
                priority = rec.get('priority', 'medium').upper()
                plain_lang = rec.get('plain_language', '')
                report.append(f"{i}. **{technique}** (Priority: {priority})\n")
                report.append(f"   - {plain_lang}\n\n")
    
    # Business Impact
    report.append("## 💼 Business Impact\n\n")
    report.append("**Benefits of Fair AI:**\n")
    report.append("- Reduced legal and reputational risk\n")
    report.append("- Improved customer trust and satisfaction\n")
    report.append("- Better decision-making across diverse populations\n")
    report.append("- Compliance with regulatory requirements\n\n")
    
    report.append("---\n\n")
    report.append("*This report provides a high-level overview for executive decision-making. ")
    report.append("For detailed technical analysis, please refer to the Technical Report.*\n")
    
    return "".join(report)


def _generate_technical_report(analysis_results: Dict, before_metrics: Optional[Dict], after_metrics: Optional[Dict]) -> str:
    """Generate detailed technical report."""
    
    report = []
    report.append("# Technical Report: AI Fairness Analysis\n\n")
    report.append(f"**Report Date:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
    report.append(f"**Dataset:** {analysis_results.get('dataset_name', 'N/A')}\n")
    report.append(f"**Target Variable:** {analysis_results.get('target_column', 'N/A')}\n")
    report.append(f"**Protected Attributes:** {', '.join(analysis_results.get('sensitive_attributes', []))}\n\n")
    
    report.append("---\n\n")
    
    # Model Performance
    report.append("## 1. Model Performance\n\n")
    model_perf = analysis_results.get('model_performance', {})
    
    report.append(f"**Model Type:** {model_perf.get('model_type', 'N/A')}\n")
    report.append(f"**Overall Accuracy:** {model_perf.get('accuracy', 0)*100:.2f}%\n\n")
    
    data_summary = analysis_results.get('data_summary', {})
    report.append("**Dataset Statistics:**\n")
    report.append(f"- Total Samples: {data_summary.get('total_samples', 0):,}\n")
    report.append(f"- Training Samples: {data_summary.get('train_samples', 0):,}\n")
    report.append(f"- Test Samples: {data_summary.get('test_samples', 0):,}\n")
    report.append(f"- Feature Count: {data_summary.get('feature_count', 0)}\n\n")
    
    # Fairness Metrics
    report.append("## 2. Comprehensive Fairness Metrics Analysis\n\n")
    
    fairness_analysis = analysis_results.get('fairness_analysis', {})
    for attr, metrics in fairness_analysis.items():
        report.append(f"### Protected Attribute: {attr}\n\n")
        
        # Use comprehensive metrics from summary if available
        summary = metrics.get('summary', [])
        
        if summary:
            # Group metrics by category
            categories = {}
            for metric in summary:
                category = metric.get('Category', 'Other')
                if category not in categories:
                    categories[category] = []
                categories[category].append(metric)
            
            # Display metrics by category
            for category, category_metrics in categories.items():
                report.append(f"**{category} Metrics:**\n\n")
                report.append("| Metric | Value | Ideal | Threshold | Status |\n")
                report.append("|--------|-------|-------|-----------|--------|\n")
                
                for metric in category_metrics:
                    name = metric['Metric Name']
                    value = metric['Value']
                    ideal = metric.get('Ideal Value', 'N/A')
                    threshold = metric.get('Threshold', 'N/A')
                    status = metric.get('Status', 'Unknown')
                    
                    report.append(f"| {name} | {value:.4f} | {ideal} | {threshold} | {status} |\n")
                
                report.append("\n")
        else:
            # Fallback to old format
            disparities = metrics.get('disparities', {})
            
            report.append("**Disparity Metrics:**\n\n")
            report.append("| Metric | Value | Threshold | Status |\n")
            report.append("|--------|-------|-----------|--------|\n")
            
            metric_defs = [
                ('Selection Rate Difference', 'selection_rate_diff', 0.1),
                ('TPR Difference (Equal Opportunity)', 'tpr_diff', 0.1),
                ('FPR Difference (Equalized Odds)', 'fpr_diff', 0.1),
                ('Demographic Parity Ratio', 'demographic_parity_ratio', 0.8)
            ]
            
            for metric_name, metric_key, threshold in metric_defs:
                value = disparities.get(metric_key, 0)
                if 'ratio' in metric_key:
                    status = "✅ Pass" if value >= threshold else "❌ Fail"
                    report.append(f"| {metric_name} | {value:.3f} | ≥{threshold} | {status} |\n")
            else:
                status = "✅ Pass" if abs(value) <= threshold else "❌ Fail"
                report.append(f"| {metric_name} | {abs(value):.3f} | ≤{threshold} | {status} |\n")
        
        report.append("\n")
        
        # Group-specific metrics
        group_metrics = metrics.get('group_metrics', {})
        if group_metrics:
            report.append("**Group-Specific Performance:**\n\n")
            report.append("| Group | Selection Rate | TPR | FPR | Sample Size |\n")
            report.append("|-------|----------------|-----|-----|-------------|\n")
            
            for group, group_data in group_metrics.items():
                sr = group_data.get('selection_rate', 0)
                tpr = group_data.get('tpr', 0)
                fpr = group_data.get('fpr', 0)
                count = group_data.get('count', 0)
                report.append(f"| {group} | {sr:.3f} | {tpr:.3f} | {fpr:.3f} | {count} |\n")
            
            report.append("\n")
    
    # Before/After Comparison
    if before_metrics and after_metrics:
        report.append("## 3. Mitigation Results\n\n")
        
        report.append("**Technique Applied:** ")
        mitigation_info = analysis_results.get('mitigation_info', {})
        report.append(f"{mitigation_info.get('technique', 'N/A')}\n\n")
        
        report.append("**Detailed Comparison:**\n\n")
        
        before_disp = before_metrics.get('disparities', {})
        after_disp = after_metrics.get('disparities', {})
        
        report.append("| Metric | Before | After | Δ | % Change |\n")
        report.append("|--------|--------|-------|---|----------|\n")
        
        for metric_key in ['selection_rate_diff', 'tpr_diff', 'fpr_diff']:
            before_val = abs(before_disp.get(metric_key, 0))
            after_val = abs(after_disp.get(metric_key, 0))
            delta = before_val - after_val
            pct_change = (delta / before_val * 100) if before_val > 0 else 0
            
            metric_name = metric_key.replace('_', ' ').title()
            report.append(f"| {metric_name} | {before_val:.4f} | {after_val:.4f} | ")
            report.append(f"{delta:+.4f} | {pct_change:+.1f}% |\n")
        
        report.append("\n")
    
    # Feature Importance
    explainability = analysis_results.get('explainability', {})
    feat_imp = explainability.get('feature_importance', {})
    
    if feat_imp and feat_imp.get('features'):
        report.append("## 4. Feature Importance\n\n")
        report.append(f"**Method:** {feat_imp.get('type', 'N/A').replace('_', ' ').title()}\n\n")
        
        features = feat_imp.get('features', [])
        importances = feat_imp.get('importances', [])
        
        if features and importances:
            # Sort by importance
            sorted_features = sorted(zip(features, importances), key=lambda x: x[1], reverse=True)[:10]
            
            report.append("**Top 10 Features:**\n\n")
            report.append("| Rank | Feature | Importance |\n")
            report.append("|------|---------|------------|\n")
            
            for i, (feat, imp) in enumerate(sorted_features, 1):
                report.append(f"| {i} | {feat} | {imp:.4f} |\n")
            
            report.append("\n")
    
    # Bias Detection Details
    bias_details = analysis_results.get('bias_detection_details', {})
    if bias_details:
        report.append("## 5. Bias Detection Analysis\n\n")
        
        for attr, details in bias_details.items():
            report.append(f"### {attr}\n\n")
            
            # Representation distribution
            rep_dist = details.get('representation_distribution', {})
            if rep_dist:
                report.append("**Representation Distribution:**\n\n")
                proportions = rep_dist.get('proportions', {})
                counts = rep_dist.get('counts', {})
                
                report.append("| Group | Count | Proportion |\n")
                report.append("|-------|-------|------------|\n")
                
                for group in proportions.keys():
                    count = counts.get(group, 0)
                    prop = proportions.get(group, 0)
                    report.append(f"| {group} | {count} | {prop*100:.1f}% |\n")
                
                report.append("\n")
    
    # Statistical Tests
    report.append("## 6. Statistical Significance\n\n")
    report.append("**Interpretation Guidelines:**\n")
    report.append("- Selection Rate Difference < 0.1: Acceptable disparity\n")
    report.append("- TPR/FPR Difference < 0.1: Fair treatment across groups\n")
    report.append("- Demographic Parity Ratio ≥ 0.8: Meets 80% rule\n\n")
    
    # Recommendations
    report.append("## 7. Technical Recommendations\n\n")
    
    if not (before_metrics and after_metrics):
        recs = analysis_results.get('mitigation_recommendations', [])
        if isinstance(recs, list) and recs:
            for i, rec in enumerate(recs, 1):
                technique = rec.get('technique', '').replace('_', ' ').title()
                rec_type = rec.get('type', 'N/A')
                priority = rec.get('priority', 'medium')
                
                report.append(f"### {i}. {technique} ({rec_type})\n\n")
                report.append(f"**Priority:** {priority.upper()}\n\n")
                report.append(f"**Description:** {rec.get('plain_language', '')}\n\n")
                report.append(f"**Expected Impact:** {rec.get('expected_impact', '')}\n\n")
                report.append(f"**Trade-offs:** {rec.get('trade_offs', '')}\n\n")
                report.append(f"**When to Use:** {rec.get('when_to_use', '')}\n\n")
    else:
        report.append("✅ Mitigation has been applied. Monitor model performance and conduct regular audits.\n\n")
    
    report.append("---\n\n")
    report.append("*This technical report provides detailed metrics and analysis for data scientists and ML engineers.*\n")
    
    return "".join(report)


def _generate_compliance_report(analysis_results: Dict, before_metrics: Optional[Dict], after_metrics: Optional[Dict]) -> str:
    """Generate compliance/audit report."""
    
    report = []
    report.append("# Compliance Report: AI Fairness Audit\n\n")
    report.append(f"**Audit Date:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
    report.append(f"**Dataset:** {analysis_results.get('dataset_name', 'N/A')}\n")
    report.append(f"**Model Type:** {analysis_results.get('model_performance', {}).get('model_type', 'N/A')}\n")
    report.append(f"**Protected Attributes:** {', '.join(analysis_results.get('sensitive_attributes', []))}\n\n")
    
    report.append("---\n\n")
    
    # Regulatory Context
    report.append("## 1. Regulatory Framework\n\n")
    report.append("This audit evaluates compliance with:\n")
    report.append("- **Equal Credit Opportunity Act (ECOA)** - Fair lending practices\n")
    report.append("- **Fair Housing Act (FHA)** - Non-discrimination in housing\n")
    report.append("- **Title VII of Civil Rights Act** - Employment discrimination\n")
    report.append("- **EU AI Act** - High-risk AI systems requirements\n")
    report.append("- **GDPR Article 22** - Automated decision-making\n\n")
    
    # Compliance Assessment
    report.append("## 2. Compliance Assessment\n\n")
    
    fairness_analysis = analysis_results.get('fairness_analysis', {})
    
    for attr, metrics in fairness_analysis.items():
        report.append(f"### Protected Class: {attr}\n\n")
        
        disparities = metrics.get('disparities', {})
        
        # 80% Rule (Adverse Impact)
        dp_ratio = disparities.get('demographic_parity_ratio', 1.0)
        report.append("**Four-Fifths Rule (80% Rule):**\n")
        report.append(f"- Demographic Parity Ratio: {dp_ratio:.3f}\n")
        
        if dp_ratio >= 0.8:
            report.append("- ✅ **COMPLIANT** - Meets 80% threshold\n")
        else:
            report.append("- ❌ **NON-COMPLIANT** - Below 80% threshold (adverse impact detected)\n")
        
        report.append("\n")
        
        # Equal Opportunity
        tpr_diff = abs(disparities.get('tpr_diff', 0))
        report.append("**Equal Opportunity:**\n")
        report.append(f"- TPR Difference: {tpr_diff:.3f}\n")
        
        if tpr_diff <= 0.1:
            report.append("- ✅ **COMPLIANT** - Equal opportunity maintained\n")
        else:
            report.append("- ⚠️ **REVIEW REQUIRED** - Significant TPR disparity\n")
        
        report.append("\n")
        
        # Equalized Odds
        fpr_diff = abs(disparities.get('fpr_diff', 0))
        report.append("**Equalized Odds:**\n")
        report.append(f"- FPR Difference: {fpr_diff:.3f}\n")
        
        if fpr_diff <= 0.1:
            report.append("- ✅ **COMPLIANT** - Equalized odds maintained\n")
        else:
            report.append("- ⚠️ **REVIEW REQUIRED** - Significant FPR disparity\n")
        
        report.append("\n")
    
    # Mitigation Actions
    report.append("## 3. Mitigation Actions\n\n")
    
    if before_metrics and after_metrics:
        report.append("**Status:** ✅ Mitigation Applied\n\n")
        
        mitigation_info = analysis_results.get('mitigation_info', {})
        report.append(f"**Technique:** {mitigation_info.get('technique', 'N/A')}\n")
        report.append(f"**Date Applied:** {datetime.now().strftime('%Y-%m-%d')}\n\n")
        
        report.append("**Impact Assessment:**\n\n")
        
        before_disp = before_metrics.get('disparities', {})
        after_disp = after_metrics.get('disparities', {})
        
        report.append("| Metric | Before | After | Status |\n")
        report.append("|--------|--------|-------|--------|\n")
        
        # 80% Rule
        before_dp = before_disp.get('demographic_parity_ratio', 1.0)
        after_dp = after_disp.get('demographic_parity_ratio', 1.0)
        dp_status = "✅ Compliant" if after_dp >= 0.8 else "❌ Non-compliant"
        report.append(f"| 80% Rule | {before_dp:.3f} | {after_dp:.3f} | {dp_status} |\n")
        
        # Equal Opportunity
        before_tpr = abs(before_disp.get('tpr_diff', 0))
        after_tpr = abs(after_disp.get('tpr_diff', 0))
        tpr_status = "✅ Improved" if after_tpr < before_tpr else "→ No change"
        report.append(f"| Equal Opportunity | {before_tpr:.3f} | {after_tpr:.3f} | {tpr_status} |\n")
        
        # Equalized Odds
        before_fpr = abs(before_disp.get('fpr_diff', 0))
        after_fpr = abs(after_disp.get('fpr_diff', 0))
        fpr_status = "✅ Improved" if after_fpr < before_fpr else "→ No change"
        report.append(f"| Equalized Odds | {before_fpr:.3f} | {after_fpr:.3f} | {fpr_status} |\n")
        
        report.append("\n")
    else:
        report.append("**Status:** ⚠️ No Mitigation Applied\n\n")
        report.append("**Recommended Actions:**\n\n")
        
        recs = analysis_results.get('mitigation_recommendations', [])
        if isinstance(recs, list) and recs:
            for i, rec in enumerate(recs[:3], 1):
                technique = rec.get('technique', '').replace('_', ' ').title()
                priority = rec.get('priority', 'medium')
                report.append(f"{i}. **{technique}** (Priority: {priority.upper()})\n")
                report.append(f"   - Rationale: {rec.get('plain_language', '')}\n\n")
    
    # Documentation
    report.append("## 4. Documentation & Audit Trail\n\n")
    
    report.append("**Model Information:**\n")
    model_perf = analysis_results.get('model_performance', {})
    report.append(f"- Model Type: {model_perf.get('model_type', 'N/A')}\n")
    report.append(f"- Training Date: {datetime.now().strftime('%Y-%m-%d')}\n")
    report.append(f"- Model Accuracy: {model_perf.get('accuracy', 0)*100:.2f}%\n")
    report.append(f"- Features Used: {model_perf.get('feature_columns', [])}\n\n")
    
    report.append("**Data Information:**\n")
    data_summary = analysis_results.get('data_summary', {})
    report.append(f"- Total Records: {data_summary.get('total_samples', 0):,}\n")
    report.append(f"- Training Set: {data_summary.get('train_samples', 0):,}\n")
    report.append(f"- Test Set: {data_summary.get('test_samples', 0):,}\n\n")
    
    # Risk Assessment
    report.append("## 5. Risk Assessment\n\n")
    
    # Calculate overall risk
    max_disparity = 0
    for attr, metrics in fairness_analysis.items():
        disparities = metrics.get('disparities', {})
        max_disparity = max(
            max_disparity,
            abs(disparities.get('selection_rate_diff', 0)),
            abs(disparities.get('tpr_diff', 0)),
            abs(disparities.get('fpr_diff', 0))
        )
    
    if max_disparity > 0.2:
        risk_level = "🔴 HIGH"
        risk_desc = "Immediate action required to address significant disparities"
        actions = [
            "Apply bias mitigation techniques immediately",
            "Conduct thorough review of training data",
            "Consider model redesign or alternative approaches",
            "Consult legal counsel regarding deployment"
        ]
    elif max_disparity > 0.1:
        risk_level = "🟡 MODERATE"
        risk_desc = "Mitigation recommended before production deployment"
        actions = [
            "Apply appropriate bias mitigation techniques",
            "Monitor model performance closely",
            "Document mitigation efforts",
            "Establish regular audit schedule"
        ]
    else:
        risk_level = "🟢 LOW"
        risk_desc = "Model meets fairness thresholds"
        actions = [
            "Maintain current monitoring practices",
            "Conduct periodic fairness audits",
            "Document compliance procedures",
            "Update as regulations evolve"
        ]
    
    report.append(f"**Overall Risk Level:** {risk_level}\n\n")
    report.append(f"**Assessment:** {risk_desc}\n\n")
    report.append("**Required Actions:**\n")
    for action in actions:
        report.append(f"- {action}\n")
    report.append("\n")
    
    # Certification
    report.append("## 6. Audit Certification\n\n")
    report.append("This report certifies that:\n\n")
    report.append("1. ✅ Fairness analysis was conducted using industry-standard metrics\n")
    report.append("2. ✅ Protected attributes were properly identified and analyzed\n")
    report.append("3. ✅ Disparities were measured against regulatory thresholds\n")
    
    if before_metrics and after_metrics:
        report.append("4. ✅ Bias mitigation was applied and documented\n")
        report.append("5. ✅ Post-mitigation metrics show improvement\n")
    else:
        report.append("4. ⚠️ Bias mitigation pending (see recommendations)\n")
        report.append("5. ⏳ Post-mitigation audit required after implementation\n")
    
    report.append("\n")
    
    report.append("---\n\n")
    report.append(f"**Auditor:** AI Fairness & Explainability Toolkit\n")
    report.append(f"**Report ID:** {datetime.now().strftime('%Y%m%d-%H%M%S')}\n")
    report.append(f"**Next Audit Due:** {datetime.now().strftime('%Y-%m-%d')} + 90 days\n\n")
    
    report.append("*This compliance report provides documentation for regulatory audits and legal review.*\n")
    
    return "".join(report)

