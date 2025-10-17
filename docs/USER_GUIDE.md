# AI Fairness Toolkit - User Guide

**Welcome!** This guide will walk you through using the AI Fairness Toolkit, even if you don’t have a technical background. Our goal is to make fairness analysis accessible to everyone.

---

## 1. What is This Toolkit?

This toolkit helps you understand if your AI models are fair. It answers questions like:
- Is my model treating different groups of people equally?
- Where is the bias coming from?
- How can I fix the bias?

## 2. Getting Started: A 5-Minute Walkthrough

### Step 1: Upload Your Data

1.  **Go to the application:** [https://fairness-toolkit-frontend.onrender.com](https://fairness-toolkit-frontend.onrender.com)
2.  **Upload a CSV file** with your model’s data.
3.  **Or, click “Load Demo”** to use our pre-loaded COMPAS dataset.

### Step 2: Configure Your Analysis

-   **Target Column:** What are you trying to predict? (e.g., “two_year_recid”)
-   **Sensitive Attributes:** What groups do you want to compare? (e.g., “race”, “sex”)

### Step 3: Run the Analysis

-   Click **“Run Fairness Analysis”**. The toolkit will analyze your model and data.

## 3. Understanding Your Results

### Fairness Tab: Is My Model Fair?

This tab shows you fairness metrics with a simple traffic light system:

-   🟢 **Green (Fair):** No significant bias detected.
-   🟡 **Yellow (Moderate Concern):** Some bias detected. Review recommended.
-   🔴 **Red (High Risk):** Significant bias detected. Action needed.

Use the **Help Tooltips (❓)** next to each metric for plain-language explanations.

### Bias Tab: Where is the Bias?

This tab visualizes bias across different groups, helping you pinpoint where the unfairness is coming from.

### Explain Tab: Why is the Model Making These Decisions?

-   **SHAP:** Shows which features have the biggest impact on decisions.
-   **LIME:** Explains individual predictions.
-   **Counterfactuals:** Shows “what-if” scenarios (e.g., “What would need to change for this person to get approved?”).

### Mitigation Tab: How Can I Fix the Bias?

-   Click **“Get Recommendations”** to see suggested techniques.
-   The advisor will recommend techniques with high impact and low trade-offs.
-   Click **“Apply This Technique”** to fix the bias.
-   See a **before/after comparison** to measure the improvement.

### Reports Tab: How Do I Share My Findings?

-   **Generate reports** for different audiences:
    -   **Executive Summary:** High-level overview for leadership.
    -   **Technical Report:** Detailed metrics for data scientists.
    -   **Compliance Report:** Audit trail for regulators.
-   **Export data** in CSV, Excel, or JSON format.

## 4. Frequently Asked Questions (FAQ)

**Q: The app is slow on first load. Why?**

A: Our backend service sleeps after 15 minutes of inactivity (it’s on a free plan). The first request takes 30-60 seconds to wake it up. This is normal!

**Q: What do the fairness metrics mean?**

A: Use the **Help Tooltips (❓)** next to each metric for a plain-language explanation and an example.

**Q: How do I know which mitigation technique to use?**

A: The **Mitigation Advisor** will recommend the best techniques for your specific situation. It considers the type of bias, expected impact, and potential trade-offs.

**Q: Can I use this with my own data?**

A: Yes! Just upload a CSV file with your data. Ensure it has a target column and sensitive attributes.

## 5. Glossary of Terms

-   **Fairness Metric:** A number that measures how fair your model is.
-   **Bias:** When your model treats different groups of people unfairly.
-   **Mitigation:** Techniques to reduce or remove bias.
-   **Explainability:** Understanding why your model makes certain decisions.
-   **Counterfactual:** A “what-if” scenario.

---

*If you have any other questions, please contact your administrator.*

