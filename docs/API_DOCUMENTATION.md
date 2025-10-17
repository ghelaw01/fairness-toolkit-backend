# AI Fairness Toolkit - API Documentation

**Version:** 1.0  
**Base URL:** `/api/fairness`

---

## 1. Overview

This API provides a comprehensive suite of tools for fairness analysis, bias mitigation, explainability, and reporting. All endpoints are secured and require proper authentication.

## 2. Authentication

All API requests must include an `Authorization` header with a valid Bearer token.

`Authorization: Bearer <YOUR_API_KEY>`

## 3. Endpoints

### 3.1 Data Management

**POST /upload**
-   Upload a CSV file for analysis.
-   **Request Body:** `multipart/form-data` with `file` field.
-   **Response:** `{ "data_info": { "columns": [...], "total_samples": ... } }`

**GET /datasets/compas**
-   Load the demo COMPAS dataset.
-   **Response:** `{ "data_info": { ... } }`

### 3.2 Fairness Analysis

**POST /analyze**
-   Run a comprehensive fairness analysis.
-   **Request Body:**
    ```json
    {
      "target_column": "two_year_recid",
      "sensitive_attributes": ["race", "sex"],
      "model_type": "random_forest"
    }
    ```
-   **Response:** `{ "results": { ... } }`

### 3.3 Explainability

**POST /explain/shap**
-   Generate SHAP explanations (global and local).
-   **Response:** `{ "global_importance": [...], "local_explanations": [...] }`

**POST /explain/lime**
-   Generate LIME explanation for a single instance.
-   **Request Body:** `{ "instance_index": 123 }`
-   **Response:** `{ "lime_explanation": { ... } }`

**POST /explain/counterfactual**
-   Generate counterfactual explanations.
-   **Request Body:** `{ "instance_index": 123, "desired_outcome": 1 }`
-   **Response:** `{ "counterfactuals": [...] }`

### 3.4 Bias Mitigation

**POST /mitigate/recommend**
-   Get recommended mitigation techniques.
-   **Request Body:** `{ "fairness_metrics": { ... } }`
-   **Response:** `{ "recommendations": [...] }`

**POST /mitigate/apply**
-   Apply a specific mitigation technique.
-   **Request Body:** `{ "technique": "reweighing" }`
-   **Response:** `{ "mitigation_results": { ... } }`

### 3.5 Reporting & Exporting

**POST /report/generate**
-   Generate a stakeholder-specific report.
-   **Request Body:** `{ "stakeholder_type": "executive" }`
-   **Response:** `{ "report": "..." }` (Markdown format)

**POST /export/{format}**
-   Export results in a specific format (`csv`, `excel`, `json`).
-   **Response:** File download.

### 3.6 Privacy & Governance

**GET /audit/trail**
-   Retrieve the complete audit trail.
-   **Response:** `application/jsonl` stream.

**POST /anonymize**
-   Anonymize a dataset.
-   **Request Body:** `{ "dataset_id": "...", "method": "generalization" }`
-   **Response:** `{ "anonymized_dataset_id": "..." }`

## 4. Error Handling

The API uses standard HTTP status codes:
-   `200 OK`: Request successful.
-   `201 Created`: Resource created successfully.
-   `400 Bad Request`: Invalid request body or parameters.
-   `401 Unauthorized`: Invalid or missing API key.
-   `404 Not Found`: Resource not found.
-   `500 Internal Server Error`: An unexpected error occurred.

Error responses include a JSON body with details:
```json
{
  "error": "Invalid input",
  "details": "'target_column' is required"
}
```

