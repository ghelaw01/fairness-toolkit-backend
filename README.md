# AI Fairness Toolkit - Backend

A comprehensive backend API for analyzing machine learning model fairness, detecting bias, and providing explainability insights.

## Features

- **File Upload & Data Processing**: Upload CSV datasets for fairness analysis
- **Demo Data**: Pre-loaded COMPAS dataset for quick testing
- **Fairness Metrics**: Calculate demographic parity, equal opportunity, and disparate impact
- **Bias Detection**: Identify bias across protected attributes (race, gender, age)
- **Model Explainability**: SHAP and LIME explanations for model predictions
- **Mitigation Recommendations**: Get actionable suggestions to reduce bias

## Quick Start

### Prerequisites

- Python 3.8 or higher
- pip package manager

### Installation

1. Clone the repository:
```bash
git clone https://github.com/ghelaw01/fairness-toolkit-backend.git
cd fairness-toolkit-backend
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Run the server:
```bash
python main.py
```

The backend API will be available at `http://localhost:5000`

## API Endpoints

### Upload Dataset
```
POST /api/upload
Content-Type: multipart/form-data
Body: file (CSV file)
```

### Load Demo Data
```
GET /api/demo
```

### Analyze Fairness
```
POST /api/analyze
Content-Type: application/json
Body: {
  "target_column": "target_variable",
  "protected_attribute": "race"
}
```

### Get Bias Detection
```
POST /api/bias
Content-Type: application/json
Body: {
  "protected_attributes": ["race", "gender"]
}
```

### Get Explanations
```
POST /api/explain
Content-Type: application/json
Body: {
  "sample_index": 0
}
```

## Project Structure

```
backend/
├── main.py                 # Flask application entry point
├── requirements.txt        # Python dependencies
├── data/                   # Demo datasets
│   └── compas-scores-two-years.csv
├── src/
│   ├── routes/
│   │   ├── fairness_api.py           # Main API routes
│   │   └── fairness_api_recommend.py # Recommendation endpoints
│   ├── models/
│   │   └── user.py                   # User model
│   ├── bias_detection.py             # Bias detection logic
│   ├── data_processor.py             # Data processing utilities
│   ├── explainability.py             # SHAP/LIME explanations
│   └── fairness_metrics.py           # Fairness calculations
└── static/                 # Frontend build files
```

## Dependencies

- Flask: Web framework
- pandas: Data manipulation
- numpy: Numerical computing
- scikit-learn: Machine learning
- shap: Model explainability
- lime: Local interpretable explanations
- flask-cors: Cross-origin resource sharing

## Environment Variables

- `PORT`: Server port (default: 5000)
- `FLASK_ENV`: Environment mode (development/production)

## Development

To run in development mode with auto-reload:
```bash
export FLASK_ENV=development
python main.py
```

## Deployment

### Deploy to Render

1. Push code to GitHub
2. Create a new Web Service on Render
3. Connect your GitHub repository
4. Set build command: `pip install -r requirements.txt`
5. Set start command: `python main.py`
6. Deploy!

## Testing

Test the API using curl:

```bash
# Load demo data
curl http://localhost:5000/api/demo

# Analyze fairness
curl -X POST http://localhost:5000/api/analyze \
  -H "Content-Type: application/json" \
  -d '{"target_column": "two_year_recid", "protected_attribute": "race"}'
```

## License

MIT License

## Support

For issues and questions, please open an issue on GitHub.
