# Spare Part Demand Forecasting API

API documentation for the Azure ML endpoint providing demand forecasts.

## Base URL

```
https://spare-part-forecast.eastus.inference.ml.azure.com
```

## Authentication

All requests require Bearer token authentication:

```http
Authorization: Bearer YOUR_API_KEY
Content-Type: application/json
```

Get your API key from Azure ML Studio → Endpoints → spare-part-forecast → Consume.

---

## Endpoints

### POST /score

Generate demand forecasts using Prophet or XGBoost models.

---

## Request Examples

### 1. Prophet Forecast (Time Series)

Predict demand for the next N days:

```bash
curl -X POST "https://spare-part-forecast.eastus.inference.ml.azure.com/score" \
  -H "Authorization: Bearer YOUR_API_KEY" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "prophet",
    "periods": 7
  }'
```

**Response:**
```json
{
  "status": "success",
  "model": "prophet",
  "predictions": [
    {"date": "2024-01-15", "yhat": 5014.87, "yhat_lower": 3498.03, "yhat_upper": 6435.15},
    {"date": "2024-01-16", "yhat": 5052.79, "yhat_lower": 3506.00, "yhat_upper": 6514.64}
  ],
  "timestamp": "2024-01-05T10:30:00"
}
```

### 2. Prophet with Specific Dates

Predict for specific dates:

```bash
curl -X POST "https://spare-part-forecast.eastus.inference.ml.azure.com/score" \
  -H "Authorization: Bearer YOUR_API_KEY" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "prophet",
    "dates": ["2024-02-01", "2024-02-15", "2024-03-01"]
  }'
```

### 3. XGBoost Prediction (ML Model)

Predict demand based on features:

```bash
curl -X POST "https://spare-part-forecast.eastus.inference.ml.azure.com/score" \
  -H "Authorization: Bearer YOUR_API_KEY" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "xgboost",
    "features": {
      "day_of_week": 1,
      "month": 6,
      "day_of_month": 15,
      "quarter": 2,
      "year": 2024,
      "week_of_year": 24,
      "is_weekend": 0,
      "is_month_start": 0,
      "is_month_end": 0,
      "lag_1": 45,
      "lag_7": 50,
      "lag_14": 48,
      "lag_30": 52,
      "rolling_mean_7": 45,
      "rolling_mean_14": 47,
      "rolling_mean_30": 49,
      "rolling_std_7": 10,
      "rolling_std_14": 12,
      "rolling_std_30": 15
    }
  }'
```

**Response:**
```json
{
  "status": "success",
  "model": "xgboost",
  "predictions": [{"prediction": 9351.75}],
  "timestamp": "2024-01-05T10:30:00"
}
```

### 4. Batch XGBoost Predictions

Predict multiple samples at once:

```bash
curl -X POST "https://spare-part-forecast.eastus.inference.ml.azure.com/score" \
  -H "Authorization: Bearer YOUR_API_KEY" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "xgboost",
    "features": [
      {"day_of_week": 1, "month": 6, ...},
      {"day_of_week": 2, "month": 6, ...}
    ]
  }'
```

### 5. Health Check

Check if models are loaded:

```bash
curl -X POST "https://spare-part-forecast.eastus.inference.ml.azure.com/score" \
  -H "Authorization: Bearer YOUR_API_KEY" \
  -H "Content-Type: application/json" \
  -d '{"health_check": true}'
```

**Response:**
```json
{
  "status": "healthy",
  "models": {"prophet": true, "xgboost": true},
  "timestamp": "2024-01-05T10:30:00"
}
```

---

## XGBoost Feature Reference

| Feature | Type | Description | Range |
|---------|------|-------------|-------|
| `day_of_week` | int | Day of week (Monday=0) | 0-6 |
| `month` | int | Month | 1-12 |
| `day_of_month` | int | Day of month | 1-31 |
| `quarter` | int | Quarter | 1-4 |
| `year` | int | Year | 2020+ |
| `week_of_year` | int | Week number | 1-53 |
| `is_weekend` | int | Weekend flag | 0, 1 |
| `is_month_start` | int | First day of month | 0, 1 |
| `is_month_end` | int | Last day of month | 0, 1 |
| `lag_1` | float | Demand 1 day ago | ≥0 |
| `lag_7` | float | Demand 7 days ago | ≥0 |
| `lag_14` | float | Demand 14 days ago | ≥0 |
| `lag_30` | float | Demand 30 days ago | ≥0 |
| `rolling_mean_7` | float | 7-day rolling average | ≥0 |
| `rolling_mean_14` | float | 14-day rolling average | ≥0 |
| `rolling_mean_30` | float | 30-day rolling average | ≥0 |
| `rolling_std_7` | float | 7-day rolling std dev | ≥0 |
| `rolling_std_14` | float | 14-day rolling std dev | ≥0 |
| `rolling_std_30` | float | 30-day rolling std dev | ≥0 |

---

## Error Responses

### 400 Bad Request

```json
{
  "status": "error",
  "message": "Missing required features: {'lag_1', 'lag_7'}",
  "timestamp": "2024-01-05T10:30:00"
}
```

### 401 Unauthorized

Missing or invalid API key.

### 500 Server Error

```json
{
  "status": "error",
  "message": "Internal prediction error",
  "timestamp": "2024-01-05T10:30:00"
}
```

---

## Python SDK Example

```python
import requests

class DemandForecastClient:
    def __init__(self, endpoint_url, api_key):
        self.endpoint_url = endpoint_url
        self.headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json"
        }
    
    def forecast_prophet(self, periods=7):
        response = requests.post(
            f"{self.endpoint_url}/score",
            json={"model": "prophet", "periods": periods},
            headers=self.headers
        )
        return response.json()
    
    def predict_xgboost(self, features):
        response = requests.post(
            f"{self.endpoint_url}/score",
            json={"model": "xgboost", "features": features},
            headers=self.headers
        )
        return response.json()

# Usage
client = DemandForecastClient(
    "https://spare-part-forecast.eastus.inference.ml.azure.com",
    "YOUR_API_KEY"
)

# Get 7-day forecast
forecast = client.forecast_prophet(periods=7)
print(forecast)
```

---

## Rate Limits

- **Requests per second**: 100
- **Max request size**: 1 MB
- **Max batch size**: 1000 samples

---

## Support

- **OpenAPI Spec**: [openapi.json](./openapi.json)
- **Issues**: [GitHub Issues](https://github.com/shan31/spare-part-demand-forecasting/issues)
