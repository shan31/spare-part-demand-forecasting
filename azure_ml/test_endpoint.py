import requests
import json
import time
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Get endpoint URL and key from environment variables
endpoint_url = os.getenv("AZURE_ML_ENDPOINT_URL", "https://spare-part-forecast.eastus.inference.ml.azure.com/score")
api_key = os.getenv("AZURE_ML_API_KEY", "")

if not api_key:
    print("WARNING: AZURE_ML_API_KEY not set in environment!")
    print("Please add AZURE_ML_API_KEY to your .env file")


# Test Prophet prediction - matches score.py expected format
prophet_data = {
    "model": "prophet",
    "periods": 7
}

# Test XGBoost prediction - matches score.py expected format
# All features required by the trained XGBoost model
xgboost_data = {
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
}

# Use XGBoost for this test
data = xgboost_data

headers = {
    "Content-Type": "application/json",
    "Authorization": f"Bearer {api_key}"
}

# Measure response time
start_time = time.time()
response = requests.post(endpoint_url, json=data, headers=headers)
end_time = time.time()
response_time_ms = (end_time - start_time) * 1000

# Debug output
print(f"\n{'='*50}")
print(f"RESPONSE TIME: {response_time_ms:.2f} ms")
print(f"Server Execution Time: {response.headers.get('x-ms-run-fn-exec-ms', 'N/A')} ms")
print(f"{'='*50}\n")
print(f"Status Code: {response.status_code}")

# Parse JSON if available
if response.status_code == 200 and response.text:
    try:
        print(f"JSON Response: {response.json()}")
    except:
        print("Could not parse JSON response")
else:
    print(f"\nError: {response.status_code}")
    if response.status_code == 404:
        print("Endpoint not found - check if deployment is complete")
    elif response.status_code == 401:
        print("Unauthorized - check your API key")
    elif response.status_code == 500:
        print("Server error - check scoring script logs")