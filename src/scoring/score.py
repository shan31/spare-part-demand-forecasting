"""
Azure ML Scoring Script
Entry point for Azure ML managed endpoint
"""

import os
import json
import pickle
import logging
import pandas as pd
import numpy as np
from pathlib import Path

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def init():
    """
    Initialize the model.
    Called once when the endpoint starts.
    """
    global prophet_model, xgboost_model
    
    # Get model path from environment
    model_dir = os.getenv('AZUREML_MODEL_DIR', 'models')
    
    prophet_path = Path(model_dir) / 'prophet_model.pkl'
    xgboost_path = Path(model_dir) / 'xgboost_model.pkl'
    
    # Load models
    prophet_model = None
    xgboost_model = None
    
    if prophet_path.exists():
        with open(prophet_path, 'rb') as f:
            prophet_model = pickle.load(f)
        logger.info("Prophet model loaded successfully")
    
    if xgboost_path.exists():
        with open(xgboost_path, 'rb') as f:
            xgboost_model = pickle.load(f)
        logger.info("XGBoost model loaded successfully")
    
    if prophet_model is None and xgboost_model is None:
        logger.warning("No models found. Using sample predictions.")


def run(raw_data: str) -> str:
    """
    Run inference on input data.
    
    Expected input format:
    {
        "model": "prophet" or "xgboost",
        "periods": 30,  # for prophet
        "features": {...},  # for xgboost
        "dates": ["2024-01-01", ...]  # optional, for prophet
    }
    """
    try:
        # Parse input
        data = json.loads(raw_data)
        model_type = data.get('model', 'prophet').lower()
        
        if model_type == 'prophet':
            result = predict_prophet(data)
        elif model_type == 'xgboost':
            result = predict_xgboost(data)
        else:
            raise ValueError(f"Unknown model type: {model_type}")
        
        return json.dumps({
            'status': 'success',
            'predictions': result
        })
    
    except Exception as e:
        logger.error(f"Prediction error: {str(e)}")
        return json.dumps({
            'status': 'error',
            'message': str(e)
        })


def predict_prophet(data: dict) -> list:
    """Generate Prophet predictions."""
    global prophet_model
    
    if prophet_model is None:
        # Return sample predictions if model not loaded
        periods = data.get('periods', 30)
        return [{'date': f'day_{i}', 'prediction': float(np.random.randint(10, 100))} for i in range(periods)]
    
    periods = data.get('periods', 30)
    dates = data.get('dates', None)
    
    if dates:
        # Predict for specific dates
        dates_df = pd.DataFrame({'ds': pd.to_datetime(dates)})
        forecast = prophet_model.predict(dates_df)
    else:
        # Predict future periods
        future = prophet_model.make_future_dataframe(periods=periods, include_history=False)
        forecast = prophet_model.predict(future)
    
    # Format output
    result = forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].to_dict('records')
    
    # Convert timestamps to strings
    for r in result:
        r['ds'] = r['ds'].strftime('%Y-%m-%d')
    
    return result


def predict_xgboost(data: dict) -> list:
    """Generate XGBoost predictions."""
    global xgboost_model
    
    features = data.get('features', None)
    
    if features is None:
        raise ValueError("Features required for XGBoost prediction")
    
    if xgboost_model is None:
        # Return sample predictions if model not loaded
        n_samples = len(features) if isinstance(features, list) else 1
        return [{'prediction': float(np.random.randint(10, 100))} for _ in range(n_samples)]
    
    # Convert features to DataFrame
    if isinstance(features, dict):
        X = pd.DataFrame([features])
    else:
        X = pd.DataFrame(features)
    
    # Get model and feature names
    model = xgboost_model['model'] if isinstance(xgboost_model, dict) else xgboost_model
    feature_names = xgboost_model.get('feature_names', X.columns.tolist()) if isinstance(xgboost_model, dict) else X.columns.tolist()
    
    # Ensure correct column order
    X = X[feature_names]
    
    # Predict
    predictions = model.predict(X)
    
    result = [{'prediction': float(p)} for p in predictions]
    return result


# For local testing
if __name__ == "__main__":
    init()
    
    # Test Prophet prediction
    test_input = json.dumps({
        'model': 'prophet',
        'periods': 7
    })
    
    result = run(test_input)
    print("Prophet test:", result)
    
    # Test XGBoost prediction
    test_input = json.dumps({
        'model': 'xgboost',
        'features': {
            'day_of_week': 1,
            'month': 6,
            'lag_7': 50,
            'rolling_mean_7': 45
        }
    })
    
    result = run(test_input)
    print("XGBoost test:", result)
