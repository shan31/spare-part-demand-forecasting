"""
Azure ML Scoring Script
Entry point for Azure ML managed endpoint for Spare Part Demand Forecasting
"""

import os
import json
import pickle
import logging
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime, timedelta

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Global model variables
prophet_model = None
xgboost_model = None
model_loaded = False


def init():
    """
    Initialize the models.
    Called once when the endpoint container starts.
    """
    global prophet_model, xgboost_model, model_loaded
    
    # Get model directory from Azure ML environment
    # AZUREML_MODEL_DIR points to the registered model location
    model_dir = os.getenv('AZUREML_MODEL_DIR', '')
    
    logger.info(f"AZUREML_MODEL_DIR: {model_dir}")
    logger.info(f"Current working directory: {os.getcwd()}")
    
    # List all files in model directory for debugging
    if model_dir and os.path.exists(model_dir):
        logger.info(f"Files in model directory: {os.listdir(model_dir)}")
        
        # Check for nested directories (Azure ML sometimes nests models)
        for root, dirs, files in os.walk(model_dir):
            logger.info(f"Walking: {root}, dirs: {dirs}, files: {files}")
    
    # Define potential model paths
    search_paths = [
        Path(model_dir),
        Path(model_dir) / 'models',
        Path('models'),
        Path('.') / 'models',
    ]
    
    # Try to find and load Prophet model
    prophet_model = None
    for base_path in search_paths:
        prophet_path = base_path / 'prophet_model.pkl'
        if prophet_path.exists():
            try:
                with open(prophet_path, 'rb') as f:
                    prophet_model = pickle.load(f)
                logger.info(f"Prophet model loaded from: {prophet_path}")
                break
            except Exception as e:
                logger.error(f"Failed to load Prophet from {prophet_path}: {e}")
    
    # Try to find and load XGBoost model
    xgboost_model = None
    for base_path in search_paths:
        xgboost_path = base_path / 'xgboost_model.pkl'
        if xgboost_path.exists():
            try:
                with open(xgboost_path, 'rb') as f:
                    xgboost_model = pickle.load(f)
                logger.info(f"XGBoost model loaded from: {xgboost_path}")
                break
            except Exception as e:
                logger.error(f"Failed to load XGBoost from {xgboost_path}: {e}")
    
    # Log model loading status
    model_loaded = prophet_model is not None or xgboost_model is not None
    
    if model_loaded:
        logger.info("Model initialization complete!")
        logger.info(f"  Prophet model: {'Loaded' if prophet_model else 'Not found'}")
        logger.info(f"  XGBoost model: {'Loaded' if xgboost_model else 'Not found'}")
    else:
        logger.warning("No models found! Endpoint will return sample predictions.")
        logger.warning("Ensure models are registered and deployed correctly.")


def run(raw_data: str) -> str:
    """
    Run inference on input data.
    
    Expected input formats:
    
    1. Prophet forecast:
    {
        "model": "prophet",
        "periods": 30,              # Number of days to forecast
        "dates": ["2024-01-01", ...]  # Optional: specific dates to predict
    }
    
    2. XGBoost prediction:
    {
        "model": "xgboost",
        "features": {
            "day_of_week": 1,
            "month": 6,
            "lag_7": 50,
            "rolling_mean_7": 45,
            "rolling_std_7": 10,
            "day_of_month": 15,
            "quarter": 2,
            "is_weekend": 0
        }
    }
    
    3. Health check:
    {
        "health_check": true
    }
    """
    try:
        logger.info(f"Received request: {raw_data[:200]}...")  # Log first 200 chars
        
        # Parse input
        data = json.loads(raw_data)
        
        # Health check endpoint
        if data.get('health_check', False):
            return json.dumps({
                'status': 'healthy',
                'models': {
                    'prophet': prophet_model is not None,
                    'xgboost': xgboost_model is not None
                },
                'timestamp': datetime.now().isoformat()
            })
        
        # Get model type (default to prophet)
        model_type = data.get('model', 'prophet').lower()
        
        if model_type == 'prophet':
            result = predict_prophet(data)
        elif model_type == 'xgboost':
            result = predict_xgboost(data)
        else:
            raise ValueError(f"Unknown model type: {model_type}. Use 'prophet' or 'xgboost'.")
        
        response = {
            'status': 'success',
            'model': model_type,
            'predictions': result,
            'timestamp': datetime.now().isoformat()
        }
        
        logger.info(f"Response: {len(result)} predictions generated")
        return json.dumps(response)
    
    except json.JSONDecodeError as e:
        logger.error(f"Invalid JSON input: {e}")
        return json.dumps({
            'status': 'error',
            'message': f'Invalid JSON input: {str(e)}',
            'timestamp': datetime.now().isoformat()
        })
    
    except Exception as e:
        logger.error(f"Prediction error: {str(e)}", exc_info=True)
        return json.dumps({
            'status': 'error',
            'message': str(e),
            'timestamp': datetime.now().isoformat()
        })


def predict_prophet(data: dict) -> list:
    """Generate Prophet forecast predictions."""
    global prophet_model
    
    periods = data.get('periods', 30)
    dates = data.get('dates', None)
    
    # If model not loaded, return sample predictions
    if prophet_model is None:
        logger.warning("Prophet model not loaded, returning sample predictions")
        base_date = datetime.now()
        return [
            {
                'date': (base_date + timedelta(days=i)).strftime('%Y-%m-%d'),
                'yhat': float(np.random.randint(10, 100)),
                'yhat_lower': float(np.random.randint(5, 50)),
                'yhat_upper': float(np.random.randint(50, 150))
            }
            for i in range(periods)
        ]
    
    # Use actual Prophet model
    if dates:
        # Predict for specific dates
        dates_df = pd.DataFrame({'ds': pd.to_datetime(dates)})
        forecast = prophet_model.predict(dates_df)
    else:
        # Predict future periods
        future = prophet_model.make_future_dataframe(periods=periods, include_history=False)
        forecast = prophet_model.predict(future)
    
    # Format output
    result = []
    for _, row in forecast.iterrows():
        result.append({
            'date': row['ds'].strftime('%Y-%m-%d'),
            'yhat': round(float(row['yhat']), 2),
            'yhat_lower': round(float(row['yhat_lower']), 2),
            'yhat_upper': round(float(row['yhat_upper']), 2)
        })
    
    return result


def predict_xgboost(data: dict) -> list:
    """Generate XGBoost predictions."""
    global xgboost_model
    
    features = data.get('features', None)
    
    if features is None:
        raise ValueError("'features' field is required for XGBoost prediction")
    
    # If model not loaded, return sample predictions
    if xgboost_model is None:
        logger.warning("XGBoost model not loaded, returning sample predictions")
        n_samples = len(features) if isinstance(features, list) else 1
        return [
            {'prediction': float(np.random.randint(10, 100))}
            for _ in range(n_samples)
        ]
    
    # Convert features to DataFrame
    if isinstance(features, dict):
        X = pd.DataFrame([features])
    elif isinstance(features, list):
        X = pd.DataFrame(features)
    else:
        raise ValueError("'features' must be a dict (single prediction) or list of dicts (batch)")
    
    # Handle model that might be stored as dict or object
    if isinstance(xgboost_model, dict):
        model = xgboost_model.get('model', xgboost_model)
        feature_names = xgboost_model.get('feature_names', None)
        
        # Reorder columns if feature names are available
        if feature_names:
            missing = set(feature_names) - set(X.columns)
            if missing:
                raise ValueError(f"Missing required features: {missing}")
            X = X[feature_names]
    else:
        model = xgboost_model
    
    # Make predictions
    predictions = model.predict(X)
    
    # Format results
    result = [
        {'prediction': round(float(p), 2)}
        for p in predictions
    ]
    
    return result


# Local testing
if __name__ == "__main__":
    print("Testing scoring script locally...\n")
    
    # Initialize models
    init()
    
    # Test health check
    print("=" * 50)
    print("Testing Health Check:")
    health_input = json.dumps({'health_check': True})
    result = run(health_input)
    print(json.dumps(json.loads(result), indent=2))
    
    # Test Prophet prediction
    print("\n" + "=" * 50)
    print("Testing Prophet Prediction:")
    prophet_input = json.dumps({
        'model': 'prophet',
        'periods': 7
    })
    result = run(prophet_input)
    print(json.dumps(json.loads(result), indent=2))
    
    # Test XGBoost prediction with all required features
    print("\n" + "=" * 50)
    print("Testing XGBoost Prediction:")
    xgboost_input = json.dumps({
        'model': 'xgboost',
        'features': {
            'day_of_week': 1,
            'month': 6,
            'day_of_month': 15,
            'quarter': 2,
            'year': 2024,
            'week_of_year': 24,
            'is_weekend': 0,
            'is_month_start': 0,
            'is_month_end': 0,
            'lag_1': 45,
            'lag_7': 50,
            'lag_14': 48,
            'lag_30': 52,
            'rolling_mean_7': 45,
            'rolling_mean_14': 47,
            'rolling_mean_30': 49,
            'rolling_std_7': 10,
            'rolling_std_14': 12,
            'rolling_std_30': 15
        }
    })
    result = run(xgboost_input)
    print(json.dumps(json.loads(result), indent=2))
    
    # Test batch XGBoost prediction
    print("\n" + "=" * 50)
    print("Testing Batch XGBoost Prediction:")
    batch_input = json.dumps({
        'model': 'xgboost',
        'features': [
            {'day_of_week': 1, 'month': 6, 'day_of_month': 15, 'quarter': 2, 'year': 2024,
             'week_of_year': 24, 'is_weekend': 0, 'is_month_start': 0, 'is_month_end': 0,
             'lag_1': 45, 'lag_7': 50, 'lag_14': 48, 'lag_30': 52,
             'rolling_mean_7': 45, 'rolling_mean_14': 47, 'rolling_mean_30': 49,
             'rolling_std_7': 10, 'rolling_std_14': 12, 'rolling_std_30': 15},
            {'day_of_week': 2, 'month': 6, 'day_of_month': 16, 'quarter': 2, 'year': 2024,
             'week_of_year': 24, 'is_weekend': 0, 'is_month_start': 0, 'is_month_end': 0,
             'lag_1': 50, 'lag_7': 55, 'lag_14': 52, 'lag_30': 54,
             'rolling_mean_7': 48, 'rolling_mean_14': 50, 'rolling_mean_30': 51,
             'rolling_std_7': 12, 'rolling_std_14': 14, 'rolling_std_30': 16}
        ]
    })
    result = run(batch_input)
    print(json.dumps(json.loads(result), indent=2))

