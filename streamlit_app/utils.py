"""
Utility functions for loading data and models
"""

import pandas as pd
import pickle
from pathlib import Path
import os


def get_project_root():
    """Get the project root directory."""
    return Path(__file__).parent.parent


def load_raw_data():
    """Load raw spare parts demand data."""
    data_path = get_project_root() / 'data' / 'raw' / 'spare_parts_demand.csv'
    if data_path.exists():
        return pd.read_csv(data_path, parse_dates=['date'])
    return None


def load_daily_demand():
    """Load daily aggregated demand data."""
    data_path = get_project_root() / 'data' / 'processed' / 'daily_demand.csv'
    if data_path.exists():
        return pd.read_csv(data_path, parse_dates=['date'])
    return None


def load_prophet_model():
    """Load trained Prophet model."""
    model_path = get_project_root() / 'models' / 'prophet_model.pkl'
    if model_path.exists():
        with open(model_path, 'rb') as f:
            return pickle.load(f)
    return None


def load_xgboost_model():
    """Load trained XGBoost model."""
    model_path = get_project_root() / 'models' / 'xgboost_model.pkl'
    if model_path.exists():
        with open(model_path, 'rb') as f:
            return pickle.load(f)
    return None


def load_model_metrics():
    """Load model comparison metrics."""
    prophet_path = get_project_root() / 'models' / 'prophet_metrics.csv'
    xgb_path = get_project_root() / 'models' / 'xgboost_metrics.csv'
    
    metrics = []
    if prophet_path.exists():
        metrics.append(pd.read_csv(prophet_path))
    if xgb_path.exists():
        metrics.append(pd.read_csv(xgb_path))
    
    if metrics:
        return pd.concat(metrics, ignore_index=True)
    return None


def load_forecast():
    """Load saved forecast data."""
    forecast_path = get_project_root() / 'data' / 'processed' / 'prophet_forecast.csv'
    if forecast_path.exists():
        return pd.read_csv(forecast_path, parse_dates=['Date'])
    return None


def get_data_summary():
    """Get summary statistics for the dashboard."""
    df = load_raw_data()
    if df is not None:
        return {
            'total_records': len(df),
            'unique_parts': df['part_id'].nunique(),
            'service_centers': df['service_center'].nunique(),
            'categories': df['category'].nunique(),
            'date_min': df['date'].min(),
            'date_max': df['date'].max()
        }
    return None


def get_top_parts(n=10):
    """Get top N parts by demand."""
    df = load_raw_data()
    if df is not None:
        top = df.groupby(['part_id', 'part_name', 'category'])['demand_quantity'].sum().reset_index()
        return top.nlargest(n, 'demand_quantity')
    return None


def get_demand_by_category():
    """Get total demand by category."""
    df = load_raw_data()
    if df is not None:
        return df.groupby('category')['demand_quantity'].sum().reset_index()
    return None


def get_demand_by_service_center():
    """Get total demand by service center."""
    df = load_raw_data()
    if df is not None:
        return df.groupby('service_center')['demand_quantity'].sum().reset_index()
    return None
