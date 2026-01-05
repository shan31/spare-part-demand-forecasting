"""
Prophet Model Module
Time series forecasting using Facebook Prophet
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Any
import logging
import pickle
from pathlib import Path

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ProphetForecaster:
    """Prophet-based demand forecasting model."""
    
    def __init__(
        self,
        seasonality_mode: str = 'multiplicative',
        yearly_seasonality: bool = True,
        weekly_seasonality: bool = True,
        daily_seasonality: bool = False,
        changepoint_prior_scale: float = 0.05,
        seasonality_prior_scale: float = 10.0,
        holidays_prior_scale: float = 10.0,
        **kwargs
    ):
        self.model = None
        self.model_params = {
            'seasonality_mode': seasonality_mode,
            'yearly_seasonality': yearly_seasonality,
            'weekly_seasonality': weekly_seasonality,
            'daily_seasonality': daily_seasonality,
            'changepoint_prior_scale': changepoint_prior_scale,
            'seasonality_prior_scale': seasonality_prior_scale,
            'holidays_prior_scale': holidays_prior_scale,
            **kwargs
        }
        self.is_fitted = False
    
    def _create_model(self):
        """Create Prophet model instance."""
        try:
            from prophet import Prophet
        except ImportError:
            raise ImportError("Prophet not installed. Run: pip install prophet")
        
        return Prophet(**self.model_params)
    
    def add_country_holidays(self, country_name: str = 'IN') -> None:
        """Add country holidays to the model."""
        if self.model is None:
            self.model = self._create_model()
        
        self.model.add_country_holidays(country_name=country_name)
        logger.info(f"Added holidays for {country_name}")
    
    def add_custom_seasonality(
        self, 
        name: str, 
        period: float, 
        fourier_order: int
    ) -> None:
        """Add custom seasonality component."""
        if self.model is None:
            self.model = self._create_model()
        
        self.model.add_seasonality(
            name=name, 
            period=period, 
            fourier_order=fourier_order
        )
        logger.info(f"Added custom seasonality: {name}")
    
    def fit(
        self, 
        df: pd.DataFrame,
        ds_col: str = 'ds',
        y_col: str = 'y'
    ) -> 'ProphetForecaster':
        """Fit the Prophet model."""
        if self.model is None:
            self.model = self._create_model()
        
        # Ensure correct column names
        train_df = df[[ds_col, y_col]].copy()
        train_df.columns = ['ds', 'y']
        train_df['ds'] = pd.to_datetime(train_df['ds'])
        
        logger.info(f"Fitting Prophet model on {len(train_df)} samples...")
        self.model.fit(train_df)
        self.is_fitted = True
        
        logger.info("Prophet model fitted successfully")
        return self
    
    def predict(
        self, 
        periods: int = 30,
        freq: str = 'D',
        include_history: bool = False
    ) -> pd.DataFrame:
        """Generate predictions for future periods."""
        if not self.is_fitted:
            raise ValueError("Model must be fitted before prediction")
        
        future = self.model.make_future_dataframe(
            periods=periods, 
            freq=freq,
            include_history=include_history
        )
        
        forecast = self.model.predict(future)
        
        logger.info(f"Generated {periods} predictions")
        return forecast
    
    def predict_on_dates(self, dates: pd.DataFrame) -> pd.DataFrame:
        """Predict for specific dates."""
        if not self.is_fitted:
            raise ValueError("Model must be fitted before prediction")
        
        dates_df = dates.copy()
        if 'ds' not in dates_df.columns:
            dates_df = pd.DataFrame({'ds': pd.to_datetime(dates)})
        else:
            dates_df['ds'] = pd.to_datetime(dates_df['ds'])
        
        forecast = self.model.predict(dates_df)
        return forecast
    
    def get_components(self, forecast: pd.DataFrame) -> Dict[str, pd.Series]:
        """Extract forecast components (trend, seasonality, etc.)."""
        components = {}
        
        if 'trend' in forecast.columns:
            components['trend'] = forecast['trend']
        if 'weekly' in forecast.columns:
            components['weekly'] = forecast['weekly']
        if 'yearly' in forecast.columns:
            components['yearly'] = forecast['yearly']
        if 'holidays' in forecast.columns:
            components['holidays'] = forecast['holidays']
        
        return components
    
    def evaluate(
        self, 
        y_true: pd.Series, 
        y_pred: pd.Series
    ) -> Dict[str, float]:
        """Calculate evaluation metrics."""
        from sklearn.metrics import mean_absolute_error, mean_squared_error
        
        mae = mean_absolute_error(y_true, y_pred)
        rmse = np.sqrt(mean_squared_error(y_true, y_pred))
        mape = np.mean(np.abs((y_true - y_pred) / y_true)) * 100
        
        metrics = {
            'MAE': round(mae, 4),
            'RMSE': round(rmse, 4),
            'MAPE': round(mape, 2)
        }
        
        logger.info(f"Evaluation metrics: {metrics}")
        return metrics
    
    def cross_validate(
        self, 
        df: pd.DataFrame,
        initial: str = '365 days',
        period: str = '90 days',
        horizon: str = '30 days'
    ) -> pd.DataFrame:
        """Perform time series cross-validation."""
        from prophet.diagnostics import cross_validation, performance_metrics
        
        if not self.is_fitted:
            raise ValueError("Model must be fitted before cross-validation")
        
        cv_results = cross_validation(
            self.model,
            initial=initial,
            period=period,
            horizon=horizon
        )
        
        metrics_df = performance_metrics(cv_results)
        logger.info(f"Cross-validation completed with {len(cv_results)} folds")
        
        return metrics_df
    
    def save_model(self, filepath: str) -> None:
        """Save model to disk."""
        filepath = Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)
        
        with open(filepath, 'wb') as f:
            pickle.dump(self.model, f)
        
        logger.info(f"Model saved to {filepath}")
    
    def load_model(self, filepath: str) -> None:
        """Load model from disk."""
        with open(filepath, 'rb') as f:
            self.model = pickle.load(f)
        
        self.is_fitted = True
        logger.info(f"Model loaded from {filepath}")


if __name__ == "__main__":
    # Test Prophet model
    import sys
    sys.path.append('..')
    from data_loader import DataLoader
    from preprocessing import FeatureEngineer
    
    # Generate sample data
    loader = DataLoader()
    df = loader.get_sample_data(n_rows=365)
    
    # Aggregate daily demand
    df['date'] = pd.to_datetime(df['date'])
    daily_demand = df.groupby('date')['demand_quantity'].sum().reset_index()
    
    # Prepare for Prophet
    fe = FeatureEngineer()
    prophet_df = fe.prepare_for_prophet(daily_demand, 'date', 'demand_quantity')
    
    # Train and forecast
    model = ProphetForecaster()
    model.fit(prophet_df)
    
    forecast = model.predict(periods=30)
    print(forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].tail(10))
