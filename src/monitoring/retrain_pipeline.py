"""
Automated Model Retraining Pipeline
Triggers retraining based on drift, schedule, or performance degradation
"""

import os
import logging
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional, Dict
import json

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class RetrainingPipeline:
    """Automated model retraining pipeline."""
    
    def __init__(
        self,
        drift_threshold: float = 0.3,
        performance_threshold: float = 0.15,
        min_days_between_retrain: int = 7
    ):
        self.drift_threshold = drift_threshold
        self.performance_threshold = performance_threshold
        self.min_days_between_retrain = min_days_between_retrain
        self.last_retrain_date = None
        self.retrain_history = []
    
    def should_retrain(
        self,
        drift_score: Optional[float] = None,
        performance_drop: Optional[float] = None,
        last_retrain: Optional[datetime] = None
    ) -> Dict:
        """Determine if retraining should be triggered."""
        reasons = []
        should_trigger = False
        
        # Check drift threshold
        if drift_score is not None and drift_score > self.drift_threshold:
            reasons.append(f"Data drift detected: {drift_score:.3f} > {self.drift_threshold}")
            should_trigger = True
        
        # Check performance degradation
        if performance_drop is not None and performance_drop > self.performance_threshold:
            reasons.append(f"Performance degraded: {performance_drop:.1%} > {self.performance_threshold:.1%}")
            should_trigger = True
        
        # Check minimum time since last retrain
        if last_retrain:
            days_since = (datetime.now() - last_retrain).days
            if days_since < self.min_days_between_retrain:
                reasons.append(f"Too soon since last retrain: {days_since} < {self.min_days_between_retrain} days")
                should_trigger = False
        
        return {
            'should_retrain': should_trigger,
            'reasons': reasons,
            'timestamp': datetime.now().isoformat()
        }
    
    def trigger_retrain(
        self,
        model_type: str = "both",
        data_path: str = "data/processed/daily_demand.csv"
    ) -> Dict:
        """Trigger model retraining."""
        result = {
            'triggered_at': datetime.now().isoformat(),
            'model_type': model_type,
            'status': 'initiated',
            'models': {}
        }
        
        logger.info(f"Triggering retraining for: {model_type}")
        
        if model_type in ["prophet", "both"]:
            prophet_result = self._retrain_prophet(data_path)
            result['models']['prophet'] = prophet_result
        
        if model_type in ["xgboost", "both"]:
            xgboost_result = self._retrain_xgboost(data_path)
            result['models']['xgboost'] = xgboost_result
        
        # Update history
        self.last_retrain_date = datetime.now()
        self.retrain_history.append(result)
        
        result['status'] = 'completed'
        logger.info("Retraining completed!")
        
        return result
    
    def _retrain_prophet(self, data_path: str) -> Dict:
        """Retrain Prophet model."""
        try:
            import pandas as pd
            from prophet import Prophet
            import pickle
            
            logger.info("Retraining Prophet model...")
            
            # Load data
            df = pd.read_csv(data_path, parse_dates=['date'])
            prophet_df = df[['date', 'demand_quantity']].copy()
            prophet_df.columns = ['ds', 'y']
            
            # Train model
            model = Prophet(
                seasonality_mode='multiplicative',
                yearly_seasonality=True,
                weekly_seasonality=True
            )
            model.add_country_holidays(country_name='IN')
            model.fit(prophet_df)
            
            # Save model
            model_path = Path('models/prophet_model.pkl')
            model_path.parent.mkdir(exist_ok=True)
            with open(model_path, 'wb') as f:
                pickle.dump(model, f)
            
            return {
                'status': 'success',
                'training_samples': len(prophet_df),
                'model_path': str(model_path),
                'timestamp': datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Prophet retraining failed: {e}")
            return {'status': 'failed', 'error': str(e)}
    
    def _retrain_xgboost(self, data_path: str) -> Dict:
        """Retrain XGBoost model."""
        try:
            import pandas as pd
            import numpy as np
            import xgboost as xgb
            import pickle
            
            logger.info("Retraining XGBoost model...")
            
            # Load data
            df = pd.read_csv(data_path, parse_dates=['date'])
            
            # Feature engineering
            df['day_of_week'] = df['date'].dt.dayofweek
            df['month'] = df['date'].dt.month
            df['year'] = df['date'].dt.year
            
            for lag in [1, 7, 14, 30]:
                df[f'lag_{lag}'] = df['demand_quantity'].shift(lag)
            
            for window in [7, 14, 30]:
                df[f'rolling_mean_{window}'] = df['demand_quantity'].shift(1).rolling(window).mean()
            
            df = df.dropna()
            
            # Prepare features
            feature_cols = [c for c in df.columns if c not in ['date', 'demand_quantity', 'revenue']]
            X = df[feature_cols]
            y = df['demand_quantity']
            
            # Train model
            model = xgb.XGBRegressor(
                n_estimators=100,
                max_depth=6,
                learning_rate=0.1,
                random_state=42
            )
            model.fit(X, y)
            
            # Save model
            model_path = Path('models/xgboost_model.pkl')
            model_path.parent.mkdir(exist_ok=True)
            
            model_data = {
                'model': model,
                'feature_names': feature_cols
            }
            with open(model_path, 'wb') as f:
                pickle.dump(model_data, f)
            
            return {
                'status': 'success',
                'training_samples': len(df),
                'features': len(feature_cols),
                'model_path': str(model_path),
                'timestamp': datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"XGBoost retraining failed: {e}")
            return {'status': 'failed', 'error': str(e)}
    
    def save_history(self, filepath: str):
        """Save retraining history."""
        with open(filepath, 'w') as f:
            json.dump(self.retrain_history, f, indent=2)


def run_scheduled_retrain():
    """Run scheduled retraining (for cron jobs or Azure ML pipelines)."""
    pipeline = RetrainingPipeline()
    
    # Check if retraining should be triggered
    check = pipeline.should_retrain(last_retrain=pipeline.last_retrain_date)
    
    if check['should_retrain']:
        result = pipeline.trigger_retrain(model_type="both")
        pipeline.save_history('models/retrain_history.json')
        return result
    
    logger.info("Retraining not needed at this time")
    return check


if __name__ == "__main__":
    # Run retraining
    pipeline = RetrainingPipeline()
    result = pipeline.trigger_retrain(
        model_type="both",
        data_path="data/processed/daily_demand.csv"
    )
    
    print("\n" + "="*50)
    print("RETRAINING RESULTS")
    print("="*50)
    print(json.dumps(result, indent=2))
