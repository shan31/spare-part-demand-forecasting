"""
XGBoost Model Module
Gradient boosting-based demand forecasting
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple, Any
import logging
import pickle
from pathlib import Path

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class XGBoostForecaster:
    """XGBoost-based demand forecasting model."""
    
    def __init__(
        self,
        n_estimators: int = 100,
        max_depth: int = 6,
        learning_rate: float = 0.1,
        subsample: float = 0.8,
        colsample_bytree: float = 0.8,
        reg_alpha: float = 0.0,
        reg_lambda: float = 1.0,
        random_state: int = 42,
        **kwargs
    ):
        self.model = None
        self.model_params = {
            'n_estimators': n_estimators,
            'max_depth': max_depth,
            'learning_rate': learning_rate,
            'subsample': subsample,
            'colsample_bytree': colsample_bytree,
            'reg_alpha': reg_alpha,
            'reg_lambda': reg_lambda,
            'random_state': random_state,
            'objective': 'reg:squarederror',
            **kwargs
        }
        self.feature_names = []
        self.is_fitted = False
    
    def _create_model(self):
        """Create XGBoost model instance."""
        try:
            import xgboost as xgb
        except ImportError:
            raise ImportError("XGBoost not installed. Run: pip install xgboost")
        
        return xgb.XGBRegressor(**self.model_params)
    
    def fit(
        self, 
        X: pd.DataFrame, 
        y: pd.Series,
        eval_set: Optional[List[Tuple]] = None,
        early_stopping_rounds: Optional[int] = None,
        verbose: bool = True
    ) -> 'XGBoostForecaster':
        """Fit the XGBoost model."""
        self.model = self._create_model()
        self.feature_names = X.columns.tolist()
        
        fit_params = {'X': X, 'y': y}
        
        if eval_set:
            fit_params['eval_set'] = eval_set
        if early_stopping_rounds:
            fit_params['early_stopping_rounds'] = early_stopping_rounds
        
        logger.info(f"Fitting XGBoost model on {X.shape[0]} samples, {X.shape[1]} features...")
        self.model.fit(**fit_params)
        self.is_fitted = True
        
        logger.info("XGBoost model fitted successfully")
        return self
    
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Generate predictions."""
        if not self.is_fitted:
            raise ValueError("Model must be fitted before prediction")
        
        predictions = self.model.predict(X)
        logger.info(f"Generated {len(predictions)} predictions")
        
        return predictions
    
    def evaluate(
        self, 
        y_true: pd.Series, 
        y_pred: np.ndarray
    ) -> Dict[str, float]:
        """Calculate evaluation metrics."""
        from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
        
        mae = mean_absolute_error(y_true, y_pred)
        rmse = np.sqrt(mean_squared_error(y_true, y_pred))
        mape = np.mean(np.abs((y_true - y_pred) / (y_true + 1e-10))) * 100
        r2 = r2_score(y_true, y_pred)
        
        metrics = {
            'MAE': round(mae, 4),
            'RMSE': round(rmse, 4),
            'MAPE': round(mape, 2),
            'R2': round(r2, 4)
        }
        
        logger.info(f"Evaluation metrics: {metrics}")
        return metrics
    
    def get_feature_importance(self, importance_type: str = 'gain') -> pd.DataFrame:
        """Get feature importance scores."""
        if not self.is_fitted:
            raise ValueError("Model must be fitted before getting feature importance")
        
        importance = self.model.get_booster().get_score(importance_type=importance_type)
        
        importance_df = pd.DataFrame({
            'feature': list(importance.keys()),
            'importance': list(importance.values())
        }).sort_values('importance', ascending=False)
        
        return importance_df
    
    def tune_hyperparameters(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        param_grid: Optional[Dict] = None,
        cv: int = 5,
        scoring: str = 'neg_mean_absolute_error'
    ) -> Dict[str, Any]:
        """Tune hyperparameters using GridSearchCV."""
        from sklearn.model_selection import GridSearchCV, TimeSeriesSplit
        import xgboost as xgb
        
        if param_grid is None:
            param_grid = {
                'n_estimators': [50, 100, 200],
                'max_depth': [3, 5, 7],
                'learning_rate': [0.01, 0.1, 0.2],
                'subsample': [0.7, 0.8, 0.9]
            }
        
        model = xgb.XGBRegressor(random_state=42, objective='reg:squarederror')
        tscv = TimeSeriesSplit(n_splits=cv)
        
        logger.info("Starting hyperparameter tuning...")
        grid_search = GridSearchCV(
            model, 
            param_grid, 
            cv=tscv, 
            scoring=scoring,
            n_jobs=-1,
            verbose=1
        )
        
        grid_search.fit(X, y)
        
        best_params = grid_search.best_params_
        best_score = grid_search.best_score_
        
        logger.info(f"Best parameters: {best_params}")
        logger.info(f"Best score: {best_score}")
        
        # Update model with best parameters
        self.model_params.update(best_params)
        
        return {
            'best_params': best_params,
            'best_score': best_score,
            'cv_results': grid_search.cv_results_
        }
    
    def cross_validate(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        n_splits: int = 5
    ) -> Dict[str, List[float]]:
        """Perform time series cross-validation."""
        from sklearn.model_selection import TimeSeriesSplit, cross_val_score
        
        model = self._create_model()
        tscv = TimeSeriesSplit(n_splits=n_splits)
        
        mae_scores = -cross_val_score(model, X, y, cv=tscv, scoring='neg_mean_absolute_error')
        rmse_scores = np.sqrt(-cross_val_score(model, X, y, cv=tscv, scoring='neg_mean_squared_error'))
        
        results = {
            'MAE': mae_scores.tolist(),
            'MAE_mean': round(mae_scores.mean(), 4),
            'MAE_std': round(mae_scores.std(), 4),
            'RMSE': rmse_scores.tolist(),
            'RMSE_mean': round(rmse_scores.mean(), 4),
            'RMSE_std': round(rmse_scores.std(), 4)
        }
        
        logger.info(f"CV Results - MAE: {results['MAE_mean']} (+/- {results['MAE_std']})")
        return results
    
    def save_model(self, filepath: str) -> None:
        """Save model to disk."""
        filepath = Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)
        
        model_data = {
            'model': self.model,
            'feature_names': self.feature_names,
            'params': self.model_params
        }
        
        with open(filepath, 'wb') as f:
            pickle.dump(model_data, f)
        
        logger.info(f"Model saved to {filepath}")
    
    def load_model(self, filepath: str) -> None:
        """Load model from disk."""
        with open(filepath, 'rb') as f:
            model_data = pickle.load(f)
        
        self.model = model_data['model']
        self.feature_names = model_data['feature_names']
        self.model_params = model_data['params']
        self.is_fitted = True
        
        logger.info(f"Model loaded from {filepath}")


def train_test_split_timeseries(
    df: pd.DataFrame,
    target_col: str,
    test_size: float = 0.2,
    date_col: Optional[str] = None
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """Split data for time series (no shuffling)."""
    n = len(df)
    train_size = int(n * (1 - test_size))
    
    if date_col:
        df = df.sort_values(date_col)
    
    train_df = df.iloc[:train_size]
    test_df = df.iloc[train_size:]
    
    feature_cols = [c for c in df.columns if c != target_col and c != date_col]
    
    X_train = train_df[feature_cols]
    X_test = test_df[feature_cols]
    y_train = train_df[target_col]
    y_test = test_df[target_col]
    
    logger.info(f"Train size: {len(X_train)}, Test size: {len(X_test)}")
    
    return X_train, X_test, y_train, y_test


if __name__ == "__main__":
    # Test XGBoost model
    import sys
    sys.path.append('..')
    from data_loader import DataLoader
    from preprocessing import preprocess_pipeline, FeatureEngineer
    
    # Generate and preprocess sample data
    loader = DataLoader()
    df = loader.get_sample_data(n_rows=500)
    
    processed_df = preprocess_pipeline(
        df,
        group_cols=['part_id', 'service_center'],
        categorical_cols=['part_id', 'service_center', 'category']
    )
    
    # Prepare for XGBoost
    fe = FeatureEngineer()
    X, y = fe.prepare_for_xgboost(processed_df, 'demand_quantity')
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split_timeseries(
        pd.concat([X, y], axis=1), 'demand_quantity'
    )
    
    # Train and evaluate
    model = XGBoostForecaster(n_estimators=50, max_depth=4)
    model.fit(X_train, y_train)
    
    y_pred = model.predict(X_test)
    metrics = model.evaluate(y_test, y_pred)
    
    print("\nFeature Importance:")
    print(model.get_feature_importance().head(10))
