"""
Preprocessing Module
Feature engineering and data transformation for demand forecasting
"""

import pandas as pd
import numpy as np
from typing import List, Optional, Tuple
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class FeatureEngineer:
    """Create features for demand forecasting models."""
    
    def __init__(self):
        self.feature_columns = []
    
    def add_date_features(self, df: pd.DataFrame, date_col: str = 'date') -> pd.DataFrame:
        """Add date-based features."""
        df = df.copy()
        df[date_col] = pd.to_datetime(df[date_col])
        
        df['day_of_week'] = df[date_col].dt.dayofweek
        df['day_of_month'] = df[date_col].dt.day
        df['week_of_year'] = df[date_col].dt.isocalendar().week.astype(int)
        df['month'] = df[date_col].dt.month
        df['quarter'] = df[date_col].dt.quarter
        df['year'] = df[date_col].dt.year
        df['is_weekend'] = df['day_of_week'].isin([5, 6]).astype(int)
        df['is_month_start'] = df[date_col].dt.is_month_start.astype(int)
        df['is_month_end'] = df[date_col].dt.is_month_end.astype(int)
        
        logger.info("Added date features")
        return df
    
    def add_lag_features(
        self, 
        df: pd.DataFrame, 
        target_col: str = 'demand_quantity',
        group_cols: Optional[List[str]] = None,
        lags: List[int] = [1, 7, 14, 30]
    ) -> pd.DataFrame:
        """Add lagged features for time series."""
        df = df.copy()
        
        if group_cols:
            for lag in lags:
                df[f'lag_{lag}'] = df.groupby(group_cols)[target_col].shift(lag)
        else:
            for lag in lags:
                df[f'lag_{lag}'] = df[target_col].shift(lag)
        
        logger.info(f"Added lag features: {lags}")
        return df
    
    def add_rolling_features(
        self, 
        df: pd.DataFrame, 
        target_col: str = 'demand_quantity',
        group_cols: Optional[List[str]] = None,
        windows: List[int] = [7, 14, 30]
    ) -> pd.DataFrame:
        """Add rolling statistics features."""
        df = df.copy()
        
        for window in windows:
            if group_cols:
                df[f'rolling_mean_{window}'] = df.groupby(group_cols)[target_col].transform(
                    lambda x: x.shift(1).rolling(window=window, min_periods=1).mean()
                )
                df[f'rolling_std_{window}'] = df.groupby(group_cols)[target_col].transform(
                    lambda x: x.shift(1).rolling(window=window, min_periods=1).std()
                )
            else:
                df[f'rolling_mean_{window}'] = df[target_col].shift(1).rolling(window=window, min_periods=1).mean()
                df[f'rolling_std_{window}'] = df[target_col].shift(1).rolling(window=window, min_periods=1).std()
        
        logger.info(f"Added rolling features for windows: {windows}")
        return df
    
    def add_holiday_features(
        self, 
        df: pd.DataFrame, 
        date_col: str = 'date',
        country: str = 'IN'
    ) -> pd.DataFrame:
        """Add holiday indicator features."""
        df = df.copy()
        
        try:
            import holidays
            
            year_range = range(df[date_col].dt.year.min(), df[date_col].dt.year.max() + 1)
            holiday_dates = holidays.country_holidays(country, years=year_range)
            
            df['is_holiday'] = df[date_col].dt.date.isin(holiday_dates).astype(int)
            logger.info(f"Added holiday features for {country}")
            
        except ImportError:
            logger.warning("holidays package not installed. Skipping holiday features.")
            df['is_holiday'] = 0
        
        return df
    
    def encode_categoricals(
        self, 
        df: pd.DataFrame, 
        categorical_cols: List[str],
        method: str = 'onehot'
    ) -> pd.DataFrame:
        """Encode categorical variables."""
        df = df.copy()
        
        if method == 'onehot':
            df = pd.get_dummies(df, columns=categorical_cols, prefix=categorical_cols)
        elif method == 'label':
            from sklearn.preprocessing import LabelEncoder
            for col in categorical_cols:
                le = LabelEncoder()
                df[f'{col}_encoded'] = le.fit_transform(df[col].astype(str))
        
        logger.info(f"Encoded categorical columns: {categorical_cols}")
        return df
    
    def prepare_for_prophet(
        self, 
        df: pd.DataFrame,
        date_col: str = 'date',
        target_col: str = 'demand_quantity'
    ) -> pd.DataFrame:
        """Prepare data for Prophet model (requires 'ds' and 'y' columns)."""
        prophet_df = df[[date_col, target_col]].copy()
        prophet_df.columns = ['ds', 'y']
        prophet_df['ds'] = pd.to_datetime(prophet_df['ds'])
        
        logger.info("Prepared data for Prophet format")
        return prophet_df
    
    def prepare_for_xgboost(
        self,
        df: pd.DataFrame,
        target_col: str = 'demand_quantity',
        feature_cols: Optional[List[str]] = None
    ) -> Tuple[pd.DataFrame, pd.Series]:
        """Prepare data for XGBoost model."""
        df = df.copy()
        
        # Drop rows with NaN (from lag features)
        df = df.dropna()
        
        if feature_cols is None:
            # Use all numeric columns except target
            feature_cols = df.select_dtypes(include=[np.number]).columns.tolist()
            feature_cols = [c for c in feature_cols if c != target_col]
        
        X = df[feature_cols]
        y = df[target_col]
        
        logger.info(f"Prepared X shape: {X.shape}, y shape: {y.shape}")
        return X, y


def preprocess_pipeline(
    df: pd.DataFrame,
    date_col: str = 'date',
    target_col: str = 'demand_quantity',
    group_cols: Optional[List[str]] = None,
    categorical_cols: Optional[List[str]] = None
) -> pd.DataFrame:
    """Full preprocessing pipeline."""
    fe = FeatureEngineer()
    
    # Add features
    df = fe.add_date_features(df, date_col)
    df = fe.add_lag_features(df, target_col, group_cols)
    df = fe.add_rolling_features(df, target_col, group_cols)
    df = fe.add_holiday_features(df, date_col)
    
    if categorical_cols:
        df = fe.encode_categoricals(df, categorical_cols)
    
    return df


if __name__ == "__main__":
    # Test preprocessing
    from data_loader import DataLoader
    
    loader = DataLoader()
    df = loader.get_sample_data()
    
    processed_df = preprocess_pipeline(
        df, 
        group_cols=['part_id', 'service_center'],
        categorical_cols=['part_id', 'service_center', 'category']
    )
    
    print(processed_df.head())
    print(f"\nShape after preprocessing: {processed_df.shape}")
