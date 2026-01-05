"""
Generate Sample Spare Parts Demand Dataset
Creates realistic data for demonstration and testing
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import random


def generate_spare_parts_data(
    n_days: int = 730,  # 2 years
    n_parts: int = 50,
    n_service_centers: int = 4,
    start_date: str = '2022-01-01'
) -> pd.DataFrame:
    """
    Generate realistic spare parts demand data.
    
    Features:
    - Seasonal patterns (monthly, yearly)
    - Weekly patterns (lower demand on weekends)
    - Random spikes (promotional events)
    - Part category characteristics
    """
    np.random.seed(42)
    random.seed(42)
    
    # Define parts with categories
    categories = ['Engine', 'Electrical', 'Body', 'Transmission', 'Brakes', 'Suspension']
    parts = []
    
    for i in range(n_parts):
        category = random.choice(categories)
        parts.append({
            'part_id': f'P-{str(i+1).zfill(4)}',
            'part_name': f'{category}_Part_{i+1}',
            'category': category,
            'base_demand': np.random.randint(5, 100),  # Base daily demand
            'price': round(np.random.uniform(50, 5000), 2)
        })
    
    parts_df = pd.DataFrame(parts)
    
    # Service centers
    service_centers = [f'SC-{region}' for region in ['North', 'South', 'East', 'West'][:n_service_centers]]
    
    # Generate date range
    start = pd.to_datetime(start_date)
    dates = pd.date_range(start=start, periods=n_days, freq='D')
    
    # Generate records
    records = []
    
    for date in dates:
        day_of_week = date.dayofweek
        month = date.month
        
        # Weekly factor (lower on weekends)
        if day_of_week >= 5:
            weekly_factor = 0.6
        elif day_of_week == 4:  # Friday
            weekly_factor = 1.1
        else:
            weekly_factor = 1.0
        
        # Monthly seasonality (higher in summer months for AC parts, etc.)
        monthly_factor = 1 + 0.2 * np.sin(2 * np.pi * (month - 3) / 12)
        
        # Random event factor (promotional days)
        event_factor = 1.5 if np.random.random() < 0.05 else 1.0
        
        for part in parts:
            for sc in service_centers:
                # Service center factor (different regions have different demand)
                sc_factor = {'SC-North': 1.0, 'SC-South': 1.2, 'SC-East': 0.9, 'SC-West': 1.1}.get(sc, 1.0)
                
                # Category seasonality
                cat_factor = 1.0
                if part['category'] == 'Electrical' and month in [11, 12, 1, 2]:
                    cat_factor = 1.3  # Higher electrical demand in winter
                elif part['category'] == 'Body' and month in [6, 7, 8]:
                    cat_factor = 1.2  # Higher body repairs in monsoon
                
                # Calculate demand
                base = part['base_demand']
                demand = int(base * weekly_factor * monthly_factor * event_factor * sc_factor * cat_factor)
                demand = max(0, demand + np.random.randint(-5, 10))
                
                records.append({
                    'date': date,
                    'part_id': part['part_id'],
                    'part_name': part['part_name'],
                    'category': part['category'],
                    'service_center': sc,
                    'demand_quantity': demand,
                    'unit_price': part['price']
                })
    
    df = pd.DataFrame(records)
    
    # Add some derived columns
    df['revenue'] = df['demand_quantity'] * df['unit_price']
    df['day_of_week'] = df['date'].dt.dayofweek
    df['month'] = df['date'].dt.month
    df['year'] = df['date'].dt.year
    df['quarter'] = df['date'].dt.quarter
    
    return df


def save_data(df: pd.DataFrame, output_path: str = 'data/raw/spare_parts_demand.csv'):
    """Save the generated data to CSV."""
    import os
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df.to_csv(output_path, index=False)
    print(f"[OK] Data saved to {output_path}")
    print(f"   Shape: {df.shape}")
    print(f"   Date range: {df['date'].min()} to {df['date'].max()}")
    print(f"   Unique parts: {df['part_id'].nunique()}")
    print(f"   Unique service centers: {df['service_center'].nunique()}")


if __name__ == "__main__":
    # Generate and save data
    df = generate_spare_parts_data()
    save_data(df)
    
    print("\n[DATA] Sample Data Preview:")
    print(df.head(10))
    
    print("\n[STATS] Summary Statistics:")
    print(df.describe())
