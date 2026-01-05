"""
Test script for Data Drift Monitor
Demonstrates drift detection between two datasets
"""

import sys
sys.path.insert(0, '.')

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from src.monitoring.drift_monitor import DriftMonitor

def generate_baseline_data(n_rows=500):
    """Generate baseline demand data."""
    np.random.seed(42)
    
    start_date = datetime(2023, 1, 1)
    dates = [start_date + timedelta(days=i) for i in range(n_rows)]
    
    data = {
        'date': dates,
        'demand_quantity': np.random.normal(loc=100, scale=15, size=n_rows).clip(min=0),
        'price': np.random.normal(loc=50, scale=10, size=n_rows).clip(min=10),
        'inventory_level': np.random.normal(loc=200, scale=30, size=n_rows).clip(min=0)
    }
    return pd.DataFrame(data)

def generate_drifted_data(n_rows=500, drift_level='moderate'):
    """Generate data with drift compared to baseline."""
    np.random.seed(123)
    
    start_date = datetime(2023, 7, 1)
    dates = [start_date + timedelta(days=i) for i in range(n_rows)]
    
    if drift_level == 'none':
        # No drift - same distribution
        demand = np.random.normal(loc=100, scale=15, size=n_rows)
        price = np.random.normal(loc=50, scale=10, size=n_rows)
    elif drift_level == 'moderate':
        # Moderate drift - shifted mean
        demand = np.random.normal(loc=120, scale=20, size=n_rows)
        price = np.random.normal(loc=55, scale=12, size=n_rows)
    elif drift_level == 'severe':
        # Severe drift - very different distribution
        demand = np.random.normal(loc=150, scale=40, size=n_rows)
        price = np.random.normal(loc=70, scale=20, size=n_rows)
    
    data = {
        'date': dates,
        'demand_quantity': demand.clip(min=0),
        'price': price.clip(min=10),
        'inventory_level': np.random.normal(loc=180, scale=40, size=n_rows).clip(min=0)
    }
    return pd.DataFrame(data)

def main():
    print("=" * 60)
    print("DATA DRIFT MONITOR TEST")
    print("=" * 60)
    
    # Generate baseline data
    baseline = generate_baseline_data()
    print(f"\nBaseline data: {len(baseline)} rows")
    print(f"  Demand mean: {baseline['demand_quantity'].mean():.2f}")
    print(f"  Price mean: {baseline['price'].mean():.2f}")
    
    # Test 1: No drift
    print("\n" + "-" * 40)
    print("Test 1: No Drift")
    print("-" * 40)
    current_no_drift = generate_drifted_data(drift_level='none')
    print(f"Current data: {len(current_no_drift)} rows")
    print(f"  Demand mean: {current_no_drift['demand_quantity'].mean():.2f}")
    
    monitor = DriftMonitor(reference_data=baseline, threshold=0.05)
    results = monitor.detect_drift(
        current_data=current_no_drift,
        columns=['demand_quantity', 'price', 'inventory_level']
    )
    
    print(f"\nResult: {'DRIFT DETECTED' if results['drift_detected'] else 'NO DRIFT'}")
    print(f"Overall drift score: {results['overall_drift_score']}")
    for col, metrics in results['columns'].items():
        status = "[DRIFT]" if metrics['drift_detected'] else "[OK]"
        print(f"  {col}: PSI={metrics['psi']:.4f}, KS p-value={metrics['ks_pvalue']:.4f} {status}")
    
    # Test 2: Moderate drift
    print("\n" + "-" * 40)
    print("Test 2: Moderate Drift")
    print("-" * 40)
    current_moderate = generate_drifted_data(drift_level='moderate')
    print(f"Current data: {len(current_moderate)} rows")
    print(f"  Demand mean: {current_moderate['demand_quantity'].mean():.2f} (baseline: 100)")
    
    monitor2 = DriftMonitor(reference_data=baseline, threshold=0.05)
    results2 = monitor2.detect_drift(
        current_data=current_moderate,
        columns=['demand_quantity', 'price', 'inventory_level']
    )
    
    print(f"\nResult: {'DRIFT DETECTED' if results2['drift_detected'] else 'NO DRIFT'}")
    print(f"Overall drift score: {results2['overall_drift_score']}")
    for col, metrics in results2['columns'].items():
        status = "[DRIFT]" if metrics['drift_detected'] else "[OK]"
        shift = metrics.get('mean_shift_pct', 0)
        print(f"  {col}: PSI={metrics['psi']:.4f}, Mean shift={shift:.1f}% {status}")
    
    # Test 3: Severe drift
    print("\n" + "-" * 40)
    print("Test 3: Severe Drift")
    print("-" * 40)
    current_severe = generate_drifted_data(drift_level='severe')
    print(f"Current data: {len(current_severe)} rows")
    print(f"  Demand mean: {current_severe['demand_quantity'].mean():.2f} (baseline: 100)")
    
    monitor3 = DriftMonitor(reference_data=baseline, threshold=0.05)
    results3 = monitor3.detect_drift(
        current_data=current_severe,
        columns=['demand_quantity', 'price', 'inventory_level']
    )
    
    print(f"\nResult: {'DRIFT DETECTED' if results3['drift_detected'] else 'NO DRIFT'}")
    print(f"Overall drift score: {results3['overall_drift_score']}")
    for col, metrics in results3['columns'].items():
        status = "[DRIFT]" if metrics['drift_detected'] else "[OK]"
        shift = metrics.get('mean_shift_pct', 0)
        print(f"  {col}: PSI={metrics['psi']:.4f}, Mean shift={shift:.1f}% {status}")
    
    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"Test 1 (No drift):       {'DRIFT DETECTED' if results['drift_detected'] else 'OK - No drift'}")
    print(f"Test 2 (Moderate drift): {'DRIFT DETECTED' if results2['drift_detected'] else 'OK - No drift'}")
    print(f"Test 3 (Severe drift):   {'DRIFT DETECTED' if results3['drift_detected'] else 'OK - No drift'}")
    
    print("\nDrift Monitor test complete!")

if __name__ == "__main__":
    main()
