"""
Data Drift Monitoring Script
Monitors for data distribution changes that may require model retraining
"""

import pandas as pd
import numpy as np
from scipy import stats
from typing import Dict, List, Optional, Tuple
import logging
from datetime import datetime, timedelta
import json
from pathlib import Path

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class DriftMonitor:
    """Monitor for data drift in demand forecasting data."""
    
    def __init__(
        self,
        reference_data: pd.DataFrame,
        threshold: float = 0.05,
        baseline_window: int = 30
    ):
        self.reference_data = reference_data
        self.threshold = threshold
        self.baseline_window = baseline_window
        self.drift_history = []
    
    def calculate_ks_statistic(
        self,
        reference: np.ndarray,
        current: np.ndarray
    ) -> Tuple[float, float]:
        """Calculate Kolmogorov-Smirnov statistic."""
        statistic, p_value = stats.ks_2samp(reference, current)
        return statistic, p_value
    
    def calculate_psi(
        self,
        reference: np.ndarray,
        current: np.ndarray,
        bins: int = 10
    ) -> float:
        """Calculate Population Stability Index."""
        # Create bins based on reference distribution
        _, bin_edges = np.histogram(reference, bins=bins)
        
        ref_counts, _ = np.histogram(reference, bins=bin_edges)
        cur_counts, _ = np.histogram(current, bins=bin_edges)
        
        # Add small constant to avoid division by zero
        ref_pct = (ref_counts + 0.001) / len(reference)
        cur_pct = (cur_counts + 0.001) / len(current)
        
        psi = np.sum((cur_pct - ref_pct) * np.log(cur_pct / ref_pct))
        return psi
    
    def detect_drift(
        self,
        current_data: pd.DataFrame,
        columns: Optional[List[str]] = None
    ) -> Dict:
        """Detect drift across specified columns."""
        if columns is None:
            columns = self.reference_data.select_dtypes(include=[np.number]).columns.tolist()
        
        results = {
            'timestamp': datetime.now().isoformat(),
            'drift_detected': False,
            'columns': {},
            'overall_drift_score': 0.0
        }
        
        drift_scores = []
        
        for col in columns:
            if col not in current_data.columns or col not in self.reference_data.columns:
                continue
            
            ref_values = self.reference_data[col].dropna().values
            cur_values = current_data[col].dropna().values
            
            if len(ref_values) == 0 or len(cur_values) == 0:
                continue
            
            # Calculate metrics
            ks_stat, ks_pvalue = self.calculate_ks_statistic(ref_values, cur_values)
            psi = self.calculate_psi(ref_values, cur_values)
            
            # Determine if drift detected
            is_drift = ks_pvalue < self.threshold or psi > 0.25
            
            results['columns'][col] = {
                'ks_statistic': round(ks_stat, 4),
                'ks_pvalue': round(ks_pvalue, 4),
                'psi': round(psi, 4),
                'drift_detected': is_drift,
                'reference_mean': round(float(np.mean(ref_values)), 2),
                'current_mean': round(float(np.mean(cur_values)), 2),
                'mean_shift_pct': round((np.mean(cur_values) - np.mean(ref_values)) / np.mean(ref_values) * 100, 2)
            }
            
            drift_scores.append(psi)
            
            if is_drift:
                results['drift_detected'] = True
        
        results['overall_drift_score'] = round(float(np.mean(drift_scores)), 4) if drift_scores else 0.0
        
        # Log and store results
        self.drift_history.append(results)
        self._log_results(results)
        
        return results
    
    def _log_results(self, results: Dict):
        """Log drift detection results."""
        if results['drift_detected']:
            logger.warning(f"DRIFT DETECTED! Overall score: {results['overall_drift_score']}")
            for col, metrics in results['columns'].items():
                if metrics['drift_detected']:
                    logger.warning(f"  - {col}: PSI={metrics['psi']}, KS p-value={metrics['ks_pvalue']}")
        else:
            logger.info(f"No drift detected. Overall score: {results['overall_drift_score']}")
    
    def get_drift_report(self) -> pd.DataFrame:
        """Generate drift report from history."""
        if not self.drift_history:
            return pd.DataFrame()
        
        rows = []
        for record in self.drift_history:
            row = {
                'timestamp': record['timestamp'],
                'drift_detected': record['drift_detected'],
                'overall_score': record['overall_drift_score']
            }
            rows.append(row)
        
        return pd.DataFrame(rows)
    
    def save_results(self, filepath: str):
        """Save drift monitoring results."""
        with open(filepath, 'w') as f:
            json.dump(self.drift_history, f, indent=2)
        logger.info(f"Results saved to {filepath}")


def monitor_demand_data(
    reference_path: str,
    current_path: str,
    output_path: str = None
) -> Dict:
    """Run drift monitoring on demand data."""
    # Load data
    reference_df = pd.read_csv(reference_path)
    current_df = pd.read_csv(current_path)
    
    # Initialize monitor
    monitor = DriftMonitor(reference_df, threshold=0.05)
    
    # Detect drift
    results = monitor.detect_drift(
        current_df,
        columns=['demand_quantity', 'revenue', 'unit_price']
    )
    
    # Save results
    if output_path:
        monitor.save_results(output_path)
    
    return results


if __name__ == "__main__":
    # Example usage
    import argparse
    
    parser = argparse.ArgumentParser(description="Monitor for data drift")
    parser.add_argument("--reference", required=True, help="Path to reference data")
    parser.add_argument("--current", required=True, help="Path to current data")
    parser.add_argument("--output", default="drift_results.json", help="Output file path")
    
    args = parser.parse_args()
    
    results = monitor_demand_data(args.reference, args.current, args.output)
    
    print("\n" + "="*50)
    print("DRIFT MONITORING RESULTS")
    print("="*50)
    print(f"Drift Detected: {results['drift_detected']}")
    print(f"Overall Score: {results['overall_drift_score']}")
    
    if results['drift_detected']:
        print("\nDrifted Columns:")
        for col, metrics in results['columns'].items():
            if metrics['drift_detected']:
                print(f"  - {col}: PSI={metrics['psi']}, Mean Shift={metrics['mean_shift_pct']}%")
