"""
Data Loader Module
Handles loading data from various sources (CSV, Azure Blob, etc.)
"""

import pandas as pd
from pathlib import Path
from typing import Optional, Union
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class DataLoader:
    """Load and manage spare parts demand data."""
    
    def __init__(self, data_dir: str = "data"):
        self.data_dir = Path(data_dir)
        self.raw_dir = self.data_dir / "raw"
        self.processed_dir = self.data_dir / "processed"
    
    def load_csv(self, filename: str, subfolder: str = "raw") -> pd.DataFrame:
        """Load data from CSV file."""
        if subfolder == "raw":
            filepath = self.raw_dir / filename
        else:
            filepath = self.processed_dir / filename
        
        logger.info(f"Loading data from {filepath}")
        
        if not filepath.exists():
            raise FileNotFoundError(f"File not found: {filepath}")
        
        df = pd.read_csv(filepath)
        logger.info(f"Loaded {len(df)} rows, {len(df.columns)} columns")
        
        return df
    
    def save_csv(self, df: pd.DataFrame, filename: str, subfolder: str = "processed") -> None:
        """Save DataFrame to CSV."""
        if subfolder == "raw":
            filepath = self.raw_dir / filename
        else:
            filepath = self.processed_dir / filename
        
        filepath.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(filepath, index=False)
        logger.info(f"Saved {len(df)} rows to {filepath}")
    
    def load_from_azure(
        self, 
        storage_account: str, 
        container: str, 
        blob_name: str,
        connection_string: Optional[str] = None
    ) -> pd.DataFrame:
        """Load data from Azure Blob Storage."""
        try:
            from azure.storage.blob import BlobServiceClient
            import io
            
            blob_service_client = BlobServiceClient.from_connection_string(connection_string)
            blob_client = blob_service_client.get_blob_client(container=container, blob=blob_name)
            
            download_stream = blob_client.download_blob()
            df = pd.read_csv(io.StringIO(download_stream.readall().decode('utf-8')))
            
            logger.info(f"Loaded {len(df)} rows from Azure Blob: {blob_name}")
            return df
            
        except ImportError:
            logger.error("azure-storage-blob not installed. Run: pip install azure-storage-blob")
            raise
    
    def get_sample_data(self, n_rows: int = 1000) -> pd.DataFrame:
        """Generate sample data for testing."""
        import numpy as np
        from datetime import datetime, timedelta
        
        np.random.seed(42)
        
        # Generate dates
        start_date = datetime(2022, 1, 1)
        dates = [start_date + timedelta(days=i) for i in range(n_rows)]
        
        # Generate sample data
        data = {
            'date': dates,
            'part_id': np.random.choice(['P001', 'P002', 'P003', 'P004', 'P005'], n_rows),
            'service_center': np.random.choice(['SC-North', 'SC-South', 'SC-East', 'SC-West'], n_rows),
            'category': np.random.choice(['Engine', 'Electrical', 'Body', 'Transmission'], n_rows),
            'demand_quantity': np.random.poisson(lam=50, size=n_rows) + np.random.randint(0, 20, n_rows)
        }
        
        df = pd.DataFrame(data)
        logger.info(f"Generated sample data with {len(df)} rows")
        
        return df


if __name__ == "__main__":
    # Test the data loader
    loader = DataLoader()
    sample_df = loader.get_sample_data()
    print(sample_df.head())
