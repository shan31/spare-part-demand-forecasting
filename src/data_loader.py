"""
Data Loader Module
Handles loading data from various sources (CSV, Azure Blob, SQL, API)
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Optional, Union, Dict, List, Any
from datetime import datetime, timedelta
import logging
import os
import json

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class DataLoader:
    """Load and manage spare parts demand data from multiple sources."""
    
    def __init__(self, data_dir: str = "data"):
        self.data_dir = Path(data_dir)
        self.raw_dir = self.data_dir / "raw"
        self.processed_dir = self.data_dir / "processed"
        self._ensure_directories()
    
    def _ensure_directories(self):
        """Create data directories if they don't exist."""
        self.raw_dir.mkdir(parents=True, exist_ok=True)
        self.processed_dir.mkdir(parents=True, exist_ok=True)
    
    # ==================== LOCAL FILE OPERATIONS ====================
    
    def load_csv(self, filename: str, subfolder: str = "raw") -> pd.DataFrame:
        """Load data from CSV file."""
        if subfolder == "raw":
            filepath = self.raw_dir / filename
        else:
            filepath = self.processed_dir / filename
        
        logger.info(f"Loading data from {filepath}")
        
        if not filepath.exists():
            raise FileNotFoundError(f"File not found: {filepath}")
        
        df = pd.read_csv(filepath, parse_dates=['date'] if 'date' in pd.read_csv(filepath, nrows=0).columns else None)
        logger.info(f"Loaded {len(df)} rows, {len(df.columns)} columns")
        
        return df
    
    def save_csv(self, df: pd.DataFrame, filename: str, subfolder: str = "processed") -> str:
        """Save DataFrame to CSV."""
        if subfolder == "raw":
            filepath = self.raw_dir / filename
        else:
            filepath = self.processed_dir / filename
        
        filepath.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(filepath, index=False)
        logger.info(f"Saved {len(df)} rows to {filepath}")
        return str(filepath)
    
    # ==================== AZURE BLOB STORAGE ====================
    
    def load_from_azure_blob(
        self, 
        container: str, 
        blob_name: str,
        connection_string: Optional[str] = None
    ) -> pd.DataFrame:
        """Load data from Azure Blob Storage."""
        try:
            from azure.storage.blob import BlobServiceClient
            import io
            
            conn_str = connection_string or os.getenv("AZURE_STORAGE_CONNECTION_STRING")
            if not conn_str:
                raise ValueError("Azure Storage connection string not provided")
            
            blob_service_client = BlobServiceClient.from_connection_string(conn_str)
            blob_client = blob_service_client.get_blob_client(container=container, blob=blob_name)
            
            download_stream = blob_client.download_blob()
            content = download_stream.readall().decode('utf-8')
            
            # Detect file type
            if blob_name.endswith('.json'):
                df = pd.read_json(io.StringIO(content))
            elif blob_name.endswith('.parquet'):
                df = pd.read_parquet(io.BytesIO(download_stream.readall()))
            else:
                df = pd.read_csv(io.StringIO(content))
            
            logger.info(f"Loaded {len(df)} rows from Azure Blob: {container}/{blob_name}")
            return df
            
        except ImportError:
            logger.error("azure-storage-blob not installed. Run: pip install azure-storage-blob")
            raise
    
    def save_to_azure_blob(
        self,
        df: pd.DataFrame,
        container: str,
        blob_name: str,
        connection_string: Optional[str] = None
    ) -> str:
        """Save DataFrame to Azure Blob Storage."""
        try:
            from azure.storage.blob import BlobServiceClient
            import io
            
            conn_str = connection_string or os.getenv("AZURE_STORAGE_CONNECTION_STRING")
            if not conn_str:
                raise ValueError("Azure Storage connection string not provided")
            
            blob_service_client = BlobServiceClient.from_connection_string(conn_str)
            blob_client = blob_service_client.get_blob_client(container=container, blob=blob_name)
            
            # Convert to appropriate format
            if blob_name.endswith('.json'):
                content = df.to_json(orient='records')
            elif blob_name.endswith('.parquet'):
                buffer = io.BytesIO()
                df.to_parquet(buffer, index=False)
                content = buffer.getvalue()
            else:
                content = df.to_csv(index=False)
            
            blob_client.upload_blob(content, overwrite=True)
            logger.info(f"Saved {len(df)} rows to Azure Blob: {container}/{blob_name}")
            return f"azure://{container}/{blob_name}"
            
        except ImportError:
            logger.error("azure-storage-blob not installed")
            raise
    
    def list_azure_blobs(self, container: str, prefix: str = "") -> List[str]:
        """List blobs in Azure container."""
        try:
            from azure.storage.blob import BlobServiceClient
            
            conn_str = os.getenv("AZURE_STORAGE_CONNECTION_STRING")
            blob_service_client = BlobServiceClient.from_connection_string(conn_str)
            container_client = blob_service_client.get_container_client(container)
            
            blobs = [blob.name for blob in container_client.list_blobs(name_starts_with=prefix)]
            return blobs
        except Exception as e:
            logger.error(f"Error listing blobs: {e}")
            return []
    
    # ==================== SQL DATABASE ====================
    
    def load_from_sql(
        self,
        query: str,
        connection_string: Optional[str] = None
    ) -> pd.DataFrame:
        """Load data from SQL database."""
        try:
            import pyodbc
            
            conn_str = connection_string or os.getenv("SQL_CONNECTION_STRING")
            if not conn_str:
                raise ValueError("SQL connection string not provided")
            
            conn = pyodbc.connect(conn_str)
            df = pd.read_sql(query, conn)
            conn.close()
            
            logger.info(f"Loaded {len(df)} rows from SQL")
            return df
            
        except ImportError:
            logger.error("pyodbc not installed. Run: pip install pyodbc")
            raise
    
    # ==================== REST API ====================
    
    def load_from_api(
        self,
        url: str,
        headers: Optional[Dict] = None,
        params: Optional[Dict] = None,
        auth_token: Optional[str] = None
    ) -> pd.DataFrame:
        """Load data from REST API."""
        import requests
        
        if auth_token:
            headers = headers or {}
            headers["Authorization"] = f"Bearer {auth_token}"
        
        response = requests.get(url, headers=headers, params=params)
        response.raise_for_status()
        
        data = response.json()
        
        # Handle nested data
        if isinstance(data, dict):
            if 'data' in data:
                data = data['data']
            elif 'results' in data:
                data = data['results']
        
        df = pd.DataFrame(data)
        logger.info(f"Loaded {len(df)} rows from API: {url}")
        return df
    
    # ==================== DATA VALIDATION ====================
    
    def validate_data(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Validate demand data and return quality report."""
        report = {
            "total_rows": len(df),
            "total_columns": len(df.columns),
            "missing_values": df.isnull().sum().to_dict(),
            "duplicate_rows": df.duplicated().sum(),
            "data_types": df.dtypes.astype(str).to_dict(),
            "issues": []
        }
        
        # Check required columns
        required_cols = ['date', 'demand_quantity']
        for col in required_cols:
            if col not in df.columns:
                report["issues"].append(f"Missing required column: {col}")
        
        # Check for negative demand
        if 'demand_quantity' in df.columns:
            neg_count = (df['demand_quantity'] < 0).sum()
            if neg_count > 0:
                report["issues"].append(f"Negative demand values: {neg_count}")
        
        # Check date range
        if 'date' in df.columns:
            report["date_range"] = {
                "min": str(df['date'].min()),
                "max": str(df['date'].max())
            }
        
        report["is_valid"] = len(report["issues"]) == 0
        return report
    
    def clean_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Clean and preprocess demand data."""
        df = df.copy()
        
        # Convert date column
        if 'date' in df.columns:
            df['date'] = pd.to_datetime(df['date'])
        
        # Remove duplicates
        df = df.drop_duplicates()
        
        # Handle missing values
        if 'demand_quantity' in df.columns:
            df['demand_quantity'] = df['demand_quantity'].fillna(0)
            df['demand_quantity'] = df['demand_quantity'].clip(lower=0)
        
        # Sort by date
        if 'date' in df.columns:
            df = df.sort_values('date').reset_index(drop=True)
        
        logger.info(f"Cleaned data: {len(df)} rows remaining")
        return df
    
    # ==================== SAMPLE DATA ====================
    
    def get_sample_data(self, n_rows: int = 1000) -> pd.DataFrame:
        """Generate sample data for testing."""
        np.random.seed(42)
        
        start_date = datetime(2022, 1, 1)
        dates = [start_date + timedelta(days=i) for i in range(n_rows)]
        
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
    
    # ==================== UNIFIED LOAD ====================
    
    def load(self, source: str, **kwargs) -> pd.DataFrame:
        """
        Unified data loading from any source.
        
        Args:
            source: Data source path or URL
                - Local: "data/raw/file.csv"
                - Azure: "azure://container/blob.csv"
                - SQL: "sql://query" 
                - API: "https://api.example.com/data"
                - Sample: "sample:1000"
        """
        if source.startswith("azure://"):
            parts = source[8:].split("/", 1)
            return self.load_from_azure_blob(container=parts[0], blob_name=parts[1], **kwargs)
        
        elif source.startswith("sql://"):
            query = source[6:]
            return self.load_from_sql(query=query, **kwargs)
        
        elif source.startswith("http://") or source.startswith("https://"):
            return self.load_from_api(url=source, **kwargs)
        
        elif source.startswith("sample:"):
            n_rows = int(source[7:])
            return self.get_sample_data(n_rows=n_rows)
        
        else:
            # Assume local file
            return self.load_csv(filename=source, **kwargs)


if __name__ == "__main__":
    # Test the data loader
    loader = DataLoader()
    
    # Test sample data
    print("=" * 50)
    print("Testing Sample Data Generation:")
    sample_df = loader.get_sample_data(100)
    print(sample_df.head())
    
    # Test validation
    print("\n" + "=" * 50)
    print("Testing Data Validation:")
    report = loader.validate_data(sample_df)
    print(json.dumps(report, indent=2, default=str))
    
    # Test unified loader
    print("\n" + "=" * 50)
    print("Testing Unified Loader:")
    df = loader.load("sample:50")
    print(f"Loaded {len(df)} rows via unified loader")

