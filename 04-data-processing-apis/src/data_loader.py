"""
Data Loader Module
==================
Handles loading data from multiple sources (CSV, JSON, APIs)
with built-in error handling and logging.

Author: Alexy Louis
Email: alexy.louis.scholar@gmail.com
"""

import pandas as pd
import json
import logging
from pathlib import Path
from typing import Dict, Any, Optional, Union
from datetime import datetime
import requests


class DataLoader:
    """
    Multi-source data loader with support for CSV, JSON, and API endpoints.
    
    Features:
    - Unified interface for multiple data formats
    - Automatic encoding detection
    - Error handling and logging
    - Data source metadata tracking
    
    Example:
        >>> loader = DataLoader(base_path='data/raw')
        >>> sales_df = loader.load_csv('sales_transactions.csv')
        >>> products = loader.load_json('product_catalog.json')
    """
    
    def __init__(self, base_path: str = 'data/raw', log_level: int = logging.INFO):
        """
        Initialize the DataLoader.
        
        Args:
            base_path: Base directory for data files
            log_level: Logging level (default: INFO)
        """
        self.base_path = Path(base_path)
        self.load_history: Dict[str, Dict[str, Any]] = {}
        
        # Setup logging
        self.logger = logging.getLogger(self.__class__.__name__)
        self.logger.setLevel(log_level)
        
        if not self.logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
            handler.setFormatter(formatter)
            self.logger.addHandler(handler)
    
    def load_csv(
        self,
        filename: str,
        encoding: str = 'utf-8',
        parse_dates: Optional[list] = None,
        dtype: Optional[Dict] = None,
        **kwargs
    ) -> pd.DataFrame:
        """
        Load data from a CSV file.
        
        Args:
            filename: Name of the CSV file
            encoding: File encoding (default: utf-8)
            parse_dates: List of columns to parse as dates
            dtype: Dictionary of column data types
            **kwargs: Additional arguments for pd.read_csv
            
        Returns:
            DataFrame containing the loaded data
            
        Raises:
            FileNotFoundError: If file doesn't exist
            pd.errors.ParserError: If file cannot be parsed
        """
        filepath = self.base_path / filename
        self.logger.info(f"Loading CSV: {filepath}")
        
        start_time = datetime.now()
        
        try:
            df = pd.read_csv(
                filepath,
                encoding=encoding,
                parse_dates=parse_dates,
                dtype=dtype,
                **kwargs
            )
            
            load_time = (datetime.now() - start_time).total_seconds()
            
            # Track metadata
            self._record_load(
                source=str(filepath),
                source_type='csv',
                rows=len(df),
                columns=len(df.columns),
                load_time=load_time
            )
            
            self.logger.info(
                f"✅ Loaded {len(df):,} rows × {len(df.columns)} columns "
                f"in {load_time:.2f}s"
            )
            
            return df
            
        except FileNotFoundError:
            self.logger.error(f"File not found: {filepath}")
            raise
        except Exception as e:
            self.logger.error(f"Error loading CSV: {str(e)}")
            raise
    
    def load_json(
        self,
        filename: str,
        normalize: bool = False,
        record_path: Optional[str] = None,
        encoding: str = 'utf-8'
    ) -> Union[pd.DataFrame, Dict[str, Any]]:
        """
        Load data from a JSON file.
        
        Args:
            filename: Name of the JSON file
            normalize: If True, normalize nested JSON to DataFrame
            record_path: Path to records in nested JSON structure
            encoding: File encoding (default: utf-8)
            
        Returns:
            DataFrame if normalize=True, else dictionary
        """
        filepath = self.base_path / filename
        self.logger.info(f"Loading JSON: {filepath}")
        
        start_time = datetime.now()
        
        try:
            with open(filepath, 'r', encoding=encoding) as f:
                data = json.load(f)
            
            if normalize:
                if record_path:
                    df = pd.json_normalize(data, record_path=record_path)
                else:
                    df = pd.json_normalize(data)
                
                load_time = (datetime.now() - start_time).total_seconds()
                
                self._record_load(
                    source=str(filepath),
                    source_type='json',
                    rows=len(df),
                    columns=len(df.columns),
                    load_time=load_time
                )
                
                self.logger.info(
                    f"✅ Loaded and normalized {len(df):,} rows in {load_time:.2f}s"
                )
                
                return df
            else:
                load_time = (datetime.now() - start_time).total_seconds()
                
                self._record_load(
                    source=str(filepath),
                    source_type='json',
                    rows=len(data) if isinstance(data, list) else 1,
                    columns=None,
                    load_time=load_time
                )
                
                self.logger.info(f"✅ Loaded JSON in {load_time:.2f}s")
                
                return data
                
        except FileNotFoundError:
            self.logger.error(f"File not found: {filepath}")
            raise
        except json.JSONDecodeError as e:
            self.logger.error(f"Invalid JSON: {str(e)}")
            raise
    
    def load_from_api(
        self,
        url: str,
        method: str = 'GET',
        headers: Optional[Dict] = None,
        params: Optional[Dict] = None,
        timeout: int = 30,
        normalize: bool = True
    ) -> Union[pd.DataFrame, Dict[str, Any]]:
        """
        Load data from a REST API endpoint.
        
        Args:
            url: API endpoint URL
            method: HTTP method (GET, POST)
            headers: Request headers
            params: Query parameters
            timeout: Request timeout in seconds
            normalize: If True, convert response to DataFrame
            
        Returns:
            DataFrame if normalize=True, else dictionary
        """
        self.logger.info(f"Loading from API: {url}")
        
        start_time = datetime.now()
        
        try:
            if method.upper() == 'GET':
                response = requests.get(
                    url, headers=headers, params=params, timeout=timeout
                )
            elif method.upper() == 'POST':
                response = requests.post(
                    url, headers=headers, json=params, timeout=timeout
                )
            else:
                raise ValueError(f"Unsupported HTTP method: {method}")
            
            response.raise_for_status()
            data = response.json()
            
            load_time = (datetime.now() - start_time).total_seconds()
            
            if normalize and isinstance(data, list):
                df = pd.json_normalize(data)
                
                self._record_load(
                    source=url,
                    source_type='api',
                    rows=len(df),
                    columns=len(df.columns),
                    load_time=load_time
                )
                
                self.logger.info(
                    f"✅ Loaded {len(df):,} rows from API in {load_time:.2f}s"
                )
                
                return df
            else:
                self._record_load(
                    source=url,
                    source_type='api',
                    rows=None,
                    columns=None,
                    load_time=load_time
                )
                
                self.logger.info(f"✅ API response received in {load_time:.2f}s")
                
                return data
                
        except requests.exceptions.RequestException as e:
            self.logger.error(f"API request failed: {str(e)}")
            raise
    
    def _record_load(
        self,
        source: str,
        source_type: str,
        rows: Optional[int],
        columns: Optional[int],
        load_time: float
    ) -> None:
        """Record metadata about a data load operation."""
        self.load_history[source] = {
            'source_type': source_type,
            'rows': rows,
            'columns': columns,
            'load_time': load_time,
            'timestamp': datetime.now().isoformat()
        }
    
    def get_load_summary(self) -> pd.DataFrame:
        """
        Get a summary of all data loads.
        
        Returns:
            DataFrame with load statistics
        """
        if not self.load_history:
            return pd.DataFrame()
        
        records = []
        for source, metadata in self.load_history.items():
            records.append({
                'source': source,
                **metadata
            })
        
        return pd.DataFrame(records)
    
    def __repr__(self) -> str:
        return f"DataLoader(base_path='{self.base_path}', loads={len(self.load_history)})"


# Convenience function for quick loading
def quick_load(filepath: str, **kwargs) -> pd.DataFrame:
    """
    Quick load function for common file types.
    
    Args:
        filepath: Path to data file
        **kwargs: Additional arguments passed to loader
        
    Returns:
        DataFrame with loaded data
    """
    path = Path(filepath)
    
    if path.suffix.lower() == '.csv':
        return pd.read_csv(filepath, **kwargs)
    elif path.suffix.lower() == '.json':
        return pd.read_json(filepath, **kwargs)
    elif path.suffix.lower() in ['.xlsx', '.xls']:
        return pd.read_excel(filepath, **kwargs)
    else:
        raise ValueError(f"Unsupported file type: {path.suffix}")


if __name__ == '__main__':
    # Example usage
    loader = DataLoader(base_path='data/raw')
    
    # Load CSV
    sales = loader.load_csv('sales_transactions.csv', 
                            parse_dates=['transaction_date'])
    
    # Load JSON
    products = loader.load_json('product_catalog.json', 
                                normalize=True, 
                                record_path='products')
    
    # Print summary
    print("\n" + "="*60)
    print("LOAD SUMMARY")
    print("="*60)
    print(loader.get_load_summary())
