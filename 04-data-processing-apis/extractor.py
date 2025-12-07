"""
============================================================
DATA EXTRACTION MODULE
============================================================
Extracts data from multiple sources:
- CSV files
- JSON files
- XML files
- Excel files
- REST APIs

Author: Alexy Louis
============================================================
"""

import pandas as pd
import json
import xml.etree.ElementTree as ET
import requests
import os
from datetime import datetime
from pipeline_logger import PipelineLogger, log_execution_time

logger = PipelineLogger('Extractor')


class DataExtractor:
    """
    Multi-source data extractor supporting CSV, JSON, XML, Excel, and APIs.
    """
    
    def __init__(self, base_path=None):
        """
        Initialize the extractor.
        
        Args:
            base_path: Base directory for file sources
        """
        self.base_path = base_path or os.path.dirname(os.path.abspath(__file__))
        self.extraction_stats = {}
        logger.info(f"DataExtractor initialized with base path: {self.base_path}")
    
    def extract_csv(self, filepath, **kwargs):
        """
        Extract data from CSV file.
        
        Args:
            filepath: Path to CSV file
            **kwargs: Additional pandas read_csv arguments
            
        Returns:
            pandas DataFrame
        """
        full_path = os.path.join(self.base_path, filepath) if not os.path.isabs(filepath) else filepath
        logger.info(f"Extracting CSV: {os.path.basename(filepath)}")
        
        try:
            df = pd.read_csv(full_path, **kwargs)
            self._log_extraction_stats('csv', filepath, len(df), len(df.columns))
            return df
        except Exception as e:
            logger.error(f"Failed to extract CSV {filepath}: {str(e)}")
            raise
    
    def extract_json(self, filepath, record_path=None):
        """
        Extract data from JSON file.
        
        Args:
            filepath: Path to JSON file
            record_path: Path to records in JSON structure (e.g., 'products')
            
        Returns:
            pandas DataFrame
        """
        full_path = os.path.join(self.base_path, filepath) if not os.path.isabs(filepath) else filepath
        logger.info(f"Extracting JSON: {os.path.basename(filepath)}")
        
        try:
            with open(full_path, 'r') as f:
                data = json.load(f)
            
            # Navigate to record path if specified
            if record_path:
                for key in record_path.split('.'):
                    data = data[key]
            
            df = pd.DataFrame(data)
            self._log_extraction_stats('json', filepath, len(df), len(df.columns))
            return df
        except Exception as e:
            logger.error(f"Failed to extract JSON {filepath}: {str(e)}")
            raise
    
    def extract_xml(self, filepath, record_tag='item'):
        """
        Extract data from XML file.
        
        Args:
            filepath: Path to XML file
            record_tag: XML tag containing each record
            
        Returns:
            pandas DataFrame
        """
        full_path = os.path.join(self.base_path, filepath) if not os.path.isabs(filepath) else filepath
        logger.info(f"Extracting XML: {os.path.basename(filepath)}")
        
        try:
            tree = ET.parse(full_path)
            root = tree.getroot()
            
            records = []
            for item in root.iter(record_tag):
                record = {}
                for child in item:
                    record[child.tag] = child.text
                records.append(record)
            
            df = pd.DataFrame(records)
            self._log_extraction_stats('xml', filepath, len(df), len(df.columns))
            return df
        except Exception as e:
            logger.error(f"Failed to extract XML {filepath}: {str(e)}")
            raise
    
    def extract_excel(self, filepath, sheet_name=0, **kwargs):
        """
        Extract data from Excel file.
        
        Args:
            filepath: Path to Excel file
            sheet_name: Sheet name or index
            **kwargs: Additional pandas read_excel arguments
            
        Returns:
            pandas DataFrame
        """
        full_path = os.path.join(self.base_path, filepath) if not os.path.isabs(filepath) else filepath
        logger.info(f"Extracting Excel: {os.path.basename(filepath)} (sheet: {sheet_name})")
        
        try:
            df = pd.read_excel(full_path, sheet_name=sheet_name, **kwargs)
            self._log_extraction_stats('excel', filepath, len(df), len(df.columns))
            return df
        except Exception as e:
            logger.error(f"Failed to extract Excel {filepath}: {str(e)}")
            raise
    
    def extract_api(self, url, params=None, headers=None, json_path=None):
        """
        Extract data from REST API.
        
        Args:
            url: API endpoint URL
            params: Query parameters
            headers: Request headers
            json_path: Path to data in JSON response
            
        Returns:
            pandas DataFrame or dict
        """
        logger.info(f"Extracting from API: {url}")
        
        try:
            response = requests.get(url, params=params, headers=headers, timeout=30)
            response.raise_for_status()
            
            data = response.json()
            
            # Navigate to data path if specified
            if json_path:
                for key in json_path.split('.'):
                    data = data[key]
            
            # Convert to DataFrame if possible
            if isinstance(data, list):
                df = pd.DataFrame(data)
                self._log_extraction_stats('api', url, len(df), len(df.columns))
                return df
            else:
                self._log_extraction_stats('api', url, 1, len(data) if isinstance(data, dict) else 0)
                return data
                
        except requests.exceptions.RequestException as e:
            logger.warning(f"API request failed: {str(e)}")
            return None
        except Exception as e:
            logger.error(f"Failed to extract from API {url}: {str(e)}")
            raise
    
    def _log_extraction_stats(self, source_type, source_name, rows, cols):
        """Log extraction statistics"""
        source_key = os.path.basename(source_name)
        self.extraction_stats[source_key] = {
            'type': source_type,
            'rows': rows,
            'columns': cols,
            'extracted_at': datetime.now().isoformat()
        }
        logger.metric(f"Rows extracted", rows)
        logger.metric(f"Columns", cols)
    
    def get_extraction_summary(self):
        """Get summary of all extractions"""
        return pd.DataFrame(self.extraction_stats).T


def extract_all_sources(raw_dir):
    """
    Extract data from all configured sources.
    
    Args:
        raw_dir: Directory containing raw data files
        
    Returns:
        dict of DataFrames
    """
    extractor = DataExtractor(raw_dir)
    data = {}
    
    logger.stage_start("DATA EXTRACTION")
    
    # 1. Extract stores (JSON)
    logger.info("─" * 40)
    data['stores'] = extractor.extract_json('stores.json', record_path='stores')
    
    # 2. Extract products (JSON)
    logger.info("─" * 40)
    data['products'] = extractor.extract_json('product_catalog.json', record_path='products')
    
    # 3. Extract customers (CSV)
    logger.info("─" * 40)
    data['customers'] = extractor.extract_csv('customers.csv')
    
    # 4. Extract transactions (CSV)
    logger.info("─" * 40)
    data['transactions'] = extractor.extract_csv('sales_transactions.csv')
    
    # 5. Extract inventory (XML)
    logger.info("─" * 40)
    data['inventory'] = extractor.extract_xml('inventory.xml', record_tag='item')
    
    # 6. Extract suppliers (Excel)
    logger.info("─" * 40)
    data['suppliers'] = extractor.extract_excel('suppliers.xlsx', sheet_name='Suppliers')
    
    # 7. Extract exchange rates (API) - with fallback
    logger.info("─" * 40)
    try:
        exchange_data = extractor.extract_api(
            'https://api.exchangerate-api.com/v4/latest/USD'
        )
        if exchange_data is not None:
            data['exchange_rates'] = exchange_data
        else:
            # Fallback data
            logger.warning("Using fallback exchange rates")
            data['exchange_rates'] = {
                'base': 'USD',
                'rates': {'EUR': 0.92, 'GBP': 0.79, 'JPY': 149.50, 'CNY': 7.24}
            }
    except Exception:
        logger.warning("API extraction failed, using fallback data")
        data['exchange_rates'] = {
            'base': 'USD',
            'rates': {'EUR': 0.92, 'GBP': 0.79, 'JPY': 149.50, 'CNY': 7.24}
        }
    
    logger.stage_end("DATA EXTRACTION")
    
    # Log summary
    logger.info("\nEXTRACTION SUMMARY:")
    for name, df in data.items():
        if isinstance(df, pd.DataFrame):
            logger.info(f"  {name}: {len(df):,} rows × {len(df.columns)} columns")
        else:
            logger.info(f"  {name}: dict with {len(df)} keys")
    
    return data, extractor.get_extraction_summary()


if __name__ == "__main__":
    # Test extraction
    raw_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'data', 'raw')
    data, summary = extract_all_sources(raw_dir)
    print("\n" + summary.to_string())
