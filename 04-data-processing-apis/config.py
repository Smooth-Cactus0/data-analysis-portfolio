# ============================================================
# GLOBAL RETAIL DATA PIPELINE - Configuration
# ============================================================
# Author: Alexy Louis
# Project: Multi-Source ETL Pipeline for Retail Analytics
# ============================================================

import os
from datetime import datetime

# Project paths
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(PROJECT_ROOT, 'data')
RAW_DIR = os.path.join(DATA_DIR, 'raw')
PROCESSED_DIR = os.path.join(DATA_DIR, 'processed')
OUTPUT_DIR = os.path.join(DATA_DIR, 'output')
LOGS_DIR = os.path.join(PROJECT_ROOT, 'logs')

# Create directories if they don't exist
for directory in [RAW_DIR, PROCESSED_DIR, OUTPUT_DIR, LOGS_DIR]:
    os.makedirs(directory, exist_ok=True)

# Data source configurations
DATA_SOURCES = {
    'sales_transactions': {
        'type': 'csv',
        'path': os.path.join(RAW_DIR, 'sales_transactions.csv'),
        'description': 'Daily sales transactions from POS system'
    },
    'product_catalog': {
        'type': 'json',
        'path': os.path.join(RAW_DIR, 'product_catalog.json'),
        'description': 'Product master data with categories and pricing'
    },
    'customer_data': {
        'type': 'csv',
        'path': os.path.join(RAW_DIR, 'customers.csv'),
        'description': 'Customer CRM data with demographics'
    },
    'inventory_levels': {
        'type': 'xml',
        'path': os.path.join(RAW_DIR, 'inventory.xml'),
        'description': 'Current inventory levels from warehouse system'
    },
    'store_locations': {
        'type': 'json',
        'path': os.path.join(RAW_DIR, 'stores.json'),
        'description': 'Store location and metadata'
    },
    'exchange_rates': {
        'type': 'api',
        'url': 'https://api.exchangerate-api.com/v4/latest/USD',
        'description': 'Live currency exchange rates'
    },
    'supplier_data': {
        'type': 'excel',
        'path': os.path.join(RAW_DIR, 'suppliers.xlsx'),
        'description': 'Supplier contracts and lead times'
    }
}

# Pipeline configuration
PIPELINE_CONFIG = {
    'batch_size': 10000,
    'date_format': '%Y-%m-%d',
    'datetime_format': '%Y-%m-%d %H:%M:%S',
    'currency': 'USD',
    'fiscal_year_start_month': 1,
    'log_level': 'INFO'
}

# Data quality thresholds
QUALITY_THRESHOLDS = {
    'max_null_percentage': 5.0,
    'max_duplicate_percentage': 1.0,
    'min_records': 100,
    'date_range_days': 365
}

# Output configurations
OUTPUT_CONFIG = {
    'reports': {
        'sales_summary': os.path.join(OUTPUT_DIR, 'sales_summary.csv'),
        'customer_analytics': os.path.join(OUTPUT_DIR, 'customer_analytics.csv'),
        'inventory_report': os.path.join(OUTPUT_DIR, 'inventory_report.csv'),
        'executive_dashboard': os.path.join(OUTPUT_DIR, 'executive_dashboard.xlsx')
    },
    'database': {
        'type': 'sqlite',
        'path': os.path.join(OUTPUT_DIR, 'retail_analytics.db')
    }
}

print(f"✅ Configuration loaded - Project: Global Retail Data Pipeline")
print(f"   Data directory: {DATA_DIR}")
print(f"   Sources configured: {len(DATA_SOURCES)}")
