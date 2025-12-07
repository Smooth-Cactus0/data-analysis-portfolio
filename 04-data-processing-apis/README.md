# 🔄 Multi-Source Data Pipeline: ETL Framework

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![Pandas](https://img.shields.io/badge/Pandas-2.0+-green.svg)](https://pandas.pydata.org/)
[![License](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

A production-grade ETL (Extract, Transform, Load) pipeline demonstrating professional data engineering practices for processing multi-source e-commerce data.

## 📋 Table of Contents

- [Overview](#overview)
- [Features](#features)
- [Architecture](#architecture)
- [Data Sources](#data-sources)
- [Pipeline Components](#pipeline-components)
- [Results](#results)
- [Usage](#usage)
- [Project Structure](#project-structure)
- [Key Learnings](#key-learnings)
- [Author](#author)

## Overview

This project showcases a complete data engineering solution for an e-commerce company that needs to:

1. **Consolidate** data from 6 different sources (CSV, JSON, API)
2. **Validate** data quality with comprehensive business rules
3. **Clean** and transform data for downstream analytics
4. **Integrate** multiple datasets into enriched analytical tables
5. **Generate** business insights and reports

### Business Problem

> *"Our e-commerce data is scattered across multiple systems. Sales transactions are in CSV files, product catalog is in JSON, customer data comes from various sources, and we need to integrate external market data. How do we build a reliable, maintainable pipeline to process all this data?"*

### Solution

A modular, object-oriented ETL framework with:
- **DataLoader**: Multi-source data extraction
- **DataValidator**: Comprehensive quality validation
- **DataTransformer**: Chainable data cleaning operations
- **ETLPipeline**: Orchestrated workflow execution

## Features

### 🔌 Multi-Source Data Loading
- CSV files with automatic type inference
- JSON files with nested structure handling
- API endpoints with authentication support
- Automatic encoding detection
- Load time tracking and metadata

### ✅ Data Quality Validation
- **15+ validation rules** including:
  - Null/missing value detection
  - Duplicate detection
  - Range constraints
  - Pattern matching (email, phone)
  - Referential integrity
  - Custom business rules
- Severity levels (INFO, WARNING, ERROR, CRITICAL)
- Detailed validation reports

### 🔄 Data Transformation
- **Chainable API** for readable pipelines
- Null handling strategies (mean, median, mode, custom)
- Duplicate removal
- Outlier detection (IQR, percentile)
- Text standardization
- Type conversion
- Custom transformations
- Complete audit trail

### 📊 Pipeline Orchestration
- Step-based execution
- Error handling and recovery
- Execution metrics
- Comprehensive logging
- JSON report generation

## Architecture

![Pipeline Architecture](images/04_pipeline_architecture.png)

```
┌─────────────────────────────────────────────────────────────────────┐
│                           EXTRACT                                    │
├──────────┬──────────┬──────────┬──────────┬──────────┬──────────────┤
│  Sales   │ Products │ Customers│ Inventory│  Market  │  Suppliers   │
│  (CSV)   │  (JSON)  │  (CSV)   │  (CSV)   │  (API)   │   (JSON)     │
└────┬─────┴────┬─────┴────┬─────┴────┬─────┴────┬─────┴──────┬───────┘
     │          │          │          │          │            │
     └──────────┴──────────┴────┬─────┴──────────┴────────────┘
                                │
                                ▼
┌─────────────────────────────────────────────────────────────────────┐
│                    TRANSFORM & VALIDATE                              │
│  • Data Quality Checks    • Null Handling      • Deduplication      │
│  • Type Conversion        • Outlier Removal    • Enrichment         │
└─────────────────────────────────┬───────────────────────────────────┘
                                  │
                                  ▼
┌─────────────────────────────────────────────────────────────────────┐
│                            LOAD                                      │
├──────────────┬──────────────┬──────────────┬──────────────┬─────────┤
│    Sales     │    Sales     │  Customers   │  Inventory   │ Reports │
│   Cleaned    │   Enriched   │   Cleaned    │   Cleaned    │  JSON   │
└──────────────┴──────────────┴──────────────┴──────────────┴─────────┘
```

## Data Sources

| Source | Format | Records | Description |
|--------|--------|---------|-------------|
| Sales Transactions | CSV | 5,025 | Order data with intentional quality issues |
| Product Catalog | JSON | 150 | Product master data with nested attributes |
| Customers | CSV | 1,200 | Customer demographics and segments |
| Inventory | CSV | 450 | Warehouse stock levels |
| Market Data | JSON | - | External API simulation (trends, competitors) |
| Suppliers | JSON | 20 | Supplier information |

### Data Quality Issues (Intentional)

To demonstrate real-world data cleaning:
- **50+ missing values** across datasets
- **25 duplicate** transactions
- **30 invalid emails** in customer data
- **20 negative values** in quantities
- **10 invalid warehouse IDs**

## Pipeline Components

### DataLoader

```python
from src.data_loader import DataLoader

loader = DataLoader(base_path='data/raw')

# Load CSV with date parsing
sales = loader.load_csv('sales.csv', parse_dates=['transaction_date'])

# Load and normalize JSON
products = loader.load_json('products.json', normalize=True, record_path='products')

# Get load summary
print(loader.get_load_summary())
```

### DataValidator

```python
from src.data_validator import DataValidator, ValidationSeverity

validator = DataValidator(name="SalesValidator")

# Chain validation rules
validator.add_uniqueness_check('transaction_id')
validator.add_null_check(['customer_id', 'product_id'])
validator.add_range_check('quantity', min_val=1, max_val=1000)
validator.add_email_check('email')
validator.add_allowed_values_check('status', ['Active', 'Inactive'])

# Run validation
results = validator.validate(df)

# Generate report
print(validator.generate_report())
```

### DataTransformer

```python
from src.data_transformer import DataTransformer

# Chainable transformations
transformer = DataTransformer(df, name="SalesTransformer")

clean_df = (transformer
    .remove_duplicates(['transaction_id'])
    .filter_rows(lambda df: df['quantity'] > 0)
    .fill_nulls('payment_method', strategy='mode')
    .clip_outliers('price', method='iqr')
    .add_column('total', lambda df: df['quantity'] * df['price'])
    .standardize_text(['category'])
    .get_result()
)

# View transformation log
print(transformer.get_transformation_log())
```

### ETLPipeline (Orchestrator)

```python
from src.pipeline_orchestrator import ETLPipeline

pipeline = ETLPipeline(
    name="EcommercePipeline",
    input_path="data/raw",
    output_path="data/processed"
)

# Define pipeline
pipeline.add_extract_csv('sales', 'sales.csv')
pipeline.add_validation('sales', sales_validator)
pipeline.add_transform('sales', clean_function)
pipeline.add_load_csv('sales', 'sales_cleaned.csv')

# Execute
result = pipeline.run()
print(f"Status: {result.status}")
```

## Results

### Data Quality Improvements

![Data Quality](images/01_data_quality_overview.png)

| Metric | Before | After | Improvement |
|--------|--------|-------|-------------|
| Sales Records | 5,025 | 4,985 | 40 invalid removed |
| Customer Records | 1,200 | 1,180 | 20 invalid removed |
| Inventory Records | 450 | 440 | 10 invalid removed |
| Missing Values | 150+ | 0 | 100% resolved |
| Duplicates | 25 | 0 | 100% resolved |

### Validation Summary

![Validation Results](images/02_validation_results.png)

### Business Analytics

![Sales Analytics](images/03_sales_analytics.png)

**Key Metrics (Completed Orders):**
- **Total Revenue**: $2.4M+
- **Average Order Value**: $620
- **Top Category**: Electronics (35% of revenue)
- **Top Channel**: Website (45% of transactions)

### Transformation Summary

![Transformation Summary](images/05_transformation_summary.png)

## Usage

### Prerequisites

```bash
pip install pandas numpy matplotlib seaborn
```

### Quick Start

```python
# 1. Clone and navigate to project
cd 04-data-processing-apis

# 2. Generate sample data (if needed)
python src/generate_data.py

# 3. Run the pipeline
cd notebooks
jupyter notebook etl_pipeline_demo.ipynb
```

### Running as Script

```python
import sys
sys.path.insert(0, 'src')

from data_loader import DataLoader
from data_validator import DataValidator
from data_transformer import DataTransformer

# Load
loader = DataLoader(base_path='data/raw')
df = loader.load_csv('sales_transactions.csv')

# Validate
validator = DataValidator()
validator.add_null_check(['transaction_id'])
validator.validate(df)

# Transform
transformer = DataTransformer(df)
clean_df = transformer.remove_duplicates().get_result()

# Save
clean_df.to_csv('data/processed/sales_cleaned.csv', index=False)
```

## Project Structure

```
04-data-processing-apis/
│
├── README.md                    # This file
├── PROGRESS.md                  # Development progress tracker
│
├── data/
│   ├── raw/                     # Source data files
│   │   ├── sales_transactions.csv
│   │   ├── product_catalog.json
│   │   ├── customers.csv
│   │   ├── inventory.csv
│   │   └── suppliers.json
│   │
│   ├── external/               # External API data
│   │   └── market_data.json
│   │
│   └── processed/              # Output files
│       ├── sales_cleaned.csv
│       ├── sales_enriched.csv
│       ├── customers_cleaned.csv
│       ├── inventory_cleaned.csv
│       ├── inventory_summary.csv
│       ├── category_analytics.csv
│       └── pipeline_report.json
│
├── src/                        # Source code modules
│   ├── __init__.py
│   ├── data_loader.py          # Multi-source data loading
│   ├── data_validator.py       # Data quality validation
│   ├── data_transformer.py     # Data transformation
│   ├── pipeline_orchestrator.py # ETL orchestration
│   └── generate_data.py        # Sample data generator
│
├── notebooks/
│   └── etl_pipeline_demo.ipynb # Main demonstration notebook
│
├── images/                     # Generated visualizations
│   ├── 01_data_quality_overview.png
│   ├── 02_validation_results.png
│   ├── 03_sales_analytics.png
│   ├── 04_pipeline_architecture.png
│   └── 05_transformation_summary.png
│
└── docs/                       # Additional documentation
    └── api_reference.md        # API documentation
```

## Key Learnings

### Technical Skills Demonstrated

1. **Object-Oriented Design**
   - Clean separation of concerns
   - Reusable, testable components
   - Fluent/chainable APIs

2. **Data Engineering Best Practices**
   - Comprehensive logging and audit trails
   - Error handling and graceful degradation
   - Configuration management

3. **ETL Patterns**
   - Multi-source data integration
   - Validation-first approach
   - Incremental transformation

4. **Software Engineering**
   - Type hints for documentation
   - Docstrings for API clarity
   - Modular architecture

### Business Value

- **Data Quality**: Automated validation catches issues before they propagate
- **Reliability**: Logged transformations provide audit trail
- **Maintainability**: Modular design allows easy updates
- **Scalability**: Can be extended for larger datasets or new sources

## Technologies Used

- **Python 3.8+**
- **Pandas** - Data manipulation
- **NumPy** - Numerical operations
- **Matplotlib/Seaborn** - Visualization
- **JSON** - Configuration and data exchange
- **Logging** - Audit trails

## Future Enhancements

- [ ] Add database connectors (PostgreSQL, MongoDB)
- [ ] Implement incremental loading
- [ ] Add data lineage tracking
- [ ] Create REST API for pipeline execution
- [ ] Add unit tests with pytest
- [ ] Implement parallel processing for large datasets

## Author

**Alexy Louis**

- 📧 Email: alexy.louis.scholar@gmail.com
- 💼 LinkedIn: [linkedin.com/in/alexy-louis-19a5a9262](https://www.linkedin.com/in/alexy-louis-19a5a9262/)
- 🐙 GitHub: [github.com/Smooth-Cactus0](https://github.com/Smooth-Cactus0)

---

## License

This project is licensed under the MIT License - see the [LICENSE](../LICENSE) file for details.

---

*Part of the [Data Analysis Portfolio](../README.md) project collection.*
