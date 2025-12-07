"""
============================================================
DATA QUALITY & VALIDATION MODULE
============================================================
Provides comprehensive data quality checks:
- Completeness (null values)
- Uniqueness (duplicates)
- Validity (data types, ranges, formats)
- Consistency (cross-field validation)
- Timeliness (date ranges)

Author: Alexy Louis
============================================================
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import re
from pipeline_logger import PipelineLogger

logger = PipelineLogger('DataQuality')


class DataQualityChecker:
    """
    Comprehensive data quality validation framework.
    """
    
    def __init__(self, df, name='Dataset'):
        """
        Initialize the quality checker.
        
        Args:
            df: pandas DataFrame to check
            name: Name of the dataset for reporting
        """
        self.df = df.copy()
        self.name = name
        self.issues = []
        self.metrics = {}
        self.passed_checks = 0
        self.failed_checks = 0
    
    def run_all_checks(self, config=None):
        """
        Run all quality checks based on configuration.
        
        Args:
            config: Dict with column-specific validation rules
            
        Returns:
            Dict with quality report
        """
        logger.info(f"Running quality checks on: {self.name}")
        logger.info(f"Dataset shape: {self.df.shape[0]:,} rows × {self.df.shape[1]} columns")
        
        # Basic checks
        self.check_completeness()
        self.check_duplicates()
        self.check_data_types()
        
        # Apply custom rules if provided
        if config:
            self.apply_validation_rules(config)
        
        return self.generate_report()
    
    def check_completeness(self, threshold=5.0):
        """Check for missing values"""
        null_counts = self.df.isnull().sum()
        null_pct = (null_counts / len(self.df) * 100).round(2)
        
        self.metrics['null_summary'] = pd.DataFrame({
            'null_count': null_counts,
            'null_percent': null_pct
        }).sort_values('null_percent', ascending=False)
        
        columns_exceeding = null_pct[null_pct > threshold]
        
        if len(columns_exceeding) == 0:
            self._log_check("Completeness check", True, f"All columns < {threshold}% null")
            self.passed_checks += 1
        else:
            for col, pct in columns_exceeding.items():
                self.issues.append({
                    'check': 'completeness',
                    'column': col,
                    'issue': f'{pct}% null values',
                    'severity': 'high' if pct > 20 else 'medium'
                })
            self._log_check("Completeness check", False, 
                          f"{len(columns_exceeding)} columns exceed {threshold}% null")
            self.failed_checks += 1
        
        return self
    
    def check_duplicates(self, subset=None, threshold=1.0):
        """Check for duplicate rows"""
        if subset:
            duplicates = self.df.duplicated(subset=subset).sum()
        else:
            duplicates = self.df.duplicated().sum()
        
        dup_pct = (duplicates / len(self.df) * 100)
        self.metrics['duplicate_count'] = duplicates
        self.metrics['duplicate_percent'] = round(dup_pct, 2)
        
        if dup_pct <= threshold:
            self._log_check("Duplicate check", True, f"{duplicates} duplicates ({dup_pct:.2f}%)")
            self.passed_checks += 1
        else:
            self.issues.append({
                'check': 'duplicates',
                'column': 'all' if not subset else str(subset),
                'issue': f'{duplicates} duplicate rows ({dup_pct:.2f}%)',
                'severity': 'high' if dup_pct > 5 else 'medium'
            })
            self._log_check("Duplicate check", False, f"{duplicates} duplicates ({dup_pct:.2f}%)")
            self.failed_checks += 1
        
        return self
    
    def check_data_types(self):
        """Analyze data types and detect potential issues"""
        type_info = {}
        for col in self.df.columns:
            type_info[col] = {
                'dtype': str(self.df[col].dtype),
                'unique_values': self.df[col].nunique(),
                'sample_values': self.df[col].dropna().head(3).tolist()
            }
        self.metrics['type_info'] = type_info
        self._log_check("Data type analysis", True, f"{len(self.df.columns)} columns analyzed")
        self.passed_checks += 1
        return self
    
    def check_range(self, column, min_val=None, max_val=None):
        """Check if values are within expected range"""
        if column not in self.df.columns:
            return self
            
        col_data = pd.to_numeric(self.df[column], errors='coerce')
        
        out_of_range = 0
        if min_val is not None:
            out_of_range += (col_data < min_val).sum()
        if max_val is not None:
            out_of_range += (col_data > max_val).sum()
        
        if out_of_range == 0:
            self._log_check(f"Range check [{column}]", True, 
                          f"All values in range [{min_val}, {max_val}]")
            self.passed_checks += 1
        else:
            self.issues.append({
                'check': 'range',
                'column': column,
                'issue': f'{out_of_range} values outside [{min_val}, {max_val}]',
                'severity': 'medium'
            })
            self._log_check(f"Range check [{column}]", False, 
                          f"{out_of_range} values out of range")
            self.failed_checks += 1
        
        return self
    
    def check_date_format(self, column, date_format='%Y-%m-%d'):
        """Validate date format"""
        if column not in self.df.columns:
            return self
            
        def is_valid_date(val):
            if pd.isna(val):
                return True
            try:
                datetime.strptime(str(val), date_format)
                return True
            except:
                return False
        
        invalid = (~self.df[column].apply(is_valid_date)).sum()
        
        if invalid == 0:
            self._log_check(f"Date format [{column}]", True, f"All dates valid")
            self.passed_checks += 1
        else:
            self.issues.append({
                'check': 'date_format',
                'column': column,
                'issue': f'{invalid} invalid date values',
                'severity': 'medium'
            })
            self._log_check(f"Date format [{column}]", False, f"{invalid} invalid dates")
            self.failed_checks += 1
        
        return self
    
    def check_regex(self, column, pattern, description='pattern'):
        """Validate values match regex pattern"""
        if column not in self.df.columns:
            return self
            
        regex = re.compile(pattern)
        
        def matches_pattern(val):
            if pd.isna(val):
                return True
            return bool(regex.match(str(val)))
        
        invalid = (~self.df[column].apply(matches_pattern)).sum()
        
        if invalid == 0:
            self._log_check(f"Pattern check [{column}]", True, f"All values match {description}")
            self.passed_checks += 1
        else:
            self.issues.append({
                'check': 'pattern',
                'column': column,
                'issue': f'{invalid} values don\'t match {description}',
                'severity': 'low'
            })
            self._log_check(f"Pattern check [{column}]", False, f"{invalid} pattern mismatches")
            self.failed_checks += 1
        
        return self
    
    def check_referential_integrity(self, column, reference_values, ref_name='reference'):
        """Check if values exist in reference set"""
        if column not in self.df.columns:
            return self
            
        non_null = self.df[column].dropna()
        invalid = (~non_null.isin(reference_values)).sum()
        
        if invalid == 0:
            self._log_check(f"Referential integrity [{column}]", True, 
                          f"All values in {ref_name}")
            self.passed_checks += 1
        else:
            self.issues.append({
                'check': 'referential_integrity',
                'column': column,
                'issue': f'{invalid} values not in {ref_name}',
                'severity': 'high'
            })
            self._log_check(f"Referential integrity [{column}]", False, 
                          f"{invalid} orphan values")
            self.failed_checks += 1
        
        return self
    
    def check_unique(self, column):
        """Check if column values are unique (for IDs)"""
        if column not in self.df.columns:
            return self
            
        duplicates = self.df[column].duplicated().sum()
        
        if duplicates == 0:
            self._log_check(f"Uniqueness [{column}]", True, "All values unique")
            self.passed_checks += 1
        else:
            self.issues.append({
                'check': 'uniqueness',
                'column': column,
                'issue': f'{duplicates} duplicate values',
                'severity': 'high'
            })
            self._log_check(f"Uniqueness [{column}]", False, f"{duplicates} duplicates")
            self.failed_checks += 1
        
        return self
    
    def check_positive(self, column):
        """Check if numeric column has only positive values"""
        if column not in self.df.columns:
            return self
            
        col_data = pd.to_numeric(self.df[column], errors='coerce')
        negative = (col_data < 0).sum()
        
        if negative == 0:
            self._log_check(f"Positive values [{column}]", True, "All values >= 0")
            self.passed_checks += 1
        else:
            self.issues.append({
                'check': 'positive',
                'column': column,
                'issue': f'{negative} negative values',
                'severity': 'high'
            })
            self._log_check(f"Positive values [{column}]", False, f"{negative} negative values")
            self.failed_checks += 1
        
        return self
    
    def apply_validation_rules(self, config):
        """Apply custom validation rules from config"""
        for column, rules in config.items():
            if column not in self.df.columns:
                continue
                
            for rule, value in rules.items():
                if rule == 'min':
                    self.check_range(column, min_val=value)
                elif rule == 'max':
                    self.check_range(column, max_val=value)
                elif rule == 'unique':
                    if value:
                        self.check_unique(column)
                elif rule == 'positive':
                    if value:
                        self.check_positive(column)
                elif rule == 'date_format':
                    self.check_date_format(column, value)
                elif rule == 'pattern':
                    self.check_regex(column, value)
        
        return self
    
    def generate_report(self):
        """Generate quality report"""
        total_checks = self.passed_checks + self.failed_checks
        score = (self.passed_checks / total_checks * 100) if total_checks > 0 else 0
        
        report = {
            'dataset': self.name,
            'rows': len(self.df),
            'columns': len(self.df.columns),
            'checks_passed': self.passed_checks,
            'checks_failed': self.failed_checks,
            'quality_score': round(score, 1),
            'issues': self.issues,
            'metrics': self.metrics
        }
        
        logger.info(f"\n{'='*50}")
        logger.info(f"QUALITY SCORE: {score:.1f}% ({self.passed_checks}/{total_checks} checks passed)")
        logger.info(f"{'='*50}")
        
        return report
    
    def _log_check(self, check_name, passed, details):
        """Log check result"""
        logger.data_quality(check_name, passed, details)


def validate_all_datasets(data):
    """
    Run validation on all extracted datasets.
    
    Args:
        data: Dict of DataFrames
        
    Returns:
        Dict of quality reports
    """
    logger.stage_start("DATA QUALITY VALIDATION")
    
    reports = {}
    
    # Validate transactions
    if 'transactions' in data:
        logger.info("\n" + "─" * 50)
        checker = DataQualityChecker(data['transactions'], 'Transactions')
        checker.check_completeness()
        checker.check_duplicates()
        checker.check_positive('quantity')
        checker.check_positive('unit_price')
        checker.check_date_format('transaction_date')
        reports['transactions'] = checker.generate_report()
    
    # Validate customers
    if 'customers' in data:
        logger.info("\n" + "─" * 50)
        checker = DataQualityChecker(data['customers'], 'Customers')
        checker.check_completeness()
        checker.check_unique('customer_id')
        checker.check_date_format('signup_date')
        checker.check_date_format('date_of_birth')
        reports['customers'] = checker.generate_report()
    
    # Validate products
    if 'products' in data:
        logger.info("\n" + "─" * 50)
        checker = DataQualityChecker(data['products'], 'Products')
        checker.check_completeness()
        checker.check_unique('product_id')
        checker.check_positive('unit_price')
        checker.check_positive('cost_price')
        checker.check_range('margin_percent', min_val=0, max_val=100)
        reports['products'] = checker.generate_report()
    
    # Validate stores
    if 'stores' in data:
        logger.info("\n" + "─" * 50)
        checker = DataQualityChecker(data['stores'], 'Stores')
        checker.check_completeness()
        checker.check_unique('store_id')
        checker.check_range('latitude', min_val=-90, max_val=90)
        checker.check_range('longitude', min_val=-180, max_val=180)
        reports['stores'] = checker.generate_report()
    
    # Validate inventory
    if 'inventory' in data:
        logger.info("\n" + "─" * 50)
        checker = DataQualityChecker(data['inventory'], 'Inventory')
        checker.check_completeness()
        checker.check_positive('current_stock')
        reports['inventory'] = checker.generate_report()
    
    # Validate suppliers
    if 'suppliers' in data:
        logger.info("\n" + "─" * 50)
        checker = DataQualityChecker(data['suppliers'], 'Suppliers')
        checker.check_completeness()
        checker.check_unique('supplier_id')
        checker.check_range('reliability_score', min_val=0, max_val=1)
        reports['suppliers'] = checker.generate_report()
    
    logger.stage_end("DATA QUALITY VALIDATION")
    
    # Summary
    logger.info("\nVALIDATION SUMMARY:")
    for name, report in reports.items():
        status = "✓" if report['quality_score'] >= 80 else "⚠" if report['quality_score'] >= 60 else "✗"
        logger.info(f"  {status} {name}: {report['quality_score']}% quality score")
    
    return reports


if __name__ == "__main__":
    # Test with sample data
    test_df = pd.DataFrame({
        'id': [1, 2, 2, 3, 4],
        'value': [10, -5, 30, 40, None],
        'date': ['2024-01-01', '2024-01-02', 'invalid', '2024-01-04', '2024-01-05']
    })
    
    checker = DataQualityChecker(test_df, 'Test Data')
    report = checker.run_all_checks()
    print("\nIssues found:", len(report['issues']))
