"""
Data Validator Module
=====================
Comprehensive data quality validation with detailed reporting.

Author: Alexy Louis
Email: alexy.louis.scholar@gmail.com
"""

import pandas as pd
import numpy as np
import re
import logging
from typing import Dict, List, Any, Optional, Callable, Tuple
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum


class ValidationSeverity(Enum):
    """Severity levels for validation issues."""
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"


@dataclass
class ValidationResult:
    """Container for a single validation result."""
    check_name: str
    passed: bool
    severity: ValidationSeverity
    message: str
    affected_rows: int = 0
    affected_columns: List[str] = field(default_factory=list)
    details: Dict[str, Any] = field(default_factory=dict)
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())


class DataValidator:
    """
    Comprehensive data quality validator.
    
    Performs multiple validation checks including:
    - Schema validation (column names, data types)
    - Null/missing value detection
    - Duplicate detection
    - Range and constraint validation
    - Pattern matching (email, phone, etc.)
    - Referential integrity checks
    
    Example:
        >>> validator = DataValidator()
        >>> validator.add_null_check(['customer_id', 'email'])
        >>> validator.add_range_check('age', min_val=0, max_val=120)
        >>> results = validator.validate(df)
        >>> validator.generate_report()
    """
    
    def __init__(self, name: str = "DataValidator", log_level: int = logging.INFO):
        """
        Initialize the DataValidator.
        
        Args:
            name: Name for this validator instance
            log_level: Logging level
        """
        self.name = name
        self.checks: List[Callable] = []
        self.results: List[ValidationResult] = []
        self.validated_data: Optional[pd.DataFrame] = None
        
        # Setup logging
        self.logger = logging.getLogger(f"{self.__class__.__name__}.{name}")
        self.logger.setLevel(log_level)
        
        if not self.logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
            handler.setFormatter(formatter)
            self.logger.addHandler(handler)
    
    def validate(self, df: pd.DataFrame) -> List[ValidationResult]:
        """
        Run all registered validation checks.
        
        Args:
            df: DataFrame to validate
            
        Returns:
            List of ValidationResult objects
        """
        self.logger.info(f"Starting validation of {len(df):,} rows × {len(df.columns)} columns")
        self.validated_data = df
        self.results = []
        
        for check in self.checks:
            try:
                result = check(df)
                self.results.append(result)
                
                status = "✅ PASS" if result.passed else "❌ FAIL"
                self.logger.info(f"  {status} - {result.check_name}")
                
            except Exception as e:
                self.logger.error(f"Check failed with error: {str(e)}")
                self.results.append(ValidationResult(
                    check_name="Error",
                    passed=False,
                    severity=ValidationSeverity.CRITICAL,
                    message=f"Check failed: {str(e)}"
                ))
        
        passed = sum(1 for r in self.results if r.passed)
        failed = len(self.results) - passed
        
        self.logger.info(f"Validation complete: {passed} passed, {failed} failed")
        
        return self.results
    
    # =========================================================================
    # Built-in Validation Checks
    # =========================================================================
    
    def add_null_check(
        self,
        columns: List[str],
        severity: ValidationSeverity = ValidationSeverity.ERROR,
        threshold: float = 0.0
    ) -> 'DataValidator':
        """
        Add null/missing value check for specified columns.
        
        Args:
            columns: Columns to check for nulls
            severity: Severity level if check fails
            threshold: Acceptable null ratio (0.0 = no nulls allowed)
        """
        def check(df: pd.DataFrame) -> ValidationResult:
            null_counts = {}
            total_nulls = 0
            
            for col in columns:
                if col in df.columns:
                    null_count = df[col].isnull().sum()
                    if null_count > 0:
                        null_counts[col] = null_count
                        total_nulls += null_count
            
            null_ratio = total_nulls / (len(df) * len(columns)) if columns else 0
            passed = null_ratio <= threshold
            
            return ValidationResult(
                check_name=f"Null Check ({', '.join(columns[:3])}{'...' if len(columns) > 3 else ''})",
                passed=passed,
                severity=severity,
                message=f"Found {total_nulls:,} null values across {len(null_counts)} columns" if not passed else "No null values found",
                affected_rows=total_nulls,
                affected_columns=list(null_counts.keys()),
                details={'null_counts': null_counts, 'null_ratio': null_ratio}
            )
        
        self.checks.append(check)
        return self
    
    def add_duplicate_check(
        self,
        columns: Optional[List[str]] = None,
        severity: ValidationSeverity = ValidationSeverity.WARNING
    ) -> 'DataValidator':
        """
        Add duplicate row detection.
        
        Args:
            columns: Columns to check for duplicates (None = all columns)
            severity: Severity level if duplicates found
        """
        def check(df: pd.DataFrame) -> ValidationResult:
            subset = columns if columns else None
            duplicates = df.duplicated(subset=subset, keep=False)
            dup_count = duplicates.sum()
            
            return ValidationResult(
                check_name=f"Duplicate Check ({', '.join(columns) if columns else 'all columns'})",
                passed=dup_count == 0,
                severity=severity,
                message=f"Found {dup_count:,} duplicate rows" if dup_count > 0 else "No duplicates found",
                affected_rows=dup_count,
                affected_columns=columns or list(df.columns),
                details={'duplicate_count': dup_count}
            )
        
        self.checks.append(check)
        return self
    
    def add_range_check(
        self,
        column: str,
        min_val: Optional[float] = None,
        max_val: Optional[float] = None,
        severity: ValidationSeverity = ValidationSeverity.ERROR
    ) -> 'DataValidator':
        """
        Add numeric range validation.
        
        Args:
            column: Column to check
            min_val: Minimum allowed value
            max_val: Maximum allowed value
            severity: Severity level if check fails
        """
        def check(df: pd.DataFrame) -> ValidationResult:
            if column not in df.columns:
                return ValidationResult(
                    check_name=f"Range Check ({column})",
                    passed=False,
                    severity=ValidationSeverity.CRITICAL,
                    message=f"Column '{column}' not found"
                )
            
            violations = 0
            
            if min_val is not None:
                violations += (df[column] < min_val).sum()
            if max_val is not None:
                violations += (df[column] > max_val).sum()
            
            range_str = f"[{min_val}, {max_val}]"
            
            return ValidationResult(
                check_name=f"Range Check ({column}: {range_str})",
                passed=violations == 0,
                severity=severity,
                message=f"Found {violations:,} values outside range {range_str}" if violations > 0 else "All values within range",
                affected_rows=violations,
                affected_columns=[column],
                details={'min': min_val, 'max': max_val, 'violations': violations}
            )
        
        self.checks.append(check)
        return self
    
    def add_pattern_check(
        self,
        column: str,
        pattern: str,
        pattern_name: str = "pattern",
        severity: ValidationSeverity = ValidationSeverity.WARNING
    ) -> 'DataValidator':
        """
        Add regex pattern validation.
        
        Args:
            column: Column to check
            pattern: Regex pattern to match
            pattern_name: Human-readable pattern name
            severity: Severity level if check fails
        """
        def check(df: pd.DataFrame) -> ValidationResult:
            if column not in df.columns:
                return ValidationResult(
                    check_name=f"Pattern Check ({column})",
                    passed=False,
                    severity=ValidationSeverity.CRITICAL,
                    message=f"Column '{column}' not found"
                )
            
            # Convert to string and check pattern
            str_col = df[column].astype(str)
            matches = str_col.str.match(pattern, na=False)
            violations = (~matches & df[column].notna()).sum()
            
            return ValidationResult(
                check_name=f"Pattern Check ({column}: {pattern_name})",
                passed=violations == 0,
                severity=severity,
                message=f"Found {violations:,} values not matching {pattern_name}" if violations > 0 else "All values match pattern",
                affected_rows=violations,
                affected_columns=[column],
                details={'pattern': pattern, 'violations': violations}
            )
        
        self.checks.append(check)
        return self
    
    def add_email_check(
        self,
        column: str,
        severity: ValidationSeverity = ValidationSeverity.WARNING
    ) -> 'DataValidator':
        """Add email format validation."""
        email_pattern = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
        return self.add_pattern_check(column, email_pattern, "email format", severity)
    
    def add_allowed_values_check(
        self,
        column: str,
        allowed_values: List[Any],
        severity: ValidationSeverity = ValidationSeverity.ERROR
    ) -> 'DataValidator':
        """
        Add check for allowed categorical values.
        
        Args:
            column: Column to check
            allowed_values: List of allowed values
            severity: Severity level if check fails
        """
        def check(df: pd.DataFrame) -> ValidationResult:
            if column not in df.columns:
                return ValidationResult(
                    check_name=f"Allowed Values ({column})",
                    passed=False,
                    severity=ValidationSeverity.CRITICAL,
                    message=f"Column '{column}' not found"
                )
            
            invalid_mask = ~df[column].isin(allowed_values) & df[column].notna()
            violations = invalid_mask.sum()
            invalid_values = df.loc[invalid_mask, column].unique().tolist()[:10]
            
            return ValidationResult(
                check_name=f"Allowed Values ({column})",
                passed=violations == 0,
                severity=severity,
                message=f"Found {violations:,} invalid values" if violations > 0 else "All values are valid",
                affected_rows=violations,
                affected_columns=[column],
                details={
                    'allowed': allowed_values,
                    'invalid_values': invalid_values,
                    'violations': violations
                }
            )
        
        self.checks.append(check)
        return self
    
    def add_uniqueness_check(
        self,
        column: str,
        severity: ValidationSeverity = ValidationSeverity.ERROR
    ) -> 'DataValidator':
        """Add uniqueness constraint validation."""
        def check(df: pd.DataFrame) -> ValidationResult:
            if column not in df.columns:
                return ValidationResult(
                    check_name=f"Uniqueness Check ({column})",
                    passed=False,
                    severity=ValidationSeverity.CRITICAL,
                    message=f"Column '{column}' not found"
                )
            
            duplicates = df[column].duplicated().sum()
            
            return ValidationResult(
                check_name=f"Uniqueness Check ({column})",
                passed=duplicates == 0,
                severity=severity,
                message=f"Found {duplicates:,} duplicate values" if duplicates > 0 else "All values are unique",
                affected_rows=duplicates,
                affected_columns=[column],
                details={'duplicates': duplicates}
            )
        
        self.checks.append(check)
        return self
    
    def add_referential_check(
        self,
        column: str,
        reference_df: pd.DataFrame,
        reference_column: str,
        severity: ValidationSeverity = ValidationSeverity.ERROR
    ) -> 'DataValidator':
        """
        Add referential integrity check.
        
        Args:
            column: Column to check
            reference_df: Reference DataFrame
            reference_column: Column in reference DataFrame
            severity: Severity level if check fails
        """
        def check(df: pd.DataFrame) -> ValidationResult:
            valid_values = set(reference_df[reference_column].dropna())
            invalid_mask = ~df[column].isin(valid_values) & df[column].notna()
            violations = invalid_mask.sum()
            
            return ValidationResult(
                check_name=f"Referential Integrity ({column})",
                passed=violations == 0,
                severity=severity,
                message=f"Found {violations:,} values not in reference" if violations > 0 else "All references valid",
                affected_rows=violations,
                affected_columns=[column],
                details={'violations': violations}
            )
        
        self.checks.append(check)
        return self
    
    def add_custom_check(
        self,
        check_func: Callable[[pd.DataFrame], ValidationResult]
    ) -> 'DataValidator':
        """Add a custom validation function."""
        self.checks.append(check_func)
        return self
    
    # =========================================================================
    # Reporting
    # =========================================================================
    
    def generate_report(self) -> pd.DataFrame:
        """
        Generate a summary report of all validation results.
        
        Returns:
            DataFrame with validation results
        """
        if not self.results:
            return pd.DataFrame()
        
        records = []
        for result in self.results:
            records.append({
                'check_name': result.check_name,
                'passed': '✅' if result.passed else '❌',
                'severity': result.severity.value,
                'message': result.message,
                'affected_rows': result.affected_rows,
                'timestamp': result.timestamp
            })
        
        return pd.DataFrame(records)
    
    def get_summary(self) -> Dict[str, Any]:
        """Get validation summary statistics."""
        if not self.results:
            return {}
        
        passed = sum(1 for r in self.results if r.passed)
        failed = len(self.results) - passed
        
        severity_counts = {}
        for result in self.results:
            if not result.passed:
                sev = result.severity.value
                severity_counts[sev] = severity_counts.get(sev, 0) + 1
        
        return {
            'total_checks': len(self.results),
            'passed': passed,
            'failed': failed,
            'pass_rate': passed / len(self.results) * 100,
            'severity_breakdown': severity_counts,
            'total_affected_rows': sum(r.affected_rows for r in self.results if not r.passed)
        }
    
    def get_failed_checks(self) -> List[ValidationResult]:
        """Get list of failed validation checks."""
        return [r for r in self.results if not r.passed]
    
    def is_valid(self, ignore_warnings: bool = False) -> bool:
        """
        Check if data passed all validations.
        
        Args:
            ignore_warnings: If True, only consider errors and critical issues
        """
        for result in self.results:
            if not result.passed:
                if ignore_warnings and result.severity == ValidationSeverity.WARNING:
                    continue
                return False
        return True
    
    def __repr__(self) -> str:
        return f"DataValidator(name='{self.name}', checks={len(self.checks)}, results={len(self.results)})"


if __name__ == '__main__':
    # Example usage
    import pandas as pd
    
    # Sample data
    df = pd.DataFrame({
        'id': [1, 2, 2, 3, 4],
        'email': ['a@b.com', 'invalid', 'c@d.com', None, 'e@f.com'],
        'age': [25, 30, -5, 150, 40],
        'status': ['active', 'inactive', 'unknown', 'active', 'active']
    })
    
    # Create validator
    validator = DataValidator(name="TestValidator")
    
    # Add checks
    validator.add_uniqueness_check('id')
    validator.add_email_check('email')
    validator.add_range_check('age', min_val=0, max_val=120)
    validator.add_allowed_values_check('status', ['active', 'inactive'])
    validator.add_null_check(['id', 'email'])
    
    # Run validation
    results = validator.validate(df)
    
    # Print report
    print("\n" + "="*60)
    print("VALIDATION REPORT")
    print("="*60)
    print(validator.generate_report().to_string(index=False))
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    print(validator.get_summary())
