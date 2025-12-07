"""
Data Transformer Module
=======================
Comprehensive data cleaning and transformation utilities.

Author: Alexy Louis
Email: alexy.louis.scholar@gmail.com
"""

import pandas as pd
import numpy as np
import logging
from typing import Dict, List, Any, Optional, Callable, Union
from datetime import datetime
from dataclasses import dataclass


@dataclass
class TransformationLog:
    """Record of a transformation operation."""
    operation: str
    columns_affected: List[str]
    rows_before: int
    rows_after: int
    timestamp: str
    details: Dict[str, Any]


class DataTransformer:
    """
    Data transformation and cleaning pipeline.
    
    Features:
    - Chainable transformation methods
    - Automatic logging of all operations
    - Null value handling strategies
    - Duplicate removal
    - Type conversion
    - Outlier detection and handling
    - Custom transformation support
    
    Example:
        >>> transformer = DataTransformer(df)
        >>> result = (transformer
        ...     .remove_duplicates(['id'])
        ...     .handle_nulls({'age': 'median', 'name': 'drop'})
        ...     .clip_outliers('salary', method='iqr')
        ...     .standardize_text(['name', 'city'])
        ...     .get_result())
    """
    
    def __init__(self, df: pd.DataFrame, name: str = "DataTransformer"):
        """
        Initialize the DataTransformer.
        
        Args:
            df: DataFrame to transform
            name: Name for this transformer instance
        """
        self.original_df = df.copy()
        self.df = df.copy()
        self.name = name
        self.log: List[TransformationLog] = []
        
        # Setup logging
        self.logger = logging.getLogger(f"{self.__class__.__name__}.{name}")
        self.logger.setLevel(logging.INFO)
        
        if not self.logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
            handler.setFormatter(formatter)
            self.logger.addHandler(handler)
        
        self.logger.info(f"Initialized with {len(df):,} rows × {len(df.columns)} columns")
    
    def _log_operation(
        self,
        operation: str,
        columns: List[str],
        rows_before: int,
        details: Dict[str, Any] = None
    ) -> None:
        """Log a transformation operation."""
        log_entry = TransformationLog(
            operation=operation,
            columns_affected=columns,
            rows_before=rows_before,
            rows_after=len(self.df),
            timestamp=datetime.now().isoformat(),
            details=details or {}
        )
        self.log.append(log_entry)
        
        rows_changed = rows_before - len(self.df)
        if rows_changed != 0:
            self.logger.info(f"  {operation}: {abs(rows_changed):,} rows {'removed' if rows_changed > 0 else 'added'}")
        else:
            self.logger.info(f"  {operation}: completed")
    
    # =========================================================================
    # Null Handling
    # =========================================================================
    
    def drop_nulls(
        self,
        columns: Optional[List[str]] = None,
        how: str = 'any'
    ) -> 'DataTransformer':
        """
        Drop rows with null values.
        
        Args:
            columns: Columns to check (None = all)
            how: 'any' or 'all'
        """
        rows_before = len(self.df)
        self.df = self.df.dropna(subset=columns, how=how)
        
        self._log_operation(
            f"Drop nulls ({how})",
            columns or list(self.df.columns),
            rows_before
        )
        
        return self
    
    def fill_nulls(
        self,
        column: str,
        strategy: str = 'mean',
        value: Any = None
    ) -> 'DataTransformer':
        """
        Fill null values using specified strategy.
        
        Args:
            column: Column to fill
            strategy: 'mean', 'median', 'mode', 'ffill', 'bfill', 'value'
            value: Value to use if strategy='value'
        """
        rows_before = len(self.df)
        null_count = self.df[column].isnull().sum()
        
        if strategy == 'mean':
            fill_value = self.df[column].mean()
        elif strategy == 'median':
            fill_value = self.df[column].median()
        elif strategy == 'mode':
            fill_value = self.df[column].mode().iloc[0] if not self.df[column].mode().empty else None
        elif strategy == 'ffill':
            self.df[column] = self.df[column].ffill()
            fill_value = 'forward fill'
        elif strategy == 'bfill':
            self.df[column] = self.df[column].bfill()
            fill_value = 'backward fill'
        elif strategy == 'value':
            fill_value = value
        else:
            raise ValueError(f"Unknown strategy: {strategy}")
        
        if strategy not in ['ffill', 'bfill']:
            self.df[column] = self.df[column].fillna(fill_value)
        
        self._log_operation(
            f"Fill nulls ({column}: {strategy})",
            [column],
            rows_before,
            {'nulls_filled': null_count, 'fill_value': str(fill_value)}
        )
        
        return self
    
    def handle_nulls(
        self,
        strategies: Dict[str, Union[str, tuple]]
    ) -> 'DataTransformer':
        """
        Handle nulls for multiple columns with different strategies.
        
        Args:
            strategies: Dict mapping columns to strategies
                       e.g., {'age': 'median', 'name': 'drop', 'city': ('value', 'Unknown')}
        """
        for column, strategy in strategies.items():
            if column not in self.df.columns:
                self.logger.warning(f"Column '{column}' not found, skipping")
                continue
            
            if isinstance(strategy, tuple):
                self.fill_nulls(column, strategy[0], strategy[1])
            elif strategy == 'drop':
                self.drop_nulls([column])
            else:
                self.fill_nulls(column, strategy)
        
        return self
    
    # =========================================================================
    # Duplicate Handling
    # =========================================================================
    
    def remove_duplicates(
        self,
        columns: Optional[List[str]] = None,
        keep: str = 'first'
    ) -> 'DataTransformer':
        """
        Remove duplicate rows.
        
        Args:
            columns: Columns to consider (None = all)
            keep: 'first', 'last', or False
        """
        rows_before = len(self.df)
        dup_count = self.df.duplicated(subset=columns, keep=keep).sum()
        
        self.df = self.df.drop_duplicates(subset=columns, keep=keep)
        
        self._log_operation(
            f"Remove duplicates",
            columns or list(self.df.columns),
            rows_before,
            {'duplicates_removed': dup_count}
        )
        
        return self
    
    # =========================================================================
    # Outlier Handling
    # =========================================================================
    
    def clip_outliers(
        self,
        column: str,
        method: str = 'iqr',
        factor: float = 1.5
    ) -> 'DataTransformer':
        """
        Clip outliers using IQR or percentile method.
        
        Args:
            column: Column to process
            method: 'iqr' or 'percentile'
            factor: IQR multiplier (default 1.5) or percentile (e.g., 0.01 for 1st/99th)
        """
        rows_before = len(self.df)
        
        if method == 'iqr':
            Q1 = self.df[column].quantile(0.25)
            Q3 = self.df[column].quantile(0.75)
            IQR = Q3 - Q1
            lower = Q1 - factor * IQR
            upper = Q3 + factor * IQR
        elif method == 'percentile':
            lower = self.df[column].quantile(factor)
            upper = self.df[column].quantile(1 - factor)
        else:
            raise ValueError(f"Unknown method: {method}")
        
        outliers_low = (self.df[column] < lower).sum()
        outliers_high = (self.df[column] > upper).sum()
        
        self.df[column] = self.df[column].clip(lower=lower, upper=upper)
        
        self._log_operation(
            f"Clip outliers ({column})",
            [column],
            rows_before,
            {
                'method': method,
                'lower_bound': lower,
                'upper_bound': upper,
                'outliers_clipped': outliers_low + outliers_high
            }
        )
        
        return self
    
    def remove_outliers(
        self,
        column: str,
        method: str = 'iqr',
        factor: float = 1.5
    ) -> 'DataTransformer':
        """Remove rows with outlier values instead of clipping."""
        rows_before = len(self.df)
        
        if method == 'iqr':
            Q1 = self.df[column].quantile(0.25)
            Q3 = self.df[column].quantile(0.75)
            IQR = Q3 - Q1
            lower = Q1 - factor * IQR
            upper = Q3 + factor * IQR
        else:
            lower = self.df[column].quantile(factor)
            upper = self.df[column].quantile(1 - factor)
        
        self.df = self.df[(self.df[column] >= lower) & (self.df[column] <= upper)]
        
        self._log_operation(
            f"Remove outliers ({column})",
            [column],
            rows_before
        )
        
        return self
    
    # =========================================================================
    # Text Processing
    # =========================================================================
    
    def standardize_text(
        self,
        columns: List[str],
        lowercase: bool = True,
        strip: bool = True,
        remove_special: bool = False
    ) -> 'DataTransformer':
        """
        Standardize text columns.
        
        Args:
            columns: Columns to process
            lowercase: Convert to lowercase
            strip: Strip whitespace
            remove_special: Remove special characters
        """
        rows_before = len(self.df)
        
        for col in columns:
            if col not in self.df.columns:
                continue
            
            if self.df[col].dtype == 'object':
                if strip:
                    self.df[col] = self.df[col].str.strip()
                if lowercase:
                    self.df[col] = self.df[col].str.lower()
                if remove_special:
                    self.df[col] = self.df[col].str.replace(r'[^\w\s]', '', regex=True)
        
        self._log_operation(
            "Standardize text",
            columns,
            rows_before,
            {'lowercase': lowercase, 'strip': strip, 'remove_special': remove_special}
        )
        
        return self
    
    # =========================================================================
    # Type Conversion
    # =========================================================================
    
    def convert_types(
        self,
        type_map: Dict[str, str]
    ) -> 'DataTransformer':
        """
        Convert column data types.
        
        Args:
            type_map: Dict mapping columns to target types
                     e.g., {'age': 'int', 'price': 'float', 'date': 'datetime'}
        """
        rows_before = len(self.df)
        
        for col, dtype in type_map.items():
            if col not in self.df.columns:
                continue
            
            try:
                if dtype == 'datetime':
                    self.df[col] = pd.to_datetime(self.df[col], errors='coerce')
                elif dtype == 'int':
                    self.df[col] = pd.to_numeric(self.df[col], errors='coerce').astype('Int64')
                elif dtype == 'float':
                    self.df[col] = pd.to_numeric(self.df[col], errors='coerce')
                elif dtype == 'str':
                    self.df[col] = self.df[col].astype(str)
                elif dtype == 'category':
                    self.df[col] = self.df[col].astype('category')
                else:
                    self.df[col] = self.df[col].astype(dtype)
            except Exception as e:
                self.logger.warning(f"Failed to convert {col} to {dtype}: {str(e)}")
        
        self._log_operation(
            "Convert types",
            list(type_map.keys()),
            rows_before,
            {'conversions': type_map}
        )
        
        return self
    
    # =========================================================================
    # Filter Operations
    # =========================================================================
    
    def filter_rows(
        self,
        condition: Callable[[pd.DataFrame], pd.Series]
    ) -> 'DataTransformer':
        """
        Filter rows based on a condition.
        
        Args:
            condition: Function that returns boolean Series
        """
        rows_before = len(self.df)
        self.df = self.df[condition(self.df)]
        
        self._log_operation(
            "Filter rows",
            list(self.df.columns),
            rows_before
        )
        
        return self
    
    def filter_by_values(
        self,
        column: str,
        valid_values: List[Any],
        keep: bool = True
    ) -> 'DataTransformer':
        """
        Filter rows by column values.
        
        Args:
            column: Column to filter on
            valid_values: List of values
            keep: If True, keep matching rows; if False, remove them
        """
        rows_before = len(self.df)
        
        if keep:
            self.df = self.df[self.df[column].isin(valid_values)]
        else:
            self.df = self.df[~self.df[column].isin(valid_values)]
        
        self._log_operation(
            f"Filter by values ({column})",
            [column],
            rows_before
        )
        
        return self
    
    # =========================================================================
    # Column Operations
    # =========================================================================
    
    def rename_columns(
        self,
        column_map: Dict[str, str]
    ) -> 'DataTransformer':
        """Rename columns."""
        rows_before = len(self.df)
        self.df = self.df.rename(columns=column_map)
        
        self._log_operation(
            "Rename columns",
            list(column_map.keys()),
            rows_before,
            {'renames': column_map}
        )
        
        return self
    
    def select_columns(
        self,
        columns: List[str]
    ) -> 'DataTransformer':
        """Select specific columns."""
        rows_before = len(self.df)
        self.df = self.df[columns]
        
        self._log_operation(
            "Select columns",
            columns,
            rows_before
        )
        
        return self
    
    def drop_columns(
        self,
        columns: List[str]
    ) -> 'DataTransformer':
        """Drop specified columns."""
        rows_before = len(self.df)
        self.df = self.df.drop(columns=columns, errors='ignore')
        
        self._log_operation(
            "Drop columns",
            columns,
            rows_before
        )
        
        return self
    
    def add_column(
        self,
        name: str,
        func: Callable[[pd.DataFrame], pd.Series]
    ) -> 'DataTransformer':
        """
        Add a new calculated column.
        
        Args:
            name: New column name
            func: Function that takes DataFrame and returns Series
        """
        rows_before = len(self.df)
        self.df[name] = func(self.df)
        
        self._log_operation(
            f"Add column ({name})",
            [name],
            rows_before
        )
        
        return self
    
    # =========================================================================
    # Custom Operations
    # =========================================================================
    
    def apply_custom(
        self,
        func: Callable[[pd.DataFrame], pd.DataFrame],
        description: str = "Custom transformation"
    ) -> 'DataTransformer':
        """
        Apply a custom transformation function.
        
        Args:
            func: Function that takes and returns DataFrame
            description: Description of the transformation
        """
        rows_before = len(self.df)
        self.df = func(self.df)
        
        self._log_operation(
            description,
            list(self.df.columns),
            rows_before
        )
        
        return self
    
    # =========================================================================
    # Output
    # =========================================================================
    
    def get_result(self) -> pd.DataFrame:
        """Get the transformed DataFrame."""
        return self.df.copy()
    
    def get_transformation_log(self) -> pd.DataFrame:
        """Get log of all transformations."""
        if not self.log:
            return pd.DataFrame()
        
        records = []
        for entry in self.log:
            records.append({
                'operation': entry.operation,
                'columns_affected': ', '.join(entry.columns_affected[:3]) + ('...' if len(entry.columns_affected) > 3 else ''),
                'rows_before': entry.rows_before,
                'rows_after': entry.rows_after,
                'rows_changed': entry.rows_before - entry.rows_after,
                'timestamp': entry.timestamp
            })
        
        return pd.DataFrame(records)
    
    def get_summary(self) -> Dict[str, Any]:
        """Get transformation summary."""
        return {
            'original_rows': len(self.original_df),
            'final_rows': len(self.df),
            'rows_removed': len(self.original_df) - len(self.df),
            'original_columns': len(self.original_df.columns),
            'final_columns': len(self.df.columns),
            'transformations_applied': len(self.log)
        }
    
    def compare_before_after(self) -> pd.DataFrame:
        """Compare original and transformed data statistics."""
        comparison = []
        
        for col in self.df.columns:
            if col in self.original_df.columns:
                record = {
                    'column': col,
                    'dtype_before': str(self.original_df[col].dtype),
                    'dtype_after': str(self.df[col].dtype),
                    'nulls_before': self.original_df[col].isnull().sum(),
                    'nulls_after': self.df[col].isnull().sum()
                }
                
                if self.original_df[col].dtype in ['int64', 'float64']:
                    record['mean_before'] = self.original_df[col].mean()
                    record['mean_after'] = self.df[col].mean()
                
                comparison.append(record)
        
        return pd.DataFrame(comparison)
    
    def __repr__(self) -> str:
        return f"DataTransformer(name='{self.name}', rows={len(self.df)}, transformations={len(self.log)})"


if __name__ == '__main__':
    # Example usage
    df = pd.DataFrame({
        'id': [1, 2, 2, 3, 4],
        'name': ['  John  ', 'JANE', 'jane', 'Mike', None],
        'age': [25, 30, 30, -5, 150],
        'salary': [50000, 60000, 60000, 45000, 1000000]
    })
    
    print("Original DataFrame:")
    print(df)
    
    transformer = DataTransformer(df, name="Example")
    
    result = (transformer
        .remove_duplicates(['id'])
        .standardize_text(['name'])
        .clip_outliers('age', method='percentile', factor=0.05)
        .clip_outliers('salary', method='iqr')
        .fill_nulls('name', strategy='value', value='unknown')
        .get_result())
    
    print("\nTransformed DataFrame:")
    print(result)
    
    print("\nTransformation Log:")
    print(transformer.get_transformation_log())
    
    print("\nSummary:")
    print(transformer.get_summary())
