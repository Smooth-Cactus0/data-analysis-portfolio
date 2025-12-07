"""
Pipeline Orchestrator Module
============================
Orchestrates the complete ETL pipeline workflow.

Author: Alexy Louis
Email: alexy.louis.scholar@gmail.com
"""

import pandas as pd
import numpy as np
import json
import logging
from pathlib import Path
from typing import Dict, List, Any, Optional, Callable
from datetime import datetime
from dataclasses import dataclass
from enum import Enum

from data_loader import DataLoader
from data_validator import DataValidator, ValidationSeverity, ValidationResult
from data_transformer import DataTransformer


class PipelineStatus(Enum):
    """Pipeline execution status."""
    NOT_STARTED = "not_started"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    COMPLETED_WITH_WARNINGS = "completed_with_warnings"


@dataclass
class PipelineStep:
    """Represents a single pipeline step."""
    name: str
    function: Callable
    args: tuple = ()
    kwargs: Dict = None
    required: bool = True
    
    def __post_init__(self):
        if self.kwargs is None:
            self.kwargs = {}


@dataclass
class PipelineResult:
    """Container for pipeline execution results."""
    status: PipelineStatus
    start_time: datetime
    end_time: datetime
    duration_seconds: float
    steps_completed: int
    steps_total: int
    data_sources_loaded: int
    rows_processed: int
    rows_output: int
    validation_passed: bool
    warnings: List[str]
    errors: List[str]
    output_files: List[str]


class ETLPipeline:
    """
    Complete ETL Pipeline Orchestrator.
    
    Coordinates the entire data processing workflow:
    1. Extract - Load data from multiple sources
    2. Transform - Clean, validate, and transform data
    3. Load - Save processed data to destination
    
    Features:
    - Modular step-based execution
    - Comprehensive logging
    - Error handling and recovery
    - Execution metrics and reporting
    - Support for multiple data sources
    - Automated validation
    
    Example:
        >>> pipeline = ETLPipeline(name="SalesPipeline")
        >>> pipeline.add_extract_step("sales", "sales.csv", "csv")
        >>> pipeline.add_transform_step("clean", clean_function)
        >>> pipeline.add_load_step("output", "processed_sales.csv")
        >>> result = pipeline.run()
    """
    
    def __init__(
        self,
        name: str = "ETLPipeline",
        input_path: str = "data/raw",
        output_path: str = "data/processed",
        log_level: int = logging.INFO
    ):
        """
        Initialize the ETL Pipeline.
        
        Args:
            name: Pipeline name
            input_path: Path for input data files
            output_path: Path for output data files
            log_level: Logging level
        """
        self.name = name
        self.input_path = Path(input_path)
        self.output_path = Path(output_path)
        self.status = PipelineStatus.NOT_STARTED
        
        # Components
        self.loader = DataLoader(base_path=str(input_path), log_level=log_level)
        self.validators: Dict[str, DataValidator] = {}
        self.transformers: Dict[str, DataTransformer] = {}
        
        # Data storage
        self.data: Dict[str, pd.DataFrame] = {}
        self.metadata: Dict[str, Any] = {}
        
        # Pipeline steps
        self.extract_steps: List[PipelineStep] = []
        self.transform_steps: List[PipelineStep] = []
        self.validate_steps: List[PipelineStep] = []
        self.load_steps: List[PipelineStep] = []
        
        # Execution tracking
        self.execution_log: List[Dict[str, Any]] = []
        self.warnings: List[str] = []
        self.errors: List[str] = []
        
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
        
        # Ensure output directory exists
        self.output_path.mkdir(parents=True, exist_ok=True)
    
    # =========================================================================
    # Extract Phase
    # =========================================================================
    
    def add_extract_csv(
        self,
        name: str,
        filename: str,
        parse_dates: Optional[List[str]] = None,
        **kwargs
    ) -> 'ETLPipeline':
        """Add CSV extraction step."""
        def extract():
            self.data[name] = self.loader.load_csv(
                filename, parse_dates=parse_dates, **kwargs
            )
            return self.data[name]
        
        self.extract_steps.append(PipelineStep(
            name=f"Extract CSV: {name}",
            function=extract
        ))
        return self
    
    def add_extract_json(
        self,
        name: str,
        filename: str,
        normalize: bool = True,
        record_path: Optional[str] = None,
        **kwargs
    ) -> 'ETLPipeline':
        """Add JSON extraction step."""
        def extract():
            result = self.loader.load_json(
                filename, normalize=normalize, record_path=record_path, **kwargs
            )
            if isinstance(result, pd.DataFrame):
                self.data[name] = result
            else:
                self.metadata[name] = result
            return result
        
        self.extract_steps.append(PipelineStep(
            name=f"Extract JSON: {name}",
            function=extract
        ))
        return self
    
    def add_extract_custom(
        self,
        name: str,
        extract_func: Callable[[], pd.DataFrame]
    ) -> 'ETLPipeline':
        """Add custom extraction step."""
        def extract():
            self.data[name] = extract_func()
            return self.data[name]
        
        self.extract_steps.append(PipelineStep(
            name=f"Extract Custom: {name}",
            function=extract
        ))
        return self
    
    # =========================================================================
    # Validation Phase
    # =========================================================================
    
    def add_validation(
        self,
        data_name: str,
        validator: DataValidator,
        fail_on_error: bool = True
    ) -> 'ETLPipeline':
        """Add validation step for a data source."""
        def validate():
            if data_name not in self.data:
                raise ValueError(f"Data source '{data_name}' not found")
            
            results = validator.validate(self.data[data_name])
            self.validators[data_name] = validator
            
            # Check results
            failed = [r for r in results if not r.passed]
            errors = [r for r in failed if r.severity in [ValidationSeverity.ERROR, ValidationSeverity.CRITICAL]]
            warnings = [r for r in failed if r.severity == ValidationSeverity.WARNING]
            
            for w in warnings:
                self.warnings.append(f"{data_name}: {w.message}")
            
            if errors and fail_on_error:
                for e in errors:
                    self.errors.append(f"{data_name}: {e.message}")
                raise ValueError(f"Validation failed for {data_name}")
            
            return results
        
        self.validate_steps.append(PipelineStep(
            name=f"Validate: {data_name}",
            function=validate,
            required=fail_on_error
        ))
        return self
    
    # =========================================================================
    # Transform Phase
    # =========================================================================
    
    def add_transform(
        self,
        data_name: str,
        transform_func: Callable[[pd.DataFrame], pd.DataFrame],
        description: str = "Transform"
    ) -> 'ETLPipeline':
        """Add transformation step."""
        def transform():
            if data_name not in self.data:
                raise ValueError(f"Data source '{data_name}' not found")
            
            self.data[data_name] = transform_func(self.data[data_name])
            return self.data[data_name]
        
        self.transform_steps.append(PipelineStep(
            name=f"{description}: {data_name}",
            function=transform
        ))
        return self
    
    def add_transformer_chain(
        self,
        data_name: str,
        transformer: DataTransformer,
        description: str = "Transform Chain"
    ) -> 'ETLPipeline':
        """Add a DataTransformer chain."""
        def transform():
            self.data[data_name] = transformer.get_result()
            self.transformers[data_name] = transformer
            return self.data[data_name]
        
        self.transform_steps.append(PipelineStep(
            name=f"{description}: {data_name}",
            function=transform
        ))
        return self
    
    def add_merge(
        self,
        output_name: str,
        left_name: str,
        right_name: str,
        on: Optional[str] = None,
        left_on: Optional[str] = None,
        right_on: Optional[str] = None,
        how: str = 'inner'
    ) -> 'ETLPipeline':
        """Add merge/join step."""
        def merge():
            left_df = self.data[left_name]
            right_df = self.data[right_name]
            
            self.data[output_name] = pd.merge(
                left_df, right_df,
                on=on, left_on=left_on, right_on=right_on, how=how
            )
            return self.data[output_name]
        
        self.transform_steps.append(PipelineStep(
            name=f"Merge: {left_name} + {right_name} -> {output_name}",
            function=merge
        ))
        return self
    
    def add_aggregate(
        self,
        input_name: str,
        output_name: str,
        group_by: List[str],
        agg_dict: Dict[str, Any]
    ) -> 'ETLPipeline':
        """Add aggregation step."""
        def aggregate():
            df = self.data[input_name]
            self.data[output_name] = df.groupby(group_by).agg(agg_dict).reset_index()
            return self.data[output_name]
        
        self.transform_steps.append(PipelineStep(
            name=f"Aggregate: {input_name} -> {output_name}",
            function=aggregate
        ))
        return self
    
    # =========================================================================
    # Load Phase
    # =========================================================================
    
    def add_load_csv(
        self,
        data_name: str,
        filename: str,
        index: bool = False,
        **kwargs
    ) -> 'ETLPipeline':
        """Add CSV output step."""
        def load():
            if data_name not in self.data:
                raise ValueError(f"Data source '{data_name}' not found")
            
            output_file = self.output_path / filename
            self.data[data_name].to_csv(output_file, index=index, **kwargs)
            return str(output_file)
        
        self.load_steps.append(PipelineStep(
            name=f"Load CSV: {data_name} -> {filename}",
            function=load
        ))
        return self
    
    def add_load_json(
        self,
        data_name: str,
        filename: str,
        orient: str = 'records',
        **kwargs
    ) -> 'ETLPipeline':
        """Add JSON output step."""
        def load():
            if data_name not in self.data:
                raise ValueError(f"Data source '{data_name}' not found")
            
            output_file = self.output_path / filename
            self.data[data_name].to_json(output_file, orient=orient, **kwargs)
            return str(output_file)
        
        self.load_steps.append(PipelineStep(
            name=f"Load JSON: {data_name} -> {filename}",
            function=load
        ))
        return self
    
    def add_load_custom(
        self,
        data_name: str,
        load_func: Callable[[pd.DataFrame, Path], str],
        description: str = "Custom Load"
    ) -> 'ETLPipeline':
        """Add custom load step."""
        def load():
            if data_name not in self.data:
                raise ValueError(f"Data source '{data_name}' not found")
            
            return load_func(self.data[data_name], self.output_path)
        
        self.load_steps.append(PipelineStep(
            name=f"{description}: {data_name}",
            function=load
        ))
        return self
    
    # =========================================================================
    # Execution
    # =========================================================================
    
    def run(self) -> PipelineResult:
        """
        Execute the complete ETL pipeline.
        
        Returns:
            PipelineResult with execution details
        """
        start_time = datetime.now()
        self.status = PipelineStatus.RUNNING
        self.warnings = []
        self.errors = []
        self.execution_log = []
        output_files = []
        
        total_steps = (len(self.extract_steps) + len(self.validate_steps) + 
                      len(self.transform_steps) + len(self.load_steps))
        steps_completed = 0
        
        self.logger.info("="*60)
        self.logger.info(f"STARTING PIPELINE: {self.name}")
        self.logger.info("="*60)
        
        try:
            # EXTRACT PHASE
            self.logger.info("\n📥 EXTRACT PHASE")
            self.logger.info("-"*40)
            for step in self.extract_steps:
                self._execute_step(step)
                steps_completed += 1
            
            # VALIDATE PHASE
            if self.validate_steps:
                self.logger.info("\n✅ VALIDATION PHASE")
                self.logger.info("-"*40)
                for step in self.validate_steps:
                    try:
                        self._execute_step(step)
                        steps_completed += 1
                    except ValueError as e:
                        if step.required:
                            raise
                        self.warnings.append(str(e))
                        steps_completed += 1
            
            # TRANSFORM PHASE
            self.logger.info("\n🔄 TRANSFORM PHASE")
            self.logger.info("-"*40)
            for step in self.transform_steps:
                self._execute_step(step)
                steps_completed += 1
            
            # LOAD PHASE
            self.logger.info("\n📤 LOAD PHASE")
            self.logger.info("-"*40)
            for step in self.load_steps:
                result = self._execute_step(step)
                if result:
                    output_files.append(result)
                steps_completed += 1
            
            # Determine final status
            if self.warnings:
                self.status = PipelineStatus.COMPLETED_WITH_WARNINGS
            else:
                self.status = PipelineStatus.COMPLETED
            
        except Exception as e:
            self.status = PipelineStatus.FAILED
            self.errors.append(str(e))
            self.logger.error(f"Pipeline failed: {str(e)}")
        
        end_time = datetime.now()
        duration = (end_time - start_time).total_seconds()
        
        # Calculate stats
        total_rows_in = sum(len(df) for df in self.data.values())
        total_rows_out = sum(len(df) for df in self.data.values())
        
        self.logger.info("\n" + "="*60)
        self.logger.info(f"PIPELINE {self.status.value.upper()}")
        self.logger.info(f"Duration: {duration:.2f} seconds")
        self.logger.info(f"Steps: {steps_completed}/{total_steps}")
        self.logger.info("="*60)
        
        return PipelineResult(
            status=self.status,
            start_time=start_time,
            end_time=end_time,
            duration_seconds=duration,
            steps_completed=steps_completed,
            steps_total=total_steps,
            data_sources_loaded=len(self.data),
            rows_processed=total_rows_in,
            rows_output=total_rows_out,
            validation_passed=len(self.errors) == 0,
            warnings=self.warnings,
            errors=self.errors,
            output_files=output_files
        )
    
    def _execute_step(self, step: PipelineStep) -> Any:
        """Execute a single pipeline step."""
        step_start = datetime.now()
        self.logger.info(f"  → {step.name}")
        
        try:
            result = step.function(*step.args, **step.kwargs)
            
            step_end = datetime.now()
            duration = (step_end - step_start).total_seconds()
            
            self.execution_log.append({
                'step': step.name,
                'status': 'success',
                'duration': duration,
                'timestamp': step_end.isoformat()
            })
            
            return result
            
        except Exception as e:
            self.execution_log.append({
                'step': step.name,
                'status': 'failed',
                'error': str(e),
                'timestamp': datetime.now().isoformat()
            })
            raise
    
    # =========================================================================
    # Reporting
    # =========================================================================
    
    def get_execution_log(self) -> pd.DataFrame:
        """Get detailed execution log."""
        return pd.DataFrame(self.execution_log)
    
    def get_data_summary(self) -> pd.DataFrame:
        """Get summary of all loaded/processed data."""
        records = []
        for name, df in self.data.items():
            records.append({
                'name': name,
                'rows': len(df),
                'columns': len(df.columns),
                'memory_mb': df.memory_usage(deep=True).sum() / 1024 / 1024,
                'null_count': df.isnull().sum().sum()
            })
        return pd.DataFrame(records)
    
    def get_data(self, name: str) -> pd.DataFrame:
        """Get a specific data source."""
        if name not in self.data:
            raise ValueError(f"Data source '{name}' not found")
        return self.data[name].copy()
    
    def generate_report(self) -> Dict[str, Any]:
        """Generate comprehensive pipeline report."""
        return {
            'pipeline_name': self.name,
            'status': self.status.value,
            'data_sources': list(self.data.keys()),
            'execution_log': self.execution_log,
            'warnings': self.warnings,
            'errors': self.errors,
            'data_summary': self.get_data_summary().to_dict('records'),
            'generated_at': datetime.now().isoformat()
        }
    
    def save_report(self, filename: str = "pipeline_report.json") -> str:
        """Save pipeline report to file."""
        report = self.generate_report()
        output_file = self.output_path / filename
        
        with open(output_file, 'w') as f:
            json.dump(report, f, indent=2, default=str)
        
        return str(output_file)
    
    def __repr__(self) -> str:
        return (f"ETLPipeline(name='{self.name}', "
                f"extract_steps={len(self.extract_steps)}, "
                f"transform_steps={len(self.transform_steps)}, "
                f"load_steps={len(self.load_steps)}, "
                f"status={self.status.value})")


if __name__ == '__main__':
    # Example usage will be in the notebook
    print("ETL Pipeline module loaded successfully")
    print("See notebook for complete usage examples")
