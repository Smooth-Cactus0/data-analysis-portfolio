"""
Multi-Source ETL Pipeline
=========================

A comprehensive data processing pipeline for handling multiple data sources,
performing validation, transformation, and loading operations.

Author: Alexy Louis
Email: alexy.louis.scholar@gmail.com
"""

from .data_loader import DataLoader, quick_load
from .data_validator import DataValidator, ValidationResult, ValidationSeverity
from .data_transformer import DataTransformer, TransformationLog
from .pipeline_orchestrator import ETLPipeline, PipelineResult, PipelineStatus

__all__ = [
    'DataLoader', 'quick_load',
    'DataValidator', 'ValidationResult', 'ValidationSeverity',
    'DataTransformer', 'TransformationLog',
    'ETLPipeline', 'PipelineResult', 'PipelineStatus'
]

__version__ = '1.0.0'
__author__ = 'Alexy Louis'
