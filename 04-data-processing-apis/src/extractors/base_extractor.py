"""
Base Extractor Module
=====================
Abstract base class for all data extractors in the pipeline.

Author: Alexy Louis
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional, Union
from pathlib import Path
import pandas as pd

import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from src.utils.logger import get_logger


@dataclass
class ExtractionResult:
    """
    Container for extraction results with metadata.
    
    Attributes:
        data: Extracted data (DataFrame or dict)
        source: Source identifier
        records_extracted: Number of records extracted
        extraction_time: Time taken for extraction
        timestamp: When extraction occurred
        success: Whether extraction was successful
        errors: List of any errors encountered
        metadata: Additional extraction metadata
    """
    data: Union[pd.DataFrame, Dict, List]
    source: str
    records_extracted: int = 0
    extraction_time: float = 0.0
    timestamp: datetime = field(default_factory=datetime.now)
    success: bool = True
    errors: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert result to dictionary for serialization."""
        return {
            'source': self.source,
            'records_extracted': self.records_extracted,
            'extraction_time': self.extraction_time,
            'timestamp': self.timestamp.isoformat(),
            'success': self.success,
            'errors': self.errors,
            'metadata': self.metadata
        }


class BaseExtractor(ABC):
    """
    Abstract base class for all data extractors.
    
    Provides common functionality for extracting data from various sources
    including logging, error handling, and result standardization.
    
    Subclasses must implement:
        - extract(): Main extraction logic
        - validate_source(): Validate source configuration
    """
    
    def __init__(self, name: str, config: Optional[Dict] = None):
        """
        Initialize extractor.
        
        Args:
            name: Extractor name for logging
            config: Configuration dictionary
        """
        self.name = name
        self.config = config or {}
        self.logger = get_logger(f'extractor.{name}')
        self._extraction_count = 0
        self._error_count = 0
    
    @abstractmethod
    def extract(self, **kwargs) -> ExtractionResult:
        """
        Extract data from source.
        
        Must be implemented by subclasses.
        
        Returns:
            ExtractionResult containing extracted data and metadata
        """
        pass
    
    @abstractmethod
    def validate_source(self) -> bool:
        """
        Validate that the source is accessible and properly configured.
        
        Must be implemented by subclasses.
        
        Returns:
            True if source is valid, False otherwise
        """
        pass
    
    def _create_result(
        self,
        data: Union[pd.DataFrame, Dict, List],
        source: str,
        extraction_time: float,
        errors: Optional[List[str]] = None,
        metadata: Optional[Dict] = None
    ) -> ExtractionResult:
        """
        Create standardized extraction result.
        
        Args:
            data: Extracted data
            source: Source identifier
            extraction_time: Time taken for extraction
            errors: Any errors encountered
            metadata: Additional metadata
            
        Returns:
            ExtractionResult instance
        """
        records = 0
        if isinstance(data, pd.DataFrame):
            records = len(data)
        elif isinstance(data, (list, dict)):
            records = len(data)
        
        errors = errors or []
        success = len(errors) == 0
        
        self._extraction_count += 1
        if not success:
            self._error_count += 1
        
        return ExtractionResult(
            data=data,
            source=source,
            records_extracted=records,
            extraction_time=extraction_time,
            success=success,
            errors=errors,
            metadata=metadata or {}
        )
    
    def get_stats(self) -> Dict[str, Any]:
        """Get extractor statistics."""
        return {
            'name': self.name,
            'extractions': self._extraction_count,
            'errors': self._error_count,
            'success_rate': (
                (self._extraction_count - self._error_count) / self._extraction_count * 100
                if self._extraction_count > 0 else 0
            )
        }
    
    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(name='{self.name}')"
