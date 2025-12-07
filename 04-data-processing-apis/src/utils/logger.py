"""
Logging Utility Module
======================
Provides centralized logging configuration for the data pipeline.

Features:
- Colored console output
- File logging with rotation
- Structured logging support
- Performance metrics tracking

Author: Alexy Louis
"""

import logging
import sys
from datetime import datetime
from pathlib import Path
from typing import Optional
import json


class ColoredFormatter(logging.Formatter):
    """Custom formatter with colored output for console logging."""
    
    COLORS = {
        'DEBUG': '\033[36m',      # Cyan
        'INFO': '\033[32m',       # Green
        'WARNING': '\033[33m',    # Yellow
        'ERROR': '\033[31m',      # Red
        'CRITICAL': '\033[35m',   # Magenta
        'RESET': '\033[0m'        # Reset
    }
    
    def format(self, record: logging.LogRecord) -> str:
        color = self.COLORS.get(record.levelname, self.COLORS['RESET'])
        reset = self.COLORS['RESET']
        timestamp = datetime.fromtimestamp(record.created).strftime('%Y-%m-%d %H:%M:%S')
        formatted = f"{color}[{timestamp}] [{record.levelname:8}]{reset} {record.getMessage()}"
        if record.exc_info:
            formatted += f"\n{self.formatException(record.exc_info)}"
        return formatted


class JSONFormatter(logging.Formatter):
    """JSON formatter for structured logging."""
    
    def format(self, record: logging.LogRecord) -> str:
        log_data = {
            'timestamp': datetime.fromtimestamp(record.created).isoformat(),
            'level': record.levelname,
            'logger': record.name,
            'message': record.getMessage(),
            'module': record.module,
            'function': record.funcName,
            'line': record.lineno
        }
        if hasattr(record, 'extra_data'):
            log_data['extra'] = record.extra_data
        if record.exc_info:
            log_data['exception'] = self.formatException(record.exc_info)
        return json.dumps(log_data)


class PipelineLogger:
    """
    Centralized logger for the data pipeline.
    
    Provides consistent logging across all pipeline components with
    support for console output, file logging, and metrics tracking.
    """
    
    _instances = {}
    _log_dir = None
    
    def __new__(cls, name: str = 'pipeline', log_dir: Optional[str] = None):
        if name not in cls._instances:
            instance = super().__new__(cls)
            cls._instances[name] = instance
        return cls._instances[name]
    
    def __init__(self, name: str = 'pipeline', log_dir: Optional[str] = None):
        if hasattr(self, '_logger'):
            return
        self.name = name
        self._logger = logging.getLogger(name)
        self._logger.setLevel(logging.DEBUG)
        self._metrics = {}
        
        if log_dir:
            PipelineLogger._log_dir = Path(log_dir)
        elif PipelineLogger._log_dir is None:
            PipelineLogger._log_dir = Path('logs')
        self._log_dir = PipelineLogger._log_dir
        self._log_dir.mkdir(parents=True, exist_ok=True)
        
        if not self._logger.handlers:
            self._setup_handlers()
    
    def _setup_handlers(self):
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(logging.INFO)
        console_handler.setFormatter(ColoredFormatter())
        self._logger.addHandler(console_handler)
        
        log_file = self._log_dir / f"{self.name}_{datetime.now().strftime('%Y%m%d')}.log"
        file_handler = logging.FileHandler(log_file, mode='a')
        file_handler.setLevel(logging.DEBUG)
        file_handler.setFormatter(logging.Formatter(
            '%(asctime)s | %(levelname)-8s | %(name)s | %(funcName)s:%(lineno)d | %(message)s'
        ))
        self._logger.addHandler(file_handler)
    
    def _log(self, level: int, message: str, **kwargs):
        extra = {'extra_data': kwargs} if kwargs else {}
        if kwargs:
            extra_str = ' | '.join(f"{k}={v}" for k, v in kwargs.items())
            message = f"{message} | {extra_str}"
        self._logger.log(level, message, extra=extra)
    
    def debug(self, message: str, **kwargs): self._log(logging.DEBUG, message, **kwargs)
    def info(self, message: str, **kwargs): self._log(logging.INFO, message, **kwargs)
    def warning(self, message: str, **kwargs): self._log(logging.WARNING, message, **kwargs)
    def error(self, message: str, **kwargs): self._log(logging.ERROR, message, **kwargs)
    def critical(self, message: str, **kwargs): self._log(logging.CRITICAL, message, **kwargs)
    def exception(self, message: str, **kwargs): self._logger.exception(message)
    
    def start_timer(self, operation: str):
        self._metrics[f"{operation}_start"] = datetime.now()
        self.debug(f"Timer started: {operation}")
    
    def stop_timer(self, operation: str) -> float:
        start_key = f"{operation}_start"
        if start_key not in self._metrics:
            return 0.0
        duration = (datetime.now() - self._metrics[start_key]).total_seconds()
        self._metrics[f"{operation}_duration"] = duration
        self.info(f"Timer stopped: {operation}", duration_seconds=round(duration, 2))
        return duration
    
    def record_metric(self, name: str, value):
        self._metrics[name] = value
    
    def get_metrics(self) -> dict:
        return self._metrics.copy()
    
    def log_separator(self, char: str = '=', length: int = 60):
        self.info(char * length)
    
    def log_section(self, title: str):
        self.log_separator()
        self.info(f"  {title.upper()}")
        self.log_separator()


def get_logger(name: str = 'pipeline', log_dir: Optional[str] = None) -> PipelineLogger:
    return PipelineLogger(name, log_dir)
