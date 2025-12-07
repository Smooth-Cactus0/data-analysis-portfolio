"""
============================================================
PIPELINE LOGGING UTILITY
============================================================
Provides structured logging for ETL pipeline operations.
Logs are written to both console and file.

Author: Alexy Louis
============================================================
"""

import logging
import os
from datetime import datetime
from functools import wraps
import time

# Create logs directory
LOGS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'logs')
os.makedirs(LOGS_DIR, exist_ok=True)

class PipelineLogger:
    """Custom logger for ETL pipeline with structured output"""
    
    def __init__(self, name='RetailPipeline', log_level=logging.INFO):
        self.logger = logging.getLogger(name)
        self.logger.setLevel(log_level)
        self.logger.handlers = []  # Clear existing handlers
        
        # Create formatters
        console_format = logging.Formatter(
            '%(asctime)s | %(levelname)-8s | %(message)s',
            datefmt='%H:%M:%S'
        )
        file_format = logging.Formatter(
            '%(asctime)s | %(levelname)-8s | %(name)s | %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        
        # Console handler
        console_handler = logging.StreamHandler()
        console_handler.setLevel(log_level)
        console_handler.setFormatter(console_format)
        self.logger.addHandler(console_handler)
        
        # File handler
        log_file = os.path.join(LOGS_DIR, f'pipeline_{datetime.now().strftime("%Y%m%d")}.log')
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(log_level)
        file_handler.setFormatter(file_format)
        self.logger.addHandler(file_handler)
        
        self.log_file = log_file
    
    def info(self, message):
        self.logger.info(message)
    
    def warning(self, message):
        self.logger.warning(message)
    
    def error(self, message):
        self.logger.error(message)
    
    def debug(self, message):
        self.logger.debug(message)
    
    def success(self, message):
        """Log success message with checkmark"""
        self.logger.info(f"✓ {message}")
    
    def stage_start(self, stage_name):
        """Log start of a pipeline stage"""
        self.logger.info("=" * 60)
        self.logger.info(f"STAGE: {stage_name}")
        self.logger.info("=" * 60)
    
    def stage_end(self, stage_name, status='completed'):
        """Log end of a pipeline stage"""
        self.logger.info(f"Stage '{stage_name}' {status}")
        self.logger.info("-" * 60)
    
    def metric(self, name, value, unit=''):
        """Log a metric"""
        self.logger.info(f"  📊 {name}: {value} {unit}")
    
    def data_quality(self, check_name, passed, details=''):
        """Log data quality check result"""
        status = "✓ PASS" if passed else "✗ FAIL"
        self.logger.info(f"  {status} | {check_name} | {details}")


def log_execution_time(logger=None):
    """Decorator to log function execution time"""
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            start_time = time.time()
            result = func(*args, **kwargs)
            elapsed = time.time() - start_time
            
            if logger:
                logger.info(f"  ⏱ {func.__name__} completed in {elapsed:.2f}s")
            
            return result
        return wrapper
    return decorator


# Create default logger instance
pipeline_logger = PipelineLogger()

if __name__ == "__main__":
    # Test the logger
    logger = PipelineLogger()
    logger.stage_start("Test Stage")
    logger.info("This is an info message")
    logger.success("Operation completed successfully")
    logger.metric("Records processed", 1000)
    logger.data_quality("Null check", True, "0.5% null values")
    logger.stage_end("Test Stage")
    print(f"\nLog file: {logger.log_file}")
