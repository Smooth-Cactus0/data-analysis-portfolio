"""
Helper Utilities Module
=======================
Common utility functions used across the pipeline.

Author: Alexy Louis
"""

import hashlib
import re
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional, Union
import yaml
import json


def load_config(config_path: Union[str, Path]) -> Dict[str, Any]:
    """
    Load configuration from YAML file.
    
    Args:
        config_path: Path to configuration file
        
    Returns:
        Configuration dictionary
    """
    config_path = Path(config_path)
    if not config_path.exists():
        raise FileNotFoundError(f"Configuration file not found: {config_path}")
    
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def save_json(data: Any, filepath: Union[str, Path], indent: int = 2):
    """Save data to JSON file."""
    filepath = Path(filepath)
    filepath.parent.mkdir(parents=True, exist_ok=True)
    with open(filepath, 'w') as f:
        json.dump(data, f, indent=indent, default=str)


def load_json(filepath: Union[str, Path]) -> Any:
    """Load data from JSON file."""
    with open(filepath, 'r') as f:
        return json.load(f)


def generate_hash(data: str) -> str:
    """Generate SHA256 hash of string data."""
    return hashlib.sha256(data.encode()).hexdigest()


def generate_batch_id() -> str:
    """Generate unique batch ID for pipeline runs."""
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    unique = generate_hash(str(datetime.now().timestamp()))[:8]
    return f"BATCH_{timestamp}_{unique}"


def parse_date(date_string: str, formats: Optional[List[str]] = None) -> Optional[datetime]:
    """
    Parse date string with multiple format support.
    
    Args:
        date_string: Date string to parse
        formats: List of date formats to try
        
    Returns:
        Parsed datetime or None if parsing fails
    """
    if not date_string or str(date_string).lower() in ['nan', 'nat', 'none', '']:
        return None
        
    formats = formats or [
        '%Y-%m-%d',
        '%Y-%m-%d %H:%M:%S',
        '%m/%d/%Y',
        '%d/%m/%Y',
        '%Y/%m/%d',
        '%d-%m-%Y',
        '%m-%d-%Y',
        '%Y%m%d',
        '%d %b %Y',
        '%d %B %Y',
        '%b %d, %Y',
        '%B %d, %Y',
    ]
    
    for fmt in formats:
        try:
            return datetime.strptime(str(date_string).strip(), fmt)
        except ValueError:
            continue
    return None


def clean_string(text: str, lowercase: bool = False, strip: bool = True) -> str:
    """
    Clean and normalize string data.
    
    Args:
        text: Input string
        lowercase: Convert to lowercase
        strip: Strip whitespace
        
    Returns:
        Cleaned string
    """
    if not isinstance(text, str):
        text = str(text) if text is not None else ''
    
    if strip:
        text = text.strip()
        text = re.sub(r'\s+', ' ', text)  # Normalize whitespace
    
    if lowercase:
        text = text.lower()
    
    return text


def validate_email(email: str) -> bool:
    """Validate email format."""
    pattern = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
    return bool(re.match(pattern, str(email)))


def validate_phone(phone: str) -> bool:
    """Validate phone number format (US)."""
    cleaned = re.sub(r'[^\d]', '', str(phone))
    return len(cleaned) in [10, 11]


def format_currency(value: float, currency: str = 'USD', decimals: int = 2) -> str:
    """Format number as currency string."""
    symbols = {'USD': '$', 'EUR': '€', 'GBP': '£', 'JPY': '¥'}
    symbol = symbols.get(currency, currency)
    return f"{symbol}{value:,.{decimals}f}"


def format_percentage(value: float, decimals: int = 1) -> str:
    """Format number as percentage string."""
    return f"{value:.{decimals}f}%"


def calculate_date_range(days_back: int = 30) -> tuple:
    """Calculate date range from today."""
    end_date = datetime.now()
    start_date = end_date - timedelta(days=days_back)
    return start_date, end_date


def chunk_list(lst: List, chunk_size: int) -> List[List]:
    """Split list into chunks of specified size."""
    return [lst[i:i + chunk_size] for i in range(0, len(lst), chunk_size)]


def flatten_dict(d: Dict, parent_key: str = '', sep: str = '_') -> Dict:
    """Flatten nested dictionary."""
    items = []
    for k, v in d.items():
        new_key = f"{parent_key}{sep}{k}" if parent_key else k
        if isinstance(v, dict):
            items.extend(flatten_dict(v, new_key, sep).items())
        else:
            items.append((new_key, v))
    return dict(items)


def safe_divide(numerator: float, denominator: float, default: float = 0.0) -> float:
    """Safely divide numbers, returning default if division by zero."""
    try:
        if denominator == 0:
            return default
        return numerator / denominator
    except (TypeError, ZeroDivisionError):
        return default


def detect_encoding(file_path: Union[str, Path]) -> str:
    """Detect file encoding (simplified version)."""
    encodings = ['utf-8', 'latin-1', 'cp1252', 'iso-8859-1']
    
    for encoding in encodings:
        try:
            with open(file_path, 'r', encoding=encoding) as f:
                f.read(1024)
            return encoding
        except (UnicodeDecodeError, UnicodeError):
            continue
    
    return 'utf-8'  # Default fallback


def ensure_directory(path: Union[str, Path]) -> Path:
    """Ensure directory exists, create if not."""
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def get_file_size(file_path: Union[str, Path]) -> str:
    """Get human-readable file size."""
    size = Path(file_path).stat().st_size
    for unit in ['B', 'KB', 'MB', 'GB']:
        if size < 1024:
            return f"{size:.1f} {unit}"
        size /= 1024
    return f"{size:.1f} TB"


class RetryHandler:
    """Handle retries with exponential backoff."""
    
    def __init__(self, max_retries: int = 3, base_delay: float = 1.0, max_delay: float = 60.0):
        self.max_retries = max_retries
        self.base_delay = base_delay
        self.max_delay = max_delay
    
    def get_delay(self, attempt: int) -> float:
        """Calculate delay for current attempt using exponential backoff."""
        delay = self.base_delay * (2 ** attempt)
        return min(delay, self.max_delay)
    
    def should_retry(self, attempt: int) -> bool:
        """Check if should retry based on attempt number."""
        return attempt < self.max_retries
