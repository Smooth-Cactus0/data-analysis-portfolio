"""
EEG Signal Analysis Package

Modules:
    preprocessing: EEG data cleaning and epoch extraction
    features: Feature extraction (time, frequency, spatial)
    models: Classical ML and Deep Learning models
    visualization: Plotting utilities for EEG data
"""

from . import preprocessing
from . import features
from . import models
from . import visualization

__version__ = "1.0.0"
__author__ = "Alexy Louis"
