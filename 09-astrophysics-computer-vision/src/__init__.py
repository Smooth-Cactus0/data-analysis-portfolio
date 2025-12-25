"""
Astrophysics Computer Vision Package
====================================

This package provides tools for:
- Galaxy morphology classification from imaging surveys
- Transient/variable object classification from light curves
- Anomaly detection in astronomical data

Modules:
--------
- preprocessing: Image and light curve preprocessing utilities
- models: CNN, LSTM, Transfer Learning, and Autoencoder architectures
- visualization: Astronomy-specific plotting functions
"""

from . import preprocessing
from . import models
from . import visualization

__version__ = "1.0.0"
__author__ = "Alexy Louis"
