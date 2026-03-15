# model/__init__.py

from .ecg_model import ECGClassifier
from .inference import get_prediction

__all__ = ["ECGClassifier", "get_prediction"]
