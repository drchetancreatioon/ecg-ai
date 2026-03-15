# model/__init__.py

from .ecg_model import ECGClassifier
from .inference import predict_ecg  # Changed from get_prediction to predict_ecg

__all__ = ["ECGClassifier", "predict_ecg"]
