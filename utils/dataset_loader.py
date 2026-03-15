import wfdb
import numpy as np
import pandas as pd
import torch

def load_raw_data(df, sampling_rate, path):
    if sampling_rate == 100:
        data = [wfdb.rdsamp(path + f) for f in df.filename_lr]
    else:
        data = [wfdb.rdsamp(path + f) for f in df.filename_hr]
    data = np.array([signal for signal, meta in data])
    return data

def preprocess_signal(signal):
    # Ensure signal is (Channels, Length) -> (12, 1000)
    if signal.shape[1] == 12:
        signal = signal.T
    return torch.tensor(signal, dtype=torch.float32).unsqueeze(0)
