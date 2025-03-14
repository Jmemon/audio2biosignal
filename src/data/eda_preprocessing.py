import torch
import numpy as np
import pandas as pd
from scipy.signal import butter, filtfilt, resample
from pathlib import Path
from src.configs import AudioEDAFeatureConfig

def preprocess_eda(eda_signal: torch.Tensor, feature_config: AudioEDAFeatureConfig) -> torch.Tensor:
    eda_data = eda_signal.numpy() if isinstance(eda_signal, torch.Tensor) else eda_signal
    original_sample_rate = 1000  # Assuming original EDA sample rate
    if original_sample_rate != feature_config.mutual_sample_rate:
        resample_factor = feature_config.mutual_sample_rate / original_sample_rate
        eda_data = resample(eda_data, int(len(eda_data) * resample_factor))
    if feature_config.eda_normalize:
        eda_data = (eda_data - eda_data.mean()) / eda_data.std()
    if feature_config.filter_highpass:
        b, a = butter(2, 0.05, btype='highpass', fs=feature_config.mutual_sample_rate)
        eda_data = filtfilt(b, a, eda_data)
    if feature_config.filter_lowpass:
        b, a = butter(2, 8, btype='lowpass', fs=feature_config.mutual_sample_rate)
        eda_data = filtfilt(b, a, eda_data)
    if not isinstance(eda_data, torch.Tensor):
        # Ensure array is contiguous in memory before tensor conversion
        eda_data = np.ascontiguousarray(eda_data)
        eda_tensor = torch.tensor(eda_data, dtype=torch.float32).unsqueeze(0)  # Shape: (1, time_steps)
    else:
        eda_tensor = eda_data.unsqueeze(0) if eda_data.dim() == 1 else eda_data
    return eda_tensor
