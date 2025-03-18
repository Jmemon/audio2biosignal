import torch
import numpy as np
import pandas as pd
from scipy.signal import butter, filtfilt, resample
from pathlib import Path
from src.configs import AudioEDAFeatureConfig

def preprocess_eda(eda_signal: torch.Tensor, feature_config: AudioEDAFeatureConfig) -> torch.Tensor:
    """
    Preprocess electrodermal activity (EDA) signals for neural network input with configurable signal processing.
    
    Transforms raw EDA signals through a pipeline of resampling, normalization, and filtering operations
    to produce standardized tensor representations suitable for model training. Implements a flexible
    signal processing chain with O(n) time complexity where n is the signal length.
    
    Parameters
    ----------
    eda_signal : torch.Tensor
        Raw EDA signal as a 1D tensor or numpy array. Expected to be sampled at 1000Hz originally.
        Can handle both torch.Tensor and numpy.ndarray inputs transparently.
    
    feature_config : AudioEDAFeatureConfig
        Configuration object controlling signal processing parameters:
        - mutual_sample_rate: Target sampling rate for resampling (Hz)
        - eda_normalize: Whether to apply z-score normalization
        - filter_highpass: Whether to apply highpass filtering (0.05Hz cutoff)
        - filter_lowpass: Whether to apply lowpass filtering (8Hz cutoff)
    
    Returns
    -------
    torch.Tensor
        Processed EDA signal as a 2D tensor with shape (1, time_steps), where time_steps
        depends on the original signal length and resampling factor. Values are normalized
        if specified in the configuration.
    
    Notes
    -----
    - Assumes original EDA sampling rate of 1000Hz if resampling is needed
    - Applies Butterworth filters of order 2 when filtering is enabled
    - Ensures memory contiguity for efficient tensor operations
    - Thread-safe but not optimized for batch processing
    """
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
