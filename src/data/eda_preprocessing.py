import torch
import numpy as np
import pandas as pd
from scipy.signal import butter, filtfilt, resample
from pathlib import Path
from src.configs import AudioEDAFeatureConfig

def preprocess_eda(eda_signal: torch.Tensor, sample_rate: int, feature_config: AudioEDAFeatureConfig) -> torch.Tensor:
    """
    Preprocess electrodermal activity (EDA) signals for neural network input with configurable signal processing.
    
    Transforms raw EDA signals through a pipeline of resampling, normalization, and filtering operations
    to produce standardized tensor representations suitable for model training. Implements a flexible
    signal processing chain with O(n) time complexity where n is the signal length.
    
    Parameters
    ----------
    eda_signal : torch.Tensor
        Raw EDA signal as a 1D tensor or numpy array.
        Can handle both torch.Tensor and numpy.ndarray inputs transparently.
    
    sample_rate : int
        Original sampling rate of the EDA signal in Hz.
    
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
    - Applies Butterworth filters of order 2 when filtering is enabled
    - Ensures memory contiguity for efficient tensor operations
    - Thread-safe but not optimized for batch processing
    """
    print(f"[preprocess_eda] Input eda_signal shape: {eda_signal.shape if isinstance(eda_signal, torch.Tensor) else eda_signal.shape if hasattr(eda_signal, 'shape') else len(eda_signal)}")
    eda_data = eda_signal.numpy() if isinstance(eda_signal, torch.Tensor) else eda_signal
    print(f"[preprocess_eda] After conversion to numpy, eda_data shape: {eda_data.shape}")
    
    if sample_rate != feature_config.mutual_sample_rate:
        resample_factor = feature_config.mutual_sample_rate / sample_rate
        eda_data = resample(eda_data, int(len(eda_data) * resample_factor))
        print(f"[preprocess_eda] After resampling, eda_data shape: {eda_data.shape}")
    if feature_config.eda_normalize:
        eda_data = (eda_data - eda_data.mean()) / eda_data.std()
        print(f"[preprocess_eda] After normalization, eda_data shape: {eda_data.shape}")
    if feature_config.filter_highpass:
        b, a = butter(2, 0.05, btype='highpass', fs=feature_config.mutual_sample_rate)
        eda_data = filtfilt(b, a, eda_data)
        print(f"[preprocess_eda] After highpass filter, eda_data shape: {eda_data.shape}")
    if feature_config.filter_lowpass:
        b, a = butter(2, 8, btype='lowpass', fs=feature_config.mutual_sample_rate)
        eda_data = filtfilt(b, a, eda_data)
        print(f"[preprocess_eda] After lowpass filter, eda_data shape: {eda_data.shape}")
    if not isinstance(eda_data, torch.Tensor):
        # Ensure array is contiguous in memory before tensor conversion
        eda_data = np.ascontiguousarray(eda_data)
        eda_tensor = torch.tensor(eda_data, dtype=torch.float32).unsqueeze(0)  # Shape: (1, time_steps)
        print(f"[preprocess_eda] After tensor conversion and unsqueeze, eda_tensor shape: {eda_tensor.shape}")
    else:
        eda_tensor = eda_data.unsqueeze(0) if eda_data.dim() == 1 else eda_data
        print(f"[preprocess_eda] After tensor processing, eda_tensor shape: {eda_tensor.shape}")
    
    print(f"[preprocess_eda] Final output shape: {eda_tensor.shape}")
    return eda_tensor
