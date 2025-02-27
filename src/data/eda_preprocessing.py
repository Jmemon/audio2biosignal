import torch
import pandas as pd
from scipy.signal import butter, filtfilt
from src.configs import EDAFeatureConfig, AudioFeatureConfig

def preprocess_eda(eda_file_path: str, eda_feature_config: EDAFeatureConfig,
                   audio_feature_config: AudioFeatureConfig) -> torch.Tensor:
    eda_data = pd.read_csv(eda_file_path).values.squeeze()
    # Resampling logic here (if needed)
    if eda_feature_config.eda_normalize:
        eda_data = (eda_data - eda_data.mean()) / eda_data.std()
    if eda_feature_config.filter_highpass:
        b, a = butter(2, 0.05, btype='highpass', fs=audio_feature_config.audio_sample_rate)
        eda_data = filtfilt(b, a, eda_data)
    if eda_feature_config.filter_lowpass:
        b, a = butter(2, 8, btype='lowpass', fs=audio_feature_config.audio_sample_rate)
        eda_data = filtfilt(b, a, eda_data)
    eda_tensor = torch.tensor(eda_data).unsqueeze(0)  # Shape: (1, time_steps)
    return eda_tensor
