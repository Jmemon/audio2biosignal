import torch
import torchaudio
from pathlib import Path
from src.configs import AudioEDAFeatureConfig

def preprocess_audio(waveform: torch.Tensor, sample_rate: int, feature_config: AudioEDAFeatureConfig) -> torch.Tensor:
    """
    Transforms raw audio waveforms into MFCC features for neural network processing.
    
    Performs a standardized preprocessing pipeline including resampling to a target rate,
    optional amplitude normalization, and MFCC feature extraction with configurable parameters.
    Time complexity is O(n log n) due to FFT operations in MFCC computation.
    
    Parameters
    ----------
    waveform : torch.Tensor
        Raw audio waveform tensor with shape (channels, samples).
        Must be a non-empty tensor with finite values.
    
    sample_rate : int
        Original sampling rate of the waveform in Hz.
        Must be positive.
    
    feature_config : AudioEDAFeatureConfig
        Configuration object containing all preprocessing parameters:
        - mutual_sample_rate: Target sample rate for resampling
        - audio_normalize: Whether to normalize amplitude to [-1, 1]
        - audio_n_mfcc: Number of MFCC coefficients to extract
        - audio_n_mels: Number of mel filterbanks
        - audio_window_size: Window size for STFT in samples
        - audio_hop_length: Hop length for STFT in samples
    
    Returns
    -------
    torch.Tensor
        MFCC features with shape (channels, n_mfcc, time_steps).
        Time steps depend on input length, window size, and hop length.
    
    Notes
    -----
    - Preserves multichannel audio structure if present in input
    - Normalization is performed per-tensor, not per-channel
    - Zero-length inputs may produce undefined behavior in MFCC transform
    """
    if sample_rate != feature_config.mutual_sample_rate:
        resample_transform = torchaudio.transforms.Resample(
            orig_freq=sample_rate,
            new_freq=feature_config.mutual_sample_rate
        )
        waveform = resample_transform(waveform)
    if feature_config.audio_normalize:
        waveform = waveform / waveform.abs().max()
    mfcc_transform = torchaudio.transforms.MFCC(
        sample_rate=feature_config.mutual_sample_rate,
        n_mfcc=feature_config.audio_n_mfcc,
        melkwargs={
            'n_mels': feature_config.audio_n_mels,
            'win_length': feature_config.audio_window_size,
            'hop_length': feature_config.audio_hop_length
        }
    )
    mfcc = mfcc_transform(waveform)
    return mfcc  # Shape: (1, n_mfcc, time_steps)
