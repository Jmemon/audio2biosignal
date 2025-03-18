import torch
import torchaudio
import torchaudio.functional as F
import torchaudio.transforms as T
import numpy as np
from pydantic import BaseModel
from typing import Tuple

def preprocess_audio(audio_signal: torch.Tensor, sample_rate: int, feature_config: AudioEDAFeatureConfig) -> torch.Tensor:
    """
    Preprocess audio signal by converting sample rate and extracting MFCCs.
    
    Args:
        audio_signal: Raw audio signal with shape (channels, time_steps)
        sample_rate: Original sample rate of the audio signal in Hz
        feature_config: Configuration for feature extraction
        
    Returns:
        torch.Tensor: MFCCs with shape (num_channels, num_mfccs, time_steps)
    """
    # Handle multi-channel or single-channel input
    if audio_signal.dim() == 1:
        # Convert single channel to (1, time_steps) shape
        audio_signal = audio_signal.unsqueeze(0)
    
    num_channels = audio_signal.shape[0]
    target_sr = feature_config.mutual_sample_rate
    
    # Apply normalization if configured
    if feature_config.audio_normalize:
        audio_signal = normalize_audio(audio_signal)
    
    # Resample audio to target sample rate if needed
    if sample_rate != target_sr:
        # Convert to target sample rate
        audio_signal = resample_audio(audio_signal, sample_rate, target_sr)
    
    # Compute MFCCs for each channel
    mfcc_features = []
    for channel_idx in range(num_channels):
        channel_data = audio_signal[channel_idx]
        
        # Extract MFCCs
        mfccs = extract_mfccs(
            channel_data, 
            target_sr,
            n_mfcc=feature_config.audio_n_mfcc,
            n_mels=feature_config.audio_n_mels,
            window_size=feature_config.audio_window_size,
            hop_length=feature_config.audio_hop_length
        )
        
        mfcc_features.append(mfccs)
    
    # Stack all channel features
    mfcc_tensor = torch.stack(mfcc_features)
    
    return mfcc_tensor

def normalize_audio(audio: torch.Tensor) -> torch.Tensor:
    """
    Normalize audio to have zero mean and unit variance per channel.
    
    Args:
        audio: Audio tensor with shape (channels, time_steps)
    
    Returns:
        torch.Tensor: Normalized audio
    """
    # Normalize each channel separately
    mean = torch.mean(audio, dim=1, keepdim=True)
    std = torch.std(audio, dim=1, keepdim=True)
    
    # Add small epsilon to avoid division by zero
    eps = 1e-8
    return (audio - mean) / (std + eps)

def resample_audio(audio: torch.Tensor, orig_sr: int, target_sr: int) -> torch.Tensor:
    """
    Resample audio from original sample rate to target sample rate.
    
    Args:
        audio: Audio tensor with shape (channels, time_steps)
        orig_sr: Original sample rate in Hz
        target_sr: Target sample rate in Hz
    
    Returns:
        torch.Tensor: Resampled audio with shape (channels, new_time_steps)
    """
    # Use torchaudio's resample function for each channel
    resampler = T.Resample(orig_freq=orig_sr, new_freq=target_sr)
    
    # Apply to each channel
    resampled_channels = []
    for channel_idx in range(audio.shape[0]):
        resampled = resampler(audio[channel_idx])
        resampled_channels.append(resampled)
    
    # Stack channels back together
    return torch.stack(resampled_channels)

def extract_mfccs(audio: torch.Tensor, sample_rate: int, n_mfcc: int = 40, 
                 n_mels: int = 128, window_size: int = 400, hop_length: int = 160) -> torch.Tensor:
    """
    Extract MFCCs from audio signal.
    
    Args:
        audio: Single channel audio tensor with shape (time_steps,)
        sample_rate: Sample rate of the audio in Hz
        n_mfcc: Number of MFCC coefficients to extract
        n_mels: Number of Mel filter banks
        window_size: STFT window size
        hop_length: STFT hop length
    
    Returns:
        torch.Tensor: MFCCs with shape (n_mfcc, time_steps)
    """
    # Create MFCC transform
    mfcc_transform = T.MFCC(
        sample_rate=sample_rate,
        n_mfcc=n_mfcc,
        melkwargs={
            'n_fft': window_size,
            'hop_length': hop_length,
            'n_mels': n_mels,
            'center': True,
            'pad_mode': 'reflect',
            'normalized': True
        }
    )
    
    # Extract MFCCs
    mfccs = mfcc_transform(audio)
    return mfccs

def preprocess_eda(eda_signal: torch.Tensor, sample_rate: int, feature_config: AudioEDAFeatureConfig) -> torch.Tensor:
    """
    Preprocess EDA signal by resampling and applying filters.
    
    Args:
        eda_signal: EDA signal with shape (channels, time_steps)
        sample_rate: Original sample rate of the EDA signal in Hz
        feature_config: Configuration for feature extraction
        
    Returns:
        torch.Tensor: Processed EDA with shape (channels, time_steps)
    """
    # Handle single channel input
    if eda_signal.dim() == 1:
        eda_signal = eda_signal.unsqueeze(0)
    
    target_sr = feature_config.mutual_sample_rate
    
    # Apply normalization if configured
    if feature_config.eda_normalize:
        eda_signal = normalize_audio(eda_signal)
    
    # Resample EDA to target sample rate if needed
    if sample_rate != target_sr:
        eda_signal = resample_eda(eda_signal, sample_rate, target_sr)
    
    # Apply filters if configured
    if feature_config.filter_lowpass or feature_config.filter_highpass:
        eda_signal = apply_filters(
            eda_signal, 
            target_sr, 
            lowpass=feature_config.filter_lowpass,
            highpass=feature_config.filter_highpass
        )
    
    return eda_signal

def resample_eda(eda: torch.Tensor, orig_sr: int, target_sr: int) -> torch.Tensor:
    """
    Resample EDA signal from original sample rate to target sample rate.
    
    Args:
        eda: EDA tensor with shape (channels, time_steps)
        orig_sr: Original sample rate in Hz
        target_sr: Target sample rate in Hz
    
    Returns:
        torch.Tensor: Resampled EDA with shape (channels, new_time_steps)
    """
    # Use torchaudio's resample function for each channel
    resampler = T.Resample(orig_freq=orig_sr, new_freq=target_sr)
    
    # Apply to each channel
    resampled_channels = []
    for channel_idx in range(eda.shape[0]):
        resampled = resampler(eda[channel_idx])
        resampled_channels.append(resampled)
    
    # Stack channels back together
    return torch.stack(resampled_channels)

def apply_filters(eda: torch.Tensor, sample_rate: int, lowpass: bool = True, highpass: bool = False) -> torch.Tensor:
    """
    Apply lowpass and/or highpass filters to EDA signal.
    
    Args:
        eda: EDA tensor with shape (channels, time_steps)
        sample_rate: Sample rate in Hz
        lowpass: Whether to apply 8Hz lowpass filter
        highpass: Whether to apply 0.05Hz highpass filter
    
    Returns:
        torch.Tensor: Filtered EDA with shape (channels, time_steps)
    """
    # Apply filters for each channel separately
    filtered_channels = []
    
    for channel_idx in range(eda.shape[0]):
        channel_data = eda[channel_idx]
        
        # Convert to frequency domain
        fft_data = torch.fft.rfft(channel_data)
        freqs = torch.fft.rfftfreq(len(channel_data), d=1.0/sample_rate)
        
        # Create filter mask
        mask = torch.ones_like(freqs, dtype=torch.float32)
        
        # Apply lowpass filter (8Hz)
        if lowpass:
            mask = mask * (freqs <= 8.0)
        
        # Apply highpass filter (0.05Hz)
        if highpass:
            mask = mask * (freqs >= 0.05)
        
        # Apply filter
        filtered_fft = fft_data * mask
        
        # Convert back to time domain
        filtered_data = torch.fft.irfft(filtered_fft, n=len(channel_data))
        filtered_channels.append(filtered_data)
    
    # Stack channels back together
    return torch.stack(filtered_channels)

def joint_preprocess(audio_signal: torch.Tensor, audio_sr: int, 
                     eda_signal: torch.Tensor, eda_sr: int, 
                     feature_config: AudioEDAFeatureConfig) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Joint preprocessing of audio and EDA signals.
    
    Args:
        audio_signal: Audio signal with shape (channels, time_steps)
        audio_sr: Audio sample rate in Hz
        eda_signal: EDA signal with shape (channels, time_steps)
        eda_sr: EDA sample rate in Hz
        feature_config: Feature configuration
        
    Returns:
        Tuple[torch.Tensor, torch.Tensor]: (audio_mfccs, processed_eda)
    """
    # Process audio to extract MFCCs
    audio_mfccs = preprocess_audio(audio_signal, audio_sr, feature_config)
    
    # Process EDA
    processed_eda = preprocess_eda(eda_signal, eda_sr, feature_config)
    
    return audio_mfccs, processed_eda