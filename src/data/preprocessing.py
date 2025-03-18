import torch
import torchaudio
import torchaudio.functional as F
import torchaudio.transforms as T
import numpy as np
from pydantic import BaseModel
from typing import Tuple
from src.configs import AudioEDAFeatureConfig

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
        # Normalize each channel separately
        mean = torch.mean(audio_signal, dim=1, keepdim=True)
        std = torch.std(audio_signal, dim=1, keepdim=True)
        # Add small epsilon to avoid division by zero
        eps = 1e-8
        audio_signal = (audio_signal - mean) / (std + eps)
    
    # Resample audio to target sample rate if needed
    if sample_rate != target_sr:
        # Use torchaudio's resample function for each channel
        resampler = T.Resample(orig_freq=sample_rate, new_freq=target_sr)
        
        # Apply to each channel
        resampled_channels = []
        for channel_idx in range(audio_signal.shape[0]):
            resampled = resampler(audio_signal[channel_idx])
            resampled_channels.append(resampled)
        
        # Stack channels back together
        audio_signal = torch.stack(resampled_channels)
    
    # Compute MFCCs for each channel
    mfcc_features = []
    for channel_idx in range(num_channels):
        channel_data = audio_signal[channel_idx]
        
        # Create MFCC transform
        mfcc_transform = T.MFCC(
            sample_rate=target_sr,
            n_mfcc=feature_config.audio_n_mfcc,
            melkwargs={
                'n_fft': feature_config.audio_window_size,
                'hop_length': feature_config.audio_hop_length,
                'n_mels': feature_config.audio_n_mels,
                'center': True,
                'pad_mode': 'reflect',
                'normalized': True
            }
        )
        
        # Extract MFCCs
        mfccs = mfcc_transform(channel_data)
        mfcc_features.append(mfccs)
    
    # Stack all channel features
    mfcc_tensor = torch.stack(mfcc_features)
    
    return mfcc_tensor

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
        # Normalize each channel separately
        mean = torch.mean(eda_signal, dim=1, keepdim=True)
        std = torch.std(eda_signal, dim=1, keepdim=True)
        # Add small epsilon to avoid division by zero
        eps = 1e-8
        eda_signal = (eda_signal - mean) / (std + eps)
    
    # Resample EDA to target sample rate if needed
    if sample_rate != target_sr:
        # Use torchaudio's resample function for each channel
        resampler = T.Resample(orig_freq=sample_rate, new_freq=target_sr)
        
        # Apply to each channel
        resampled_channels = []
        for channel_idx in range(eda_signal.shape[0]):
            resampled = resampler(eda_signal[channel_idx])
            resampled_channels.append(resampled)
        
        # Stack channels back together
        eda_signal = torch.stack(resampled_channels)
    
    # Apply filters if configured
    if feature_config.filter_lowpass or feature_config.filter_highpass:
        # Apply filters for each channel separately
        filtered_channels = []
        
        for channel_idx in range(eda_signal.shape[0]):
            channel_data = eda_signal[channel_idx]
            
            # Convert to frequency domain
            fft_data = torch.fft.rfft(channel_data)
            freqs = torch.fft.rfftfreq(len(channel_data), d=1.0/target_sr)
            
            # Create filter mask
            mask = torch.ones_like(freqs, dtype=torch.float32)
            
            # Apply lowpass filter (8Hz)
            if feature_config.filter_lowpass:
                mask = mask * (freqs <= 8.0)
            
            # Apply highpass filter (0.05Hz)
            if feature_config.filter_highpass:
                mask = mask * (freqs >= 0.05)
            
            # Apply filter
            filtered_fft = fft_data * mask
            
            # Convert back to time domain
            filtered_data = torch.fft.irfft(filtered_fft, n=len(channel_data))
            filtered_channels.append(filtered_data)
        
        # Stack channels back together
        eda_signal = torch.stack(filtered_channels)
    
    return eda_signal

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
