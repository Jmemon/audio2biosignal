import torch
import torchaudio
from pathlib import Path
from src.configs import AudioEDAFeatureConfig

def preprocess_audio(waveform: torch.Tensor, sample_rate: int, feature_config: AudioEDAFeatureConfig) -> torch.Tensor:
    """
    Transforms raw audio waveforms into MFCC features for neural network processing.
    
    Performs a standardized preprocessing pipeline including optional amplitude normalization,
    MFCC feature extraction with configurable parameters, and resampling of the MFCC features
    along the time dimension. Time complexity is O(n log n) due to FFT operations in MFCC computation.
    
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
        - mutual_sample_rate: Target sample rate for resampling MFCC features
        - audio_normalize: Whether to normalize amplitude to [-1, 1]
        - audio_n_mfcc: Number of MFCC coefficients to extract
        - audio_n_mels: Number of mel filterbanks
        - audio_window_size: Window size for STFT in samples
        - audio_hop_length: Hop length for STFT in samples
    
    Returns
    -------
    torch.Tensor
        Resampled MFCC features with shape (channels, n_mfcc, time_steps).
        Time steps are adjusted according to the mutual_sample_rate.
    
    Notes
    -----
    - Preserves multichannel audio structure if present in input
    - Normalization is performed per-tensor, not per-channel
    - Zero-length inputs may produce undefined behavior in MFCC transform
    - Resampling is performed on the MFCC features, not the raw audio
    """
    print(f"[preprocess_audio] Input waveform shape: {waveform.shape}, sample_rate: {sample_rate}")
    if feature_config.audio_normalize:
        waveform = waveform / waveform.abs().max()
        print(f"[preprocess_audio] After normalization, waveform shape: {waveform.shape}")
    
    # Compute MFCC at original sample rate
    mfcc_transform = torchaudio.transforms.MFCC(
        sample_rate=sample_rate,  # Use original sample rate
        n_mfcc=feature_config.audio_n_mfcc,
        melkwargs={
            'n_mels': feature_config.audio_n_mels,
            'win_length': feature_config.audio_window_size,
            'hop_length': feature_config.audio_hop_length
        }
    )
    mfcc = mfcc_transform(waveform)  # Shape: (channels, n_mfcc, time_steps)
    print(f"[preprocess_audio] After MFCC transform, shape: {mfcc.shape}")
    
    # Resample MFCC features along the time dimension if needed
    if sample_rate != feature_config.mutual_sample_rate:
        # Calculate the scaling factor for time dimension
        scale_factor = feature_config.mutual_sample_rate / sample_rate
        
        # Get current dimensions
        channels, n_mfcc, time_steps = mfcc.shape
        
        # Calculate new time steps after resampling
        new_time_steps = int(time_steps * scale_factor)
        
        # Use interpolate to resample along the time dimension (dim=2)
        mfcc = torch.nn.functional.interpolate(
            mfcc, 
            size=new_time_steps,
            mode='linear',
            align_corners=False
        )
        print(f"[preprocess_audio] After resampling, mfcc shape: {mfcc.shape}")
    
    print(f"[preprocess_audio] Final output shape: {mfcc.shape}")
    return mfcc  # Shape: (channels, n_mfcc, resampled_time_steps)
