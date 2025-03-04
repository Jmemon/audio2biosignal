import torch
import torchaudio
from pathlib import Path
from src.configs import AudioEDAFeatureConfig

def preprocess_audio(audio_file_path: Path, feature_config: AudioEDAFeatureConfig) -> torch.Tensor:
    waveform, sample_rate = torchaudio.load(audio_file_path)
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
