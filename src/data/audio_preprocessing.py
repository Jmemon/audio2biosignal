import torch
import torchaudio
from src.configs import AudioFeatureConfig

def preprocess_audio(audio_file_path: str, audio_feature_config: AudioFeatureConfig) -> torch.Tensor:
    waveform, sample_rate = torchaudio.load(audio_file_path)
    if sample_rate != audio_feature_config.audio_sample_rate:
        resample_transform = torchaudio.transforms.Resample(
            orig_freq=sample_rate,
            new_freq=audio_feature_config.audio_sample_rate
        )
        waveform = resample_transform(waveform)
    if audio_feature_config.audio_normalize:
        waveform = waveform / waveform.abs().max()
    mfcc_transform = torchaudio.transforms.MFCC(
        sample_rate=audio_feature_config.audio_sample_rate,
        n_mfcc=audio_feature_config.audio_n_mfcc,
        melkwargs={
            'n_mels': audio_feature_config.audio_n_mels,
            'win_length': audio_feature_config.audio_window_size,
            'hop_length': audio_feature_config.audio_hop_length
        }
    )
    mfcc = mfcc_transform(waveform)
    return mfcc  # Shape: (1, n_mfcc, time_steps)
