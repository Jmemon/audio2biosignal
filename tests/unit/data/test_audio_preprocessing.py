"""
Test suite for audio preprocessing functionality.

This module provides comprehensive testing for:
1. Basic audio preprocessing functionality
2. Resampling behavior
3. Normalization behavior
4. MFCC transformation
5. Error handling
"""

import os
import pytest
import torch
import torchaudio
import numpy as np
from pathlib import Path
from unittest.mock import patch, MagicMock, mock_open

from src.configs import AudioEDAFeatureConfig
from src.data.audio_preprocessing import preprocess_audio


class TestAudioPreprocessing:
    """Test suite for audio preprocessing functions."""

    @pytest.fixture
    def feature_config(self):
        """Standard feature configuration for testing."""
        return AudioEDAFeatureConfig(
            mutual_sample_rate=16000,
            audio_normalize=True,
            audio_n_mfcc=13,
            audio_n_mels=40,
            audio_window_size=400,
            audio_hop_length=160
        )

    @pytest.fixture
    def mock_audio_file(self, tmp_path):
        """Create a temporary audio file for testing."""
        # Create a simple sine wave
        sample_rate = 22050
        duration = 1  # seconds
        t = torch.linspace(0, duration, int(sample_rate * duration))
        waveform = torch.sin(2 * np.pi * 440 * t).unsqueeze(0)  # 440 Hz sine wave
        
        # Save as wav file
        file_path = tmp_path / "test_audio.wav"
        torchaudio.save(file_path, waveform, sample_rate)
        
        return file_path

    def test_basic_preprocessing(self, feature_config, mock_audio_file):
        """
        GIVEN a valid audio file and feature configuration
        WHEN preprocess_audio is called
        THEN it should return a properly formatted MFCC tensor
        """
        # Process the audio file
        mfcc = preprocess_audio(mock_audio_file, feature_config)
        
        # Verify the output shape and type
        assert isinstance(mfcc, torch.Tensor)
        assert mfcc.dim() == 3
        assert mfcc.size(0) == 1  # Batch dimension
        assert mfcc.size(1) == feature_config.audio_n_mfcc  # Number of MFCC coefficients
        assert mfcc.size(2) > 0  # Time steps

    def test_resampling(self, feature_config, mock_audio_file):
        """
        GIVEN an audio file with a different sample rate than the config
        WHEN preprocess_audio is called
        THEN it should correctly resample the audio
        """
        # Get original sample rate
        waveform, original_sample_rate = torchaudio.load(mock_audio_file)
        
        # Modify config to have a different sample rate
        feature_config.mutual_sample_rate = original_sample_rate // 2
        
        # Mock the resample transform to verify it's called
        with patch('torchaudio.transforms.Resample', autospec=True) as mock_resample:
            # Setup the mock to return a properly transformed waveform
            mock_instance = mock_resample.return_value
            mock_instance.return_value = waveform
            
            # Process the audio
            preprocess_audio(mock_audio_file, feature_config)
            
            # Verify resample was called with correct parameters
            mock_resample.assert_called_once_with(
                orig_freq=original_sample_rate,
                new_freq=feature_config.mutual_sample_rate
            )
            mock_instance.assert_called_once()

    def test_normalization_enabled(self, feature_config, mock_audio_file):
        """
        GIVEN feature config with normalization enabled
        WHEN preprocess_audio is called
        THEN the waveform should be normalized
        """
        feature_config.audio_normalize = True
        
        # Create a mock waveform with known values
        mock_waveform = torch.tensor([[2.0, -4.0, 1.0]])  # Max abs value is 4.0
        expected_normalized = torch.tensor([[0.5, -1.0, 0.25]])  # Divided by 4.0
        
        with patch('torchaudio.load', return_value=(mock_waveform, feature_config.mutual_sample_rate)):
            with patch('torchaudio.transforms.MFCC') as mock_mfcc:
                # Configure mock MFCC transform
                mock_mfcc_instance = mock_mfcc.return_value
                mock_mfcc_instance.return_value = torch.zeros((1, feature_config.audio_n_mfcc, 10))
                
                # Process the audio
                preprocess_audio(mock_audio_file, feature_config)
                
                # Get the waveform that was passed to MFCC transform
                args, kwargs = mock_mfcc_instance.call_args
                passed_waveform = args[0]
                
                # Verify normalization was applied
                assert torch.allclose(passed_waveform, expected_normalized)

    def test_normalization_disabled(self, feature_config, mock_audio_file):
        """
        GIVEN feature config with normalization disabled
        WHEN preprocess_audio is called
        THEN the waveform should not be normalized
        """
        feature_config.audio_normalize = False
        
        # Create a mock waveform with known values
        mock_waveform = torch.tensor([[2.0, -4.0, 1.0]])
        
        with patch('torchaudio.load', return_value=(mock_waveform, feature_config.mutual_sample_rate)):
            with patch('torchaudio.transforms.MFCC') as mock_mfcc:
                # Configure mock MFCC transform
                mock_mfcc_instance = mock_mfcc.return_value
                mock_mfcc_instance.return_value = torch.zeros((1, feature_config.audio_n_mfcc, 10))
                
                # Process the audio
                preprocess_audio(mock_audio_file, feature_config)
                
                # Get the waveform that was passed to MFCC transform
                args, kwargs = mock_mfcc_instance.call_args
                passed_waveform = args[0]
                
                # Verify original waveform was used (not normalized)
                assert torch.allclose(passed_waveform, mock_waveform)

    def test_mfcc_transform_parameters(self, feature_config, mock_audio_file):
        """
        GIVEN a feature configuration with specific MFCC parameters
        WHEN preprocess_audio is called
        THEN the MFCC transform should be created with those parameters
        """
        with patch('torchaudio.transforms.MFCC', autospec=True) as mock_mfcc:
            # Configure the mock
            mock_instance = mock_mfcc.return_value
            mock_instance.return_value = torch.zeros((1, feature_config.audio_n_mfcc, 10))
            
            # Process the audio
            preprocess_audio(mock_audio_file, feature_config)
            
            # Verify MFCC was created with correct parameters
            mock_mfcc.assert_called_once_with(
                sample_rate=feature_config.mutual_sample_rate,
                n_mfcc=feature_config.audio_n_mfcc,
                melkwargs={
                    'n_mels': feature_config.audio_n_mels,
                    'win_length': feature_config.audio_window_size,
                    'hop_length': feature_config.audio_hop_length
                }
            )

    def test_file_not_found(self, feature_config):
        """
        GIVEN a non-existent audio file path
        WHEN preprocess_audio is called
        THEN it should raise FileNotFoundError
        """
        non_existent_path = Path("/path/to/nonexistent/audio.wav")
        
        with pytest.raises(FileNotFoundError):
            preprocess_audio(non_existent_path, feature_config)

    def test_invalid_audio_file(self, feature_config, tmp_path):
        """
        GIVEN an invalid audio file (not a proper audio format)
        WHEN preprocess_audio is called
        THEN it should raise an appropriate error
        """
        # Create an invalid "audio" file (just a text file)
        invalid_file = tmp_path / "invalid_audio.wav"
        with open(invalid_file, 'w') as f:
            f.write("This is not audio data")
        
        with pytest.raises(Exception) as excinfo:
            preprocess_audio(invalid_file, feature_config)
        
        # The exact exception type might vary depending on torchaudio implementation
        # but we should get some kind of error
        assert excinfo.value is not None

    def test_zero_length_audio(self, feature_config, tmp_path):
        """
        GIVEN an audio file with zero length
        WHEN preprocess_audio is called
        THEN it should handle the edge case appropriately
        """
        # Create a zero-length waveform
        zero_waveform = torch.zeros((1, 0))
        zero_sample_rate = feature_config.mutual_sample_rate
        
        with patch('torchaudio.load', return_value=(zero_waveform, zero_sample_rate)):
            # This test verifies the function doesn't crash with zero-length input
            # The exact behavior (error or empty tensor) depends on implementation details
            try:
                result = preprocess_audio(Path("dummy_path.wav"), feature_config)
                # If it returns a result, it should be a valid tensor with zero time steps
                assert isinstance(result, torch.Tensor)
                assert result.size(2) == 0 or result.numel() == 0
            except Exception as e:
                # If it raises an exception, it should be a specific, handled exception
                # not a general crash
                assert isinstance(e, (ValueError, RuntimeError))
                assert "empty" in str(e).lower() or "zero" in str(e).lower()
    
    def test_multichannel_audio(self, feature_config):
        """
        GIVEN an audio file with multiple channels
        WHEN preprocess_audio is called
        THEN it should process all channels correctly
        """
        # Create a multi-channel waveform (2 channels)
        multi_waveform = torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])  # 2 channels, 3 samples
        
        with patch('torchaudio.load', return_value=(multi_waveform, feature_config.mutual_sample_rate)):
            with patch('torchaudio.transforms.MFCC') as mock_mfcc:
                # Configure mock MFCC transform to return appropriate shape
                mock_mfcc_instance = mock_mfcc.return_value
                # Return tensor with shape matching input channels (2 channels)
                mock_mfcc_instance.return_value = torch.zeros((2, feature_config.audio_n_mfcc, 10))
                
                # Process the audio
                result = preprocess_audio(Path("dummy_path.wav"), feature_config)
                
                # Verify MFCC was called with the multi-channel waveform
                args, _ = mock_mfcc_instance.call_args
                passed_waveform = args[0]
                assert passed_waveform.size(0) == 2, "Should preserve multiple channels"
                
                # Verify result has the expected number of channels
                assert result.size(0) == 2, "Output should have same number of channels as input"
    
    def test_extreme_values(self, feature_config):
        """
        GIVEN an audio file with extreme amplitude values
        WHEN preprocess_audio is called with normalization enabled
        THEN it should normalize correctly without numerical issues
        """
        # Create a waveform with extreme values
        extreme_waveform = torch.tensor([[1e-10, 1e10]])  # Very small and very large values
        
        with patch('torchaudio.load', return_value=(extreme_waveform, feature_config.mutual_sample_rate)):
            with patch('torchaudio.transforms.MFCC') as mock_mfcc:
                # Configure mock MFCC transform
                mock_mfcc_instance = mock_mfcc.return_value
                mock_mfcc_instance.return_value = torch.zeros((1, feature_config.audio_n_mfcc, 10))
                
                # Enable normalization
                feature_config.audio_normalize = True
                
                # Process the audio
                preprocess_audio(Path("dummy_path.wav"), feature_config)
                
                # Get the waveform that was passed to MFCC transform
                args, _ = mock_mfcc_instance.call_args
                normalized_waveform = args[0]
                
                # Verify normalization handled extreme values correctly
                assert torch.isfinite(normalized_waveform).all(), "Normalized values should be finite"
                assert normalized_waveform.abs().max() == 1.0, "Maximum absolute value should be 1.0"
                assert not torch.isnan(normalized_waveform).any(), "No NaN values should be present"
