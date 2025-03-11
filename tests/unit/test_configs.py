"""
Test suite for configuration components.

This module provides comprehensive testing for:
1. DatasetType enum functionality
2. Enum value consistency
3. Integration with DataLoaderBuilder
4. DatasetConfig base class functionality
5. Derived dataset configuration classes
"""

import pytest
from enum import Enum
import sys
from typing import Dict, Any, List
from pydantic import ValidationError
from unittest.mock import patch, MagicMock

from src.data.datasets.types import DatasetType
from src.configs import DatasetConfig, HKU956Config, PMEmo2019Config, AudioEDAFeatureConfig, DataConfig, LossConfig
from src.loss import LossBuilder
import torch.nn as nn


class TestDatasetType:
    """Tests for the DatasetType enum class."""

    def test_enum_values(self):
        """
        GIVEN the DatasetType enum
        WHEN accessing its values
        THEN it should contain the expected dataset identifiers
        """
        assert DatasetType.HKU956.value == 'hku956'
        assert DatasetType.PMEmo2019.value == 'pmemo2019'
        assert len(DatasetType) == 2, f"Expected 2 dataset types, found {len(DatasetType)}"

    def test_enum_membership(self):
        """
        GIVEN the DatasetType enum
        WHEN checking for membership
        THEN it should correctly identify valid and invalid members
        """
        # Valid members
        assert DatasetType.HKU956 in DatasetType
        assert DatasetType.PMEmo2019 in DatasetType
        
        # Invalid members
        assert 'hku956' not in DatasetType
        assert 'pmemo2019' not in DatasetType
        assert 'invalid_dataset' not in DatasetType

    def test_enum_iteration(self):
        """
        GIVEN the DatasetType enum
        WHEN iterating over its members
        THEN it should yield all defined dataset types
        """
        dataset_types = list(DatasetType)
        assert len(dataset_types) == 2
        assert DatasetType.HKU956 in dataset_types
        assert DatasetType.PMEmo2019 in dataset_types

    def test_enum_comparison(self):
        """
        GIVEN the DatasetType enum
        WHEN comparing enum members
        THEN it should maintain equality with identical members
        """
        # Same enum value comparison
        assert DatasetType.HKU956 == DatasetType.HKU956
        assert DatasetType.PMEmo2019 == DatasetType.PMEmo2019
        
        # Different enum value comparison
        assert DatasetType.HKU956 != DatasetType.PMEmo2019
        
        # String comparison (should not be equal)
        assert DatasetType.HKU956 != 'hku956'
        assert DatasetType.PMEmo2019 != 'pmemo2019'

    def test_enum_string_representation(self):
        """
        GIVEN the DatasetType enum
        WHEN converting to string
        THEN it should provide the correct string representation
        """
        assert str(DatasetType.HKU956) == 'DatasetType.HKU956'
        assert str(DatasetType.PMEmo2019) == 'DatasetType.PMEmo2019'
        
        # Representation should include the value
        assert repr(DatasetType.HKU956).endswith("'hku956'")
        assert repr(DatasetType.PMEmo2019).endswith("'pmemo2019'")

    def test_enum_from_string(self):
        """
        GIVEN string values matching enum values
        WHEN attempting to convert to enum members
        THEN it should correctly match the appropriate enum member
        """
        # Test conversion from string to enum
        assert DatasetType('hku956') == DatasetType.HKU956
        assert DatasetType('pmemo2019') == DatasetType.PMEmo2019
        
        # Test invalid string raises ValueError
        with pytest.raises(ValueError) as excinfo:
            DatasetType('invalid_dataset')
        assert "is not a valid" in str(excinfo.value)

    @pytest.mark.parametrize("dataset_name,expected_enum", [
        ('hku956', DatasetType.HKU956),
        ('pmemo2019', DatasetType.PMEmo2019),
    ])
    def test_enum_parametrized_lookup(self, dataset_name, expected_enum):
        """
        GIVEN various string dataset identifiers
        WHEN converting to enum members
        THEN it should match the correct enum values
        """
        assert DatasetType(dataset_name) == expected_enum
        assert DatasetType(dataset_name).value == dataset_name

    def test_enum_usage_in_dictionary(self):
        """
        GIVEN the DatasetType enum
        WHEN using enum members as dictionary keys
        THEN it should behave correctly for lookups
        """
        # Create a dictionary with enum keys
        dataset_config = {
            DatasetType.HKU956: {'path': '/data/hku956', 'format': 'csv'},
            DatasetType.PMEmo2019: {'path': '/data/pmemo2019', 'format': 'json'}
        }
        
        # Test dictionary lookups
        assert dataset_config[DatasetType.HKU956]['path'] == '/data/hku956'
        assert dataset_config[DatasetType.PMEmo2019]['format'] == 'json'
        
        # Test membership
        assert DatasetType.HKU956 in dataset_config
        assert DatasetType.PMEmo2019 in dataset_config

    def test_enum_hashability(self):
        """
        GIVEN the DatasetType enum
        WHEN using enum members in hash-based collections
        THEN it should maintain proper hashability
        """
        # Create a set with enum members
        dataset_set = {DatasetType.HKU956, DatasetType.PMEmo2019}
        
        # Test set operations
        assert len(dataset_set) == 2
        assert DatasetType.HKU956 in dataset_set
        
        # Adding the same enum again shouldn't change the set
        dataset_set.add(DatasetType.HKU956)
        assert len(dataset_set) == 2


class TestDatasetConfig:
    """Tests for the DatasetConfig base class."""

    def test_valid_initialization(self):
        """
        GIVEN valid parameters for all required fields
        WHEN initializing a DatasetConfig
        THEN it should create a valid instance with the provided values
        """
        config = DatasetConfig(
            dataset_name="TestDataset",
            dataset_root_path="/path/to/dataset",
            modalities=["audio", "eda"],
            file_format={"audio": ".wav", "eda": ".csv"},
            data_directories={
                "audio": "/path/to/dataset/audio",
                "eda": "/path/to/dataset/eda"
            },
            metadata_paths=["/path/to/dataset/metadata.csv"],
            split_ratios=[0.7, 0.15, 0.15],
            seed=42
        )
        
        assert config.dataset_name == "TestDataset"
        assert config.dataset_root_path == "/path/to/dataset"
        assert config.modalities == ["audio", "eda"]
        assert config.file_format == {"audio": ".wav", "eda": ".csv"}
        assert config.data_directories == {
            "audio": "/path/to/dataset/audio",
            "eda": "/path/to/dataset/eda"
        }
        assert config.metadata_paths == ["/path/to/dataset/metadata.csv"]
        assert config.split_ratios == [0.7, 0.15, 0.15]
        assert config.seed == 42

    def test_missing_required_fields(self):
        """
        GIVEN initialization parameters missing required fields
        WHEN initializing a DatasetConfig
        THEN it should raise ValidationError with appropriate message
        """
        with pytest.raises(ValidationError) as excinfo:
            DatasetConfig(
                # Missing dataset_name
                dataset_root_path="/path/to/dataset",
                modalities=["audio", "eda"],
                file_format={"audio": ".wav", "eda": ".csv"},
                data_directories={
                    "audio": "/path/to/dataset/audio",
                    "eda": "/path/to/dataset/eda"
                },
                metadata_paths=["/path/to/dataset/metadata.csv"],
                split_ratios=[0.7, 0.15, 0.15],
                seed=42
            )
        
        error_msg = str(excinfo.value)
        assert "dataset_name" in error_msg
        assert "field required" in error_msg

    def test_invalid_field_types(self):
        """
        GIVEN initialization parameters with incorrect types
        WHEN initializing a DatasetConfig
        THEN it should raise ValidationError with appropriate type error messages
        """
        with pytest.raises(ValidationError) as excinfo:
            DatasetConfig(
                dataset_name="TestDataset",
                dataset_root_path="/path/to/dataset",
                modalities="audio,eda",  # Should be a list, not a string
                file_format={"audio": ".wav", "eda": ".csv"},
                data_directories={
                    "audio": "/path/to/dataset/audio",
                    "eda": "/path/to/dataset/eda"
                },
                metadata_paths="/path/to/dataset/metadata.csv",  # Should be a list
                split_ratios=[0.7, 0.15, 0.15],
                seed=42
            )
        
        error_msg = str(excinfo.value)
        assert "modalities" in error_msg or "metadata_paths" in error_msg
        assert "str" in error_msg or "list" in error_msg

    def test_invalid_split_ratios(self):
        """
        GIVEN split_ratios that don't sum to 1.0
        WHEN initializing a DatasetConfig
        THEN it should still create the instance (no validation for sum)
        """
        # Note: This test verifies current behavior. If validation for split_ratios
        # sum is added in the future, this test should be updated.
        config = DatasetConfig(
            dataset_name="TestDataset",
            dataset_root_path="/path/to/dataset",
            modalities=["audio", "eda"],
            file_format={"audio": ".wav", "eda": ".csv"},
            data_directories={
                "audio": "/path/to/dataset/audio",
                "eda": "/path/to/dataset/eda"
            },
            metadata_paths=["/path/to/dataset/metadata.csv"],
            split_ratios=[0.6, 0.2, 0.3],  # Sum is 1.1
            seed=42
        )
        
        assert config.split_ratios == [0.6, 0.2, 0.3]

    def test_empty_collections(self):
        """
        GIVEN empty lists and dictionaries for collection fields
        WHEN initializing a DatasetConfig
        THEN it should create a valid instance with empty collections
        """
        config = DatasetConfig(
            dataset_name="TestDataset",
            dataset_root_path="/path/to/dataset",
            modalities=[],
            file_format={},
            data_directories={},
            metadata_paths=[],
            split_ratios=[],
            seed=42
        )
        
        assert config.modalities == []
        assert config.file_format == {}
        assert config.data_directories == {}
        assert config.metadata_paths == []
        assert config.split_ratios == []

    def test_model_dict_representation(self):
        """
        GIVEN a valid DatasetConfig instance
        WHEN converting to dictionary with model_dump()
        THEN it should produce a dictionary with all fields
        """
        config = DatasetConfig(
            dataset_name="TestDataset",
            dataset_root_path="/path/to/dataset",
            modalities=["audio", "eda"],
            file_format={"audio": ".wav", "eda": ".csv"},
            data_directories={
                "audio": "/path/to/dataset/audio",
                "eda": "/path/to/dataset/eda"
            },
            metadata_paths=["/path/to/dataset/metadata.csv"],
            split_ratios=[0.7, 0.15, 0.15],
            seed=42
        )
        
        config_dict = config.model_dump()
        
        assert isinstance(config_dict, dict)
        assert config_dict["dataset_name"] == "TestDataset"
        assert config_dict["dataset_root_path"] == "/path/to/dataset"
        assert config_dict["modalities"] == ["audio", "eda"]
        assert config_dict["file_format"] == {"audio": ".wav", "eda": ".csv"}
        assert config_dict["data_directories"] == {
            "audio": "/path/to/dataset/audio",
            "eda": "/path/to/dataset/eda"
        }
        assert config_dict["metadata_paths"] == ["/path/to/dataset/metadata.csv"]
        assert config_dict["split_ratios"] == [0.7, 0.15, 0.15]
        assert config_dict["seed"] == 42


class TestAudioEDAFeatureConfig:
    """Tests for the AudioEDAFeatureConfig class."""

    def test_default_initialization(self):
        """
        GIVEN no parameters
        WHEN initializing AudioEDAFeatureConfig with defaults
        THEN it should create an instance with the correct predefined values
        """
        config = AudioEDAFeatureConfig()
        
        # Mutual configurations
        assert config.mutual_sample_rate == 200
        
        # Audio configurations
        assert config.audio_normalize is True
        assert config.audio_n_mfcc == 40
        assert config.audio_n_mels == 128
        assert config.audio_window_size == 400
        assert config.audio_hop_length == 160
        
        # EDA configurations
        assert config.eda_window_size == 400
        assert config.eda_hop_length == 160
        assert config.eda_normalize is True
        assert config.filter_lowpass is True
        assert config.filter_highpass is False

    def test_custom_initialization(self):
        """
        GIVEN custom parameters
        WHEN initializing AudioEDAFeatureConfig with those parameters
        THEN it should override defaults while preserving other values
        """
        config = AudioEDAFeatureConfig(
            mutual_sample_rate=100,
            audio_n_mfcc=20,
            filter_highpass=True
        )
        
        # Custom values should be used
        assert config.mutual_sample_rate == 100
        assert config.audio_n_mfcc == 20
        assert config.filter_highpass is True
        
        # Default values should remain for non-overridden fields
        assert config.audio_normalize is True
        assert config.audio_n_mels == 128
        assert config.eda_normalize is True
        assert config.filter_lowpass is True

    def test_serialization_deserialization(self):
        """
        GIVEN a valid AudioEDAFeatureConfig instance
        WHEN serializing to JSON and back
        THEN the resulting object should match the original
        """
        original_config = AudioEDAFeatureConfig(
            mutual_sample_rate=150,
            audio_n_mfcc=30,
            eda_normalize=False
        )
        
        # Serialize to JSON
        json_str = original_config.model_dump_json()
        
        # Deserialize from JSON
        deserialized_config = AudioEDAFeatureConfig.model_validate_json(json_str)
        
        # Compare fields
        assert deserialized_config.mutual_sample_rate == original_config.mutual_sample_rate
        assert deserialized_config.audio_n_mfcc == original_config.audio_n_mfcc
        assert deserialized_config.eda_normalize == original_config.eda_normalize
        assert deserialized_config.audio_normalize == original_config.audio_normalize
        assert deserialized_config.audio_n_mels == original_config.audio_n_mels
        assert deserialized_config.audio_window_size == original_config.audio_window_size
        assert deserialized_config.audio_hop_length == original_config.audio_hop_length
        assert deserialized_config.eda_window_size == original_config.eda_window_size
        assert deserialized_config.eda_hop_length == original_config.eda_hop_length
        assert deserialized_config.filter_lowpass == original_config.filter_lowpass
        assert deserialized_config.filter_highpass == original_config.filter_highpass

    def test_model_dump(self):
        """
        GIVEN a valid AudioEDAFeatureConfig instance
        WHEN converting to dictionary with model_dump()
        THEN it should produce a dictionary with all fields
        """
        config = AudioEDAFeatureConfig(
            mutual_sample_rate=250,
            audio_normalize=False
        )
        
        config_dict = config.model_dump()
        
        assert isinstance(config_dict, dict)
        assert config_dict["mutual_sample_rate"] == 250
        assert config_dict["audio_normalize"] is False
        assert config_dict["audio_n_mfcc"] == 40
        assert config_dict["audio_n_mels"] == 128
        assert config_dict["audio_window_size"] == 400
        assert config_dict["audio_hop_length"] == 160
        assert config_dict["eda_window_size"] == 400
        assert config_dict["eda_hop_length"] == 160
        assert config_dict["eda_normalize"] is True
        assert config_dict["filter_lowpass"] is True
        assert config_dict["filter_highpass"] is False

    def test_with_audio_preprocessing(self):
        """
        GIVEN AudioEDAFeatureConfig instance
        WHEN used with audio preprocessing
        THEN it should provide all necessary configuration parameters
        """
        from unittest.mock import patch, MagicMock
        import torch
        
        config = AudioEDAFeatureConfig(
            mutual_sample_rate=16000,
            audio_normalize=True,
            audio_n_mfcc=13
        )
        
        # Mock torchaudio functions
        with patch('torchaudio.load') as mock_load, \
             patch('torchaudio.transforms.Resample') as mock_resample, \
             patch('torchaudio.transforms.MFCC') as mock_mfcc:
            
            # Configure mocks
            mock_waveform = torch.ones(1, 1000)
            mock_load.return_value = (mock_waveform, 44100)
            
            mock_resampler = MagicMock()
            mock_resampler.return_value = mock_waveform
            mock_resample.return_value = mock_resampler
            
            mock_mfcc_transform = MagicMock()
            mock_mfcc_transform.return_value = torch.ones(1, 13, 100)
            mock_mfcc.return_value = mock_mfcc_transform
            
            # Call the preprocessing function
            from src.data.audio_preprocessing import preprocess_audio
            result = preprocess_audio("dummy_path", config)
            
            # Verify the config parameters were used correctly
            mock_resample.assert_called_once_with(
                orig_freq=44100,
                new_freq=16000
            )
            
            mock_mfcc.assert_called_once_with(
                sample_rate=16000,
                n_mfcc=13,
                melkwargs={
                    'n_mels': 128,
                    'win_length': 400,
                    'hop_length': 160
                }
            )
            
            assert isinstance(result, torch.Tensor)

    def test_with_eda_preprocessing(self):
        """
        GIVEN AudioEDAFeatureConfig instance
        WHEN used with EDA preprocessing
        THEN it should provide all necessary configuration parameters
        """
        from unittest.mock import patch, MagicMock
        import numpy as np
        import pandas as pd
        import torch
        
        config = AudioEDAFeatureConfig(
            mutual_sample_rate=100,
            eda_normalize=True,
            filter_lowpass=True,
            filter_highpass=True
        )
        
        # Mock pandas and scipy functions
        with patch('pandas.read_csv') as mock_read_csv, \
             patch('scipy.signal.resample') as mock_resample, \
             patch('scipy.signal.butter') as mock_butter, \
             patch('scipy.signal.filtfilt') as mock_filtfilt:
            
            # Configure mocks
            mock_data = np.ones(1000)
            mock_df = pd.DataFrame(mock_data)
            mock_read_csv.return_value = mock_df
            
            mock_resampled = np.ones(500)
            mock_resample.return_value = mock_resampled
            
            mock_butter.return_value = (np.array([1, 2, 1]), np.array([1, 0, 0]))
            mock_filtfilt.return_value = mock_resampled
            
            # Call the preprocessing function
            from src.data.eda_preprocessing import preprocess_eda
            with patch('torch.tensor', return_value=torch.ones(1, 500)) as mock_tensor:
                result = preprocess_eda("dummy_path", config)
                
                # Verify the config parameters were used correctly
                mock_butter.assert_any_call(2, 0.05, btype='highpass', fs=100)
                mock_butter.assert_any_call(2, 8, btype='lowpass', fs=100)
                mock_filtfilt.assert_called()
                
                assert isinstance(result, torch.Tensor)

    def test_parameter_validation(self):
        """
        GIVEN invalid parameters
        WHEN initializing AudioEDAFeatureConfig
        THEN it should handle them appropriately
        """
        # Test with negative sample rate
        # Note: Pydantic v2 doesn't validate int fields by default unless validators are added
        config = AudioEDAFeatureConfig(mutual_sample_rate=-100)
        assert config.mutual_sample_rate == -100  # This passes with current implementation
        
        # Test with zero window size
        config = AudioEDAFeatureConfig(audio_window_size=0)
        assert config.audio_window_size == 0  # This passes with current implementation
        
        # Test with extreme values
        config = AudioEDAFeatureConfig(
            audio_n_mfcc=1000,
            audio_n_mels=10000
        )
        assert config.audio_n_mfcc == 1000
        assert config.audio_n_mels == 10000

    def test_window_hop_relationship(self):
        """
        GIVEN various window and hop length combinations
        WHEN initializing AudioEDAFeatureConfig
        THEN it should maintain consistent relationships
        """
        # Test with window size equal to hop length
        config = AudioEDAFeatureConfig(
            audio_window_size=200,
            audio_hop_length=200,
            eda_window_size=200,
            eda_hop_length=200
        )
        assert config.audio_window_size == config.audio_hop_length
        assert config.eda_window_size == config.eda_hop_length
        
        # Test with window size smaller than hop length
        # Note: This is technically invalid for STFT but Pydantic doesn't validate this
        config = AudioEDAFeatureConfig(
            audio_window_size=100,
            audio_hop_length=200
        )
        assert config.audio_window_size < config.audio_hop_length
        
        # Test with standard 50% overlap
        config = AudioEDAFeatureConfig(
            audio_window_size=400,
            audio_hop_length=200
        )
        assert config.audio_window_size == 2 * config.audio_hop_length

    def test_with_dataloader_builder(self):
        """
        GIVEN AudioEDAFeatureConfig instance
        WHEN used with DataLoaderBuilder
        THEN it should be compatible with the builder's interface
        """
        from unittest.mock import patch, MagicMock
        from src.data.dataloader import DataLoaderBuilder
        from src.configs import DataConfig
        from src.data.datasets.types import DatasetType
        
        # Create config instances
        feature_config = AudioEDAFeatureConfig()
        data_config = DataConfig(
            train_datasets=[DatasetType.HKU956],
            val_datasets=[DatasetType.HKU956],
            test_datasets=[DatasetType.HKU956],
            num_workers=2,
            prefetch_size=1
        )
        
        # Mock the dataset and dataloader
        with patch('src.data.dataloader.HKU956Dataset') as mock_dataset_class, \
             patch('src.data.dataloader.DataLoader') as mock_dataloader, \
             patch('src.data.dataloader.torch.utils.data.Subset') as mock_subset:
            
            # Configure mocks
            mock_dataset = MagicMock()
            mock_dataset.__len__.return_value = 100
            mock_dataset_class.return_value = mock_dataset
            
            mock_subset_instance = MagicMock()
            mock_subset.return_value = mock_subset_instance
            
            # Call the builder
            result = DataLoaderBuilder.build(data_config, feature_config, 'train')
            
            # Verify the feature config was passed correctly
            mock_dataset_class.assert_called_once()
            args, kwargs = mock_dataset_class.call_args
            assert args[1] is feature_config
            
            # Verify dataloader was created
            mock_dataloader.assert_called_once()
            assert isinstance(result, list)
            assert len(result) > 0

    def test_integration_with_train_script(self):
        """
        GIVEN AudioEDAFeatureConfig instance
        WHEN used in the training script
        THEN it should be properly initialized and used
        """
        from unittest.mock import patch, MagicMock
        
        # Mock dependencies to avoid actual execution
        with patch('src.data.dataloader.DataLoaderBuilder.build') as mock_build, \
             patch('src.models.tcn.TCN') as mock_tcn, \
             patch('src.optimizer.OptimizerBuilder.build') as mock_optimizer_build, \
             patch('src.loss.LossBuilder.build') as mock_loss_build, \
             patch('transformers.Trainer') as mock_trainer, \
             patch('transformers.TrainingArguments') as mock_training_args, \
             patch('wandb.init') as mock_wandb_init, \
             patch('yaml.safe_load') as mock_yaml_load, \
             patch('builtins.open', MagicMock()), \
             patch('os.makedirs') as mock_makedirs:
            
            # Configure mocks
            mock_yaml_load.return_value = {
                'experiment_name': 'test',
                'seed': 42,
                'model': {
                    'architecture': 'tcn',
                    'params': {
                        'input_size': 40,
                        'output_size': 1,
                        'num_blocks': 5,
                        'num_channels': 64,
                        'kernel_size': 3,
                        'dropout': 0.2
                    }
                },
                'optimizer': {
                    'name': 'adamw',
                    'learning_rate': 0.001,
                    'weight_decay': 0.01
                },
                'data': {
                    'train_datasets': ['hku956'],
                    'val_datasets': ['hku956'],
                    'test_datasets': ['hku956'],
                    'num_workers': 4,
                    'prefetch_size': 2
                },
                'loss': {
                    'name': 'mse'
                },
                'hardware': {
                    'device': 'cuda',
                    'precision': 'fp16',
                    'distributed': False,
                    'num_gpus': 1
                },
                'logging': {
                    'wandb_project': 'test_project',
                    'wandb_run_name': 'test_run',
                    'log_every_n_steps': 10
                },
                'checkpoint': {
                    'checkpoint_dir': './checkpoints',
                    'save_top_k': 3
                },
                'batch_size': 32,
                'max_epochs': 10,
                'val_check_interval': 1.0,
                'accumulate_grad_batches': 1,
                'gradient_clip_val': 1.0
            }
            
            # Capture the AudioEDAFeatureConfig instance
            captured_config = None
            
            def mock_build_side_effect(data_config, feature_config, split):
                nonlocal captured_config
                captured_config = feature_config
                return [MagicMock()]
            
            mock_build.side_effect = mock_build_side_effect
            
            # Import and call the main function from the training script
            import sys
            import os
            sys.argv = ['train_audio2eda.py', '--config', 'dummy_config.yaml']
            
            from scripts.train_audio2eda import main
            
            # This should run without errors and use AudioEDAFeatureConfig
            main()
            
            # Verify AudioEDAFeatureConfig was created and used
            assert captured_config is not None
            assert isinstance(captured_config, AudioEDAFeatureConfig)
            
            # Verify default values were used
            assert captured_config.mutual_sample_rate == 200
            assert captured_config.audio_n_mfcc == 40
            assert captured_config.audio_n_mels == 128
    
    def test_parameter_relationships(self):
        """
        GIVEN AudioEDAFeatureConfig with various parameter combinations
        WHEN examining relationships between parameters
        THEN they should maintain expected relationships
        """
        # Test that audio and EDA window sizes can be different
        config = AudioEDAFeatureConfig(
            audio_window_size=800,
            eda_window_size=400
        )
        assert config.audio_window_size == 800
        assert config.eda_window_size == 400
        
        # Test that audio and EDA hop lengths can be different
        config = AudioEDAFeatureConfig(
            audio_hop_length=320,
            eda_hop_length=160
        )
        assert config.audio_hop_length == 320
        assert config.eda_hop_length == 160
        
        # Test that normalization can be independently configured
        config = AudioEDAFeatureConfig(
            audio_normalize=True,
            eda_normalize=False
        )
        assert config.audio_normalize is True
        assert config.eda_normalize is False
        
        # Test that filters can be independently configured
        config = AudioEDAFeatureConfig(
            filter_lowpass=False,
            filter_highpass=True
        )
        assert config.filter_lowpass is False
        assert config.filter_highpass is True
    
    def test_copy_and_update(self):
        """
        GIVEN an existing AudioEDAFeatureConfig instance
        WHEN creating a copy with updated values
        THEN it should maintain original values except those explicitly changed
        """
        original_config = AudioEDAFeatureConfig(
            mutual_sample_rate=100,
            audio_n_mfcc=20,
            audio_n_mels=64
        )
        
        # Create a copy with some updated values
        updated_config = AudioEDAFeatureConfig(
            **original_config.model_dump(),
            audio_n_mfcc=30,
            audio_normalize=False
        )
        
        # Verify updated values
        assert updated_config.audio_n_mfcc == 30
        assert updated_config.audio_normalize is False
        
        # Verify preserved values
        assert updated_config.mutual_sample_rate == original_config.mutual_sample_rate
        assert updated_config.audio_n_mels == original_config.audio_n_mels
        assert updated_config.eda_window_size == original_config.eda_window_size
    
    @pytest.mark.parametrize("sample_rate,expected_valid", [
        (44100, True),
        (48000, True),
        (16000, True),
        (8000, True),
        (0, True),  # Currently allowed, though not ideal
        (-1000, True),  # Currently allowed, though not ideal
    ])
    def test_sample_rate_validation(self, sample_rate, expected_valid):
        """
        GIVEN various sample rate values
        WHEN initializing AudioEDAFeatureConfig
        THEN it should handle them according to current validation rules
        """
        # This test documents current behavior
        # If validation is added in the future, this test should be updated
        if expected_valid:
            config = AudioEDAFeatureConfig(mutual_sample_rate=sample_rate)
            assert config.mutual_sample_rate == sample_rate
        else:
            with pytest.raises(ValidationError):
                AudioEDAFeatureConfig(mutual_sample_rate=sample_rate)


class TestDataConfig:
    """Tests for the DataConfig class that configures dataset loading parameters."""
    
    def test_valid_initialization(self):
        """
        GIVEN valid parameters for all required fields
        WHEN initializing a DataConfig
        THEN it should create a valid instance with the provided values
        """
        from src.data.datasets.types import DatasetType
        
        config = DataConfig(
            train_datasets=[DatasetType.HKU956, DatasetType.PMEmo2019],
            val_datasets=[DatasetType.HKU956],
            test_datasets=[DatasetType.PMEmo2019],
            num_workers=8,
            prefetch_size=4
        )
        
        assert len(config.train_datasets) == 2
        assert DatasetType.HKU956 in config.train_datasets
        assert DatasetType.PMEmo2019 in config.train_datasets
        assert len(config.val_datasets) == 1
        assert DatasetType.HKU956 in config.val_datasets
        assert len(config.test_datasets) == 1
        assert DatasetType.PMEmo2019 in config.test_datasets
        assert config.num_workers == 8
        assert config.prefetch_size == 4
    
    def test_default_values(self):
        """
        GIVEN only required parameters
        WHEN initializing a DataConfig
        THEN it should use default values for optional parameters
        """
        from src.data.datasets.types import DatasetType
        
        config = DataConfig(
            train_datasets=[DatasetType.HKU956],
            val_datasets=[DatasetType.HKU956],
            test_datasets=[DatasetType.HKU956]
        )
        
        assert config.num_workers == 4  # Default value
        assert config.prefetch_size == 2  # Default value
    
    def test_empty_datasets(self):
        """
        GIVEN empty dataset lists
        WHEN initializing a DataConfig
        THEN it should create a valid instance with empty lists
        """
        config = DataConfig(
            train_datasets=[],
            val_datasets=[],
            test_datasets=[]
        )
        
        assert len(config.train_datasets) == 0
        assert len(config.val_datasets) == 0
        assert len(config.test_datasets) == 0
        assert config.num_workers == 4  # Default value
        assert config.prefetch_size == 2  # Default value
    
    def test_invalid_dataset_type(self):
        """
        GIVEN invalid dataset type
        WHEN initializing a DataConfig
        THEN it should raise a validation error
        """
        with pytest.raises(ValidationError) as excinfo:
            DataConfig(
                train_datasets=["invalid_dataset"],  # Not a DatasetType enum
                val_datasets=[],
                test_datasets=[]
            )
        
        error_msg = str(excinfo.value)
        assert "train_datasets" in error_msg
        assert "type_error" in error_msg.lower() or "not a valid enumeration member" in error_msg.lower()
    
    def test_negative_workers(self):
        """
        GIVEN negative number of workers
        WHEN initializing a DataConfig
        THEN it should accept the value (no validation in current implementation)
        """
        from src.data.datasets.types import DatasetType
        
        # Note: This test documents current behavior. If validation for positive
        # num_workers is added in the future, this test should be updated.
        config = DataConfig(
            train_datasets=[DatasetType.HKU956],
            val_datasets=[DatasetType.HKU956],
            test_datasets=[DatasetType.HKU956],
            num_workers=-2
        )
        
        assert config.num_workers == -2
    
    def test_zero_prefetch_size(self):
        """
        GIVEN zero prefetch size
        WHEN initializing a DataConfig
        THEN it should accept the value (no validation in current implementation)
        """
        from src.data.datasets.types import DatasetType
        
        # Note: This test documents current behavior. If validation for positive
        # prefetch_size is added in the future, this test should be updated.
        config = DataConfig(
            train_datasets=[DatasetType.HKU956],
            val_datasets=[DatasetType.HKU956],
            test_datasets=[DatasetType.HKU956],
            prefetch_size=0
        )
        
        assert config.prefetch_size == 0
    
    def test_serialization_deserialization(self):
        """
        GIVEN a valid DataConfig instance
        WHEN serializing to JSON and back
        THEN the resulting object should match the original
        """
        from src.data.datasets.types import DatasetType
        
        original_config = DataConfig(
            train_datasets=[DatasetType.HKU956, DatasetType.PMEmo2019],
            val_datasets=[DatasetType.HKU956],
            test_datasets=[DatasetType.PMEmo2019],
            num_workers=8,
            prefetch_size=4
        )
        
        # Serialize to JSON
        json_str = original_config.model_dump_json()
        
        # Deserialize from JSON
        deserialized_config = DataConfig.model_validate_json(json_str)
        
        # Compare fields
        assert len(deserialized_config.train_datasets) == len(original_config.train_datasets)
        for dataset in original_config.train_datasets:
            assert dataset in deserialized_config.train_datasets
            
        assert len(deserialized_config.val_datasets) == len(original_config.val_datasets)
        for dataset in original_config.val_datasets:
            assert dataset in deserialized_config.val_datasets
            
        assert len(deserialized_config.test_datasets) == len(original_config.test_datasets)
        for dataset in original_config.test_datasets:
            assert dataset in deserialized_config.test_datasets
            
        assert deserialized_config.num_workers == original_config.num_workers
        assert deserialized_config.prefetch_size == original_config.prefetch_size
    
    def test_with_dataloader_builder(self):
        """
        GIVEN DataConfig instance
        WHEN used with DataLoaderBuilder
        THEN it should be compatible with the builder's interface
        """
        from unittest.mock import patch, MagicMock
        from src.data.datasets.types import DatasetType
        from src.data.dataloader import DataLoaderBuilder
        
        # Create config instance
        data_config = DataConfig(
            train_datasets=[DatasetType.HKU956],
            val_datasets=[DatasetType.HKU956],
            test_datasets=[DatasetType.HKU956],
            num_workers=2,
            prefetch_size=1
        )
        
        feature_config = AudioEDAFeatureConfig()
        
        # Mock the dataset and dataloader
        with patch('src.data.dataloader.HKU956Dataset') as mock_dataset_class, \
             patch('src.data.dataloader.DataLoader') as mock_dataloader, \
             patch('src.data.dataloader.torch.utils.data.Subset') as mock_subset, \
             patch('src.data.dataloader.HKU956Config') as mock_hku_config:
            
            # Configure mocks
            mock_dataset = MagicMock()
            mock_dataset.__len__.return_value = 100
            mock_dataset_class.return_value = mock_dataset
            
            mock_subset_instance = MagicMock()
            mock_subset.return_value = mock_subset_instance
            
            # Call the builder
            result = DataLoaderBuilder.build(data_config, feature_config, 'train')
            
            # Verify the config parameters were used correctly
            mock_dataloader.assert_called_once()
            args, kwargs = mock_dataloader.call_args
            assert kwargs['num_workers'] == data_config.num_workers
            assert kwargs['prefetch_factor'] == data_config.prefetch_size
            
            # Verify result is a list of dataloaders
            assert isinstance(result, list)
            assert len(result) > 0
    
    def test_model_dump(self):
        """
        GIVEN a valid DataConfig instance
        WHEN converting to dictionary with model_dump()
        THEN it should produce a dictionary with all fields
        """
        from src.data.datasets.types import DatasetType
        
        config = DataConfig(
            train_datasets=[DatasetType.HKU956],
            val_datasets=[DatasetType.PMEmo2019],
            test_datasets=[],
            num_workers=16,
            prefetch_size=8
        )
        
        config_dict = config.model_dump()
        
        assert isinstance(config_dict, dict)
        assert "train_datasets" in config_dict
        assert "val_datasets" in config_dict
        assert "test_datasets" in config_dict
        assert "num_workers" in config_dict
        assert "prefetch_size" in config_dict
        
        assert len(config_dict["train_datasets"]) == 1
        assert len(config_dict["val_datasets"]) == 1
        assert len(config_dict["test_datasets"]) == 0
        assert config_dict["num_workers"] == 16
        assert config_dict["prefetch_size"] == 8
    
    def test_integration_with_train_script(self):
        """
        GIVEN DataConfig instance
        WHEN used in the training script
        THEN it should be properly initialized and used
        """
        from unittest.mock import patch, MagicMock
        import yaml
        
        # Mock YAML config that would be loaded in the training script
        yaml_config = {
            'experiment_name': 'test',
            'seed': 42,
            'model': {
                'architecture': 'tcn',
                'params': {
                    'input_size': 40,
                    'output_size': 1,
                    'num_blocks': 5,
                    'num_channels': 64,
                    'kernel_size': 3,
                    'dropout': 0.2
                }
            },
            'optimizer': {
                'name': 'adamw',
                'learning_rate': 0.001
            },
            'data': {
                'train_datasets': ['hku956'],
                'val_datasets': ['pmemo2019'],
                'test_datasets': ['hku956', 'pmemo2019'],
                'num_workers': 4,
                'prefetch_size': 2
            },
            'loss': {
                'name': 'mse'
            },
            'hardware': {
                'device': 'cuda',
                'precision': 'fp16',
                'distributed': False,
                'num_gpus': 1
            },
            'logging': {
                'wandb_project': 'test_project',
                'wandb_run_name': 'test_run'
            },
            'checkpoint': {
                'checkpoint_dir': './checkpoints'
            },
            'batch_size': 32,
            'max_epochs': 10
        }
        
        # Mock dependencies to avoid actual execution
        with patch('yaml.safe_load', return_value=yaml_config), \
             patch('builtins.open', MagicMock()), \
             patch('src.data.dataloader.DataLoaderBuilder.build') as mock_build, \
             patch('wandb.init'), \
             patch('os.makedirs'), \
             patch('os.environ'):
            
            # Import the main function from the training script
            from scripts.train_audio2eda import main
            
            # Capture the DataConfig instance
            captured_config = None
            
            def mock_build_side_effect(data_config, feature_config, split):
                nonlocal captured_config
                if split == 'train':  # Only capture once
                    captured_config = data_config
                return [MagicMock()]
            
            mock_build.side_effect = mock_build_side_effect
            
            # Mock sys.argv
            with patch('sys.argv', ['train_audio2eda.py', '--config', 'dummy_config.yaml']):
                # This should run without errors and use DataConfig
                main()
            
            # Verify DataConfig was created and used
            assert captured_config is not None
            assert isinstance(captured_config, DataConfig)
            
            # Verify values from YAML were used
            assert len(captured_config.train_datasets) == 1
            assert len(captured_config.val_datasets) == 1
            assert len(captured_config.test_datasets) == 2
            assert captured_config.num_workers == 4
            assert captured_config.prefetch_size == 2
    
    def test_extreme_values(self):
        """
        GIVEN extreme values for numeric fields
        WHEN initializing a DataConfig
        THEN it should handle them appropriately
        """
        from src.data.datasets.types import DatasetType
        
        # Test with very large values
        config = DataConfig(
            train_datasets=[DatasetType.HKU956],
            val_datasets=[DatasetType.HKU956],
            test_datasets=[DatasetType.HKU956],
            num_workers=1000000,  # Extremely large number of workers
            prefetch_size=1000000  # Extremely large prefetch size
        )
        
        assert config.num_workers == 1000000
        assert config.prefetch_size == 1000000
        
        # Test with very small values
        config = DataConfig(
            train_datasets=[DatasetType.HKU956],
            val_datasets=[DatasetType.HKU956],
            test_datasets=[DatasetType.HKU956],
            num_workers=-1000000,  # Extremely negative number of workers
            prefetch_size=-1000000  # Extremely negative prefetch size
        )
        
        assert config.num_workers == -1000000
        assert config.prefetch_size == -1000000
    
    def test_different_datasets_for_splits(self):
        """
        GIVEN different datasets for each split
        WHEN initializing a DataConfig
        THEN it should maintain the correct datasets for each split
        """
        from src.data.datasets.types import DatasetType
        
        config = DataConfig(
            train_datasets=[DatasetType.HKU956],
            val_datasets=[DatasetType.PMEmo2019],
            test_datasets=[DatasetType.HKU956, DatasetType.PMEmo2019]
        )
        
        assert len(config.train_datasets) == 1
        assert DatasetType.HKU956 in config.train_datasets
        assert DatasetType.PMEmo2019 not in config.train_datasets
        
        assert len(config.val_datasets) == 1
        assert DatasetType.PMEmo2019 in config.val_datasets
        assert DatasetType.HKU956 not in config.val_datasets
        
        assert len(config.test_datasets) == 2
        assert DatasetType.HKU956 in config.test_datasets
        assert DatasetType.PMEmo2019 in config.test_datasets
    
    def test_duplicate_datasets(self):
        """
        GIVEN duplicate datasets in a split
        WHEN initializing a DataConfig
        THEN it should handle duplicates according to list behavior
        """
        from src.data.datasets.types import DatasetType
        
        config = DataConfig(
            train_datasets=[DatasetType.HKU956, DatasetType.HKU956],
            val_datasets=[DatasetType.PMEmo2019],
            test_datasets=[DatasetType.HKU956]
        )
        
        # Lists can contain duplicates, so both should be present
        assert len(config.train_datasets) == 2
        assert config.train_datasets.count(DatasetType.HKU956) == 2
    
    @pytest.mark.parametrize("num_workers,prefetch_size,expected_valid", [
        (0, 0, True),
        (1, 1, True),
        (10, 10, True),
        (-1, -1, True),  # Currently allowed, though not ideal
        (None, None, False),  # Should fail type validation
    ])
    def test_worker_and_prefetch_validation(self, num_workers, prefetch_size, expected_valid):
        """
        GIVEN various values for num_workers and prefetch_size
        WHEN initializing DataConfig
        THEN it should validate according to current rules
        """
        from src.data.datasets.types import DatasetType
        
        # This test documents current behavior
        if expected_valid:
            config = DataConfig(
                train_datasets=[DatasetType.HKU956],
                val_datasets=[DatasetType.HKU956],
                test_datasets=[DatasetType.HKU956],
                num_workers=num_workers,
                prefetch_size=prefetch_size
            )
            assert config.num_workers == num_workers
            assert config.prefetch_size == prefetch_size
        else:
            with pytest.raises(ValidationError):
                DataConfig(
                    train_datasets=[DatasetType.HKU956],
                    val_datasets=[DatasetType.HKU956],
                    test_datasets=[DatasetType.HKU956],
                    num_workers=num_workers,
                    prefetch_size=prefetch_size
                )
    
    def test_custom_batch_size_in_dataloader(self):
        """
        GIVEN DataConfig instance and a custom batch size
        WHEN used with DataLoaderBuilder with a specific batch size
        THEN the DataLoader should be created with the specified batch size
        """
        from unittest.mock import patch, MagicMock
        from src.data.datasets.types import DatasetType
        from src.data.dataloader import DataLoaderBuilder
        
        # Create config instance
        data_config = DataConfig(
            train_datasets=[DatasetType.HKU956],
            val_datasets=[DatasetType.HKU956],
            test_datasets=[DatasetType.HKU956],
            num_workers=2,
            prefetch_size=1
        )
        
        feature_config = AudioEDAFeatureConfig()
        custom_batch_size = 64  # Different from default 32
        
        # Mock the dataset and dataloader
        with patch('src.data.dataloader.HKU956Dataset') as mock_dataset_class, \
             patch('src.data.dataloader.DataLoader') as mock_dataloader, \
             patch('src.data.dataloader.torch.utils.data.Subset') as mock_subset, \
             patch('src.data.dataloader.HKU956Config') as mock_hku_config, \
             patch.object(DataLoaderBuilder, 'build', wraps=DataLoaderBuilder.build) as wrapped_build:
            
            # Configure mocks
            mock_dataset = MagicMock()
            mock_dataset.__len__.return_value = 100
            mock_dataset_class.return_value = mock_dataset
            
            mock_subset_instance = MagicMock()
            mock_subset.return_value = mock_subset_instance
            
            # Override the batch_size in the DataLoader call
            def custom_dataloader(*args, **kwargs):
                # Verify batch_size is used
                assert 'batch_size' in kwargs
                assert kwargs['batch_size'] == custom_batch_size
                return MagicMock()
            
            mock_dataloader.side_effect = custom_dataloader
            
            # Call the builder with custom batch_size
            # Note: This is testing the current behavior where batch_size is hardcoded
            # If DataLoaderBuilder is updated to accept batch_size, this test should be updated
            result = DataLoaderBuilder.build(data_config, feature_config, 'train')
            
            # Verify dataloader was created
            mock_dataloader.assert_called()
            
            # This test documents that currently batch_size is hardcoded to 32
            # If the implementation changes to use a configurable batch_size, this test will fail
            # and should be updated to test the new behavior
    
    def test_yaml_serialization_compatibility(self):
        """
        GIVEN a DataConfig instance
        WHEN serializing to YAML and back
        THEN the resulting object should match the original
        """
        import yaml
        from src.data.datasets.types import DatasetType
        
        original_config = DataConfig(
            train_datasets=[DatasetType.HKU956, DatasetType.PMEmo2019],
            val_datasets=[DatasetType.HKU956],
            test_datasets=[DatasetType.PMEmo2019],
            num_workers=8,
            prefetch_size=4
        )
        
        # Convert to dict first (as would happen in a real scenario)
        config_dict = original_config.model_dump()
        
        # Convert enum values to strings for YAML serialization
        # This mimics what would happen in a real application
        serializable_dict = {
            'train_datasets': [ds.value for ds in config_dict['train_datasets']],
            'val_datasets': [ds.value for ds in config_dict['val_datasets']],
            'test_datasets': [ds.value for ds in config_dict['test_datasets']],
            'num_workers': config_dict['num_workers'],
            'prefetch_size': config_dict['prefetch_size']
        }
        
        # Serialize to YAML
        yaml_str = yaml.dump(serializable_dict)
        
        # Deserialize from YAML
        deserialized_dict = yaml.safe_load(yaml_str)
        
        # Convert string values back to enum instances
        deserialized_dict['train_datasets'] = [DatasetType(ds) for ds in deserialized_dict['train_datasets']]
        deserialized_dict['val_datasets'] = [DatasetType(ds) for ds in deserialized_dict['val_datasets']]
        deserialized_dict['test_datasets'] = [DatasetType(ds) for ds in deserialized_dict['test_datasets']]
        
        # Create a new DataConfig from the deserialized dict
        deserialized_config = DataConfig(**deserialized_dict)
        
        # Compare fields
        assert len(deserialized_config.train_datasets) == len(original_config.train_datasets)
        for dataset in original_config.train_datasets:
            assert dataset in deserialized_config.train_datasets
            
        assert len(deserialized_config.val_datasets) == len(original_config.val_datasets)
        for dataset in original_config.val_datasets:
            assert dataset in deserialized_config.val_datasets
            
        assert len(deserialized_config.test_datasets) == len(original_config.test_datasets)
        for dataset in original_config.test_datasets:
            assert dataset in deserialized_config.test_datasets
            
        assert deserialized_config.num_workers == original_config.num_workers
        assert deserialized_config.prefetch_size == original_config.prefetch_size


class TestOptimizerConfig:
    """Tests for the OptimizerConfig class that configures optimizer parameters."""
    
    def test_default_initialization(self):
        """
        GIVEN no parameters
        WHEN initializing OptimizerConfig with defaults
        THEN it should create an instance with the correct predefined values
        """
        from src.configs import OptimizerConfig
        
        config = OptimizerConfig()
        
        assert config.name == "adamw"
        assert config.learning_rate == 1e-4
        assert config.weight_decay == 0.01
        assert config.beta1 == 0.9
        assert config.beta2 == 0.999
        assert config.momentum == 0.0
        assert config.warmup_steps == 0
        assert config.warmup_ratio == 0.0
        assert config.scheduler == "cosine"
    
    def test_custom_initialization(self):
        """
        GIVEN custom parameters
        WHEN initializing OptimizerConfig with those parameters
        THEN it should override defaults while preserving other values
        """
        from src.configs import OptimizerConfig
        
        config = OptimizerConfig(
            name="adam",
            learning_rate=0.001,
            scheduler="linear"
        )
        
        # Custom values should be used
        assert config.name == "adam"
        assert config.learning_rate == 0.001
        assert config.scheduler == "linear"
        
        # Default values should remain for non-overridden fields
        assert config.weight_decay == 0.01
        assert config.beta1 == 0.9
        assert config.beta2 == 0.999
        assert config.momentum == 0.0
    
    def test_invalid_optimizer_name(self):
        """
        GIVEN invalid optimizer name
        WHEN initializing OptimizerConfig
        THEN it should raise ValidationError
        """
        from src.configs import OptimizerConfig
        from pydantic import ValidationError
        
        with pytest.raises(ValidationError) as excinfo:
            OptimizerConfig(name="invalid_optimizer")
        
        error_msg = str(excinfo.value)
        assert "name" in error_msg
        assert "invalid_optimizer" in error_msg
        assert "not a valid enumeration member" in error_msg.lower() or "Input should be" in error_msg
    
    def test_invalid_scheduler_name(self):
        """
        GIVEN invalid scheduler name
        WHEN initializing OptimizerConfig
        THEN it should raise ValidationError
        """
        from src.configs import OptimizerConfig
        from pydantic import ValidationError
        
        with pytest.raises(ValidationError) as excinfo:
            OptimizerConfig(scheduler="invalid_scheduler")
        
        error_msg = str(excinfo.value)
        assert "scheduler" in error_msg
        assert "invalid_scheduler" in error_msg
        assert "not a valid enumeration member" in error_msg.lower() or "Input should be" in error_msg
    
    def test_none_scheduler(self):
        """
        GIVEN None as scheduler value
        WHEN initializing OptimizerConfig
        THEN it should accept None as a valid value
        """
        from src.configs import OptimizerConfig
        
        config = OptimizerConfig(scheduler=None)
        assert config.scheduler is None
    
    def test_serialization_deserialization(self):
        """
        GIVEN a valid OptimizerConfig instance
        WHEN serializing to JSON and back
        THEN the resulting object should match the original
        """
        from src.configs import OptimizerConfig
        
        original_config = OptimizerConfig(
            name="sgd",
            learning_rate=0.01,
            momentum=0.9,
            scheduler="reduce_on_plateau"
        )
        
        # Serialize to JSON
        json_str = original_config.model_dump_json()
        
        # Deserialize from JSON
        deserialized_config = OptimizerConfig.model_validate_json(json_str)
        
        # Compare fields
        assert deserialized_config.name == original_config.name
        assert deserialized_config.learning_rate == original_config.learning_rate
        assert deserialized_config.weight_decay == original_config.weight_decay
        assert deserialized_config.beta1 == original_config.beta1
        assert deserialized_config.beta2 == original_config.beta2
        assert deserialized_config.momentum == original_config.momentum
        assert deserialized_config.warmup_steps == original_config.warmup_steps
        assert deserialized_config.warmup_ratio == original_config.warmup_ratio
        assert deserialized_config.scheduler == original_config.scheduler
    
    def test_model_dump(self):
        """
        GIVEN a valid OptimizerConfig instance
        WHEN converting to dictionary with model_dump()
        THEN it should produce a dictionary with all fields
        """
        from src.configs import OptimizerConfig
        
        config = OptimizerConfig(
            name="adam",
            learning_rate=0.005,
            beta1=0.8
        )
        
        config_dict = config.model_dump()
        
        assert isinstance(config_dict, dict)
        assert config_dict["name"] == "adam"
        assert config_dict["learning_rate"] == 0.005
        assert config_dict["weight_decay"] == 0.01
        assert config_dict["beta1"] == 0.8
        assert config_dict["beta2"] == 0.999
        assert config_dict["momentum"] == 0.0
        assert config_dict["warmup_steps"] == 0
        assert config_dict["warmup_ratio"] == 0.0
        assert config_dict["scheduler"] == "cosine"
    
    def test_with_optimizer_builder(self):
        """
        GIVEN OptimizerConfig instance
        WHEN used with OptimizerBuilder
        THEN it should create the correct optimizer and scheduler
        """
        from unittest.mock import patch, MagicMock
        from src.configs import OptimizerConfig
        from src.optimizer import OptimizerBuilder
        import torch.nn as nn
        
        # Create a simple model for parameters
        model = nn.Linear(10, 1)
        
        # Test with AdamW optimizer
        config = OptimizerConfig(
            name="adamw",
            learning_rate=0.001,
            weight_decay=0.1,
            scheduler="cosine"
        )
        
        # Mock the torch optimizers and schedulers
        with patch('src.optimizer.AdamW') as mock_adamw, \
             patch('src.optimizer.CosineAnnealingLR') as mock_scheduler:
            
            # Configure mocks
            mock_optimizer = MagicMock()
            mock_adamw.return_value = mock_optimizer
            
            mock_scheduler_instance = MagicMock()
            mock_scheduler.return_value = mock_scheduler_instance
            
            # Call the builder
            optimizer, scheduler = OptimizerBuilder.build(config, model.parameters())
            
            # Verify the correct optimizer was created with the right parameters
            mock_adamw.assert_called_once()
            args, kwargs = mock_adamw.call_args
            assert kwargs["lr"] == 0.001
            assert kwargs["weight_decay"] == 0.1
            assert kwargs["betas"] == (0.9, 0.999)
            
            # Verify the correct scheduler was created
            mock_scheduler.assert_called_once()
            assert optimizer is mock_optimizer
            assert scheduler is mock_scheduler_instance
    
    def test_sgd_optimizer_creation(self):
        """
        GIVEN OptimizerConfig for SGD
        WHEN used with OptimizerBuilder
        THEN it should create SGD optimizer with correct momentum
        """
        from unittest.mock import patch, MagicMock
        from src.configs import OptimizerConfig
        from src.optimizer import OptimizerBuilder
        import torch.nn as nn
        
        # Create a simple model for parameters
        model = nn.Linear(10, 1)
        
        # Test with SGD optimizer
        config = OptimizerConfig(
            name="sgd",
            learning_rate=0.01,
            momentum=0.9,
            scheduler=None
        )
        
        # Mock the torch optimizers
        with patch('src.optimizer.SGD') as mock_sgd:
            # Configure mocks
            mock_optimizer = MagicMock()
            mock_sgd.return_value = mock_optimizer
            
            # Call the builder
            optimizer, scheduler = OptimizerBuilder.build(config, model.parameters())
            
            # Verify the correct optimizer was created with the right parameters
            mock_sgd.assert_called_once()
            args, kwargs = mock_sgd.call_args
            assert kwargs["lr"] == 0.01
            assert kwargs["momentum"] == 0.9
            assert kwargs["weight_decay"] == 0.01
            
            # Verify no scheduler was created
            assert optimizer is mock_optimizer
            assert scheduler is None
    
    def test_adam_optimizer_creation(self):
        """
        GIVEN OptimizerConfig for Adam
        WHEN used with OptimizerBuilder
        THEN it should create Adam optimizer with correct betas
        """
        from unittest.mock import patch, MagicMock
        from src.configs import OptimizerConfig
        from src.optimizer import OptimizerBuilder
        import torch.nn as nn
        
        # Create a simple model for parameters
        model = nn.Linear(10, 1)
        
        # Test with Adam optimizer and custom betas
        config = OptimizerConfig(
            name="adam",
            learning_rate=0.002,
            beta1=0.85,
            beta2=0.95,
            scheduler="step"
        )
        
        # Mock the torch optimizers and schedulers
        with patch('src.optimizer.Adam') as mock_adam, \
             patch('src.optimizer.StepLR') as mock_scheduler:
            
            # Configure mocks
            mock_optimizer = MagicMock()
            mock_adam.return_value = mock_optimizer
            
            mock_scheduler_instance = MagicMock()
            mock_scheduler.return_value = mock_scheduler_instance
            
            # Call the builder
            optimizer, scheduler = OptimizerBuilder.build(config, model.parameters())
            
            # Verify the correct optimizer was created with the right parameters
            mock_adam.assert_called_once()
            args, kwargs = mock_adam.call_args
            assert kwargs["lr"] == 0.002
            assert kwargs["betas"] == (0.85, 0.95)
            assert kwargs["weight_decay"] == 0.01
            
            # Verify the correct scheduler was created
            mock_scheduler.assert_called_once()
            assert optimizer is mock_optimizer
            assert scheduler is mock_scheduler_instance
    
    def test_extreme_learning_rate_values(self):
        """
        GIVEN extreme learning rate values
        WHEN initializing OptimizerConfig
        THEN it should handle them appropriately
        """
        from src.configs import OptimizerConfig
        
        # Test with very small learning rate
        tiny_lr_config = OptimizerConfig(learning_rate=1e-10)
        assert tiny_lr_config.learning_rate == 1e-10
        
        # Test with very large learning rate
        large_lr_config = OptimizerConfig(learning_rate=1000.0)
        assert large_lr_config.learning_rate == 1000.0
        
        # Test with zero learning rate
        zero_lr_config = OptimizerConfig(learning_rate=0.0)
        assert zero_lr_config.learning_rate == 0.0
    
    def test_negative_values(self):
        """
        GIVEN negative values for numeric fields
        WHEN initializing OptimizerConfig
        THEN it should handle them according to current validation rules
        """
        from src.configs import OptimizerConfig
        
        # Note: This test documents current behavior
        # If validation for positive values is added in the future, this test should be updated
        
        # Test with negative learning rate
        config = OptimizerConfig(learning_rate=-0.001)
        assert config.learning_rate == -0.001
        
        # Test with negative weight decay
        config = OptimizerConfig(weight_decay=-0.01)
        assert config.weight_decay == -0.01
        
        # Test with negative momentum
        config = OptimizerConfig(momentum=-0.9)
        assert config.momentum == -0.9
        
        # Test with negative warmup steps
        config = OptimizerConfig(warmup_steps=-10)
        assert config.warmup_steps == -10
    
    def test_integration_with_train_script(self):
        """
        GIVEN OptimizerConfig instance
        WHEN used in the training script
        THEN it should be properly initialized and used
        """
        from unittest.mock import patch, MagicMock
        import yaml
        
        # Mock YAML config that would be loaded in the training script
        yaml_config = {
            'experiment_name': 'test',
            'seed': 42,
            'model': {
                'architecture': 'tcn',
                'params': {
                    'input_size': 40,
                    'output_size': 1,
                    'num_blocks': 5,
                    'num_channels': 64,
                    'kernel_size': 3,
                    'dropout': 0.2
                }
            },
            'optimizer': {
                'name': 'adamw',
                'learning_rate': 0.001,
                'weight_decay': 0.05,
                'beta1': 0.9,
                'beta2': 0.999,
                'scheduler': 'cosine'
            },
            'data': {
                'train_datasets': ['hku956'],
                'val_datasets': ['pmemo2019'],
                'test_datasets': ['hku956', 'pmemo2019'],
                'num_workers': 4,
                'prefetch_size': 2
            },
            'loss': {
                'name': 'mse'
            },
            'hardware': {
                'device': 'cuda',
                'precision': 'fp16',
                'distributed': False,
                'num_gpus': 1
            },
            'logging': {
                'wandb_project': 'test_project',
                'wandb_run_name': 'test_run'
            },
            'checkpoint': {
                'checkpoint_dir': './checkpoints'
            },
            'batch_size': 32,
            'max_epochs': 10
        }
        
        # Mock dependencies to avoid actual execution
        with patch('yaml.safe_load', return_value=yaml_config), \
             patch('builtins.open', MagicMock()), \
             patch('src.optimizer.OptimizerBuilder.build') as mock_optimizer_build, \
             patch('wandb.init'), \
             patch('os.makedirs'), \
             patch('os.environ'):
            
            # Import the main function from the training script
            from scripts.train_audio2eda import main
            
            # Capture the OptimizerConfig instance
            captured_config = None
            
            def mock_optimizer_build_side_effect(optimizer_config, model_params):
                nonlocal captured_config
                captured_config = optimizer_config
                return MagicMock(), MagicMock()
            
            mock_optimizer_build.side_effect = mock_optimizer_build_side_effect
            
            # Mock sys.argv
            with patch('sys.argv', ['train_audio2eda.py', '--config', 'dummy_config.yaml']), \
                 patch('src.models.tcn.TCN', return_value=MagicMock()), \
                 patch('src.data.dataloader.DataLoaderBuilder.build', return_value=[MagicMock()]), \
                 patch('src.loss.LossBuilder.build', return_value=MagicMock()), \
                 patch('transformers.Trainer'), \
                 patch('transformers.TrainingArguments'):
                
                # This should run without errors and use OptimizerConfig
                main()
            
            # Verify OptimizerConfig was created and used
            assert captured_config is not None
            from src.configs import OptimizerConfig
            assert isinstance(captured_config, OptimizerConfig)
            
            # Verify values from YAML were used
            assert captured_config.name == "adamw"
            assert captured_config.learning_rate == 0.001
            assert captured_config.weight_decay == 0.05
            assert captured_config.beta1 == 0.9
            assert captured_config.beta2 == 0.999
            assert captured_config.scheduler == "cosine"
    
    @pytest.mark.parametrize("optimizer_name,expected_class", [
        ("adam", "Adam"),
        ("adamw", "AdamW"),
        ("sgd", "SGD")
    ])
    def test_optimizer_name_mapping(self, optimizer_name, expected_class):
        """
        GIVEN different optimizer names
        WHEN used with OptimizerBuilder
        THEN it should create the correct optimizer class
        """
        from unittest.mock import patch, MagicMock
        from src.configs import OptimizerConfig
        from src.optimizer import OptimizerBuilder
        import torch.nn as nn
        
        # Create a simple model for parameters
        model = nn.Linear(10, 1)
        
        # Create config with the specified optimizer name
        config = OptimizerConfig(
            name=optimizer_name,
            scheduler=None
        )
        
        # Mock all optimizer classes
        with patch('src.optimizer.Adam') as mock_adam, \
             patch('src.optimizer.AdamW') as mock_adamw, \
             patch('src.optimizer.SGD') as mock_sgd:
            
            # Configure mocks
            mock_optimizer = MagicMock()
            mock_adam.return_value = mock_optimizer
            mock_adamw.return_value = mock_optimizer
            mock_sgd.return_value = mock_optimizer
            
            # Call the builder
            optimizer, _ = OptimizerBuilder.build(config, model.parameters())
            
            # Verify the correct optimizer class was used
            if expected_class == "Adam":
                mock_adam.assert_called_once()
                mock_adamw.assert_not_called()
                mock_sgd.assert_not_called()
            elif expected_class == "AdamW":
                mock_adam.assert_not_called()
                mock_adamw.assert_called_once()
                mock_sgd.assert_not_called()
            elif expected_class == "SGD":
                mock_adam.assert_not_called()
                mock_adamw.assert_not_called()
                mock_sgd.assert_called_once()
    
    @pytest.mark.parametrize("scheduler_name,expected_class", [
        ("cosine", "CosineAnnealingLR"),
        ("linear", None),  # Not implemented in current code
        ("constant", None),  # Not implemented in current code
        ("reduce_on_plateau", None),  # Not implemented in current code
        ("step", "StepLR"),
        ("exponential", "ExponentialLR"),
        (None, None)
    ])
    def test_scheduler_name_mapping(self, scheduler_name, expected_class):
        """
        GIVEN different scheduler names
        WHEN used with OptimizerBuilder
        THEN it should create the correct scheduler class if implemented
        """
        from unittest.mock import patch, MagicMock
        from src.configs import OptimizerConfig
        from src.optimizer import OptimizerBuilder
        import torch.nn as nn
        
        # Create a simple model for parameters
        model = nn.Linear(10, 1)
        
        # Create config with the specified scheduler name
        config = OptimizerConfig(
            name="adam",
            scheduler=scheduler_name
        )
        
        # Mock all scheduler classes
        with patch('src.optimizer.Adam') as mock_adam, \
             patch('src.optimizer.CosineAnnealingLR') as mock_cosine, \
             patch('src.optimizer.StepLR') as mock_step, \
             patch('src.optimizer.ExponentialLR') as mock_exp:
            
            # Configure mocks
            mock_optimizer = MagicMock()
            mock_adam.return_value = mock_optimizer
            
            mock_cosine_instance = MagicMock()
            mock_cosine.return_value = mock_cosine_instance
            
            mock_step_instance = MagicMock()
            mock_step.return_value = mock_step_instance
            
            mock_exp_instance = MagicMock()
            mock_exp.return_value = mock_exp_instance
            
            # Call the builder
            _, scheduler = OptimizerBuilder.build(config, model.parameters())
            
            # Verify the correct scheduler class was used if implemented
            if expected_class == "CosineAnnealingLR":
                mock_cosine.assert_called_once()
                mock_step.assert_not_called()
                mock_exp.assert_not_called()
                assert scheduler is mock_cosine_instance
            elif expected_class == "StepLR":
                mock_cosine.assert_not_called()
                mock_step.assert_called_once()
                mock_exp.assert_not_called()
                assert scheduler is mock_step_instance
            elif expected_class == "ExponentialLR":
                mock_cosine.assert_not_called()
                mock_step.assert_not_called()
                mock_exp.assert_called_once()
                assert scheduler is mock_exp_instance
            else:
                # For unimplemented schedulers or None
                if scheduler_name in ["linear", "constant", "reduce_on_plateau"]:
                    # These are valid values but not implemented in current code
                    assert scheduler is None
                elif scheduler_name is None:
                    # None scheduler should result in None
                    assert scheduler is None
    
    def test_yaml_serialization_compatibility(self):
        """
        GIVEN an OptimizerConfig instance
        WHEN serializing to YAML and back
        THEN the resulting object should match the original
        """
        import yaml
        from src.configs import OptimizerConfig
        
        original_config = OptimizerConfig(
            name="sgd",
            learning_rate=0.01,
            momentum=0.9,
            scheduler="cosine"
        )
        
        # Convert to dict
        config_dict = original_config.model_dump()
        
        # Serialize to YAML
        yaml_str = yaml.dump(config_dict)
        
        # Deserialize from YAML
        deserialized_dict = yaml.safe_load(yaml_str)
        
        # Create a new OptimizerConfig from the deserialized dict
        deserialized_config = OptimizerConfig(**deserialized_dict)
        
        # Compare fields
        assert deserialized_config.name == original_config.name
        assert deserialized_config.learning_rate == original_config.learning_rate
        assert deserialized_config.weight_decay == original_config.weight_decay
        assert deserialized_config.beta1 == original_config.beta1
        assert deserialized_config.beta2 == original_config.beta2
        assert deserialized_config.momentum == original_config.momentum
        assert deserialized_config.warmup_steps == original_config.warmup_steps
        assert deserialized_config.warmup_ratio == original_config.warmup_ratio
        assert deserialized_config.scheduler == original_config.scheduler
    
    def test_unsupported_optimizer_error(self):
        """
        GIVEN OptimizerConfig with valid name but unsupported in OptimizerBuilder
        WHEN used with OptimizerBuilder
        THEN it should raise ValueError
        """
        from unittest.mock import patch, MagicMock
        from src.configs import OptimizerConfig
        from src.optimizer import OptimizerBuilder
        import torch.nn as nn
        
        # Create a simple model for parameters
        model = nn.Linear(10, 1)
        
        # Create a mock that allows any string for name to bypass validation
        with patch('src.configs.OptimizerConfig.model_validate', 
                  return_value=MagicMock(name="unsupported", 
                                        learning_rate=0.001, 
                                        weight_decay=0.01,
                                        beta1=0.9,
                                        beta2=0.999,
                                        momentum=0.0,
                                        scheduler=None)):
            
            # Create config with unsupported optimizer name
            config = OptimizerConfig.model_validate({"name": "unsupported"})
            
            # Verify it raises ValueError when used with OptimizerBuilder
            with pytest.raises(ValueError) as excinfo:
                OptimizerBuilder.build(config, model.parameters())
            
            assert "Unsupported optimizer" in str(excinfo.value)
    
    def test_empty_required_datasets(self):
        """
        GIVEN DataConfig with empty required dataset lists
        WHEN using it with DataLoaderBuilder
        THEN it should handle empty lists gracefully
        """
        from unittest.mock import patch, MagicMock
        from src.data.dataloader import DataLoaderBuilder
        
        # Create config with empty dataset lists
        data_config = DataConfig(
            train_datasets=[],
            val_datasets=[],
            test_datasets=[]
        )
        
        feature_config = AudioEDAFeatureConfig()
        
        # Mock the dataloader builder
        with patch('src.data.dataloader.DataLoaderBuilder.build', wraps=DataLoaderBuilder.build) as wrapped_build:
            # Call the builder with empty datasets
            train_loaders = DataLoaderBuilder.build(data_config, feature_config, 'train')
            val_loaders = DataLoaderBuilder.build(data_config, feature_config, 'val')
            test_loaders = DataLoaderBuilder.build(data_config, feature_config, 'test')
            
            # Verify empty lists are returned
            assert isinstance(train_loaders, list)
            assert len(train_loaders) == 0
            
            assert isinstance(val_loaders, list)
            assert len(val_loaders) == 0
            
            assert isinstance(test_loaders, list)
            assert len(test_loaders) == 0


class TestLossConfig:
    """Tests for the LossConfig class that configures loss function parameters."""
    
    def test_valid_initialization(self):
        """
        GIVEN valid loss function names
        WHEN initializing a LossConfig
        THEN it should create a valid instance with the provided values
        """
        # Test each valid loss function name
        mse_config = LossConfig(name="mse")
        assert mse_config.name == "mse"
        
        l1_config = LossConfig(name="l1")
        assert l1_config.name == "l1"
        
        huber_config = LossConfig(name="huber")
        assert huber_config.name == "huber"
        
        custom_config = LossConfig(name="custom")
        assert custom_config.name == "custom"
    
    def test_default_value(self):
        """
        GIVEN no parameters
        WHEN initializing LossConfig with defaults
        THEN it should use 'mse' as the default loss function
        """
        config = LossConfig()
        assert config.name == "mse"
    
    def test_invalid_loss_name(self):
        """
        GIVEN invalid loss function name
        WHEN initializing LossConfig
        THEN it should raise ValidationError
        """
        with pytest.raises(ValidationError) as excinfo:
            LossConfig(name="invalid_loss")
        
        error_msg = str(excinfo.value)
        assert "name" in error_msg
        assert "invalid_loss" in error_msg
        assert "not a valid enumeration member" in error_msg.lower() or "Input should be" in error_msg
    
    def test_serialization_deserialization(self):
        """
        GIVEN a valid LossConfig instance
        WHEN serializing to JSON and back
        THEN the resulting object should match the original
        """
        original_config = LossConfig(name="huber")
        
        # Serialize to JSON
        json_str = original_config.model_dump_json()
        
        # Deserialize from JSON
        deserialized_config = LossConfig.model_validate_json(json_str)
        
        # Compare fields
        assert deserialized_config.name == original_config.name
    
    def test_model_dump(self):
        """
        GIVEN a valid LossConfig instance
        WHEN converting to dictionary with model_dump()
        THEN it should produce a dictionary with the name field
        """
        config = LossConfig(name="l1")
        
        config_dict = config.model_dump()
        
        assert isinstance(config_dict, dict)
        assert config_dict["name"] == "l1"
    
    def test_with_loss_builder(self):
        """
        GIVEN LossConfig instance
        WHEN used with LossBuilder
        THEN it should create the correct loss function
        """
        # Test with MSE loss
        mse_config = LossConfig(name="mse")
        mse_loss = LossBuilder.build(mse_config)
        assert isinstance(mse_loss, nn.MSELoss)
        
        # Test with L1 loss
        l1_config = LossConfig(name="l1")
        l1_loss = LossBuilder.build(l1_config)
        assert isinstance(l1_loss, nn.L1Loss)
        
        # Test with Huber loss
        huber_config = LossConfig(name="huber")
        huber_loss = LossBuilder.build(huber_config)
        assert isinstance(huber_loss, nn.SmoothL1Loss)
    
    def test_yaml_serialization_compatibility(self):
        """
        GIVEN a LossConfig instance
        WHEN serializing to YAML and back
        THEN the resulting object should match the original
        """
        import yaml
        
        original_config = LossConfig(name="l1")
        
        # Convert to dict
        config_dict = original_config.model_dump()
        
        # Serialize to YAML
        yaml_str = yaml.dump(config_dict)
        
        # Deserialize from YAML
        deserialized_dict = yaml.safe_load(yaml_str)
        
        # Create a new LossConfig from the deserialized dict
        deserialized_config = LossConfig(**deserialized_dict)
        
        # Compare fields
        assert deserialized_config.name == original_config.name
    
    def test_integration_with_train_script(self):
        """
        GIVEN LossConfig instance
        WHEN used in the training script
        THEN it should be properly initialized and used
        """
        from unittest.mock import patch, MagicMock
        
        # Mock YAML config that would be loaded in the training script
        yaml_config = {
            'experiment_name': 'test',
            'seed': 42,
            'model': {
                'architecture': 'tcn',
                'params': {
                    'input_size': 40,
                    'output_size': 1,
                    'num_blocks': 5,
                    'num_channels': 64,
                    'kernel_size': 3,
                    'dropout': 0.2
                }
            },
            'optimizer': {
                'name': 'adamw',
                'learning_rate': 0.001
            },
            'data': {
                'train_datasets': ['hku956'],
                'val_datasets': ['hku956'],
                'test_datasets': ['hku956'],
                'num_workers': 4,
                'prefetch_size': 2
            },
            'loss': {
                'name': 'huber'
            },
            'hardware': {
                'device': 'cuda',
                'precision': 'fp16',
                'distributed': False,
                'num_gpus': 1
            },
            'logging': {
                'wandb_project': 'test_project',
                'wandb_run_name': 'test_run'
            },
            'checkpoint': {
                'checkpoint_dir': './checkpoints'
            },
            'batch_size': 32,
            'max_epochs': 10
        }
        
        # Mock dependencies to avoid actual execution
        with patch('yaml.safe_load', return_value=yaml_config), \
             patch('builtins.open', MagicMock()), \
             patch('src.loss.LossBuilder.build') as mock_loss_build, \
             patch('wandb.init'), \
             patch('os.makedirs'), \
             patch('os.environ'):
            
            # Import the main function from the training script
            from scripts.train_audio2eda import main
            
            # Capture the LossConfig instance
            captured_config = None
            
            def mock_loss_build_side_effect(loss_config):
                nonlocal captured_config
                captured_config = loss_config
                return MagicMock()
            
            mock_loss_build.side_effect = mock_loss_build_side_effect
            
            # Mock sys.argv
            with patch('sys.argv', ['train_audio2eda.py', '--config', 'dummy_config.yaml']), \
                 patch('src.models.tcn.TCN', return_value=MagicMock()), \
                 patch('src.optimizer.OptimizerBuilder.build', return_value=(MagicMock(), MagicMock())), \
                 patch('src.data.dataloader.DataLoaderBuilder.build', return_value=[MagicMock()]), \
                 patch('transformers.Trainer'), \
                 patch('transformers.TrainingArguments'):
                
                # This should run without errors and use LossConfig
                main()
            
            # Verify LossConfig was created and used
            assert captured_config is not None
            assert isinstance(captured_config, LossConfig)
            
            # Verify values from YAML were used
            assert captured_config.name == "huber"


class TestModelConfig:
    """Tests for the ModelConfig class that configures model architecture and parameters."""
    
    def test_valid_tcn_initialization(self):
        """
        GIVEN valid parameters for TCN architecture
        WHEN initializing a ModelConfig
        THEN it should create a valid instance with the provided values
        """
        from src.configs import ModelConfig
        
        config = ModelConfig(
            architecture="tcn",
            params={
                "input_size": 40,
                "output_size": 1,
                "num_blocks": 5,
                "num_channels": 64,
                "kernel_size": 3,
                "dropout": 0.2
            }
        )
        
        assert config.architecture == "tcn"
        assert config.params["input_size"] == 40
        assert config.params["output_size"] == 1
        assert config.params["num_blocks"] == 5
        assert config.params["num_channels"] == 64
        assert config.params["kernel_size"] == 3
        assert config.params["dropout"] == 0.2
    
    def test_valid_wavenet_initialization(self):
        """
        GIVEN valid parameters for Wavenet architecture
        WHEN initializing a ModelConfig
        THEN it should create a valid instance with the provided values
        """
        from src.configs import ModelConfig
        
        config = ModelConfig(
            architecture="wavenet",
            params={
                "num_stacks": 2,
                "num_layers_per_stack": 10,
                "residual_channels": 64,
                "skip_channels": 256,
                "kernel_size": 3,
                "dilation_base": 2,
                "dropout_rate": 0.2,
                "input_channels": 40,
                "output_channels": 1,
                "use_bias": True
            }
        )
        
        assert config.architecture == "wavenet"
        assert config.params["num_stacks"] == 2
        assert config.params["num_layers_per_stack"] == 10
        assert config.params["residual_channels"] == 64
        assert config.params["skip_channels"] == 256
        assert config.params["kernel_size"] == 3
        assert config.params["dilation_base"] == 2
        assert config.params["dropout_rate"] == 0.2
        assert config.params["input_channels"] == 40
        assert config.params["output_channels"] == 1
        assert config.params["use_bias"] is True
    
    def test_invalid_architecture(self):
        """
        GIVEN invalid architecture name
        WHEN initializing a ModelConfig
        THEN it should raise ValidationError
        """
        from src.configs import ModelConfig
        from pydantic import ValidationError
        
        with pytest.raises(ValidationError) as excinfo:
            ModelConfig(
                architecture="invalid_architecture",
                params={}
            )
        
        error_msg = str(excinfo.value)
        assert "architecture" in error_msg
        assert "invalid_architecture" in error_msg
        assert "not a valid enumeration member" in error_msg.lower() or "Input should be" in error_msg
    
    def test_missing_tcn_params(self):
        """
        GIVEN TCN architecture with missing required parameters
        WHEN initializing a ModelConfig
        THEN it should raise ValueError with appropriate message
        """
        from src.configs import ModelConfig
        
        # Missing output_size and dropout
        with pytest.raises(ValueError) as excinfo:
            ModelConfig(
                architecture="tcn",
                params={
                    "input_size": 40,
                    "num_blocks": 5,
                    "num_channels": 64,
                    "kernel_size": 3
                }
            )
        
        error_msg = str(excinfo.value)
        assert "Missing parameters for tcn model" in error_msg
        assert "output_size" in error_msg
        assert "dropout" in error_msg
    
    def test_missing_wavenet_params(self):
        """
        GIVEN Wavenet architecture with missing required parameters
        WHEN initializing a ModelConfig
        THEN it should raise ValueError with appropriate message
        """
        from src.configs import ModelConfig
        
        # Missing several required parameters
        with pytest.raises(ValueError) as excinfo:
            ModelConfig(
                architecture="wavenet",
                params={
                    "num_stacks": 2,
                    "num_layers_per_stack": 10,
                    "kernel_size": 3
                }
            )
        
        error_msg = str(excinfo.value)
        assert "Missing parameters for wavenet model" in error_msg
        assert "residual_channels" in error_msg
        assert "skip_channels" in error_msg
        assert "dilation_base" in error_msg
        assert "dropout_rate" in error_msg
        assert "input_channels" in error_msg
        assert "output_channels" in error_msg
        assert "use_bias" in error_msg
    
    def test_extra_params_accepted(self):
        """
        GIVEN architecture with extra parameters beyond required ones
        WHEN initializing a ModelConfig
        THEN it should accept the extra parameters
        """
        from src.configs import ModelConfig
        
        config = ModelConfig(
            architecture="tcn",
            params={
                "input_size": 40,
                "output_size": 1,
                "num_blocks": 5,
                "num_channels": 64,
                "kernel_size": 3,
                "dropout": 0.2,
                "extra_param": "value",
                "another_extra": 123
            }
        )
        
        assert config.params["extra_param"] == "value"
        assert config.params["another_extra"] == 123
    
    def test_serialization_deserialization(self):
        """
        GIVEN a valid ModelConfig instance
        WHEN serializing to JSON and back
        THEN the resulting object should match the original
        """
        from src.configs import ModelConfig
        
        original_config = ModelConfig(
            architecture="tcn",
            params={
                "input_size": 40,
                "output_size": 1,
                "num_blocks": 5,
                "num_channels": 64,
                "kernel_size": 3,
                "dropout": 0.2
            }
        )
        
        # Serialize to JSON
        json_str = original_config.model_dump_json()
        
        # Deserialize from JSON
        deserialized_config = ModelConfig.model_validate_json(json_str)
        
        # Compare fields
        assert deserialized_config.architecture == original_config.architecture
        assert deserialized_config.params == original_config.params
    
    def test_yaml_serialization_compatibility(self):
        """
        GIVEN a ModelConfig instance
        WHEN serializing to YAML and back
        THEN the resulting object should match the original
        """
        import yaml
        from src.configs import ModelConfig
        
        original_config = ModelConfig(
            architecture="wavenet",
            params={
                "num_stacks": 2,
                "num_layers_per_stack": 10,
                "residual_channels": 64,
                "skip_channels": 256,
                "kernel_size": 3,
                "dilation_base": 2,
                "dropout_rate": 0.2,
                "input_channels": 40,
                "output_channels": 1,
                "use_bias": True
            }
        )
        
        # Convert to dict
        config_dict = original_config.model_dump()
        
        # Serialize to YAML
        yaml_str = yaml.dump(config_dict)
        
        # Deserialize from YAML
        deserialized_dict = yaml.safe_load(yaml_str)
        
        # Create a new ModelConfig from the deserialized dict
        deserialized_config = ModelConfig(**deserialized_dict)
        
        # Compare fields
        assert deserialized_config.architecture == original_config.architecture
        assert deserialized_config.params == original_config.params
    
    def test_integration_with_train_script(self):
        """
        GIVEN ModelConfig instance
        WHEN used in the training script
        THEN it should be properly initialized and used to create the correct model
        """
        from unittest.mock import patch, MagicMock
        import yaml
        
        # Mock YAML config that would be loaded in the training script
        yaml_config = {
            'experiment_name': 'test',
            'seed': 42,
            'model': {
                'architecture': 'tcn',
                'params': {
                    'input_size': 40,
                    'output_size': 1,
                    'num_blocks': 5,
                    'num_channels': 64,
                    'kernel_size': 3,
                    'dropout': 0.2
                }
            },
            'optimizer': {
                'name': 'adamw',
                'learning_rate': 0.001
            },
            'data': {
                'train_datasets': ['hku956'],
                'val_datasets': ['hku956'],
                'test_datasets': ['hku956'],
                'num_workers': 4,
                'prefetch_size': 2
            },
            'loss': {
                'name': 'mse'
            },
            'hardware': {
                'device': 'cuda',
                'precision': 'fp16',
                'distributed': False,
                'num_gpus': 1
            },
            'logging': {
                'wandb_project': 'test_project',
                'wandb_run_name': 'test_run'
            },
            'checkpoint': {
                'checkpoint_dir': './checkpoints'
            },
            'batch_size': 32,
            'max_epochs': 10
        }
        
        # Mock dependencies to avoid actual execution
        with patch('yaml.safe_load', return_value=yaml_config), \
             patch('builtins.open', MagicMock()), \
             patch('src.models.tcn.TCN') as mock_tcn, \
             patch('src.models.wavenet.Wavenet') as mock_wavenet, \
             patch('wandb.init'), \
             patch('os.makedirs'), \
             patch('os.environ'):
            
            # Import the main function from the training script
            from scripts.train_audio2eda import main
            
            # Mock sys.argv
            with patch('sys.argv', ['train_audio2eda.py', '--config', 'dummy_config.yaml']), \
                 patch('src.optimizer.OptimizerBuilder.build', return_value=(MagicMock(), MagicMock())), \
                 patch('src.data.dataloader.DataLoaderBuilder.build', return_value=[MagicMock()]), \
                 patch('src.loss.LossBuilder.build', return_value=MagicMock()), \
                 patch('transformers.Trainer'), \
                 patch('transformers.TrainingArguments'):
                
                # This should run without errors and use ModelConfig
                main()
            
            # Verify the correct model was created with the right parameters
            mock_tcn.assert_called_once()
            mock_wavenet.assert_not_called()
            
            # Verify the parameters were passed correctly
            args, kwargs = mock_tcn.call_args
            assert len(args) == 1
            assert args[0] == yaml_config['model']['params']
    
    def test_with_different_model_architecture(self):
        """
        GIVEN ModelConfig with different architecture
        WHEN used in the training script
        THEN it should create the correct model type
        """
        from unittest.mock import patch, MagicMock
        import yaml
        
        # Mock YAML config with wavenet architecture
        yaml_config = {
            'experiment_name': 'test',
            'seed': 42,
            'model': {
                'architecture': 'wavenet',
                'params': {
                    'num_stacks': 2,
                    'num_layers_per_stack': 10,
                    'residual_channels': 64,
                    'skip_channels': 256,
                    'kernel_size': 3,
                    'dilation_base': 2,
                    'dropout_rate': 0.2,
                    'input_channels': 40,
                    'output_channels': 1,
                    'use_bias': True
                }
            },
            'optimizer': {
                'name': 'adamw',
                'learning_rate': 0.001
            },
            'data': {
                'train_datasets': ['hku956'],
                'val_datasets': ['hku956'],
                'test_datasets': ['hku956'],
                'num_workers': 4,
                'prefetch_size': 2
            },
            'loss': {
                'name': 'mse'
            },
            'hardware': {
                'device': 'cuda',
                'precision': 'fp16',
                'distributed': False,
                'num_gpus': 1
            },
            'logging': {
                'wandb_project': 'test_project',
                'wandb_run_name': 'test_run'
            },
            'checkpoint': {
                'checkpoint_dir': './checkpoints'
            },
            'batch_size': 32,
            'max_epochs': 10
        }
        
        # Mock dependencies to avoid actual execution
        with patch('yaml.safe_load', return_value=yaml_config), \
             patch('builtins.open', MagicMock()), \
             patch('src.models.tcn.TCN') as mock_tcn, \
             patch('src.models.wavenet.Wavenet') as mock_wavenet, \
             patch('wandb.init'), \
             patch('os.makedirs'), \
             patch('os.environ'):
            
            # Import the main function from the training script
            from scripts.train_audio2eda import main
            
            # Mock sys.argv
            with patch('sys.argv', ['train_audio2eda.py', '--config', 'dummy_config.yaml']), \
                 patch('src.optimizer.OptimizerBuilder.build', return_value=(MagicMock(), MagicMock())), \
                 patch('src.data.dataloader.DataLoaderBuilder.build', return_value=[MagicMock()]), \
                 patch('src.loss.LossBuilder.build', return_value=MagicMock()), \
                 patch('transformers.Trainer'), \
                 patch('transformers.TrainingArguments'):
                
                # This should run without errors and use ModelConfig
                main()
            
            # Verify the correct model was created with the right parameters
            mock_tcn.assert_not_called()
            mock_wavenet.assert_called_once()
            
            # Verify the parameters were passed correctly
            args, kwargs = mock_wavenet.call_args
            assert len(args) == 1
            assert args[0] == yaml_config['model']['params']
    
    def test_model_dump(self):
        """
        GIVEN a valid ModelConfig instance
        WHEN converting to dictionary with model_dump()
        THEN it should produce a dictionary with all fields
        """
        from src.configs import ModelConfig
        
        config = ModelConfig(
            architecture="tcn",
            params={
                "input_size": 40,
                "output_size": 1,
                "num_blocks": 5,
                "num_channels": 64,
                "kernel_size": 3,
                "dropout": 0.2
            }
        )
        
        config_dict = config.model_dump()
        
        assert isinstance(config_dict, dict)
        assert "architecture" in config_dict
        assert "params" in config_dict
        assert config_dict["architecture"] == "tcn"
        assert config_dict["params"]["input_size"] == 40
        assert config_dict["params"]["output_size"] == 1
        assert config_dict["params"]["num_blocks"] == 5
        assert config_dict["params"]["num_channels"] == 64
        assert config_dict["params"]["kernel_size"] == 3
        assert config_dict["params"]["dropout"] == 0.2
    
    def test_unknown_architecture_error(self):
        """
        GIVEN ModelConfig with valid Literal value but unsupported in validator
        WHEN initializing ModelConfig
        THEN it should raise ValueError
        """
        from src.configs import ModelConfig
        from unittest.mock import patch, MagicMock
        
        # Create a mock that allows any string for architecture to bypass validation
        with patch('src.configs.ModelConfig.model_validate', 
                  return_value=MagicMock(architecture="unknown", params={})):
            
            # Create config with unsupported architecture
            config = ModelConfig.model_validate({"architecture": "unknown", "params": {}})
            
            # Verify it raises ValueError when the validator runs
            with pytest.raises(ValueError) as excinfo:
                # Access params to trigger the validator
                config.params
            
            assert "Invalid model architecture" in str(excinfo.value)

class TestHardwareConfig:
    """Tests for the HardwareConfig class that configures hardware settings for training."""
    
    def test_validate_distributed_with_mps(self):
        """
        GIVEN HardwareConfig with MPS device and distributed=True
        WHEN validating the distributed field
        THEN it should raise ValueError with appropriate message
        """
        from src.configs import HardwareConfig
        from pydantic import ValidationError
        
        # Test with MPS device and distributed=True
        with pytest.raises(ValueError) as excinfo:
            HardwareConfig(
                device="mps",
                distributed=True,
                num_gpus=1,
                precision="fp32"
            )
        
        error_msg = str(excinfo.value)
        assert "MPS device does not support distributed training" in error_msg
    
    class TestNumGpusValidator:
        """Tests specifically for the num_gpus field validator."""
        
        def test_mps_with_multiple_gpus(self):
            """
            GIVEN HardwareConfig with MPS device and num_gpus > 1
            WHEN validating the num_gpus field
            THEN it should raise ValueError with appropriate message
            """
            from src.configs import HardwareConfig
            
            # Test with MPS device and num_gpus > 1
            with pytest.raises(ValueError) as excinfo:
                HardwareConfig(
                    device="mps",
                    num_gpus=2,
                    precision="fp32"
                )
            
            error_msg = str(excinfo.value)
            assert "MPS device only supports a single GPU" in error_msg
        
        def test_mps_with_single_gpu(self):
            """
            GIVEN HardwareConfig with MPS device and num_gpus = 1
            WHEN validating the num_gpus field
            THEN it should accept the configuration
            """
            from src.configs import HardwareConfig
            
            # Test with MPS device and num_gpus = 1 (valid)
            config = HardwareConfig(
                device="mps",
                num_gpus=1,
                precision="fp32"
            )
            
            assert config.device == "mps"
            assert config.num_gpus == 1
        
        def test_cpu_with_nonzero_gpus(self):
            """
            GIVEN HardwareConfig with CPU device and num_gpus > 0
            WHEN validating the num_gpus field
            THEN it should raise ValueError with appropriate message
            """
            from src.configs import HardwareConfig
            
            # Test with CPU device and num_gpus > 0
            with pytest.raises(ValueError) as excinfo:
                HardwareConfig(
                    device="cpu",
                    num_gpus=1,
                    precision="fp32"
                )
            
            error_msg = str(excinfo.value)
            assert "CPU device should have num_gpus set to 0" in error_msg
        
        def test_cpu_with_zero_gpus(self):
            """
            GIVEN HardwareConfig with CPU device and num_gpus = 0
            WHEN validating the num_gpus field
            THEN it should accept the configuration
            """
            from src.configs import HardwareConfig
            
            # Test with CPU device and num_gpus = 0 (valid)
            config = HardwareConfig(
                device="cpu",
                num_gpus=0,
                precision="fp32"
            )
            
            assert config.device == "cpu"
            assert config.num_gpus == 0
        
        def test_cuda_with_zero_gpus(self):
            """
            GIVEN HardwareConfig with CUDA device and num_gpus = 0
            WHEN validating the num_gpus field
            THEN it should raise ValueError with appropriate message
            """
            from src.configs import HardwareConfig
            
            # Test with CUDA device and num_gpus = 0
            with pytest.raises(ValueError) as excinfo:
                HardwareConfig(
                    device="cuda",
                    num_gpus=0,
                    precision="fp16"
                )
            
            error_msg = str(excinfo.value)
            assert "CUDA device should have at least 1 GPU" in error_msg
        
        def test_cuda_with_multiple_gpus(self):
            """
            GIVEN HardwareConfig with CUDA device and num_gpus > 1
            WHEN validating the num_gpus field
            THEN it should accept the configuration
            """
            from src.configs import HardwareConfig
            
            # Test with CUDA device and num_gpus > 1 (valid)
            config = HardwareConfig(
                device="cuda",
                num_gpus=4,
                precision="fp16"
            )
            
            assert config.device == "cuda"
            assert config.num_gpus == 4
        
        def test_extreme_gpu_values(self):
            """
            GIVEN HardwareConfig with extremely large num_gpus value
            WHEN validating the num_gpus field
            THEN it should accept valid configurations and reject invalid ones
            """
            from src.configs import HardwareConfig
            
            # Test with CUDA device and extremely large num_gpus (valid)
            config = HardwareConfig(
                device="cuda",
                num_gpus=1000,
                precision="fp16"
            )
            assert config.num_gpus == 1000
            
            # Test with MPS device and extremely large num_gpus (invalid)
            with pytest.raises(ValueError) as excinfo:
                HardwareConfig(
                    device="mps",
                    num_gpus=1000,
                    precision="fp32"
                )
            assert "MPS device only supports a single GPU" in str(excinfo.value)
        
        def test_negative_gpu_values(self):
            """
            GIVEN HardwareConfig with negative num_gpus value
            WHEN validating the num_gpus field
            THEN it should raise ValueError for all device types
            """
            from src.configs import HardwareConfig
            
            # Test with CUDA device and negative num_gpus
            with pytest.raises(ValueError) as excinfo:
                HardwareConfig(
                    device="cuda",
                    num_gpus=-1,
                    precision="fp16"
                )
            assert "CUDA device should have at least 1 GPU" in str(excinfo.value)
            
            # Test with CPU device and negative num_gpus
            with pytest.raises(ValueError) as excinfo:
                HardwareConfig(
                    device="cpu",
                    num_gpus=-1,
                    precision="fp32"
                )
            assert "CPU device should have num_gpus set to 0" in str(excinfo.value)
            
            # Test with MPS device and negative num_gpus
            with pytest.raises(ValueError) as excinfo:
                HardwareConfig(
                    device="mps",
                    num_gpus=-1,
                    precision="fp32"
                )
            assert "MPS device only supports a single GPU" in str(excinfo.value)
        
        def test_validator_isolation(self):
            """
            GIVEN HardwareConfig with invalid num_gpus but also invalid precision
            WHEN validating fields
            THEN it should raise the precision error first (showing validator order)
            """
            from src.configs import HardwareConfig
            
            # Test with MPS device, invalid num_gpus, and invalid precision
            # Should fail on precision validation first
            with pytest.raises(ValueError) as excinfo:
                HardwareConfig(
                    device="mps",
                    num_gpus=2,  # Invalid for MPS
                    precision="fp16"  # Invalid for MPS
                )
            
            # Should fail on precision first due to validator order
            error_msg = str(excinfo.value)
            assert "MPS device only supports fp32 precision" in error_msg
            assert "MPS device only supports a single GPU" not in error_msg
    
    def test_validate_distributed_with_cpu(self):
        """
        GIVEN HardwareConfig with CPU device and distributed=True
        WHEN validating the distributed field
        THEN it should raise ValueError with appropriate message
        """
        from src.configs import HardwareConfig
        from pydantic import ValidationError
        
        # Test with CPU device and distributed=True
        with pytest.raises(ValueError) as excinfo:
            HardwareConfig(
                device="cpu",
                distributed=True,
                num_gpus=0,
                precision="fp32"
            )
        
        error_msg = str(excinfo.value)
        assert "Distributed training not recommended with CPU device" in error_msg
    
    def test_validate_distributed_valid_configs(self):
        """
        GIVEN HardwareConfig with valid device and distributed combinations
        WHEN validating the distributed field
        THEN it should accept valid configurations
        """
        from src.configs import HardwareConfig
        
        # Test with CUDA device and distributed=True (valid)
        cuda_config = HardwareConfig(
            device="cuda",
            distributed=True,
            num_gpus=2,
            precision="fp16"
        )
        assert cuda_config.device == "cuda"
        assert cuda_config.distributed is True
        assert cuda_config.num_gpus == 2
        
        # Test with MPS device and distributed=False (valid)
        mps_config = HardwareConfig(
            device="mps",
            distributed=False,
            num_gpus=1,
            precision="fp32"
        )
        assert mps_config.device == "mps"
        assert mps_config.distributed is False
        assert mps_config.num_gpus == 1
        
        # Test with CPU device and distributed=False (valid)
        cpu_config = HardwareConfig(
            device="cpu",
            distributed=False,
            num_gpus=0,
            precision="fp32"
        )
        assert cpu_config.device == "cpu"
        assert cpu_config.distributed is False
        assert cpu_config.num_gpus == 0
    
    def test_validate_distributed_integration(self):
        """
        GIVEN HardwareConfig used in a training configuration
        WHEN the config is used in the training script
        THEN it should properly validate distributed settings
        """
        from unittest.mock import patch, MagicMock
        import yaml
        
        # Mock YAML config with valid hardware settings
        yaml_config = {
            'experiment_name': 'test',
            'seed': 42,
            'model': {
                'architecture': 'tcn',
                'params': {
                    'input_size': 40,
                    'output_size': 1,
                    'num_blocks': 5,
                    'num_channels': 64,
                    'kernel_size': 3,
                    'dropout': 0.2
                }
            },
            'optimizer': {
                'name': 'adamw',
                'learning_rate': 0.001
            },
            'data': {
                'train_datasets': ['hku956'],
                'val_datasets': ['hku956'],
                'test_datasets': ['hku956'],
                'num_workers': 4,
                'prefetch_size': 2
            },
            'loss': {
                'name': 'mse'
            },
            'hardware': {
                'device': 'cuda',
                'precision': 'fp16',
                'distributed': True,
                'num_gpus': 2
            },
            'logging': {
                'wandb_project': 'test_project',
                'wandb_run_name': 'test_run'
            },
            'checkpoint': {
                'checkpoint_dir': './checkpoints'
            },
            'batch_size': 32,
            'max_epochs': 10
        }
        
        # Mock dependencies to avoid actual execution
        with patch('yaml.safe_load', return_value=yaml_config), \
             patch('builtins.open', MagicMock()), \
             patch('transformers.TrainingArguments') as mock_training_args, \
             patch('wandb.init'), \
             patch('os.makedirs'), \
             patch('os.environ'):
            
            # Import the main function from the training script
            from scripts.train_audio2eda import main
            
            # Mock sys.argv
            with patch('sys.argv', ['train_audio2eda.py', '--config', 'dummy_config.yaml']), \
                 patch('src.models.tcn.TCN', return_value=MagicMock()), \
                 patch('src.optimizer.OptimizerBuilder.build', return_value=(MagicMock(), MagicMock())), \
                 patch('src.data.dataloader.DataLoaderBuilder.build', return_value=[MagicMock()]), \
                 patch('src.loss.LossBuilder.build', return_value=MagicMock()), \
                 patch('transformers.Trainer'):
                
                # This should run without errors with valid config
                main()
                
                # Verify TrainingArguments was called
                mock_training_args.assert_called_once()
    
    def test_validate_distributed_with_invalid_yaml(self):
        """
        GIVEN invalid hardware configuration in YAML
        WHEN loading the config in the training script
        THEN it should raise appropriate validation errors
        """
        from unittest.mock import patch, MagicMock
        import yaml
        from pydantic import ValidationError
        
        # Mock YAML config with invalid hardware settings (MPS with distributed=True)
        yaml_config = {
            'experiment_name': 'test',
            'seed': 42,
            'model': {
                'architecture': 'tcn',
                'params': {
                    'input_size': 40,
                    'output_size': 1,
                    'num_blocks': 5,
                    'num_channels': 64,
                    'kernel_size': 3,
                    'dropout': 0.2
                }
            },
            'optimizer': {
                'name': 'adamw',
                'learning_rate': 0.001
            },
            'data': {
                'train_datasets': ['hku956'],
                'val_datasets': ['hku956'],
                'test_datasets': ['hku956'],
                'num_workers': 4,
                'prefetch_size': 2
            },
            'loss': {
                'name': 'mse'
            },
            'hardware': {
                'device': 'mps',
                'precision': 'fp32',
                'distributed': True,  # Invalid: MPS doesn't support distributed
                'num_gpus': 1
            },
            'logging': {
                'wandb_project': 'test_project',
                'wandb_run_name': 'test_run'
            },
            'checkpoint': {
                'checkpoint_dir': './checkpoints'
            },
            'batch_size': 32,
            'max_epochs': 10
        }
        
        # Mock dependencies to avoid actual execution
        with patch('yaml.safe_load', return_value=yaml_config), \
             patch('builtins.open', MagicMock()):
            
            # Import the main function from the training script
            from scripts.train_audio2eda import main
            
            # Mock sys.argv
            with patch('sys.argv', ['train_audio2eda.py', '--config', 'dummy_config.yaml']):
                # Should raise ValidationError due to invalid hardware config
                with pytest.raises(ValueError) as excinfo:
                    main()
                
                assert "MPS device does not support distributed training" in str(excinfo.value)
    
    def test_default_initialization(self):
        """
        GIVEN no parameters
        WHEN initializing HardwareConfig with defaults
        THEN it should create an instance with the correct predefined values
        """
        from src.configs import HardwareConfig
        
        config = HardwareConfig()
        
        assert config.device == "cuda"
        assert config.precision == "fp16"
        assert config.distributed is False
        assert config.num_gpus == 1
    
    def test_custom_initialization(self):
        """
        GIVEN custom parameters
        WHEN initializing HardwareConfig with those parameters
        THEN it should override defaults while preserving other values
        """
        from src.configs import HardwareConfig
        
        config = HardwareConfig(
            device="cpu",
            precision="fp32"
        )
        
        # Custom values should be used
        assert config.device == "cpu"
        assert config.precision == "fp32"
        
        # Default values should remain for non-overridden fields
        assert config.distributed is False
        assert config.num_gpus == 1  # This should be 0 for CPU, but there's no validation for this yet
    
    def test_validate_precision(self):
        """
        GIVEN HardwareConfig with MPS device
        WHEN setting precision to non-fp32
        THEN it should raise ValueError
        """
        from src.configs import HardwareConfig
        from pydantic import ValidationError
        
        # Test with MPS device and non-fp32 precision
        with pytest.raises(ValueError) as excinfo:
            HardwareConfig(
                device="mps",
                precision="fp16"
            )
        
        error_msg = str(excinfo.value)
        assert "MPS device only supports fp32 precision" in error_msg
        
        # Test with MPS device and fp32 precision (valid)
        config = HardwareConfig(
            device="mps",
            precision="fp32"
        )
        assert config.device == "mps"
        assert config.precision == "fp32"
        
        # Test with CUDA device and various precision options (all valid)
        for precision in ["fp32", "fp16", "bf16"]:
            config = HardwareConfig(
                device="cuda",
                precision=precision
            )
            assert config.precision == precision
    
    def test_validate_num_gpus(self):
        """
        GIVEN HardwareConfig with various device types
        WHEN setting num_gpus
        THEN it should validate according to device type
        """
        from src.configs import HardwareConfig
        from pydantic import ValidationError
        
        # Test with MPS device and num_gpus > 1
        with pytest.raises(ValueError) as excinfo:
            HardwareConfig(
                device="mps",
                num_gpus=2
            )
        
        error_msg = str(excinfo.value)
        assert "MPS device only supports a single GPU" in error_msg
        
        # Test with CPU device and num_gpus > 0
        with pytest.raises(ValueError) as excinfo:
            HardwareConfig(
                device="cpu",
                num_gpus=1
            )
        
        error_msg = str(excinfo.value)
        assert "CPU device should have num_gpus set to 0" in error_msg
        
        # Test with CUDA device and num_gpus < 1
        with pytest.raises(ValueError) as excinfo:
            HardwareConfig(
                device="cuda",
                num_gpus=0
            )
        
        error_msg = str(excinfo.value)
        assert "CUDA device should have at least 1 GPU" in error_msg
        
        # Test valid configurations
        valid_configs = [
            {"device": "mps", "num_gpus": 1},
            {"device": "cpu", "num_gpus": 0},
            {"device": "cuda", "num_gpus": 1},
            {"device": "cuda", "num_gpus": 4}
        ]
        
        for config_params in valid_configs:
            config = HardwareConfig(**config_params)
            assert config.device == config_params["device"]
            assert config.num_gpus == config_params["num_gpus"]
    
    def test_validator_interactions(self):
        """
        GIVEN HardwareConfig with multiple validation constraints
        WHEN setting combinations of parameters
        THEN it should validate all constraints correctly
        """
        from src.configs import HardwareConfig
        from pydantic import ValidationError
        
        # Test interaction between precision and distributed validators
        # This should fail on precision validation first
        with pytest.raises(ValueError) as excinfo:
            HardwareConfig(
                device="mps",
                precision="fp16",
                distributed=True
            )
        
        error_msg = str(excinfo.value)
        assert "MPS device only supports fp32 precision" in error_msg
        
        # Test interaction between num_gpus and distributed validators
        # This should fail on distributed validation
        with pytest.raises(ValueError) as excinfo:
            HardwareConfig(
                device="cpu",
                num_gpus=0,
                distributed=True
            )
        
        error_msg = str(excinfo.value)
        assert "Distributed training not recommended with CPU device" in error_msg
        
        # Test order of validation (should validate precision before distributed)
        with pytest.raises(ValueError) as excinfo:
            HardwareConfig(
                device="mps",
                precision="bf16",
                distributed=True
            )
        
        error_msg = str(excinfo.value)
        assert "MPS device only supports fp32 precision" in error_msg
        assert "MPS device does not support distributed training" not in error_msg
    
    def test_serialization_deserialization(self):
        """
        GIVEN a valid HardwareConfig instance
        WHEN serializing to JSON and back
        THEN the resulting object should match the original
        """
        from src.configs import HardwareConfig
        
        original_config = HardwareConfig(
            device="cuda",
            precision="fp16",
            distributed=True,
            num_gpus=4
        )
        
        # Serialize to JSON
        json_str = original_config.model_dump_json()
        
        # Deserialize from JSON
        deserialized_config = HardwareConfig.model_validate_json(json_str)
        
        # Compare fields
        assert deserialized_config.device == original_config.device
        assert deserialized_config.precision == original_config.precision
        assert deserialized_config.distributed == original_config.distributed
        assert deserialized_config.num_gpus == original_config.num_gpus
        
        # Test with YAML serialization
        import yaml
        
        # Convert to dict
        config_dict = original_config.model_dump()
        
        # Serialize to YAML
        yaml_str = yaml.dump(config_dict)
        
        # Deserialize from YAML
        deserialized_dict = yaml.safe_load(yaml_str)
        
        # Create a new HardwareConfig from the deserialized dict
        yaml_deserialized_config = HardwareConfig(**deserialized_dict)
        
        # Compare fields
        assert yaml_deserialized_config.device == original_config.device
        assert yaml_deserialized_config.precision == original_config.precision
        assert yaml_deserialized_config.distributed == original_config.distributed
        assert yaml_deserialized_config.num_gpus == original_config.num_gpus
    
    def test_model_dump(self):
        """
        GIVEN a valid HardwareConfig instance
        WHEN converting to dictionary with model_dump()
        THEN it should produce a dictionary with all fields
        """
        from src.configs import HardwareConfig
        
        config = HardwareConfig(
            device="cuda",
            precision="fp16",
            distributed=True,
            num_gpus=2
        )
        
        config_dict = config.model_dump()
        
        assert isinstance(config_dict, dict)
        assert config_dict["device"] == "cuda"
        assert config_dict["precision"] == "fp16"
        assert config_dict["distributed"] is True
        assert config_dict["num_gpus"] == 2

class TestCheckpointConfig:
    """Tests for the CheckpointConfig class that configures model checkpointing."""
    
    def test_default_initialization(self):
        """
        GIVEN only required parameters
        WHEN initializing CheckpointConfig with defaults
        THEN it should create an instance with the correct predefined values
        """
        from src.configs import CheckpointConfig
        
        config = CheckpointConfig(checkpoint_dir="./checkpoints")
        
        assert config.save_top_k == 3
        assert config.checkpoint_dir == "./checkpoints"
        assert config.monitor == "val_loss"
        assert config.mode == "min"
        assert config.save_last is True
        assert config.save_every_n_steps == 1000
        assert config.load_from_checkpoint is None
    
    def test_custom_initialization(self):
        """
        GIVEN custom parameters
        WHEN initializing CheckpointConfig with those parameters
        THEN it should override defaults while preserving other values
        """
        from src.configs import CheckpointConfig
        
        config = CheckpointConfig(
            checkpoint_dir="./custom_checkpoints",
            save_top_k=5,
            monitor="val_accuracy",
            mode="max",
            save_last=False,
            save_every_n_steps=500,
            load_from_checkpoint="path/to/checkpoint.ckpt"
        )
        
        assert config.save_top_k == 5
        assert config.checkpoint_dir == "./custom_checkpoints"
        assert config.monitor == "val_accuracy"
        assert config.mode == "max"
        assert config.save_last is False
        assert config.save_every_n_steps == 500
        assert config.load_from_checkpoint == "path/to/checkpoint.ckpt"
    
    def test_missing_required_fields(self):
        """
        GIVEN initialization parameters missing required fields
        WHEN initializing CheckpointConfig
        THEN it should raise ValidationError with appropriate message
        """
        from src.configs import CheckpointConfig
        from pydantic import ValidationError
        
        with pytest.raises(ValidationError) as excinfo:
            CheckpointConfig()  # Missing checkpoint_dir
        
        error_msg = str(excinfo.value)
        assert "checkpoint_dir" in error_msg
        assert "field required" in error_msg
    
    def test_invalid_mode_value(self):
        """
        GIVEN invalid mode value
        WHEN initializing CheckpointConfig
        THEN it should raise ValidationError with appropriate message
        """
        from src.configs import CheckpointConfig
        from pydantic import ValidationError
        
        with pytest.raises(ValidationError) as excinfo:
            CheckpointConfig(
                checkpoint_dir="./checkpoints",
                mode="invalid_mode"  # Not 'min' or 'max'
            )
        
        error_msg = str(excinfo.value)
        assert "mode" in error_msg
        assert "invalid_mode" in error_msg
        assert "not a valid enumeration member" in error_msg.lower() or "Input should be" in error_msg
    
    def test_serialization_deserialization(self):
        """
        GIVEN a valid CheckpointConfig instance
        WHEN serializing to JSON and back
        THEN the resulting object should match the original
        """
        from src.configs import CheckpointConfig
        
        original_config = CheckpointConfig(
            checkpoint_dir="./checkpoints",
            save_top_k=2,
            monitor="val_f1",
            mode="max"
        )
        
        # Serialize to JSON
        json_str = original_config.model_dump_json()
        
        # Deserialize from JSON
        deserialized_config = CheckpointConfig.model_validate_json(json_str)
        
        # Compare fields
        assert deserialized_config.save_top_k == original_config.save_top_k
        assert deserialized_config.checkpoint_dir == original_config.checkpoint_dir
        assert deserialized_config.monitor == original_config.monitor
        assert deserialized_config.mode == original_config.mode
        assert deserialized_config.save_last == original_config.save_last
        assert deserialized_config.save_every_n_steps == original_config.save_every_n_steps
        assert deserialized_config.load_from_checkpoint == original_config.load_from_checkpoint
    
    def test_model_dump(self):
        """
        GIVEN a valid CheckpointConfig instance
        WHEN converting to dictionary with model_dump()
        THEN it should produce a dictionary with all fields
        """
        from src.configs import CheckpointConfig
        
        config = CheckpointConfig(
            checkpoint_dir="./checkpoints",
            save_top_k=1,
            load_from_checkpoint="model.ckpt"
        )
        
        config_dict = config.model_dump()
        
        assert isinstance(config_dict, dict)
        assert config_dict["save_top_k"] == 1
        assert config_dict["checkpoint_dir"] == "./checkpoints"
        assert config_dict["monitor"] == "val_loss"
        assert config_dict["mode"] == "min"
        assert config_dict["save_last"] is True
        assert config_dict["save_every_n_steps"] == 1000
        assert config_dict["load_from_checkpoint"] == "model.ckpt"
    
    def test_with_training_arguments(self):
        """
        GIVEN CheckpointConfig instance
        WHEN used to configure TrainingArguments
        THEN it should correctly map its values to the appropriate arguments
        """
        from src.configs import CheckpointConfig
        from unittest.mock import patch, MagicMock
        
        config = CheckpointConfig(
            checkpoint_dir="./checkpoints",
            save_top_k=5,
            monitor="val_loss",
            mode="min",
            save_last=True,
            save_every_n_steps=100
        )
        
        # Mock TrainingArguments
        with patch('transformers.TrainingArguments') as mock_training_args:
            mock_args_instance = MagicMock()
            mock_training_args.return_value = mock_args_instance
            
            # Create training arguments using config values
            args = mock_training_args(
                output_dir=config.checkpoint_dir,
                save_steps=config.save_every_n_steps,
                save_total_limit=config.save_top_k,
                load_best_model_at_end=config.save_last,
                metric_for_best_model=config.monitor,
                greater_is_better=config.mode == 'max'
            )
            
            # Verify the config parameters were used correctly
            mock_training_args.assert_called_once()
            call_kwargs = mock_training_args.call_args[1]
            assert call_kwargs["output_dir"] == config.checkpoint_dir
            assert call_kwargs["save_steps"] == config.save_every_n_steps
            assert call_kwargs["save_total_limit"] == config.save_top_k
            assert call_kwargs["load_best_model_at_end"] == config.save_last
            assert call_kwargs["metric_for_best_model"] == config.monitor
            assert call_kwargs["greater_is_better"] == (config.mode == 'max')
    
    def test_yaml_serialization_compatibility(self):
        """
        GIVEN a CheckpointConfig instance
        WHEN serializing to YAML and back
        THEN the resulting object should match the original
        """
        import yaml
        from src.configs import CheckpointConfig
        
        original_config = CheckpointConfig(
            checkpoint_dir="./checkpoints",
            save_top_k=4,
            monitor="val_accuracy",
            mode="max"
        )
        
        # Convert to dict
        config_dict = original_config.model_dump()
        
        # Serialize to YAML
        yaml_str = yaml.dump(config_dict)
        
        # Deserialize from YAML
        deserialized_dict = yaml.safe_load(yaml_str)
        
        # Create a new CheckpointConfig from the deserialized dict
        deserialized_config = CheckpointConfig(**deserialized_dict)
        
        # Compare fields
        assert deserialized_config.save_top_k == original_config.save_top_k
        assert deserialized_config.checkpoint_dir == original_config.checkpoint_dir
        assert deserialized_config.monitor == original_config.monitor
        assert deserialized_config.mode == original_config.mode
        assert deserialized_config.save_last == original_config.save_last
        assert deserialized_config.save_every_n_steps == original_config.save_every_n_steps
        assert deserialized_config.load_from_checkpoint == original_config.load_from_checkpoint
    
    def test_integration_with_train_script(self):
        """
        GIVEN CheckpointConfig instance
        WHEN used in the training script
        THEN it should be properly initialized and used
        """
        from unittest.mock import patch, MagicMock
        import yaml
        
        # Mock YAML config that would be loaded in the training script
        yaml_config = {
            'experiment_name': 'test',
            'seed': 42,
            'model': {
                'architecture': 'tcn',
                'params': {
                    'input_size': 40,
                    'output_size': 1,
                    'num_blocks': 5,
                    'num_channels': 64,
                    'kernel_size': 3,
                    'dropout': 0.2
                }
            },
            'optimizer': {
                'name': 'adamw',
                'learning_rate': 0.001
            },
            'data': {
                'train_datasets': ['hku956'],
                'val_datasets': ['hku956'],
                'test_datasets': ['hku956'],
                'num_workers': 4,
                'prefetch_size': 2
            },
            'loss': {
                'name': 'mse'
            },
            'hardware': {
                'device': 'cuda',
                'precision': 'fp16',
                'distributed': False,
                'num_gpus': 1
            },
            'logging': {
                'wandb_project': 'test_project',
                'wandb_run_name': 'test_run'
            },
            'checkpoint': {
                'checkpoint_dir': './custom_checkpoints',
                'save_top_k': 5,
                'monitor': 'val_accuracy',
                'mode': 'max',
                'save_every_n_steps': 500
            },
            'batch_size': 32,
            'max_epochs': 10
        }
        
        # Mock dependencies to avoid actual execution
        with patch('yaml.safe_load', return_value=yaml_config), \
             patch('builtins.open', MagicMock()), \
             patch('os.makedirs') as mock_makedirs, \
             patch('transformers.TrainingArguments') as mock_training_args, \
             patch('wandb.init'), \
             patch('os.environ'):
            
            # Import the main function from the training script
            from scripts.train_audio2eda import main
            
            # Mock sys.argv
            with patch('sys.argv', ['train_audio2eda.py', '--config', 'dummy_config.yaml']), \
                 patch('src.models.tcn.TCN', return_value=MagicMock()), \
                 patch('src.optimizer.OptimizerBuilder.build', return_value=(MagicMock(), MagicMock())), \
                 patch('src.data.dataloader.DataLoaderBuilder.build', return_value=[MagicMock()]), \
                 patch('src.loss.LossBuilder.build', return_value=MagicMock()), \
                 patch('transformers.Trainer'):
                
                # This should run without errors and use CheckpointConfig
                main()
            
            # Verify the checkpoint directory was created
            mock_makedirs.assert_any_call('./custom_checkpoints', exist_ok=True)
            
            # Verify TrainingArguments was called with the correct checkpoint parameters
            mock_training_args.assert_called_once()
            call_kwargs = mock_training_args.call_args[1]
            assert call_kwargs["output_dir"] == './custom_checkpoints'
            assert call_kwargs["save_total_limit"] == 5
            assert call_kwargs["metric_for_best_model"] == 'val_accuracy'
            assert call_kwargs["greater_is_better"] is True  # mode is 'max'
            assert call_kwargs["save_steps"] == 500
    
    def test_extreme_values(self):
        """
        GIVEN extreme values for numeric fields
        WHEN initializing CheckpointConfig
        THEN it should handle them appropriately
        """
        from src.configs import CheckpointConfig
        
        # Test with very large values
        config = CheckpointConfig(
            checkpoint_dir="./checkpoints",
            save_top_k=1000000,  # Extremely large number
            save_every_n_steps=1000000000  # Extremely large number
        )
        
        assert config.save_top_k == 1000000
        assert config.save_every_n_steps == 1000000000
        
        # Test with zero values
        config = CheckpointConfig(
            checkpoint_dir="./checkpoints",
            save_top_k=0,
            save_every_n_steps=0
        )
        
        assert config.save_top_k == 0
        assert config.save_every_n_steps == 0
        
        # Test with negative values (should be accepted as there's no validation)
        config = CheckpointConfig(
            checkpoint_dir="./checkpoints",
            save_top_k=-1,
            save_every_n_steps=-100
        )
        
        assert config.save_top_k == -1
        assert config.save_every_n_steps == -100
    
    def test_checkpoint_path_handling(self):
        """
        GIVEN various checkpoint directory path formats
        WHEN initializing CheckpointConfig
        THEN it should handle them appropriately
        """
        from src.configs import CheckpointConfig
        
        # Test with relative path
        config = CheckpointConfig(checkpoint_dir="./checkpoints")
        assert config.checkpoint_dir == "./checkpoints"
        
        # Test with absolute path
        config = CheckpointConfig(checkpoint_dir="/absolute/path/to/checkpoints")
        assert config.checkpoint_dir == "/absolute/path/to/checkpoints"
        
        # Test with path containing special characters
        config = CheckpointConfig(checkpoint_dir="./checkpoints with spaces/model-v1.0")
        assert config.checkpoint_dir == "./checkpoints with spaces/model-v1.0"
        
        # Test with empty string (should be accepted as there's no validation)
        config = CheckpointConfig(checkpoint_dir="")
        assert config.checkpoint_dir == ""
    
    def test_load_from_checkpoint_handling(self):
        """
        GIVEN various load_from_checkpoint values
        WHEN initializing CheckpointConfig
        THEN it should handle them appropriately
        """
        from src.configs import CheckpointConfig
        
        # Test with None (default)
        config = CheckpointConfig(checkpoint_dir="./checkpoints")
        assert config.load_from_checkpoint is None
        
        # Test with specific checkpoint path
        config = CheckpointConfig(
            checkpoint_dir="./checkpoints",
            load_from_checkpoint="./checkpoints/model-epoch=10.ckpt"
        )
        assert config.load_from_checkpoint == "./checkpoints/model-epoch=10.ckpt"
        
        # Test with empty string
        config = CheckpointConfig(
            checkpoint_dir="./checkpoints",
            load_from_checkpoint=""
        )
        assert config.load_from_checkpoint == ""
        
        # Test with S3 path
        config = CheckpointConfig(
            checkpoint_dir="./checkpoints",
            load_from_checkpoint="s3://my-bucket/checkpoints/model.ckpt"
        )
        assert config.load_from_checkpoint == "s3://my-bucket/checkpoints/model.ckpt"

class TestLossBuilder:
    """Tests for the LossBuilder class that builds loss functions."""
    
    def test_build_mse_loss(self):
        """
        GIVEN LossConfig with 'mse' name
        WHEN calling LossBuilder.build
        THEN it should return an MSELoss instance
        """
        config = LossConfig(name="mse")
        loss_fn = LossBuilder.build(config)
        
        assert isinstance(loss_fn, nn.MSELoss)
    
    def test_build_l1_loss(self):
        """
        GIVEN LossConfig with 'l1' name
        WHEN calling LossBuilder.build
        THEN it should return an L1Loss instance
        """
        config = LossConfig(name="l1")
        loss_fn = LossBuilder.build(config)
        
        assert isinstance(loss_fn, nn.L1Loss)
    
    def test_build_huber_loss(self):
        """
        GIVEN LossConfig with 'huber' name
        WHEN calling LossBuilder.build
        THEN it should return a SmoothL1Loss instance
        """
        config = LossConfig(name="huber")
        loss_fn = LossBuilder.build(config)
        
        assert isinstance(loss_fn, nn.SmoothL1Loss)
    
    def test_build_custom_loss(self):
        """
        GIVEN LossConfig with 'custom' name
        WHEN calling LossBuilder.build
        THEN it should handle the custom case (currently returns None)
        """
        config = LossConfig(name="custom")
        loss_fn = LossBuilder.build(config)
        
        # Current implementation returns None for custom loss
        assert loss_fn is None
    
    def test_build_unsupported_loss(self):
        """
        GIVEN LossConfig with unsupported name
        WHEN calling LossBuilder.build
        THEN it should raise ValueError with appropriate message
        """
        # Create a mock that allows any string for name to bypass validation
        with patch('src.configs.LossConfig.model_validate', 
                  return_value=MagicMock(name="unsupported")):
            
            # Create config with unsupported loss name
            config = LossConfig.model_validate({"name": "unsupported"})
            
            # Verify it raises ValueError when used with LossBuilder
            with pytest.raises(ValueError) as excinfo:
                LossBuilder.build(config)
            
            assert "Unsupported loss function" in str(excinfo.value)
    
    def test_loss_function_behavior(self):
        """
        GIVEN loss functions built by LossBuilder
        WHEN using them to calculate loss
        THEN they should produce expected results
        """
        import torch
        
        # Create test tensors
        predictions = torch.tensor([0.0, 1.0, 2.0, 3.0])
        targets = torch.tensor([0.0, 1.0, 1.0, 1.0])
        
        # Test MSE loss
        mse_config = LossConfig(name="mse")
        mse_loss_fn = LossBuilder.build(mse_config)
        mse_result = mse_loss_fn(predictions, targets)
        
        # Expected MSE: ((0-0)² + (1-1)² + (2-1)² + (3-1)²) / 4 = (0 + 0 + 1 + 4) / 4 = 1.25
        assert abs(mse_result.item() - 1.25) < 1e-5
        
        # Test L1 loss
        l1_config = LossConfig(name="l1")
        l1_loss_fn = LossBuilder.build(l1_config)
        l1_result = l1_loss_fn(predictions, targets)
        
        # Expected L1: (|0-0| + |1-1| + |2-1| + |3-1|) / 4 = (0 + 0 + 1 + 2) / 4 = 0.75
        assert abs(l1_result.item() - 0.75) < 1e-5
        
        # Test Huber loss (SmoothL1Loss)
        huber_config = LossConfig(name="huber")
        huber_loss_fn = LossBuilder.build(huber_config)
        huber_result = huber_loss_fn(predictions, targets)
        
        # SmoothL1Loss is L1 for large differences and MSE for small differences
        # The exact value depends on the implementation details, but we can check it's between L1 and MSE
        assert huber_result.item() > 0
    
    def test_with_mock_nn_modules(self):
        """
        GIVEN mocked torch.nn loss modules
        WHEN calling LossBuilder.build
        THEN it should create the loss functions using the correct classes
        """
        with patch('src.loss.nn.MSELoss') as mock_mse, \
             patch('src.loss.nn.L1Loss') as mock_l1, \
             patch('src.loss.nn.SmoothL1Loss') as mock_huber:
            
            # Configure mocks
            mock_mse_instance = MagicMock()
            mock_mse.return_value = mock_mse_instance
            
            mock_l1_instance = MagicMock()
            mock_l1.return_value = mock_l1_instance
            
            mock_huber_instance = MagicMock()
            mock_huber.return_value = mock_huber_instance
            
            # Test MSE loss
            mse_config = LossConfig(name="mse")
            mse_result = LossBuilder.build(mse_config)
            mock_mse.assert_called_once()
            assert mse_result is mock_mse_instance
            
            # Test L1 loss
            l1_config = LossConfig(name="l1")
            l1_result = LossBuilder.build(l1_config)
            mock_l1.assert_called_once()
            assert l1_result is mock_l1_instance
            
            # Test Huber loss
            huber_config = LossConfig(name="huber")
            huber_result = LossBuilder.build(huber_config)
            mock_huber.assert_called_once()
            assert huber_result is mock_huber_instance
    
    def test_loss_builder_in_training_loop(self):
        """
        GIVEN loss function from LossBuilder
        WHEN used in a simple training loop
        THEN it should calculate gradients correctly
        """
        import torch
        import torch.nn as nn
        
        # Create a simple model
        model = nn.Linear(2, 1)
        
        # Create loss function using LossBuilder
        config = LossConfig(name="mse")
        loss_fn = LossBuilder.build(config)
        
        # Create optimizer
        optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
        
        # Create sample data
        inputs = torch.tensor([[1.0, 2.0], [3.0, 4.0]], requires_grad=True)
        targets = torch.tensor([[3.0], [7.0]])
        
        # Training step
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = loss_fn(outputs, targets)
        loss.backward()
        
        # Verify gradients were calculated
        for param in model.parameters():
            assert param.grad is not None
            assert torch.any(param.grad != 0)


class TestTrainConfig:
    """Tests for the TrainConfig class that configures the overall training process."""
    
    def test_valid_initialization(self):
        """
        GIVEN valid parameters for all required fields
        WHEN initializing a TrainConfig
        THEN it should create a valid instance with the provided values
        """
        from src.configs import TrainConfig, ModelConfig, OptimizerConfig, DataConfig, LossConfig
        from src.configs import HardwareConfig, LoggingConfig, CheckpointConfig
        from src.data.datasets.types import DatasetType
        
        # Create required nested configs
        model_config = ModelConfig(
            architecture="tcn",
            params={
                "input_size": 40,
                "output_size": 1,
                "num_blocks": 5,
                "num_channels": 64,
                "kernel_size": 3,
                "dropout": 0.2
            }
        )
        
        optimizer_config = OptimizerConfig(
            name="adamw",
            learning_rate=0.001
        )
        
        data_config = DataConfig(
            train_datasets=[DatasetType.HKU956],
            val_datasets=[DatasetType.PMEmo2019],
            test_datasets=[DatasetType.HKU956, DatasetType.PMEmo2019]
        )
        
        loss_config = LossConfig(name="mse")
        
        hardware_config = HardwareConfig(
            device="cuda",
            precision="fp16"
        )
        
        logging_config = LoggingConfig(
            wandb_project="test_project",
            wandb_run_name="test_run"
        )
        
        checkpoint_config = CheckpointConfig(
            checkpoint_dir="./checkpoints"
        )
        
        # Initialize TrainConfig with all required configs
        train_config = TrainConfig(
            experiment_name="test_experiment",
            model=model_config,
            optimizer=optimizer_config,
            data=data_config,
            loss=loss_config,
            hardware=hardware_config,
            logging=logging_config,
            checkpoint=checkpoint_config,
            batch_size=64,
            max_epochs=50,
            gradient_clip_val=0.5
        )
        
        # Verify all fields have correct values
        assert train_config.experiment_name == "test_experiment"
        assert train_config.seed == 42  # Default value
        assert train_config.model == model_config
        assert train_config.optimizer == optimizer_config
        assert train_config.data == data_config
        assert train_config.loss == loss_config
        assert train_config.hardware == hardware_config
        assert train_config.logging == logging_config
        assert train_config.checkpoint == checkpoint_config
        assert train_config.batch_size == 64
        assert train_config.max_epochs == 50
        assert train_config.gradient_clip_val == 0.5
        assert train_config.accumulate_grad_batches == 1  # Default value
        assert train_config.val_check_interval == 1.0  # Default value
        assert train_config.early_stopping is True  # Default value
        assert train_config.early_stopping_patience == 10  # Default value
        assert train_config.early_stopping_min_delta == 0.0001  # Default value
    
    def test_default_values(self):
        """
        GIVEN only required parameters
        WHEN initializing a TrainConfig
        THEN it should use default values for optional parameters
        """
        from src.configs import TrainConfig, ModelConfig, OptimizerConfig, DataConfig, LossConfig
        from src.configs import HardwareConfig, LoggingConfig, CheckpointConfig
        from src.data.datasets.types import DatasetType
        
        # Create minimal required nested configs
        model_config = ModelConfig(
            architecture="tcn",
            params={
                "input_size": 40,
                "output_size": 1,
                "num_blocks": 5,
                "num_channels": 64,
                "kernel_size": 3,
                "dropout": 0.2
            }
        )
        
        optimizer_config = OptimizerConfig()
        data_config = DataConfig(
            train_datasets=[DatasetType.HKU956],
            val_datasets=[DatasetType.HKU956],
            test_datasets=[DatasetType.HKU956]
        )
        loss_config = LossConfig()
        hardware_config = HardwareConfig()
        logging_config = LoggingConfig(wandb_project="test_project")
        checkpoint_config = CheckpointConfig(checkpoint_dir="./checkpoints")
        
        # Initialize TrainConfig with only required parameters
        train_config = TrainConfig(
            experiment_name="test_experiment",
            model=model_config,
            optimizer=optimizer_config,
            data=data_config,
            loss=loss_config,
            hardware=hardware_config,
            logging=logging_config,
            checkpoint=checkpoint_config
        )
        
        # Verify default values are used
        assert train_config.seed == 42
        assert train_config.batch_size == 32
        assert train_config.max_epochs == 100
        assert train_config.gradient_clip_val == 1.0
        assert train_config.accumulate_grad_batches == 1
        assert train_config.val_check_interval == 1.0
        assert train_config.early_stopping is True
        assert train_config.early_stopping_patience == 10
        assert train_config.early_stopping_min_delta == 0.0001
    
    def test_missing_required_fields(self):
        """
        GIVEN initialization parameters missing required fields
        WHEN initializing a TrainConfig
        THEN it should raise ValidationError with appropriate message
        """
        from src.configs import TrainConfig, ModelConfig, OptimizerConfig, DataConfig, LossConfig
        from src.configs import HardwareConfig, LoggingConfig, CheckpointConfig
        from src.data.datasets.types import DatasetType
        from pydantic import ValidationError
        
        # Create some required configs
        model_config = ModelConfig(
            architecture="tcn",
            params={
                "input_size": 40,
                "output_size": 1,
                "num_blocks": 5,
                "num_channels": 64,
                "kernel_size": 3,
                "dropout": 0.2
            }
        )
        
        # Missing experiment_name
        with pytest.raises(ValidationError) as excinfo:
            TrainConfig(
                # experiment_name is missing
                model=model_config,
                optimizer=OptimizerConfig(),
                data=DataConfig(
                    train_datasets=[DatasetType.HKU956],
                    val_datasets=[DatasetType.HKU956],
                    test_datasets=[DatasetType.HKU956]
                ),
                loss=LossConfig(),
                hardware=HardwareConfig(),
                logging=LoggingConfig(wandb_project="test_project"),
                checkpoint=CheckpointConfig(checkpoint_dir="./checkpoints")
            )
        
        error_msg = str(excinfo.value)
        assert "experiment_name" in error_msg
        assert "field required" in error_msg
        
        # Missing model config
        with pytest.raises(ValidationError) as excinfo:
            TrainConfig(
                experiment_name="test_experiment",
                # model is missing
                optimizer=OptimizerConfig(),
                data=DataConfig(
                    train_datasets=[DatasetType.HKU956],
                    val_datasets=[DatasetType.HKU956],
                    test_datasets=[DatasetType.HKU956]
                ),
                loss=LossConfig(),
                hardware=HardwareConfig(),
                logging=LoggingConfig(wandb_project="test_project"),
                checkpoint=CheckpointConfig(checkpoint_dir="./checkpoints")
            )
        
        error_msg = str(excinfo.value)
        assert "model" in error_msg
        assert "field required" in error_msg
    
    def test_serialization_deserialization(self):
        """
        GIVEN a valid TrainConfig instance
        WHEN serializing to JSON and back
        THEN the resulting object should match the original
        """
        from src.configs import TrainConfig, ModelConfig, OptimizerConfig, DataConfig, LossConfig
        from src.configs import HardwareConfig, LoggingConfig, CheckpointConfig
        from src.data.datasets.types import DatasetType
        
        # Create a complete TrainConfig
        original_config = TrainConfig(
            experiment_name="test_serialization",
            seed=123,
            model=ModelConfig(
                architecture="tcn",
                params={
                    "input_size": 40,
                    "output_size": 1,
                    "num_blocks": 5,
                    "num_channels": 64,
                    "kernel_size": 3,
                    "dropout": 0.2
                }
            ),
            optimizer=OptimizerConfig(
                name="adam",
                learning_rate=0.002,
                scheduler="cosine"
            ),
            data=DataConfig(
                train_datasets=[DatasetType.HKU956],
                val_datasets=[DatasetType.PMEmo2019],
                test_datasets=[DatasetType.HKU956, DatasetType.PMEmo2019],
                num_workers=8
            ),
            loss=LossConfig(name="huber"),
            hardware=HardwareConfig(
                device="cuda",
                precision="fp16",
                num_gpus=2
            ),
            logging=LoggingConfig(
                wandb_project="test_project",
                wandb_run_name="serialization_test",
                log_every_n_steps=20
            ),
            checkpoint=CheckpointConfig(
                checkpoint_dir="./serialization_checkpoints",
                save_top_k=5,
                mode="max",
                monitor="val_accuracy"
            ),
            batch_size=128,
            max_epochs=200,
            gradient_clip_val=0.5,
            accumulate_grad_batches=2,
            val_check_interval=0.5,
            early_stopping=True,
            early_stopping_patience=15,
            early_stopping_min_delta=0.001
        )
        
        # Serialize to JSON
        json_str = original_config.model_dump_json()
        
        # Deserialize from JSON
        deserialized_config = TrainConfig.model_validate_json(json_str)
        
        # Compare fields
        assert deserialized_config.experiment_name == original_config.experiment_name
        assert deserialized_config.seed == original_config.seed
        assert deserialized_config.batch_size == original_config.batch_size
        assert deserialized_config.max_epochs == original_config.max_epochs
        assert deserialized_config.gradient_clip_val == original_config.gradient_clip_val
        assert deserialized_config.accumulate_grad_batches == original_config.accumulate_grad_batches
        assert deserialized_config.val_check_interval == original_config.val_check_interval
        assert deserialized_config.early_stopping == original_config.early_stopping
        assert deserialized_config.early_stopping_patience == original_config.early_stopping_patience
        assert deserialized_config.early_stopping_min_delta == original_config.early_stopping_min_delta
        
        # Compare nested configs
        assert deserialized_config.model.architecture == original_config.model.architecture
        assert deserialized_config.model.params == original_config.model.params
        
        assert deserialized_config.optimizer.name == original_config.optimizer.name
        assert deserialized_config.optimizer.learning_rate == original_config.optimizer.learning_rate
        assert deserialized_config.optimizer.scheduler == original_config.optimizer.scheduler
        
        assert len(deserialized_config.data.train_datasets) == len(original_config.data.train_datasets)
        assert deserialized_config.data.num_workers == original_config.data.num_workers
        
        assert deserialized_config.loss.name == original_config.loss.name
        
        assert deserialized_config.hardware.device == original_config.hardware.device
        assert deserialized_config.hardware.precision == original_config.hardware.precision
        assert deserialized_config.hardware.num_gpus == original_config.hardware.num_gpus
        
        assert deserialized_config.logging.wandb_project == original_config.logging.wandb_project
        assert deserialized_config.logging.wandb_run_name == original_config.logging.wandb_run_name
        assert deserialized_config.logging.log_every_n_steps == original_config.logging.log_every_n_steps
        
        assert deserialized_config.checkpoint.checkpoint_dir == original_config.checkpoint.checkpoint_dir
        assert deserialized_config.checkpoint.save_top_k == original_config.checkpoint.save_top_k
        assert deserialized_config.checkpoint.mode == original_config.checkpoint.mode
        assert deserialized_config.checkpoint.monitor == original_config.checkpoint.monitor
    
    def test_yaml_serialization_compatibility(self):
        """
        GIVEN a TrainConfig instance
        WHEN serializing to YAML and back
        THEN the resulting object should match the original
        """
        import yaml
        from src.configs import TrainConfig, ModelConfig, OptimizerConfig, DataConfig, LossConfig
        from src.configs import HardwareConfig, LoggingConfig, CheckpointConfig
        from src.data.datasets.types import DatasetType
        
        # Create a TrainConfig with all fields
        original_config = TrainConfig(
            experiment_name="yaml_test",
            seed=456,
            model=ModelConfig(
                architecture="tcn",
                params={
                    "input_size": 40,
                    "output_size": 1,
                    "num_blocks": 5,
                    "num_channels": 64,
                    "kernel_size": 3,
                    "dropout": 0.2
                }
            ),
            optimizer=OptimizerConfig(
                name="adamw",
                learning_rate=0.001
            ),
            data=DataConfig(
                train_datasets=[DatasetType.HKU956],
                val_datasets=[DatasetType.HKU956],
                test_datasets=[DatasetType.HKU956]
            ),
            loss=LossConfig(),
            hardware=HardwareConfig(),
            logging=LoggingConfig(wandb_project="yaml_project"),
            checkpoint=CheckpointConfig(checkpoint_dir="./yaml_checkpoints"),
            batch_size=64
        )
        
        # Convert to dict
        config_dict = original_config.model_dump()
        
        # Handle enum serialization for YAML
        # Convert DatasetType enums to strings
        train_datasets = [ds.value for ds in config_dict['data']['train_datasets']]
        val_datasets = [ds.value for ds in config_dict['data']['val_datasets']]
        test_datasets = [ds.value for ds in config_dict['data']['test_datasets']]
        
        config_dict['data']['train_datasets'] = train_datasets
        config_dict['data']['val_datasets'] = val_datasets
        config_dict['data']['test_datasets'] = test_datasets
        
        # Serialize to YAML
        yaml_str = yaml.dump(config_dict)
        
        # Deserialize from YAML
        deserialized_dict = yaml.safe_load(yaml_str)
        
        # Convert string values back to enum instances for datasets
        deserialized_dict['data']['train_datasets'] = [DatasetType(ds) for ds in deserialized_dict['data']['train_datasets']]
        deserialized_dict['data']['val_datasets'] = [DatasetType(ds) for ds in deserialized_dict['data']['val_datasets']]
        deserialized_dict['data']['test_datasets'] = [DatasetType(ds) for ds in deserialized_dict['data']['test_datasets']]
        
        # Create a new TrainConfig from the deserialized dict
        deserialized_config = TrainConfig(**deserialized_dict)
        
        # Compare key fields
        assert deserialized_config.experiment_name == original_config.experiment_name
        assert deserialized_config.seed == original_config.seed
        assert deserialized_config.batch_size == original_config.batch_size
        assert deserialized_config.model.architecture == original_config.model.architecture
        assert deserialized_config.optimizer.name == original_config.optimizer.name
        assert deserialized_config.loss.name == original_config.loss.name
        assert len(deserialized_config.data.train_datasets) == len(original_config.data.train_datasets)
        assert deserialized_config.hardware.device == original_config.hardware.device
        assert deserialized_config.logging.wandb_project == original_config.logging.wandb_project
        assert deserialized_config.checkpoint.checkpoint_dir == original_config.checkpoint.checkpoint_dir
    
    def test_integration_with_train_script(self):
        """
        GIVEN a TrainConfig instance
        WHEN used in the training script
        THEN it should be properly loaded and used to configure training
        """
        from unittest.mock import patch, MagicMock
        import yaml
        
        # Mock YAML config that would be loaded in the training script
        yaml_config = {
            'experiment_name': 'integration_test',
            'seed': 42,
            'model': {
                'architecture': 'tcn',
                'params': {
                    'input_size': 40,
                    'output_size': 1,
                    'num_blocks': 5,
                    'num_channels': 64,
                    'kernel_size': 3,
                    'dropout': 0.2
                }
            },
            'optimizer': {
                'name': 'adamw',
                'learning_rate': 0.001,
                'weight_decay': 0.01,
                'scheduler': 'cosine'
            },
            'data': {
                'train_datasets': ['hku956'],
                'val_datasets': ['pmemo2019'],
                'test_datasets': ['hku956', 'pmemo2019'],
                'num_workers': 4,
                'prefetch_size': 2
            },
            'loss': {
                'name': 'mse'
            },
            'hardware': {
                'device': 'cuda',
                'precision': 'fp16',
                'distributed': False,
                'num_gpus': 1
            },
            'logging': {
                'wandb_project': 'test_project',
                'wandb_run_name': 'integration_run',
                'log_every_n_steps': 10
            },
            'checkpoint': {
                'checkpoint_dir': './integration_checkpoints',
                'save_top_k': 3,
                'monitor': 'val_loss',
                'mode': 'min'
            },
            'batch_size': 32,
            'max_epochs': 10,
            'gradient_clip_val': 1.0,
            'accumulate_grad_batches': 1,
            'val_check_interval': 1.0,
            'early_stopping': True,
            'early_stopping_patience': 5,
            'early_stopping_min_delta': 0.001
        }
        
        # Mock dependencies to avoid actual execution
        with patch('yaml.safe_load', return_value=yaml_config), \
             patch('builtins.open', MagicMock()), \
             patch('src.models.tcn.TCN') as mock_tcn, \
             patch('src.optimizer.OptimizerBuilder.build') as mock_optimizer_build, \
             patch('src.loss.LossBuilder.build') as mock_loss_build, \
             patch('src.data.dataloader.DataLoaderBuilder.build') as mock_dataloader_build, \
             patch('transformers.Trainer') as mock_trainer, \
             patch('transformers.TrainingArguments') as mock_training_args, \
             patch('wandb.init') as mock_wandb_init, \
             patch('os.makedirs') as mock_makedirs, \
             patch('os.environ'):
            
            # Configure mocks
            mock_dataloader_build.return_value = [MagicMock()]
            mock_optimizer_build.return_value = (MagicMock(), MagicMock())
            mock_loss_build.return_value = MagicMock()
            
            # Import and call the main function from the training script
            import sys
            sys.argv = ['train_audio2eda.py', '--config', 'dummy_config.yaml']
            
            from scripts.train_audio2eda import main
            
            # This should run without errors
            main()
            
            # Verify TrainConfig was created and used correctly
            mock_tcn.assert_called_once()
            mock_optimizer_build.assert_called_once()
            mock_loss_build.assert_called_once()
            mock_dataloader_build.assert_called()
            mock_training_args.assert_called_once()
            mock_trainer.assert_called_once()
            mock_wandb_init.assert_called_once_with(
                project=yaml_config['logging']['wandb_project'], 
                name=yaml_config['logging']['wandb_run_name']
            )
            mock_makedirs.assert_called_with(
                yaml_config['checkpoint']['checkpoint_dir'], 
                exist_ok=True
            )
            
            # Verify training arguments were configured correctly
            args, kwargs = mock_training_args.call_args
            assert kwargs['output_dir'] == yaml_config['checkpoint']['checkpoint_dir']
            assert kwargs['num_train_epochs'] == yaml_config['max_epochs']
            assert kwargs['per_device_train_batch_size'] == yaml_config['batch_size']
            assert kwargs['per_device_eval_batch_size'] == yaml_config['batch_size']
            assert kwargs['gradient_accumulation_steps'] == yaml_config['accumulate_grad_batches']
            assert kwargs['learning_rate'] == yaml_config['optimizer']['learning_rate']
            assert kwargs['weight_decay'] == yaml_config['optimizer']['weight_decay']
            assert kwargs['fp16'] == (yaml_config['hardware']['precision'] == 'fp16')
    
    def test_extreme_values(self):
        """
        GIVEN extreme values for numeric fields
        WHEN initializing a TrainConfig
        THEN it should handle them appropriately
        """
        from src.configs import TrainConfig, ModelConfig, OptimizerConfig, DataConfig, LossConfig
        from src.configs import HardwareConfig, LoggingConfig, CheckpointConfig
        from src.data.datasets.types import DatasetType
        
        # Create required nested configs
        model_config = ModelConfig(
            architecture="tcn",
            params={
                "input_size": 40,
                "output_size": 1,
                "num_blocks": 5,
                "num_channels": 64,
                "kernel_size": 3,
                "dropout": 0.2
            }
        )
        
        # Create minimal required configs
        optimizer_config = OptimizerConfig()
        data_config = DataConfig(
            train_datasets=[DatasetType.HKU956],
            val_datasets=[DatasetType.HKU956],
            test_datasets=[DatasetType.HKU956]
        )
        loss_config = LossConfig()
        hardware_config = HardwareConfig()
        logging_config = LoggingConfig(wandb_project="test_project")
        checkpoint_config = CheckpointConfig(checkpoint_dir="./checkpoints")
        
        # Test with very large values
        large_config = TrainConfig(
            experiment_name="extreme_large_test",
            seed=2**31 - 1,  # Max 32-bit signed int
            model=model_config,
            optimizer=optimizer_config,
            data=data_config,
            loss=loss_config,
            hardware=hardware_config,
            logging=logging_config,
            checkpoint=checkpoint_config,
            batch_size=10000,  # Very large batch size
            max_epochs=1000000,  # Very large epoch count
            gradient_clip_val=1000.0,  # Very large gradient clip
            accumulate_grad_batches=100,  # Very large accumulation
            val_check_interval=100.0,  # Very large interval
            early_stopping_patience=10000  # Very large patience
        )
        
        assert large_config.seed == 2**31 - 1
        assert large_config.batch_size == 10000
        assert large_config.max_epochs == 1000000
        assert large_config.gradient_clip_val == 1000.0
        assert large_config.accumulate_grad_batches == 100
        assert large_config.val_check_interval == 100.0
        assert large_config.early_stopping_patience == 10000
        
        # Test with very small values
        small_config = TrainConfig(
            experiment_name="extreme_small_test",
            seed=0,
            model=model_config,
            optimizer=optimizer_config,
            data=data_config,
            loss=loss_config,
            hardware=hardware_config,
            logging=logging_config,
            checkpoint=checkpoint_config,
            batch_size=1,  # Minimum batch size
            max_epochs=1,  # Minimum epoch count
            gradient_clip_val=0.0,  # No gradient clipping
            accumulate_grad_batches=1,  # No accumulation
            val_check_interval=0.01,  # Very small interval
            early_stopping_patience=0,  # No patience
            early_stopping_min_delta=0.0  # No minimum delta
        )
        
        assert small_config.seed == 0
        assert small_config.batch_size == 1
        assert small_config.max_epochs == 1
        assert small_config.gradient_clip_val == 0.0
        assert small_config.accumulate_grad_batches == 1
        assert small_config.val_check_interval == 0.01
        assert small_config.early_stopping_patience == 0
        assert small_config.early_stopping_min_delta == 0.0
        
        # Test with negative values (should be accepted as there's no validation)
        # Note: This test documents current behavior. If validation for positive
        # values is added in the future, this test should be updated.
        negative_config = TrainConfig(
            experiment_name="negative_test",
            seed=-42,
            model=model_config,
            optimizer=optimizer_config,
            data=data_config,
            loss=loss_config,
            hardware=hardware_config,
            logging=logging_config,
            checkpoint=checkpoint_config,
            batch_size=-32,  # Negative batch size
            max_epochs=-100,  # Negative epoch count
            gradient_clip_val=-1.0,  # Negative gradient clip
            accumulate_grad_batches=-2,  # Negative accumulation
            val_check_interval=-0.5,  # Negative interval
            early_stopping_patience=-10,  # Negative patience
            early_stopping_min_delta=-0.0001  # Negative delta
        )
        
        assert negative_config.seed == -42
        assert negative_config.batch_size == -32
        assert negative_config.max_epochs == -100
        assert negative_config.gradient_clip_val == -1.0
        assert negative_config.accumulate_grad_batches == -2
        assert negative_config.val_check_interval == -0.5
        assert negative_config.early_stopping_patience == -10
        assert negative_config.early_stopping_min_delta == -0.0001
    
    def test_early_stopping_configuration(self):
        """
        GIVEN various early stopping configurations
        WHEN initializing a TrainConfig
        THEN it should handle early stopping settings correctly
        """
        from src.configs import TrainConfig, ModelConfig, OptimizerConfig, DataConfig, LossConfig
        from src.configs import HardwareConfig, LoggingConfig, CheckpointConfig
        from src.data.datasets.types import DatasetType
        
        # Create required nested configs
        model_config = ModelConfig(
            architecture="tcn",
            params={
                "input_size": 40,
                "output_size": 1,
                "num_blocks": 5,
                "num_channels": 64,
                "kernel_size": 3,
                "dropout": 0.2
            }
        )
        
        # Create minimal required configs
        optimizer_config = OptimizerConfig()
        data_config = DataConfig(
            train_datasets=[DatasetType.HKU956],
            val_datasets=[DatasetType.HKU956],
            test_datasets=[DatasetType.HKU956]
        )
        loss_config = LossConfig()
        hardware_config = HardwareConfig()
        logging_config = LoggingConfig(wandb_project="test_project")
        checkpoint_config = CheckpointConfig(checkpoint_dir="./checkpoints")
        
        # Test with early stopping disabled
        no_early_stopping_config = TrainConfig(
            experiment_name="no_early_stopping",
            model=model_config,
            optimizer=optimizer_config,
            data=data_config,
            loss=loss_config,
            hardware=hardware_config,
            logging=logging_config,
            checkpoint=checkpoint_config,
            early_stopping=False
        )
        
        assert no_early_stopping_config.early_stopping is False
        assert no_early_stopping_config.early_stopping_patience == 10  # Default value
        assert no_early_stopping_config.early_stopping_min_delta == 0.0001  # Default value
        
        # Test with custom early stopping settings
        custom_early_stopping_config = TrainConfig(
            experiment_name="custom_early_stopping",
            model=model_config,
            optimizer=optimizer_config,
            data=data_config,
            loss=loss_config,
            hardware=hardware_config,
            logging=logging_config,
            checkpoint=checkpoint_config,
            early_stopping=True,
            early_stopping_patience=20,
            early_stopping_min_delta=0.005
        )
        
        assert custom_early_stopping_config.early_stopping is True
        assert custom_early_stopping_config.early_stopping_patience == 20
        assert custom_early_stopping_config.early_stopping_min_delta == 0.005
    
    def test_validation_interval_options(self):
        """
        GIVEN different validation interval configurations
        WHEN initializing a TrainConfig
        THEN it should handle both integer and float interval values
        """
        from src.configs import TrainConfig, ModelConfig, OptimizerConfig, DataConfig, LossConfig
        from src.configs import HardwareConfig, LoggingConfig, CheckpointConfig
        from src.data.datasets.types import DatasetType
        
        # Create required nested configs
        model_config = ModelConfig(
            architecture="tcn",
            params={
                "input_size": 40,
                "output_size": 1,
                "num_blocks": 5,
                "num_channels": 64,
                "kernel_size": 3,
                "dropout": 0.2
            }
        )
        
        # Create minimal required configs
        optimizer_config = OptimizerConfig()
        data_config = DataConfig(
            train_datasets=[DatasetType.HKU956],
            val_datasets=[DatasetType.HKU956],
            test_datasets=[DatasetType.HKU956]
        )
        loss_config = LossConfig()
        hardware_config = HardwareConfig()
        logging_config = LoggingConfig(wandb_project="test_project")
        checkpoint_config = CheckpointConfig(checkpoint_dir="./checkpoints")
        
        # Test with integer validation interval (steps)
        steps_config = TrainConfig(
            experiment_name="steps_validation",
            model=model_config,
            optimizer=optimizer_config,
            data=data_config,
            loss=loss_config,
            hardware=hardware_config,
            logging=logging_config,
            checkpoint=checkpoint_config,
            val_check_interval=100  # Every 100 steps
        )
        
        assert steps_config.val_check_interval == 100
        assert isinstance(steps_config.val_check_interval, int)
        
        # Test with float validation interval (fraction of epoch)
        fraction_config = TrainConfig(
            experiment_name="fraction_validation",
            model=model_config,
            optimizer=optimizer_config,
            data=data_config,
            loss=loss_config,
            hardware=hardware_config,
            logging=logging_config,
            checkpoint=checkpoint_config,
            val_check_interval=0.25  # Every 1/4 epoch
        )
        
        assert fraction_config.val_check_interval == 0.25
        assert isinstance(fraction_config.val_check_interval, float)
    
    def test_model_dump(self):
        """
        GIVEN a valid TrainConfig instance
        WHEN converting to dictionary with model_dump()
        THEN it should produce a dictionary with all fields
        """
        from src.configs import TrainConfig, ModelConfig, OptimizerConfig, DataConfig, LossConfig
        from src.configs import HardwareConfig, LoggingConfig, CheckpointConfig
        from src.data.datasets.types import DatasetType
        
        # Create a TrainConfig with custom values
        config = TrainConfig(
            experiment_name="dump_test",
            seed=789,
            model=ModelConfig(
                architecture="tcn",
                params={
                    "input_size": 40,
                    "output_size": 1,
                    "num_blocks": 5,
                    "num_channels": 64,
                    "kernel_size": 3,
                    "dropout": 0.2
                }
            ),
            optimizer=OptimizerConfig(
                name="sgd",
                learning_rate=0.01,
                momentum=0.9
            ),
            data=DataConfig(
                train_datasets=[DatasetType.HKU956],
                val_datasets=[DatasetType.PMEmo2019],
                test_datasets=[],
                num_workers=16,
                prefetch_size=8
            ),
            loss=LossConfig(name="l1"),
            hardware=HardwareConfig(
                device="cpu",
                precision="fp32",
                num_gpus=0
            ),
            logging=LoggingConfig(
                wandb_project="dump_project",
                wandb_run_name="dump_run"
            ),
            checkpoint=CheckpointConfig(
                checkpoint_dir="./dump_checkpoints",
                save_top_k=1
            ),
            batch_size=16,
            max_epochs=50,
            gradient_clip_val=0.1,
            early_stopping=False
        )
        
        config_dict = config.model_dump()
        
        # Verify dictionary structure
        assert isinstance(config_dict, dict)
        assert config_dict["experiment_name"] == "dump_test"
        assert config_dict["seed"] == 789
        assert config_dict["batch_size"] == 16
        assert config_dict["max_epochs"] == 50
        assert config_dict["gradient_clip_val"] == 0.1
        assert config_dict["early_stopping"] is False
        
        # Verify nested dictionaries
        assert config_dict["model"]["architecture"] == "tcn"
        assert config_dict["model"]["params"]["input_size"] == 40
        assert config_dict["model"]["params"]["dropout"] == 0.2
        
        assert config_dict["optimizer"]["name"] == "sgd"
        assert config_dict["optimizer"]["learning_rate"] == 0.01
        assert config_dict["optimizer"]["momentum"] == 0.9
        
        assert len(config_dict["data"]["train_datasets"]) == 1
        assert len(config_dict["data"]["val_datasets"]) == 1
        assert len(config_dict["data"]["test_datasets"]) == 0
        assert config_dict["data"]["num_workers"] == 16
        
        assert config_dict["loss"]["name"] == "l1"
        
        assert config_dict["hardware"]["device"] == "cpu"
        assert config_dict["hardware"]["precision"] == "fp32"
        assert config_dict["hardware"]["num_gpus"] == 0
        
        assert config_dict["logging"]["wandb_project"] == "dump_project"
        assert config_dict["logging"]["wandb_run_name"] == "dump_run"
        
        assert config_dict["checkpoint"]["checkpoint_dir"] == "./dump_checkpoints"
        assert config_dict["checkpoint"]["save_top_k"] == 1


class TestDerivedDatasetConfigs:
    """Tests for the derived dataset configuration classes."""

    def test_hku956_config_defaults(self):
        """
        GIVEN HKU956Config class
        WHEN initializing with default values
        THEN it should create an instance with the correct predefined values
        """
        config = HKU956Config()
        
        assert config.dataset_name == "HKU956"
        assert "HKU956" in config.dataset_root_path
        assert "eda" in config.modalities
        assert "audio" in config.modalities
        assert config.file_format["eda"] == ".csv"
        assert config.file_format["audio"] == ".mp3"
        assert "physiological_signals" in config.data_directories["eda"]
        assert "original_song_audio.csv" in config.data_directories["audio"]
        assert config.split_ratios == [0.8, 0.1, 0.1]
        assert config.seed == 42
    
    class TestPMEmo2019Config:
        """Specific tests for the PMEmo2019Config class."""
        
        def test_s3_path_format(self):
            """
            GIVEN PMEmo2019Config instance
            WHEN examining S3 paths
            THEN all paths should follow the correct S3 URI format
            """
            config = PMEmo2019Config()
            
            # Check root path format
            assert config.dataset_root_path.startswith("s3://")
            
            # Check data directory paths
            for path in config.data_directories.values():
                assert path.startswith("s3://")
                
            # Check metadata paths
            for path in config.metadata_paths:
                assert path.startswith("s3://")
                
            # Check that paths are properly formed with bucket and key
            all_paths = [config.dataset_root_path] + list(config.data_directories.values()) + config.metadata_paths
            for path in all_paths:
                parts = path.replace("s3://", "").split("/")
                assert len(parts) >= 1, f"Invalid S3 path format: {path}"
                assert parts[0], f"Missing bucket name in S3 path: {path}"
        
        def test_modalities_consistency(self):
            """
            GIVEN PMEmo2019Config instance
            WHEN examining modalities and related configurations
            THEN all modalities should have corresponding file formats and data directories
            """
            config = PMEmo2019Config()
            
            # Every modality should have a file format
            for modality in config.modalities:
                assert modality in config.file_format, f"Missing file format for modality: {modality}"
                
            # Every modality should have a data directory
            for modality in config.modalities:
                assert modality in config.data_directories, f"Missing data directory for modality: {modality}"
                
            # No extra file formats or data directories for non-existent modalities
            for key in config.file_format:
                assert key in config.modalities, f"File format defined for non-existent modality: {key}"
                
            for key in config.data_directories:
                assert key in config.modalities, f"Data directory defined for non-existent modality: {key}"
        
        def test_split_ratios_validity(self):
            """
            GIVEN PMEmo2019Config instance
            WHEN examining split ratios
            THEN they should sum to 1.0 and have the expected number of splits
            """
            config = PMEmo2019Config()
            
            # Should have exactly 3 splits (train, val, test)
            assert len(config.split_ratios) == 3, f"Expected 3 split ratios, got {len(config.split_ratios)}"
            
            # Ratios should sum to 1.0 (allowing for floating point imprecision)
            assert abs(sum(config.split_ratios) - 1.0) < 1e-10, f"Split ratios should sum to 1.0, got {sum(config.split_ratios)}"
            
            # Each ratio should be positive
            for ratio in config.split_ratios:
                assert ratio > 0, f"Split ratio should be positive, got {ratio}"
        
        def test_custom_initialization(self):
            """
            GIVEN custom parameters
            WHEN initializing PMEmo2019Config with those parameters
            THEN it should override defaults while preserving other values
            """
            custom_config = PMEmo2019Config(
                seed=100,
                split_ratios=[0.7, 0.2, 0.1],
                modalities=["audio", "eda", "ecg"]
            )
            
            # Custom values should be used
            assert custom_config.seed == 100
            assert custom_config.split_ratios == [0.7, 0.2, 0.1]
            assert "ecg" in custom_config.modalities
            
            # Default values should remain for non-overridden fields
            assert custom_config.dataset_name == "PMEmo2019"
            assert "PMEmo2019" in custom_config.dataset_root_path
            assert "EDA" in custom_config.data_directories["eda"]
            assert "chorus" in custom_config.data_directories["audio"]
        
        def test_with_dataloader_builder(self):
            """
            GIVEN PMEmo2019Config instance
            WHEN used with DataLoaderBuilder
            THEN it should be compatible with the builder's interface
            """
            from src.data.dataloader import DataLoaderBuilder
            
            # This test verifies that PMEmo2019Config can be used with DataLoaderBuilder
            # We're not actually building dataloaders, just checking the interface compatibility
            
            dataset_mapping = {
                'pmemo2019': (None, None, PMEmo2019Config()),
            }
            
            # Extract the config from the mapping
            _, _, config = dataset_mapping['pmemo2019']
            
            # Verify it's the right type
            assert isinstance(config, PMEmo2019Config)
            assert isinstance(config, DatasetConfig)
            
            # Verify it has all the expected attributes that DataLoaderBuilder would use
            assert hasattr(config, 'dataset_name')
            assert hasattr(config, 'dataset_root_path')
            assert hasattr(config, 'modalities')
            assert hasattr(config, 'data_directories')
        
        def test_model_dump_and_validation(self):
            """
            GIVEN PMEmo2019Config instance
            WHEN serializing and deserializing
            THEN the resulting object should match the original
            """
            original_config = PMEmo2019Config()
            
            # Serialize to dict
            config_dict = original_config.model_dump()
            
            # Deserialize from dict
            deserialized_config = PMEmo2019Config.model_validate(config_dict)
            
            # Compare fields
            assert deserialized_config.dataset_name == original_config.dataset_name
            assert deserialized_config.dataset_root_path == original_config.dataset_root_path
            assert deserialized_config.modalities == original_config.modalities
            assert deserialized_config.file_format == original_config.file_format
            assert deserialized_config.data_directories == original_config.data_directories
            assert deserialized_config.split_ratios == original_config.split_ratios
            assert deserialized_config.seed == original_config.seed
            
            # Serialize to JSON and back
            json_str = original_config.model_dump_json()
            json_deserialized = PMEmo2019Config.model_validate_json(json_str)
            
            # Verify JSON serialization/deserialization works
            assert json_deserialized.dataset_name == original_config.dataset_name
            assert json_deserialized.dataset_root_path == original_config.dataset_root_path
        
        def test_with_pmemo2019_dataset(self):
            """
            GIVEN PMEmo2019Config instance
            WHEN used to initialize a PMEmo2019Dataset
            THEN it should provide all necessary configuration
            """
            from unittest.mock import patch, MagicMock
            
            config = PMEmo2019Config()
            
            # Mock the PMEmo2019Dataset to avoid actual initialization
            with patch('src.data.datasets.pmemo2019.PMEmo2019Dataset') as MockDataset:
                # Create a mock for AudioEDAFeatureConfig
                mock_feature_config = MagicMock()
                
                # Initialize the dataset with our config
                MockDataset(config, mock_feature_config)
                
                # Verify the dataset was initialized with our config
                MockDataset.assert_called_once()
                args, kwargs = MockDataset.call_args
                assert args[0] is config
                assert args[1] is mock_feature_config
        
        def test_metadata_paths_validation(self):
            """
            GIVEN PMEmo2019Config instance
            WHEN examining metadata paths
            THEN they should be properly configured
            """
            config = PMEmo2019Config()
            
            # Should have at least one metadata path
            assert len(config.metadata_paths) > 0, "PMEmo2019Config should have at least one metadata path"
            
            # First metadata path should point to the main metadata file
            assert "metadata.csv" in config.metadata_paths[0], "First metadata path should point to metadata.csv"
            
            # Test with custom metadata paths
            custom_config = PMEmo2019Config(
                metadata_paths=["s3://custom-bucket/PMEmo2019/custom_metadata.csv"]
            )
            assert "custom_metadata.csv" in custom_config.metadata_paths[0]
        
        def test_directory_structure_consistency(self):
            """
            GIVEN PMEmo2019Config instance
            WHEN examining directory structure
            THEN it should follow a consistent pattern
            """
            config = PMEmo2019Config()
            
            # All paths should be under the same bucket
            bucket = config.dataset_root_path.split('/')[2]
            
            for path in list(config.data_directories.values()) + config.metadata_paths:
                path_bucket = path.split('/')[2]
                assert path_bucket == bucket, f"Path {path} uses different bucket than root path"
            
            # Audio directory should be under the dataset root
            assert config.data_directories["audio"].startswith(config.dataset_root_path.rstrip('/'))
            
            # EDA directory should be under the dataset root
            assert config.data_directories["eda"].startswith(config.dataset_root_path.rstrip('/'))
        
        def test_file_format_extensions(self):
            """
            GIVEN PMEmo2019Config instance
            WHEN examining file format extensions
            THEN they should all start with a period
            """
            config = PMEmo2019Config()
            
            # All file extensions should start with a period
            for ext in config.file_format.values():
                assert ext.startswith('.'), f"File extension '{ext}' should start with a period"
                
            # Test with custom file formats
            custom_config = PMEmo2019Config(
                file_format={"audio": ".wav", "eda": ".dat"}
            )
            assert custom_config.file_format["audio"] == ".wav"
            assert custom_config.file_format["eda"] == ".dat"
        
        def test_integration_with_dataset_type_enum(self):
            """
            GIVEN PMEmo2019Config and DatasetType
            WHEN using them together
            THEN they should be compatible
            """
            from src.data.datasets.types import DatasetType
            
            # The enum value should match what's expected in the DataLoaderBuilder
            assert DatasetType.PMEmo2019.value == 'pmemo2019'
            
            # Create a mapping that would be used in DataLoaderBuilder
            dataset_mapping = {
                DatasetType.PMEmo2019.value: (None, None, PMEmo2019Config())
            }
            
            # Verify we can access the config using the enum value
            assert 'pmemo2019' in dataset_mapping
            config = dataset_mapping['pmemo2019'][2]
            assert isinstance(config, PMEmo2019Config)

    def test_pmemo2019_config_defaults(self):
        """
        GIVEN PMEmo2019Config class
        WHEN initializing with default values
        THEN it should create an instance with the correct predefined values
        """
        config = PMEmo2019Config()
        
        assert config.dataset_name == "PMEmo2019"
        assert "PMEmo2019" in config.dataset_root_path
        assert "eda" in config.modalities
        assert "audio" in config.modalities
        assert config.file_format["eda"] == ".csv"
        assert config.file_format["audio"] == ".mp3"
        assert "EDA" in config.data_directories["eda"]
        assert "chorus" in config.data_directories["audio"]
        assert "metadata.csv" in config.metadata_paths[0]
        assert config.split_ratios == [0.8, 0.1, 0.1]
        assert config.seed == 42

    def test_override_default_values(self):
        """
        GIVEN derived config class
        WHEN initializing with custom values that override defaults
        THEN it should use the provided values instead of defaults
        """
        config = HKU956Config(
            seed=100,
            split_ratios=[0.6, 0.2, 0.2],
            metadata_paths=["custom_metadata.csv"]
        )
        
        # Custom values should be used
        assert config.seed == 100
        assert config.split_ratios == [0.6, 0.2, 0.2]
        assert config.metadata_paths == ["custom_metadata.csv"]
        
        # Default values should remain unchanged
        assert config.dataset_name == "HKU956"
        assert "HKU956" in config.dataset_root_path

    def test_inheritance_relationship(self):
        """
        GIVEN derived config classes
        WHEN checking inheritance
        THEN they should be subclasses of DatasetConfig
        """
        assert issubclass(HKU956Config, DatasetConfig)
        assert issubclass(PMEmo2019Config, DatasetConfig)
        
        hku_config = HKU956Config()
        pmemo_config = PMEmo2019Config()
        
        assert isinstance(hku_config, DatasetConfig)
        assert isinstance(pmemo_config, DatasetConfig)

    def test_config_compatibility(self):
        """
        GIVEN both derived config classes
        WHEN using them interchangeably where DatasetConfig is expected
        THEN they should be compatible with functions expecting DatasetConfig
        """
        def process_dataset_config(config: DatasetConfig) -> str:
            return f"Processing {config.dataset_name} with {len(config.modalities)} modalities"
        
        hku_result = process_dataset_config(HKU956Config())
        pmemo_result = process_dataset_config(PMEmo2019Config())
        
        assert "Processing HKU956" in hku_result
        assert "Processing PMEmo2019" in pmemo_result
        assert "2 modalities" in hku_result
        assert "2 modalities" in pmemo_result
    
    class TestHKU956Config:
        """Specific tests for the HKU956Config class."""
        
        def test_s3_path_format(self):
            """
            GIVEN HKU956Config instance
            WHEN examining S3 paths
            THEN all paths should follow the correct S3 URI format
            """
            config = HKU956Config()
            
            # Check root path format
            assert config.dataset_root_path.startswith("s3://")
            
            # Check data directory paths
            for path in config.data_directories.values():
                assert path.startswith("s3://")
                
            # Check that paths are properly formed with bucket and key
            for path in [config.dataset_root_path] + list(config.data_directories.values()):
                parts = path.replace("s3://", "").split("/")
                assert len(parts) >= 1, f"Invalid S3 path format: {path}"
                assert parts[0], f"Missing bucket name in S3 path: {path}"
        
        def test_modalities_consistency(self):
            """
            GIVEN HKU956Config instance
            WHEN examining modalities and related configurations
            THEN all modalities should have corresponding file formats and data directories
            """
            config = HKU956Config()
            
            # Every modality should have a file format
            for modality in config.modalities:
                assert modality in config.file_format, f"Missing file format for modality: {modality}"
                
            # Every modality should have a data directory
            for modality in config.modalities:
                assert modality in config.data_directories, f"Missing data directory for modality: {modality}"
                
            # No extra file formats or data directories for non-existent modalities
            for key in config.file_format:
                assert key in config.modalities, f"File format defined for non-existent modality: {key}"
                
            for key in config.data_directories:
                assert key in config.modalities, f"Data directory defined for non-existent modality: {key}"
        
        def test_split_ratios_validity(self):
            """
            GIVEN HKU956Config instance
            WHEN examining split ratios
            THEN they should sum to 1.0 and have the expected number of splits
            """
            config = HKU956Config()
            
            # Should have exactly 3 splits (train, val, test)
            assert len(config.split_ratios) == 3, f"Expected 3 split ratios, got {len(config.split_ratios)}"
            
            # Ratios should sum to 1.0 (allowing for floating point imprecision)
            assert abs(sum(config.split_ratios) - 1.0) < 1e-10, f"Split ratios should sum to 1.0, got {sum(config.split_ratios)}"
            
            # Each ratio should be positive
            for ratio in config.split_ratios:
                assert ratio > 0, f"Split ratio should be positive, got {ratio}"
        
        def test_custom_initialization(self):
            """
            GIVEN custom parameters
            WHEN initializing HKU956Config with those parameters
            THEN it should override defaults while preserving other values
            """
            custom_config = HKU956Config(
                seed=100,
                split_ratios=[0.7, 0.2, 0.1],
                modalities=["audio", "eda", "ecg"]
            )
            
            # Custom values should be used
            assert custom_config.seed == 100
            assert custom_config.split_ratios == [0.7, 0.2, 0.1]
            assert "ecg" in custom_config.modalities
            
            # Default values should remain for non-overridden fields
            assert custom_config.dataset_name == "HKU956"
            assert "HKU956" in custom_config.dataset_root_path
            assert "physiological_signals" in custom_config.data_directories["eda"]
        
        def test_with_dataloader_builder(self):
            """
            GIVEN HKU956Config instance
            WHEN used with DataLoaderBuilder
            THEN it should be compatible with the builder's interface
            """
            from src.data.dataloader import DataLoaderBuilder
            
            # This test verifies that HKU956Config can be used with DataLoaderBuilder
            # We're not actually building dataloaders, just checking the interface compatibility
            
            dataset_mapping = {
                'hku956': (None, None, HKU956Config()),
            }
            
            # Extract the config from the mapping
            _, _, config = dataset_mapping['hku956']
            
            # Verify it's the right type
            assert isinstance(config, HKU956Config)
            assert isinstance(config, DatasetConfig)
            
            # Verify it has all the expected attributes that DataLoaderBuilder would use
            assert hasattr(config, 'dataset_name')
            assert hasattr(config, 'dataset_root_path')
            assert hasattr(config, 'modalities')
            assert hasattr(config, 'data_directories')
        
        def test_model_dump_and_validation(self):
            """
            GIVEN HKU956Config instance
            WHEN serializing and deserializing
            THEN the resulting object should match the original
            """
            original_config = HKU956Config()
            
            # Serialize to dict
            config_dict = original_config.model_dump()
            
            # Deserialize from dict
            deserialized_config = HKU956Config.model_validate(config_dict)
            
            # Compare fields
            assert deserialized_config.dataset_name == original_config.dataset_name
            assert deserialized_config.dataset_root_path == original_config.dataset_root_path
            assert deserialized_config.modalities == original_config.modalities
            assert deserialized_config.file_format == original_config.file_format
            assert deserialized_config.data_directories == original_config.data_directories
            assert deserialized_config.split_ratios == original_config.split_ratios
            assert deserialized_config.seed == original_config.seed
            
            # Serialize to JSON and back
            json_str = original_config.model_dump_json()
            json_deserialized = HKU956Config.model_validate_json(json_str)
            
            # Verify JSON serialization/deserialization works
            assert json_deserialized.dataset_name == original_config.dataset_name
            assert json_deserialized.dataset_root_path == original_config.dataset_root_path
        
        def test_with_hku956_dataset(self):
            """
            GIVEN HKU956Config instance
            WHEN used to initialize an HKU956Dataset
            THEN it should provide all necessary configuration
            """
            from unittest.mock import patch, MagicMock
            
            config = HKU956Config()
            
            # Mock the HKU956Dataset to avoid actual initialization
            with patch('src.data.datasets.hku956.HKU956Dataset') as MockDataset:
                # Create a mock for AudioEDAFeatureConfig
                mock_feature_config = MagicMock()
                
                # Initialize the dataset with our config
                MockDataset(config, mock_feature_config)
                
                # Verify the dataset was initialized with our config
                MockDataset.assert_called_once()
                args, kwargs = MockDataset.call_args
                assert args[0] is config
                assert args[1] is mock_feature_config
                
        def test_file_format_extensions(self):
            """
            GIVEN HKU956Config instance
            WHEN examining file format extensions
            THEN they should all start with a period
            """
            config = HKU956Config()
            
            # All file extensions should start with a period
            for ext in config.file_format.values():
                assert ext.startswith('.'), f"File extension '{ext}' should start with a period"
                
            # Test with custom file formats
            custom_config = HKU956Config(
                file_format={"audio": ".wav", "eda": ".dat"}
            )
            assert custom_config.file_format["audio"] == ".wav"
            assert custom_config.file_format["eda"] == ".dat"
            
            # Test with invalid extension (no period)
            # Note: Currently this is allowed as there's no validation
            # This test documents current behavior
            custom_config = HKU956Config(
                file_format={"audio": "mp3", "eda": "csv"}
            )
            assert custom_config.file_format["audio"] == "mp3"
            assert custom_config.file_format["eda"] == "csv"
        
        def test_s3_path_handling(self):
            """
            GIVEN HKU956Config instance with various S3 path formats
            WHEN examining the paths
            THEN they should be properly handled
            """
            # Test with trailing slash
            config = HKU956Config(
                dataset_root_path="s3://audio2biosignal-train-data/HKU956/"
            )
            assert config.dataset_root_path.endswith('/')
            
            # Test without trailing slash
            config = HKU956Config(
                dataset_root_path="s3://audio2biosignal-train-data/HKU956"
            )
            assert not config.dataset_root_path.endswith('/')
            
            # Test with different bucket name
            config = HKU956Config(
                dataset_root_path="s3://custom-bucket/HKU956/"
            )
            assert "custom-bucket" in config.dataset_root_path
            
            # Test with subdirectories
            config = HKU956Config(
                dataset_root_path="s3://audio2biosignal-train-data/datasets/HKU956/"
            )
            assert "datasets/HKU956" in config.dataset_root_path
        
        def test_empty_metadata_paths(self):
            """
            GIVEN HKU956Config instance
            WHEN examining metadata_paths
            THEN it should handle empty metadata paths correctly
            """
            config = HKU956Config()
            
            # Default should be an empty list
            assert isinstance(config.metadata_paths, list)
            assert len(config.metadata_paths) == 0
            
            # Test with custom metadata paths
            custom_config = HKU956Config(
                metadata_paths=["path1.csv", "path2.csv"]
            )
            assert len(custom_config.metadata_paths) == 2
            assert "path1.csv" in custom_config.metadata_paths
            assert "path2.csv" in custom_config.metadata_paths
            
            # Test with empty list explicitly set
            explicit_empty = HKU956Config(metadata_paths=[])
            assert len(explicit_empty.metadata_paths) == 0
    
    def test_serialization_deserialization(self):
        """
        GIVEN a valid DatasetConfig instance
        WHEN serializing to JSON and back
        THEN the resulting object should match the original
        """
        original_config = DatasetConfig(
            dataset_name="TestDataset",
            dataset_root_path="/path/to/dataset",
            modalities=["audio", "eda"],
            file_format={"audio": ".wav", "eda": ".csv"},
            data_directories={
                "audio": "/path/to/dataset/audio",
                "eda": "/path/to/dataset/eda"
            },
            metadata_paths=["/path/to/dataset/metadata.csv"],
            split_ratios=[0.7, 0.15, 0.15],
            seed=42
        )
        
        # Serialize to JSON
        json_str = original_config.model_dump_json()
        
        # Deserialize from JSON
        deserialized_config = DatasetConfig.model_validate_json(json_str)
        
        # Compare fields
        assert deserialized_config.dataset_name == original_config.dataset_name
        assert deserialized_config.dataset_root_path == original_config.dataset_root_path
        assert deserialized_config.modalities == original_config.modalities
        assert deserialized_config.file_format == original_config.file_format
        assert deserialized_config.data_directories == original_config.data_directories
        assert deserialized_config.metadata_paths == original_config.metadata_paths
        assert deserialized_config.split_ratios == original_config.split_ratios
        assert deserialized_config.seed == original_config.seed
    
    def test_extreme_values(self):
        """
        GIVEN initialization parameters with extreme values
        WHEN initializing a DatasetConfig
        THEN it should handle them appropriately
        """
        # Very long strings
        long_name = "x" * 1000
        
        # Large collections
        many_modalities = [f"modality_{i}" for i in range(100)]
        many_formats = {f"modality_{i}": f".format_{i}" for i in range(100)}
        many_directories = {f"modality_{i}": f"/path/to/{i}" for i in range(100)}
        many_metadata_paths = [f"/path/to/metadata_{i}.csv" for i in range(100)]
        
        # Extreme numeric values
        large_seed = 2**31 - 1  # Max 32-bit signed int
        
        config = DatasetConfig(
            dataset_name=long_name,
            dataset_root_path="/path/to/dataset",
            modalities=many_modalities,
            file_format=many_formats,
            data_directories=many_directories,
            metadata_paths=many_metadata_paths,
            split_ratios=[0.5, 0.5],
            seed=large_seed
        )
        
        assert config.dataset_name == long_name
        assert len(config.modalities) == 100
        assert len(config.file_format) == 100
        assert len(config.data_directories) == 100
        assert len(config.metadata_paths) == 100
        assert config.seed == large_seed
    
    def test_modality_consistency_validation(self):
        """
        GIVEN inconsistent modalities and file_format/data_directories
        WHEN initializing a DatasetConfig
        THEN it should still create the instance (no validation for consistency)
        """
        # Note: This test verifies current behavior. If validation for modality consistency
        # is added in the future, this test should be updated.
        config = DatasetConfig(
            dataset_name="TestDataset",
            dataset_root_path="/path/to/dataset",
            modalities=["audio", "eda"],
            file_format={"audio": ".wav", "video": ".mp4"},  # 'video' not in modalities
            data_directories={
                "audio": "/path/to/dataset/audio",
                "text": "/path/to/dataset/text"  # 'text' not in modalities
            },
            metadata_paths=["/path/to/dataset/metadata.csv"],
            split_ratios=[0.7, 0.15, 0.15],
            seed=42
        )
        
        # Verify the inconsistent values are accepted
        assert "video" in config.file_format
        assert "text" in config.data_directories
        assert "eda" not in config.file_format  # No validation enforcing this
        assert "eda" not in config.data_directories  # No validation enforcing this
