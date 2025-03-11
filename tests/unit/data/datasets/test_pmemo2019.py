"""
Test suite for PMEmo2019 dataset.

This module provides comprehensive testing for:
1. Basic collate functionality
2. Padding behavior with different tensor sizes
3. Edge cases (empty batch, single item batch)
4. Tensor dimension preservation
5. Dataset initialization and configuration
6. Data loading and preprocessing
7. Indexing and iteration behavior
8. Prefetching mechanism
"""

import pytest
import torch
import pandas as pd
import numpy as np
from unittest.mock import patch, MagicMock, call
from pathlib import Path
from src.data.datasets.pmemo2019 import collate_fn, PMEmo2019Dataset
from src.configs import PMEmo2019Config, AudioEDAFeatureConfig
from src.utilities import S3FileManager


class TestPMEmo2019Dataset:
    """Tests for the PMEmo2019Dataset class."""
    
    @pytest.fixture
    def mock_s3_manager(self):
        """
        Provides a mocked S3FileManager.
        
        Returns:
            MagicMock: Mocked S3FileManager
        """
        with patch('src.data.datasets.pmemo2019.S3FileManager', autospec=True) as mock:
            # Configure the mock
            mock_instance = mock.return_value
            mock_instance.download_file.return_value = Path("/tmp/mock_file.csv")
            mock_instance.prefetch_files.return_value = None
            yield mock_instance
    
    @pytest.fixture
    def mock_preprocess_audio(self):
        """
        Provides a mocked preprocess_audio function.
        
        Returns:
            MagicMock: Mocked preprocess_audio function
        """
        with patch('src.data.datasets.pmemo2019.preprocess_audio') as mock:
            mock.return_value = torch.rand(5, 100)  # 5 features, 100 time steps
            yield mock
    
    @pytest.fixture
    def mock_preprocess_eda(self):
        """
        Provides a mocked preprocess_eda function.
        
        Returns:
            MagicMock: Mocked preprocess_eda function
        """
        with patch('src.data.datasets.pmemo2019.preprocess_eda') as mock:
            mock.return_value = torch.rand(100)  # 100 time steps
            yield mock
    
    @pytest.fixture
    def mock_pd_read_csv(self):
        """
        Provides a mocked pandas.read_csv function.
        
        Returns:
            MagicMock: Mocked pd.read_csv function
        """
        with patch('pandas.read_csv') as mock:
            # For metadata CSV
            metadata_df = pd.DataFrame({
                'musicId': ['1', '2', '3'],
                'fileName': ['song1.mp3', 'song2.mp3', 'song3.mp3']
            })
            
            # For EDA CSV
            eda_df = pd.DataFrame({
                'time': [0.1, 0.2, 0.3],
                'subject1': [0.5, 0.6, 0.7],
                'subject2': [0.8, 0.9, 1.0]
            })
            
            # Configure mock to return different DataFrames based on path
            def side_effect(path, *args, **kwargs):
                if 'metadata' in str(path):
                    return metadata_df
                else:
                    return eda_df
            
            mock.side_effect = side_effect
            yield mock
    
    @pytest.fixture
    def dataset_config(self):
        """
        Provides a standard PMEmo2019Config.
        
        Returns:
            PMEmo2019Config: Standard configuration
        """
        return PMEmo2019Config()
    
    @pytest.fixture
    def feature_config(self):
        """
        Provides a standard AudioEDAFeatureConfig.
        
        Returns:
            AudioEDAFeatureConfig: Standard configuration
        """
        return AudioEDAFeatureConfig(
            audio_sample_rate=16000,
            audio_n_mels=40,
            eda_sample_rate=4,
            sequence_length=10
        )
    
    def test_initialization(self, mock_s3_manager, mock_pd_read_csv, dataset_config, feature_config):
        """
        GIVEN valid configuration objects
        WHEN PMEmo2019Dataset is initialized
        THEN it should properly set up internal state and load metadata
        """
        # Setup _eda_files with test data
        with patch.object(PMEmo2019Dataset, '_eda_files', {
            ('1', 'subject1'): 's3://path/to/eda1.csv',
            ('2', 'subject2'): 's3://path/to/eda2.csv',
            ('3', 'subject1'): 's3://path/to/eda3.csv',
        }):
            dataset = PMEmo2019Dataset(dataset_config, feature_config)
            
            # Verify S3 manager was created
            assert dataset.s3_manager == mock_s3_manager
            
            # Verify metadata was downloaded
            mock_s3_manager.download_file.assert_called_with(
                "s3://audio2biosignal-train-data/PMEmo2019/metadata.csv"
            )
            
            # Verify audio files were loaded from metadata
            assert len(dataset._audio_files) == 3
            assert dataset._audio_files['1'] == "s3://audio2biosignal-train-data/PMEmo2019/chorus/song1.mp3"
            assert dataset._audio_files['2'] == "s3://audio2biosignal-train-data/PMEmo2019/chorus/song2.mp3"
            assert dataset._audio_files['3'] == "s3://audio2biosignal-train-data/PMEmo2019/chorus/song3.mp3"
            
            # Verify examples were created correctly
            assert len(dataset.examples) == 3  # One for each valid (music_id, subject_id) pair
            
            # Verify the structure of examples
            for example in dataset.examples:
                # Each example should be a dict with a single key-value pair
                assert len(example) == 1
                
                # Extract the key (subject_id, music_id) and value (audio_path, eda_path)
                (subject_id, music_id), (audio_path, eda_path) = list(example.items())[0]
                
                # Verify the key format
                assert isinstance(subject_id, str)
                assert isinstance(music_id, str)
                
                # Verify the value format
                assert isinstance(audio_path, str)
                assert isinstance(eda_path, str)
                assert audio_path.startswith("s3://")
                assert eda_path.startswith("s3://")
                
                # Verify audio path matches the one in _audio_files
                assert audio_path == dataset._audio_files[music_id]
    
    def test_load_audio_file(self, mock_s3_manager, mock_preprocess_audio, dataset_config, feature_config):
        """
        GIVEN a valid audio file S3 path
        WHEN _load_audio_file is called
        THEN it should download the file and preprocess it correctly
        """
        dataset = PMEmo2019Dataset(dataset_config, feature_config)
        
        # Test loading an audio file
        audio_s3_path = "s3://audio2biosignal-train-data/PMEmo2019/chorus/test.mp3"
        audio_tensor = dataset._load_audio_file(audio_s3_path)
        
        # Verify file was downloaded
        mock_s3_manager.download_file.assert_called_with(audio_s3_path)
        
        # Verify preprocessing was called
        mock_preprocess_audio.assert_called_once()
        assert torch.is_tensor(audio_tensor)
        assert audio_tensor.shape == (5, 100)  # From our mock
    
    def test_load_audio_file_with_download_failure(self, mock_s3_manager, mock_preprocess_audio, dataset_config, feature_config):
        """
        GIVEN an S3 path that fails to download
        WHEN _load_audio_file is called
        THEN it should propagate the exception from the S3 manager
        """
        # Configure S3 manager to raise an exception
        mock_s3_manager.download_file.side_effect = Exception("Failed to download audio file")
        
        dataset = PMEmo2019Dataset(dataset_config, feature_config)
        
        # Verify exception is propagated
        with pytest.raises(Exception, match="Failed to download audio file"):
            dataset._load_audio_file("s3://audio2biosignal-train-data/PMEmo2019/chorus/nonexistent.mp3")
        
        # Verify preprocessing was not called
        mock_preprocess_audio.assert_not_called()
    
    def test_load_audio_file_with_preprocessing_failure(self, mock_s3_manager, mock_preprocess_audio, dataset_config, feature_config):
        """
        GIVEN an audio file that fails during preprocessing
        WHEN _load_audio_file is called
        THEN it should propagate the exception from the preprocessing function
        """
        # Configure preprocess_audio to raise an exception
        mock_preprocess_audio.side_effect = ValueError("Invalid audio format")
        
        dataset = PMEmo2019Dataset(dataset_config, feature_config)
        
        # Verify exception is propagated
        with pytest.raises(ValueError, match="Invalid audio format"):
            dataset._load_audio_file("s3://audio2biosignal-train-data/PMEmo2019/chorus/test.mp3")
        
        # Verify download was still called
        mock_s3_manager.download_file.assert_called_once()
    
    def test_load_audio_file_caching(self, mock_s3_manager, mock_preprocess_audio, dataset_config, feature_config):
        """
        GIVEN the same audio file path called multiple times
        WHEN _load_audio_file is called
        THEN it should reuse the downloaded file and not download it again
        """
        # Configure S3 manager to return the same path for multiple calls
        local_path = Path("/tmp/cached_audio.mp3")
        mock_s3_manager.download_file.return_value = local_path
        
        dataset = PMEmo2019Dataset(dataset_config, feature_config)
        
        # Call _load_audio_file twice with the same path
        audio_s3_path = "s3://audio2biosignal-train-data/PMEmo2019/chorus/test.mp3"
        dataset._load_audio_file(audio_s3_path)
        dataset._load_audio_file(audio_s3_path)
        
        # Verify download was called only once with the same path
        mock_s3_manager.download_file.assert_called_once_with(audio_s3_path)
        
        # Verify preprocessing was called twice (once per call)
        assert mock_preprocess_audio.call_count == 2
        
        # Verify both calls to preprocess_audio used the same local path
        for call_args in mock_preprocess_audio.call_args_list:
            assert call_args[0][0] == local_path
    
    def test_load_audio_file_with_empty_path(self, mock_s3_manager, dataset_config, feature_config):
        """
        GIVEN an empty S3 path
        WHEN _load_audio_file is called
        THEN it should raise a ValueError
        """
        dataset = PMEmo2019Dataset(dataset_config, feature_config)
        
        # Verify empty path raises ValueError
        with pytest.raises(ValueError, match="Empty audio file path"):
            dataset._load_audio_file("")
        
        # Verify download was not called
        mock_s3_manager.download_file.assert_not_called()
    
    def test_load_audio_file_with_different_feature_configs(self, mock_s3_manager, dataset_config, feature_config):
        """
        GIVEN different feature configurations
        WHEN _load_audio_file is called
        THEN it should pass the correct feature config to preprocess_audio
        """
        dataset = PMEmo2019Dataset(dataset_config, feature_config)
        
        # Create a patched preprocess_audio that captures the feature_config
        with patch('src.data.datasets.pmemo2019.preprocess_audio') as mock_preprocess:
            mock_preprocess.return_value = torch.rand(5, 100)
            
            # Call _load_audio_file
            audio_s3_path = "s3://audio2biosignal-train-data/PMEmo2019/chorus/test.mp3"
            dataset._load_audio_file(audio_s3_path)
            
            # Verify preprocess_audio was called with the correct feature_config
            mock_preprocess.assert_called_once()
            _, kwargs = mock_preprocess.call_args
            assert kwargs.get('feature_config') == feature_config or mock_preprocess.call_args[0][1] == feature_config
    
    def test_load_eda_file(self, mock_s3_manager, mock_pd_read_csv, mock_preprocess_eda, dataset_config, feature_config):
        """
        GIVEN a valid EDA file S3 path and subject ID
        WHEN _load_eda_file is called
        THEN it should download the file, extract the subject's data, and preprocess it
        """
        dataset = PMEmo2019Dataset(dataset_config, feature_config)
        
        # Test loading an EDA file
        eda_s3_path = "s3://audio2biosignal-train-data/PMEmo2019/eda/test.csv"
        subject_id = "subject1"
        eda_tensor = dataset._load_eda_file(eda_s3_path, subject_id)
        
        # Verify file was downloaded
        mock_s3_manager.download_file.assert_called_with(eda_s3_path)
        
        # Verify CSV was read
        mock_pd_read_csv.assert_called()
        
        # Verify preprocessing was called with the correct subject's data
        mock_preprocess_eda.assert_called_once()
        assert torch.is_tensor(eda_tensor)
        assert eda_tensor.shape == (100,)  # From our mock
    
    def test_len(self, dataset_config, feature_config):
        """
        GIVEN a dataset with examples
        WHEN __len__ is called
        THEN it should return the correct number of examples
        """
        with patch.object(PMEmo2019Dataset, 'examples', [1, 2, 3]):
            dataset = PMEmo2019Dataset(dataset_config, feature_config)
            assert len(dataset) == 3
    
    @pytest.mark.parametrize("examples_data,expected_length", [
        ([], 0),                                # Empty list
        ([{}], 1),                              # Single empty dict
        ([{}, {}, {}], 3),                      # Multiple empty dicts
        ([{'key': 'value'}], 1),                # Single dict with content
        ([{'key1': 'value1'}, {'key2': 'value2'}, {'key3': 'value3'}], 3)  # Multiple dicts with content
    ])
    def test_len_with_various_examples(self, dataset_config, feature_config, examples_data, expected_length):
        """
        GIVEN a dataset with various examples configurations
        WHEN __len__ is called
        THEN it should return the correct number of examples regardless of their content
        """
        with patch.object(PMEmo2019Dataset, 'examples', examples_data):
            dataset = PMEmo2019Dataset(dataset_config, feature_config)
            assert len(dataset) == expected_length, f"Expected length {expected_length} but got {len(dataset)} for examples {examples_data}"
    
    def test_len_consistency_with_getitem(self, dataset_config, feature_config):
        """
        GIVEN a dataset with examples
        WHEN __len__ is called and items are accessed via __getitem__
        THEN the length should be consistent with the number of accessible items
        """
        examples = [
            {('subject1', '1'): ('s3://path/to/audio1.mp3', 's3://path/to/eda1.csv')},
            {('subject2', '2'): ('s3://path/to/audio2.mp3', 's3://path/to/eda2.csv')},
            {('subject3', '3'): ('s3://path/to/audio3.mp3', 's3://path/to/eda3.csv')}
        ]
        
        with patch.object(PMEmo2019Dataset, 'examples', examples):
            dataset = PMEmo2019Dataset(dataset_config, feature_config)
            
            # Mock _load_audio_file and _load_eda_file to avoid actual loading
            with patch.object(dataset, '_load_audio_file', return_value=torch.rand(5, 100)):
                with patch.object(dataset, '_load_eda_file', return_value=torch.rand(100)):
                    # Verify length
                    length = len(dataset)
                    assert length == 3
                    
                    # Verify we can access each item by index
                    for i in range(length):
                        item = dataset[i]
                        assert isinstance(item, tuple)
                        assert len(item) == 2
                        assert torch.is_tensor(item[0])
                        assert torch.is_tensor(item[1])
    
    def test_getitem(self, mock_s3_manager, dataset_config, feature_config, mock_preprocess_audio, mock_preprocess_eda):
        """
        GIVEN a valid index
        WHEN __getitem__ is called
        THEN it should return the corresponding audio and EDA tensors and prefetch next examples
        """
        # Setup examples with test data
        with patch.object(PMEmo2019Dataset, 'examples', [
            {('subject1', '1'): ('s3://path/to/audio1.mp3', 's3://path/to/eda1.csv')},
            {('subject2', '2'): ('s3://path/to/audio2.mp3', 's3://path/to/eda2.csv')},
            {('subject1', '3'): ('s3://path/to/audio3.mp3', 's3://path/to/eda3.csv')},
        ]):
            dataset = PMEmo2019Dataset(dataset_config, feature_config)
            
            # Test getting an item
            audio_tensor, eda_tensor = dataset[0]
            
            # Verify tensors were returned
            assert torch.is_tensor(audio_tensor)
            assert torch.is_tensor(eda_tensor)
            assert audio_tensor.shape == (5, 100)  # From our mock
            assert eda_tensor.shape == (100,)  # From our mock
            
            # Verify prefetching was triggered for next examples
            mock_s3_manager.prefetch_files.assert_called_once()
            # Should prefetch audio and EDA for next examples
            assert mock_s3_manager.prefetch_files.call_args[0][0] == [
                's3://path/to/audio2.mp3', 's3://path/to/eda2.csv',
                's3://path/to/audio3.mp3', 's3://path/to/eda3.csv'
            ]
    
    def test_getitem_last_example(self, mock_s3_manager, dataset_config, feature_config):
        """
        GIVEN the last index in the dataset
        WHEN __getitem__ is called
        THEN it should return the corresponding tensors without prefetching
        """
        # Setup examples with test data - only one example
        with patch.object(PMEmo2019Dataset, 'examples', [
            {('subject1', '1'): ('s3://path/to/audio1.mp3', 's3://path/to/eda1.csv')},
        ]):
            dataset = PMEmo2019Dataset(dataset_config, feature_config)
            
            # Mock the load methods to return tensors directly
            with patch.object(dataset, '_load_audio_file', return_value=torch.rand(5, 100)):
                with patch.object(dataset, '_load_eda_file', return_value=torch.rand(100)):
                    # Test getting the last item
                    audio_tensor, eda_tensor = dataset[0]
                    
                    # Verify tensors were returned
                    assert torch.is_tensor(audio_tensor)
                    assert torch.is_tensor(eda_tensor)
                    
                    # Verify prefetching was NOT triggered (no next examples)
                    mock_s3_manager.prefetch_files.assert_not_called()
    
    def test_empty_dataset(self, dataset_config, feature_config):
        """
        GIVEN a dataset with no examples
        WHEN operations are performed on it
        THEN it should behave appropriately
        """
        # Setup empty examples
        with patch.object(PMEmo2019Dataset, 'examples', []):
            dataset = PMEmo2019Dataset(dataset_config, feature_config)
            
            # Verify length is 0
            assert len(dataset) == 0
            
            # Verify indexing raises IndexError
            with pytest.raises(IndexError):
                dataset[0]
    
    def test_error_handling_missing_audio(self, dataset_config, feature_config):
        """
        GIVEN a music_id without corresponding audio file
        WHEN examples are created
        THEN it should skip that example
        """
        # Setup _eda_files with test data including one with missing audio
        with patch.object(PMEmo2019Dataset, '_eda_files', {
            ('1', 'subject1'): 's3://path/to/eda1.csv',
            ('999', 'subject2'): 's3://path/to/eda2.csv',  # No audio for this music_id
        }):
            # Setup _audio_files with test data missing music_id 999
            with patch.object(PMEmo2019Dataset, '_audio_files', {
                '1': 's3://path/to/audio1.mp3',
            }):
                dataset = PMEmo2019Dataset(dataset_config, feature_config)
                
                # Verify only valid examples were created
                assert len(dataset.examples) == 1
                assert ('subject1', '1') in list(dataset.examples[0].keys())[0]
                
                # Verify the invalid example was skipped
                for example in dataset.examples:
                    keys = list(example.keys())
                    assert ('subject2', '999') not in keys
    
    def test_error_handling_download_failure(self, mock_s3_manager, dataset_config, feature_config):
        """
        GIVEN an S3 download failure
        WHEN _load_audio_file or _load_eda_file is called
        THEN it should propagate the exception
        """
        # Configure S3 manager to raise an exception
        mock_s3_manager.download_file.side_effect = Exception("S3 download failed")
        
        dataset = PMEmo2019Dataset(dataset_config, feature_config)
        
        # Setup examples with test data
        with patch.object(dataset, 'examples', [
            {('subject1', '1'): ('s3://path/to/audio1.mp3', 's3://path/to/eda1.csv')},
        ]):
            # Verify exception is propagated from _load_audio_file
            with pytest.raises(Exception, match="S3 download failed"):
                dataset._load_audio_file('s3://path/to/audio1.mp3')
            
            # Verify exception is propagated from _load_eda_file
            with pytest.raises(Exception, match="S3 download failed"):
                dataset._load_eda_file('s3://path/to/eda1.csv', 'subject1')
    
    def test_load_audio_file_with_non_s3_path(self, mock_s3_manager, mock_preprocess_audio, dataset_config, feature_config):
        """
        GIVEN a non-S3 path (e.g., local file path)
        WHEN _load_audio_file is called
        THEN it should still attempt to process it through the S3 manager
        """
        dataset = PMEmo2019Dataset(dataset_config, feature_config)
        
        # Test with a local file path
        local_path = "/path/to/local/audio.mp3"
        dataset._load_audio_file(local_path)
        
        # Verify S3 manager was still called with the local path
        mock_s3_manager.download_file.assert_called_once_with(local_path)
        mock_preprocess_audio.assert_called_once()
    
    def test_load_audio_file_with_unsupported_format(self, mock_s3_manager, dataset_config, feature_config):
        """
        GIVEN an audio file with unsupported format
        WHEN _load_audio_file is called
        THEN it should attempt to process it and let preprocess_audio handle format validation
        """
        dataset = PMEmo2019Dataset(dataset_config, feature_config)
        
        # Create a patched preprocess_audio that raises an exception for unsupported format
        with patch('src.data.datasets.pmemo2019.preprocess_audio') as mock_preprocess:
            mock_preprocess.side_effect = ValueError("Unsupported audio format: .xyz")
            
            # Test with an unsupported format
            unsupported_path = "s3://audio2biosignal-train-data/PMEmo2019/chorus/test.xyz"
            
            # Verify the exception from preprocess_audio is propagated
            with pytest.raises(ValueError, match="Unsupported audio format"):
                dataset._load_audio_file(unsupported_path)
            
            # Verify download was still attempted
            mock_s3_manager.download_file.assert_called_once_with(unsupported_path)
    
    def test_invalid_subject_id(self, mock_s3_manager, mock_pd_read_csv, dataset_config, feature_config):
        """
        GIVEN an invalid subject ID
        WHEN _load_eda_file is called
        THEN it should raise a KeyError
        """
        dataset = PMEmo2019Dataset(dataset_config, feature_config)
        
        # Test loading an EDA file with non-existent subject ID
        eda_s3_path = "s3://audio2biosignal-train-data/PMEmo2019/eda/test.csv"
        non_existent_subject = "non_existent_subject"
        
        # Mock pandas DataFrame to raise KeyError for non-existent column
        with patch('pandas.DataFrame.__getitem__', side_effect=KeyError(non_existent_subject)):
            with pytest.raises(KeyError):
                dataset._load_eda_file(eda_s3_path, non_existent_subject)
    
    @pytest.mark.parametrize("index", [-1, 100])
    def test_getitem_index_out_of_range(self, dataset_config, feature_config, index):
        """
        GIVEN an index that is out of range
        WHEN __getitem__ is called
        THEN it should raise IndexError
        """
        # Setup examples with test data
        with patch.object(PMEmo2019Dataset, 'examples', [
            {('subject1', '1'): ('s3://path/to/audio1.mp3', 's3://path/to/eda1.csv')},
        ]):
            dataset = PMEmo2019Dataset(dataset_config, feature_config)
            
            # Verify IndexError is raised for out-of-range index
            with pytest.raises(IndexError):
                dataset[index]
    
    def test_getitem_with_malformed_example(self, mock_s3_manager, dataset_config, feature_config):
        """
        GIVEN an example with incorrect structure (empty dict)
        WHEN __getitem__ is called
        THEN it should raise KeyError
        """
        # Setup examples with malformed data (empty dict)
        with patch.object(PMEmo2019Dataset, 'examples', [{}]):
            dataset = PMEmo2019Dataset(dataset_config, feature_config)
            
            # Verify KeyError is raised for malformed example
            with pytest.raises(KeyError):
                dataset[0]
    
    def test_getitem_with_invalid_example_key_type(self, mock_s3_manager, dataset_config, feature_config):
        """
        GIVEN an example with incorrect key type (not a tuple)
        WHEN __getitem__ is called
        THEN it should raise TypeError or KeyError
        """
        # Setup examples with invalid key type
        with patch.object(PMEmo2019Dataset, 'examples', [
            {"invalid_key": ('s3://path/to/audio1.mp3', 's3://path/to/eda1.csv')}
        ]):
            dataset = PMEmo2019Dataset(dataset_config, feature_config)
            
            # Should raise an exception when trying to unpack the key
            with pytest.raises((TypeError, KeyError)):
                dataset[0]
    
    def test_getitem_with_invalid_example_value_type(self, mock_s3_manager, dataset_config, feature_config):
        """
        GIVEN an example with incorrect value type (not a tuple)
        WHEN __getitem__ is called
        THEN it should raise TypeError or ValueError
        """
        # Setup examples with invalid value type
        with patch.object(PMEmo2019Dataset, 'examples', [
            {('subject1', '1'): "not_a_tuple"}
        ]):
            dataset = PMEmo2019Dataset(dataset_config, feature_config)
            
            # Should raise an exception when trying to unpack the value
            with pytest.raises((TypeError, ValueError)):
                dataset[0]
    
    def test_prefetch_limit(self, mock_s3_manager, dataset_config, feature_config):
        """
        GIVEN a dataset with many examples
        WHEN __getitem__ is called
        THEN it should prefetch only up to the next 4 examples
        """
        # Setup examples with test data - more than the prefetch limit
        with patch.object(PMEmo2019Dataset, 'examples', [
            {(f'subject{i}', f'{i}'): (f's3://path/to/audio{i}.mp3', f's3://path/to/eda{i}.csv')}
            for i in range(10)
        ]):
            dataset = PMEmo2019Dataset(dataset_config, feature_config)
            
            # Mock the load methods to return tensors directly
            with patch.object(dataset, '_load_audio_file', return_value=torch.rand(5, 100)):
                with patch.object(dataset, '_load_eda_file', return_value=torch.rand(100)):
                    # Test getting the first item
                    dataset[0]
                    
                    # Verify prefetching was called with the right number of examples
                    mock_s3_manager.prefetch_files.assert_called_once()
                    # Should have 8 paths (4 examples * 2 files each)
                    assert len(mock_s3_manager.prefetch_files.call_args[0][0]) == 8
    
    def test_prefetch_exact_limit(self, mock_s3_manager, dataset_config, feature_config):
        """
        GIVEN a dataset with exactly 5 examples (current + 4 more)
        WHEN __getitem__ is called for the first example
        THEN it should prefetch exactly 4 examples (8 paths)
        """
        # Setup examples with test data - exactly at the prefetch limit
        with patch.object(PMEmo2019Dataset, 'examples', [
            {(f'subject{i}', f'{i}'): (f's3://path/to/audio{i}.mp3', f's3://path/to/eda{i}.csv')}
            for i in range(5)
        ]):
            dataset = PMEmo2019Dataset(dataset_config, feature_config)
            
            # Mock the load methods to return tensors directly
            with patch.object(dataset, '_load_audio_file', return_value=torch.rand(5, 100)):
                with patch.object(dataset, '_load_eda_file', return_value=torch.rand(100)):
                    # Test getting the first item
                    dataset[0]
                    
                    # Verify prefetching was called with exactly 4 examples
                    mock_s3_manager.prefetch_files.assert_called_once()
                    prefetched_paths = mock_s3_manager.prefetch_files.call_args[0][0]
                    assert len(prefetched_paths) == 8  # 4 examples * 2 files each
                    
                    # Verify the correct paths were prefetched
                    expected_paths = []
                    for i in range(1, 5):  # Examples 1-4
                        expected_paths.extend([
                            f's3://path/to/audio{i}.mp3', 
                            f's3://path/to/eda{i}.csv'
                        ])
                    assert set(prefetched_paths) == set(expected_paths)
    
    def test_prefetch_fewer_than_limit(self, mock_s3_manager, dataset_config, feature_config):
        """
        GIVEN a dataset with fewer than 5 examples (less than current + 4 more)
        WHEN __getitem__ is called for the first example
        THEN it should prefetch only the remaining examples
        """
        # Setup examples with test data - fewer than the prefetch limit
        with patch.object(PMEmo2019Dataset, 'examples', [
            {(f'subject{i}', f'{i}'): (f's3://path/to/audio{i}.mp3', f's3://path/to/eda{i}.csv')}
            for i in range(3)
        ]):
            dataset = PMEmo2019Dataset(dataset_config, feature_config)
            
            # Mock the load methods to return tensors directly
            with patch.object(dataset, '_load_audio_file', return_value=torch.rand(5, 100)):
                with patch.object(dataset, '_load_eda_file', return_value=torch.rand(100)):
                    # Test getting the first item
                    dataset[0]
                    
                    # Verify prefetching was called with only the remaining examples
                    mock_s3_manager.prefetch_files.assert_called_once()
                    prefetched_paths = mock_s3_manager.prefetch_files.call_args[0][0]
                    assert len(prefetched_paths) == 4  # 2 examples * 2 files each
                    
                    # Verify the correct paths were prefetched
                    expected_paths = []
                    for i in range(1, 3):  # Examples 1-2
                        expected_paths.extend([
                            f's3://path/to/audio{i}.mp3', 
                            f's3://path/to/eda{i}.csv'
                        ])
                    assert set(prefetched_paths) == set(expected_paths)

    def test_initialization_with_empty_eda_files(self, mock_s3_manager, mock_pd_read_csv, dataset_config, feature_config):
        """
        GIVEN an empty _eda_files dictionary
        WHEN PMEmo2019Dataset is initialized
        THEN it should create an empty examples list
        """
        # Setup empty _eda_files
        with patch.object(PMEmo2019Dataset, '_eda_files', {}):
            dataset = PMEmo2019Dataset(dataset_config, feature_config)
            
            # Verify audio files were still loaded from metadata
            assert len(dataset._audio_files) == 3
            
            # Verify no examples were created
            assert len(dataset.examples) == 0
    
    def test_initialization_with_malformed_metadata(self, mock_s3_manager, dataset_config, feature_config):
        """
        GIVEN malformed metadata CSV
        WHEN PMEmo2019Dataset is initialized
        THEN it should handle the error gracefully
        """
        # Mock pandas to raise an exception when reading the CSV
        with patch('pandas.read_csv', side_effect=Exception("Malformed CSV")):
            # Expect initialization to fail
            with pytest.raises(Exception) as excinfo:
                dataset = PMEmo2019Dataset(dataset_config, feature_config)
            
            # Verify the error is propagated
            assert "Malformed CSV" in str(excinfo.value)
    
    def test_initialization_with_missing_columns(self, mock_s3_manager, dataset_config, feature_config):
        """
        GIVEN metadata CSV with missing required columns
        WHEN PMEmo2019Dataset is initialized
        THEN it should handle the error gracefully
        """
        # Create a DataFrame missing required columns
        incomplete_df = pd.DataFrame({
            'musicId': ['1', '2', '3'],
            # Missing 'fileName' column
        })
        
        # Mock pandas to return the incomplete DataFrame
        with patch('pandas.read_csv', return_value=incomplete_df):
            # Expect initialization to fail when accessing missing column
            with pytest.raises(KeyError):
                dataset = PMEmo2019Dataset(dataset_config, feature_config)
    
    def test_initialization_with_duplicate_music_ids(self, mock_s3_manager, dataset_config, feature_config):
        """
        GIVEN metadata CSV with duplicate music IDs
        WHEN PMEmo2019Dataset is initialized
        THEN it should use the last occurrence of each music ID
        """
        # Create a DataFrame with duplicate music IDs
        duplicate_df = pd.DataFrame({
            'musicId': ['1', '1', '2'],  # Duplicate ID '1'
            'fileName': ['song1a.mp3', 'song1b.mp3', 'song2.mp3']
        })
        
        # Mock pandas to return the DataFrame with duplicates
        with patch('pandas.read_csv', return_value=duplicate_df):
            # Setup _eda_files with test data
            with patch.object(PMEmo2019Dataset, '_eda_files', {
                ('1', 'subject1'): 's3://path/to/eda1.csv',
                ('2', 'subject1'): 's3://path/to/eda2.csv',
            }):
                dataset = PMEmo2019Dataset(dataset_config, feature_config)
                
                # Verify the last occurrence of music ID '1' was used
                assert dataset._audio_files['1'] == "s3://audio2biosignal-train-data/PMEmo2019/chorus/song1b.mp3"
                
                # Verify examples were created correctly
                assert len(dataset.examples) == 2
    
    def test_initialization_with_invalid_config_types(self):
        """
        GIVEN invalid configuration object types
        WHEN PMEmo2019Dataset is initialized
        THEN it should raise TypeError
        """
        # Test with invalid dataset_config type
        with pytest.raises(TypeError):
            PMEmo2019Dataset(dataset_config="invalid_type", feature_config=AudioEDAFeatureConfig())
        
        # Test with invalid feature_config type
        with pytest.raises(TypeError):
            PMEmo2019Dataset(dataset_config=PMEmo2019Config(), feature_config="invalid_type")
    
    def test_initialization_with_s3_manager_singleton(self, mock_s3_manager, dataset_config, feature_config):
        """
        GIVEN multiple dataset instances
        WHEN PMEmo2019Dataset is initialized
        THEN it should reuse the same S3FileManager instance
        """
        # Setup _eda_files with test data
        with patch.object(PMEmo2019Dataset, '_eda_files', {}):
            # Create first dataset instance
            dataset1 = PMEmo2019Dataset(dataset_config, feature_config)
            
            # Create second dataset instance
            dataset2 = PMEmo2019Dataset(dataset_config, feature_config)
            
            # Verify both instances use the same S3FileManager (singleton pattern)
            assert dataset1.s3_manager is dataset2.s3_manager
            
            # Verify S3FileManager was only instantiated once
            from src.data.datasets.pmemo2019 import S3FileManager
            S3FileManager.assert_called_once()

class TestGetItemMethod:
    """Tests specifically focused on the __getitem__ method."""
    
    @pytest.fixture
    def dataset(self, mock_s3_manager, dataset_config, feature_config):
        """
        Provides a standard PMEmo2019Dataset instance with mocked dependencies.
        
        Returns:
            PMEmo2019Dataset: Configured dataset instance
        """
        with patch.object(PMEmo2019Dataset, '_eda_files', {}):
            dataset = PMEmo2019Dataset(dataset_config, feature_config)
            return dataset
            
    @pytest.fixture
    def example_factory(self):
        """
        Provides a factory function to create example dictionaries.
        
        Returns:
            function: Factory function for creating examples
        """
        def _create_example(subject_id, music_id, audio_path, eda_path):
            return {(subject_id, music_id): (audio_path, eda_path)}
        return _create_example
    
    def test_getitem_error_propagation_from_load_audio(self, dataset, mock_s3_manager):
        """
        GIVEN a scenario where _load_audio_file raises an exception
        WHEN __getitem__ is called
        THEN it should propagate the exception
        """
        # Setup examples with test data
        with patch.object(dataset, 'examples', [
            {('subject1', '1'): ('s3://path/to/audio1.mp3', 's3://path/to/eda1.csv')}
        ]):
            # Configure _load_audio_file to raise an exception
            with patch.object(dataset, '_load_audio_file', side_effect=IOError("Audio file error")):
                # Verify exception is propagated
                with pytest.raises(IOError, match="Audio file error"):
                    dataset[0]
    
    def test_getitem_error_propagation_from_load_eda(self, dataset, mock_s3_manager):
        """
        GIVEN a scenario where _load_eda_file raises an exception
        WHEN __getitem__ is called
        THEN it should propagate the exception
        """
        # Setup examples with test data
        with patch.object(dataset, 'examples', [
            {('subject1', '1'): ('s3://path/to/audio1.mp3', 's3://path/to/eda1.csv')}
        ]):
            # Configure _load_audio_file to return a tensor
            with patch.object(dataset, '_load_audio_file', return_value=torch.rand(5, 100)):
                # Configure _load_eda_file to raise an exception
                with patch.object(dataset, '_load_eda_file', side_effect=ValueError("EDA file error")):
                    # Verify exception is propagated
                    with pytest.raises(ValueError, match="EDA file error"):
                        dataset[0]
    
    def test_getitem_with_empty_path_strings(self, dataset, mock_s3_manager):
        """
        GIVEN examples with empty path strings
        WHEN __getitem__ is called
        THEN it should raise ValueError from the load methods
        """
        # Setup examples with empty path strings
        with patch.object(dataset, 'examples', [
            {('subject1', '1'): ('', 's3://path/to/eda1.csv')}
        ]):
            # Configure _load_audio_file to propagate the ValueError for empty path
            with patch.object(dataset, '_load_audio_file', side_effect=ValueError("Empty audio file path")):
                # Verify ValueError is raised
                with pytest.raises(ValueError, match="Empty audio file path"):
                    dataset[0]
        
        # Test empty EDA path
        with patch.object(dataset, 'examples', [
            {('subject1', '1'): ('s3://path/to/audio1.mp3', '')}
        ]):
            # Configure _load_audio_file to return a tensor
            with patch.object(dataset, '_load_audio_file', return_value=torch.rand(5, 100)):
                # Configure _load_eda_file to propagate the ValueError for empty path
                with patch.object(dataset, '_load_eda_file', side_effect=ValueError("Empty EDA file path")):
                    # Verify ValueError is raised
                    with pytest.raises(ValueError, match="Empty EDA file path"):
                        dataset[0]
    
    def test_getitem_with_multiple_items_in_example(self, dataset, mock_s3_manager):
        """
        GIVEN an example with multiple key-value pairs (violating expected format)
        WHEN __getitem__ is called
        THEN it should use only the first key-value pair
        """
        # Setup examples with multiple items
        with patch.object(dataset, 'examples', [
            {
                ('subject1', '1'): ('s3://path/to/audio1.mp3', 's3://path/to/eda1.csv'),
                ('subject2', '2'): ('s3://path/to/audio2.mp3', 's3://path/to/eda2.csv')
            }
        ]):
            # Mock the load methods to track which paths are used
            with patch.object(dataset, '_load_audio_file', return_value=torch.rand(5, 100)) as mock_load_audio:
                with patch.object(dataset, '_load_eda_file', return_value=torch.rand(100)) as mock_load_eda:
                    # Get the item
                    dataset[0]
                    
                    # Verify only the first item was used
                    mock_load_audio.assert_called_once_with('s3://path/to/audio1.mp3')
                    mock_load_eda.assert_called_once_with('s3://path/to/eda1.csv', 'subject1')
    
    def test_getitem_with_prefetch_exception(self, dataset, mock_s3_manager):
        """
        GIVEN a scenario where prefetch_files raises an exception
        WHEN __getitem__ is called
        THEN it should still return the tensors and not propagate the prefetch exception
        """
        # Setup examples with test data
        with patch.object(dataset, 'examples', [
            {('subject1', '1'): ('s3://path/to/audio1.mp3', 's3://path/to/eda1.csv')},
            {('subject2', '2'): ('s3://path/to/audio2.mp3', 's3://path/to/eda2.csv')}
        ]):
            # Configure prefetch_files to raise an exception
            mock_s3_manager.prefetch_files.side_effect = Exception("Prefetch error")
            
            # Mock the load methods to return tensors
            with patch.object(dataset, '_load_audio_file', return_value=torch.rand(5, 100)):
                with patch.object(dataset, '_load_eda_file', return_value=torch.rand(100)):
                    # Should not raise an exception despite prefetch error
                    audio_tensor, eda_tensor = dataset[0]
                    
                    # Verify tensors were returned
                    assert torch.is_tensor(audio_tensor)
                    assert torch.is_tensor(eda_tensor)
                    
                    # Verify prefetch was attempted
                    mock_s3_manager.prefetch_files.assert_called_once()
    
    def test_getitem_middle_of_dataset(self, dataset, mock_s3_manager, example_factory):
        """
        GIVEN a dataset with multiple examples
        WHEN __getitem__ is called with an index in the middle
        THEN it should return the correct item and prefetch subsequent examples
        """
        # Create a dataset with 7 examples
        examples = [
            example_factory(f"subject{i}", f"{i}", f"s3://path/to/audio{i}.mp3", f"s3://path/to/eda{i}.csv")
            for i in range(7)
        ]
        
        with patch.object(dataset, 'examples', examples):
            # Mock the load methods to return tensors
            with patch.object(dataset, '_load_audio_file', return_value=torch.rand(5, 100)) as mock_load_audio:
                with patch.object(dataset, '_load_eda_file', return_value=torch.rand(100)) as mock_load_eda:
                    # Get an item from the middle of the dataset (index 3)
                    audio_tensor, eda_tensor = dataset[3]
                    
                    # Verify correct item was loaded
                    mock_load_audio.assert_called_once_with('s3://path/to/audio3.mp3')
                    mock_load_eda.assert_called_once_with('s3://path/to/eda3.csv', 'subject3')
                    
                    # Verify prefetching was triggered for next examples (4, 5, 6)
                    mock_s3_manager.prefetch_files.assert_called_once()
                    prefetched_paths = mock_s3_manager.prefetch_files.call_args[0][0]
                    
                    # Should have 6 paths (3 examples * 2 files each)
                    assert len(prefetched_paths) == 6
                    
                    # Verify the correct paths were prefetched
                    expected_paths = []
                    for i in range(4, 7):  # Examples 4, 5, 6
                        expected_paths.extend([
                            f's3://path/to/audio{i}.mp3', 
                            f's3://path/to/eda{i}.csv'
                        ])
                    assert set(prefetched_paths) == set(expected_paths)
    
    def test_getitem_second_to_last_example(self, dataset, mock_s3_manager, example_factory):
        """
        GIVEN a dataset with multiple examples
        WHEN __getitem__ is called with the second-to-last index
        THEN it should return the correct item and prefetch only the last example
        """
        # Create a dataset with 5 examples
        examples = [
            example_factory(f"subject{i}", f"{i}", f"s3://path/to/audio{i}.mp3", f"s3://path/to/eda{i}.csv")
            for i in range(5)
        ]
        
        with patch.object(dataset, 'examples', examples):
            # Mock the load methods to return tensors
            with patch.object(dataset, '_load_audio_file', return_value=torch.rand(5, 100)) as mock_load_audio:
                with patch.object(dataset, '_load_eda_file', return_value=torch.rand(100)) as mock_load_eda:
                    # Get the second-to-last item (index 3)
                    audio_tensor, eda_tensor = dataset[3]
                    
                    # Verify correct item was loaded
                    mock_load_audio.assert_called_once_with('s3://path/to/audio3.mp3')
                    mock_load_eda.assert_called_once_with('s3://path/to/eda3.csv', 'subject3')
                    
                    # Verify prefetching was triggered for only the last example
                    mock_s3_manager.prefetch_files.assert_called_once()
                    prefetched_paths = mock_s3_manager.prefetch_files.call_args[0][0]
                    
                    # Should have 2 paths (1 example * 2 files)
                    assert len(prefetched_paths) == 2
                    
                    # Verify the correct paths were prefetched
                    expected_paths = [
                        's3://path/to/audio4.mp3', 
                        's3://path/to/eda4.csv'
                    ]
                    assert set(prefetched_paths) == set(expected_paths)
    
    def test_getitem_with_unexpected_example_structure(self, dataset, mock_s3_manager):
        """
        GIVEN examples with unexpected structure (not a dict with tuple key and tuple value)
        WHEN __getitem__ is called
        THEN it should handle the error gracefully
        """
        # Setup examples with completely unexpected structure
        with patch.object(dataset, 'examples', [
            "not_a_dict"  # String instead of dict
        ]):
            # Verify AttributeError is raised when trying to call .items() on a string
            with pytest.raises(AttributeError):
                dataset[0]
        
        # Setup examples with dict but unexpected content
        with patch.object(dataset, 'examples', [
            {"string_key": "string_value"}  # Not tuple key and tuple value
        ]):
            # Verify ValueError or TypeError is raised when unpacking
            with pytest.raises((ValueError, TypeError)):
                dataset[0]

class TestEDAFileLoading:
    """Tests specifically focused on EDA file loading functionality."""
    
    @pytest.fixture
    def dataset(self, mock_s3_manager, dataset_config, feature_config):
        """
        Provides a standard PMEmo2019Dataset instance.
        
        Returns:
            PMEmo2019Dataset: Configured dataset instance
        """
        with patch.object(PMEmo2019Dataset, '_eda_files', {}):
            dataset = PMEmo2019Dataset(dataset_config, feature_config)
            return dataset
    
    def test_load_eda_file_basic(self, dataset, mock_s3_manager, mock_preprocess_eda):
        """
        GIVEN a valid EDA file S3 path and subject ID
        WHEN _load_eda_file is called
        THEN it should download the file, extract the subject's data, and preprocess it correctly
        """
        # Configure mock_pd_read_csv to return a DataFrame with the subject column
        eda_df = pd.DataFrame({
            'time': [0.1, 0.2, 0.3],
            'subject1': [0.5, 0.6, 0.7]
        })
        
        with patch('pandas.read_csv', return_value=eda_df):
            # Test loading an EDA file
            eda_s3_path = "s3://audio2biosignal-train-data/PMEmo2019/eda/test.csv"
            subject_id = "subject1"
            eda_tensor = dataset._load_eda_file(eda_s3_path, subject_id)
            
            # Verify file was downloaded
            mock_s3_manager.download_file.assert_called_with(eda_s3_path)
            
            # Verify preprocessing was called with the correct data
            mock_preprocess_eda.assert_called_once()
            # Check that the values passed to preprocess_eda match the subject's column
            np.testing.assert_array_equal(
                mock_preprocess_eda.call_args[0][0], 
                eda_df['subject1'].values
            )
            
            # Verify feature_config was passed correctly
            assert mock_preprocess_eda.call_args[0][1] == dataset.feature_config
            
            # Verify return value
            assert torch.is_tensor(eda_tensor)
    
    def test_load_eda_file_with_empty_path(self, dataset, mock_s3_manager):
        """
        GIVEN an empty EDA file path
        WHEN _load_eda_file is called
        THEN it should raise a ValueError
        """
        with pytest.raises(ValueError, match="Empty EDA file path"):
            dataset._load_eda_file("", "subject1")
        
        # Verify download was not called
        mock_s3_manager.download_file.assert_not_called()
    
    def test_load_eda_file_with_empty_subject_id(self, dataset, mock_s3_manager):
        """
        GIVEN an empty subject ID
        WHEN _load_eda_file is called
        THEN it should raise a ValueError
        """
        with pytest.raises(ValueError, match="Empty subject ID"):
            dataset._load_eda_file("s3://path/to/eda.csv", "")
        
        # Verify download was not called
        mock_s3_manager.download_file.assert_not_called()
    
    def test_load_eda_file_with_download_failure(self, dataset, mock_s3_manager):
        """
        GIVEN an S3 path that fails to download
        WHEN _load_eda_file is called
        THEN it should propagate the exception from the S3 manager
        """
        # Configure S3 manager to raise an exception
        mock_s3_manager.download_file.side_effect = Exception("Failed to download EDA file")
        
        # Verify exception is propagated
        with pytest.raises(Exception, match="Failed to download EDA file"):
            dataset._load_eda_file("s3://path/to/nonexistent.csv", "subject1")
    
    def test_load_eda_file_with_csv_parsing_failure(self, dataset, mock_s3_manager):
        """
        GIVEN a file that cannot be parsed as CSV
        WHEN _load_eda_file is called
        THEN it should propagate the pandas exception
        """
        # Configure pandas.read_csv to raise an exception
        with patch('pandas.read_csv', side_effect=pd.errors.ParserError("CSV parsing error")):
            with pytest.raises(pd.errors.ParserError, match="CSV parsing error"):
                dataset._load_eda_file("s3://path/to/malformed.csv", "subject1")
        
        # Verify download was still called
        mock_s3_manager.download_file.assert_called_once()
    
    def test_load_eda_file_with_missing_subject_column(self, dataset, mock_s3_manager):
        """
        GIVEN a CSV file without the requested subject column
        WHEN _load_eda_file is called
        THEN it should raise a KeyError
        """
        # Create DataFrame without the requested subject column
        eda_df = pd.DataFrame({
            'time': [0.1, 0.2, 0.3],
            'other_subject': [0.5, 0.6, 0.7]  # Different subject
        })
        
        with patch('pandas.read_csv', return_value=eda_df):
            with pytest.raises(KeyError):
                dataset._load_eda_file("s3://path/to/eda.csv", "subject1")
    
    def test_load_eda_file_with_empty_dataframe(self, dataset, mock_s3_manager, mock_preprocess_eda):
        """
        GIVEN a CSV file with no data rows
        WHEN _load_eda_file is called
        THEN it should still process it correctly
        """
        # Create empty DataFrame but with the required columns
        empty_df = pd.DataFrame({
            'time': [],
            'subject1': []
        })
        
        with patch('pandas.read_csv', return_value=empty_df):
            dataset._load_eda_file("s3://path/to/empty.csv", "subject1")
            
            # Verify preprocessing was called with empty array
            mock_preprocess_eda.assert_called_once()
            assert len(mock_preprocess_eda.call_args[0][0]) == 0
    
    def test_load_eda_file_with_preprocessing_failure(self, dataset, mock_s3_manager):
        """
        GIVEN EDA data that fails during preprocessing
        WHEN _load_eda_file is called
        THEN it should propagate the exception from the preprocessing function
        """
        # Create valid DataFrame
        eda_df = pd.DataFrame({
            'time': [0.1, 0.2, 0.3],
            'subject1': [0.5, 0.6, 0.7]
        })
        
        with patch('pandas.read_csv', return_value=eda_df):
            # Configure preprocess_eda to raise an exception
            with patch('src.data.datasets.pmemo2019.preprocess_eda', 
                      side_effect=ValueError("Invalid EDA format")):
                
                with pytest.raises(ValueError, match="Invalid EDA format"):
                    dataset._load_eda_file("s3://path/to/eda.csv", "subject1")
    
    def test_load_eda_file_caching(self, dataset, mock_s3_manager, mock_preprocess_eda):
        """
        GIVEN the same EDA file path called multiple times
        WHEN _load_eda_file is called
        THEN it should reuse the downloaded file and not download it again
        """
        # Configure S3 manager to return the same path for multiple calls
        local_path = Path("/tmp/cached_eda.csv")
        mock_s3_manager.download_file.return_value = local_path
        
        # Create DataFrame with subject data
        eda_df = pd.DataFrame({
            'time': [0.1, 0.2, 0.3],
            'subject1': [0.5, 0.6, 0.7]
        })
        
        with patch('pandas.read_csv', return_value=eda_df) as mock_read_csv:
            # Call _load_eda_file twice with the same path but different subjects
            eda_s3_path = "s3://path/to/eda.csv"
            dataset._load_eda_file(eda_s3_path, "subject1")
            dataset._load_eda_file(eda_s3_path, "subject1")
            
            # Verify download was called only once with the same path
            mock_s3_manager.download_file.assert_called_once_with(eda_s3_path)
            
            # Verify CSV was read twice (once per call)
            assert mock_read_csv.call_count == 2
            
            # Verify both calls to read_csv used the same local path
            for call_args in mock_read_csv.call_args_list:
                assert call_args[0][0] == local_path
    
    def test_load_eda_file_with_different_feature_configs(self, dataset, mock_s3_manager):
        """
        GIVEN different feature configurations
        WHEN _load_eda_file is called
        THEN it should pass the correct feature config to preprocess_eda
        """
        # Create DataFrame with subject data
        eda_df = pd.DataFrame({
            'time': [0.1, 0.2, 0.3],
            'subject1': [0.5, 0.6, 0.7]
        })
        
        with patch('pandas.read_csv', return_value=eda_df):
            # Create a patched preprocess_eda that captures the feature_config
            with patch('src.data.datasets.pmemo2019.preprocess_eda') as mock_preprocess:
                mock_preprocess.return_value = torch.rand(100)
                
                # Call _load_eda_file
                eda_s3_path = "s3://path/to/eda.csv"
                dataset._load_eda_file(eda_s3_path, "subject1")
                
                # Verify preprocess_eda was called with the correct feature_config
                mock_preprocess.assert_called_once()
                assert mock_preprocess.call_args[0][1] == dataset.feature_config
    
    def test_load_eda_file_with_non_s3_path(self, dataset, mock_s3_manager, mock_preprocess_eda):
        """
        GIVEN a non-S3 path (e.g., local file path)
        WHEN _load_eda_file is called
        THEN it should still attempt to process it through the S3 manager
        """
        # Create DataFrame with subject data
        eda_df = pd.DataFrame({
            'time': [0.1, 0.2, 0.3],
            'subject1': [0.5, 0.6, 0.7]
        })
        
        with patch('pandas.read_csv', return_value=eda_df):
            # Test with a local file path
            local_path = "/path/to/local/eda.csv"
            dataset._load_eda_file(local_path, "subject1")
            
            # Verify S3 manager was still called with the local path
            mock_s3_manager.download_file.assert_called_once_with(local_path)
            mock_preprocess_eda.assert_called_once()
    
    def test_load_eda_file_with_numeric_subject_id(self, dataset, mock_s3_manager, mock_preprocess_eda):
        """
        GIVEN a numeric subject ID
        WHEN _load_eda_file is called
        THEN it should convert it to string and process correctly
        """
        # Create DataFrame with numeric column names
        eda_df = pd.DataFrame({
            'time': [0.1, 0.2, 0.3],
            '123': [0.5, 0.6, 0.7]  # Numeric column name
        })
        
        with patch('pandas.read_csv', return_value=eda_df):
            # Call with numeric subject ID
            eda_tensor = dataset._load_eda_file("s3://path/to/eda.csv", "123")
            
            # Verify preprocessing was called with the correct data
            mock_preprocess_eda.assert_called_once()
            np.testing.assert_array_equal(
                mock_preprocess_eda.call_args[0][0], 
                eda_df['123'].values
            )
            
            # Verify return value
            assert torch.is_tensor(eda_tensor)
    
    def test_load_eda_file_performance(self, dataset, mock_s3_manager, benchmark):
        """
        GIVEN a need to measure performance of EDA file loading
        WHEN _load_eda_file is called repeatedly
        THEN it should complete within acceptable time limits
        """
        # Skip if benchmark fixture is not available
        pytest.importorskip("pytest_benchmark")
        
        # Create DataFrame with subject data
        eda_df = pd.DataFrame({
            'time': [0.1, 0.2, 0.3],
            'subject1': [0.5, 0.6, 0.7]
        })
        
        # Configure mocks for performance testing
        with patch('pandas.read_csv', return_value=eda_df):
            with patch('src.data.datasets.pmemo2019.preprocess_eda') as mock_preprocess:
                mock_preprocess.return_value = torch.rand(100)
                
                # Define function to benchmark
                def load_eda():
                    return dataset._load_eda_file("s3://path/to/eda.csv", "subject1")
                
                # Run benchmark
                result = benchmark(load_eda)
                
                # Basic verification that function completed successfully
                assert isinstance(result, torch.Tensor)
    
    def test_load_eda_file_concurrent_access(self, dataset_config, feature_config):
        """
        GIVEN multiple threads accessing the same dataset
        WHEN _load_eda_file is called concurrently
        THEN it should handle concurrent access safely
        """
        # Skip if running in an environment where threading is problematic
        import platform
        if platform.system() == "Windows":
            pytest.skip("Skipping thread test on Windows")
        
        # Create a real dataset with mocked dependencies
        with patch('src.data.datasets.pmemo2019.S3FileManager') as mock_s3_manager_class:
            mock_s3_manager_instance = mock_s3_manager_class.return_value
            mock_s3_manager_instance.download_file.return_value = Path("/tmp/test_eda.csv")
            
            # Create DataFrame with subject data
            eda_df = pd.DataFrame({
                'time': [0.1, 0.2, 0.3],
                'subject1': [0.5, 0.6, 0.7],
                'subject2': [0.8, 0.9, 1.0]
            })
            
            with patch('pandas.read_csv', return_value=eda_df):
                with patch('src.data.datasets.pmemo2019.preprocess_eda') as mock_preprocess:
                    mock_preprocess.return_value = torch.rand(100)
                    
                    # Create dataset
                    dataset = PMEmo2019Dataset(dataset_config, feature_config)
                    
                    # Test concurrent access
                    import concurrent.futures
                    import threading
                    
                    # Track successful completions
                    results = []
                    lock = threading.Lock()
                    
                    def load_file(path, subject):
                        try:
                            tensor = dataset._load_eda_file(path, subject)
                            with lock:
                                results.append((path, subject, tensor))
                            return True
                        except Exception:
                            return False
                    
                    # Create different combinations of paths and subjects
                    test_cases = [
                        (f"s3://path/to/eda{i}.csv", f"subject{j}")
                        for i in range(5) for j in range(1, 3)
                    ]
                    
                    # Execute concurrently
                    with concurrent.futures.ThreadPoolExecutor(max_workers=5) as executor:
                        futures = [executor.submit(load_file, path, subject) 
                                  for path, subject in test_cases]
                        concurrent.futures.wait(futures)
                    
                    # Verify all operations completed successfully
                    assert all(future.result() for future in futures)
                    assert len(results) == len(test_cases)
    
    def test_load_eda_file_with_time_column_extraction(self, dataset, mock_s3_manager, mock_preprocess_eda):
        """
        GIVEN a CSV file with a time column
        WHEN _load_eda_file is called
        THEN it should correctly identify the time column and extract the subject data
        """
        # Create DataFrame with time column and subject data
        eda_df = pd.DataFrame({
            'timestamp': [0.1, 0.2, 0.3],  # First column is time
            'subject1': [0.5, 0.6, 0.7]
        })
        
        with patch('pandas.read_csv', return_value=eda_df):
            # Test loading an EDA file
            eda_s3_path = "s3://path/to/eda.csv"
            subject_id = "subject1"
            eda_tensor = dataset._load_eda_file(eda_s3_path, subject_id)
            
            # Verify preprocessing was called with the correct data
            mock_preprocess_eda.assert_called_once()
            np.testing.assert_array_equal(
                mock_preprocess_eda.call_args[0][0], 
                eda_df['subject1'].values
            )
            
            # Verify return value
            assert torch.is_tensor(eda_tensor)
    
    def test_load_eda_file_with_malformed_csv_structure(self, dataset, mock_s3_manager):
        """
        GIVEN a CSV file with unexpected structure (no time column)
        WHEN _load_eda_file is called
        THEN it should handle the error gracefully
        """
        # Create DataFrame with no columns (empty DataFrame)
        empty_df = pd.DataFrame()
        
        with patch('pandas.read_csv', return_value=empty_df):
            # Should raise an IndexError when trying to access columns[0]
            with pytest.raises(IndexError):
                dataset._load_eda_file("s3://path/to/malformed.csv", "subject1")
    
    def test_load_eda_file_with_nan_values(self, dataset, mock_s3_manager, mock_preprocess_eda):
        """
        GIVEN a CSV file with NaN values in the subject column
        WHEN _load_eda_file is called
        THEN it should pass the data with NaNs to preprocess_eda
        """
        # Create DataFrame with NaN values
        eda_df = pd.DataFrame({
            'time': [0.1, 0.2, 0.3],
            'subject1': [0.5, float('nan'), 0.7]  # Include a NaN value
        })
        
        with patch('pandas.read_csv', return_value=eda_df):
            # Test loading an EDA file with NaN values
            dataset._load_eda_file("s3://path/to/eda_with_nans.csv", "subject1")
            
            # Verify preprocessing was called with data containing NaN
            mock_preprocess_eda.assert_called_once()
            assert np.isnan(mock_preprocess_eda.call_args[0][0][1])  # Second value should be NaN

class TestAudioFileLoading:
    """Tests specifically focused on audio file loading functionality."""
    
    @pytest.fixture
    def dataset(self, mock_s3_manager, dataset_config, feature_config):
        """
        Provides a standard PMEmo2019Dataset instance.
        
        Returns:
            PMEmo2019Dataset: Configured dataset instance
        """
        with patch.object(PMEmo2019Dataset, '_eda_files', {}):
            dataset = PMEmo2019Dataset(dataset_config, feature_config)
            return dataset
    
    def test_load_audio_file_performance(self, dataset, mock_s3_manager, benchmark):
        """
        GIVEN a need to measure performance of audio file loading
        WHEN _load_audio_file is called repeatedly
        THEN it should complete within acceptable time limits
        """
        # Skip if benchmark fixture is not available
        pytest.importorskip("pytest_benchmark")
        
        # Configure mocks for performance testing
        with patch('src.data.datasets.pmemo2019.preprocess_audio') as mock_preprocess:
            mock_preprocess.return_value = torch.rand(5, 100)
            
            # Define function to benchmark
            def load_audio():
                return dataset._load_audio_file("s3://audio2biosignal-train-data/PMEmo2019/chorus/test.mp3")
            
            # Run benchmark
            result = benchmark(load_audio)
            
            # Basic verification that function completed successfully
            assert isinstance(result, torch.Tensor)
    
    def test_load_audio_file_with_corrupted_local_file(self, dataset, mock_s3_manager):
        """
        GIVEN a scenario where the downloaded file is corrupted
        WHEN _load_audio_file is called
        THEN it should handle the error appropriately
        """
        # Configure S3 manager to return a path
        mock_s3_manager.download_file.return_value = Path("/tmp/corrupted_audio.mp3")
        
        # Configure preprocess_audio to simulate file corruption
        with patch('src.data.datasets.pmemo2019.preprocess_audio') as mock_preprocess:
            mock_preprocess.side_effect = IOError("Corrupted audio file")
            
            # Verify IOError is propagated
            with pytest.raises(IOError, match="Corrupted audio file"):
                dataset._load_audio_file("s3://audio2biosignal-train-data/PMEmo2019/chorus/corrupted.mp3")
    
    def test_load_audio_file_with_very_large_file(self, dataset, mock_s3_manager):
        """
        GIVEN a very large audio file
        WHEN _load_audio_file is called
        THEN it should process it correctly without memory issues
        """
        # Configure preprocess_audio to return a large tensor
        with patch('src.data.datasets.pmemo2019.preprocess_audio') as mock_preprocess:
            # Create a relatively large tensor (but not too large for testing)
            mock_preprocess.return_value = torch.rand(5, 10000)  # 5 features, 10000 time steps
            
            # Load the "large" audio file
            audio_tensor = dataset._load_audio_file("s3://audio2biosignal-train-data/PMEmo2019/chorus/large.mp3")
            
            # Verify tensor dimensions
            assert audio_tensor.shape == (5, 10000)
            
            # Verify memory usage is reasonable (basic check)
            import sys
            tensor_size_bytes = audio_tensor.element_size() * audio_tensor.nelement()
            assert tensor_size_bytes < 10 * 1024 * 1024  # Less than 10MB
    
    def test_load_audio_file_concurrent_access(self, dataset_config, feature_config):
        """
        GIVEN multiple threads accessing the same dataset
        WHEN _load_audio_file is called concurrently
        THEN it should handle concurrent access safely
        """
        # Skip if running in an environment where threading is problematic
        import platform
        if platform.system() == "Windows":
            pytest.skip("Skipping thread test on Windows")
        
        # Create a real dataset with mocked dependencies
        with patch('src.data.datasets.pmemo2019.S3FileManager') as mock_s3_manager_class:
            mock_s3_manager_instance = mock_s3_manager_class.return_value
            mock_s3_manager_instance.download_file.return_value = Path("/tmp/test_audio.mp3")
            
            with patch('src.data.datasets.pmemo2019.preprocess_audio') as mock_preprocess:
                mock_preprocess.return_value = torch.rand(5, 100)
                
                # Create dataset
                dataset = PMEmo2019Dataset(dataset_config, feature_config)
                
                # Test concurrent access
                import concurrent.futures
                import threading
                
                # Track successful completions
                results = []
                lock = threading.Lock()
                
                def load_file(path):
                    try:
                        tensor = dataset._load_audio_file(path)
                        with lock:
                            results.append(tensor)
                        return True
                    except Exception:
                        return False
                
                # Create different paths to avoid caching effects
                paths = [f"s3://audio2biosignal-train-data/PMEmo2019/chorus/test{i}.mp3" for i in range(10)]
                
                # Execute concurrently
                with concurrent.futures.ThreadPoolExecutor(max_workers=5) as executor:
                    futures = [executor.submit(load_file, path) for path in paths]
                    concurrent.futures.wait(futures)
                
                # Verify all operations completed successfully
                assert all(future.result() for future in futures)
                assert len(results) == 10

class TestCollateFn:
    """Tests for the PMEmo2019 dataset collate_fn function."""

    @pytest.fixture
    def standard_batch(self):
        """
        Provides a standard batch of audio and EDA tensors.
        
        Returns:
            list: List of (audio_tensor, eda_tensor) tuples
        """
        # Create 3 examples with different lengths
        return [
            (torch.rand(5, 100), torch.rand(100)),  # 5 features, length 100
            (torch.rand(5, 150), torch.rand(150)),  # 5 features, length 150
            (torch.rand(5, 80), torch.rand(80)),    # 5 features, length 80
        ]

    def test_basic_collate(self, standard_batch):
        """
        GIVEN a batch of audio and EDA tensors with different lengths
        WHEN collate_fn is called
        THEN it should return properly padded tensors
        """
        # Call the collate function
        padded_audio, padded_eda = collate_fn(standard_batch)
        
        # Check shapes
        assert padded_audio.shape == (3, 5, 150)  # batch_size, features, max_length
        assert padded_eda.shape == (3, 150)       # batch_size, max_length
        
        # Check types
        assert isinstance(padded_audio, torch.Tensor)
        assert isinstance(padded_eda, torch.Tensor)

    def test_padding_alignment(self, standard_batch):
        """
        GIVEN a batch of tensors with different lengths
        WHEN collate_fn is called
        THEN the original data should be aligned to the right in the padded tensors
        """
        # Get original tensors for comparison
        original_audio = standard_batch[0][0]
        original_eda = standard_batch[0][1]
        original_length = original_audio.size(1)
        
        # Call the collate function
        padded_audio, padded_eda = collate_fn(standard_batch)
        
        # Check that the original data is preserved at the right end of the padded tensor
        max_length = padded_audio.size(2)
        padding_offset = max_length - original_length
        
        # Check audio tensor alignment
        assert torch.all(torch.eq(
            padded_audio[0, :, padding_offset:],
            original_audio
        ))
        
        # Check EDA tensor alignment
        assert torch.all(torch.eq(
            padded_eda[0, padding_offset:],
            original_eda
        ))
        
        # Check that padding (zeros) is applied to the left
        assert torch.all(padded_audio[0, :, :padding_offset] == 0)
        assert torch.all(padded_eda[0, :padding_offset] == 0)

    def test_empty_batch(self):
        """
        GIVEN an empty batch
        WHEN collate_fn is called
        THEN it should raise an appropriate error
        """
        with pytest.raises(IndexError):
            collate_fn([])

    def test_single_item_batch(self):
        """
        GIVEN a batch with a single item
        WHEN collate_fn is called
        THEN it should return properly shaped tensors
        """
        single_item = [(torch.rand(5, 100), torch.rand(100))]
        
        padded_audio, padded_eda = collate_fn(single_item)
        
        # Check shapes
        assert padded_audio.shape == (1, 5, 100)
        assert padded_eda.shape == (1, 100)
        
        # Check that the original data is preserved
        assert torch.all(torch.eq(padded_audio[0], single_item[0][0]))
        assert torch.all(torch.eq(padded_eda[0], single_item[0][1]))

    def test_different_feature_dimensions(self):
        """
        GIVEN tensors with inconsistent feature dimensions
        WHEN collate_fn is called
        THEN it should raise an appropriate error
        """
        inconsistent_batch = [
            (torch.rand(5, 100), torch.rand(100)),
            (torch.rand(7, 150), torch.rand(150)),  # Different feature dimension
        ]
        
        with pytest.raises(RuntimeError, match="inconsistent"):
            collate_fn(inconsistent_batch)

    def test_zero_length_tensors(self):
        """
        GIVEN tensors with zero length
        WHEN collate_fn is called
        THEN it should handle them appropriately
        """
        zero_length_batch = [
            (torch.rand(5, 0), torch.rand(0)),
            (torch.rand(5, 10), torch.rand(10)),
        ]
        
        padded_audio, padded_eda = collate_fn(zero_length_batch)
        
        # Check shapes
        assert padded_audio.shape == (2, 5, 10)
        assert padded_eda.shape == (2, 10)
        
        # Check that the first item is all zeros
        assert torch.all(padded_audio[0] == 0)
        assert torch.all(padded_eda[0] == 0)

    def test_large_size_differences(self):
        """
        GIVEN tensors with very different sizes
        WHEN collate_fn is called
        THEN it should handle the padding efficiently
        """
        batch = [
            (torch.rand(5, 10), torch.rand(10)),      # Very short
            (torch.rand(5, 1000), torch.rand(1000)),  # Very long
        ]
        
        padded_audio, padded_eda = collate_fn(batch)
        
        # Check shapes
        assert padded_audio.shape == (2, 5, 1000)
        assert padded_eda.shape == (2, 1000)
        
        # Check that the short tensor is properly padded
        assert torch.all(padded_audio[0, :, :990] == 0)  # First 990 elements should be 0
        assert torch.all(padded_eda[0, :990] == 0)       # First 990 elements should be 0
        
        # Check that the long tensor is preserved
        assert torch.all(torch.eq(padded_audio[1], batch[1][0]))
        assert torch.all(torch.eq(padded_eda[1], batch[1][1]))

    def test_memory_efficiency(self):
        """
        GIVEN a large batch of tensors
        WHEN collate_fn is called
        THEN it should not use excessive memory
        """
        # Create a large batch (this is a basic check, not a precise measurement)
        large_batch = [
            (torch.rand(5, 1000), torch.rand(1000)) for _ in range(32)
        ]
        
        # This should complete without OOM errors
        padded_audio, padded_eda = collate_fn(large_batch)
        
        # Basic shape check
        assert padded_audio.shape == (32, 5, 1000)
        assert padded_eda.shape == (32, 1000)
    
    def test_mismatched_audio_eda_lengths(self):
        """
        GIVEN audio and EDA tensors with different lengths
        WHEN collate_fn is called
        THEN it should handle the mismatch appropriately
        """
        # Create batch with mismatched audio and EDA lengths
        mismatched_batch = [
            (torch.rand(5, 100), torch.rand(80)),  # EDA shorter than audio
            (torch.rand(5, 120), torch.rand(150))  # EDA longer than audio
        ]
        
        padded_audio, padded_eda = collate_fn(mismatched_batch)
        
        # Check shapes - should use max length of each type
        assert padded_audio.shape == (2, 5, 120)
        assert padded_eda.shape == (2, 150)
        
        # Verify padding was applied correctly
        assert torch.all(padded_audio[0, :, -100:] == mismatched_batch[0][0])
        assert torch.all(padded_eda[0, -80:] == mismatched_batch[0][1])
    
    def test_device_preservation(self):
        """
        GIVEN tensors on specific devices
        WHEN collate_fn is called
        THEN it should preserve the device of input tensors
        """
        # Skip test if CUDA is not available
        if not torch.cuda.is_available():
            pytest.skip("CUDA not available, skipping device test")
            
        # Create tensors on CUDA device
        cuda_batch = [
            (torch.rand(5, 100, device="cuda"), torch.rand(100, device="cuda")),
            (torch.rand(5, 150, device="cuda"), torch.rand(150, device="cuda"))
        ]
        
        padded_audio, padded_eda = collate_fn(cuda_batch)
        
        # Check that output tensors are on the same device
        assert padded_audio.device.type == "cuda"
        assert padded_eda.device.type == "cuda"
    
    def test_dtype_preservation(self):
        """
        GIVEN tensors with specific dtypes
        WHEN collate_fn is called
        THEN it should preserve the dtype of input tensors
        """
        # Create tensors with float64 dtype
        float64_batch = [
            (torch.rand(5, 100, dtype=torch.float64), torch.rand(100, dtype=torch.float64)),
            (torch.rand(5, 150, dtype=torch.float64), torch.rand(150, dtype=torch.float64))
        ]
        
        padded_audio, padded_eda = collate_fn(float64_batch)
        
        # Check that output tensors have the same dtype
        assert padded_audio.dtype == torch.float64
        assert padded_eda.dtype == torch.float64
