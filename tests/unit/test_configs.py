"""
Tests for configuration components in the audio2biosignal project.

This module provides comprehensive testing for:
1. DatasetType enum values and behavior
2. Enum integration with other components
3. Edge cases and error handling
"""

import pytest
from enum import Enum
import sys
import os
from typing import Dict, Any

# Add the project root to the path so we can import the src modules
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from src.data.datasets.types import DatasetType


class TestDatasetType:
    """Test suite for the DatasetType enum."""

    def test_enum_values(self):
        """
        GIVEN the DatasetType enum
        WHEN accessing its values
        THEN it should contain the expected dataset identifiers
        """
        # Verify all expected datasets are defined
        assert hasattr(DatasetType, 'HKU956')
        assert hasattr(DatasetType, 'PMEmo2019')
        
        # Verify the string values
        assert DatasetType.HKU956.value == 'hku956'
        assert DatasetType.PMEmo2019.value == 'pmemo2019'
        
        # Verify the total number of datasets
        assert len(DatasetType) == 2, f"Expected 2 dataset types, found {len(DatasetType)}"

    def test_enum_equality(self):
        """
        GIVEN the DatasetType enum
        WHEN comparing enum members
        THEN they should follow expected equality behavior
        """
        # Same enum members should be equal
        assert DatasetType.HKU956 == DatasetType.HKU956
        assert DatasetType.PMEmo2019 == DatasetType.PMEmo2019
        
        # Different enum members should not be equal
        assert DatasetType.HKU956 != DatasetType.PMEmo2019
        
        # Enum members should not equal their string values
        assert DatasetType.HKU956 != 'hku956'
        assert DatasetType.PMEmo2019 != 'pmemo2019'

    def test_enum_identity(self):
        """
        GIVEN the DatasetType enum
        WHEN checking identity of enum members
        THEN they should be singletons
        """
        # Enum members should be singletons (same object)
        assert DatasetType.HKU956 is DatasetType.HKU956
        assert DatasetType.PMEmo2019 is DatasetType.PMEmo2019
        
        # Different enum members should be different objects
        assert DatasetType.HKU956 is not DatasetType.PMEmo2019

    def test_enum_from_string(self):
        """
        GIVEN the DatasetType enum
        WHEN creating enum members from strings
        THEN it should correctly map to the appropriate enum member
        """
        # Test valid string conversions
        assert DatasetType('hku956') == DatasetType.HKU956
        assert DatasetType('pmemo2019') == DatasetType.PMEmo2019
        
        # Test case sensitivity (should be case-sensitive)
        with pytest.raises(ValueError):
            DatasetType('HKU956')
        
        with pytest.raises(ValueError):
            DatasetType('PMEMO2019')
        
        # Test invalid dataset name
        with pytest.raises(ValueError) as excinfo:
            DatasetType('nonexistent_dataset')
        assert "nonexistent_dataset" in str(excinfo.value)

    def test_enum_iteration(self):
        """
        GIVEN the DatasetType enum
        WHEN iterating over its members
        THEN it should include all defined datasets
        """
        datasets = list(DatasetType)
        assert len(datasets) == 2
        assert DatasetType.HKU956 in datasets
        assert DatasetType.PMEmo2019 in datasets

    def test_enum_in_dataloader_mapping(self):
        """
        GIVEN the DatasetType enum
        WHEN using its values as keys in the DataLoaderBuilder mapping
        THEN the keys should match the expected string values
        """
        # This test verifies that the enum values match what's expected in DataLoaderBuilder
        dataset_mapping_keys = ['hku956', 'pmemo2019']
        
        for dataset_type in DatasetType:
            assert dataset_type.value in dataset_mapping_keys, \
                f"Dataset type {dataset_type.value} not found in expected mapping keys"

    def test_enum_serialization(self):
        """
        GIVEN the DatasetType enum
        WHEN serializing enum members to strings
        THEN it should produce the expected string representations
        """
        # Test string conversion
        assert str(DatasetType.HKU956) == 'DatasetType.HKU956'
        assert str(DatasetType.PMEmo2019) == 'DatasetType.PMEmo2019'
        
        # Test value access
        assert DatasetType.HKU956.value == 'hku956'
        assert DatasetType.PMEmo2019.value == 'pmemo2019'
        
        # Test repr
        assert repr(DatasetType.HKU956) == 'DatasetType.HKU956'
        assert repr(DatasetType.PMEmo2019) == 'DatasetType.PMEmo2019'

    def test_enum_in_config(self):
        """
        GIVEN the DatasetType enum
        WHEN used in a DataConfig object
        THEN it should be properly recognized and processed
        """
        from src.configs import DataConfig
        
        # Create a simple config with both dataset types
        config = DataConfig(
            train_datasets=[DatasetType.HKU956],
            val_datasets=[DatasetType.PMEmo2019],
            test_datasets=[DatasetType.HKU956, DatasetType.PMEmo2019],
            num_workers=4,
            prefetch_size=2
        )
        
        # Verify the datasets were properly stored
        assert len(config.train_datasets) == 1
        assert config.train_datasets[0] == DatasetType.HKU956
        
        assert len(config.val_datasets) == 1
        assert config.val_datasets[0] == DatasetType.PMEmo2019
        
        assert len(config.test_datasets) == 2
        assert DatasetType.HKU956 in config.test_datasets
        assert DatasetType.PMEmo2019 in config.test_datasets
    
    def test_enum_name_and_value_access(self):
        """
        GIVEN the DatasetType enum
        WHEN accessing name and value attributes
        THEN it should return the correct properties
        """
        # Test name property
        assert DatasetType.HKU956.name == 'HKU956'
        assert DatasetType.PMEmo2019.name == 'PMEmo2019'
        
        # Test direct value access
        assert DatasetType.HKU956.value == 'hku956'
        assert DatasetType.PMEmo2019.value == 'pmemo2019'
        
        # Test __str__ and __repr__
        assert str(DatasetType.HKU956) == 'DatasetType.HKU956'
        assert repr(DatasetType.HKU956) == 'DatasetType.HKU956'
    
    def test_enum_in_dictionary(self):
        """
        GIVEN the DatasetType enum
        WHEN used as dictionary keys
        THEN it should behave correctly
        """
        # Create a dictionary with enum keys
        dataset_config = {
            DatasetType.HKU956: {"sample_rate": 16000, "channels": 1},
            DatasetType.PMEmo2019: {"sample_rate": 44100, "channels": 2}
        }
        
        # Test dictionary access with enum keys
        assert dataset_config[DatasetType.HKU956]["sample_rate"] == 16000
        assert dataset_config[DatasetType.PMEmo2019]["channels"] == 2
        
        # Test dictionary membership
        assert DatasetType.HKU956 in dataset_config
        assert DatasetType.PMEmo2019 in dataset_config
        
        # Test dictionary iteration
        keys = list(dataset_config.keys())
        assert len(keys) == 2
        assert DatasetType.HKU956 in keys
        assert DatasetType.PMEmo2019 in keys
