from enum import Enum

class DatasetType(Enum):
    """
    Enumeration of supported datasets for audio-to-biosignal modeling.
    
    DatasetType provides a type-safe registry of available datasets, ensuring consistent
    identification across configuration, data loading, and model training components.
    Each enum member maps a symbolic name to a string identifier used in configuration files.
    
    Architecture:
        - Implements Python's Enum pattern for type safety and IDE autocompletion
        - String values match dataset directory names in S3 storage
        - Hashable for use as dictionary keys in dataset mappings
    
    Members:
        HKU956: The HKU music-evoked physiological dataset with 956 samples
        PMEmo2019: The PMEmo dataset (2019) with continuous emotion annotations
    
    Integration:
        - Used by DataConfig to specify datasets for training/validation/testing
        - Consumed by DataLoaderBuilder to instantiate appropriate dataset classes
        - String values (e.g., 'hku956') used in YAML configuration files
        - Can be converted from string with DatasetType(str_value)
    
    Example:
        ```python
        # Type-safe dataset specification
        config = DataConfig(
            train_datasets=[DatasetType.HKU956],
            val_datasets=[DatasetType.PMEmo2019],
            test_datasets=[DatasetType.HKU956, DatasetType.PMEmo2019]
        )
        
        # String conversion for configuration parsing
        dataset_type = DatasetType('hku956')  # Returns DatasetType.HKU956
        ```
    
    Limitations:
        - Adding new datasets requires code changes to this enum
        - No validation that dataset identifiers exist in storage
        - No metadata about dataset characteristics (size, modalities, etc.)
    """
    HKU956 = 'hku956'
    PMEmo2019 = 'pmemo2019'
