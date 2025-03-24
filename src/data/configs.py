from enum import Enum
from pydantic import BaseModel
from typing import List, Dict, Optional

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

class DatasetConfig(BaseModel):
    """
    Base configuration class for dataset management in multimodal biosignal processing.
    
    Provides a standardized interface for dataset access, defining the structure,
    location, and processing parameters for multimodal datasets (primarily audio and EDA).
    Serves as the foundation for dataset-specific configurations through inheritance.
    
    Architecture:
        - Implements a declarative configuration pattern using Pydantic BaseModel
        - Supports S3-based remote storage with path resolution
        - Enables consistent dataset splitting for reproducible machine learning experiments
    
    Attributes:
        dataset_name (str): Unique identifier for the dataset
        dataset_root_path (str): Base path to dataset storage (local or S3 URI)
        modalities (List[str]): Available data modalities (e.g., ["eda", "audio"])
        file_format (Dict[str, str]): File extensions for each modality (e.g., {"audio": ".mp3"})
        data_directories (Dict[str, str]): Storage paths for each modality's data
        metadata_paths (List[str]): Paths to metadata files containing annotations or mappings
        split_ratios (List[float]): Train/validation/test split proportions (should sum to 1.0)
        seed (int): Random seed for reproducible dataset splitting
    
    Integration:
        - Used by DataLoaderBuilder to construct PyTorch DataLoader instances
        - Consumed by dataset classes (e.g., HKU956Dataset, PMEmo2019Dataset)
        - Extended by dataset-specific configurations (HKU956Config, PMEmo2019Config)
    
    Example:
        ```python
        config = DatasetConfig(
            dataset_name="CustomDataset",
            dataset_root_path="s3://my-bucket/datasets/custom/",
            modalities=["audio", "eda"],
            file_format={"audio": ".wav", "eda": ".csv"},
            data_directories={
                "audio": "s3://my-bucket/datasets/custom/audio/",
                "eda": "s3://my-bucket/datasets/custom/biosignals/"
            },
            metadata_paths=["s3://my-bucket/datasets/custom/metadata.csv"],
            split_ratios=[0.8, 0.1, 0.1],
            seed=42
        )
        ```
    
    Limitations:
        - No validation for consistency between modalities and file_format/data_directories
        - No validation that split_ratios sum to 1.0
        - S3 paths require appropriate AWS credentials in the environment
    """
    dataset_name: str
    dataset_root_path: str
    modalities: List[str]
    file_format: Dict[str, str]
    data_directories: Dict[str, str]
    metadata_paths: List[str]
    split_ratios: List[float]
    seed: int

class HKU956Config(DatasetConfig):
    """
    Configuration for the HKU956 dataset, providing standardized access to music-evoked physiological signals.
    
    HKU956Config encapsulates the structure and location of the HKU956 dataset, which contains
    956 samples of synchronized audio and electrodermal activity (EDA) recordings from subjects
    listening to music. It defines S3 paths, file formats, and dataset organization to enable
    reproducible experiments with this multimodal dataset.
    
    Architecture:
        - Inherits from DatasetConfig, implementing a concrete configuration for HKU956
        - Maintains fixed S3 paths to standardize access across experiments
        - Structures data access by modality (audio/EDA) with predefined directory mappings
        - Supports 80/10/10 train/validation/test splitting with fixed random seed
    
    Integration:
        - Used by HKU956Dataset to locate and load dataset files from S3
        - Consumed by DataLoaderBuilder to construct PyTorch DataLoader instances
        - Compatible with AudioEDAFeatureConfig for signal preprocessing parameters
        - Provides empty metadata_paths as HKU956 uses directory structure for organization
    
    Example:
        ```python
        # Standard usage with default parameters
        config = HKU956Config()
        dataset = HKU956Dataset(config, feature_config)
        
        # Custom configuration with modified split ratios
        custom_config = HKU956Config(
            split_ratios=[0.7, 0.15, 0.15],
            seed=100
        )
        ```
    
    Limitations:
        - Fixed to specific S3 bucket structure (audio2biosignal-train-data)
        - Audio path points to a CSV file containing audio file references
        - No validation for S3 path accessibility or file existence
        - Requires S3FileManager utility for downloading files from S3
    """
    dataset_name: str = "HKU956"
    dataset_root_path: str = "s3://audio2biosignal-train-data/HKU956/"
    modalities: List[str] = ["eda", "audio"]
    file_format: Dict[str, str] = {
        "eda": ".csv",
        "audio": ".mp3"
    }
    data_directories: Dict[str, str] = {
        "eda": "s3://audio2biosignal-train-data/HKU956/1. physiological_signals/",
        "audio": "s3://audio2biosignal-train-data/HKU956/2. original_song_audio.csv"
    }
    metadata_paths: List[str] = []
    split_ratios: List[float] = [0.8, 0.1, 0.1]
    seed: int = 42

class PMEmo2019Config(DatasetConfig):
    """
    Configuration for the PMEmo2019 dataset, providing standardized access to music-evoked physiological signals.
    
    PMEmo2019Config encapsulates the structure and location of the PMEmo2019 dataset, which contains
    synchronized audio and electrodermal activity (EDA) recordings from multiple subjects listening to
    music excerpts. It defines S3 paths, file formats, and dataset organization to enable reproducible
    experiments with this multimodal dataset.
    
    Architecture:
        - Inherits from DatasetConfig, implementing a concrete configuration for PMEmo2019
        - Maintains fixed S3 paths to standardize access across experiments
        - Structures data access by modality (audio/EDA) with predefined directory mappings
        - Supports 80/10/10 train/validation/test splitting with fixed random seed
    
    Integration:
        - Used by PMEmo2019Dataset to locate and load dataset files from S3
        - Consumed by DataLoaderBuilder to construct PyTorch DataLoader instances
        - Provides metadata path for subject-music-EDA mappings required by the dataset
        - Compatible with AudioEDAFeatureConfig for signal preprocessing parameters
    
    Example:
        ```python
        # Standard usage with default parameters
        config = PMEmo2019Config()
        dataset = PMEmo2019Dataset(config, feature_config)
        
        # Custom configuration with modified split ratios
        custom_config = PMEmo2019Config(
            split_ratios=[0.7, 0.15, 0.15],
            seed=100
        )
        ```
    
    Limitations:
        - Fixed to specific S3 bucket structure (audio2biosignal-train-data)
        - Assumes chorus excerpts for audio files rather than full songs
        - Requires metadata.csv file in the expected S3 location
        - No validation for S3 path accessibility or file existence
    """
    dataset_name: str = "PMEmo2019"
    dataset_root_path: str = "s3://audio2biosignal-train-data/PMEmo2019/"
    modalities: List[str] = ["eda", "audio"]
    file_format: Dict[str, str] = {
        "eda": ".csv",
        "audio": ".mp3"
    }
    data_directories: Dict[str, str] = {
        "eda": "s3://audio2biosignal-train-data/PMEmo2019/EDA/",
        "audio": "s3://audio2biosignal-train-data/PMEmo2019/chorus"
    }
    metadata_paths: List[str] = ["s3://audio2biosignal-train-data/PMEmo2019/metadata.csv"]
    split_ratios: List[float] = [0.8, 0.1, 0.1]
    seed: int = 42

class AudioEDAFeatureConfig(BaseModel):
    """
    Configuration for audio and electrodermal activity (EDA) signal preprocessing in biosignal modeling.
    
    AudioEDAFeatureConfig encapsulates the complete specification of signal processing parameters
    for both audio and EDA modalities, ensuring consistent preprocessing across dataset loading,
    feature extraction, and model training. It provides unified control over sampling rates,
    normalization, and time-frequency transformations with sensible defaults for biosignal analysis.
    
    Architecture:
        - Implements a declarative configuration pattern using Pydantic BaseModel
        - Maintains modality-specific parameter groups with coordinated defaults
        - Supports synchronized time-domain representations through aligned window sizes
        - Enables independent control of filtering strategies for each modality
    
    Attributes:
        # Audio processing parameters
        audio_target_sample_rate (Optional[int]): Target sample rate in Hz for audio (default: None, no resampling)
        audio_normalize (bool): Whether to normalize audio amplitude to [-1, 1] (default: True)
        audio_n_mfcc (int): Number of Mel-frequency cepstral coefficients (default: 40)
        audio_n_mels (int): Number of Mel filterbank features (default: 128)
        audio_window_size (int): STFT window size in samples (default: 400)
        audio_hop_length (int): STFT hop length in samples (default: 160)
        
        # EDA processing parameters
        eda_target_sample_rate (Optional[int]): Target sample rate in Hz for EDA (default: None, no resampling)
        eda_window_size (int): Analysis window size in samples (default: 400)
        eda_hop_length (int): Analysis hop length in samples (default: 160)
        eda_normalize (bool): Whether to normalize EDA signals (default: True)
        filter_lowpass (bool): Apply 8Hz low-pass filter to EDA (default: True)
        filter_highpass (bool): Apply 0.05Hz high-pass filter to EDA (default: False)
    
    Integration:
        - Used by dataset classes to configure signal preprocessing
        - Consumed by audio_preprocessing and eda_preprocessing modules
        - Passed to dataset constructors alongside dataset-specific configs
        - Controls feature extraction dimensionality and temporal resolution
    
    Example:
        ```python
        # Standard configuration with default parameters
        feature_config = AudioEDAFeatureConfig()
        
        # Custom configuration for higher resolution processing
        custom_config = AudioEDAFeatureConfig(
            audio_target_sample_rate=500,
            eda_target_sample_rate=500,
            audio_n_mfcc=60,
            audio_n_mels=256,
            filter_highpass=True
        )
        
        # Use with dataset initialization
        dataset = PMEmo2019Dataset(dataset_config, feature_config)
        ```
    
    Limitations:
        - No validation for parameter value ranges or relationships
        - Window sizes must be manually coordinated for temporal alignment
        - Fixed filter cutoff frequencies (8Hz lowpass, 0.05Hz highpass)
        - No support for advanced audio features (e.g., spectral contrast, chroma)
    """
    # Audio configurations
    audio_target_sample_rate: Optional[int] = None  # Hz, None means no resampling
    audio_normalize: bool = True
    audio_n_mfcc: int = 40
    audio_n_mels: int = 128
    audio_window_size: int = 400  # STFT window size
    audio_hop_length: int = 160   # STFT hop length

    # EDA configurations
    eda_target_sample_rate: Optional[int] = None  # Hz, None means no resampling
    eda_window_size: int = 400
    eda_hop_length: int = 160
    eda_normalize: bool = True
    filter_lowpass: bool = True   # 8Hz low-pass filter
    filter_highpass: bool = False  # 0.05Hz high-pass filter
