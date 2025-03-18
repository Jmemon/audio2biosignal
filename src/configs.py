from enum import Enum
from pydantic import BaseModel, validator, field_validator
from typing import List, Dict, Optional, Literal, Union, Any

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
        mutual_sample_rate (int): Target sample rate in Hz for both modalities (default: 200)
        
        # Audio processing parameters
        audio_normalize (bool): Whether to normalize audio amplitude to [-1, 1] (default: True)
        audio_n_mfcc (int): Number of Mel-frequency cepstral coefficients (default: 40)
        audio_n_mels (int): Number of Mel filterbank features (default: 128)
        audio_window_size (int): STFT window size in samples (default: 400)
        audio_hop_length (int): STFT hop length in samples (default: 160)
        
        # EDA processing parameters
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
            mutual_sample_rate=500,
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
    # Mutual configurations
    mutual_sample_rate: int = 200  # Hz

    # Audio configurations
    audio_normalize: bool = True
    audio_n_mfcc: int = 40
    audio_n_mels: int = 128
    audio_window_size: int = 400  # STFT window size
    audio_hop_length: int = 160   # STFT hop length

    # EDA configurations
    eda_window_size: int = 400
    eda_hop_length: int = 160
    eda_normalize: bool = True
    filter_lowpass: bool = True   # 8Hz low-pass filter
    filter_highpass: bool = False  # 0.05Hz high-pass filter

class DataConfig(BaseModel):
    """
    Configuration for dataset selection and data loading parameters in audio-to-biosignal modeling.
    
    DataConfig manages the specification of which datasets to use for each training phase
    (train/validation/test) and controls the data loading process parameters. It serves as
    the central configuration point for dataset selection and PyTorch DataLoader behavior.
    
    Architecture:
        - Implements a declarative configuration pattern using Pydantic BaseModel
        - Uses DatasetType enum for type-safe dataset selection
        - Supports multi-dataset training with separate dataset lists for each phase
        - Controls parallelism and memory management for data loading pipeline
    
    Attributes:
        train_datasets (List[DatasetType]): Datasets to use for training
        val_datasets (List[DatasetType]): Datasets to use for validation
        test_datasets (List[DatasetType]): Datasets to use for testing
        batch_size (int): Number of samples per batch (default: 32)
        num_workers (int): Number of subprocesses for data loading (default: 4)
        prefetch_size (int): Number of batches to prefetch (default: 2)
    
    Integration:
        - Used by DataLoaderBuilder to construct PyTorch DataLoader instances
        - Consumed by training scripts to configure dataset selection
        - Works with dataset-specific configurations (HKU956Config, PMEmo2019Config)
        - Batch size affects memory usage and gradient accumulation strategy
    
    Example:
        ```python
        config = DataConfig(
            train_datasets=[DatasetType.HKU956],
            val_datasets=[DatasetType.PMEmo2019],
            test_datasets=[DatasetType.HKU956, DatasetType.PMEmo2019],
            batch_size=64,
            num_workers=8,
            prefetch_size=4
        )
        train_loaders = DataLoaderBuilder.build(config, feature_config, 'train')
        ```
    
    Limitations:
        - No validation for dataset existence or compatibility
        - No automatic adjustment of num_workers based on available CPU cores
        - No memory usage estimation based on batch_size
        - Requires DatasetType enum to be updated when adding new datasets
    """
    train_datasets: List[DatasetType]
    val_datasets: List[DatasetType]
    test_datasets: List[DatasetType]
    batch_size: int = 32
    num_workers: int = 4
    prefetch_size: int = 2

class OptimizerConfig(BaseModel):
    """
    Configuration for optimizer selection and hyperparameters in audio-to-biosignal modeling.
    
    OptimizerConfig encapsulates the complete specification of optimization algorithms,
    learning rate schedules, and gradient handling for neural network training. It provides
    a unified interface for configuring the optimization process with sensible defaults
    while allowing fine-grained control over training dynamics.
    
    Architecture:
        - Implements a declarative configuration pattern using Pydantic BaseModel
        - Supports multiple optimizer types with algorithm-specific parameters
        - Integrates learning rate scheduling and gradient clipping in a unified interface
        - Maintains type safety through Literal types for constrained parameter choices
    
    Attributes:
        name (Literal["adam", "adamw", "sgd"]): Optimizer algorithm selection (default: "adamw")
        learning_rate (float): Initial learning rate for optimization (default: 1e-4)
        weight_decay (float): L2 regularization strength (default: 0.01)
        beta1 (float): Exponential decay rate for first moment estimates in Adam/AdamW (default: 0.9)
        beta2 (float): Exponential decay rate for second moment estimates in Adam/AdamW (default: 0.999)
        momentum (float): Momentum factor for SGD optimizer (default: 0.0, only used with SGD)
        warmup_steps (int): Number of steps for learning rate warmup (default: 0)
        warmup_ratio (float): Proportion of training steps for learning rate warmup (default: 0.0)
        scheduler (Optional[Literal["cosine", "linear", "constant", "reduce_on_plateau"]]): 
            Learning rate scheduler type (default: "cosine")
        gradient_clip_val (float): Maximum allowed gradient norm for gradient clipping (default: 1.0)
    
    Integration:
        - Used by OptimizerBuilder to construct PyTorch optimizer and scheduler instances
        - Consumed by training scripts to configure optimization strategy
        - Supports different parameter sets for different optimizer types (Adam/AdamW/SGD)
        - Gradient clipping value directly configures gradient norm constraints
    
    Example:
        ```python
        # AdamW with cosine learning rate schedule
        config = OptimizerConfig(
            name="adamw",
            learning_rate=0.001,
            weight_decay=0.01,
            scheduler="cosine"
        )
        
        # SGD with momentum and no scheduler
        config = OptimizerConfig(
            name="sgd",
            learning_rate=0.01,
            momentum=0.9,
            scheduler=None
        )
        
        # Create optimizer and scheduler
        optimizer, scheduler = OptimizerBuilder.build(config, model.parameters())
        ```
    
    Limitations:
        - Limited to three optimizer types (Adam, AdamW, SGD)
        - Not all schedulers are fully implemented in OptimizerBuilder
        - No support for parameter-group-specific learning rates
        - Warmup implementation depends on the training loop integration
    """
    name: Literal["adam", "adamw", "sgd"] = "adamw"
    learning_rate: float = 1e-4
    weight_decay: float = 0.01
    beta1: float = 0.9
    beta2: float = 0.999
    momentum: float = 0.0  # Only used for SGD
    warmup_steps: int = 0
    warmup_ratio: float = 0.0
    scheduler: Optional[Literal["cosine", "linear", "constant", "reduce_on_plateau"]] = "cosine"
    gradient_clip_val: float = 1.0

class LossConfig(BaseModel):
    name: Literal["mse", "l1", "huber", "custom"] = "mse"

class ModelConfig(BaseModel):
    """
    Configuration for neural network architecture selection and hyperparameters in audio-to-biosignal modeling.
    
    ModelConfig encapsulates the complete specification of neural network architectures
    and their hyperparameters, providing a unified interface for model instantiation with
    architecture-specific validation. It ensures that all required parameters for each
    supported architecture are provided while allowing flexible extension.
    
    Architecture:
        - Implements a declarative configuration pattern using Pydantic BaseModel
        - Uses field validation to enforce architecture-specific parameter requirements
        - Supports multiple model architectures with type-safe selection
        - Maintains extensibility through dictionary-based parameter specification
    
    Attributes:
        architecture (Literal["tcn", "wavenet"]): Neural network architecture selection
        params (Dict[str, Any]): Architecture-specific parameters dictionary
            
            TCN architecture required parameters:
                input_size (int): Dimensionality of input features
                output_size (int): Dimensionality of output predictions
                num_blocks (int): Number of TCN blocks in the network
                num_channels (int): Number of channels in convolutional layers
                kernel_size (int): Size of convolutional kernels
                dropout (float): Dropout probability for regularization
                
            Wavenet architecture required parameters:
                num_stacks (int): Number of stacked dilated convolution blocks
                num_layers_per_stack (int): Number of layers in each stack
                residual_channels (int): Number of channels in residual connections
                skip_channels (int): Number of channels in skip connections
                kernel_size (int): Size of convolutional kernels
                dilation_base (int): Base for exponential dilation rates
                dropout_rate (float): Dropout probability for regularization
                input_channels (int): Dimensionality of input features
                output_channels (int): Dimensionality of output predictions
                use_bias (bool): Whether to include bias terms in convolutions
    
    Integration:
        - Used by model factory functions to instantiate neural network models
        - Consumed by training scripts to configure model architecture
        - Validated during configuration loading to ensure all required parameters are present
        - Supports serialization to/from YAML configuration files
    
    Example:
        ```python
        # TCN model configuration
        tcn_config = ModelConfig(
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
        
        # Wavenet model configuration
        wavenet_config = ModelConfig(
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
        
        # Create model instance
        model = TCN(**tcn_config.params)
        ```
    
    Limitations:
        - Limited to two model architectures (TCN, Wavenet)
        - No validation for parameter value ranges or types
        - No support for custom or composite architectures
        - Architecture-specific parameters must be updated when adding new architectures
    """
    architecture: Literal["tcn", "wavenet"]
    params: Dict[str, Any]

    @field_validator('params')
    @classmethod
    def validate_params(cls, v, info):
        """
        Validates architecture-specific parameters for model configuration.
        
        This validator ensures that all required parameters for the selected neural network
        architecture are present in the params dictionary. It enforces architecture-specific
        schema validation by checking for the presence of mandatory parameters based on the
        architecture type (tcn or wavenet).
        
        Architecture:
            - Implements a Pydantic field validator pattern with architecture-specific validation
            - Uses dictionary comprehension for efficient missing parameter detection
            - Maintains O(n) time complexity where n is the number of required parameters
            - Preserves input dictionary structure without modification when valid
        
        Parameters:
            cls (Type[ModelConfig]): The ModelConfig class
            v (Dict[str, Any]): The params dictionary to validate
            info (ValidationInfo): Validation context containing the parent data
        
        Returns:
            Dict[str, Any]: The validated params dictionary (unchanged if valid)
        
        Raises:
            ValueError: If architecture is invalid or required parameters are missing
                - When architecture is not 'tcn' or 'wavenet'
                - When any required parameter for the specified architecture is missing
        
        Integration:
            - Called automatically by Pydantic during ModelConfig instantiation
            - Enables early validation before model construction attempts
            - Provides detailed error messages identifying specific missing parameters
            - Supports configuration loading from YAML with precise validation errors
        
        Limitations:
            - Only validates parameter presence, not parameter types or value ranges
            - Limited to two predefined architectures (tcn, wavenet)
            - No validation for extraneous parameters that may be ignored
            - Architecture-specific parameter lists must be updated when adding new architectures
        """
        architecture = info.data.get('architecture')
        if architecture == 'tcn':
            required_params = ["input_size", "output_size", "num_blocks", "num_channels", "kernel_size", "dropout"]
        elif architecture == 'wavenet':
            required_params = [
                "num_stacks", "num_layers_per_stack", "residual_channels", "skip_channels",
                "kernel_size", "dilation_base", "dropout_rate", "input_channels",
                "output_channels", "use_bias"
            ]
        else:
            raise ValueError(f"Invalid model architecture: {architecture}")

        missing = [key for key in required_params if key not in v]
        if missing:
            raise ValueError(f"Missing parameters for {architecture} model: {missing}")
        return v

class MetricsConfig(BaseModel):
    compute_metrics: bool = True
    train_metrics: List[Literal["loss", "mse", "dtw", "frechet"]] = ["loss"]
    val_metrics: List[Literal["loss", "mse", "dtw", "frechet"]] = ["loss", "mse", "dtw"]
    val_check_interval: Union[int, float] = 1.0
    early_stopping: bool = True
    early_stopping_patience: int = 10
    early_stopping_min_delta: float = 0.0001

class LoggingConfig(BaseModel):
    """
    Configuration for experiment logging and monitoring in audio-to-biosignal modeling.
    
    LoggingConfig encapsulates the parameters for Weights & Biases (wandb) integration,
    controlling experiment tracking, visualization, and reporting. It provides a unified
    interface for configuring logging behavior with sensible defaults while enabling
    experiment organization through projects, entities, and tags.
    
    Architecture:
        - Implements a declarative configuration pattern using Pydantic BaseModel
        - Supports hierarchical experiment organization through projects and optional entities
        - Enables experiment identification and filtering through run names and tags
        - Controls logging frequency to balance visibility and performance
    
    Attributes:
        wandb_project (str): Project name for grouping related experiments in wandb
        wandb_entity (Optional[str]): Team or username for organizational hierarchy (default: None)
        wandb_run_name (Optional[str]): Unique identifier for the specific experiment run (default: None)
        wandb_tags (List[str]): Categorical labels for filtering and organizing runs (default: [])
        log_every_n_steps (int): Frequency of metric logging during training (default: 50)
    
    Integration:
        - Used by training scripts to initialize and configure wandb logging
        - Consumed by experiment tracking systems for run organization
        - Controls metric visualization granularity and storage requirements
        - Supports reproducibility through consistent experiment naming
    
    Example:
        ```python
        config = LoggingConfig(
            wandb_project="audio2biosignal",
            wandb_entity="research-team",
            wandb_run_name="tcn-model-v1",
            wandb_tags=["production", "tcn", "eda-prediction"],
            log_every_n_steps=100
        )
        
        # Initialize wandb with this configuration
        wandb.init(
            project=config.wandb_project,
            entity=config.wandb_entity,
            name=config.wandb_run_name,
            tags=config.wandb_tags
        )
        ```
    
    Limitations:
        - Limited to wandb as the only supported logging platform
        - No validation for project/entity existence in wandb
        - No support for custom metric logging configurations
        - No automatic handling of API keys or authentication
    """
    wandb_project: str
    wandb_entity: Optional[str] = None
    wandb_run_name: Optional[str] = None
    wandb_tags: List[str] = []
    log_every_n_steps: int = 50

class CheckpointConfig(BaseModel):
    """
    Configuration for model checkpointing and persistence in audio-to-biosignal modeling.
    
    CheckpointConfig encapsulates the complete specification of model saving, loading, and
    selection strategies during training. It provides fine-grained control over checkpoint
    frequency, selection criteria, and storage locations while maintaining compatibility
    with the Hugging Face Transformers training infrastructure.
    
    Architecture:
        - Implements a declarative configuration pattern using Pydantic BaseModel
        - Supports both metric-based and frequency-based checkpoint selection
        - Enables deterministic model selection through configurable monitoring criteria
        - Maintains filesystem organization through centralized directory management
    
    Attributes:
        checkpoint_dir (str): Directory path where model checkpoints will be saved
        save_top_k (int): Number of best-performing models to retain (default: 3)
        monitor (str): Metric name to monitor for checkpoint selection (default: "val_loss")
        mode (Literal["min", "max"]): Whether to minimize or maximize the monitored metric (default: "min")
        save_last (bool): Whether to always save the most recent model (default: True)
        save_every_n_steps (int): Frequency of checkpointing in training steps (default: 1000)
        load_from_checkpoint (Optional[str]): Path to a specific checkpoint for resuming training (default: None)
    
    Integration:
        - Used by training scripts to configure model persistence strategy
        - Consumed by Hugging Face TrainingArguments for checkpoint configuration
        - Controls disk space usage through save_top_k parameter
        - Enables training resumption through load_from_checkpoint parameter
    
    Example:
        ```python
        config = CheckpointConfig(
            checkpoint_dir="./checkpoints/experiment_1",
            save_top_k=5,
            monitor="val_accuracy",
            mode="max",
            save_every_n_steps=500,
            load_from_checkpoint="./checkpoints/pretrained/model.ckpt"
        )
        
        # Configure TrainingArguments with this checkpoint configuration
        training_args = TrainingArguments(
            output_dir=config.checkpoint_dir,
            save_steps=config.save_every_n_steps,
            save_total_limit=config.save_top_k,
            load_best_model_at_end=config.save_last,
            metric_for_best_model=config.monitor,
            greater_is_better=config.mode == 'max'
        )
        ```
    
    Limitations:
        - No validation for directory existence or write permissions
        - No automatic handling of checkpoint file naming conventions
        - No support for distributed checkpoint strategies (e.g., sharded checkpoints)
        - Limited to scalar metrics for checkpoint selection criteria
    """
    save_top_k: int = 3
    checkpoint_dir: str
    monitor: str = "val_loss"
    mode: Literal["min", "max"] = "min"
    save_last: bool = True
    save_every_n_steps: int = 1000
    load_from_checkpoint: Optional[str] = None

class HardwareConfig(BaseModel):
    """
    Configuration for hardware resources and computational precision in audio-to-biosignal modeling.
    
    HardwareConfig encapsulates the complete specification of hardware utilization strategy,
    controlling device selection, numerical precision, and distributed training capabilities.
    It provides a unified interface for configuring computational resources with appropriate
    validation to prevent incompatible settings across different hardware platforms.
    
    Architecture:
        - Implements a declarative configuration pattern using Pydantic BaseModel
        - Enforces hardware-specific constraints through field validators
        - Prevents invalid combinations of device types, precision, and distribution settings
        - Maintains sensible defaults for common GPU-accelerated training scenarios
    
    Attributes:
        device (Literal["cpu", "cuda", "mps"]): Computation device type (default: "cuda")
            - "cpu": Use CPU for all computations
            - "cuda": Use NVIDIA GPU(s) with CUDA
            - "mps": Use Apple Metal Performance Shaders (M-series chips)
        precision (Literal["fp32", "fp16", "bf16"]): Numerical precision for computations (default: "fp16")
            - "fp32": 32-bit floating point (full precision)
            - "fp16": 16-bit floating point (half precision)
            - "bf16": 16-bit brain floating point (alternative half precision format)
        distributed (bool): Whether to use distributed training across multiple devices (default: False)
        num_gpus (int): Number of GPU devices to utilize (default: 1)
    
    Integration:
        - Used by training scripts to configure PyTorch device selection
        - Consumed by Hugging Face Trainer for mixed precision training setup
        - Controls distributed training configuration in multi-GPU environments
        - Validates hardware configurations before expensive model initialization
    
    Example:
        ```python
        # Standard GPU configuration with mixed precision
        config = HardwareConfig(
            device="cuda",
            precision="fp16",
            num_gpus=1
        )
        
        # Multi-GPU distributed training
        config = HardwareConfig(
            device="cuda",
            precision="fp16",
            distributed=True,
            num_gpus=4
        )
        
        # Apple Silicon configuration
        config = HardwareConfig(
            device="mps",
            precision="fp32",  # MPS only supports fp32
            num_gpus=1         # MPS only supports single GPU
        )
        ```
    
    Limitations:
        - MPS device limited to fp32 precision and single GPU
        - CPU device cannot use distributed training effectively
        - No automatic detection of available hardware resources
        - No support for heterogeneous device configurations
        - Limited validation of actual hardware availability at runtime
    """
    device: Literal["cpu", "cuda", "mps"] = "cuda"
    precision: Literal["fp32", "fp16", "bf16"] = "fp16"
    distributed: bool = False
    num_gpus: int = 1
    
    @field_validator('precision')
    @classmethod
    def validate_precision(cls, v, info):
        """
        Validates numerical precision compatibility with the selected device type.
        
        This validator ensures that the precision setting is compatible with the hardware
        device selection, enforcing platform-specific constraints such as the fp32-only
        limitation of Apple's Metal Performance Shaders (MPS). It prevents configuration
        of unsupported precision modes that would cause runtime errors during training.
        
        Architecture:
            - Implements a Pydantic field validator pattern with cross-field validation
            - Uses conditional logic to enforce device-specific precision constraints
            - Maintains O(1) time complexity with direct field access and comparison
            - Preserves the input precision value when valid for the selected device
        
        Parameters:
            cls (Type[HardwareConfig]): The HardwareConfig class
            v (Literal["fp32", "fp16", "bf16"]): The precision value to validate
            info (ValidationInfo): Validation context containing the parent data
                                   with the 'device' field
        
        Returns:
            Literal["fp32", "fp16", "bf16"]: The validated precision value (unchanged if valid)
        
        Raises:
            ValueError: When an incompatible precision is specified for the selected device
                - When 'mps' device is selected with any precision other than 'fp32'
        
        Integration:
            - Called automatically by Pydantic during HardwareConfig instantiation
            - Enables early validation before PyTorch device initialization
            - Works with other hardware validators to ensure consistent configuration
            - Prevents runtime errors from incompatible precision/device combinations
        
        Limitations:
            - Only enforces MPS-specific constraints, not CUDA or CPU limitations
            - No validation for hardware-specific bf16 support which varies by GPU generation
            - Does not check for actual hardware availability at runtime
            - Cannot detect driver or CUDA version compatibility issues
        """
        device = info.data.get('device')
        if device == 'mps' and v != 'fp32':
            raise ValueError("MPS device only supports fp32 precision")
        return v
    
    @field_validator('distributed')
    @classmethod
    def validate_distributed(cls, v, info):
        """
        Validates distributed training compatibility with the selected device type.
        
        This validator ensures that the distributed training setting is compatible with the hardware
        device selection, enforcing platform-specific constraints such as the lack of distributed
        training support in Apple's Metal Performance Shaders (MPS) and discouraging distributed
        training on CPU devices which is typically inefficient.
        
        Architecture:
            - Implements a Pydantic field validator pattern with cross-field validation
            - Uses conditional logic to enforce device-specific distributed training constraints
            - Maintains O(1) time complexity with direct field access and comparison
            - Preserves the input distributed value when valid for the selected device
        
        Parameters:
            cls (Type[HardwareConfig]): The HardwareConfig class
            v (bool): The distributed training flag value to validate
            info (ValidationInfo): Validation context containing the parent data
                                   with the 'device' field
        
        Returns:
            bool: The validated distributed flag value (unchanged if valid)
        
        Raises:
            ValueError: When distributed training is incompatible with the selected device
                - When 'mps' device is selected with distributed=True
                - When 'cpu' device is selected with distributed=True
        
        Integration:
            - Called automatically by Pydantic during HardwareConfig instantiation
            - Enables early validation before PyTorch distributed initialization
            - Works with other hardware validators to ensure consistent configuration
            - Prevents runtime errors from incompatible distributed/device combinations
        
        Limitations:
            - Only enforces basic device compatibility, not detailed distributed requirements
            - No validation for actual distributed environment availability
            - Cannot detect network configuration issues for distributed training
            - Does not validate distributed backend compatibility (NCCL, Gloo, etc.)
        """
        device = info.data.get('device')
        if device == 'mps' and v:
            raise ValueError("MPS device does not support distributed training")
        if device == 'cpu' and v:
            raise ValueError("Distributed training not recommended with CPU device")
        return v
    
    @field_validator('num_gpus')
    @classmethod
    def validate_num_gpus(cls, v, info):
        """
        Validates GPU count compatibility with the selected device type.
        
        This validator ensures that the number of GPUs specified is compatible with the hardware
        device selection, enforcing platform-specific constraints such as the single-GPU limitation
        of Apple's Metal Performance Shaders (MPS), zero GPUs for CPU devices, and at least one
        GPU for CUDA devices. It prevents configuration of invalid GPU counts that would cause
        runtime errors or resource allocation issues during training.
        
        Architecture:
            - Implements a Pydantic field validator pattern with cross-field validation
            - Uses conditional logic to enforce device-specific GPU count constraints
            - Maintains O(1) time complexity with direct field access and comparison
            - Preserves the input GPU count when valid for the selected device
        
        Parameters:
            cls (Type[HardwareConfig]): The HardwareConfig class
            v (int): The number of GPUs to validate
            info (ValidationInfo): Validation context containing the parent data
                                   with the 'device' field
        
        Returns:
            int: The validated GPU count (unchanged if valid)
        
        Raises:
            ValueError: When an incompatible GPU count is specified for the selected device
                - When 'mps' device is selected with num_gpus > 1
                - When 'cpu' device is selected with num_gpus > 0
                - When 'cuda' device is selected with num_gpus < 1
        
        Integration:
            - Called automatically by Pydantic during HardwareConfig instantiation
            - Enables early validation before PyTorch device initialization
            - Works with other hardware validators to ensure consistent configuration
            - Prevents runtime errors from incompatible device/GPU count combinations
        
        Limitations:
            - No validation for actual hardware availability at runtime
            - Cannot detect if the requested number of CUDA GPUs exceeds available hardware
            - Does not validate GPU memory requirements for the model architecture
            - No support for heterogeneous GPU configurations or specific device selection
        """
        device = info.data.get('device')
        if device == 'mps' and v > 1:
            raise ValueError("MPS device only supports a single GPU")
        if device == 'cpu' and v > 0:
            raise ValueError("CPU device should have num_gpus set to 0")
        if device == 'cuda' and v < 1:
            raise ValueError("CUDA device should have at least 1 GPU")
        return v

class TrainConfig(BaseModel):
    """
    Configuration for training loop parameters in audio-to-biosignal modeling.
    
    TrainConfig encapsulates the core training loop parameters that control convergence
    behavior, computational efficiency, and resource utilization during model training.
    It provides a minimal but essential set of parameters that directly influence the
    training dynamics and termination conditions.
    
    Architecture:
        - Implements a declarative configuration pattern using Pydantic BaseModel
        - Maintains separation of concerns from optimizer and hardware configurations
        - Provides sensible defaults for general deep learning scenarios
        - Supports integration with Hugging Face Trainer and PyTorch training loops
    
    Attributes:
        max_epochs (int): Maximum number of complete passes through the training dataset (default: 100)
        accumulate_grad_batches (int): Number of batches to accumulate gradients before optimizer step (default: 1)
            - Values > 1 enable effective batch size increase without memory overhead
            - Useful for simulating larger batch sizes on memory-constrained hardware
    
    Integration:
        - Used by training scripts to configure training loop termination
        - Consumed by Hugging Face TrainingArguments for gradient accumulation
        - Complements OptimizerConfig by separating training loop from optimization parameters
        - Works with MetricsConfig to coordinate validation frequency and early stopping
    
    Example:
        ```python
        # Standard configuration with default parameters
        train_config = TrainConfig()
        
        # Configuration for longer training with gradient accumulation
        train_config = TrainConfig(
            max_epochs=200,
            accumulate_grad_batches=4  # Effective batch size = batch_size * 4
        )
        
        # Use with TrainingArguments
        training_args = TrainingArguments(
            num_train_epochs=train_config.max_epochs,
            gradient_accumulation_steps=train_config.accumulate_grad_batches,
            # ... other arguments
        )
        ```
    
    Limitations:
        - Limited to fixed epoch-based training (no step-based termination)
        - No support for dynamic or scheduled gradient accumulation
        - No explicit learning rate scaling for accumulated gradients
        - Requires coordination with batch size in DataConfig for effective batch size calculation
    """
    max_epochs: int = 100
    accumulate_grad_batches: int = 1

class RunConfig(BaseModel):
    """
    Unified configuration container for audio-to-biosignal modeling experiments.
    
    RunConfig serves as the central integration point for all configuration components,
    providing a comprehensive specification for reproducible machine learning experiments.
    It encapsulates the complete experiment definition from model architecture to hardware
    utilization, enabling consistent configuration serialization and experiment tracking.
    
    Architecture:
        - Implements a hierarchical configuration pattern using Pydantic BaseModel
        - Composes specialized configuration components into a unified experiment definition
        - Enables YAML-based configuration with automatic validation and type checking
        - Maintains separation of concerns through modular configuration components
        - Supports serialization/deserialization for experiment reproducibility
    
    Attributes:
        experiment_name (str): Unique identifier for the experiment
        seed (int): Global random seed for reproducibility (default: 42)
        
        model (ModelConfig): Neural network architecture and hyperparameters
        optimizer (OptimizerConfig): Optimization algorithm and learning rate schedule
        data (DataConfig): Dataset selection and data loading parameters
        loss (LossConfig): Loss function specification
        hardware (HardwareConfig): Computational resources and precision settings
        logging (LoggingConfig): Experiment tracking and visualization parameters
        checkpoint (CheckpointConfig): Model persistence and selection strategy
        train (TrainConfig): Training loop parameters and termination conditions (default: TrainConfig())
    
    Integration:
        - Used by training scripts as the single entry point for experiment configuration
        - Loaded from YAML configuration files in deployment pipelines
        - Passed to the train() function to execute the complete training workflow
        - Enables experiment reproducibility through consistent random seed initialization
        - Supports configuration inheritance and composition through YAML anchors
    
    Example:
        ```python
        # Load configuration from YAML file
        with open('configs/experiment.yaml', 'r') as f:
            config_dict = yaml.safe_load(f)
        run_config = RunConfig(**config_dict)
        
        # Execute training with the configuration
        best_checkpoint_path = train(run_config)
        ```
    
    Limitations:
        - No validation for cross-component parameter consistency
        - No automatic hardware resource detection or configuration adjustment
        - Requires manual coordination of feature dimensions across components
        - Limited to predefined model architectures and optimization strategies
    """
    experiment_name: str
    seed: int = 42

    model: ModelConfig
    optimizer: OptimizerConfig
    data: DataConfig
    loss: LossConfig
    hardware: HardwareConfig
    logging: LoggingConfig
    checkpoint: CheckpointConfig
    train: TrainConfig = TrainConfig()
