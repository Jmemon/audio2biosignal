from enum import Enum
from pydantic import BaseModel, validator, field_validator
from typing import List, Dict, Optional, Literal, Union, Any

class DatasetType(Enum):
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
    train_datasets: List[DatasetType]
    val_datasets: List[DatasetType]
    test_datasets: List[DatasetType]
    batch_size: int = 32
    num_workers: int = 4
    prefetch_size: int = 2

class OptimizerConfig(BaseModel):
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
    architecture: Literal["tcn", "wavenet"]
    params: Dict[str, Any]

    @field_validator('params')
    @classmethod
    def validate_params(cls, v, info):
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
    wandb_project: str
    wandb_entity: Optional[str] = None
    wandb_run_name: Optional[str] = None
    wandb_tags: List[str] = []
    log_every_n_steps: int = 50

class CheckpointConfig(BaseModel):
    save_top_k: int = 3
    checkpoint_dir: str
    monitor: str = "val_loss"
    mode: Literal["min", "max"] = "min"
    save_last: bool = True
    save_every_n_steps: int = 1000
    load_from_checkpoint: Optional[str] = None

class HardwareConfig(BaseModel):
    device: Literal["cpu", "cuda", "mps"] = "cuda"
    precision: Literal["fp32", "fp16", "bf16"] = "fp16"
    distributed: bool = False
    num_gpus: int = 1
    
    @field_validator('precision')
    @classmethod
    def validate_precision(cls, v, info):
        device = info.data.get('device')
        if device == 'mps' and v != 'fp32':
            raise ValueError("MPS device only supports fp32 precision")
        return v
    
    @field_validator('distributed')
    @classmethod
    def validate_distributed(cls, v, info):
        device = info.data.get('device')
        if device == 'mps' and v:
            raise ValueError("MPS device does not support distributed training")
        if device == 'cpu' and v:
            raise ValueError("Distributed training not recommended with CPU device")
        return v
    
    @field_validator('num_gpus')
    @classmethod
    def validate_num_gpus(cls, v, info):
        device = info.data.get('device')
        if device == 'mps' and v > 1:
            raise ValueError("MPS device only supports a single GPU")
        if device == 'cpu' and v > 0:
            raise ValueError("CPU device should have num_gpus set to 0")
        if device == 'cuda' and v < 1:
            raise ValueError("CUDA device should have at least 1 GPU")
        return v

class TrainConfig(BaseModel):
    max_epochs: int = 100
    accumulate_grad_batches: int = 1

class RunConfig(BaseModel):
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
