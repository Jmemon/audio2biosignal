from enum import Enum
from pydantic import BaseModel, validator, field_validator
from typing import List, Dict, Optional, Literal, Union, Any

class DatasetType(Enum):
    HKU956 = 'hku956'
    PMEmo2019 = 'pmemo2019'

class DatasetConfig(BaseModel):
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

class LoggingConfig(BaseModel):
    wandb_project: str
    wandb_entity: Optional[str] = None
    wandb_run_name: Optional[str] = None
    wandb_tags: List[str] = []
    log_every_n_steps: int = 50
    log_config: bool = True
    compute_metrics: bool = True
    train_metrics: List[Literal["loss", "mse", "dtw", "frechet"]] = ["loss"]
    val_metrics: List[Literal["loss", "mse", "dtw", "frechet"]] = ["loss", "mse", "dtw"]
    val_check_interval: Union[int, float] = 1.0
    early_stopping: bool = True
    early_stopping_patience: int = 10
    early_stopping_min_delta: float = 0.0001

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
