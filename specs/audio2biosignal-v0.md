# audio2biosignal

## High-Level Objective
Build a system to train and evaluate a model that translate audio to EDA signals.

## Mid-Level Objective
- Create dataset models, classes, and dataloader for each dataset with eda data
- Create model classes, registry, and optimizer builder
- Create loss classes
- Create optimizer classes
- Create a trainer invocation script configurable by a config fig file that includes model fields, optimizer fields, datasets fields, hardware fields, loss fields, train hyperparameters, wandb logging fields, checkpoint fields. Properly instantiating all components based on config file and making sure nothing conflicts. Then invoking the train/eval loop.

## Implementation Notes
- Use pyproject.toml for dependencies (add minimally where needed). 
- Use src/ for code.
- Use HF Trainer for training.
- Use torchaudio for audio processing where possible (else librosa).
- ANYWHERE you have to make an assumption, include a warning using the warn package. Eg assuming what column of a csv file contains eda data.

## Context
### Beginning Context
- scripts/train_audio2eda.py

### Ending Context
- src/configs.py
- src/data/datasets/hku956.py
- src/data/datasets/pmemo2019.py
- src/data/dataloader.py
- src/data/datasets/types.py
- src/data/audio_preprocessing.py
- src/data/eda_preprocessing.py
- src/optimizer.py
- src/loss.py
- src/models/base.py
- src/models/encoders/wavenet.py
- src/models/encoders/transformer.py
- src/models/decoders/cnn.py
- src/models/decoders/lstm.py
- src/models/decoders/transformer.py
- src/models/audio2eda/transformer.py
- src/models/audio2eda/wavenet2eda.py
- src/models/registry.py
- src/logging.py
- src/checkpoint.py
- src/trainer.py
- scripts/train_audio2eda.py

## Low-Level Tasks
1. Create pydantic model for dataset configuration and for each dataset with eda data
```aider
CREATE src/configs.py:
    CREATE pydantic DatasetConfig(BaseModel): {
            dataset_name: str, 
            dataset_path: str,
            modalities: List[str],
            file_format: Dict[str, str]
            data_directories: Dict[str, str]
            metadata_paths: List[str]
            split_ratios: List[float]
            seed: int
        }

        pydantic HKU956Config(DatasetConfig): {
            dataset_name: "HKU956",
            dataset_root_path: "s3://audio2biosignal-train-data/HKU956/",
            modalities: ["eda", "audio"],
            file_format: {
                "eda": ".csv",
                "audio": ".mp3"
            }
            data_directories: {
                "eda": "s3://audio2biosignal-train-data/HKU956/1. physiological_signals/",
                "audio": "s3://audio2biosignal-train-data/HKU956/2. original_song_audio.csv"
            }
            metadata_paths: []
            split_ratios: (0.8, 0.1, 0.1)
            seed: 42
        }

        pydantic PMEmo2019Config(DatasetConfig): {
            dataset_name: "PMEmo2019",
            dataset_root_path: "s3://audio2biosignal-train-data/PMEmo2019/",
            modalities: ["eda", "audio"],
            file_format: {
                "eda": ".csv",
                "audio": ".mp3"
            }
            data_directories: {
                "eda": "s3://audio2biosignal-train-data/PMEmo2019/EDA/",
                "audio": "s3://audio2biosignal-train-data/PMEmo2019/chorus"
            }
            metadata_paths: ["s3://audio2biosignal-train-data/PMEmo2019/metadata.csv]
            split_ratios: (0.8, 0.1, 0.1)
            seed: 42
        }
```
2. Create AudioFeatureConfig pydantic model.
```aider
UPDATE src/configs.py:
    CREATE pydantic AudioFeatureConfig(BaseModel):
        audio_sample_rate: int = 44_100  # 44.1 kHz
        audio_normalize: bool = True
        audio_n_mfcc: int = 40
        audio_n_mels: int = 128
        audio_window_size: int = 400  # number of samples in each STFT window
        audio_hop_length: int = 160  # number of samples we increment by for each STFT window
```
3. Create pre-processing function for audio.
```aider
CREATE src/data/audio_preprocessing.py:
    CREATE def preprocess_audio(audio_file_path: str, audio_feature_config: AudioFeatureConfig) -> torch.Tensor:
        Resample audio to match `audio_sample_rate` if necessary.
        If audio_normalize is True, normalize the audio to be between -1 and 1.
        Compute MFCCs with given parameters.
        Return the MFCCs as a tensor of shape (1, n_mfcc, time_steps).
```
4. Create EDAFeatureConfig pydantic model.
```aider
UPDATE src/configs.py:
    CREATE pydantic EDAFeatureConfig(BaseModel):
        eda_window_size: int = 400
        eda_hop_length: int = 160
        eda_normalize: bool = True
        filter_lowpass: bool = True  # 8Hz low-pass filter
        filter_highpass: bool = False  # 0.05Hz high-pass filter for baseline correction
```
5. Create pre-processing function for eda signals.
```aider
CREATE src/data/eda_preprocessing.py:
    CREATE def preprocess_eda(eda_file_path: str, eda_feature_config: EDAFeatureConfig, audio_feature_config: AudioFeatureConfig) -> torch.Tensor:
        Load the waveform from the file.
        Resample to match audio sample rate.
        If eda_normalize is True, perform z-score normalization.
        Perform baseline correction with high-pass filter.
        Remove artifacts and noise with low-pass filter.
        Return the EDA signal as a tensor of shape (1, n_eda_features, time_steps).
```
6. Create classes for HKU956 and PMEmo2019 datasets.
```aider
CREATE src/data/datasets/hku956.py:
    CREATE class HKU956Dataset(Dataset):
        CREATE def __init__(self, dataset_config: HKU956Config, audio_feature_config: AudioFeatureConfig, eda_feature_config: EDAFeatureConfig, split: str="train"):
            CREATE self split: str = split
            For the following two variables, load the corresponding files based on the split ratios in the config. First 80% if train, 80%<90% if val, 90%<100% if test.
            CREATE eda_files: List[Tuple[str, str]]: From the s3 directory hierarchy, load the list of csv files in EDA subdirectories (not necessarily immediate) of physiological_signals directories into a list of tuples (subject, file_path) to be iterated over, ordered by subject_id, then by eda_file. The dir structure format to eda files will be `.../1. physiological_signals/<subject_id>/EDA/<num>_<song_id>.csv`.
            CREATE audio_files: Dict[str, str]: From the original_song_audios.csv file, store song_id as key and link as value.
        CREATE def load_audio_file(self, song_id: str) -> str:
            Load the audio file from the url at `audio_files[song_id]`.
            Pass this file to `preprocess_audio` with the given `audio_feature_config`.
        CREATE def load_eda_file(self, subject_id: str, song_id: str) -> Tuple[torch.Tensor, torch.Tensor]:
            Load the eda file from the s3 directory at `.../1. physiological_signals/<subject_id>/EDA/<num>_<song_id>.csv`.
            Pass this file to `preprocess_eda` with the given `eda_feature_config` and `audio_feature_config`.
        CREATE def __len__(self) -> int:
            length of eda_files
        CREATE def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
            Compute index into eda_files, download the files, and load the eda file and corresponding audio file into torch tensors.

CREATE src/data/datasets/pmemo2019.py:
    CREATE class PMEmo2019Dataset(Dataset):
        CREATE def __init__(self, dataset_config: PMEmo2019Config, audio_feature_config: AudioFeatureConfig, eda_feature_config: EDAFeatureConfig, split: str="train"):
            CREATE self split: str = split
            For the following two variables, load the corresponding files based on the split ratios in the config. First 80% if train, 80%<90% if val, 90%<100% if test.
            CREATE eda_files: List[str]: From the s3 directory hierarchy, load all s3 links to csv files in directory `EDA/` into a list.
            CREATE audio_files: List[str]: From the s3 directory hierarchy, load all s3 links to mp3 files in directory `chorus/` into a list.
        CREATE def load_audio_file(self, song_id: str) -> str: 
            MIRROR HKU956Dataset.load_audio_file
        CREATE def load_eda_file(self, song_id: str) -> Tuple[torch.Tensor, torch.Tensor]:
            MIRROR HKU956Dataset.load_eda_file
        CREATE def __len__(self) -> int:
            MIRROR HKU956Dataset.__len__
        CREATE def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
            MIRROR HKU956Dataset.__getitem__
```
7. Create dataset enum to be used for dataloader building.
```aider
CREATE src/data/datasets/types.py:
    CREATE enum DatasetType:
        CREATE HKU956='hku956'
        CREATE PMEmo2019='pmemo2019'
```
8. Create DataConfig pydantic model.
```aider
UPDATE src/configs.py:
    CREATE pydantic DataConfig(BaseModel):
        train_datasets: List[DatasetType]
        val_datasets: List[DatasetType]
        test_datasets: List[DatasetType]
        batch_size: int = 32
        num_workers: int = 4
        prefetch_size: int = 2
```
9. Create DataLoader builder. Pre-fetches configurable amount of data from S3. batch_size, num_workers, collate_fn.
```aider
CREATE src/data/dataloader.py:
    CREATE class DataLoaderBuilder:
        CREATE staticmethod def build(self, data_config: DataConfig, audio_feature_config: AudioFeatureConfig, eda_feature_config: EDAFeatureConfig) -> DataLoader:
            For each of datasets in `data_config`, instantiate a dataloader with the given parameters and return it.
            For pre-fetching make sure to load `prefetch_size` batches of data from S3.
            When building collate_fn, make sure to pad on the left so you can concatenate it into one tensor.
            A batch tensor should have shape (batch_size, channels, time_steps).
```
10. Create OptimizerConfig pydantic model.
```aider
UPDATE src/configs.py:
    CREATE pydantic OptimizerConfig(BaseModel):
        name: Literal["adam", "adamw", "sgd"] = "adamw"
        learning_rate: float = 1e-4
        weight_decay: float = 0.01
        beta1: float = 0.9
        beta2: float = 0.999
        momentum: float = 0.0  # Only used for SGD
        warmup_steps: int = 0
        warmup_ratio: float = 0.0
        scheduler: Optional[Literal["cosine", "linear", "constant", "reduce_on_plateau"]] = "cosine"
```
11. Create Optimizer builder.
```aider
CREATE src/optimizer.py
    CREATE class OptimizerBuilder:
        CREATE staticmethod def build(self, optimizer_config: OptimizerConfig) -> Optimizer:
            Instantiate the optimizer with the given parameters.
            If warmup_steps > 0, instantiate a warmup scheduler with the given parameters.
            Return the optimizer and scheduler.
```
12. Create LossConfig pydantic model.
```aider
UPDATE src/configs.py:
    CREATE pydantic LossConfig(BaseModel):
        name: Literal["mse", "l1", "huber", "custom"] = "mse"
```
13. Create Loss builder.
```aider
CREATE src/loss.py
    CREATE class LossBuilder:
        CREATE staticmethod def build(self, loss_config: LossConfig) -> Loss:
            Instantiate the loss function with the given parameters.
            Return the loss function.
```
14. Create ModelConfig pydantic model.
```aider
UPDATE src/configs.py:
    CREATE pydantic ModelConfig(BaseModel):
        pass TODO later – need to get into architectures first to know how to parameterize them
```
15. Create model files.
```aider
CREATE src/models/base.py:
    CREATE class BaseEncoder(nn.Module):
        CREATE def __init__(self, config: Dict):
            CREATE self config: Dict = config
        CREATE def forward(self, audio: torch.Tensor) -> torch.Tensor:
            raise NotImplementedError
    CREATE class BaseDecoder(nn.Module):
        CREATE def __init__(self, config: Dict):
            MIRROR BaseEncoder.__init__
        CREATE def forward(self, encoded_audio: torch.Tensor) -> torch.Tensor:
            MIRROR BaseEncoder.forward
    CREATE class BaseAudio2EDA(nn.Module):
        CREATE def __init__(self, encoder: BaseEncoder, decoder: BaseDecoder)
            CREATE self encoder: BaseEncoder = encoder
            CREATE self decoder: BaseDecoder = decoder
        CREATE def forward(self, audio: torch.Tensor) -> torch.Tensor:
            Pass the audio through the encoder and decoder.
            Return the predicted EDA signal.

CREATE src/models/encoders/wavenet.py:
    CREATE class WavenetEncoder(BaseEncoder):
        No implementation. TODO later.
CREATE src/models/encoders/transformer.py:
    CREATE class TransformerEncoder(BaseEncoder):
        No implementation. TODO later.
CREATE src/models/decoders/cnn.py:
    CREATE class CNNDecoder(BaseDecoder):
        No implementation. TODO later.
CREATE src/models/decoders/lstm.py:
    CREATE class LSTMDecoder(BaseDecoder):
        No implementation. TODO later.
CREATE src/models/decoders/transformer.py:
    CREATE class TransformerDecoder(BaseDecoder):
        No implementation. TODO later.
CREATE src/models/audio2eda/transformer.py:
    CREATE class TransformerAudio2EDA(BaseAudio2EDA):
        No implementation. TODO later.
CREATE src/models/audio2eda/wavenet2eda.py:
    CREATE class Wavenet2EDA(BaseAudio2EDA):
        No implementation. TODO later.

CREATE src/models/registry.py:
    CREATE class ModelRegistry:
        CREATE def __init__(self):
        CREATE def get_encoder(self, encoder_config: Dict) -> BaseEncoder:
        CREATE def get_decoder(self, decoder_config: Dict) -> BaseDecoder:
        CREATE def get_model(self, model_config: ModelConfig) -> BaseAudio2EDA:
            call `get_encoder` passing in encoder params from model_config.
            call `get_decoder` passing in decoder params from model_config.
            return subclass of `BaseAudio2EDA` with the encoder and decoder.
```
16. Create LoggingConfig pydantic model.
```aider
UPDATE src/configs.py:
    CREATE pydantic LoggingConfig(BaseModel):
        wandb_project: str
        wandb_entity: Optional[str] = None
        wandb_run_name: Optional[str] = None
        wandb_tags: List[str] = []
        log_every_n_steps: int = 50
        log_config: bool = True
        compute_metrics: bool = True
        train_metrics: List[Union[str, Literal["loss", "mse", "dtw", "frechet"]]] = ["loss"]
        val_metrics: List[Union[str, Literal["loss", "mse", "dtw", "frechet"]]] = ["loss", "mse", "dtw"]
17. Create CheckpointConfig pydantic model.
```aider
UPDATE src/configs.py:
    CREATE pydantic CheckpointConfig(BaseModel):
        save_top_k: int = 3
        checkpoint_dir: str
        monitor: str = "val_loss"
        mode: Literal["min", "max"] = "min"
        save_last: bool = True
        save_every_n_steps: int = 1000
        load_from_checkpoint: Optional[str] = None
```
18. Create TrainConfig pydantic model.
```aider
UPDATE src/configs.py:
    CREATE pydantic HardwareConfig(BaseModel):
            device: Literal["cpu", "cuda", "mps"] = "cuda"
            precision: Literal["fp32", "fp16", "bf16"] = "fp16"
            distributed: bool = False
            num_gpus: int = 1

        class TrainConfig(BaseModel):
            # High-level configuration
            experiment_name: str
            seed: int = 42
            
            # Component configurations
            model: ModelConfig
            optimizer: OptimizerConfig
            data: DataConfig
            loss: LossConfig
            hardware: HardwareConfig
            logging: LoggingConfig
            checkpoint: CheckpointConfig
            
            # Training hyperparameters
            max_epochs: int = 100
            gradient_clip_val: float = 1.0
            accumulate_grad_batches: int = 1
            val_check_interval: Union[int, float] = 1.0  # Can be epoch fraction or steps
            early_stopping: bool = True
            early_stopping_patience: int = 10
            early_stopping_min_delta: float = 0.0001
```
18. Create multiple config files.
```aider
CREATE configs/example1.yml, configs/example2.yml, configs/example3.yml:
    Fill in all fields with example values, and mock fields for model config since it hasn't been implemented yet.
    These should be a valid config file useful for debugging all the capabilities we've specified above.
    Vary parameters across the configs to test that everything is instantiated correctly.
```
19. Update Trainer invocation script.
```aider
UPDATE scripts/train_audio2eda.py:
    CREATE def main():
        Take the config filepath as a command line argument.
        train_config = TrainConfig.model_validate_json(open(train_config_path).read())
        Instantiate dataloaders for train, val, and test datasets.
        Instantiate optimizer, scheduler, loss, model, logging, checkpoint appropriately based on configs.
        Make sure checkpoint output directory exists and if not create it (with a name unique to the run if that is not specified).
        Instantiate a Trainer with the parameters in `train_config` and everything else prior.
        Kickoff training.
```