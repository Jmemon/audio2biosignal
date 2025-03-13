# Audio2Biosignal v0 Trainer

## High-Level Objective
Create a custom trainer class for the audio2biosignal project. The models are seq2seq models that take in audio and output biosignals. It needs to support experimentation with different models and training strategies.

## Mid-Level Objectives

## Implementation Notes

## Context
### Beginning Context
src/configs.py

### Ending Context
src/configs.py
src/data/meta_dataloader.py
src/state.py
src/train.py

## Low-Level Tasks
1. Create the meta-dataloader class.
```aider
CREATE src/data/meta_dataloader.py:
    CREATE class MetaDataLoader:
        CREATE def __init__(self, dataloaders: List[DataLoader], split: str, sampling_weights: List[float]) -> MetaDataLoader:
            len(dataloaders) == len(sampling_weights)
        CREATE def __iter__(self) -> Iterator[Tuple[torch.Tensor, torch.Tensor]]:
            get next batch from one of the dataloaders, each with probability sampling_weights[i] (unless its empty, then skip it) (so sampling_weights impacts data order)
```

1. Create a `TrainState` class and supporting classes. Hypothetically, we should be able to load this from a checkpoint and use it to resume training. This should outline a checkpoint schema.
```aider
CREATE src/state.py:
    CREATE pydantic RNGState(BaseModel):  
        seed: int
        torch_rng_state: Dict[str, List]
        numpy_rng_state: bytes
        python_rng_state: tuple

    CREATE pydantic ModelState(BaseModel): 
        model_loaded: bool
        model_ram_size_bytes: int
        model_dtype: torch.dtype
        model_device: torch.device
        best_model_path: Path
        last_model_path: Path

    CREATE pydantic DataLoaderState(BaseModel): 
        last_batch_idx: int
        dataset: DatasetType
        split: Literal["train", "val", "test"]

    CREATE pydantic MetaDataLoaderState(BaseModel): 
        split: Literal["train", "val", "test"]
        sampling_weights: List[float]
        dataloader_states: List[DataLoaderState] # len(dataloader_states) == len(sampling_weights)

    CREATE pydantic MetricState(BaseModel): 
        metrics: Dict[str, Union[float, int]]  # validate names are train/<name> or val/<name>

    CREATE pydantic OptimizerState(BaseModel): 
        last_step: int
        moments_ram_size_bytes: int
        moments_dtype: torch.dtype
        moments_device: torch.device
        moments_path: Path

    CREATE pydantic CheckpointState(BaseModel): 
        steps_since_last_checkpoint: int
        checkpoint_dir: Path

    CREATE pydantic TrainState(BaseModel): 
        status: Literal["initialized", "in_progress", "stopped", "complete"]
        rng_state: RNGState
        optimizer_state: OptimizerState
        checkpoint_state: CheckpointState
        metric_state: MetricState
        meta_dataloader_state: MetaDataLoaderState
        model_state: ModelState

        @classmethod
        CREATE def from_config(cls, config: RunConfig) -> TrainState:
            ...

        @classmethod
        CREATE def from_checkpoint(cls, checkpoint_path: Path) -> TrainState:
            ...
        
        CREATE def to_checkpoint(self, checkpoint_path: Path) -> None:
            ...
```

1. Create a `train` function.
```aider
CREATE src/train.py:
    CREATE def train(cfg: TrainState) -> TrainState:
```

## Scratch
- def fit() function runs the train loop. Flow:
    (model, dataloaders, optimizers, losses) 
     -> ensure everything is initialized
     -> Move everything to the accelerator
     -> connect to wandb
     -> run train loop for cfg.num_epochs, optimizer stepping, logging metrics to wandb, storing checkpoints, performing validation
     -> Return TrainOutput with metrics, paths_to_artifacts (checkpoints), path to best checkpoint filled in
