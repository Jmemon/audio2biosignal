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
src/output.py
src/state.py
src/trainer.py

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

2. Update config schema to abstract out training parameters as `TrainConfig` and group everything else as `RunConfig`.
```aider
UPDATE src/configs.py:
    UPDATE pydantic TrainConfig(BaseModel):
        Contains batch_size, max_epochs, gradient_clip_val, early_stopping, early_stopping_patience
    CREATE pydantic RunConfig(BaseModel):
        Is basically the same as the old TrainConfig, except the fields that were at the root level are now in the new TrainConfig.
```

3. Create a `TrainState` class and supporting classes. Hypothetically, we should be able to load this from a checkpoint and use it to resume training.
```aider
CREATE src/state.py:
    CREATE pydantic RNGState(BaseModel):  
        seed: int
        times_sampled: int

    CREATE pydantic ModelState(BaseModel): 
        config: ModelConfig
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
        config: LoggingConfig
        metrics: Dict[str, Union[float, int]]  # names are train/<name> or val/<name>

    CREATE pydantic OptimizerState(BaseModel): 
        config: OptimizerConfig
        last_step: int
        moments_ckpt_path: Path

    CREATE pydantic CheckpointState(BaseModel): 
        config: CheckpointConfig

    CREATE pydantic TrainState(BaseModel): 
        status: Literal["initialized", "in_progress", "stopped", "complete"]
        rng_state: RNGState
        optimizer_state: OptimizerState
        checkpoint_state: CheckpointState
        metric_state: MetricState
        meta_dataloader_state: MetaDataLoaderState
        model_state: ModelState
        loss_config: LossConfig
        train_config: TrainConfig
        hardware_config: HardwareConfig

        CREATE def from_config(config: RunConfig) -> TrainState:
            ...
```

1. Create a `train` function.
```aider
CREATE src/train.py:
    CREATE def train(state: TrainState) -> TrainState:
```

## Scratch
- def fit() function runs the train loop. Flow:
    (model, dataloaders, optimizers, losses) 
     -> ensure everything is initialized
     -> Move everything to the accelerator
     -> connect to wandb
     -> run train loop for cfg.num_epochs, optimizer stepping, logging metrics to wandb, storing checkpoints, performing validation
     -> Return TrainOutput with metrics, paths_to_artifacts (checkpoints), path to best checkpoint filled in
