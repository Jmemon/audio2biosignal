# Audio2Biosignal v0 Trainer

## High-Level Objective
Create a custom training function for the audio2biosignal project. The models are seq2seq models that take in audio and output biosignals. It needs to support experimentation with different models and training strategies.

## Mid-Level Objectives

## Implementation Notes
Perform rigorous input validation.
Ensure all failure points are handled gracefully via logging and/or raising exceptions with helpful debugging messages.
Include information-dense comments.

## Context
### Beginning Context
src/configs.py (read-only)

### Ending Context
src/configs.py (read-only)
src/train.py

## Low-Level Tasks
1. Create a `train` function.
```aider
CREATE src/train.py:
    CREATE def train(cfg: RunConfig) -> Path:
        Initialize everything according to config: model, optimizer, dataloaders, loss, hardware, metrics, logging, checkpointing.
        Connect to wandb.
        Move everything to the accelerator.
        Run train loop for cfg.num_epochs, optimizer stepping, logging metrics to wandb, storing checkpoints, performing validation.
        Return path to best checkpoint.
```
