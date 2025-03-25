from pathlib import Path
import torch
import wandb
import logging
import time
import math
from typing import Optional, Dict, Any, List, Tuple
from torch.utils.data import DataLoader

from src.configs import RunConfig, AudioEDAFeatureConfig
from src.optimizer import OptimizerBuilder
from src.loss import LossBuilder
from src.data.dataloader import DataLoaderBuilder
from src.architectures.tcn import TCN
from src.architectures.wavenet import Wavenet
from src.metrics import MetricsCalculator

def train(cfg: RunConfig) -> Path:
    """
    Executes the complete training pipeline for audio-to-EDA models with comprehensive monitoring and checkpointing.
    
    This function orchestrates the end-to-end training process, including model initialization,
    optimization, validation, early stopping, and checkpoint management. It implements a multi-dataset
    training strategy with configurable metrics tracking and hardware acceleration support.
    
    Architecture:
        - Implements a modular training loop with O(epochs * batches) time complexity
        - Supports mixed precision training with automatic gradient scaling
        - Maintains model state through configurable checkpointing strategy
        - Integrates with W&B for experiment tracking with O(log_interval) logging frequency
        - Implements early stopping with patience-based termination
    
    Parameters:
        cfg (RunConfig): Comprehensive configuration object containing nested configurations for:
            - experiment_name (str): Unique identifier for the training run (required)
            - model (ModelConfig): Model architecture and parameters (must specify "tcn" or "wavenet")
            - optimizer (OptimizerConfig): Optimization algorithm and learning rate schedule
            - data (DataConfig): Dataset selection and dataloader parameters
            - loss (LossConfig): Loss function specification
            - hardware (HardwareConfig): Device and precision settings (must be "cpu", "cuda", or "mps")
            - logging (LoggingConfig): W&B integration parameters
            - checkpoint (CheckpointConfig): Model persistence strategy (must specify checkpoint_dir)
            - metrics (MetricsConfig): Evaluation metrics and validation frequency
            - train (TrainConfig): Training loop parameters (epochs, gradient accumulation)
    
    Returns:
        Path: Filesystem path to the best checkpoint file, or the last checkpoint if no best was saved.
            Returns None if no checkpoints were saved during training.
    
    Raises:
        ValueError: If required configuration fields are missing or invalid:
            - When experiment_name is not specified
            - When model.architecture is not "tcn" or "wavenet"
            - When checkpoint.checkpoint_dir is not specified
            - When hardware.device is not "cpu", "cuda", or "mps"
            - When no training or validation datasets are available
    
    Behavior:
        - Sets random seed from cfg.seed for reproducibility
        - Initializes W&B logging if cfg.logging.wandb_project is specified
        - Moves model to specified device (CPU/CUDA/MPS)
        - Enables mixed precision training when using fp16 precision on CUDA
        - Performs gradient clipping when cfg.optimizer.gradient_clip_val > 0
        - Validates at intervals specified by cfg.metrics.val_check_interval
        - Saves checkpoints based on monitored metric (min/max mode)
        - Implements early stopping when enabled in cfg.metrics
        - Saves final model state when cfg.checkpoint.save_last is True
    
    Integration:
        - Called by training scripts with a fully populated RunConfig
        - Consumes datasets via DataLoaderBuilder with AudioEDAFeatureConfig
        - Instantiates models via architecture-specific constructors (TCN/Wavenet)
        - Integrates with OptimizerBuilder for optimization strategy
        - Uses LossBuilder for criterion construction
        - Leverages MetricsCalculator for performance evaluation
    
    Limitations:
        - Currently supports only TCN and Wavenet architectures
        - Mixed precision (fp16) only available with CUDA devices
        - Requires both training and validation datasets
        - Checkpoint paths are determined by experiment_name
        - Memory usage scales with batch size and model size
        - No distributed training support
    """
    # (a) Validate critical fields
    if not cfg.experiment_name:
        raise ValueError("experiment_name must be specified in RunConfig")
    if not cfg.model.architecture in ["tcn", "wavenet"]:
        raise ValueError(f"Unsupported model architecture: {cfg.model.architecture}")
    if not cfg.checkpoint.checkpoint_dir:
        raise ValueError("checkpoint_dir must be specified in CheckpointConfig")
    if cfg.hardware.device not in ["cpu", "cuda", "mps"]:
        raise ValueError(f"Unsupported device: {cfg.hardware.device}")
    
    # (b) Set up logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    logger = logging.getLogger("train")
    logger.info(f"Starting training for experiment: {cfg.experiment_name}")
    
    # (c) Initialize W&B if enabled
    if cfg.logging.wandb_project:
        wandb.init(
            project=cfg.logging.wandb_project,
            entity=cfg.logging.wandb_entity,
            name=cfg.logging.wandb_run_name or cfg.experiment_name,
            tags=cfg.logging.wandb_tags
        )
        logger.info(f"Initialized W&B project: {cfg.logging.wandb_project}")
    
    # Set random seed for reproducibility
    torch.manual_seed(cfg.seed)
    
    # (d) Build the model
    if cfg.model.architecture == "tcn":
        model = TCN(cfg.model.params)
        logger.info("Initialized TCN model")
    elif cfg.model.architecture == "wavenet":
        model = Wavenet(cfg.model.params)
        logger.info("Initialized Wavenet model")
    else:
        raise ValueError(f"Unsupported model architecture: {cfg.model.architecture}")
    
    # (e) Build optimizer and scheduler
    optimizer, scheduler = OptimizerBuilder.build(cfg.optimizer, model.parameters())
    logger.info(f"Using optimizer: {cfg.optimizer.name}")
    
    # (f) Build loss function
    criterion = LossBuilder.build(cfg.loss)
    logger.info(f"Using loss function: {cfg.loss.name}")
    
    # (g) Construct dataloaders
    feature_config = AudioEDAFeatureConfig()
    train_dataloaders = DataLoaderBuilder.build(cfg.data, feature_config, 'train')
    val_dataloaders = DataLoaderBuilder.build(cfg.data, feature_config, 'val')
    
    if not train_dataloaders:
        raise ValueError("No training datasets available")
    if not val_dataloaders:
        raise ValueError("No validation datasets available")
    
    logger.info(f"Created dataloaders for {len(train_dataloaders)} training and {len(val_dataloaders)} validation datasets")
    
    # (h) Move model to device
    device = torch.device(cfg.hardware.device)
    model = model.to(device)
    
    # Set precision
    if cfg.hardware.precision == "fp16" and cfg.hardware.device == "cuda":
        scaler = torch.cuda.amp.GradScaler()
        use_amp = True
    else:
        use_amp = False
    
    # Create checkpoint directory
    checkpoint_dir = Path(cfg.checkpoint.checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    
    # Initialize tracking variables
    best_metric_value = float('inf') if cfg.checkpoint.mode == "min" else float('-inf')
    best_ckpt_path = None
    early_stop_counter = 0
    global_step = 0
    
    # (i) Training loop
    for epoch in range(cfg.train.max_epochs):
        logger.info(f"Starting epoch {epoch+1}/{cfg.train.max_epochs}")
        model.train()
        epoch_loss = 0.0
        epoch_start_time = time.time()
        
        # Training loop
        for dataloader_idx, train_dataloader in enumerate(train_dataloaders):
            for batch_idx, (audio, eda) in enumerate(train_dataloader):
                audio, eda = audio.to(device), eda.to(device)
                
                # Zero gradients
                optimizer.zero_grad()
                
                # Forward pass with optional AMP
                if use_amp:
                    with torch.cuda.amp.autocast():
                        outputs = model(audio)
                        loss = criterion(outputs.squeeze(), eda)
                    # Backward pass with scaler
                    scaler.scale(loss).backward()
                    if cfg.optimizer.gradient_clip_val > 0:
                        scaler.unscale_(optimizer)
                        torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.optimizer.gradient_clip_val)
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    # Standard forward/backward pass
                    outputs = model(audio)
                    loss = criterion(outputs.squeeze(), eda)
                    loss.backward()
                    if cfg.optimizer.gradient_clip_val > 0:
                        torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.optimizer.gradient_clip_val)
                    optimizer.step()
                
                # Update metrics
                epoch_loss += loss.item()
                global_step += 1
                
                # Calculate training metrics if enabled
                train_metrics_log = {}
                if cfg.metrics.compute_metrics and len(cfg.metrics.train_metrics) > 0:
                    # Skip metrics calculation for some steps to save computation
                    if global_step % cfg.logging.log_every_n_steps == 0:
                        batch_metrics = MetricsCalculator.calculate_metrics(
                            outputs.squeeze(), eda, cfg.metrics.train_metrics
                        )
                        for metric_name, metric_value in batch_metrics.items():
                            train_metrics_log[f"train/{metric_name}"] = metric_value
                
                # Log metrics
                if global_step % cfg.logging.log_every_n_steps == 0:
                    log_message = f"Epoch {epoch+1}, Batch {batch_idx+1}, Loss: {loss.item():.6f}"
                    
                    # Add any calculated metrics to the log message
                    for metric_name, metric_value in train_metrics_log.items():
                        metric_short_name = metric_name.split('/')[-1]
                        log_message += f", {metric_short_name}: {metric_value:.6f}"
                    
                    logger.info(log_message)
                    
                    if cfg.logging.wandb_project:
                        wandb_log = {
                            "train/loss": loss.item(),
                            "epoch": epoch,
                            "global_step": global_step,
                            **train_metrics_log
                        }
                        wandb.log(wandb_log)
                
                # Save checkpoint periodically
                if cfg.checkpoint.save_every_n_steps > 0 and global_step % cfg.checkpoint.save_every_n_steps == 0:
                    step_ckpt_path = checkpoint_dir / f"{cfg.experiment_name}_step_{global_step}.pt"
                    torch.save({
                        'epoch': epoch,
                        'global_step': global_step,
                        'model_state_dict': model.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                        'loss': loss.item(),
                    }, step_ckpt_path)
                    logger.info(f"Saved checkpoint at step {global_step} to {step_ckpt_path}")
        
        # Calculate average epoch loss
        avg_epoch_loss = epoch_loss / sum(len(dl) for dl in train_dataloaders)
        epoch_time = time.time() - epoch_start_time
        logger.info(f"Epoch {epoch+1} completed in {epoch_time:.2f}s, Avg Loss: {avg_epoch_loss:.6f}")
        
        if cfg.logging.wandb_project:
            wandb.log({"train/epoch_loss": avg_epoch_loss, "epoch": epoch})
        
        # Update learning rate scheduler if it exists
        if scheduler is not None:
            scheduler.step()
            if cfg.logging.wandb_project:
                wandb.log({"train/learning_rate": optimizer.param_groups[0]['lr'], "epoch": epoch})
        
        # Validation loop
        if (isinstance(cfg.metrics.val_check_interval, int) and (epoch + 1) % cfg.metrics.val_check_interval == 0) or \
           (isinstance(cfg.metrics.val_check_interval, float) and cfg.metrics.val_check_interval <= 1.0):
            model.eval()
            val_loss = 0.0
            val_metrics = {}
            
            with torch.no_grad():
                for val_dataloader in val_dataloaders:
                    for audio, eda in val_dataloader:
                        audio, eda = audio.to(device), eda.to(device)
                        outputs = model(audio)
                        batch_loss = criterion(outputs.squeeze(), eda)
                        val_loss += batch_loss.item()
                        
                        # Calculate additional metrics
                        if cfg.metrics.compute_metrics and len(cfg.metrics.val_metrics) > 0:
                            batch_metrics = MetricsCalculator.calculate_metrics(
                                outputs.squeeze(), eda, cfg.metrics.val_metrics
                            )
                            
                            # Accumulate metrics
                            for metric_name, metric_value in batch_metrics.items():
                                if metric_name not in val_metrics:
                                    val_metrics[metric_name] = 0.0
                                val_metrics[metric_name] += metric_value
            
            # Calculate average validation loss and metrics
            total_val_batches = sum(len(dl) for dl in val_dataloaders)
            avg_val_loss = val_loss / total_val_batches
            
            # Log validation metrics
            log_message = f"Validation Loss: {avg_val_loss:.6f}"
            wandb_log_dict = {"val/loss": avg_val_loss, "epoch": epoch}
            
            # Process and log other metrics
            for metric_name, metric_total in val_metrics.items():
                metric_avg = metric_total / total_val_batches
                log_message += f", {metric_name}: {metric_avg:.6f}"
                wandb_log_dict[f"val/{metric_name}"] = metric_avg
            
            logger.info(log_message)
            
            if cfg.logging.wandb_project:
                wandb.log(wandb_log_dict)
            
            # (j) Save checkpoint if it's the best so far
            # Determine which metric to monitor based on config
            monitor_metric = cfg.checkpoint.monitor
            if monitor_metric == "val_loss":
                current_metric = avg_val_loss
            elif monitor_metric in val_metrics:
                current_metric = val_metrics[monitor_metric] / total_val_batches
            else:
                logger.warning(f"Monitored metric '{monitor_metric}' not found, using val_loss instead")
                current_metric = avg_val_loss
            is_best = False
            
            if cfg.checkpoint.mode == "min" and current_metric < best_metric_value:
                best_metric_value = current_metric
                is_best = True
                early_stop_counter = 0
            elif cfg.checkpoint.mode == "max" and current_metric > best_metric_value:
                best_metric_value = current_metric
                is_best = True
                early_stop_counter = 0
            else:
                early_stop_counter += 1
            
            if is_best:
                best_ckpt_path = checkpoint_dir / f"{cfg.experiment_name}_best.pt"
                torch.save({
                    'epoch': epoch,
                    'global_step': global_step,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'loss': avg_val_loss,
                    'metric': current_metric,
                }, best_ckpt_path)
                logger.info(f"Saved best model checkpoint to {best_ckpt_path} with {cfg.checkpoint.monitor}: {best_metric_value:.6f}")
            
            # Early stopping check
            if cfg.metrics.early_stopping and early_stop_counter >= cfg.metrics.early_stopping_patience:
                logger.info(f"Early stopping triggered after {early_stop_counter} epochs without improvement")
                break
    
    # Save final model if requested
    if cfg.checkpoint.save_last:
        last_ckpt_path = checkpoint_dir / f"{cfg.experiment_name}_last.pt"
        torch.save({
            'epoch': epoch,
            'global_step': global_step,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': avg_epoch_loss,
        }, last_ckpt_path)
        logger.info(f"Saved final model checkpoint to {last_ckpt_path}")
    
    # Close wandb run if it was used
    if cfg.logging.wandb_project:
        wandb.finish()
    
    # (k) Return the path to the best checkpoint
    if best_ckpt_path is None:
        logger.warning("No best checkpoint was saved during training")
        # Fall back to the last checkpoint if available
        if cfg.checkpoint.save_last:
            best_ckpt_path = checkpoint_dir / f"{cfg.experiment_name}_last.pt"
    
    return best_ckpt_path
