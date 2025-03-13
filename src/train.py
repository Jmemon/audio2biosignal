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
from src.models.tcn import TCN
from src.models.wavenet import Wavenet

def train(cfg: RunConfig) -> Path:
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
            tags=cfg.logging.wandb_tags,
            config=cfg.dict()
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
                
                # Log metrics
                if global_step % cfg.logging.log_every_n_steps == 0:
                    logger.info(f"Epoch {epoch+1}, Batch {batch_idx+1}, Loss: {loss.item():.6f}")
                    if cfg.logging.wandb_project:
                        wandb.log({"train/loss": loss.item(), "epoch": epoch, "global_step": global_step})
                
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
                        
                        # Calculate additional metrics if needed
                        # This would be where you'd compute DTW, Frechet, etc.
            
            # Calculate average validation loss
            avg_val_loss = val_loss / sum(len(dl) for dl in val_dataloaders)
            logger.info(f"Validation Loss: {avg_val_loss:.6f}")
            
            if cfg.logging.wandb_project:
                wandb.log({"val/loss": avg_val_loss, "epoch": epoch})
            
            # (j) Save checkpoint if it's the best so far
            current_metric = avg_val_loss  # Using validation loss as the default metric
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
