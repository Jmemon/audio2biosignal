import argparse
import yaml
import os
import wandb
import torch
from src.configs import TrainConfig, AudioEDAFeatureConfig
from src.data.dataloader import DataLoaderBuilder
from src.optimizer import OptimizerBuilder
from src.loss import LossBuilder
from src.models.tcn import TCN
from src.models.wavenet import Wavenet
from src.utilities import S3FileManager
from transformers import Trainer, TrainingArguments

def main():
    parser = argparse.ArgumentParser(description="Train Audio2EDA Model")
    parser.add_argument("--config", type=str, required=True, help="Config file path")
    args = parser.parse_args()

    with open(args.config, 'r') as f:
        config_dict = yaml.safe_load(f)
    train_config = TrainConfig(**config_dict)

    # Load environment variables
    wandb_api_key = os.getenv('WANDB_API_KEY')
    aws_access_key_id = os.getenv('AWS_ACCESS_KEY_ID')
    aws_secret_access_key = os.getenv('AWS_SECRET_ACCESS_KEY')

    # Initialize WandB
    os.environ['WANDB_API_KEY'] = wandb_api_key
    wandb.init(project=train_config.logging.wandb_project, name=train_config.logging.wandb_run_name)

    # Instantiate model
    if train_config.model.architecture == 'tcn':
        model = TCN(train_config.model.params)
    elif train_config.model.architecture == 'wavenet':
        model = Wavenet(train_config.model.params)
    else:
        raise ValueError(f"Unknown model architecture: {train_config.model.architecture}")

    # Instantiate optimizer and scheduler
    optimizer, scheduler = OptimizerBuilder.build(train_config.optimizer, model.parameters())

    # Instantiate loss function
    loss_fn = LossBuilder.build(train_config.loss)

    # Instantiate data loaders
    feature_config = AudioEDAFeatureConfig()
    train_loader = DataLoaderBuilder.build(train_config.data, feature_config, split='train')
    val_loader = DataLoaderBuilder.build(train_config.data, feature_config, split='val')

    # Ensure checkpoint directory exists
    os.makedirs(train_config.checkpoint.checkpoint_dir, exist_ok=True)

    # Set up training arguments for HuggingFace Trainer
    training_args = TrainingArguments(
        output_dir=train_config.checkpoint.checkpoint_dir,
        num_train_epochs=train_config.max_epochs,
        per_device_train_batch_size=train_config.batch_size,
        per_device_eval_batch_size=train_config.batch_size,
        evaluation_strategy="steps",
        eval_steps=train_config.val_check_interval,
        save_steps=train_config.checkpoint.save_every_n_steps,
        logging_steps=train_config.logging.log_every_n_steps,
        save_total_limit=train_config.checkpoint.save_top_k,
        load_best_model_at_end=train_config.checkpoint.save_last,
        metric_for_best_model=train_config.checkpoint.monitor,
        greater_is_better=train_config.checkpoint.mode == 'max',
        fp16=train_config.hardware.precision == 'fp16',
        bf16=train_config.hardware.precision == 'bf16',
        dataloader_num_workers=train_config.data.num_workers,
        gradient_accumulation_steps=train_config.accumulate_grad_batches,
        gradient_checkpointing=True,
        learning_rate=train_config.optimizer.learning_rate,
        weight_decay=train_config.optimizer.weight_decay,
        report_to=["wandb"],
    )

    # Define compute_metrics function if needed

    # Instantiate Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_loader.dataset,
        eval_dataset=val_loader.dataset,
        optimizers=(optimizer, scheduler),
        compute_metrics=None,  # Define if needed
    )

    # Start training
    trainer.train()

if __name__ == "__main__":
    main()
