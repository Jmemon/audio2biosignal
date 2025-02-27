import argparse
import yaml
from src.configs import TrainConfig
from src.data.dataloader import DataLoaderBuilder
from src.optimizer import OptimizerBuilder
from src.loss import LossBuilder
from src.models.registry import ModelRegistry

def main():
    parser = argparse.ArgumentParser(description="Train Audio2EDA Model")
    parser.add_argument("--config", type=str, required=True, help="Config file path")
    args = parser.parse_args()

    with open(args.config, 'r') as f:
        config_dict = yaml.safe_load(f)
    train_config = TrainConfig(**config_dict)

    # Instantiate components
    data_loader = DataLoaderBuilder.build(
        train_config.data,
        train_config.model.encoder_params,
        train_config.model.decoder_params,
        split="train"
    )

    model_registry = ModelRegistry()
    model = model_registry.get_model(train_config.model)

    optimizer, scheduler = OptimizerBuilder.build(
        train_config.optimizer,
        model.parameters()
    )

    loss_fn = LossBuilder.build(train_config.loss)

    # Ensure checkpoint directory exists
    import os
    os.makedirs(train_config.checkpoint.checkpoint_dir, exist_ok=True)

    # Instantiate Trainer and start training
    # (Use appropriate training loop or framework)
    pass

if __name__ == "__main__":
    main()
