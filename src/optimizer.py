from torch.optim import Adam, AdamW, SGD, Optimizer
from torch.optim.lr_scheduler import _LRScheduler, CosineAnnealingLR, StepLR, ExponentialLR
from typing import Tuple, Optional
from src.configs import OptimizerConfig

class OptimizerBuilder:
    @staticmethod
    def build(optimizer_config: OptimizerConfig, model_params) -> Tuple[Optimizer, Optional[_LRScheduler]]:
        if optimizer_config.name == "adamw":
            optimizer =  AdamW(model_params, lr=optimizer_config.learning_rate,
                              weight_decay=optimizer_config.weight_decay,
                              betas=(optimizer_config.beta1, optimizer_config.beta2))
        elif optimizer_config.name == "adam":
            optimizer = Adam(model_params, lr=optimizer_config.learning_rate,
                             weight_decay=optimizer_config.weight_decay,
                             betas=(optimizer_config.beta1, optimizer_config.beta2))
        elif optimizer_config.name == "sgd":
            optimizer = SGD(model_params, lr=optimizer_config.learning_rate,
                            weight_decay=optimizer_config.weight_decay,
                            momentum=optimizer_config.momentum)
        else:
            raise ValueError(f"Unsupported optimizer: {optimizer_config.name}")

        # Instantiate scheduler if necessary
        scheduler = None
        if optimizer_config.scheduler == "cosine":
            scheduler = CosineAnnealingLR(optimizer, T_max=100)  # Default T_max
        elif optimizer_config.scheduler == "step":
            scheduler = StepLR(optimizer, step_size=30, gamma=0.1)  # Default values
        elif optimizer_config.scheduler == "exponential":
            scheduler = ExponentialLR(optimizer, gamma=0.95)  # Default gamma

        return optimizer, scheduler
