import torch.nn as nn
from src.configs import LossConfig

class LossBuilder:
    @staticmethod
    def build(loss_config: LossConfig) -> nn.Module:
        if loss_config.name == "mse":
            return nn.MSELoss()
        elif loss_config.name == "l1":
            return nn.L1Loss()
        elif loss_config.name == "huber":
            return nn.SmoothL1Loss()
        elif loss_config.name == "custom":
            # Implement custom loss function
            pass
        else:
            raise ValueError(f"Unsupported loss function: {loss_config.name}")
