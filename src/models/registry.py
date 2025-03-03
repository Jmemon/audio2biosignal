from typing import Any
import torch.nn as nn
from src.models.tcn import TCN
from src.models.wavenet import Wavenet
from src.configs import ModelConfig

class ModelRegistry:
    def get_model(self, model_config: ModelConfig) -> nn.Module:
        architecture = model_config.architecture
        params = model_config.params

        if architecture == 'tcn':
            return TCN(params)
        elif architecture == 'wavenet':
            return Wavenet(params)
        else:
            raise ValueError(f"Unknown architecture: {architecture}")
