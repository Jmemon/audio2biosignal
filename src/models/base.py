import torch
import torch.nn as nn
from typing import Dict

class BaseEncoder(nn.Module):
    def __init__(self, config: Dict):
        super().__init__()
        self.config = config

    def forward(self, audio: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError

class BaseDecoder(nn.Module):
    def __init__(self, config: Dict):
        super().__init__()
        self.config = config

    def forward(self, encoded_audio: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError

class BaseAudio2EDA(nn.Module):
    def __init__(self, encoder: BaseEncoder, decoder: BaseDecoder):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder

    def forward(self, audio: torch.Tensor) -> torch.Tensor:
        encoded_audio = self.encoder(audio)
        eda_output = self.decoder(encoded_audio)
        return eda_output
