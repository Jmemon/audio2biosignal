from enum import Enum
from pydantic import BaseModel, field_validator
from typing import Dict, Any, Literal

class GeneratorType(str, Enum):
    pass

class DownsamplerType(str, Enum):
    pass

class UpsamplerType(str, Enum):
    residual_upsampler = "residual_upsampler"

class ResidualUpsamplerConfig(BaseModel):
    """
    Configuration for the Residual Upsampler.
    
    Attributes:
        upsampling_factor (float): The factor by which to upsample.
        in_channels (int): Number of input channels.
        hidden_channels (int): Number of channels in the hidden layers.
        out_channels (int): Number of output channels.
    """
    upsampling_factor: float
    in_channels: int
    hidden_channels: int
    out_channels: int

class ModelConfig(BaseModel):
    """
    Configuration for generator, EDA downsampler, and EDA upsampler in audio-to-biosignal modeling.

    ModelConfig encapsulates the specification of neural network components 
    and ensures required parameters are provided while allowing flexible extension.

    Attributes:
        generator (GeneratorType): Specifies the generator type.
        eda_downsampler (DownsamplerType): Specifies the EDA downsampler type.
        eda_upsampler (UpsamplerType): Specifies the EDA upsampler type.
    """
    generator: GeneratorType
    eda_downsampler: DownsamplerType
    eda_upsampler: UpsamplerType

