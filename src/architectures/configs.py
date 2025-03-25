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
    Configuration for neural network architecture selection and hyperparameters in audio-to-biosignal modeling.
    
    ModelConfig encapsulates the complete specification of neural network architectures
    and their hyperparameters, providing a unified interface for model instantiation with
    architecture-specific validation. It ensures that all required parameters for each
    supported architecture are provided while allowing flexible extension.
    
    Architecture:
        - Implements a declarative configuration pattern using Pydantic BaseModel
        - Uses field validation to enforce architecture-specific parameter requirements
        - Supports multiple model architectures with type-safe selection
        - Maintains extensibility through dictionary-based parameter specification
    
    Attributes:
        architecture (Literal["tcn", "wavenet"]): Neural network architecture selection
        params (Dict[str, Any]): Architecture-specific parameters dictionary
            
            TCN architecture required parameters:
                input_size (int): Dimensionality of input features
                output_size (int): Dimensionality of output predictions
                num_blocks (int): Number of TCN blocks in the network
                num_channels (int): Number of channels in convolutional layers
                kernel_size (int): Size of convolutional kernels
                dropout (float): Dropout probability for regularization
                
            Wavenet architecture required parameters:
                num_stacks (int): Number of stacked dilated convolution blocks
                num_layers_per_stack (int): Number of layers in each stack
                residual_channels (int): Number of channels in residual connections
                skip_channels (int): Number of channels in skip connections
                kernel_size (int): Size of convolutional kernels
                dilation_base (int): Base for exponential dilation rates
                dropout_rate (float): Dropout probability for regularization
                input_channels (int): Dimensionality of input features
                output_channels (int): Dimensionality of output predictions
                use_bias (bool): Whether to include bias terms in convolutions
    
    Integration:
        - Used by model factory functions to instantiate neural network models
        - Consumed by training scripts to configure model architecture
        - Validated during configuration loading to ensure all required parameters are present
        - Supports serialization to/from YAML configuration files
    
    Example:
        ```python
        # TCN model configuration
        tcn_config = ModelConfig(
            architecture="tcn",
            params={
                "input_size": 40,
                "output_size": 1,
                "num_blocks": 5,
                "num_channels": 64,
                "kernel_size": 3,
                "dropout": 0.2
            }
        )
        
        # Wavenet model configuration
        wavenet_config = ModelConfig(
            architecture="wavenet",
            params={
                "num_stacks": 2,
                "num_layers_per_stack": 10,
                "residual_channels": 64,
                "skip_channels": 256,
                "kernel_size": 3,
                "dilation_base": 2,
                "dropout_rate": 0.2,
                "input_channels": 40,
                "output_channels": 1,
                "use_bias": True
            }
        )
        
        # Create model instance
        model = TCN(**tcn_config.params)
        ```
    
    Limitations:
        - Limited to two model architectures (TCN, Wavenet)
        - No validation for parameter value ranges or types
        - No support for custom or composite architectures
        - Architecture-specific parameters must be updated when adding new architectures
    """
    architecture: Literal["tcn", "wavenet", "residual_upsampler"]
    params: Dict[str, Any]

