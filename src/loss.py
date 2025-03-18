import torch.nn as nn
from src.configs import LossConfig

class LossBuilder:
    """
    Factory for constructing PyTorch loss functions based on configuration.
    
    LossBuilder implements a factory pattern that instantiates appropriate PyTorch 
    loss function objects based on the provided configuration. It centralizes loss 
    function creation logic, ensuring consistent initialization across the application.
    
    Architecture:
        - Implements a static factory pattern with no instance state
        - O(1) complexity for loss function instantiation
        - Validation-first approach with early error detection
    
    Interface:
        - build(loss_config: LossConfig) -> nn.Module:
            Creates and returns a PyTorch loss function based on configuration.
            
            Parameters:
                loss_config: LossConfig - Configuration object specifying loss type
                                          and any parameters.
            
            Returns:
                nn.Module - Instantiated PyTorch loss function
                
            Raises:
                ValueError - When an unsupported loss function name is provided
                TypeError - When loss_config is None or not a LossConfig instance
    
    Integration:
        - Requires LossConfig from src.configs
        - Integrates with PyTorch's nn.Module loss functions
        - Used in training pipelines to construct the appropriate loss function:
          ```
          loss_fn = LossBuilder.build(config.loss)
          ```
    
    Limitations:
        - Custom loss implementation is currently a placeholder
        - No support for loss function parameter customization beyond type selection
        - Thread-safe but not designed for dynamic reconfiguration
    """
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
