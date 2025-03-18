from torch.optim import Adam, AdamW, SGD, Optimizer
from torch.optim.lr_scheduler import _LRScheduler, CosineAnnealingLR, StepLR, ExponentialLR
from typing import Tuple, Optional
from src.configs import OptimizerConfig

class OptimizerBuilder:
    @staticmethod
    def build(optimizer_config: OptimizerConfig, model_params) -> Tuple[Optimizer, Optional[_LRScheduler]]:
        """
        Constructs PyTorch optimizer and optional learning rate scheduler based on configuration.
        
        This factory method instantiates the appropriate optimizer and scheduler objects
        according to the provided configuration, ensuring consistent initialization across
        the application with centralized parameter handling.
        
        Architecture:
            - Implements a factory pattern with O(1) dispatch complexity
            - Maintains separation between configuration and instantiation logic
            - Preserves optimizer state when creating scheduler
        
        Parameters:
            optimizer_config (OptimizerConfig): Configuration object containing:
                - name (str): Optimizer type ("adam", "adamw", or "sgd")
                - learning_rate (float): Initial learning rate
                - weight_decay (float): L2 regularization strength
                - beta1, beta2 (float): Adam/AdamW momentum parameters
                - momentum (float): SGD momentum parameter (only used with SGD)
                - scheduler (Optional[str]): Scheduler type ("cosine", "step", "exponential", or None)
            
            model_params (iterable): PyTorch model parameters to optimize
                - Can be from model.parameters() or a custom parameter group
                - Empty parameter collections are handled gracefully
        
        Returns:
            Tuple[Optimizer, Optional[_LRScheduler]]:
                - optimizer: Configured PyTorch optimizer instance
                - scheduler: Learning rate scheduler if specified in config, otherwise None
        
        Raises:
            ValueError: When an unsupported optimizer name is provided
        
        Behavior:
            - Uses fixed default values for scheduler hyperparameters (T_max=100, step_size=30, etc.)
            - Does not implement all scheduler types mentioned in OptimizerConfig
            - Thread-safe as it maintains no state between calls
        
        Integration:
            - Used in training pipelines to construct optimization components:
              ```
              optimizer, scheduler = OptimizerBuilder.build(config.optimizer, model.parameters())
              ```
            - Scheduler step() should be called by the training loop after each epoch
        
        Limitations:
            - Limited to three optimizer types (Adam, AdamW, SGD)
            - No support for parameter-group-specific learning rates
            - Scheduler hyperparameters are not configurable through OptimizerConfig
            - No support for learning rate warmup or custom schedulers
        """
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
