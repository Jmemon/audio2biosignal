"""
Tests for the OptimizerBuilder class.

This suite verifies:
1. Creation of different optimizer types (adamw, adam, sgd)
2. Correct parameter passing to optimizers
3. Creation of different scheduler types
4. Error handling for invalid configurations
5. Return value structure and types
"""

import pytest
import torch
import torch.nn as nn
from torch.optim import Adam, AdamW, SGD
from torch.optim.lr_scheduler import CosineAnnealingLR, StepLR, ExponentialLR

from src.configs import OptimizerConfig
from src.optimizer import OptimizerBuilder


class TestOptimizerBuilder:
    """Test suite for OptimizerBuilder class."""

    @pytest.fixture
    def simple_model(self):
        """Create a simple model for testing optimizers."""
        class SimpleModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.linear = nn.Linear(10, 2)
                
            def forward(self, x):
                return self.linear(x)
                
        return SimpleModel()

    @pytest.fixture
    def default_optimizer_config(self):
        """Create a default optimizer config for testing."""
        return OptimizerConfig(
            name="adam",
            learning_rate=0.001,
            weight_decay=0.01,
            beta1=0.9,
            beta2=0.999,
            momentum=0.0,
            scheduler=None
        )

    class TestOptimizerCreation:
        """Tests for creating different optimizer types."""

        def test_create_adam_optimizer(self, simple_model, default_optimizer_config):
            """
            GIVEN an optimizer config with name="adam"
            WHEN OptimizerBuilder.build is called
            THEN it should return an Adam optimizer instance
            """
            # Configure for Adam
            config = default_optimizer_config
            config.name = "adam"
            
            # Build optimizer
            optimizer, scheduler = OptimizerBuilder.build(config, simple_model.parameters())
            
            # Verify optimizer type and parameters
            assert isinstance(optimizer, Adam)
            assert not isinstance(optimizer, AdamW)  # Ensure it's not AdamW
            assert optimizer.defaults['lr'] == config.learning_rate
            assert optimizer.defaults['weight_decay'] == config.weight_decay
            assert optimizer.defaults['betas'] == (config.beta1, config.beta2)
            assert scheduler is None

        def test_create_adamw_optimizer(self, simple_model, default_optimizer_config):
            """
            GIVEN an optimizer config with name="adamw"
            WHEN OptimizerBuilder.build is called
            THEN it should return an AdamW optimizer instance
            """
            # Configure for AdamW
            config = default_optimizer_config
            config.name = "adamw"
            
            # Build optimizer
            optimizer, scheduler = OptimizerBuilder.build(config, simple_model.parameters())
            
            # Verify optimizer type and parameters
            assert isinstance(optimizer, AdamW)
            assert optimizer.defaults['lr'] == config.learning_rate
            assert optimizer.defaults['weight_decay'] == config.weight_decay
            assert optimizer.defaults['betas'] == (config.beta1, config.beta2)
            assert scheduler is None

        def test_create_sgd_optimizer(self, simple_model, default_optimizer_config):
            """
            GIVEN an optimizer config with name="sgd"
            WHEN OptimizerBuilder.build is called
            THEN it should return an SGD optimizer instance
            """
            # Configure for SGD
            config = default_optimizer_config
            config.name = "sgd"
            config.momentum = 0.9  # Set momentum for SGD
            
            # Build optimizer
            optimizer, scheduler = OptimizerBuilder.build(config, simple_model.parameters())
            
            # Verify optimizer type and parameters
            assert isinstance(optimizer, SGD)
            assert optimizer.defaults['lr'] == config.learning_rate
            assert optimizer.defaults['weight_decay'] == config.weight_decay
            assert optimizer.defaults['momentum'] == config.momentum
            assert scheduler is None

        def test_invalid_optimizer_name(self, simple_model, default_optimizer_config):
            """
            GIVEN an optimizer config with an invalid name
            WHEN OptimizerBuilder.build is called
            THEN it should raise a ValueError with appropriate message
            """
            # Configure with invalid optimizer name
            config = default_optimizer_config
            config.name = "invalid_optimizer"
            
            # Verify ValueError is raised
            with pytest.raises(ValueError) as excinfo:
                OptimizerBuilder.build(config, simple_model.parameters())
            
            assert "Unsupported optimizer" in str(excinfo.value)
            assert config.name in str(excinfo.value)

    class TestSchedulerCreation:
        """Tests for creating different scheduler types."""

        def test_cosine_scheduler(self, simple_model, default_optimizer_config):
            """
            GIVEN an optimizer config with scheduler="cosine"
            WHEN OptimizerBuilder.build is called
            THEN it should return a CosineAnnealingLR scheduler
            """
            # Configure for cosine scheduler
            config = default_optimizer_config
            config.scheduler = "cosine"
            
            # Build optimizer and scheduler
            optimizer, scheduler = OptimizerBuilder.build(config, simple_model.parameters())
            
            # Verify scheduler type
            assert isinstance(scheduler, CosineAnnealingLR)
            assert scheduler.T_max == 100  # Default T_max value

        def test_step_scheduler(self, simple_model, default_optimizer_config):
            """
            GIVEN an optimizer config with scheduler="step"
            WHEN OptimizerBuilder.build is called
            THEN it should return a StepLR scheduler
            """
            # Configure for step scheduler
            config = default_optimizer_config
            config.scheduler = "step"
            
            # Build optimizer and scheduler
            optimizer, scheduler = OptimizerBuilder.build(config, simple_model.parameters())
            
            # Verify scheduler type
            assert isinstance(scheduler, StepLR)
            assert scheduler.step_size == 30  # Default step_size
            assert scheduler.gamma == 0.1  # Default gamma

        def test_exponential_scheduler(self, simple_model, default_optimizer_config):
            """
            GIVEN an optimizer config with scheduler="exponential"
            WHEN OptimizerBuilder.build is called
            THEN it should return an ExponentialLR scheduler
            """
            # Configure for exponential scheduler
            config = default_optimizer_config
            config.scheduler = "exponential"
            
            # Build optimizer and scheduler
            optimizer, scheduler = OptimizerBuilder.build(config, simple_model.parameters())
            
            # Verify scheduler type
            assert isinstance(scheduler, ExponentialLR)
            assert scheduler.gamma == 0.95  # Default gamma

        def test_no_scheduler(self, simple_model, default_optimizer_config):
            """
            GIVEN an optimizer config with scheduler=None
            WHEN OptimizerBuilder.build is called
            THEN it should return None for the scheduler
            """
            # Configure with no scheduler
            config = default_optimizer_config
            config.scheduler = None
            
            # Build optimizer and scheduler
            optimizer, scheduler = OptimizerBuilder.build(config, simple_model.parameters())
            
            # Verify scheduler is None
            assert scheduler is None
            
        def test_invalid_scheduler_name(self, simple_model, default_optimizer_config):
            """
            GIVEN an optimizer config with an invalid scheduler name
            WHEN OptimizerBuilder.build is called
            THEN it should return an optimizer but no scheduler
            """
            # Configure with invalid scheduler name
            config = default_optimizer_config
            config.scheduler = "invalid_scheduler"
            
            # Build optimizer and scheduler
            optimizer, scheduler = OptimizerBuilder.build(config, simple_model.parameters())
            
            # Verify optimizer exists but scheduler is None
            assert isinstance(optimizer, Adam)
            assert scheduler is None

    class TestParameterPassing:
        """Tests for correct parameter passing to optimizers."""
        
        def test_empty_model_parameters(self):
            """
            GIVEN an optimizer config and empty model parameters
            WHEN OptimizerBuilder.build is called
            THEN it should create an optimizer with no parameter groups
            """
            # Create empty parameter list
            empty_params = []
            
            # Create config
            config = OptimizerConfig(
                name="adam",
                learning_rate=0.001,
                weight_decay=0.01,
                beta1=0.9,
                beta2=0.999,
                momentum=0.0,
                scheduler=None
            )
            
            # Build optimizer
            optimizer, _ = OptimizerBuilder.build(config, empty_params)
            
            # Verify optimizer has no parameter groups
            assert len(optimizer.param_groups) == 1
            assert len(optimizer.param_groups[0]['params']) == 0

        def test_learning_rate_parameter(self, simple_model, default_optimizer_config):
            """
            GIVEN an optimizer config with custom learning_rate
            WHEN OptimizerBuilder.build is called
            THEN the optimizer should have the correct learning rate
            """
            # Configure custom learning rate
            config = default_optimizer_config
            config.learning_rate = 0.0042
            
            # Build optimizer
            optimizer, _ = OptimizerBuilder.build(config, simple_model.parameters())
            
            # Verify learning rate
            assert optimizer.defaults['lr'] == 0.0042

        def test_weight_decay_parameter(self, simple_model, default_optimizer_config):
            """
            GIVEN an optimizer config with custom weight_decay
            WHEN OptimizerBuilder.build is called
            THEN the optimizer should have the correct weight decay
            """
            # Configure custom weight decay
            config = default_optimizer_config
            config.weight_decay = 0.05
            
            # Build optimizer
            optimizer, _ = OptimizerBuilder.build(config, simple_model.parameters())
            
            # Verify weight decay
            assert optimizer.defaults['weight_decay'] == 0.05

        def test_beta_parameters(self, simple_model, default_optimizer_config):
            """
            GIVEN an optimizer config with custom beta values
            WHEN OptimizerBuilder.build is called with Adam or AdamW
            THEN the optimizer should have the correct beta values
            """
            # Configure custom betas
            config = default_optimizer_config
            config.name = "adam"
            config.beta1 = 0.8
            config.beta2 = 0.99
            
            # Build optimizer
            optimizer, _ = OptimizerBuilder.build(config, simple_model.parameters())
            
            # Verify betas
            assert optimizer.defaults['betas'] == (0.8, 0.99)

        def test_momentum_parameter(self, simple_model, default_optimizer_config):
            """
            GIVEN an optimizer config with custom momentum
            WHEN OptimizerBuilder.build is called with SGD
            THEN the optimizer should have the correct momentum
            """
            # Configure custom momentum for SGD
            config = default_optimizer_config
            config.name = "sgd"
            config.momentum = 0.85
            
            # Build optimizer
            optimizer, _ = OptimizerBuilder.build(config, simple_model.parameters())
            
            # Verify momentum
            assert optimizer.defaults['momentum'] == 0.85

    class TestIntegration:
        """Integration tests for the complete optimizer building process."""

        def test_complete_build_process(self, simple_model):
            """
            GIVEN a complete optimizer configuration
            WHEN OptimizerBuilder.build is called
            THEN it should return correctly configured optimizer and scheduler
            """
            # Create a complete configuration
            config = OptimizerConfig(
                name="adamw",
                learning_rate=0.002,
                weight_decay=0.1,
                beta1=0.85,
                beta2=0.95,
                momentum=0.0,
                scheduler="cosine"
            )
            
            # Build optimizer and scheduler
            optimizer, scheduler = OptimizerBuilder.build(config, simple_model.parameters())
            
            # Verify optimizer
            assert isinstance(optimizer, AdamW)
            assert optimizer.defaults['lr'] == 0.002
            assert optimizer.defaults['weight_decay'] == 0.1
            assert optimizer.defaults['betas'] == (0.85, 0.95)
            
            # Verify scheduler
            assert isinstance(scheduler, CosineAnnealingLR)
            
            # Verify optimizer is connected to scheduler
            assert scheduler.optimizer == optimizer

        def test_optimizer_step(self, simple_model):
            """
            GIVEN a built optimizer
            WHEN optimizer.step() is called after loss.backward()
            THEN the model parameters should be updated
            """
            # Create configuration
            config = OptimizerConfig(
                name="adam",
                learning_rate=0.1,  # High learning rate for visible change
                weight_decay=0.0,
                beta1=0.9,
                beta2=0.999,
                momentum=0.0,
                scheduler=None
            )
            
            # Build optimizer
            optimizer, _ = OptimizerBuilder.build(config, simple_model.parameters())
            
            # Save initial parameters
            initial_params = []
            for param in simple_model.parameters():
                initial_params.append(param.clone().detach())
            
            # Create dummy input and perform forward pass
            dummy_input = torch.randn(1, 10)
            output = simple_model(dummy_input)
            
            # Create dummy loss and backward pass
            loss = output.sum()
            loss.backward()
            
            # Perform optimization step
            optimizer.step()
            
            # Verify parameters have changed
            for i, param in enumerate(simple_model.parameters()):
                assert not torch.allclose(param, initial_params[i])

        def test_scheduler_step(self, simple_model):
            """
            GIVEN a built optimizer with scheduler
            WHEN scheduler.step() is called
            THEN the learning rate should be updated
            """
            # Create configuration with scheduler
            config = OptimizerConfig(
                name="adam",
                learning_rate=0.01,
                weight_decay=0.0,
                beta1=0.9,
                beta2=0.999,
                momentum=0.0,
                scheduler="exponential"  # Use exponential for predictable changes
            )
            
            # Build optimizer and scheduler
            optimizer, scheduler = OptimizerBuilder.build(config, simple_model.parameters())
            
            # Get initial learning rate
            initial_lr = optimizer.param_groups[0]['lr']
            
            # Step the scheduler
            scheduler.step()
            
            # Get updated learning rate
            updated_lr = optimizer.param_groups[0]['lr']
            
            # Verify learning rate has changed as expected
            expected_lr = initial_lr * 0.95  # gamma=0.95 for exponential scheduler
            assert abs(updated_lr - expected_lr) < 1e-6
            
        def test_optimizer_state_preservation(self, simple_model):
            """
            GIVEN a built optimizer with scheduler
            WHEN optimizer updates parameters and scheduler steps
            THEN optimizer state should be preserved
            """
            # Create configuration with scheduler
            config = OptimizerConfig(
                name="adam",
                learning_rate=0.1,
                weight_decay=0.0,
                beta1=0.9,
                beta2=0.999,
                momentum=0.0,
                scheduler="step"
            )
            
            # Build optimizer and scheduler
            optimizer, scheduler = OptimizerBuilder.build(config, simple_model.parameters())
            
            # Create dummy input and perform forward pass
            dummy_input = torch.randn(1, 10)
            output = simple_model(dummy_input)
            
            # Create dummy loss and backward pass
            loss = output.sum()
            loss.backward()
            
            # Perform optimization step to populate optimizer state
            optimizer.step()
            
            # Save optimizer state
            state_before = {k: v.clone() if isinstance(v, torch.Tensor) else v 
                           for k, v in optimizer.state_dict()['state'].items()}
            
            # Step the scheduler
            scheduler.step()
            
            # Verify optimizer state is preserved (except for lr)
            state_after = optimizer.state_dict()['state']
            
            # Check that state keys are the same
            assert set(state_before.keys()) == set(state_after.keys())
            
            # For each parameter state, check that momentum/statistics are preserved
            for param_id in state_before:
                for k in state_before[param_id]:
                    if k != 'lr':  # Learning rate should change
                        if isinstance(state_before[param_id][k], torch.Tensor):
                            assert torch.allclose(state_before[param_id][k], state_after[param_id][k])
                        else:
                            assert state_before[param_id][k] == state_after[param_id][k]
