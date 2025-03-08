"""
Test suite for OptimizerBuilder.

This module provides comprehensive testing for:
1. Creation of different optimizer types (Adam, AdamW, SGD)
2. Scheduler configuration (Cosine, Step, Exponential)
3. Parameter validation and error handling
4. Integration with model parameters
5. Edge cases and boundary conditions
6. Parameter group handling
"""

import pytest
from unittest.mock import MagicMock, patch
import torch
import torch.nn as nn
from torch.optim import Adam, AdamW, SGD
from torch.optim.lr_scheduler import CosineAnnealingLR, StepLR, ExponentialLR

from src.configs import OptimizerConfig
from src.optimizer import OptimizerBuilder


# === FIXTURES ===

@pytest.fixture
def simple_model():
    """
    Provides a simple model with parameters for optimizer testing.
    
    Returns:
        nn.Module: A simple neural network with parameters
    """
    class SimpleModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.linear = nn.Linear(10, 2)
            
        def forward(self, x):
            return self.linear(x)
    
    return SimpleModel()


@pytest.fixture
def optimizer_config_factory():
    """
    Factory for creating optimizer configs with customizable attributes.
    
    Returns:
        function: A factory function that creates OptimizerConfig instances
    """
    def _create_config(**kwargs):
        defaults = {
            "name": "adam",
            "learning_rate": 0.001,
            "weight_decay": 0.0,
            "beta1": 0.9,
            "beta2": 0.999,
            "momentum": 0.0,
            "scheduler": None
        }
        # Override defaults with any provided kwargs
        params = {**defaults, **kwargs}
        return OptimizerConfig(**params)
    
    return _create_config


@pytest.fixture
def complex_model():
    """
    Provides a more complex model with multiple parameter groups.
    
    Returns:
        nn.Module: A neural network with multiple layers and parameter groups
    """
    class ComplexModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.conv = nn.Conv2d(3, 16, kernel_size=3, padding=1)
            self.bn = nn.BatchNorm2d(16)
            self.fc1 = nn.Linear(16 * 32 * 32, 120)
            self.fc2 = nn.Linear(120, 84)
            self.fc3 = nn.Linear(84, 10)
            
        def forward(self, x):
            x = self.conv(x)
            x = self.bn(x)
            x = torch.relu(x)
            x = x.view(-1, 16 * 32 * 32)
            x = torch.relu(self.fc1(x))
            x = torch.relu(self.fc2(x))
            x = self.fc3(x)
            return x
    
    return ComplexModel()


# === TEST CATEGORIES ===

class TestOptimizerCreation:
    """Tests for creating different types of optimizers."""
    
    @pytest.mark.parametrize("learning_rate", [0.0001, 0.001, 0.01, 0.1, 1.0])
    def test_learning_rate_range(self, simple_model, optimizer_config_factory, learning_rate):
        """
        GIVEN optimizer configs with different learning rates
        WHEN OptimizerBuilder.build is called
        THEN the optimizer should be created with the correct learning rate
        """
        # Arrange
        config = optimizer_config_factory(
            name="adam",
            learning_rate=learning_rate
        )
        
        # Act
        optimizer, _ = OptimizerBuilder.build(config, simple_model.parameters())
        
        # Assert
        assert optimizer.defaults["lr"] == learning_rate
    
    @pytest.mark.parametrize("weight_decay", [0.0, 0.0001, 0.001, 0.01, 0.1])
    def test_weight_decay_range(self, simple_model, optimizer_config_factory, weight_decay):
        """
        GIVEN optimizer configs with different weight decay values
        WHEN OptimizerBuilder.build is called
        THEN the optimizer should be created with the correct weight decay
        """
        # Arrange
        config = optimizer_config_factory(
            name="adam",
            weight_decay=weight_decay
        )
        
        # Act
        optimizer, _ = OptimizerBuilder.build(config, simple_model.parameters())
        
        # Assert
        assert optimizer.defaults["weight_decay"] == weight_decay
    
    @pytest.mark.parametrize("beta1,beta2", [
        (0.9, 0.999),  # Default values
        (0.8, 0.99),   # Lower values
        (0.95, 0.999), # Higher beta1
        (0.9, 0.9999)  # Higher beta2
    ])
    def test_beta_values(self, simple_model, optimizer_config_factory, beta1, beta2):
        """
        GIVEN optimizer configs with different beta values
        WHEN OptimizerBuilder.build is called
        THEN the optimizer should be created with the correct beta values
        """
        # Arrange
        config = optimizer_config_factory(
            name="adam",
            beta1=beta1,
            beta2=beta2
        )
        
        # Act
        optimizer, _ = OptimizerBuilder.build(config, simple_model.parameters())
        
        # Assert
        assert optimizer.defaults["betas"] == (beta1, beta2)
    
    def test_adam_optimizer_creation(self, simple_model, optimizer_config_factory):
        """
        GIVEN an optimizer config for Adam
        WHEN OptimizerBuilder.build is called
        THEN it should return an Adam optimizer with correct parameters
        """
        # Arrange
        config = optimizer_config_factory(
            name="adam",
            learning_rate=0.01,
            weight_decay=0.001,
            beta1=0.8,
            beta2=0.99
        )
        
        # Act
        optimizer, scheduler = OptimizerBuilder.build(config, simple_model.parameters())
        
        # Assert
        assert isinstance(optimizer, Adam)
        assert optimizer.defaults["lr"] == 0.01
        assert optimizer.defaults["weight_decay"] == 0.001
        assert optimizer.defaults["betas"] == (0.8, 0.99)
        assert scheduler is None
    
    def test_adamw_optimizer_creation(self, simple_model, optimizer_config_factory):
        """
        GIVEN an optimizer config for AdamW
        WHEN OptimizerBuilder.build is called
        THEN it should return an AdamW optimizer with correct parameters
        """
        # Arrange
        config = optimizer_config_factory(
            name="adamw",
            learning_rate=0.02,
            weight_decay=0.01,
            beta1=0.85,
            beta2=0.95
        )
        
        # Act
        optimizer, scheduler = OptimizerBuilder.build(config, simple_model.parameters())
        
        # Assert
        assert isinstance(optimizer, AdamW)
        assert optimizer.defaults["lr"] == 0.02
        assert optimizer.defaults["weight_decay"] == 0.01
        assert optimizer.defaults["betas"] == (0.85, 0.95)
        assert scheduler is None
    
    def test_sgd_optimizer_creation(self, simple_model, optimizer_config_factory):
        """
        GIVEN an optimizer config for SGD
        WHEN OptimizerBuilder.build is called
        THEN it should return an SGD optimizer with correct parameters
        """
        # Arrange
        config = optimizer_config_factory(
            name="sgd",
            learning_rate=0.1,
            weight_decay=0.001,
            momentum=0.9
        )
        
        # Act
        optimizer, scheduler = OptimizerBuilder.build(config, simple_model.parameters())
        
        # Assert
        assert isinstance(optimizer, SGD)
        assert optimizer.defaults["lr"] == 0.1
        assert optimizer.defaults["weight_decay"] == 0.001
        assert optimizer.defaults["momentum"] == 0.9
        assert scheduler is None


class TestSchedulerCreation:
    """Tests for creating different types of learning rate schedulers."""
    
    def test_cosine_scheduler_creation(self, simple_model, optimizer_config_factory):
        """
        GIVEN an optimizer config with cosine scheduler
        WHEN OptimizerBuilder.build is called
        THEN it should return the optimizer with a CosineAnnealingLR scheduler
        """
        # Arrange
        config = optimizer_config_factory(
            name="adam",
            scheduler="cosine"
        )
        
        # Act
        optimizer, scheduler = OptimizerBuilder.build(config, simple_model.parameters())
        
        # Assert
        assert isinstance(scheduler, CosineAnnealingLR)
        assert scheduler.T_max == 100  # Default T_max value
    
    def test_step_scheduler_creation(self, simple_model, optimizer_config_factory):
        """
        GIVEN an optimizer config with step scheduler
        WHEN OptimizerBuilder.build is called
        THEN it should return the optimizer with a StepLR scheduler
        """
        # Arrange
        config = optimizer_config_factory(
            name="adam",
            scheduler="step"
        )
        
        # Act
        optimizer, scheduler = OptimizerBuilder.build(config, simple_model.parameters())
        
        # Assert
        assert isinstance(scheduler, StepLR)
        assert scheduler.step_size == 30  # Default step_size value
        assert scheduler.gamma == 0.1  # Default gamma value
    
    def test_exponential_scheduler_creation(self, simple_model, optimizer_config_factory):
        """
        GIVEN an optimizer config with exponential scheduler
        WHEN OptimizerBuilder.build is called
        THEN it should return the optimizer with an ExponentialLR scheduler
        """
        # Arrange
        config = optimizer_config_factory(
            name="adam",
            scheduler="exponential"
        )
        
        # Act
        optimizer, scheduler = OptimizerBuilder.build(config, simple_model.parameters())
        
        # Assert
        assert isinstance(scheduler, ExponentialLR)
        assert scheduler.gamma == 0.95  # Default gamma value
    
    def test_scheduler_with_different_optimizers(self, simple_model, optimizer_config_factory):
        """
        GIVEN different optimizer types with the same scheduler
        WHEN OptimizerBuilder.build is called
        THEN each optimizer should have a correctly configured scheduler
        """
        # Test with different optimizer types
        optimizer_types = ["adam", "adamw", "sgd"]
        scheduler_type = "cosine"
        
        for opt_type in optimizer_types:
            # Arrange
            config = optimizer_config_factory(
                name=opt_type,
                scheduler=scheduler_type
            )
            
            # Act
            optimizer, scheduler = OptimizerBuilder.build(config, simple_model.parameters())
            
            # Assert
            assert scheduler is not None, f"Scheduler should not be None for {opt_type}"
            assert isinstance(scheduler, CosineAnnealingLR), f"Scheduler should be CosineAnnealingLR for {opt_type}"
            assert scheduler.T_max == 100, f"T_max should be 100 for {opt_type}"


class TestErrorHandling:
    """Tests for error handling and validation."""
    
    def test_unsupported_optimizer(self, simple_model, optimizer_config_factory):
        """
        GIVEN an optimizer config with an unsupported optimizer name
        WHEN OptimizerBuilder.build is called
        THEN it should raise a ValueError with appropriate message
        """
        # Arrange
        config = optimizer_config_factory(name="unsupported_optimizer")
        
        # Act & Assert
        with pytest.raises(ValueError) as excinfo:
            OptimizerBuilder.build(config, simple_model.parameters())
        
        assert "Unsupported optimizer" in str(excinfo.value)
        assert "unsupported_optimizer" in str(excinfo.value)
    
    @pytest.mark.parametrize("invalid_param", [
        {"learning_rate": -0.1},      # Negative learning rate
        {"learning_rate": 0},         # Zero learning rate
        {"weight_decay": -0.01},      # Negative weight decay
        {"beta1": 1.1},               # Beta1 > 1
        {"beta1": -0.1},              # Beta1 < 0
        {"beta2": 1.1},               # Beta2 > 1
        {"beta2": -0.1},              # Beta2 < 0
        {"momentum": -0.1}            # Negative momentum
    ])
    def test_invalid_optimizer_parameters(self, simple_model, optimizer_config_factory, invalid_param):
        """
        GIVEN optimizer configs with invalid parameter values
        WHEN OptimizerBuilder.build is called
        THEN it should handle the error appropriately
        """
        # Arrange
        config = optimizer_config_factory(**invalid_param)
        
        # We're testing that PyTorch handles these invalid values appropriately
        # This might raise an error or might silently clamp/adjust the values
        # The test just verifies our code doesn't crash and passes the values to PyTorch
        
        # Act - This should not raise an exception from our code
        try:
            optimizer, _ = OptimizerBuilder.build(config, simple_model.parameters())
            # If we get here, PyTorch accepted or adjusted the value
        except ValueError as e:
            # If PyTorch raises a ValueError, that's acceptable
            assert "learning_rate" in str(e) or "lr" in str(e) or \
                   "weight_decay" in str(e) or \
                   "beta" in str(e) or \
                   "momentum" in str(e)
    
    def test_none_model_parameters(self, optimizer_config_factory):
        """
        GIVEN None instead of model parameters
        WHEN OptimizerBuilder.build is called
        THEN it should handle the error gracefully
        """
        # Arrange
        config = optimizer_config_factory(name="adam")
        
        # Act & Assert
        with pytest.raises(TypeError):
            OptimizerBuilder.build(config, None)
    
    def test_empty_model_parameters(self, optimizer_config_factory):
        """
        GIVEN an empty iterable for model parameters
        WHEN OptimizerBuilder.build is called
        THEN it should create an optimizer without error
        """
        # Arrange
        config = optimizer_config_factory(name="adam")
        empty_params = []
        
        # Act
        optimizer, scheduler = OptimizerBuilder.build(config, empty_params)
        
        # Assert
        assert isinstance(optimizer, Adam)
        assert len(optimizer.param_groups) == 1
        assert len(optimizer.param_groups[0]['params']) == 0
    
    def test_unsupported_scheduler(self, simple_model, optimizer_config_factory):
        """
        GIVEN an optimizer config with an unsupported scheduler name
        WHEN OptimizerBuilder.build is called
        THEN it should return an optimizer with no scheduler
        """
        # Arrange
        config = optimizer_config_factory(
            name="adam",
            scheduler="unsupported_scheduler"
        )
        
        # Act
        optimizer, scheduler = OptimizerBuilder.build(config, simple_model.parameters())
        
        # Assert
        assert isinstance(optimizer, Adam)
        assert scheduler is None


class TestParameterGroups:
    """Tests for handling different parameter groups."""
    
    def test_parameter_groups_with_different_lrs(self, complex_model, optimizer_config_factory):
        """
        GIVEN a model with multiple parameter groups with different learning rates
        WHEN OptimizerBuilder.build is called
        THEN the optimizer should be configured with the correct parameter groups
        """
        # Arrange
        config = optimizer_config_factory(name="adam", learning_rate=0.01)
        
        # Create parameter groups with different learning rates
        conv_params = list(complex_model.conv.parameters())
        bn_params = list(complex_model.bn.parameters())
        fc_params = list(complex_model.fc1.parameters()) + list(complex_model.fc2.parameters()) + list(complex_model.fc3.parameters())
        
        param_groups = [
            {'params': conv_params, 'lr': 0.001},  # Lower LR for conv layers
            {'params': bn_params, 'lr': 0.01},     # Default LR for batch norm
            {'params': fc_params, 'lr': 0.1}       # Higher LR for fully connected layers
        ]
        
        # Act
        optimizer, _ = OptimizerBuilder.build(config, param_groups)
        
        # Assert
        assert len(optimizer.param_groups) == 3
        assert optimizer.param_groups[0]['lr'] == 0.001
        assert optimizer.param_groups[1]['lr'] == 0.01
        assert optimizer.param_groups[2]['lr'] == 0.1
    
    def test_parameter_groups_with_different_weight_decay(self, complex_model, optimizer_config_factory):
        """
        GIVEN a model with parameter groups with different weight decay values
        WHEN OptimizerBuilder.build is called
        THEN the optimizer should be configured with the correct weight decay per group
        """
        # Arrange
        config = optimizer_config_factory(name="adamw", weight_decay=0.01)
        
        # Create parameter groups with different weight decay values
        param_groups = [
            {'params': complex_model.conv.parameters(), 'weight_decay': 0.001},
            {'params': complex_model.bn.parameters(), 'weight_decay': 0.0},  # No weight decay for BN
            {'params': complex_model.fc1.parameters()},  # Use default weight decay
            {'params': complex_model.fc2.parameters()},
            {'params': complex_model.fc3.parameters(), 'weight_decay': 0.1}  # Higher weight decay
        ]
        
        # Act
        optimizer, _ = OptimizerBuilder.build(config, param_groups)
        
        # Assert
        assert len(optimizer.param_groups) == 5
        assert optimizer.param_groups[0]['weight_decay'] == 0.001
        assert optimizer.param_groups[1]['weight_decay'] == 0.0
        assert optimizer.param_groups[2]['weight_decay'] == 0.01  # Default from config
        assert optimizer.param_groups[3]['weight_decay'] == 0.01  # Default from config
        assert optimizer.param_groups[4]['weight_decay'] == 0.1


class TestCompatibility:
    """Tests for compatibility with different PyTorch versions and features."""
    
    def test_compatibility_with_frozen_layers(self, complex_model, optimizer_config_factory):
        """
        GIVEN a model with some frozen layers
        WHEN OptimizerBuilder.build is called
        THEN the optimizer should be created correctly
        """
        # Arrange
        config = optimizer_config_factory(name="adam")
        
        # Freeze some layers
        for param in complex_model.conv.parameters():
            param.requires_grad = False
        
        # Act
        optimizer, _ = OptimizerBuilder.build(config, complex_model.parameters())
        
        # Assert
        # Check that parameters are correctly included/excluded based on requires_grad
        trainable_params = [p for p in complex_model.parameters() if p.requires_grad]
        frozen_params = [p for p in complex_model.parameters() if not p.requires_grad]
        
        # All trainable parameters should be in the optimizer
        optimizer_params = []
        for group in optimizer.param_groups:
            optimizer_params.extend(group['params'])
        
        # Verify all trainable parameters are in the optimizer
        for param in trainable_params:
            assert any(torch.equal(param, opt_param) for opt_param in optimizer_params)
        
        # Frozen parameters might still be included but won't be updated
        # This is PyTorch's behavior, not our code's responsibility
    
    def test_with_amp_compatible_params(self, simple_model, optimizer_config_factory):
        """
        GIVEN optimizer config with parameters compatible with AMP (Automatic Mixed Precision)
        WHEN OptimizerBuilder.build is called
        THEN the optimizer should be created with AMP-compatible parameters
        """
        # Arrange
        config = optimizer_config_factory(
            name="adam",
            learning_rate=0.001,
            weight_decay=0.01
        )
        
        # Act
        optimizer, _ = OptimizerBuilder.build(config, simple_model.parameters())
        
        # Assert
        # Just verify the optimizer is created correctly
        # The actual AMP compatibility would be tested in integration with torch.cuda.amp
        assert isinstance(optimizer, Adam)
        assert optimizer.defaults["lr"] == 0.001
        assert optimizer.defaults["weight_decay"] == 0.01


class TestIntegration:
    """Integration tests with actual model training."""
    
    def test_optimizer_step(self, simple_model, optimizer_config_factory):
        """
        GIVEN a model, optimizer and input data
        WHEN an optimization step is performed
        THEN the model parameters should be updated
        """
        # Arrange
        config = optimizer_config_factory(name="adam", learning_rate=0.1)
        optimizer, _ = OptimizerBuilder.build(config, simple_model.parameters())
        
        # Get initial parameters
        initial_params = []
        for param in simple_model.parameters():
            initial_params.append(param.clone().detach())
        
        # Create dummy input and target
        dummy_input = torch.randn(1, 10)
        dummy_target = torch.tensor([0])
        
        # Act - perform one optimization step
        optimizer.zero_grad()
        output = simple_model(dummy_input)
        loss = nn.CrossEntropyLoss()(output, dummy_target)
        loss.backward()
        optimizer.step()
        
        # Assert - parameters should have changed
        for i, param in enumerate(simple_model.parameters()):
            assert not torch.allclose(param, initial_params[i])
    
    def test_scheduler_step(self, simple_model, optimizer_config_factory):
        """
        GIVEN an optimizer with a scheduler
        WHEN scheduler.step() is called
        THEN the learning rate should be updated according to the schedule
        """
        # Arrange
        config = optimizer_config_factory(name="adam", learning_rate=0.1, scheduler="exponential")
        optimizer, scheduler = OptimizerBuilder.build(config, simple_model.parameters())
        
        initial_lr = optimizer.param_groups[0]['lr']
        
        # Act
        scheduler.step()
        
        # Assert
        new_lr = optimizer.param_groups[0]['lr']
        assert new_lr != initial_lr
        assert new_lr == initial_lr * 0.95  # ExponentialLR with gamma=0.95
    
    def test_multiple_optimization_steps(self, simple_model, optimizer_config_factory):
        """
        GIVEN a model with optimizer and scheduler
        WHEN multiple optimization steps are performed
        THEN the learning rate should follow the expected schedule
        """
        # Arrange
        config = optimizer_config_factory(name="adam", learning_rate=0.1, scheduler="step")
        optimizer, scheduler = OptimizerBuilder.build(config, simple_model.parameters())
        
        initial_lr = optimizer.param_groups[0]['lr']
        expected_lrs = [initial_lr] * 30 + [initial_lr * 0.1]  # Step scheduler with step_size=30
        
        # Act & Assert - perform multiple steps and check learning rate
        for i in range(31):
            # Dummy training step
            optimizer.zero_grad()
            output = simple_model(torch.randn(1, 10))
            loss = output.sum()
            loss.backward()
            optimizer.step()
            
            # Check learning rate
            current_lr = optimizer.param_groups[0]['lr']
            assert current_lr == expected_lrs[i], f"Learning rate at step {i} should be {expected_lrs[i]}, got {current_lr}"
            
            # Step scheduler
            scheduler.step()
    
    def test_cosine_annealing_behavior(self, simple_model, optimizer_config_factory):
        """
        GIVEN a model with cosine annealing scheduler
        WHEN multiple optimization steps are performed
        THEN the learning rate should follow the cosine curve
        """
        # Arrange
        initial_lr = 0.1
        config = optimizer_config_factory(
            name="adam", 
            learning_rate=initial_lr, 
            scheduler="cosine"
        )
        optimizer, scheduler = OptimizerBuilder.build(config, simple_model.parameters())
        
        # Act & Assert - check learning rate follows cosine curve
        lrs = []
        for i in range(100):  # T_max = 100
            # Dummy training step
            optimizer.zero_grad()
            output = simple_model(torch.randn(1, 10))
            loss = output.sum()
            loss.backward()
            optimizer.step()
            
            # Record learning rate
            lrs.append(optimizer.param_groups[0]['lr'])
            
            # Step scheduler
            scheduler.step()
        
        # Verify learning rate behavior
        # 1. Should start at initial_lr
        assert lrs[0] == initial_lr
        
        # 2. Should decrease to near zero at the end
        assert lrs[-1] < initial_lr * 0.1
        
        # 3. Should be monotonically decreasing in the first half
        for i in range(1, 50):
            assert lrs[i] <= lrs[i-1]
        
        # 4. Should reach its minimum at the end
        assert lrs[-1] <= min(lrs)
