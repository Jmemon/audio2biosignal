"""
Tests for the LossBuilder class.

This suite verifies:
1. Correct instantiation of various loss functions
2. Error handling for unsupported loss functions
3. Parameter validation
"""

import pytest
import torch
import torch.nn as nn
from unittest.mock import patch, MagicMock

from src.configs import LossConfig
from src.loss import LossBuilder


class TestLossBuilder:
    """Test suite for the LossBuilder class."""

    class TestSupportedLossFunctions:
        """Tests for successfully building supported loss functions."""

        def test_build_mse_loss(self):
            """
            GIVEN a loss config with name 'mse'
            WHEN LossBuilder.build is called
            THEN it should return an MSELoss instance
            """
            # Arrange
            loss_config = LossConfig(name="mse")
            
            # Act
            loss_fn = LossBuilder.build(loss_config)
            
            # Assert
            assert isinstance(loss_fn, nn.MSELoss)

        def test_build_l1_loss(self):
            """
            GIVEN a loss config with name 'l1'
            WHEN LossBuilder.build is called
            THEN it should return an L1Loss instance
            """
            # Arrange
            loss_config = LossConfig(name="l1")
            
            # Act
            loss_fn = LossBuilder.build(loss_config)
            
            # Assert
            assert isinstance(loss_fn, nn.L1Loss)

        def test_build_huber_loss(self):
            """
            GIVEN a loss config with name 'huber'
            WHEN LossBuilder.build is called
            THEN it should return a SmoothL1Loss instance
            """
            # Arrange
            loss_config = LossConfig(name="huber")
            
            # Act
            loss_fn = LossBuilder.build(loss_config)
            
            # Assert
            assert isinstance(loss_fn, nn.SmoothL1Loss)

    class TestErrorHandling:
        """Tests for error handling scenarios."""

        def test_unsupported_loss_function(self):
            """
            GIVEN a loss config with an unsupported name
            WHEN LossBuilder.build is called
            THEN it should raise a ValueError with appropriate message
            """
            # Arrange
            loss_config = LossConfig(name="unsupported_loss")
            
            # Act & Assert
            with pytest.raises(ValueError) as excinfo:
                LossBuilder.build(loss_config)
            
            assert "Unsupported loss function: unsupported_loss" in str(excinfo.value)

    class TestFunctionalBehavior:
        """Tests for the functional behavior of built loss functions."""

        @pytest.mark.parametrize("loss_name,loss_class", [
            ("mse", nn.MSELoss),
            ("l1", nn.L1Loss),
            ("huber", nn.SmoothL1Loss),
        ])
        def test_loss_function_computation(self, loss_name, loss_class):
            """
            GIVEN a loss function built by LossBuilder
            WHEN the loss function is used to compute loss between predictions and targets
            THEN it should compute the same result as directly instantiating the loss function
            """
            # Arrange
            loss_config = LossConfig(name=loss_name)
            builder_loss = LossBuilder.build(loss_config)
            direct_loss = loss_class()
            
            # Create sample tensors
            predictions = torch.tensor([0.1, 0.2, 0.3], dtype=torch.float32)
            targets = torch.tensor([0.0, 0.0, 1.0], dtype=torch.float32)
            
            # Act
            builder_result = builder_loss(predictions, targets)
            direct_result = direct_loss(predictions, targets)
            
            # Assert
            assert torch.isclose(builder_result, direct_result)
            assert builder_result.item() > 0  # Loss should be positive

    class TestCustomLoss:
        """Tests for the custom loss function path."""
        
        @patch("src.loss.nn.Module")
        def test_custom_loss_placeholder(self, mock_module):
            """
            GIVEN a loss config with name 'custom'
            WHEN LossBuilder.build is called
            THEN it should not raise an error (placeholder implementation)
            
            Note: This test will need to be updated once custom loss is implemented.
            """
            # This test is a placeholder that will pass with the current implementation
            # but should be updated when custom loss is implemented
            
            # Arrange
            loss_config = LossConfig(name="custom")
            
            # Act
            # Currently this will just pass through the placeholder
            # No assertion needed as we're just verifying it doesn't raise an exception
            LossBuilder.build(loss_config)
            
    class TestInputValidation:
        """Tests for input validation."""
        
        def test_none_config(self):
            """
            GIVEN a None loss config
            WHEN LossBuilder.build is called
            THEN it should raise a TypeError
            """
            # Act & Assert
            with pytest.raises(TypeError):
                LossBuilder.build(None)
                
        def test_wrong_config_type(self):
            """
            GIVEN a config of incorrect type (not LossConfig)
            WHEN LossBuilder.build is called
            THEN it should raise a TypeError
            """
            # Arrange
            invalid_config = {"name": "mse"}  # Dictionary instead of LossConfig
            
            # Act & Assert
            with pytest.raises(TypeError):
                LossBuilder.build(invalid_config)
                
    class TestIntegration:
        """Integration tests with actual PyTorch operations."""
        
        @pytest.mark.parametrize("loss_name", ["mse", "l1", "huber"])
        def test_loss_backpropagation(self, loss_name):
            """
            GIVEN a loss function built by LossBuilder
            WHEN used in a simple backpropagation scenario
            THEN it should compute gradients correctly
            """
            # Arrange
            loss_config = LossConfig(name=loss_name)
            loss_fn = LossBuilder.build(loss_config)
            
            # Create a simple tensor that requires gradients
            x = torch.tensor([0.5, 1.5, 2.5], dtype=torch.float32, requires_grad=True)
            target = torch.tensor([1.0, 2.0, 3.0], dtype=torch.float32)
            
            # Act
            output = loss_fn(x, target)
            output.backward()
            
            # Assert
            assert x.grad is not None
            assert not torch.isnan(x.grad).any(), f"NaN gradients found for {loss_name}"
            assert not torch.isinf(x.grad).any(), f"Infinite gradients found for {loss_name}"
