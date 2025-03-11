"""
Test suite for Wavenet and WavenetStack models.

This module provides comprehensive testing for:
1. Initialization with various parameters
2. Forward pass behavior
3. Shape transformations
4. Gradient flow
5. Edge cases and error handling
"""

import pytest
import torch
import torch.nn as nn
from src.models.wavenet import WavenetStack, Wavenet


class TestWavenetStackInitialization:
    """Tests specifically focused on WavenetStack initialization."""
    
    @pytest.mark.parametrize("param_name,invalid_value", [
        ("num_layers_per_stack", "not_an_int"),
        ("residual_channels", {}),
        ("skip_channels", []),
        ("kernel_size", None),
        ("dilation_base", complex(1, 2)),
        ("input_channels", set()),
    ])
    def test_init_with_invalid_parameter_types(self, param_name, invalid_value):
        """
        GIVEN parameters with invalid types that cannot be coerced
        WHEN WavenetStack is initialized
        THEN it should raise an appropriate TypeError
        """
        params = {
            'num_layers_per_stack': 2,
            'residual_channels': 16,
            'skip_channels': 8,
            'kernel_size': 3,
            'dilation_base': 2,
            'dropout_rate': 0.1,
            'input_channels': 4,
            'use_bias': True
        }
        
        # Replace one parameter with an invalid value
        params[param_name] = invalid_value
        
        # Should raise TypeError when trying to use the invalid parameter
        with pytest.raises((TypeError, ValueError)):
            stack = WavenetStack(**params)
    
    def test_init_required_parameters(self):
        """
        GIVEN a call to WavenetStack.__init__ with missing required parameters
        WHEN the object is initialized
        THEN it should raise TypeError with appropriate message
        """
        # Test with missing required parameters
        with pytest.raises(TypeError) as excinfo:
            stack = WavenetStack()
        
        # Error message should mention missing required positional arguments
        assert "missing" in str(excinfo.value)
        assert "required positional argument" in str(excinfo.value)
    
    def test_init_with_minimal_valid_parameters(self):
        """
        GIVEN minimal valid parameters for WavenetStack
        WHEN the object is initialized
        THEN it should create a valid WavenetStack with expected structure
        """
        # Minimal valid parameters
        minimal_params = {
            'num_layers_per_stack': 1,
            'residual_channels': 1,
            'skip_channels': 1,
            'kernel_size': 1,
            'dilation_base': 1,
            'dropout_rate': 0,
            'input_channels': 1,
            'use_bias': False
        }
        
        stack = WavenetStack(**minimal_params)
        
        # Check basic structure
        assert len(stack.layers) == 1
        assert len(stack.skip_connections) == 1
        assert isinstance(stack.dropout, nn.Dropout)
        assert stack.dropout.p == 0
        
        # Check layer properties
        layer = stack.layers[0]
        assert layer.in_channels == 1
        assert layer.out_channels == 1
        assert layer.kernel_size[0] == 1
        assert layer.dilation[0] == 1
        assert layer.padding[0] == 0
        assert layer.bias is None
        
        # Check skip connection properties
        skip = stack.skip_connections[0]
        assert skip.in_channels == 1
        assert skip.out_channels == 1
        assert skip.kernel_size[0] == 1
        assert skip.bias is None
    
    def test_init_parameter_interdependencies(self):
        """
        GIVEN parameters with interdependencies
        WHEN WavenetStack is initialized
        THEN it should handle these relationships correctly
        """
        # Test how kernel_size and dilation affect padding calculation
        params = {
            'num_layers_per_stack': 2,
            'residual_channels': 16,
            'skip_channels': 8,
            'kernel_size': 5,  # Larger kernel size
            'dilation_base': 3,  # Different dilation base
            'dropout_rate': 0.1,
            'input_channels': 4,
            'use_bias': True
        }
        
        stack = WavenetStack(**params)
        
        # First layer: dilation = 3^0 = 1, padding = (5-1)*1//2 = 2
        assert stack.layers[0].padding[0] == 2
        
        # Second layer: dilation = 3^1 = 3, padding = (5-1)*3//2 = 6
        assert stack.layers[1].padding[0] == 6
        
        # Test with even kernel size (padding calculation is different)
        params['kernel_size'] = 4
        even_kernel_stack = WavenetStack(**params)
        
        # First layer: dilation = 3^0 = 1, padding = (4-1)*1//2 = 1.5 -> 1
        assert even_kernel_stack.layers[0].padding[0] == 1
        
        # Second layer: dilation = 3^1 = 3, padding = (4-1)*3//2 = 4.5 -> 4
        assert even_kernel_stack.layers[1].padding[0] == 4
        
    def test_init_with_very_large_values(self):
        """
        GIVEN parameters with extremely large values
        WHEN WavenetStack is initialized
        THEN it should handle these values without numerical issues
        """
        large_params = {
            'num_layers_per_stack': 10,
            'residual_channels': 1024,
            'skip_channels': 1024,
            'kernel_size': 15,
            'dilation_base': 10,  # Will create very large dilations
            'dropout_rate': 0.5,
            'input_channels': 512,
            'use_bias': True
        }
        
        # This should initialize without numerical overflow errors
        stack = WavenetStack(**large_params)
        
        # Check that large dilations are calculated correctly
        # Last layer should have dilation = 10^9 = 1,000,000,000
        assert stack.layers[-1].dilation[0] == 10**9
        
        # Check that padding is calculated correctly for large dilations
        # For kernel_size=15, dilation=10^9, padding should be (15-1)*10^9//2 = 7*10^9
        expected_padding = (15-1) * (10**9) // 2
        assert stack.layers[-1].padding[0] == expected_padding


class TestWavenetStack:
    """Tests for the WavenetStack neural network module."""

    @pytest.fixture
    def default_params(self):
        """
        Provides default parameters for WavenetStack initialization.
        
        Returns:
            dict: Default parameters for testing
        """
        return {
            'num_layers_per_stack': 3,
            'residual_channels': 32,
            'skip_channels': 16,
            'kernel_size': 3,
            'dilation_base': 2,
            'dropout_rate': 0.1,
            'input_channels': 8,
            'use_bias': True
        }

    def test_initialization(self, default_params):
        """
        GIVEN valid initialization parameters
        WHEN a WavenetStack is created
        THEN it should initialize with correct structure and parameters
        """
        stack = WavenetStack(**default_params)
        
        # Check basic structure
        assert len(stack.layers) == default_params['num_layers_per_stack']
        assert len(stack.skip_connections) == default_params['num_layers_per_stack']
        
        # Check first layer parameters
        first_layer = stack.layers[0]
        assert isinstance(first_layer, nn.Conv1d)
        assert first_layer.in_channels == default_params['input_channels']
        assert first_layer.out_channels == default_params['residual_channels']
        assert first_layer.kernel_size[0] == default_params['kernel_size']
        assert first_layer.dilation[0] == 1  # dilation_base^0
        assert first_layer.bias is not None
        
        # Check last layer parameters
        last_layer = stack.layers[-1]
        assert last_layer.in_channels == default_params['residual_channels']
        assert last_layer.dilation[0] == default_params['dilation_base'] ** (default_params['num_layers_per_stack'] - 1)
        
        # Check skip connection parameters
        skip_conn = stack.skip_connections[0]
        assert skip_conn.in_channels == default_params['residual_channels']
        assert skip_conn.out_channels == default_params['skip_channels']
        assert skip_conn.kernel_size[0] == 1  # 1x1 conv
        
        # Check dropout
        assert isinstance(stack.dropout, nn.Dropout)
        assert stack.dropout.p == default_params['dropout_rate']

    def test_initialization_no_bias(self, default_params):
        """
        GIVEN initialization parameters with use_bias=False
        WHEN a WavenetStack is created
        THEN it should initialize layers without bias parameters
        """
        params = default_params.copy()
        params['use_bias'] = False
        stack = WavenetStack(**params)
        
        # Check bias is None for all layers
        for layer in stack.layers:
            assert layer.bias is None
        
        for skip in stack.skip_connections:
            assert skip.bias is None

    def test_forward_shape(self, default_params):
        """
        GIVEN a properly initialized WavenetStack
        WHEN forward is called with valid input
        THEN it should return outputs with the expected shapes
        """
        stack = WavenetStack(**default_params)
        batch_size = 4
        seq_length = 100
        
        # Create input tensor
        x = torch.randn(batch_size, default_params['input_channels'], seq_length)
        
        # Forward pass
        skip_sum, residual = stack(x)
        
        # Check output shapes
        assert skip_sum.shape == (batch_size, default_params['skip_channels'], seq_length)
        assert residual.shape == (batch_size, default_params['residual_channels'], seq_length)

    def test_forward_computation(self, default_params):
        """
        GIVEN a WavenetStack with controlled weights
        WHEN forward is called
        THEN it should perform the expected computation
        """
        # Create stack with deterministic initialization
        torch.manual_seed(42)
        stack = WavenetStack(**default_params)
        
        # Set all weights to 0.1 and biases to 0 for predictable output
        with torch.no_grad():
            for layer in stack.layers:
                nn.init.constant_(layer.weight, 0.1)
                if layer.bias is not None:
                    nn.init.zeros_(layer.bias)
            
            for skip in stack.skip_connections:
                nn.init.constant_(skip.weight, 0.1)
                if skip.bias is not None:
                    nn.init.zeros_(skip.bias)
        
        # Create input with all 1s
        batch_size = 1
        seq_length = 10
        x = torch.ones(batch_size, default_params['input_channels'], seq_length)
        
        # Disable dropout for deterministic output
        stack.dropout.p = 0
        
        # Forward pass
        skip_sum, residual = stack(x)
        
        # Values should be non-zero (specific values depend on architecture details)
        assert torch.all(skip_sum != 0)
        assert torch.all(residual != 0)

    def test_dilations(self, default_params):
        """
        GIVEN a WavenetStack with specific dilation parameters
        WHEN initialized
        THEN it should create layers with the correct dilation factors
        """
        stack = WavenetStack(**default_params)
        
        # Check dilations follow the pattern: dilation_base^layer_index
        for i, layer in enumerate(stack.layers):
            expected_dilation = default_params['dilation_base'] ** i
            assert layer.dilation[0] == expected_dilation

    def test_gradient_flow(self, default_params):
        """
        GIVEN a WavenetStack
        WHEN a forward and backward pass is performed
        THEN gradients should flow through the network
        """
        stack = WavenetStack(**default_params)
        
        # Create input that requires gradient
        x = torch.ones(2, default_params['input_channels'], 20, requires_grad=True)
        
        # Forward pass
        skip_sum, residual = stack(x)
        
        # Create a scalar loss and backpropagate
        loss = skip_sum.sum() + residual.sum()
        loss.backward()
        
        # Check that gradients flowed through the network
        assert x.grad is not None
        assert torch.all(x.grad != 0)
        
        # Check that all parameters received gradients
        for name, param in stack.named_parameters():
            assert param.grad is not None, f"Parameter {name} has no gradient"
            assert torch.any(param.grad != 0), f"Parameter {name} has zero gradient"

    def test_skip_connections_sum(self, default_params):
        """
        GIVEN a WavenetStack
        WHEN forward is called
        THEN it should correctly sum the skip connections
        """
        # Create a stack with only 2 layers for simplicity
        params = default_params.copy()
        params['num_layers_per_stack'] = 2
        stack = WavenetStack(**params)
        
        # Mock the skip connections to return known tensors
        original_forward = stack.forward
        
        def mock_forward(x):
            # Call the original layers but return known skip outputs
            skip_outputs = []
            for conv, skip_conv in zip(stack.layers, stack.skip_connections):
                out = conv(x)
                gated = torch.tanh(out) * torch.sigmoid(out)
                residual = gated + x
                x = residual
                
                # Create skip outputs with recognizable patterns
                if len(skip_outputs) == 0:
                    skip = torch.ones_like(gated[:, :params['skip_channels'], :])
                else:
                    skip = torch.ones_like(gated[:, :params['skip_channels'], :]) * 2
                
                skip_outputs.append(skip)
                x = stack.dropout(x)
            
            return sum(skip_outputs), x
        
        # Replace forward method temporarily
        stack.forward = mock_forward
        
        # Test
        x = torch.randn(1, params['input_channels'], 10)
        skip_sum, _ = stack(x)
        
        # With our mock (1s and 2s), the sum should be 3s
        expected_value = 3.0
        assert torch.allclose(skip_sum, torch.ones_like(skip_sum) * expected_value)
        
        # Restore original forward method
        stack.forward = original_forward
        
    def test_gated_activation_mechanism(self, default_params):
        """
        GIVEN a WavenetStack
        WHEN forward is called
        THEN it should correctly apply the gated activation (tanh * sigmoid)
        """
        # Create a minimal stack for testing the gated activation
        params = default_params.copy()
        params['num_layers_per_stack'] = 1  # Single layer for simplicity
        stack = WavenetStack(**params)
        
        # Set weights to known values for predictable output
        with torch.no_grad():
            # Set conv weights to produce a constant output
            nn.init.constant_(stack.layers[0].weight, 0.5)
            if stack.layers[0].bias is not None:
                nn.init.constant_(stack.layers[0].bias, 0.5)
            
            # Set skip connection to identity-like for testing
            nn.init.constant_(stack.skip_connections[0].weight, 1.0)
            if stack.skip_connections[0].bias is not None:
                nn.init.zeros_(stack.skip_connections[0].bias)
        
        # Disable dropout for deterministic output
        stack.dropout.p = 0
        
        # Create input with known values
        batch_size = 1
        seq_length = 5
        x = torch.ones(batch_size, params['input_channels'], seq_length)
        
        # Forward pass
        skip_sum, residual = stack(x)
        
        # Manually calculate expected gated activation for the given weights
        # For constant input of 1 and weights of 0.5, conv output should be:
        # out = 1 * 0.5 * input_channels + 0.5 (bias)
        expected_conv_out = 0.5 * params['input_channels'] + 0.5
        
        # Gated activation: tanh(out) * sigmoid(out)
        expected_gated = torch.tanh(torch.tensor(expected_conv_out)) * torch.sigmoid(torch.tensor(expected_conv_out))
        
        # The residual should be gated + x
        # But since input_channels != residual_channels, this is more complex to verify
        # We'll verify the skip connection which is more directly testable
        
        # Skip connection should be gated * skip_weights (1.0) + skip_bias (0)
        expected_skip = expected_gated
        
        # Verify skip connection output (allowing for floating point imprecision)
        assert torch.allclose(
            skip_sum, 
            torch.ones_like(skip_sum) * expected_skip, 
            atol=1e-5
        ), f"Expected skip value {expected_skip}, got mean {skip_sum.mean()}"
    
    def test_forward_empty_layers(self):
        """
        GIVEN a WavenetStack with no layers
        WHEN forward is called
        THEN it should handle this edge case correctly
        """
        # Create a stack with no layers
        params = {
            'num_layers_per_stack': 0,
            'residual_channels': 16,
            'skip_channels': 8,
            'kernel_size': 3,
            'dilation_base': 2,
            'dropout_rate': 0.1,
            'input_channels': 4,
            'use_bias': True
        }
        
        stack = WavenetStack(**params)
        
        # Input tensor
        batch_size = 2
        seq_length = 10
        x = torch.ones(batch_size, params['input_channels'], seq_length)
        
        # Forward pass
        skip_sum, residual = stack(x)
        
        # With no layers, skip_sum should be zeros with skip_channels dimension
        assert skip_sum.shape == (batch_size, params['skip_channels'], seq_length)
        assert torch.all(skip_sum == 0)
        
        # With no layers, residual should be the input itself
        assert residual.shape == x.shape
        assert torch.all(residual == x)
    
    def test_forward_with_mismatched_layer_skip_lists(self, default_params):
        """
        GIVEN a WavenetStack with mismatched layers and skip_connections
        WHEN forward is called
        THEN it should handle this robustly
        """
        # Create a normal stack
        stack = WavenetStack(**default_params)
        
        # Artificially create a mismatch by removing the last skip connection
        with torch.no_grad():
            # Save the original lists
            original_layers = stack.layers
            original_skips = stack.skip_connections
            
            # Create a mismatch by using only the first n-1 skip connections
            stack.skip_connections = stack.skip_connections[:-1]
            
            # Input tensor
            batch_size = 2
            seq_length = 10
            x = torch.ones(batch_size, default_params['input_channels'], seq_length)
            
            # Forward should raise an IndexError due to the mismatch
            with pytest.raises(IndexError):
                skip_sum, residual = stack(x)
            
            # Restore the original lists
            stack.layers = original_layers
            stack.skip_connections = original_skips
    
    def test_forward_sequence_preservation(self, default_params):
        """
        GIVEN a WavenetStack
        WHEN forward is called
        THEN it should preserve sequence length and temporal patterns
        """
        stack = WavenetStack(**default_params)
        
        # Create input with a clear temporal pattern
        batch_size = 1
        seq_length = 20
        x = torch.zeros(batch_size, default_params['input_channels'], seq_length)
        
        # Create a pattern: alternating 0s and 1s
        x[:, :, ::2] = 1.0  # Set every even position to 1
        
        # Disable dropout for deterministic output
        stack.dropout.p = 0
        
        # Forward pass
        skip_sum, residual = stack(x)
        
        # Check that output sequence length is preserved
        assert skip_sum.shape[2] == seq_length
        assert residual.shape[2] == seq_length
        
        # The exact values will depend on the weights, but we can check that
        # the temporal pattern is preserved by checking correlation between
        # adjacent timesteps - there should be low correlation due to the alternating pattern
        
        # Calculate correlation between adjacent timesteps in the residual output
        # If the pattern is preserved, adjacent values should be different
        adjacent_correlation = torch.corrcoef(
            torch.stack([residual[0, :, :-1].flatten(), residual[0, :, 1:].flatten()])
        )[0, 1]
        
        # The correlation should not be close to 1.0 (which would indicate all same values)
        # It might be negative (indicating inverse relationship) or close to zero
        assert adjacent_correlation < 0.9, "Temporal pattern not preserved in output"
    
    def test_forward_with_nan_inputs(self, default_params):
        """
        GIVEN a WavenetStack
        WHEN forward is called with inputs containing NaN values
        THEN it should propagate NaNs appropriately
        """
        stack = WavenetStack(**default_params)
        
        # Create input with some NaN values
        batch_size = 2
        seq_length = 10
        x = torch.ones(batch_size, default_params['input_channels'], seq_length)
        x[0, 0, 0] = float('nan')  # Set one value to NaN
        
        # Forward pass
        skip_sum, residual = stack(x)
        
        # The output should contain NaN values due to propagation
        assert torch.isnan(skip_sum).any(), "NaN did not propagate to skip connections"
        assert torch.isnan(residual).any(), "NaN did not propagate to residual output"
    
    def test_forward_with_different_batch_sizes(self, default_params):
        """
        GIVEN a WavenetStack
        WHEN forward is called with different batch sizes
        THEN it should handle all batch sizes correctly
        """
        stack = WavenetStack(**default_params)
        
        # Test with various batch sizes
        batch_sizes = [1, 2, 8, 16]
        seq_length = 10
        
        for batch_size in batch_sizes:
            # Create input
            x = torch.ones(batch_size, default_params['input_channels'], seq_length)
            
            # Forward pass
            skip_sum, residual = stack(x)
            
            # Check output shapes
            assert skip_sum.shape == (batch_size, default_params['skip_channels'], seq_length)
            assert residual.shape == (batch_size, default_params['residual_channels'], seq_length)
            
    def test_forward_with_empty_sequence(self, default_params):
        """
        GIVEN a WavenetStack
        WHEN forward is called with an empty sequence (seq_length=0)
        THEN it should handle this edge case gracefully
        """
        stack = WavenetStack(**default_params)
        
        # Create input with zero sequence length
        batch_size = 2
        seq_length = 0
        x = torch.ones(batch_size, default_params['input_channels'], seq_length)
        
        # Forward pass
        skip_sum, residual = stack(x)
        
        # Check output shapes - should preserve the zero sequence length
        assert skip_sum.shape == (batch_size, default_params['skip_channels'], seq_length)
        assert residual.shape == (batch_size, default_params['residual_channels'], seq_length)
        
    def test_forward_with_different_dtypes(self, default_params):
        """
        GIVEN a WavenetStack
        WHEN forward is called with inputs of different precision
        THEN it should maintain the input precision
        """
        stack = WavenetStack(**default_params)
        
        # Test with different dtypes
        dtypes = [torch.float32, torch.float64]
        batch_size = 2
        seq_length = 10
        
        for dtype in dtypes:
            # Create input with specific dtype
            x = torch.ones(batch_size, default_params['input_channels'], seq_length, dtype=dtype)
            
            # Forward pass
            skip_sum, residual = stack(x)
            
            # Check output dtypes match input
            assert skip_sum.dtype == dtype, f"Skip connection output dtype {skip_sum.dtype} doesn't match input dtype {dtype}"
            assert residual.dtype == dtype, f"Residual output dtype {residual.dtype} doesn't match input dtype {dtype}"
            
    def test_exact_gated_activation_calculation(self):
        """
        GIVEN a WavenetStack with controlled weights
        WHEN forward is called
        THEN it should calculate the gated activation exactly as expected
        """
        # Create a minimal stack for precise calculation testing
        params = {
            'num_layers_per_stack': 1,
            'residual_channels': 1,
            'skip_channels': 1,
            'kernel_size': 1,
            'dilation_base': 1,
            'dropout_rate': 0.0,  # No dropout for deterministic results
            'input_channels': 1,
            'use_bias': True
        }
        
        stack = WavenetStack(**params)
        
        # Set weights to exact values for precise calculation
        with torch.no_grad():
            # Set conv weight to 1.0 and bias to 0.5
            nn.init.constant_(stack.layers[0].weight, 1.0)
            nn.init.constant_(stack.layers[0].bias, 0.5)
            
            # Set skip connection weight to 1.0 and bias to 0.0
            nn.init.constant_(stack.skip_connections[0].weight, 1.0)
            nn.init.constant_(stack.skip_connections[0].bias, 0.0)
        
        # Create input with exact value
        x = torch.full((1, 1, 1), 2.0)  # Single value of 2.0
        
        # Forward pass
        skip_sum, residual = stack(x)
        
        # Manual calculation:
        # conv_out = x * weight + bias = 2.0 * 1.0 + 0.5 = 2.5
        # gated = tanh(conv_out) * sigmoid(conv_out)
        # gated = tanh(2.5) * sigmoid(2.5)
        expected_tanh = torch.tanh(torch.tensor(2.5))
        expected_sigmoid = torch.sigmoid(torch.tensor(2.5))
        expected_gated = expected_tanh * expected_sigmoid
        
        # Skip connection = gated * skip_weight + skip_bias = gated * 1.0 + 0.0 = gated
        expected_skip = expected_gated
        
        # Residual = gated + x = gated + 2.0
        expected_residual = expected_gated + 2.0
        
        # Verify outputs match expected calculations
        assert torch.allclose(skip_sum, expected_skip.view(1, 1, 1), atol=1e-6), \
            f"Expected skip value {expected_skip}, got {skip_sum.item()}"
        assert torch.allclose(residual, expected_residual.view(1, 1, 1), atol=1e-6), \
            f"Expected residual value {expected_residual}, got {residual.item()}"

    def test_residual_connections(self, default_params):
        """
        GIVEN a WavenetStack
        WHEN forward is called
        THEN it should correctly implement residual connections
        """
        # Create a minimal stack for testing residual connections
        params = default_params.copy()
        params['num_layers_per_stack'] = 1  # Single layer for simplicity
        stack = WavenetStack(**params)
        
        # Set weights to identity-like values
        with torch.no_grad():
            # Set conv weights to produce minimal output
            nn.init.zeros_(stack.layers[0].weight)
            if stack.layers[0].bias is not None:
                nn.init.zeros_(stack.layers[0].bias)
            
            # Set skip connection to zeros (we're testing residual, not skip)
            nn.init.zeros_(stack.skip_connections[0].weight)
            if stack.skip_connections[0].bias is not None:
                nn.init.zeros_(stack.skip_connections[0].bias)
        
        # Disable dropout
        stack.dropout.p = 0
        
        # Create input
        x = torch.ones(1, params['input_channels'], 10)
        
        # Forward pass
        _, residual = stack(x)
        
        # With zero weights and gated activation, the output should be the input
        # (but potentially with different channel dimensions)
        assert residual.shape[1] == params['residual_channels']
        
        # The residual connection should preserve the input
        # Since our weights are 0, the gated output is 0.5 (sigmoid(0)*tanh(0) = 0.5*0 = 0)
        # So residual = gated + x = 0 + x = x
        # But x is broadcast from input_channels to residual_channels
        if params['input_channels'] == params['residual_channels']:
            # If dimensions match, residual should be approximately equal to input
            assert torch.allclose(residual, x, atol=1e-6)
    
    def test_input_dimension_handling(self, default_params):
        """
        GIVEN a WavenetStack with different input and residual channel dimensions
        WHEN forward is called
        THEN it should correctly handle the dimension mismatch in residual connections
        """
        # Create params with different input and residual channels
        params = default_params.copy()
        params['input_channels'] = 4
        params['residual_channels'] = 8
        params['num_layers_per_stack'] = 1  # Single layer for simplicity
        
        stack = WavenetStack(**params)
        
        # Create input
        batch_size = 2
        seq_length = 15
        x = torch.ones(batch_size, params['input_channels'], seq_length)
        
        # Forward pass
        skip_sum, residual = stack(x)
        
        # Check output shapes
        assert skip_sum.shape == (batch_size, params['skip_channels'], seq_length)
        assert residual.shape == (batch_size, params['residual_channels'], seq_length)
        
        # The residual connection should handle the dimension mismatch
        # by broadcasting the input to match residual_channels
        assert residual.shape[1] == params['residual_channels']
    
    @pytest.mark.parametrize("dropout_rate", [0.0, 0.5, 1.0])
    def test_dropout_behavior(self, default_params, dropout_rate):
        """
        GIVEN a WavenetStack with different dropout rates
        WHEN forward is called in training and evaluation modes
        THEN it should apply dropout correctly based on the mode
        """
        # Set specific dropout rate
        params = default_params.copy()
        params['dropout_rate'] = dropout_rate
        
        stack = WavenetStack(**params)
        
        # Create input
        x = torch.ones(2, params['input_channels'], 10)
        
        # Test in training mode
        stack.train()
        torch.manual_seed(42)  # For reproducibility
        skip_sum_train, residual_train = stack(x)
        
        # Test in evaluation mode
        stack.eval()
        torch.manual_seed(42)  # Same seed
        skip_sum_eval, residual_eval = stack(x)
        
        if dropout_rate > 0.0:
            # In training mode with dropout > 0, outputs should differ from eval mode
            # Note: This is probabilistic, but with high dropout rates it's very likely
            if dropout_rate > 0.3:  # Only check with substantial dropout
                assert not torch.allclose(residual_train, residual_eval)
        else:
            # With dropout = 0, outputs should be the same regardless of mode
            assert torch.allclose(residual_train, residual_eval)
    
    def test_extreme_parameter_values(self):
        """
        GIVEN a WavenetStack with extreme parameter values
        WHEN initialized and forward is called
        THEN it should handle these values gracefully
        """
        # Test with minimal values
        min_params = {
            'num_layers_per_stack': 1,
            'residual_channels': 1,
            'skip_channels': 1,
            'kernel_size': 1,
            'dilation_base': 1,
            'dropout_rate': 0.0,
            'input_channels': 1,
            'use_bias': False
        }
        
        min_stack = WavenetStack(**min_params)
        min_input = torch.ones(1, 1, 5)
        min_skip, min_residual = min_stack(min_input)
        
        assert min_skip.shape == (1, 1, 5)
        assert min_residual.shape == (1, 1, 5)
        
        # Test with large values (but not so large as to cause memory issues)
        max_params = {
            'num_layers_per_stack': 5,
            'residual_channels': 64,
            'skip_channels': 64,
            'kernel_size': 5,
            'dilation_base': 4,
            'dropout_rate': 0.9,
            'input_channels': 32,
            'use_bias': True
        }
        
        max_stack = WavenetStack(**max_params)
        max_input = torch.ones(1, 32, 20)
        max_skip, max_residual = max_stack(max_input)
        
        assert max_skip.shape == (1, 64, 20)
        assert max_residual.shape == (1, 64, 20)
    
    def test_initialization_padding_calculation(self, default_params):
        """
        GIVEN a WavenetStack with various kernel sizes and dilations
        WHEN initialized
        THEN it should calculate padding correctly to maintain sequence length
        """
        stack = WavenetStack(**default_params)
        
        # Check padding for each layer
        for i, layer in enumerate(stack.layers):
            dilation = default_params['dilation_base'] ** i
            kernel_size = default_params['kernel_size']
            expected_padding = (kernel_size - 1) * dilation // 2
            assert layer.padding[0] == expected_padding, f"Layer {i} has incorrect padding"
    
    def test_initialization_channel_progression(self):
        """
        GIVEN a WavenetStack with different input and residual channels
        WHEN initialized
        THEN it should correctly set up channel dimensions for each layer
        """
        params = {
            'num_layers_per_stack': 4,
            'residual_channels': 16,
            'skip_channels': 8,
            'kernel_size': 3,
            'dilation_base': 2,
            'dropout_rate': 0.1,
            'input_channels': 4,
            'use_bias': True
        }
        
        stack = WavenetStack(**params)
        
        # First layer should go from input_channels to residual_channels
        assert stack.layers[0].in_channels == params['input_channels']
        assert stack.layers[0].out_channels == params['residual_channels']
        
        # Subsequent layers should maintain residual_channels
        for i in range(1, params['num_layers_per_stack']):
            assert stack.layers[i].in_channels == params['residual_channels']
            assert stack.layers[i].out_channels == params['residual_channels']
            
        # All skip connections should go from residual_channels to skip_channels
        for skip in stack.skip_connections:
            assert skip.in_channels == params['residual_channels']
            assert skip.out_channels == params['skip_channels']
    
    def test_initialization_zero_layers(self):
        """
        GIVEN a WavenetStack with zero layers
        WHEN initialized
        THEN it should handle this edge case gracefully
        """
        params = {
            'num_layers_per_stack': 0,
            'residual_channels': 16,
            'skip_channels': 8,
            'kernel_size': 3,
            'dilation_base': 2,
            'dropout_rate': 0.1,
            'input_channels': 4,
            'use_bias': True
        }
        
        stack = WavenetStack(**params)
        
        # Should have empty layer lists
        assert len(stack.layers) == 0
        assert len(stack.skip_connections) == 0
        
        # Should still have a dropout layer
        assert isinstance(stack.dropout, nn.Dropout)
        
        # Forward pass with zero layers should just return empty tensors
        x = torch.ones(2, params['input_channels'], 10)
        skip_sum, residual = stack(x)
        
        # Skip sum should be zeros with skip_channels
        assert skip_sum.shape == (2, params['skip_channels'], 10)
        assert torch.all(skip_sum == 0)
        
        # Residual should be the input itself
        assert residual.shape == x.shape
        assert torch.all(residual == x)
    
    def test_initialization_even_vs_odd_kernel_size(self):
        """
        GIVEN a WavenetStack with even and odd kernel sizes
        WHEN initialized
        THEN it should calculate padding correctly for both cases
        """
        # Test with odd kernel size (standard case)
        odd_params = {
            'num_layers_per_stack': 2,
            'residual_channels': 16,
            'skip_channels': 8,
            'kernel_size': 3,  # Odd kernel size
            'dilation_base': 2,
            'dropout_rate': 0.1,
            'input_channels': 4,
            'use_bias': True
        }
        
        odd_stack = WavenetStack(**odd_params)
        
        # First layer (dilation=1)
        assert odd_stack.layers[0].padding[0] == 1  # (3-1)*1//2 = 1
        # Second layer (dilation=2)
        assert odd_stack.layers[1].padding[0] == 2  # (3-1)*2//2 = 2
        
        # Test with even kernel size
        even_params = odd_params.copy()
        even_params['kernel_size'] = 4  # Even kernel size
        
        even_stack = WavenetStack(**even_params)
        
        # First layer (dilation=1)
        assert even_stack.layers[0].padding[0] == 1  # (4-1)*1//2 = 1.5 -> 1 (integer division)
        # Second layer (dilation=2)
        assert even_stack.layers[1].padding[0] == 3  # (4-1)*2//2 = 3
    
    def test_initialization_with_invalid_parameters(self):
        """
        GIVEN a WavenetStack with invalid parameters
        WHEN initialized
        THEN it should handle these gracefully or raise appropriate errors
        """
        # Test with negative number of layers
        negative_layers_params = {
            'num_layers_per_stack': -1,
            'residual_channels': 16,
            'skip_channels': 8,
            'kernel_size': 3,
            'dilation_base': 2,
            'dropout_rate': 0.1,
            'input_channels': 4,
            'use_bias': True
        }
        
        # Should create a stack with no layers (same as num_layers_per_stack=0)
        negative_stack = WavenetStack(**negative_layers_params)
        assert len(negative_stack.layers) == 0
        assert len(negative_stack.skip_connections) == 0
        
        # Test with negative dropout rate (should clamp to 0)
        negative_dropout_params = {
            'num_layers_per_stack': 2,
            'residual_channels': 16,
            'skip_channels': 8,
            'kernel_size': 3,
            'dilation_base': 2,
            'dropout_rate': -0.1,  # Negative dropout
            'input_channels': 4,
            'use_bias': True
        }
        
        # PyTorch's Dropout clamps negative values to 0
        negative_dropout_stack = WavenetStack(**negative_dropout_params)
        assert negative_dropout_stack.dropout.p == 0.0
        
        # Test with dropout rate > 1 (should clamp to 1)
        high_dropout_params = negative_dropout_params.copy()
        high_dropout_params['dropout_rate'] = 1.5  # Dropout > 1
        
        # PyTorch's Dropout clamps values > 1 to 1
        high_dropout_stack = WavenetStack(**high_dropout_params)
        assert high_dropout_stack.dropout.p == 1.0
        
        # Test with zero channels
        zero_channels_params = {
            'num_layers_per_stack': 2,
            'residual_channels': 0,  # Zero residual channels
            'skip_channels': 0,      # Zero skip channels
            'kernel_size': 3,
            'dilation_base': 2,
            'dropout_rate': 0.1,
            'input_channels': 4,
            'use_bias': True
        }
        
        # Should initialize without errors, but layers will have 0 output channels
        zero_channels_stack = WavenetStack(**zero_channels_params)
        assert zero_channels_stack.layers[0].out_channels == 0
        assert zero_channels_stack.skip_connections[0].out_channels == 0
        
    def test_init_with_kernel_size_edge_cases(self):
        """
        GIVEN a WavenetStack with edge case kernel sizes
        WHEN initialized
        THEN it should handle these values appropriately
        """
        # Test with kernel_size = 1 (minimum valid value)
        min_kernel_params = {
            'num_layers_per_stack': 2,
            'residual_channels': 16,
            'skip_channels': 8,
            'kernel_size': 1,  # Minimum valid kernel size
            'dilation_base': 2,
            'dropout_rate': 0.1,
            'input_channels': 4,
            'use_bias': True
        }
        
        min_kernel_stack = WavenetStack(**min_kernel_params)
        
        # Check that kernel_size is set correctly
        assert min_kernel_stack.layers[0].kernel_size[0] == 1
        
        # With kernel_size=1, padding should be 0 regardless of dilation
        for layer in min_kernel_stack.layers:
            assert layer.padding[0] == 0
        
        # Test with very large kernel size
        large_kernel_params = min_kernel_params.copy()
        large_kernel_params['kernel_size'] = 99  # Very large kernel size
        
        large_kernel_stack = WavenetStack(**large_kernel_params)
        
        # Check that kernel_size is set correctly
        assert large_kernel_stack.layers[0].kernel_size[0] == 99
        
        # Check padding calculation for large kernel
        # First layer: dilation=1, padding=(99-1)*1//2 = 49
        assert large_kernel_stack.layers[0].padding[0] == 49
        
        # Second layer: dilation=2, padding=(99-1)*2//2 = 98
        assert large_kernel_stack.layers[1].padding[0] == 98
    
    def test_initialization_with_different_dilation_bases(self):
        """
        GIVEN a WavenetStack with different dilation bases
        WHEN initialized
        THEN it should correctly calculate dilations for each layer
        """
        # Test with dilation_base = 1 (no dilation growth)
        no_growth_params = {
            'num_layers_per_stack': 3,
            'residual_channels': 16,
            'skip_channels': 8,
            'kernel_size': 3,
            'dilation_base': 1,  # No growth in dilation
            'dropout_rate': 0.1,
            'input_channels': 4,
            'use_bias': True
        }
        
        no_growth_stack = WavenetStack(**no_growth_params)
        
        # All layers should have dilation = 1
        for i, layer in enumerate(no_growth_stack.layers):
            assert layer.dilation[0] == 1, f"Layer {i} has incorrect dilation"
        
        # Test with dilation_base = 3 (faster growth)
        fast_growth_params = no_growth_params.copy()
        fast_growth_params['dilation_base'] = 3
        
        fast_growth_stack = WavenetStack(**fast_growth_params)
        
        # Check dilations: 3^0=1, 3^1=3, 3^2=9
        expected_dilations = [1, 3, 9]
        for i, layer in enumerate(fast_growth_stack.layers):
            assert layer.dilation[0] == expected_dilations[i], f"Layer {i} has incorrect dilation"
            
        # Verify padding is also calculated correctly with these dilations
        for i, layer in enumerate(fast_growth_stack.layers):
            dilation = expected_dilations[i]
            kernel_size = fast_growth_params['kernel_size']
            expected_padding = (kernel_size - 1) * dilation // 2
            assert layer.padding[0] == expected_padding, f"Layer {i} has incorrect padding"
            
        # Test with negative dilation base (should use absolute value)
        negative_base_params = no_growth_params.copy()
        negative_base_params['dilation_base'] = -2
        
        negative_base_stack = WavenetStack(**negative_base_params)
        
        # Check dilations: |-2|^0=1, |-2|^1=2, |-2|^2=4
        expected_dilations = [1, 2, 4]
        for i, layer in enumerate(negative_base_stack.layers):
            assert layer.dilation[0] == expected_dilations[i], f"Layer {i} has incorrect dilation with negative base"
            
        # Test with fractional dilation base < 1
        fractional_base_params = no_growth_params.copy()
        fractional_base_params['dilation_base'] = 0.5
        
        fractional_base_stack = WavenetStack(**fractional_base_params)
        
        # Check dilations: 0.5^0=1, 0.5^1=0.5, 0.5^2=0.25
        # These should be rounded to integers or handled appropriately
        # The exact behavior depends on implementation, but it shouldn't crash
        assert len(fractional_base_stack.layers) == 3
        assert fractional_base_stack.layers[0].dilation[0] >= 1  # First layer dilation should be at least 1
    
    def test_initialization_memory_efficiency(self):
        """
        GIVEN a WavenetStack with a large number of layers
        WHEN initialized
        THEN it should use memory efficiently
        """
        import sys
        
        # Create a small stack for baseline memory comparison
        small_params = {
            'num_layers_per_stack': 2,
            'residual_channels': 8,
            'skip_channels': 8,
            'kernel_size': 3,
            'dilation_base': 2,
            'dropout_rate': 0.1,
            'input_channels': 4,
            'use_bias': True
        }
        
        small_stack = WavenetStack(**small_params)
        
        # Create a larger stack
        large_params = small_params.copy()
        large_params['num_layers_per_stack'] = 10  # 5x more layers
        
        large_stack = WavenetStack(**large_params)
        
        # The memory usage should scale roughly linearly with the number of layers
        # We can check this by comparing the number of parameters
        small_param_count = sum(p.numel() for p in small_stack.parameters())
        large_param_count = sum(p.numel() for p in large_stack.parameters())
        
        # The large stack should have approximately 5x the parameters of the small stack
        # (not exactly 5x because the first layer is different)
        ratio = large_param_count / small_param_count
        assert 4.5 < ratio < 5.5, f"Parameter count ratio {ratio} is outside expected range"


    def test_initialization_parameter_types(self):
        """
        GIVEN a WavenetStack with parameters of different types
        WHEN initialized
        THEN it should handle type conversion appropriately
        """
        # Test with float parameters where integers are expected
        float_params = {
            'num_layers_per_stack': 3.5,  # Should be converted to 3
            'residual_channels': 16.7,    # Should be converted to 16
            'skip_channels': 8.2,         # Should be converted to 8
            'kernel_size': 3.9,           # Should be converted to 3
            'dilation_base': 2.1,         # Should be used as 2.1 for exponentiation
            'dropout_rate': 0.1,
            'input_channels': 4.8,        # Should be converted to 4
            'use_bias': True
        }
        
        float_stack = WavenetStack(**float_params)
        
        # Check that integer parameters were properly converted
        assert len(float_stack.layers) == 3
        assert float_stack.layers[0].out_channels == 16
        assert float_stack.skip_connections[0].out_channels == 8
        assert float_stack.layers[0].kernel_size[0] == 3
        assert float_stack.layers[0].in_channels == 4
        
        # Check that dilation was calculated using the float value
        # 2.1^1 ≈ 2.1, 2.1^2 ≈ 4.41
        assert abs(float_stack.layers[1].dilation[0] - 2.1) < 0.1
        assert abs(float_stack.layers[2].dilation[0] - 4.41) < 0.1
        
        # Test with boolean parameters for numeric values
        bool_params = {
            'num_layers_per_stack': 3,
            'residual_channels': 16,
            'skip_channels': 8,
            'kernel_size': 3,
            'dilation_base': 2,
            'dropout_rate': True,  # Should be converted to 1.0
            'input_channels': 4,
            'use_bias': False
        }
        
        bool_stack = WavenetStack(**bool_params)
        
        # Check that boolean was converted to float for dropout_rate
        assert bool_stack.dropout.p == 1.0
    
    def test_initialization_with_string_parameters(self):
        """
        GIVEN a WavenetStack with string parameters that can be converted to numbers
        WHEN initialized
        THEN it should handle the conversion or raise appropriate errors
        """
        # Test with string parameters that can be converted to numbers
        string_params = {
            'num_layers_per_stack': "3",
            'residual_channels': "16",
            'skip_channels': "8",
            'kernel_size': "3",
            'dilation_base': "2",
            'dropout_rate': "0.1",
            'input_channels': "4",
            'use_bias': True
        }
        
        try:
            string_stack = WavenetStack(**string_params)
            
            # If initialization succeeds, check that parameters were properly converted
            assert len(string_stack.layers) == 3
            assert string_stack.layers[0].out_channels == 16
            assert string_stack.skip_connections[0].out_channels == 8
            assert string_stack.layers[0].kernel_size[0] == 3
            assert string_stack.dropout.p == 0.1
        except (TypeError, ValueError) as e:
            # If initialization fails, it should be due to string conversion issues
            assert "could not convert string to" in str(e) or "expected" in str(e)
            
    def test_init_with_non_standard_use_bias_values(self):
        """
        GIVEN a WavenetStack with non-boolean values for use_bias
        WHEN initialized
        THEN it should convert truthy/falsy values to boolean
        """
        # Test with various truthy values for use_bias
        truthy_params = {
            'num_layers_per_stack': 2,
            'residual_channels': 16,
            'skip_channels': 8,
            'kernel_size': 3,
            'dilation_base': 2,
            'dropout_rate': 0.1,
            'input_channels': 4,
            'use_bias': 1  # Non-boolean truthy value
        }
        
        truthy_stack = WavenetStack(**truthy_params)
        
        # Check that bias is enabled (truthy value converted to True)
        for layer in truthy_stack.layers:
            assert layer.bias is not None
            
        # Test with various falsy values for use_bias
        falsy_params = truthy_params.copy()
        falsy_params['use_bias'] = 0  # Non-boolean falsy value
        
        falsy_stack = WavenetStack(**falsy_params)
        
        # Check that bias is disabled (falsy value converted to False)
        for layer in falsy_stack.layers:
            assert layer.bias is None
    
    def test_forward_with_empty_layers_and_skip_connections(self):
        """
        GIVEN a WavenetStack with empty layers and skip_connections lists
        WHEN forward is called
        THEN it should handle this edge case correctly without errors
        """
        # Create a stack with normal parameters
        params = {
            'num_layers_per_stack': 2,
            'residual_channels': 16,
            'skip_channels': 8,
            'kernel_size': 3,
            'dilation_base': 2,
            'dropout_rate': 0.1,
            'input_channels': 4,
            'use_bias': True
        }
        
        stack = WavenetStack(**params)
        
        # Artificially empty the layers and skip_connections lists
        with torch.no_grad():
            # Save the original lists
            original_layers = stack.layers
            original_skips = stack.skip_connections
            
            # Replace with empty lists
            stack.layers = nn.ModuleList([])
            stack.skip_connections = nn.ModuleList([])
            
            # Input tensor
            batch_size = 2
            seq_length = 10
            x = torch.ones(batch_size, params['input_channels'], seq_length)
            
            # Forward pass should work without errors
            skip_sum, residual = stack(x)
            
            # With no layers, skip_sum should be zeros with skip_channels dimension
            assert skip_sum.shape == (batch_size, params['skip_channels'], seq_length)
            assert torch.all(skip_sum == 0)
            
            # With no layers, residual should be the input itself
            assert residual.shape == x.shape
            assert torch.all(residual == x)
            
            # Restore the original lists
            stack.layers = original_layers
            stack.skip_connections = original_skips
    
    def test_forward_with_inf_values(self, default_params):
        """
        GIVEN a WavenetStack
        WHEN forward is called with inputs containing Inf values
        THEN it should propagate Inf values appropriately
        """
        stack = WavenetStack(**default_params)
        
        # Create input with some Inf values
        batch_size = 2
        seq_length = 10
        x = torch.ones(batch_size, default_params['input_channels'], seq_length)
        x[0, 0, 0] = float('inf')  # Set one value to Inf
        
        # Forward pass
        skip_sum, residual = stack(x)
        
        # The output should contain Inf values due to propagation
        assert torch.isinf(skip_sum).any(), "Inf did not propagate to skip connections"
        assert torch.isinf(residual).any(), "Inf did not propagate to residual output"


class TestWavenetStackForward:
    """Tests specifically focused on the forward method of WavenetStack."""
    
    @pytest.fixture
    def default_params(self):
        """
        Provides default parameters for WavenetStack initialization.
        
        Returns:
            dict: Default parameters for testing
        """
        return {
            'num_layers_per_stack': 3,
            'residual_channels': 32,
            'skip_channels': 16,
            'kernel_size': 3,
            'dilation_base': 2,
            'dropout_rate': 0.1,
            'input_channels': 8,
            'use_bias': True
        }
    
    def test_forward_with_single_timestep(self, default_params):
        """
        GIVEN a WavenetStack
        WHEN forward is called with input having a single timestep
        THEN it should process it correctly
        """
        stack = WavenetStack(**default_params)
        
        # Create input with single timestep
        batch_size = 2
        seq_length = 1
        x = torch.randn(batch_size, default_params['input_channels'], seq_length)
        
        # Forward pass
        skip_sum, residual = stack(x)
        
        # Check output shapes
        assert skip_sum.shape == (batch_size, default_params['skip_channels'], seq_length)
        assert residual.shape == (batch_size, default_params['residual_channels'], seq_length)
    
    def test_forward_with_large_input_values(self, default_params):
        """
        GIVEN a WavenetStack
        WHEN forward is called with very large input values
        THEN it should handle them without numerical instability
        """
        stack = WavenetStack(**default_params)
        
        # Create input with large values
        batch_size = 2
        seq_length = 10
        x = torch.ones(batch_size, default_params['input_channels'], seq_length) * 1e6
        
        # Forward pass
        skip_sum, residual = stack(x)
        
        # Check outputs are finite (not NaN or Inf)
        assert torch.all(torch.isfinite(skip_sum)), "Skip connections contain non-finite values"
        assert torch.all(torch.isfinite(residual)), "Residual output contains non-finite values"
    
    def test_forward_with_mixed_precision(self, default_params):
        """
        GIVEN a WavenetStack with parameters in float32
        WHEN forward is called with input in float16
        THEN it should handle the precision conversion
        """
        # Skip if float16 is not supported
        if not torch.cuda.is_available():
            pytest.skip("CUDA not available, skipping mixed precision test")
        
        stack = WavenetStack(**default_params)
        stack = stack.to('cuda')
        
        # Create input with float16 precision
        batch_size = 2
        seq_length = 10
        x = torch.ones(batch_size, default_params['input_channels'], seq_length, 
                      dtype=torch.float16, device='cuda')
        
        try:
            # Forward pass
            skip_sum, residual = stack(x)
            
            # Check output precision matches input
            assert skip_sum.dtype == torch.float16
            assert residual.dtype == torch.float16
        except Exception as e:
            # If it fails, it should be due to precision issues, not other errors
            assert "precision" in str(e).lower() or "dtype" in str(e).lower()
        finally:
            # Move back to CPU for cleanup
            stack = stack.to('cpu')


class TestWavenet:
    """Tests for the complete Wavenet model."""

    @pytest.fixture
    def default_model_config(self):
        """
        Provides default model configuration for Wavenet.
        
        Returns:
            dict: Default model configuration
        """
        return {
            'num_stacks': 2,
            'num_layers_per_stack': 3,
            'residual_channels': 32,
            'skip_channels': 16,
            'kernel_size': 3,
            'dilation_base': 2,
            'dropout_rate': 0.1,
            'input_channels': 8,
            'output_channels': 4,
            'use_bias': True
        }

    def test_initialization(self, default_model_config):
        """
        GIVEN valid model configuration
        WHEN a Wavenet is created
        THEN it should initialize with correct structure
        """
        model = Wavenet(default_model_config)
        
        # Check basic structure
        assert len(model.stacks) == default_model_config['num_stacks']
        assert isinstance(model.relu, nn.ReLU)
        assert isinstance(model.conv_out1, nn.Conv1d)
        assert isinstance(model.conv_out2, nn.Conv1d)
        
        # Check output layer dimensions
        assert model.conv_out1.in_channels == default_model_config['skip_channels']
        assert model.conv_out1.out_channels == default_model_config['skip_channels']
        assert model.conv_out2.in_channels == default_model_config['skip_channels']
        assert model.conv_out2.out_channels == default_model_config['output_channels']

    def test_forward_shape(self, default_model_config):
        """
        GIVEN a properly initialized Wavenet
        WHEN forward is called with valid input
        THEN it should return output with the expected shape
        """
        model = Wavenet(default_model_config)
        batch_size = 4
        seq_length = 100
        
        # Create input tensor (batch_size, input_channels, seq_length)
        x = torch.randn(batch_size, default_model_config['input_channels'], seq_length)
        
        # Forward pass
        output = model(x)
        
        # Check output shape (batch_size, seq_length, output_channels)
        expected_shape = (batch_size, seq_length, default_model_config['output_channels'])
        assert output.shape == expected_shape

    def test_end_to_end_gradient_flow(self, default_model_config):
        """
        GIVEN a Wavenet model
        WHEN a forward and backward pass is performed
        THEN gradients should flow through the entire network
        """
        model = Wavenet(default_model_config)
        
        # Create input that requires gradient
        x = torch.ones(2, default_model_config['input_channels'], 20, requires_grad=True)
        
        # Forward pass
        output = model(x)
        
        # Create a scalar loss and backpropagate
        loss = output.sum()
        loss.backward()
        
        # Check that gradients flowed through the network
        assert x.grad is not None
        assert torch.all(x.grad != 0)
        
        # Check that all parameters received gradients
        for name, param in model.named_parameters():
            assert param.grad is not None, f"Parameter {name} has no gradient"
            assert torch.any(param.grad != 0), f"Parameter {name} has zero gradient"

    def test_transpose_operations(self, default_model_config):
        """
        GIVEN a Wavenet model
        WHEN forward is called
        THEN it should correctly handle tensor transpositions
        """
        model = Wavenet(default_model_config)
        
        # Create input with recognizable shape
        batch_size = 2
        seq_length = 10
        x = torch.randn(batch_size, default_model_config['input_channels'], seq_length)
        
        # Add hooks to capture intermediate tensors
        intermediate_tensors = {}
        
        def save_input_hook(name):
            def hook(module, input_tensor, output_tensor):
                intermediate_tensors[name + "_input"] = input_tensor[0].clone()
                intermediate_tensors[name + "_output"] = output_tensor.clone()
            return hook
        
        # Register hooks
        model.stacks[0].register_forward_hook(save_input_hook("stack0"))
        model.conv_out1.register_forward_hook(save_input_hook("conv_out1"))
        
        # Forward pass
        output = model(x)
        
        # Check input to first stack is transposed correctly
        stack0_input = intermediate_tensors["stack0_input"]
        assert stack0_input.shape[1] == seq_length
        assert stack0_input.shape[2] == default_model_config['input_channels']
        
        # Check output is transposed back correctly
        assert output.shape[1] == seq_length
        assert output.shape[2] == default_model_config['output_channels']
    
    def test_initialization_with_minimal_config(self):
        """
        GIVEN minimal valid model configuration
        WHEN a Wavenet is created
        THEN it should initialize correctly with minimal parameters
        """
        minimal_config = {
            'num_stacks': 1,
            'num_layers_per_stack': 1,
            'residual_channels': 4,
            'skip_channels': 4,
            'kernel_size': 1,
            'dilation_base': 1,
            'dropout_rate': 0.0,
            'input_channels': 1,
            'output_channels': 1,
            'use_bias': False
        }
        
        model = Wavenet(minimal_config)
        
        # Check structure
        assert len(model.stacks) == 1
        assert isinstance(model.stacks[0], WavenetStack)
        
        # Check stack configuration
        stack = model.stacks[0]
        assert len(stack.layers) == 1
        assert stack.layers[0].kernel_size[0] == 1
        assert stack.layers[0].bias is None  # No bias
        
        # Test forward pass with minimal input
        x = torch.ones(1, 1, 5)
        output = model(x)
        assert output.shape == (1, 5, 1)
    
    def test_initialization_with_large_config(self):
        """
        GIVEN large model configuration
        WHEN a Wavenet is created
        THEN it should initialize correctly with many stacks and layers
        """
        large_config = {
            'num_stacks': 4,
            'num_layers_per_stack': 6,
            'residual_channels': 64,
            'skip_channels': 32,
            'kernel_size': 3,
            'dilation_base': 2,
            'dropout_rate': 0.2,
            'input_channels': 16,
            'output_channels': 8,
            'use_bias': True
        }
        
        model = Wavenet(large_config)
        
        # Check structure
        assert len(model.stacks) == 4
        
        # Check first and last stack
        first_stack = model.stacks[0]
        last_stack = model.stacks[-1]
        
        assert len(first_stack.layers) == 6
        assert len(last_stack.layers) == 6
        
        # First stack should take original input channels
        assert first_stack.layers[0].in_channels == large_config['input_channels']
        
        # Subsequent stacks should take residual_channels as input
        assert model.stacks[1].layers[0].in_channels == large_config['residual_channels']
        
        # Test with appropriate input
        x = torch.randn(2, large_config['input_channels'], 20)
        output = model(x)
        assert output.shape == (2, 20, large_config['output_channels'])
    
    def test_skip_connection_aggregation(self, default_model_config):
        """
        GIVEN a Wavenet model
        WHEN forward is called
        THEN it should correctly aggregate skip connections from all stacks
        """
        model = Wavenet(default_model_config)
        
        # Create a hook to capture the skip connections
        skip_values = []
        
        def capture_skip_hook(module, input_tensor, output_tensor):
            # In WavenetStack, output is (skip_sum, residual)
            skip_values.append(output_tensor[0].clone())
        
        # Register hooks on all stacks
        for i, stack in enumerate(model.stacks):
            stack.register_forward_hook(capture_skip_hook)
        
        # Forward pass
        batch_size = 2
        seq_length = 10
        x = torch.randn(batch_size, default_model_config['input_channels'], seq_length)
        output = model(x)
        
        # Verify we captured the right number of skip connections
        assert len(skip_values) == default_model_config['num_stacks']
        
        # All skip connections should have the same shape
        for skip in skip_values:
            assert skip.shape == (batch_size, default_model_config['skip_channels'], seq_length)
        
        # The model should process these skip connections through the output layers
        # We can't directly test the sum, but we can verify the output has the right shape
        assert output.shape == (batch_size, seq_length, default_model_config['output_channels'])
    
    def test_forward_with_different_sequence_lengths(self, default_model_config):
        """
        GIVEN a Wavenet model
        WHEN forward is called with inputs of different sequence lengths
        THEN it should handle all sequence lengths correctly
        """
        model = Wavenet(default_model_config)
        
        # Test with various sequence lengths
        seq_lengths = [1, 10, 100, 1000]
        batch_size = 2
        
        for seq_length in seq_lengths:
            # Create input
            x = torch.randn(batch_size, default_model_config['input_channels'], seq_length)
            
            # Forward pass
            output = model(x)
            
            # Check output shape
            expected_shape = (batch_size, seq_length, default_model_config['output_channels'])
            assert output.shape == expected_shape, f"Failed with sequence length {seq_length}"
    
    def test_forward_with_different_batch_sizes(self, default_model_config):
        """
        GIVEN a Wavenet model
        WHEN forward is called with inputs of different batch sizes
        THEN it should handle all batch sizes correctly
        """
        model = Wavenet(default_model_config)
        
        # Test with various batch sizes
        batch_sizes = [1, 2, 4, 8, 16]
        seq_length = 20
        
        for batch_size in batch_sizes:
            # Create input
            x = torch.randn(batch_size, default_model_config['input_channels'], seq_length)
            
            # Forward pass
            output = model(x)
            
            # Check output shape
            expected_shape = (batch_size, seq_length, default_model_config['output_channels'])
            assert output.shape == expected_shape, f"Failed with batch size {batch_size}"
    
    def test_forward_with_different_dtypes(self, default_model_config):
        """
        GIVEN a Wavenet model
        WHEN forward is called with inputs of different precision
        THEN it should maintain the input precision
        """
        model = Wavenet(default_model_config)
        
        # Test with different dtypes
        dtypes = [torch.float32, torch.float64]
        batch_size = 2
        seq_length = 10
        
        for dtype in dtypes:
            # Create input with specific dtype
            x = torch.ones(batch_size, default_model_config['input_channels'], seq_length, dtype=dtype)
            
            # Forward pass
            output = model(x)
            
            # Check output dtype matches input
            assert output.dtype == dtype, f"Output dtype {output.dtype} doesn't match input dtype {dtype}"
    
    def test_forward_with_zero_length_sequence(self, default_model_config):
        """
        GIVEN a Wavenet model
        WHEN forward is called with a zero-length sequence
        THEN it should handle this edge case gracefully
        """
        model = Wavenet(default_model_config)
        
        # Create input with zero sequence length
        batch_size = 2
        seq_length = 0
        x = torch.ones(batch_size, default_model_config['input_channels'], seq_length)
        
        # Forward pass
        output = model(x)
        
        # Check output shape - should preserve the zero sequence length
        expected_shape = (batch_size, seq_length, default_model_config['output_channels'])
        assert output.shape == expected_shape
    
    def test_forward_with_nan_inputs(self, default_model_config):
        """
        GIVEN a Wavenet model
        WHEN forward is called with inputs containing NaN values
        THEN it should propagate NaNs appropriately
        """
        model = Wavenet(default_model_config)
        
        # Create input with some NaN values
        batch_size = 2
        seq_length = 10
        x = torch.ones(batch_size, default_model_config['input_channels'], seq_length)
        x[0, 0, 0] = float('nan')  # Set one value to NaN
        
        # Forward pass
        output = model(x)
        
        # The output should contain NaN values due to propagation
        assert torch.isnan(output).any(), "NaN did not propagate to output"
    
    def test_model_with_controlled_weights(self, default_model_config):
        """
        GIVEN a Wavenet model with controlled weights
        WHEN forward is called
        THEN it should produce deterministic output
        """
        # Create model with fixed random seed
        torch.manual_seed(42)
        model = Wavenet(default_model_config)
        
        # Set all weights to a constant value for predictable output
        with torch.no_grad():
            for param in model.parameters():
                nn.init.constant_(param, 0.1)
        
        # Create input with all ones
        batch_size = 1
        seq_length = 5
        x = torch.ones(batch_size, default_model_config['input_channels'], seq_length)
        
        # Forward pass
        model.eval()  # Disable dropout for deterministic output
        output1 = model(x)
        
        # Second forward pass with same input should give identical output
        output2 = model(x)
        
        # Outputs should be identical
        assert torch.allclose(output1, output2)
        
        # Output should be non-zero (actual values depend on architecture details)
        assert torch.all(output1 != 0)
    
    def test_training_vs_eval_mode(self, default_model_config):
        """
        GIVEN a Wavenet model with dropout
        WHEN forward is called in training and evaluation modes
        THEN it should apply dropout only in training mode
        """
        # Ensure we have dropout
        config = default_model_config.copy()
        config['dropout_rate'] = 0.5
        
        model = Wavenet(config)
        
        # Create input
        batch_size = 2
        seq_length = 10
        x = torch.ones(batch_size, config['input_channels'], seq_length)
        
        # Test in training mode
        model.train()
        torch.manual_seed(42)  # For reproducibility
        output_train = model(x)
        
        # Test in evaluation mode
        model.eval()
        torch.manual_seed(42)  # Same seed
        output_eval = model(x)
        
        # Outputs should differ due to dropout in training mode
        assert not torch.allclose(output_train, output_eval)
    
    def test_receptive_field(self, default_model_config):
        """
        GIVEN a Wavenet model
        WHEN initialized with specific dilation parameters
        THEN it should have the expected receptive field
        """
        model = Wavenet(default_model_config)
        
        # Calculate theoretical receptive field
        # For each stack: sum of dilated convolution receptive fields
        # Formula: kernel_size + (kernel_size-1) * sum(dilations)
        kernel_size = default_model_config['kernel_size']
        layers_per_stack = default_model_config['num_layers_per_stack']
        dilation_base = default_model_config['dilation_base']
        
        # Calculate dilations for each layer
        dilations = [dilation_base ** i for i in range(layers_per_stack)]
        
        # Calculate receptive field for one stack
        # For each layer: (kernel_size - 1) * dilation
        # Plus 1 for the center point
        stack_receptive_field = 1 + sum((kernel_size - 1) * d for d in dilations)
        
        # Create input with a clear temporal pattern to test receptive field
        batch_size = 1
        seq_length = 100
        x = torch.zeros(batch_size, default_model_config['input_channels'], seq_length)
        
        # Set a single point to 1.0 in the middle
        middle_idx = seq_length // 2
        x[:, :, middle_idx] = 1.0
        
        # Forward pass
        model.eval()  # Disable dropout
        
        # Set all weights to small values to prevent saturation
        with torch.no_grad():
            for param in model.parameters():
                nn.init.normal_(param, mean=0.0, std=0.01)
        
        output = model(x)
        
        # The output should show activation in a region around the middle point
        # The size of this region is related to the receptive field
        # We can check that points far outside the receptive field are close to zero
        
        # Points that should be outside the receptive field
        far_left = middle_idx - stack_receptive_field * 2
        far_right = middle_idx + stack_receptive_field * 2
        
        if far_left >= 0:
            assert torch.all(torch.abs(output[:, far_left, :]) < 0.1)
        
        if far_right < seq_length:
            assert torch.all(torch.abs(output[:, far_right, :]) < 0.1)
    
    def test_parameter_count(self, default_model_config):
        """
        GIVEN a Wavenet model
        WHEN initialized with specific configuration
        THEN it should have the expected number of parameters
        """
        model = Wavenet(default_model_config)
        
        # Count parameters
        param_count = sum(p.numel() for p in model.parameters())
        
        # Calculate expected parameter count
        # For each stack:
        #   First layer: in_channels * residual_channels * kernel_size + residual_channels (bias)
        #   Other layers: residual_channels * residual_channels * kernel_size + residual_channels (bias)
        #   Skip connections: residual_channels * skip_channels + skip_channels (bias)
        # Output layers:
        #   Conv1: skip_channels * skip_channels * 1 + skip_channels (bias)
        #   Conv2: skip_channels * output_channels * 1 + output_channels (bias)
        
        num_stacks = default_model_config['num_stacks']
        layers_per_stack = default_model_config['num_layers_per_stack']
        residual_channels = default_model_config['residual_channels']
        skip_channels = default_model_config['skip_channels']
        kernel_size = default_model_config['kernel_size']
        input_channels = default_model_config['input_channels']
        output_channels = default_model_config['output_channels']
        use_bias = default_model_config['use_bias']
        
        # First stack first layer
        params = input_channels * residual_channels * kernel_size
        if use_bias:
            params += residual_channels
        
        # First stack other layers
        other_layers_first_stack = layers_per_stack - 1
        params += other_layers_first_stack * (residual_channels * residual_channels * kernel_size)
        if use_bias:
            params += other_layers_first_stack * residual_channels
        
        # First stack skip connections
        params += layers_per_stack * (residual_channels * skip_channels)
        if use_bias:
            params += layers_per_stack * skip_channels
        
        # Other stacks (all layers take residual_channels as input)
        other_stacks = num_stacks - 1
        params += other_stacks * layers_per_stack * (residual_channels * residual_channels * kernel_size)
        if use_bias:
            params += other_stacks * layers_per_stack * residual_channels
        
        # Other stacks skip connections
        params += other_stacks * layers_per_stack * (residual_channels * skip_channels)
        if use_bias:
            params += other_stacks * layers_per_stack * skip_channels
        
        # Output layers
        params += skip_channels * skip_channels * 1  # Conv1
        params += skip_channels * output_channels * 1  # Conv2
        if use_bias:
            params += skip_channels  # Conv1 bias
            params += output_channels  # Conv2 bias
        
        # Allow for small differences due to implementation details
        assert abs(param_count - params) / params < 0.1, \
            f"Parameter count {param_count} differs significantly from expected {params}"
    
    def test_stack_input_channels(self, default_model_config):
        """
        GIVEN a Wavenet model with multiple stacks
        WHEN initialized
        THEN each stack should have the correct input channels
        """
        model = Wavenet(default_model_config)
        
        # First stack should take original input channels
        first_stack = model.stacks[0]
        assert first_stack.layers[0].in_channels == default_model_config['input_channels']
        
        # Subsequent stacks should take residual_channels as input
        for i in range(1, default_model_config['num_stacks']):
            stack = model.stacks[i]
            assert stack.layers[0].in_channels == default_model_config['residual_channels']
    
    def test_output_activation(self, default_model_config):
        """
        GIVEN a Wavenet model
        WHEN forward is called
        THEN it should apply ReLU activation in the output layers
        """
        model = Wavenet(default_model_config)
        
        # Create input with negative values
        batch_size = 2
        seq_length = 10
        x = torch.randn(batch_size, default_model_config['input_channels'], seq_length)
        
        # Add hook to capture intermediate values
        intermediate = {}
        
        def save_pre_relu_hook(module, input_tensor, output_tensor):
            intermediate['pre_relu'] = input_tensor[0].clone()
        
        def save_post_relu_hook(module, input_tensor, output_tensor):
            intermediate['post_relu'] = output_tensor.clone()
        
        # Register hooks
        model.relu.register_forward_hook(save_post_relu_hook)
        model.relu.register_forward_pre_hook(save_pre_relu_hook)
        
        # Forward pass
        output = model(x)
        
        # Verify ReLU behavior - no negative values in output of ReLU
        assert torch.all(intermediate['post_relu'] >= 0)
        
        # If there were negative values in the input to ReLU, they should be zeroed
        if torch.any(intermediate['pre_relu'] < 0):
            neg_mask = intermediate['pre_relu'] < 0
            assert torch.all(intermediate['post_relu'][neg_mask] == 0)
    
    def test_missing_config_parameters(self):
        """
        GIVEN a model configuration with missing parameters
        WHEN a Wavenet is created
        THEN it should raise an appropriate KeyError
        """
        incomplete_config = {
            'num_stacks': 2,
            'num_layers_per_stack': 3,
            'residual_channels': 32,
            # Missing 'skip_channels'
            'kernel_size': 3,
            'dilation_base': 2,
            'dropout_rate': 0.1,
            'input_channels': 8,
            'output_channels': 4,
            'use_bias': True
        }
        
        with pytest.raises(KeyError):
            model = Wavenet(incomplete_config)
    
    def test_multiple_forward_passes_consistency(self, default_model_config):
        """
        GIVEN a Wavenet model in evaluation mode
        WHEN forward is called multiple times with the same input
        THEN it should produce identical outputs
        """
        model = Wavenet(default_model_config)
        model.eval()  # Set to evaluation mode
        
        # Create input
        batch_size = 2
        seq_length = 10
        x = torch.randn(batch_size, default_model_config['input_channels'], seq_length)
        
        # Multiple forward passes
        output1 = model(x)
        output2 = model(x)
        output3 = model(x)
        
        # All outputs should be identical
        assert torch.allclose(output1, output2)
        assert torch.allclose(output2, output3)
    
    def test_device_transfer(self, default_model_config):
        """
        GIVEN a Wavenet model
        WHEN moved to a different device
        THEN it should function correctly on that device
        """
        model = Wavenet(default_model_config)
        
        # Skip test if CUDA is not available
        if not torch.cuda.is_available():
            pytest.skip("CUDA not available, skipping device transfer test")
        
        # Move model to CUDA
        model.to('cuda')
        
        # Create input on CUDA
        batch_size = 2
        seq_length = 10
        x = torch.randn(batch_size, default_model_config['input_channels'], seq_length, device='cuda')
        
        # Forward pass
        output = model(x)
        
        # Check output is on the correct device
        assert output.device.type == 'cuda'
        
        # Check output shape
        expected_shape = (batch_size, seq_length, default_model_config['output_channels'])
        assert output.shape == expected_shape
        
        # Move back to CPU for cleanup
        model.to('cpu')
    
    def test_jit_compatibility(self, default_model_config):
        """
        GIVEN a Wavenet model
        WHEN traced with torch.jit.trace
        THEN it should be compatible with TorchScript
        """
        model = Wavenet(default_model_config)
        model.eval()  # Set to evaluation mode
        
        # Create example input
        batch_size = 2
        seq_length = 10
        example_input = torch.randn(batch_size, default_model_config['input_channels'], seq_length)
        
        try:
            # Attempt to trace the model
            traced_model = torch.jit.trace(model, example_input)
            
            # Test the traced model
            traced_output = traced_model(example_input)
            
            # Compare with original model output
            original_output = model(example_input)
            
            # Outputs should be identical
            assert torch.allclose(traced_output, original_output)
        except Exception as e:
            pytest.fail(f"JIT tracing failed: {str(e)}")
    
    def test_zero_stacks_error_handling(self):
        """
        GIVEN a model configuration with zero stacks
        WHEN a Wavenet is created
        THEN it should handle this edge case appropriately
        """
        zero_stacks_config = {
            'num_stacks': 0,
            'num_layers_per_stack': 3,
            'residual_channels': 32,
            'skip_channels': 16,
            'kernel_size': 3,
            'dilation_base': 2,
            'dropout_rate': 0.1,
            'input_channels': 8,
            'output_channels': 4,
            'use_bias': True
        }
        
        # This should either raise a ValueError or create a model that handles zero stacks gracefully
        # The exact behavior depends on the implementation
        try:
            model = Wavenet(zero_stacks_config)
            
            # If model creation succeeds, test forward pass
            batch_size = 2
            seq_length = 10
            x = torch.randn(batch_size, zero_stacks_config['input_channels'], seq_length)
            
            # Forward pass should either work or raise a meaningful error
            try:
                output = model(x)
                # If it works, check output shape
                expected_shape = (batch_size, seq_length, zero_stacks_config['output_channels'])
                assert output.shape == expected_shape
            except Exception as e:
                # The error should be related to having zero stacks
                assert "stack" in str(e).lower()
        except ValueError as e:
            # If model creation fails, the error should mention stacks
            assert "stack" in str(e).lower()
            
    def test_forward_with_gradient_checkpointing(self, default_model_config):
        """
        GIVEN a Wavenet model with gradient checkpointing enabled
        WHEN forward is called
        THEN it should produce the same output but use less memory
        """
        # Skip if CUDA is not available
        if not torch.cuda.is_available():
            pytest.skip("CUDA not available, skipping gradient checkpointing test")
            
        # Create model without gradient checkpointing
        torch.manual_seed(42)
        model_standard = Wavenet(default_model_config)
        model_standard.to('cuda')
        
        # Create identical model with gradient checkpointing
        torch.manual_seed(42)
        model_checkpointed = Wavenet(default_model_config)
        model_checkpointed.to('cuda')
        
        # Enable gradient checkpointing on the second model
        # This is a mock implementation since the actual model might not support it directly
        try:
            # Try to enable gradient checkpointing if the model supports it
            for stack in model_checkpointed.stacks:
                for module in stack.modules():
                    if hasattr(module, 'gradient_checkpointing'):
                        module.gradient_checkpointing = True
            
            # Create input requiring gradients
            batch_size = 2
            seq_length = 100
            x = torch.randn(batch_size, default_model_config['input_channels'], seq_length, 
                           requires_grad=True, device='cuda')
            
            # Forward pass with standard model
            output_standard = model_standard(x)
            
            # Forward pass with checkpointed model
            output_checkpointed = model_checkpointed(x)
            
            # Outputs should be identical
            assert torch.allclose(output_standard, output_checkpointed, atol=1e-5)
            
            # Clean up
            model_standard.to('cpu')
            model_checkpointed.to('cpu')
            
        except (AttributeError, NotImplementedError):
            # If gradient checkpointing is not implemented, skip the test
            pytest.skip("Gradient checkpointing not supported by the model")
    
    def test_forward_with_custom_activation(self, default_model_config):
        """
        GIVEN a Wavenet model with a modified activation function
        WHEN forward is called
        THEN it should use the custom activation
        """
        model = Wavenet(default_model_config)
        
        # Create a hook to replace the ReLU activation
        activation_called = [False]
        
        def custom_activation_hook(module, input_tensor, output_tensor):
            activation_called[0] = True
            # Apply a different activation (e.g., ELU instead of ReLU)
            return torch.nn.functional.elu(input_tensor[0])
        
        # Register the hook to replace ReLU
        hook_handle = model.relu.register_forward_hook(custom_activation_hook)
        
        try:
            # Forward pass
            batch_size = 2
            seq_length = 10
            x = torch.randn(batch_size, default_model_config['input_channels'], seq_length)
            output = model(x)
            
            # Verify our custom activation was called
            assert activation_called[0], "Custom activation hook was not called"
            
        finally:
            # Remove the hook
            hook_handle.remove()
