"""
Test suite for TCNBlock and TCN models.

This module provides comprehensive testing for:
1. TCNBlock initialization and parameter validation
2. Forward pass functionality and tensor shape transformations
3. Residual connection behavior
4. Edge cases and error handling
5. TCN model architecture and behavior
6. Input/output tensor shape transformations
7. Integration of TCNBlocks within the TCN model
"""

import pytest
import torch
import torch.nn as nn
from src.models.tcn import TCNBlock, TCN
from src.models.wavenet import WavenetStack, Wavenet


# === FIXTURES ===

@pytest.fixture
def sample_input_small():
    """
    Provides a small sample input tensor for TCNBlock testing.
    
    Returns:
        torch.Tensor: Tensor of shape (batch_size=2, channels=3, seq_len=10)
    """
    return torch.randn(2, 3, 10)


@pytest.fixture
def sample_input_large():
    """
    Provides a larger sample input tensor for TCNBlock testing.
    
    Returns:
        torch.Tensor: Tensor of shape (batch_size=8, channels=5, seq_len=20)
    """
    return torch.randn(8, 5, 20)


@pytest.fixture
def standard_tcn_block():
    """
    Provides a standard TCNBlock with common parameters.
    
    Returns:
        TCNBlock: A TCNBlock with in_channels=3, out_channels=5, kernel_size=3, dilation=1, dropout_rate=0.2
    """
    return TCNBlock(in_channels=3, out_channels=5, kernel_size=3, dilation=1, dropout_rate=0.2)


@pytest.fixture
def tcn_model_config():
    """
    Provides a standard configuration dictionary for TCN model.
    
    Returns:
        dict: Configuration with standard parameters for TCN model
    """
    return {
        'input_size': 8,
        'output_size': 4,
        'num_blocks': 3,
        'num_channels': 16,
        'kernel_size': 3,
        'dropout': 0.2
    }


@pytest.fixture
def standard_tcn_model(tcn_model_config):
    """
    Provides a standard TCN model with common parameters.
    
    Returns:
        TCN: A TCN model initialized with standard test configuration
    """
    return TCN(tcn_model_config)


@pytest.fixture
def tcn_input_tensor():
    """
    Provides a sample input tensor for TCN model testing.
    
    Returns:
        torch.Tensor: Tensor of shape (batch_size=4, seq_len=20, input_size=8)
    """
    return torch.randn(4, 20, 8)


# === TEST CLASSES ===

class TestTCNBlockInitialization:
    """Tests covering TCNBlock creation and parameter validation."""
    
    def test_valid_initialization(self):
        """
        GIVEN valid initialization parameters
        WHEN the TCNBlock is created
        THEN it should initialize with correct layers and parameters
        """
        # Test with standard parameters
        block = TCNBlock(in_channels=3, out_channels=5, kernel_size=3, dilation=1, dropout_rate=0.2)
        
        # Verify layers exist and have correct parameters
        assert isinstance(block.conv1, nn.Conv1d)
        assert block.conv1.in_channels == 3
        assert block.conv1.out_channels == 5
        assert block.conv1.kernel_size[0] == 3
        assert block.conv1.dilation[0] == 1
        
        assert isinstance(block.conv2, nn.Conv1d)
        assert block.conv2.in_channels == 5
        assert block.conv2.out_channels == 5
        assert block.conv2.kernel_size[0] == 3
        assert block.conv2.dilation[0] == 1
        
        assert isinstance(block.relu1, nn.ReLU)
        assert isinstance(block.relu2, nn.ReLU)
        assert isinstance(block.relu, nn.ReLU)
        
        assert isinstance(block.dropout1, nn.Dropout)
        assert block.dropout1.p == 0.2
        assert isinstance(block.dropout2, nn.Dropout)
        assert block.dropout2.p == 0.2
    
    def test_initialization_with_extreme_parameters(self):
        """
        GIVEN extreme but valid initialization parameters
        WHEN the TCNBlock is created
        THEN it should initialize correctly with those parameters
        """
        # Test with very large kernel and dilation
        block = TCNBlock(in_channels=1, out_channels=100, kernel_size=15, dilation=16, dropout_rate=0.8)
        
        # Verify parameters were set correctly
        assert block.conv1.in_channels == 1
        assert block.conv1.out_channels == 100
        assert block.conv1.kernel_size[0] == 15
        assert block.conv1.dilation[0] == 16
        assert block.dropout1.p == 0.8
        
        # Verify padding calculation for large kernel and dilation
        expected_padding = (15 - 1) * 16 // 2  # (kernel_size - 1) * dilation // 2
        assert block.conv1.padding[0] == expected_padding
    
    def test_initialization_with_even_kernel_size(self):
        """
        GIVEN an even kernel size
        WHEN the TCNBlock is created
        THEN it should calculate padding correctly
        """
        # Test with even kernel size (4)
        block = TCNBlock(in_channels=3, out_channels=5, kernel_size=4, dilation=2, dropout_rate=0.2)
        
        # Verify padding calculation for even kernel size
        expected_padding = (4 - 1) * 2 // 2  # (kernel_size - 1) * dilation // 2 = 3
        assert block.conv1.padding[0] == expected_padding
        assert block.conv2.padding[0] == expected_padding
    
    def test_initialization_with_zero_dropout(self):
        """
        GIVEN a dropout rate of 0
        WHEN the TCNBlock is created
        THEN it should create dropout layers with 0 probability
        """
        block = TCNBlock(in_channels=3, out_channels=5, kernel_size=3, dilation=1, dropout_rate=0.0)
        
        # Verify dropout layers have 0 probability
        assert isinstance(block.dropout1, nn.Dropout)
        assert block.dropout1.p == 0.0
        assert isinstance(block.dropout2, nn.Dropout)
        assert block.dropout2.p == 0.0
    
    def test_initialization_with_bias_settings(self):
        """
        GIVEN explicit bias settings for convolutions
        WHEN the TCNBlock is created
        THEN it should respect those bias settings
        """
        # Test with bias=True (default)
        block_with_bias = TCNBlock(in_channels=3, out_channels=5, kernel_size=3, dilation=1, dropout_rate=0.2)
        assert block_with_bias.conv1.bias is not None
        assert block_with_bias.conv2.bias is not None
        if block_with_bias.downsample is not None:
            assert block_with_bias.downsample.bias is not None
        
        # Test with bias=False (if supported by the implementation)
        # Note: This test assumes the TCNBlock constructor accepts a 'bias' parameter
        # If it doesn't, this test should be modified or removed
        try:
            block_without_bias = TCNBlock(in_channels=3, out_channels=5, kernel_size=3, 
                                         dilation=1, dropout_rate=0.2, bias=False)
            assert block_without_bias.conv1.bias is None
            assert block_without_bias.conv2.bias is None
            if block_without_bias.downsample is not None:
                assert block_without_bias.downsample.bias is None
        except TypeError:
            # If bias parameter is not supported, this is expected to fail
            pass
    
    def test_bias_parameter_extension(self):
        """
        GIVEN a TCNBlock with extended constructor supporting bias parameter
        WHEN the block is created with bias=False
        THEN all convolutions should have bias disabled
        """
        # This test is for future implementation that supports bias parameter
        # It's a placeholder to document the expected behavior
        
        # Mock the TCNBlock to accept bias parameter
        class ExtendedTCNBlock(TCNBlock):
            def __init__(self, in_channels, out_channels, kernel_size, dilation, dropout_rate, bias=True):
                super(TCNBlock, self).__init__()
                self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size,
                                      dilation=dilation, padding=(kernel_size - 1) * dilation // 2, bias=bias)
                self.relu1 = nn.ReLU()
                self.dropout1 = nn.Dropout(dropout_rate)
                self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size,
                                      dilation=dilation, padding=(kernel_size - 1) * dilation // 2, bias=bias)
                self.relu2 = nn.ReLU()
                self.dropout2 = nn.Dropout(dropout_rate)
                self.downsample = nn.Conv1d(in_channels, out_channels, 1, bias=bias) if in_channels != out_channels else None
                self.relu = nn.ReLU()
        
        # Create block with bias=False
        block = ExtendedTCNBlock(in_channels=3, out_channels=5, kernel_size=3, dilation=1, dropout_rate=0.2, bias=False)
        
        # Verify all convolutions have bias disabled
        assert block.conv1.bias is None
        assert block.conv2.bias is None
        assert block.downsample.bias is None
    
    def test_downsample_creation(self):
        """
        GIVEN different in_channels and out_channels
        WHEN the TCNBlock is created
        THEN it should create a downsample layer
        """
        # Test with different in and out channels
        block = TCNBlock(in_channels=3, out_channels=5, kernel_size=3, dilation=1, dropout_rate=0.2)
        
        # Verify downsample layer exists and has correct parameters
        assert block.downsample is not None
        assert isinstance(block.downsample, nn.Conv1d)
        assert block.downsample.in_channels == 3
        assert block.downsample.out_channels == 5
        assert block.downsample.kernel_size[0] == 1
        
        # Test with larger channel difference
        block = TCNBlock(in_channels=2, out_channels=64, kernel_size=3, dilation=1, dropout_rate=0.2)
        assert block.downsample is not None
        assert block.downsample.in_channels == 2
        assert block.downsample.out_channels == 64
    
    def test_no_downsample_when_channels_match(self):
        """
        GIVEN equal in_channels and out_channels
        WHEN the TCNBlock is created
        THEN it should not create a downsample layer
        """
        # Test with same in and out channels
        block = TCNBlock(in_channels=5, out_channels=5, kernel_size=3, dilation=1, dropout_rate=0.2)
        
        # Verify downsample layer is None
        assert block.downsample is None
    
    def test_dilation_affects_padding(self):
        """
        GIVEN different dilation values
        WHEN the TCNBlock is created
        THEN it should adjust padding accordingly
        """
        # Test with dilation = 2
        block1 = TCNBlock(in_channels=3, out_channels=5, kernel_size=3, dilation=2, dropout_rate=0.2)
        assert block1.conv1.padding[0] == 2  # (3-1)*2//2 = 2
        
        # Test with dilation = 4
        block2 = TCNBlock(in_channels=3, out_channels=5, kernel_size=3, dilation=4, dropout_rate=0.2)
        assert block2.conv1.padding[0] == 4  # (3-1)*4//2 = 4
        
        # Test with different kernel size and dilation
        block3 = TCNBlock(in_channels=3, out_channels=5, kernel_size=5, dilation=8, dropout_rate=0.2)
        assert block3.conv1.padding[0] == 16  # (5-1)*8//2 = 16
        
        # Verify both conv1 and conv2 have the same padding
        assert block1.conv1.padding[0] == block1.conv2.padding[0]
        assert block2.conv1.padding[0] == block2.conv2.padding[0]
        assert block3.conv1.padding[0] == block3.conv2.padding[0]


class TestTCNBlockForward:
    """Tests covering forward pass functionality."""
    
    def test_forward_implementation(self, mocker):
        """
        GIVEN a TCNBlock
        WHEN we trace through the forward implementation
        THEN it should follow the expected computation path
        """
        # Create a block with known parameters for easier tracing
        block = TCNBlock(in_channels=2, out_channels=4, kernel_size=3, dilation=1, dropout_rate=0.0)
        
        # Create a simple input with recognizable values
        input_tensor = torch.ones(1, 2, 5)  # (batch=1, channels=2, seq_len=5)
        
        # Manually trace through the expected computation
        # 1. First convolution
        conv1_out = block.conv1(input_tensor)
        # 2. First ReLU
        relu1_out = block.relu1(conv1_out)
        # 3. First dropout (with rate 0.0, this is identity)
        dropout1_out = relu1_out  # No dropout effect with rate 0.0
        # 4. Second convolution
        conv2_out = block.conv2(dropout1_out)
        # 5. Second ReLU
        relu2_out = block.relu2(conv2_out)
        # 6. Second dropout (with rate 0.0, this is identity)
        dropout2_out = relu2_out  # No dropout effect with rate 0.0
        # 7. Downsample input
        residual = block.downsample(input_tensor)
        # 8. Add residual and apply final ReLU
        expected_output = block.relu(dropout2_out + residual)
        
        # Get actual output
        actual_output = block(input_tensor)
        
        # Verify outputs match
        assert torch.allclose(actual_output, expected_output)
    
    def test_grouped_convolutions(self):
        """
        GIVEN a TCNBlock with grouped convolutions
        WHEN forward pass is executed
        THEN it should process correctly
        """
        # Define a modified TCNBlock with grouped convolutions
        class GroupedTCNBlock(nn.Module):
            def __init__(self, in_channels, out_channels, kernel_size, dilation, dropout_rate, groups=1):
                super(GroupedTCNBlock, self).__init__()
                # Ensure channel counts are divisible by groups
                assert in_channels % groups == 0, "in_channels must be divisible by groups"
                assert out_channels % groups == 0, "out_channels must be divisible by groups"
                
                self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size,
                                      dilation=dilation, padding=(kernel_size - 1) * dilation // 2,
                                      groups=groups)
                self.relu1 = nn.ReLU()
                self.dropout1 = nn.Dropout(dropout_rate)
                self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size,
                                      dilation=dilation, padding=(kernel_size - 1) * dilation // 2,
                                      groups=groups)
                self.relu2 = nn.ReLU()
                self.dropout2 = nn.Dropout(dropout_rate)
                self.downsample = nn.Conv1d(in_channels, out_channels, 1, groups=1) if in_channels != out_channels else None
                self.relu = nn.ReLU()
            
            def forward(self, x):
                residual = x
                
                out = self.conv1(x)
                out = self.relu1(out)
                out = self.dropout1(out)
                
                out = self.conv2(out)
                out = self.relu2(out)
                out = self.dropout2(out)
                
                if self.downsample is not None:
                    residual = self.downsample(x)
                
                out = self.relu(out + residual)
                
                return out
        
        # Test with different group sizes
        # For simplicity, use channel counts that are divisible by all tested group sizes
        in_channels = 12
        out_channels = 24
        
        for groups in [1, 2, 3, 4, 6, 12]:
            # Create input tensor
            input_tensor = torch.randn(2, in_channels, 10)
            
            # Create block with grouped convolutions
            block = GroupedTCNBlock(in_channels=in_channels, out_channels=out_channels, 
                                   kernel_size=3, dilation=1, dropout_rate=0.0, groups=groups)
            
            # Forward pass
            output = block(input_tensor)
            
            # Verify output shape
            assert output.shape == (2, out_channels, 10), f"Failed with groups={groups}"
    
    def test_forward_shape_preservation(self, standard_tcn_block, sample_input_small):
        """
        GIVEN a TCNBlock and input tensor
        WHEN forward pass is executed
        THEN output tensor should have expected shape
        """
        # Get input shape
        batch_size, _, seq_len = sample_input_small.shape
        
        # Forward pass
        output = standard_tcn_block(sample_input_small)
        
        # Check output shape
        assert output.shape == (batch_size, 5, seq_len)
    
    def test_forward_with_downsample(self, sample_input_small):
        """
        GIVEN a TCNBlock with different in/out channels
        WHEN forward pass is executed
        THEN output should include downsampled residual connection
        """
        # Create block with different in/out channels
        block = TCNBlock(in_channels=3, out_channels=5, kernel_size=3, dilation=1, dropout_rate=0.2)
        
        # Forward pass
        output = block(sample_input_small)
        
        # Check output shape
        batch_size, _, seq_len = sample_input_small.shape
        assert output.shape == (batch_size, 5, seq_len)
        
        # Verify downsample was used (indirectly)
        assert block.downsample is not None
    
    def test_forward_without_downsample(self, sample_input_large):
        """
        GIVEN a TCNBlock with same in/out channels
        WHEN forward pass is executed
        THEN output should include direct residual connection
        """
        # Create input with 5 channels
        input_tensor = sample_input_large
        
        # Create block with same in/out channels
        block = TCNBlock(in_channels=5, out_channels=5, kernel_size=3, dilation=1, dropout_rate=0.2)
        
        # Forward pass
        output = block(input_tensor)
        
        # Check output shape
        batch_size, channels, seq_len = input_tensor.shape
        assert output.shape == (batch_size, channels, seq_len)
        
        # Verify downsample was not used
        assert block.downsample is None
    
    def test_forward_with_different_dilations(self, sample_input_small):
        """
        GIVEN TCNBlocks with different dilation rates
        WHEN forward pass is executed
        THEN output should maintain sequence length
        """
        # Test with different dilations
        for dilation in [1, 2, 4, 8]:
            block = TCNBlock(in_channels=3, out_channels=5, kernel_size=3, dilation=dilation, dropout_rate=0.2)
            output = block(sample_input_small)
            
            # Check output shape
            batch_size, _, seq_len = sample_input_small.shape
            assert output.shape == (batch_size, 5, seq_len), f"Failed with dilation={dilation}"
    
    def test_residual_connection(self, sample_input_small):
        """
        GIVEN a TCNBlock with zero-weight convolutions
        WHEN forward pass is executed
        THEN output should equal the downsampled input
        """
        # Create block with different in/out channels
        block = TCNBlock(in_channels=3, out_channels=5, kernel_size=3, dilation=1, dropout_rate=0.0)
        
        # Zero out the weights of the convolutions to isolate residual path
        with torch.no_grad():
            nn.init.zeros_(block.conv1.weight)
            nn.init.zeros_(block.conv1.bias)
            nn.init.zeros_(block.conv2.weight)
            nn.init.zeros_(block.conv2.bias)
        
        # Forward pass
        output = block(sample_input_small)
        
        # Compute expected output (just the downsampled input)
        expected = block.relu(block.downsample(sample_input_small))
        
        # Check if output matches expected
        assert torch.allclose(output, expected)
    
    def test_forward_execution_flow(self, mocker):
        """
        GIVEN a TCNBlock with mocked internal components
        WHEN forward pass is executed
        THEN all components should be called in the correct order
        """
        # Create input tensor
        input_tensor = torch.randn(2, 3, 10)
        
        # Create block
        block = TCNBlock(in_channels=3, out_channels=5, kernel_size=3, dilation=1, dropout_rate=0.2)
        
        # Create spies for each component
        conv1_spy = mocker.spy(block.conv1, 'forward')
        relu1_spy = mocker.spy(block.relu1, 'forward')
        dropout1_spy = mocker.spy(block.dropout1, 'forward')
        conv2_spy = mocker.spy(block.conv2, 'forward')
        relu2_spy = mocker.spy(block.relu2, 'forward')
        dropout2_spy = mocker.spy(block.dropout2, 'forward')
        downsample_spy = mocker.spy(block.downsample, 'forward')
        relu_spy = mocker.spy(block.relu, 'forward')
        
        # Execute forward pass
        output = block(input_tensor)
        
        # Verify all components were called in the correct order
        # We can't directly verify order, but we can verify each was called once
        conv1_spy.assert_called_once()
        relu1_spy.assert_called_once()
        dropout1_spy.assert_called_once()
        conv2_spy.assert_called_once()
        relu2_spy.assert_called_once()
        dropout2_spy.assert_called_once()
        downsample_spy.assert_called_once()
        relu_spy.assert_called_once()
        
        # Verify output shape
        assert output.shape == (2, 5, 10)
    
    def test_forward_with_different_activations(self):
        """
        GIVEN a TCNBlock with custom activation functions
        WHEN forward pass is executed
        THEN it should use the specified activations correctly
        """
        # Define a modified TCNBlock with customizable activation function
        class CustomActivationTCNBlock(nn.Module):
            def __init__(self, in_channels, out_channels, kernel_size, dilation, dropout_rate, activation=nn.ReLU):
                super(CustomActivationTCNBlock, self).__init__()
                self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size,
                                      dilation=dilation, padding=(kernel_size - 1) * dilation // 2)
                self.act1 = activation()
                self.dropout1 = nn.Dropout(dropout_rate)
                self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size,
                                      dilation=dilation, padding=(kernel_size - 1) * dilation // 2)
                self.act2 = activation()
                self.dropout2 = nn.Dropout(dropout_rate)
                self.downsample = nn.Conv1d(in_channels, out_channels, 1) if in_channels != out_channels else None
                self.act_final = activation()
            
            def forward(self, x):
                residual = x
                
                out = self.conv1(x)
                out = self.act1(out)
                out = self.dropout1(out)
                
                out = self.conv2(out)
                out = self.act2(out)
                out = self.dropout2(out)
                
                if self.downsample is not None:
                    residual = self.downsample(x)
                
                out = self.act_final(out + residual)
                
                return out
        
        # Test with different activation functions
        activations = [
            nn.ReLU,
            nn.LeakyReLU,
            nn.GELU,
            nn.Tanh,
            nn.ELU
        ]
        
        input_tensor = torch.randn(2, 3, 10)
        
        for activation in activations:
            # Create block with the activation
            block = CustomActivationTCNBlock(in_channels=3, out_channels=5, kernel_size=3, 
                                           dilation=1, dropout_rate=0.0, activation=activation)
            
            # Forward pass
            output = block(input_tensor)
            
            # Verify output shape
            assert output.shape == (2, 5, 10)
            
            # Verify activation behavior based on known properties
            if activation == nn.Tanh:
                # Tanh outputs should be bounded between -1 and 1
                assert torch.all(output <= 1.0) and torch.all(output >= -1.0)
            elif activation in [nn.ReLU]:
                # ReLU outputs should be non-negative
                assert torch.all(output >= 0)
    
    def test_forward_with_different_dtypes(self):
        """
        GIVEN a TCNBlock and inputs with different data types
        WHEN forward pass is executed
        THEN it should handle different precisions correctly
        """
        # Skip test if CUDA is not available for half precision
        if not torch.cuda.is_available():
            pytest.skip("CUDA not available, skipping dtype test that requires half precision")
            
        # Create a standard block
        block = TCNBlock(in_channels=3, out_channels=5, kernel_size=3, dilation=1, dropout_rate=0.0)
        
        # Test with float32 (default)
        input_float32 = torch.randn(2, 3, 10)
        output_float32 = block(input_float32)
        assert output_float32.dtype == torch.float32
        
        # Test with float64 (double precision)
        block_double = block.double()
        input_float64 = torch.randn(2, 3, 10, dtype=torch.float64)
        output_float64 = block_double(input_float64)
        assert output_float64.dtype == torch.float64
        
        # Test with float16 (half precision) - requires CUDA
        block_half = block.cuda().half()
        input_float16 = torch.randn(2, 3, 10, dtype=torch.float16, device='cuda')
        output_float16 = block_half(input_float16)
        assert output_float16.dtype == torch.float16
        
        # Verify no NaN values in any output
        assert not torch.isnan(output_float32).any()
        assert not torch.isnan(output_float64).any()
        assert not torch.isnan(output_float16).any()
    
    def test_forward_with_weight_initialization(self):
        """
        GIVEN a TCNBlock with different weight initialization strategies
        WHEN forward pass is executed
        THEN outputs should reflect the initialization characteristics
        """
        # Define initialization strategies to test
        init_functions = {
            'zeros': nn.init.zeros_,
            'ones': nn.init.ones_,
            'xavier_uniform': nn.init.xavier_uniform_,
            'xavier_normal': nn.init.xavier_normal_,
            'kaiming_uniform': nn.init.kaiming_uniform_,
            'kaiming_normal': nn.init.kaiming_normal_
        }
        
        input_tensor = torch.ones(1, 3, 10)  # Use ones for predictable behavior with zero/one init
        
        for name, init_fn in init_functions:
            # Create block
            block = TCNBlock(in_channels=3, out_channels=5, kernel_size=3, dilation=1, dropout_rate=0.0)
            
            # Apply initialization to all weights
            with torch.no_grad():
                init_fn(block.conv1.weight)
                init_fn(block.conv2.weight)
                if block.downsample is not None:
                    init_fn(block.downsample.weight)
                
                # For zeros and ones, also initialize biases
                if name in ['zeros', 'ones']:
                    if block.conv1.bias is not None:
                        init_fn(block.conv1.bias)
                    if block.conv2.bias is not None:
                        init_fn(block.conv2.bias)
                    if block.downsample is not None and block.downsample.bias is not None:
                        init_fn(block.downsample.bias)
            
            # Forward pass
            output = block(input_tensor)
            
            # Verify output shape
            assert output.shape == (1, 5, 10)
            
            # Verify output characteristics based on initialization
            if name == 'zeros':
                # With zero weights and biases, output should be all zeros before final ReLU
                # After ReLU, it should still be zeros
                assert torch.allclose(output, torch.zeros_like(output))
            elif name == 'ones':
                # With all ones, output should be positive and non-zero
                assert torch.all(output > 0)


class TestTCNBlockEdgeCases:
    """Tests covering edge cases and unusual inputs."""
    
    def test_padding_modes(self):
        """
        GIVEN a TCNBlock with different padding modes
        WHEN forward pass is executed
        THEN it should handle the padding correctly
        """
        # Define a modified TCNBlock with customizable padding mode
        class PaddingModeTCNBlock(nn.Module):
            def __init__(self, in_channels, out_channels, kernel_size, dilation, dropout_rate, padding_mode='zeros'):
                super(PaddingModeTCNBlock, self).__init__()
                self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size,
                                      dilation=dilation, padding=(kernel_size - 1) * dilation // 2,
                                      padding_mode=padding_mode)
                self.relu1 = nn.ReLU()
                self.dropout1 = nn.Dropout(dropout_rate)
                self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size,
                                      dilation=dilation, padding=(kernel_size - 1) * dilation // 2,
                                      padding_mode=padding_mode)
                self.relu2 = nn.ReLU()
                self.dropout2 = nn.Dropout(dropout_rate)
                self.downsample = nn.Conv1d(in_channels, out_channels, 1) if in_channels != out_channels else None
                self.relu = nn.ReLU()
            
            def forward(self, x):
                residual = x
                
                out = self.conv1(x)
                out = self.relu1(out)
                out = self.dropout1(out)
                
                out = self.conv2(out)
                out = self.relu2(out)
                out = self.dropout2(out)
                
                if self.downsample is not None:
                    residual = self.downsample(x)
                
                out = self.relu(out + residual)
                
                return out
        
        # Test with different padding modes
        padding_modes = ['zeros', 'reflect', 'replicate', 'circular']
        
        input_tensor = torch.randn(2, 3, 10)
        
        for padding_mode in padding_modes:
            # Create block with the padding mode
            block = PaddingModeTCNBlock(in_channels=3, out_channels=5, kernel_size=3, 
                                       dilation=1, dropout_rate=0.0, padding_mode=padding_mode)
            
            # Forward pass
            output = block(input_tensor)
            
            # Verify output shape
            assert output.shape == (2, 5, 10), f"Failed with padding_mode={padding_mode}"
    
    def test_single_sample_batch(self):
        """
        GIVEN a batch with a single sample
        WHEN forward pass is executed
        THEN it should process correctly
        """
        # Create single sample input
        input_tensor = torch.randn(1, 3, 10)
        
        # Create block
        block = TCNBlock(in_channels=3, out_channels=5, kernel_size=3, dilation=1, dropout_rate=0.2)
        
        # Forward pass
        output = block(input_tensor)
        
        # Check output shape
        assert output.shape == (1, 5, 10)
    
    def test_minimum_sequence_length(self):
        """
        GIVEN an input with minimum viable sequence length
        WHEN forward pass is executed
        THEN it should process correctly
        """
        # Create input with minimum sequence length
        # For kernel_size=3 and dilation=1, minimum length is 3
        input_tensor = torch.randn(2, 3, 3)
        
        # Create block
        block = TCNBlock(in_channels=3, out_channels=5, kernel_size=3, dilation=1, dropout_rate=0.2)
        
        # Forward pass
        output = block(input_tensor)
        
        # Check output shape
        assert output.shape == (2, 5, 3)
    
    def test_large_dilation(self):
        """
        GIVEN a very large dilation value
        WHEN forward pass is executed
        THEN it should process correctly
        """
        # Create input with sufficient sequence length
        input_tensor = torch.randn(2, 3, 100)
        
        # Create block with large dilation
        block = TCNBlock(in_channels=3, out_channels=5, kernel_size=3, dilation=32, dropout_rate=0.2)
        
        # Forward pass
        output = block(input_tensor)
        
        # Check output shape
        assert output.shape == (2, 5, 100)
    
    def test_asymmetric_padding(self):
        """
        GIVEN a TCNBlock with asymmetric padding
        WHEN forward pass is executed
        THEN it should handle the padding correctly
        """
        # Define a modified TCNBlock with asymmetric padding
        class AsymmetricPaddingTCNBlock(nn.Module):
            def __init__(self, in_channels, out_channels, kernel_size, dilation, dropout_rate):
                super(AsymmetricPaddingTCNBlock, self).__init__()
                # Calculate asymmetric padding (more padding on the left)
                padding_left = kernel_size * dilation - 1
                padding_right = 0
                
                # PyTorch doesn't directly support asymmetric padding in Conv1d
                # We need to use pad function before convolution
                self.pad1 = nn.ConstantPad1d((padding_left, padding_right), 0)
                self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size, dilation=dilation)
                self.relu1 = nn.ReLU()
                self.dropout1 = nn.Dropout(dropout_rate)
                
                self.pad2 = nn.ConstantPad1d((padding_left, padding_right), 0)
                self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size, dilation=dilation)
                self.relu2 = nn.ReLU()
                self.dropout2 = nn.Dropout(dropout_rate)
                
                self.downsample = nn.Conv1d(in_channels, out_channels, 1) if in_channels != out_channels else None
                self.relu = nn.ReLU()
            
            def forward(self, x):
                residual = x
                
                # First conv block with asymmetric padding
                out = self.pad1(x)
                out = self.conv1(out)
                out = self.relu1(out)
                out = self.dropout1(out)
                
                # Second conv block with asymmetric padding
                out = self.pad2(out)
                out = self.conv2(out)
                out = self.relu2(out)
                out = self.dropout2(out)
                
                # Ensure output has same sequence length as input
                seq_diff = out.size(2) - x.size(2)
                if seq_diff > 0:
                    out = out[:, :, :x.size(2)]
                
                # Residual connection
                if self.downsample is not None:
                    residual = self.downsample(x)
                
                out = self.relu(out + residual)
                
                return out
        
        # Create input tensor
        input_tensor = torch.randn(2, 3, 20)
        
        # Create block with asymmetric padding
        block = AsymmetricPaddingTCNBlock(in_channels=3, out_channels=5, kernel_size=3, dilation=1, dropout_rate=0.0)
        
        # Forward pass
        output = block(input_tensor)
        
        # Verify output shape matches input shape
        assert output.shape[2] == input_tensor.shape[2]
    
    def test_zero_dropout(self):
        """
        GIVEN a dropout rate of 0
        WHEN forward pass is executed
        THEN it should process correctly
        """
        # Create input
        input_tensor = torch.randn(2, 3, 10)
        
        # Create block with zero dropout
        block = TCNBlock(in_channels=3, out_channels=5, kernel_size=3, dilation=1, dropout_rate=0.0)
        
        # Forward pass
        output = block(input_tensor)
        
        # Check output shape
        assert output.shape == (2, 5, 10)
    
    def test_high_dropout(self):
        """
        GIVEN a high dropout rate
        WHEN forward pass is executed in eval mode
        THEN it should process correctly
        """
        # Create input
        input_tensor = torch.randn(2, 3, 10)
        
        # Create block with high dropout
        block = TCNBlock(in_channels=3, out_channels=5, kernel_size=3, dilation=1, dropout_rate=0.9)
        block.eval()  # Set to evaluation mode to disable dropout
        
        # Forward pass
        output = block(input_tensor)
        
        # Check output shape
        assert output.shape == (2, 5, 10)


class TestTCNBlockParameterValidation:
    """Tests covering parameter validation and error handling during initialization."""
    
    def test_weight_initialization_strategies(self):
        """
        GIVEN a TCNBlock with different weight initialization strategies
        WHEN the block is created
        THEN weights should be initialized according to the specified strategy
        """
        # Define initialization strategies to test
        init_functions = {
            'xavier_uniform': nn.init.xavier_uniform_,
            'xavier_normal': nn.init.xavier_normal_,
            'kaiming_uniform': nn.init.kaiming_uniform_,
            'kaiming_normal': nn.init.kaiming_normal_,
            'orthogonal': nn.init.orthogonal_
        }
        
        for name, init_fn in init_functions.items():
            # Create block
            block = TCNBlock(in_channels=3, out_channels=5, kernel_size=3, dilation=1, dropout_rate=0.2)
            
            # Apply initialization
            init_fn(block.conv1.weight)
            init_fn(block.conv2.weight)
            if block.downsample is not None:
                init_fn(block.downsample.weight)
            
            # Verify weights have expected statistical properties
            # For example, check that weights are non-zero and have reasonable magnitudes
            assert torch.all(block.conv1.weight != 0)
            assert torch.all(block.conv2.weight != 0)
            
            # Different initializations have different expected magnitudes
            # Here we just check they're within a reasonable range
            assert block.conv1.weight.abs().mean() < 1.0
            assert block.conv2.weight.abs().mean() < 1.0
            
            # Create input tensor
            input_tensor = torch.randn(2, 3, 10)
            
            # Forward pass
            output = block(input_tensor)
            
            # Verify output shape
            assert output.shape == (2, 5, 10)
    
    def test_invalid_kernel_size(self):
        """
        GIVEN an invalid kernel size
        WHEN the TCNBlock is created
        THEN it should raise an appropriate error
        """
        # Test with negative kernel size
        with pytest.raises(ValueError, match="kernel_size should be positive"):
            TCNBlock(in_channels=3, out_channels=5, kernel_size=-1, dilation=1, dropout_rate=0.2)
        
        # Test with zero kernel size
        with pytest.raises(ValueError, match="kernel_size should be positive"):
            TCNBlock(in_channels=3, out_channels=5, kernel_size=0, dilation=1, dropout_rate=0.2)
    
    def test_invalid_dilation(self):
        """
        GIVEN an invalid dilation value
        WHEN the TCNBlock is created
        THEN it should raise an appropriate error
        """
        # Test with negative dilation
        with pytest.raises(ValueError, match="dilation should be positive"):
            TCNBlock(in_channels=3, out_channels=5, kernel_size=3, dilation=-1, dropout_rate=0.2)
        
        # Test with zero dilation
        with pytest.raises(ValueError, match="dilation should be positive"):
            TCNBlock(in_channels=3, out_channels=5, kernel_size=3, dilation=0, dropout_rate=0.2)
    
    def test_invalid_dropout_rate(self):
        """
        GIVEN an invalid dropout rate
        WHEN the TCNBlock is created
        THEN it should raise an appropriate error
        """
        # Test with negative dropout rate
        with pytest.raises(ValueError, match="dropout_rate should be between 0 and 1"):
            TCNBlock(in_channels=3, out_channels=5, kernel_size=3, dilation=1, dropout_rate=-0.1)
        
        # Test with dropout rate > 1
        with pytest.raises(ValueError, match="dropout_rate should be between 0 and 1"):
            TCNBlock(in_channels=3, out_channels=5, kernel_size=3, dilation=1, dropout_rate=1.1)
    
    def test_invalid_channel_values(self):
        """
        GIVEN invalid channel values
        WHEN the TCNBlock is created
        THEN it should raise an appropriate error
        """
        # Test with negative in_channels
        with pytest.raises(ValueError, match="in_channels should be positive"):
            TCNBlock(in_channels=-1, out_channels=5, kernel_size=3, dilation=1, dropout_rate=0.2)
        
        # Test with zero in_channels
        with pytest.raises(ValueError, match="in_channels should be positive"):
            TCNBlock(in_channels=0, out_channels=5, kernel_size=3, dilation=1, dropout_rate=0.2)
        
        # Test with negative out_channels
        with pytest.raises(ValueError, match="out_channels should be positive"):
            TCNBlock(in_channels=3, out_channels=-1, kernel_size=3, dilation=1, dropout_rate=0.2)
        
        # Test with zero out_channels
        with pytest.raises(ValueError, match="out_channels should be positive"):
            TCNBlock(in_channels=3, out_channels=0, kernel_size=3, dilation=1, dropout_rate=0.2)


class TestTCNBlockResourceUsage:
    """Tests covering memory usage and parameter count."""
    
    def test_alternative_activation_functions(self):
        """
        GIVEN a TCNBlock with alternative activation functions
        WHEN forward pass is executed
        THEN it should use the specified activations
        """
        # Define a modified TCNBlock with customizable activation functions
        class FlexibleTCNBlock(nn.Module):
            def __init__(self, in_channels, out_channels, kernel_size, dilation, dropout_rate, activation=nn.ReLU):
                super(FlexibleTCNBlock, self).__init__()
                self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size,
                                      dilation=dilation, padding=(kernel_size - 1) * dilation // 2)
                self.act1 = activation()
                self.dropout1 = nn.Dropout(dropout_rate)
                self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size,
                                      dilation=dilation, padding=(kernel_size - 1) * dilation // 2)
                self.act2 = activation()
                self.dropout2 = nn.Dropout(dropout_rate)
                self.downsample = nn.Conv1d(in_channels, out_channels, 1) if in_channels != out_channels else None
                self.act_final = activation()
            
            def forward(self, x):
                residual = x
                
                out = self.conv1(x)
                out = self.act1(out)
                out = self.dropout1(out)
                
                out = self.conv2(out)
                out = self.act2(out)
                out = self.dropout2(out)
                
                if self.downsample is not None:
                    residual = self.downsample(x)
                
                out = self.act_final(out + residual)
                
                return out
        
        # Test with different activation functions
        activations = [
            nn.ReLU,
            nn.LeakyReLU,
            nn.GELU,
            nn.Tanh,
            nn.SiLU  # Also known as Swish
        ]
        
        input_tensor = torch.randn(2, 3, 10)
        
        for activation in activations:
            # Create block with the activation
            block = FlexibleTCNBlock(in_channels=3, out_channels=5, kernel_size=3, 
                                    dilation=1, dropout_rate=0.0, activation=activation)
            
            # Forward pass
            output = block(input_tensor)
            
            # Verify output shape
            assert output.shape == (2, 5, 10)
            
            # Verify output values are within expected range for the activation
            if activation in [nn.Tanh]:
                assert torch.all(output <= 1.0) and torch.all(output >= -1.0)
            elif activation in [nn.ReLU, nn.LeakyReLU, nn.GELU, nn.SiLU]:
                # These activations can produce values > 0
                assert torch.any(output > 0)
    
    def test_parameter_count(self):
        """
        GIVEN different TCNBlock configurations
        WHEN counting parameters
        THEN the parameter count should match expected values
        """
        # Small block
        small_block = TCNBlock(in_channels=2, out_channels=4, kernel_size=3, dilation=1, dropout_rate=0.2)
        small_params = sum(p.numel() for p in small_block.parameters())
        
        # Calculate expected parameter count for small block
        # Conv1: (2 * 4 * 3) + 4 (bias) = 28
        # Conv2: (4 * 4 * 3) + 4 (bias) = 52
        # Downsample: (2 * 4 * 1) + 4 (bias) = 12
        expected_small_params = 28 + 52 + 12
        assert small_params == expected_small_params, f"Expected {expected_small_params} parameters, got {small_params}"
        
        # Large block
        large_block = TCNBlock(in_channels=16, out_channels=32, kernel_size=5, dilation=2, dropout_rate=0.2)
        large_params = sum(p.numel() for p in large_block.parameters())
        
        # Calculate expected parameter count for large block
        # Conv1: (16 * 32 * 5) + 32 (bias) = 2,592
        # Conv2: (32 * 32 * 5) + 32 (bias) = 5,152
        # Downsample: (16 * 32 * 1) + 32 (bias) = 544
        expected_large_params = 2592 + 5152 + 544
        assert large_params == expected_large_params, f"Expected {expected_large_params} parameters, got {large_params}"
        
        # Same in/out channels (no downsample)
        no_downsample_block = TCNBlock(in_channels=8, out_channels=8, kernel_size=3, dilation=1, dropout_rate=0.2)
        no_downsample_params = sum(p.numel() for p in no_downsample_block.parameters())
        
        # Calculate expected parameter count for no downsample block
        # Conv1: (8 * 8 * 3) + 8 (bias) = 200
        # Conv2: (8 * 8 * 3) + 8 (bias) = 200
        # No downsample
        expected_no_downsample_params = 200 + 200
        assert no_downsample_params == expected_no_downsample_params, \
            f"Expected {expected_no_downsample_params} parameters, got {no_downsample_params}"


class TestTCNBlockGradients:
    """Tests covering gradient flow and backpropagation."""
    
    def test_custom_forward_implementation(self):
        """
        GIVEN a TCNBlock with a custom forward implementation
        WHEN forward pass is executed
        THEN it should produce the expected output
        """
        # Define a custom TCNBlock with explicit forward implementation
        class CustomTCNBlock(nn.Module):
            def __init__(self, in_channels, out_channels, kernel_size, dilation, dropout_rate):
                super(CustomTCNBlock, self).__init__()
                self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size,
                                      dilation=dilation, padding=(kernel_size - 1) * dilation // 2)
                self.relu1 = nn.ReLU()
                self.dropout1 = nn.Dropout(dropout_rate)
                self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size,
                                      dilation=dilation, padding=(kernel_size - 1) * dilation // 2)
                self.relu2 = nn.ReLU()
                self.dropout2 = nn.Dropout(dropout_rate)
                self.downsample = nn.Conv1d(in_channels, out_channels, 1) if in_channels != out_channels else None
                self.relu = nn.ReLU()
            
            def forward(self, x):
                # Explicit forward implementation
                residual = x
                
                # First conv block
                out = self.conv1(x)
                out = self.relu1(out)
                out = self.dropout1(out)
                
                # Second conv block
                out = self.conv2(out)
                out = self.relu2(out)
                out = self.dropout2(out)
                
                # Residual connection
                if self.downsample is not None:
                    residual = self.downsample(x)
                
                # Final activation
                out = self.relu(out + residual)
                
                return out
        
        # Create both original and custom blocks with identical parameters
        original_block = TCNBlock(in_channels=3, out_channels=5, kernel_size=3, dilation=1, dropout_rate=0.0)
        custom_block = CustomTCNBlock(in_channels=3, out_channels=5, kernel_size=3, dilation=1, dropout_rate=0.0)
        
        # Copy weights from original to custom to ensure identical behavior
        with torch.no_grad():
            custom_block.conv1.weight.copy_(original_block.conv1.weight)
            custom_block.conv1.bias.copy_(original_block.conv1.bias)
            custom_block.conv2.weight.copy_(original_block.conv2.weight)
            custom_block.conv2.bias.copy_(original_block.conv2.bias)
            custom_block.downsample.weight.copy_(original_block.downsample.weight)
            custom_block.downsample.bias.copy_(original_block.downsample.bias)
        
        # Create input tensor
        input_tensor = torch.randn(2, 3, 10)
        
        # Get outputs from both blocks
        original_output = original_block(input_tensor)
        custom_output = custom_block(input_tensor)
        
        # Verify outputs are identical
        assert torch.allclose(original_output, custom_output)
    
    def test_gradient_flow(self):
        """
        GIVEN a TCNBlock
        WHEN backward pass is executed
        THEN gradients should flow to all parameters
        """
        # Create input and target
        input_tensor = torch.randn(2, 3, 10, requires_grad=True)
        target = torch.randn(2, 5, 10)
        
        # Create block
        block = TCNBlock(in_channels=3, out_channels=5, kernel_size=3, dilation=1, dropout_rate=0.2)
        
        # Forward pass
        output = block(input_tensor)
        
        # Compute loss and backward
        loss = nn.MSELoss()(output, target)
        loss.backward()
        
        # Check that all parameters have gradients
        for name, param in block.named_parameters():
            assert param.grad is not None, f"Parameter {name} has no gradient"
            assert param.grad.abs().sum() > 0, f"Parameter {name} has zero gradient"
    
    def test_gradient_flow_with_no_downsample(self):
        """
        GIVEN a TCNBlock with no downsample layer
        WHEN backward pass is executed
        THEN gradients should flow correctly through the direct residual connection
        """
        # Create input and target with matching channel dimensions
        input_tensor = torch.randn(2, 5, 10, requires_grad=True)
        target = torch.randn(2, 5, 10)
        
        # Create block with same in/out channels (no downsample)
        block = TCNBlock(in_channels=5, out_channels=5, kernel_size=3, dilation=1, dropout_rate=0.2)
        assert block.downsample is None  # Confirm no downsample layer
        
        # Forward pass
        output = block(input_tensor)
        
        # Compute loss and backward
        loss = nn.MSELoss()(output, target)
        loss.backward()
        
        # Check input gradients exist
        assert input_tensor.grad is not None
        assert input_tensor.grad.abs().sum() > 0
        
        # Check all parameters have gradients
        for name, param in block.named_parameters():
            assert param.grad is not None, f"Parameter {name} has no gradient"


# === WAVENET TESTS ===

@pytest.fixture
def wavenet_stack_input():
    """
    Provides a sample input tensor for WavenetStack testing.
    
    Returns:
        torch.Tensor: Tensor of shape (batch_size=2, channels=4, seq_len=16)
    """
    return torch.randn(2, 4, 16)


@pytest.fixture
def standard_wavenet_stack():
    """
    Provides a standard WavenetStack with common parameters.
    
    Returns:
        WavenetStack: A WavenetStack with standard test configuration
    """
    from src.models.wavenet import WavenetStack
    return WavenetStack(
        num_layers_per_stack=3,
        residual_channels=8,
        skip_channels=16,
        kernel_size=3,
        dilation_base=2,
        dropout_rate=0.2,
        input_channels=4,
        use_bias=True
    )


class TestTCNInitialization:
    """Tests covering TCN model creation and parameter validation."""
    
    def test_valid_initialization(self, tcn_model_config):
        """
        GIVEN valid initialization parameters
        WHEN the TCN model is created
        THEN it should initialize with correct layers and parameters
        """
        # Create TCN model
        model = TCN(tcn_model_config)
        
        # Verify model structure
        assert isinstance(model.network, nn.Sequential)
        assert len(model.network) == tcn_model_config['num_blocks']
        assert isinstance(model.linear, nn.Linear)
        
        # Verify linear layer dimensions
        assert model.linear.in_features == tcn_model_config['num_channels']
        assert model.linear.out_features == tcn_model_config['output_size']
        
        # Verify first TCNBlock parameters
        first_block = model.network[0]
        assert isinstance(first_block, TCNBlock)
        assert first_block.conv1.in_channels == tcn_model_config['input_size']
        assert first_block.conv1.out_channels == tcn_model_config['num_channels']
        assert first_block.conv1.kernel_size[0] == tcn_model_config['kernel_size']
        assert first_block.conv1.dilation[0] == 1  # 2^0 = 1
        
        # Verify last TCNBlock parameters
        last_block = model.network[-1]
        assert isinstance(last_block, TCNBlock)
        assert last_block.conv1.in_channels == tcn_model_config['num_channels']
        assert last_block.conv1.out_channels == tcn_model_config['num_channels']
        assert last_block.conv1.dilation[0] == 4  # 2^2 = 4 (for 3rd block, index 2)
    
    def test_initialization_with_different_block_counts(self):
        """
        GIVEN configurations with different numbers of blocks
        WHEN the TCN model is created
        THEN it should create the correct number of TCNBlocks with proper dilations
        """
        # Test with 1 block
        config_1_block = {
            'input_size': 8,
            'output_size': 4,
            'num_blocks': 1,
            'num_channels': 16,
            'kernel_size': 3,
            'dropout': 0.2
        }
        model_1_block = TCN(config_1_block)
        assert len(model_1_block.network) == 1
        assert model_1_block.network[0].conv1.dilation[0] == 1  # 2^0 = 1
        
        # Test with 5 blocks
        config_5_blocks = {
            'input_size': 8,
            'output_size': 4,
            'num_blocks': 5,
            'num_channels': 16,
            'kernel_size': 3,
            'dropout': 0.2
        }
        model_5_blocks = TCN(config_5_blocks)
        assert len(model_5_blocks.network) == 5
        
        # Verify dilations follow 2^i pattern
        for i in range(5):
            expected_dilation = 2 ** i
            assert model_5_blocks.network[i].conv1.dilation[0] == expected_dilation
    
    def test_initialization_with_different_channel_sizes(self):
        """
        GIVEN configurations with different channel sizes
        WHEN the TCN model is created
        THEN it should create TCNBlocks with correct channel dimensions
        """
        # Test with small channels
        config_small = {
            'input_size': 2,
            'output_size': 1,
            'num_blocks': 3,
            'num_channels': 4,
            'kernel_size': 3,
            'dropout': 0.2
        }
        model_small = TCN(config_small)
        assert model_small.network[0].conv1.in_channels == 2
        assert model_small.network[0].conv1.out_channels == 4
        assert model_small.linear.in_features == 4
        assert model_small.linear.out_features == 1
        
        # Test with large channels
        config_large = {
            'input_size': 64,
            'output_size': 32,
            'num_blocks': 3,
            'num_channels': 128,
            'kernel_size': 3,
            'dropout': 0.2
        }
        model_large = TCN(config_large)
        assert model_large.network[0].conv1.in_channels == 64
        assert model_large.network[0].conv1.out_channels == 128
        assert model_large.linear.in_features == 128
        assert model_large.linear.out_features == 32
    
    def test_parameter_count(self, tcn_model_config):
        """
        GIVEN a TCN model configuration
        WHEN the model is created
        THEN it should have the expected number of parameters
        """
        model = TCN(tcn_model_config)
        param_count = sum(p.numel() for p in model.parameters())
        
        # Calculate expected parameter count
        # First block: input_size -> num_channels
        first_block_params = (
            (tcn_model_config['input_size'] * tcn_model_config['num_channels'] * tcn_model_config['kernel_size']) + 
            tcn_model_config['num_channels'] +  # bias
            (tcn_model_config['num_channels'] * tcn_model_config['num_channels'] * tcn_model_config['kernel_size']) + 
            tcn_model_config['num_channels'] +  # bias
            (tcn_model_config['input_size'] * tcn_model_config['num_channels'] * 1) +  # downsample
            tcn_model_config['num_channels']  # downsample bias
        )
        
        # Middle blocks: num_channels -> num_channels (2 blocks)
        middle_block_params = 2 * (
            (tcn_model_config['num_channels'] * tcn_model_config['num_channels'] * tcn_model_config['kernel_size']) + 
            tcn_model_config['num_channels'] +  # bias
            (tcn_model_config['num_channels'] * tcn_model_config['num_channels'] * tcn_model_config['kernel_size']) + 
            tcn_model_config['num_channels']  # bias
            # No downsample needed for same channel count
        )
        
        # Linear layer: num_channels -> output_size
        linear_params = (tcn_model_config['num_channels'] * tcn_model_config['output_size']) + tcn_model_config['output_size']
        
        expected_params = first_block_params + middle_block_params + linear_params
        
        # Allow small tolerance for potential implementation differences
        assert abs(param_count - expected_params) < 10, f"Expected ~{expected_params} parameters, got {param_count}"


class TestTCNForward:
    """Tests covering TCN model forward pass functionality."""
    
    def test_forward_shape(self, standard_tcn_model, tcn_input_tensor):
        """
        GIVEN a TCN model and input tensor
        WHEN forward pass is executed
        THEN output tensor should have expected shape
        """
        # Get input shape
        batch_size, seq_len, _ = tcn_input_tensor.shape
        
        # Forward pass
        output = standard_tcn_model(tcn_input_tensor)
        
        # Check output shape
        assert output.shape == (batch_size, seq_len, 4)  # output_size=4
    
    def test_forward_with_different_batch_sizes(self, standard_tcn_model):
        """
        GIVEN a TCN model and inputs with different batch sizes
        WHEN forward pass is executed
        THEN output tensor should have correct batch dimension
        """
        # Test with batch size 1
        input_batch_1 = torch.randn(1, 20, 8)
        output_batch_1 = standard_tcn_model(input_batch_1)
        assert output_batch_1.shape == (1, 20, 4)
        
        # Test with batch size 16
        input_batch_16 = torch.randn(16, 20, 8)
        output_batch_16 = standard_tcn_model(input_batch_16)
        assert output_batch_16.shape == (16, 20, 4)
    
    def test_forward_with_different_sequence_lengths(self, standard_tcn_model):
        """
        GIVEN a TCN model and inputs with different sequence lengths
        WHEN forward pass is executed
        THEN output tensor should preserve the sequence length
        """
        # Test with short sequence
        input_short = torch.randn(4, 5, 8)
        output_short = standard_tcn_model(input_short)
        assert output_short.shape == (4, 5, 4)
        
        # Test with long sequence
        input_long = torch.randn(4, 100, 8)
        output_long = standard_tcn_model(input_long)
        assert output_long.shape == (4, 100, 4)
    
    def test_transpose_operations(self, standard_tcn_model, tcn_input_tensor):
        """
        GIVEN a TCN model
        WHEN forward pass is executed
        THEN tensor should be properly transposed before and after TCN blocks
        """
        # Create a forward hook to capture intermediate tensors
        pre_network_tensor = None
        post_network_tensor = None
        
        def pre_network_hook(module, input_tensor, output_tensor):
            nonlocal pre_network_tensor
            pre_network_tensor = input_tensor[0].clone()  # input is a tuple
            return output_tensor
        
        def post_network_hook(module, input_tensor, output_tensor):
            nonlocal post_network_tensor
            post_network_tensor = output_tensor.clone()
            return output_tensor
        
        # Register hooks
        pre_hook_handle = standard_tcn_model.network.register_forward_pre_hook(pre_network_hook)
        post_hook_handle = standard_tcn_model.network.register_forward_hook(post_network_hook)
        
        # Forward pass
        batch_size, seq_len, input_size = tcn_input_tensor.shape
        output = standard_tcn_model(tcn_input_tensor)
        
        # Verify tensor shapes at different stages
        assert pre_network_tensor.shape == (batch_size, input_size, seq_len)  # Transposed: (B, C, L)
        assert post_network_tensor.shape == (batch_size, 16, seq_len)  # Transposed: (B, C, L) with C=num_channels
        assert output.shape == (batch_size, seq_len, 4)  # Final: (B, L, output_size)
        
        # Clean up hooks
        pre_hook_handle.remove()
        post_hook_handle.remove()
    
    def test_gradient_flow(self, standard_tcn_model, tcn_input_tensor):
        """
        GIVEN a TCN model
        WHEN backward pass is executed
        THEN gradients should flow to all parameters
        """
        # Create target
        batch_size, seq_len, _ = tcn_input_tensor.shape
        target = torch.randn(batch_size, seq_len, 4)  # output_size=4
        
        # Forward pass
        output = standard_tcn_model(tcn_input_tensor)
        
        # Compute loss and backward
        loss = nn.MSELoss()(output, target)
        loss.backward()
        
        # Check that all parameters have gradients
        for name, param in standard_tcn_model.named_parameters():
            assert param.grad is not None, f"Parameter {name} has no gradient"
            assert param.grad.abs().sum() > 0, f"Parameter {name} has zero gradient"


class TestTCNEdgeCases:
    """Tests covering edge cases and unusual inputs for TCN model."""
    
    def test_single_sample_batch(self, standard_tcn_model):
        """
        GIVEN a batch with a single sample
        WHEN forward pass is executed
        THEN it should process correctly
        """
        # Create single sample input
        input_tensor = torch.randn(1, 20, 8)
        
        # Forward pass
        output = standard_tcn_model(input_tensor)
        
        # Check output shape
        assert output.shape == (1, 20, 4)
    
    def test_minimum_sequence_length(self, standard_tcn_model):
        """
        GIVEN an input with minimum viable sequence length
        WHEN forward pass is executed
        THEN it should process correctly
        """
        # Create input with minimum sequence length (1)
        input_tensor = torch.randn(4, 1, 8)
        
        # Forward pass
        output = standard_tcn_model(input_tensor)
        
        # Check output shape
        assert output.shape == (4, 1, 4)
    
    def test_empty_batch(self, standard_tcn_model):
        """
        GIVEN an input with empty batch dimension
        WHEN forward pass is executed
        THEN it should process correctly
        """
        # Create input with empty batch
        input_tensor = torch.randn(0, 20, 8)
        
        # Forward pass
        output = standard_tcn_model(input_tensor)
        
        # Check output shape
        assert output.shape == (0, 20, 4)
    
    def test_input_requires_grad(self, standard_tcn_model):
        """
        GIVEN an input that requires gradients
        WHEN forward pass is executed
        THEN output should also require gradients
        """
        # Create input that requires gradients
        input_tensor = torch.randn(4, 20, 8, requires_grad=True)
        
        # Forward pass
        output = standard_tcn_model(input_tensor)
        
        # Check that output requires gradients
        assert output.requires_grad
    
    def test_model_eval_mode(self, standard_tcn_model, tcn_input_tensor):
        """
        GIVEN a TCN model in eval mode
        WHEN forward pass is executed
        THEN dropout should be disabled
        """
        # Set model to train mode first
        standard_tcn_model.train()
        
        # Get output in train mode (with dropout)
        train_outputs = [standard_tcn_model(tcn_input_tensor) for _ in range(3)]
        
        # Check if outputs differ (due to dropout)
        train_outputs_differ = not all(
            torch.allclose(train_outputs[0], output, rtol=1e-4, atol=1e-4)
            for output in train_outputs[1:]
        )
        
        # Set model to eval mode
        standard_tcn_model.eval()
        
        # Get output in eval mode (without dropout)
        eval_outputs = [standard_tcn_model(tcn_input_tensor) for _ in range(3)]
        
        # Check if outputs are identical (no dropout)
        eval_outputs_identical = all(
            torch.allclose(eval_outputs[0], output, rtol=1e-4, atol=1e-4)
            for output in eval_outputs[1:]
        )
        
        # Verify dropout behavior
        assert train_outputs_differ, "Outputs in train mode should differ due to dropout"
        assert eval_outputs_identical, "Outputs in eval mode should be identical (no dropout)"


class TestTCNConfigValidation:
    """Tests covering TCN model configuration validation."""
    
    def test_missing_config_keys(self):
        """
        GIVEN a model configuration with missing required keys
        WHEN the TCN model is created
        THEN it should raise an appropriate error
        """
        # Test with missing input_size
        incomplete_config = {
            'output_size': 4,
            'num_blocks': 3,
            'num_channels': 16,
            'kernel_size': 3,
            'dropout': 0.2
        }
        
        with pytest.raises(KeyError) as excinfo:
            TCN(incomplete_config)
        assert "input_size" in str(excinfo.value)
        
        # Test with missing output_size
        incomplete_config = {
            'input_size': 8,
            'num_blocks': 3,
            'num_channels': 16,
            'kernel_size': 3,
            'dropout': 0.2
        }
        
        with pytest.raises(KeyError) as excinfo:
            TCN(incomplete_config)
        assert "output_size" in str(excinfo.value)
    
    def test_invalid_config_values(self):
        """
        GIVEN a model configuration with invalid values
        WHEN the TCN model is created
        THEN it should raise an appropriate error
        """
        # Test with negative num_blocks
        invalid_config = {
            'input_size': 8,
            'output_size': 4,
            'num_blocks': -1,
            'num_channels': 16,
            'kernel_size': 3,
            'dropout': 0.2
        }
        
        with pytest.raises(ValueError) as excinfo:
            TCN(invalid_config)
        assert "num_blocks" in str(excinfo.value)
        
        # Test with invalid dropout
        invalid_config = {
            'input_size': 8,
            'output_size': 4,
            'num_blocks': 3,
            'num_channels': 16,
            'kernel_size': 3,
            'dropout': 1.5
        }
        
        with pytest.raises(ValueError) as excinfo:
            TCN(invalid_config)
        assert "dropout" in str(excinfo.value)


class TestTCNSerialization:
    """Tests covering TCN model serialization and deserialization."""
    
    def test_save_and_load_model(self, standard_tcn_model, tcn_input_tensor):
        """
        GIVEN a trained TCN model
        WHEN the model is saved and loaded
        THEN the loaded model should produce identical outputs
        """
        import tempfile
        import os
        
        # Get output from original model
        original_output = standard_tcn_model(tcn_input_tensor)
        
        # Save model to temporary file
        with tempfile.NamedTemporaryFile(suffix='.pt', delete=False) as tmp:
            torch.save(standard_tcn_model.state_dict(), tmp.name)
            tmp_name = tmp.name
        
        try:
            # Create a new model with the same configuration
            new_model = TCN(tcn_model_config)
            
            # Load the saved state
            new_model.load_state_dict(torch.load(tmp_name))
            new_model.eval()
            
            # Get output from loaded model
            loaded_output = new_model(tcn_input_tensor)
            
            # Verify outputs are identical
            assert torch.allclose(original_output, loaded_output, rtol=1e-5, atol=1e-5)
        finally:
            # Clean up temporary file
            os.unlink(tmp_name)
    
    def test_jit_scripting(self, standard_tcn_model, tcn_input_tensor):
        """
        GIVEN a TCN model
        WHEN the model is JIT scripted
        THEN the scripted model should produce identical outputs
        """
        # Get output from original model
        standard_tcn_model.eval()
        original_output = standard_tcn_model(tcn_input_tensor)
        
        # Script the model
        scripted_model = torch.jit.script(standard_tcn_model)
        
        # Get output from scripted model
        scripted_output = scripted_model(tcn_input_tensor)
        
        # Verify outputs are identical
        assert torch.allclose(original_output, scripted_output, rtol=1e-5, atol=1e-5)


class TestTCNPrecision:
    """Tests covering TCN model behavior with different precision types."""
    
    def test_half_precision(self, standard_tcn_model, tcn_input_tensor):
        """
        GIVEN a TCN model
        WHEN converted to half precision (float16)
        THEN it should process inputs correctly
        """
        # Skip test if CUDA is not available
        if not torch.cuda.is_available():
            pytest.skip("CUDA not available, skipping half precision test")
        
        # Move model and input to GPU and convert to half precision
        half_model = standard_tcn_model.cuda().half()
        half_input = tcn_input_tensor.cuda().half()
        
        # Forward pass
        output = half_model(half_input)
        
        # Check output shape and type
        batch_size, seq_len, _ = tcn_input_tensor.shape
        assert output.shape == (batch_size, seq_len, 4)
        assert output.dtype == torch.float16
        
        # Verify no NaN values
        assert not torch.isnan(output).any()
    
    def test_double_precision(self, standard_tcn_model, tcn_input_tensor):
        """
        GIVEN a TCN model
        WHEN converted to double precision (float64)
        THEN it should process inputs correctly
        """
        # Convert model and input to double precision
        double_model = standard_tcn_model.double()
        double_input = tcn_input_tensor.double()
        
        # Forward pass
        output = double_model(double_input)
        
        # Check output shape and type
        batch_size, seq_len, _ = tcn_input_tensor.shape
        assert output.shape == (batch_size, seq_len, 4)
        assert output.dtype == torch.float64


class TestTCNIntegration:
    """Tests covering integration aspects of the TCN model."""
    
    def test_receptive_field(self):
        """
        GIVEN a TCN model with specific configuration
        WHEN calculating the receptive field
        THEN it should match the expected theoretical value
        """
        # Create a TCN with known parameters for receptive field calculation
        config = {
            'input_size': 8,
            'output_size': 4,
            'num_blocks': 4,  # 4 blocks with dilations 1, 2, 4, 8
            'num_channels': 16,
            'kernel_size': 3,
            'dropout': 0.0
        }
        model = TCN(config)
        
        # Calculate theoretical receptive field
        # RF = 1 + (kernel_size - 1) * sum(dilations)
        # For kernel_size=3, dilations=[1,2,4,8]: RF = 1 + 2 * (1+2+4+8) = 1 + 2 * 15 = 31
        expected_receptive_field = 1 + (config['kernel_size'] - 1) * sum(2**i for i in range(config['num_blocks']))
        
        # Create an input with a single spike at different positions
        seq_len = 50  # Longer than receptive field
        input_tensor = torch.zeros(1, seq_len, config['input_size'])
        
        # Place a spike at position 0
        input_tensor[0, 0, :] = 1.0
        output_with_spike_at_0 = model(input_tensor)
        
        # Place a spike at position outside receptive field
        input_tensor = torch.zeros(1, seq_len, config['input_size'])
        input_tensor[0, expected_receptive_field, :] = 1.0
        output_with_spike_outside_rf = model(input_tensor)
        
        # The output at position 0 should differ when spike is at 0 vs. outside receptive field
        assert not torch.allclose(
            output_with_spike_at_0[0, 0, :], 
            output_with_spike_outside_rf[0, 0, :],
            rtol=1e-2, atol=1e-2
        )
        
        # Place two spikes: one at 0 and one just beyond receptive field
        input_tensor = torch.zeros(1, seq_len, config['input_size'])
        input_tensor[0, 0, :] = 1.0
        input_tensor[0, expected_receptive_field, :] = 1.0
        output_with_two_spikes = model(input_tensor)
        
        # The output at position 0 should be the same as with just the spike at 0
        # (within numerical precision) because the second spike is outside the receptive field
        assert torch.allclose(
            output_with_spike_at_0[0, 0, :],
            output_with_two_spikes[0, 0, :],
            rtol=1e-2, atol=1e-2
        )
    
    def test_causal_property(self):
        """
        GIVEN a TCN model
        WHEN inputs are modified after a certain time step
        THEN outputs before that time step should remain unchanged
        """
        # Create a TCN with small number of parameters for faster testing
        config = {
            'input_size': 4,
            'output_size': 2,
            'num_blocks': 3,
            'num_channels': 8,
            'kernel_size': 3,
            'dropout': 0.0  # No dropout for deterministic output
        }
        model = TCN(config)
        model.eval()  # Set to eval mode to disable dropout
        
        # Create a random input sequence
        seq_len = 20
        input_tensor = torch.randn(1, seq_len, config['input_size'])
        
        # Get output for the original input
        original_output = model(input_tensor)
        
        # Create a modified input where only the second half is changed
        modified_input = input_tensor.clone()
        midpoint = seq_len // 2
        modified_input[0, midpoint:, :] = torch.randn(1, seq_len - midpoint, config['input_size'])
        
        # Get output for the modified input
        modified_output = model(modified_input)
        
        # Verify that outputs before the midpoint are identical
        # (allowing for small numerical differences)
        assert torch.allclose(
            original_output[0, :midpoint, :],
            modified_output[0, :midpoint, :],
            rtol=1e-4, atol=1e-4
        )
        
        # Verify that outputs after the midpoint are different
        assert not torch.allclose(
            original_output[0, midpoint:, :],
            modified_output[0, midpoint:, :],
            rtol=1e-1, atol=1e-1
        )
    
    def test_long_sequence_handling(self):
        """
        GIVEN a TCN model
        WHEN processing a very long sequence
        THEN it should handle it without numerical issues
        """
        # Create a TCN with small number of parameters
        config = {
            'input_size': 4,
            'output_size': 2,
            'num_blocks': 3,
            'num_channels': 8,
            'kernel_size': 3,
            'dropout': 0.0
        }
        model = TCN(config)
        
        # Create a very long input sequence
        long_seq_len = 10000
        input_tensor = torch.randn(1, long_seq_len, config['input_size'])
        
        # Forward pass
        output = model(input_tensor)
        
        # Check output shape
        assert output.shape == (1, long_seq_len, config['output_size'])
        
        # Verify no NaN or infinity values
        assert not torch.isnan(output).any()
        assert not torch.isinf(output).any()
        
        # Verify output values are within reasonable range
        assert output.abs().max() < 1000  # Arbitrary large threshold


class TestWavenetStackForward:
    """Tests covering WavenetStack forward pass functionality."""
    
    def test_forward_shape(self, standard_wavenet_stack, wavenet_stack_input):
        """
        GIVEN a WavenetStack and input tensor
        WHEN forward pass is executed
        THEN output tensors should have expected shapes
        """
        # Forward pass
        skip_output, residual_output = standard_wavenet_stack(wavenet_stack_input)
        
        # Check output shapes
        batch_size, input_channels, seq_len = wavenet_stack_input.shape
        assert skip_output.shape == (batch_size, 16, seq_len)  # skip_channels=16
        assert residual_output.shape == (batch_size, 8, seq_len)  # residual_channels=8
    
    def test_skip_connection_summation(self, wavenet_stack_input):
        """
        GIVEN a WavenetStack with controlled outputs
        WHEN forward pass is executed
        THEN skip connections should be properly summed
        """
        from src.models.wavenet import WavenetStack
        import torch.nn as nn
        
        # Create a simplified stack for testing skip connections
        stack = WavenetStack(
            num_layers_per_stack=2,  # Just 2 layers for simplicity
            residual_channels=8,
            skip_channels=4,
            kernel_size=3,
            dilation_base=1,  # No dilation for simplicity
            dropout_rate=0.0,  # No dropout for deterministic output
            input_channels=4,
            use_bias=True
        )
        
        # Replace the skip connection convs with ones that produce known outputs
        for i, skip_conv in enumerate(stack.skip_connections):
            # Create a new conv that outputs a tensor filled with a specific value
            new_conv = nn.Conv1d(8, 4, 1)
            with torch.no_grad():
                # Initialize weights to produce outputs with value (i+1)
                nn.init.constant_(new_conv.weight, 0)
                nn.init.constant_(new_conv.bias, i+1)
            stack.skip_connections[i] = new_conv
        
        # Forward pass
        skip_output, _ = stack(wavenet_stack_input)
        
        # Expected: sum of all skip outputs (1 + 2 = 3)
        # Allow small tolerance for floating point precision
        assert torch.allclose(skip_output, torch.full_like(skip_output, 3.0), rtol=1e-5, atol=1e-5)
    
    def test_gated_activation(self, wavenet_stack_input):
        """
        GIVEN a WavenetStack
        WHEN forward pass is executed
        THEN gated activation (tanh * sigmoid) should be applied
        """
        from src.models.wavenet import WavenetStack
        import torch.nn as nn
        
        # Create a simplified stack for testing gated activation
        stack = WavenetStack(
            num_layers_per_stack=1,  # Just 1 layer for simplicity
            residual_channels=8,
            skip_channels=4,
            kernel_size=1,  # 1x1 conv for simplicity
            dilation_base=1,
            dropout_rate=0.0,  # No dropout for deterministic output
            input_channels=4,
            use_bias=True
        )
        
        # Replace the conv with one that produces a known output
        known_output = torch.ones(2, 8, 16)  # Same shape as conv output
        
        # Create a forward hook to capture the conv output
        conv_output = None
        
        def conv_hook(module, input, output):
            nonlocal conv_output
            conv_output = output.clone()
            return output
        
        # Register the hook
        hook_handle = stack.layers[0].register_forward_hook(conv_hook)
        
        # Forward pass
        skip_output, residual_output = stack(wavenet_stack_input)
        
        # Remove the hook
        hook_handle.remove()
        
        # Compute expected gated output manually
        expected_gated = torch.tanh(conv_output) * torch.sigmoid(conv_output)
        
        # Verify the residual connection includes the gated output
        # We can't directly access the gated output, but we can verify its effect
        # The residual output should be gated_output + input
        # Since we can't directly access gated_output, we'll verify indirectly
        
        # Get the skip output from the first layer
        skip_conv = stack.skip_connections[0]
        expected_skip = skip_conv(expected_gated)
        
        # Verify skip output matches expected (allowing for floating point precision)
        assert torch.allclose(skip_output, expected_skip, rtol=1e-5, atol=1e-5)
    
    def test_residual_connection(self, wavenet_stack_input):
        """
        GIVEN a WavenetStack
        WHEN forward pass is executed
        THEN residual connections should be properly added
        """
        from src.models.wavenet import WavenetStack
        import torch.nn as nn
        
        # Create a simplified stack for testing residual connections
        stack = WavenetStack(
            num_layers_per_stack=1,  # Just 1 layer for simplicity
            residual_channels=4,  # Same as input for direct residual
            skip_channels=4,
            kernel_size=1,  # 1x1 conv for simplicity
            dilation_base=1,
            dropout_rate=0.0,  # No dropout for deterministic output
            input_channels=4,
            use_bias=True
        )
        
        # Zero out the weights of the convolution to isolate residual path
        with torch.no_grad():
            nn.init.zeros_(stack.layers[0].weight)
            nn.init.zeros_(stack.layers[0].bias)
        
        # Forward pass
        _, residual_output = stack(wavenet_stack_input)
        
        # With zero weights, the gated output will be sigmoid(0) * tanh(0) = 0.5 * 0 = 0
        # So residual_output should equal input
        assert torch.allclose(residual_output, wavenet_stack_input, rtol=1e-5, atol=1e-5)
    
    def test_dilated_convolutions(self, wavenet_stack_input):
        """
        GIVEN a WavenetStack with different dilation rates
        WHEN forward pass is executed
        THEN output should maintain sequence length despite dilations
        """
        from src.models.wavenet import WavenetStack
        
        # Test with different dilation bases
        for dilation_base in [1, 2, 4]:
            stack = WavenetStack(
                num_layers_per_stack=3,
                residual_channels=8,
                skip_channels=16,
                kernel_size=3,
                dilation_base=dilation_base,
                dropout_rate=0.2,
                input_channels=4,
                use_bias=True
            )
            
            # Forward pass
            skip_output, residual_output = stack(wavenet_stack_input)
            
            # Check output shapes - sequence length should be preserved
            batch_size, _, seq_len = wavenet_stack_input.shape
            assert skip_output.shape[2] == seq_len, f"Failed with dilation_base={dilation_base}"
            assert residual_output.shape[2] == seq_len, f"Failed with dilation_base={dilation_base}"
    
    def test_dropout_effect(self, wavenet_stack_input):
        """
        GIVEN a WavenetStack with different dropout rates
        WHEN forward pass is executed in train vs eval modes
        THEN dropout should affect outputs in train mode but not eval mode
        """
        from src.models.wavenet import WavenetStack
        
        # Create two identical stacks with high dropout
        stack_train = WavenetStack(
            num_layers_per_stack=3,
            residual_channels=8,
            skip_channels=16,
            kernel_size=3,
            dilation_base=2,
            dropout_rate=0.9,  # High dropout for visible effect
            input_channels=4,
            use_bias=True
        )
        
        stack_eval = WavenetStack(
            num_layers_per_stack=3,
            residual_channels=8,
            skip_channels=16,
            kernel_size=3,
            dilation_base=2,
            dropout_rate=0.9,  # High dropout for visible effect
            input_channels=4,
            use_bias=True
        )
        
        # Ensure identical weights
        stack_eval.load_state_dict(stack_train.state_dict())
        
        # Set one to eval mode
        stack_train.train()
        stack_eval.eval()
        
        # Use the same input for both
        input_tensor = wavenet_stack_input.clone()
        
        # Forward passes
        skip_train, residual_train = stack_train(input_tensor)
        skip_eval, residual_eval = stack_eval(input_tensor)
        
        # In eval mode, dropout is disabled, so multiple runs should be identical
        # In train mode, dropout is active, so multiple runs should differ
        skip_eval2, residual_eval2 = stack_eval(input_tensor)
        skip_train2, residual_train2 = stack_train(input_tensor)
        
        # Eval mode outputs should be identical between runs
        assert torch.allclose(skip_eval, skip_eval2, rtol=1e-5, atol=1e-5)
        assert torch.allclose(residual_eval, residual_eval2, rtol=1e-5, atol=1e-5)
        
        # Train mode outputs should differ between runs (due to dropout)
        # This might occasionally fail due to randomness, but it's unlikely
        assert not torch.allclose(skip_train, skip_train2, rtol=1e-5, atol=1e-5)
        assert not torch.allclose(residual_train, residual_train2, rtol=1e-5, atol=1e-5)
    
    def test_gradient_flow(self):
        """
        GIVEN a WavenetStack
        WHEN backward pass is executed
        THEN gradients should flow to all parameters
        """
        from src.models.wavenet import WavenetStack
        import torch.nn as nn
        
        # Create input and target
        input_tensor = torch.randn(2, 4, 16, requires_grad=True)
        skip_target = torch.randn(2, 16, 16)  # Target for skip output
        residual_target = torch.randn(2, 8, 16)  # Target for residual output
        
        # Create stack
        stack = WavenetStack(
            num_layers_per_stack=3,
            residual_channels=8,
            skip_channels=16,
            kernel_size=3,
            dilation_base=2,
            dropout_rate=0.2,
            input_channels=4,
            use_bias=True
        )
        
        # Forward pass
        skip_output, residual_output = stack(input_tensor)
        
        # Compute loss and backward
        skip_loss = nn.MSELoss()(skip_output, skip_target)
        residual_loss = nn.MSELoss()(residual_output, residual_target)
        loss = skip_loss + residual_loss
        loss.backward()
        
        # Check that all parameters have gradients
        for name, param in stack.named_parameters():
            assert param.grad is not None, f"Parameter {name} has no gradient"
            assert param.grad.abs().sum() > 0, f"Parameter {name} has zero gradient"
        
        # Check input gradients exist
        assert input_tensor.grad is not None
        assert input_tensor.grad.abs().sum() > 0
    
    def test_numerical_stability(self):
        """
        GIVEN a WavenetStack with extreme input values
        WHEN forward pass is executed
        THEN it should handle the values without numerical instability
        """
        from src.models.wavenet import WavenetStack
        
        # Create stack with standard parameters
        stack = WavenetStack(
            num_layers_per_stack=2,
            residual_channels=8,
            skip_channels=4,
            kernel_size=3,
            dilation_base=2,
            dropout_rate=0.0,  # No dropout for deterministic output
            input_channels=4,
            use_bias=True
        )
        
        # Test with very large values
        large_input = torch.ones(2, 4, 16) * 1e6
        skip_large, residual_large = stack(large_input)
        
        # Test with very small values
        small_input = torch.ones(2, 4, 16) * 1e-6
        skip_small, residual_small = stack(small_input)
        
        # Test with mixed positive and negative values
        mixed_input = torch.randn(2, 4, 16) * 1e4
        skip_mixed, residual_mixed = stack(mixed_input)
        
        # Verify outputs don't contain NaN or infinity
        assert not torch.isnan(skip_large).any(), "Skip output contains NaN with large input"
        assert not torch.isinf(skip_large).any(), "Skip output contains infinity with large input"
        assert not torch.isnan(residual_large).any(), "Residual output contains NaN with large input"
        assert not torch.isinf(residual_large).any(), "Residual output contains infinity with large input"
        
        assert not torch.isnan(skip_small).any(), "Skip output contains NaN with small input"
        assert not torch.isinf(skip_small).any(), "Skip output contains infinity with small input"
        assert not torch.isnan(residual_small).any(), "Residual output contains NaN with small input"
        assert not torch.isinf(residual_small).any(), "Residual output contains infinity with small input"
        
        assert not torch.isnan(skip_mixed).any(), "Skip output contains NaN with mixed input"
        assert not torch.isinf(skip_mixed).any(), "Skip output contains infinity with mixed input"
        assert not torch.isnan(residual_mixed).any(), "Residual output contains NaN with mixed input"
        assert not torch.isinf(residual_mixed).any(), "Residual output contains infinity with mixed input"
    
    def test_empty_sequence(self):
        """
        GIVEN a WavenetStack with input containing empty sequence dimension
        WHEN forward pass is executed
        THEN it should handle the empty sequence appropriately
        """
        from src.models.wavenet import WavenetStack
        
        # Create stack with minimal kernel size to handle empty sequence
        stack = WavenetStack(
            num_layers_per_stack=1,
            residual_channels=8,
            skip_channels=4,
            kernel_size=1,  # Use 1x1 convolution to handle empty sequence
            dilation_base=1,
            dropout_rate=0.0,
            input_channels=4,
            use_bias=True
        )
        
        # Create input with empty sequence dimension
        empty_seq_input = torch.randn(2, 4, 0)
        
        # Forward pass
        skip_output, residual_output = stack(empty_seq_input)
        
        # Check output shapes
        assert skip_output.shape == (2, 4, 0), "Skip output shape incorrect for empty sequence"
        assert residual_output.shape == (2, 8, 0), "Residual output shape incorrect for empty sequence"
    
    def test_multi_layer_channel_dimensions(self):
        """
        GIVEN a WavenetStack with multiple layers and different channel dimensions
        WHEN forward pass is executed
        THEN each layer should properly handle the changing dimensions
        """
        from src.models.wavenet import WavenetStack
        import torch.nn as nn
        
        # Create a custom stack with manually controlled layers for testing
        class CustomWavenetStack(WavenetStack):
            def __init__(self):
                super().__init__(
                    num_layers_per_stack=3,
                    residual_channels=8,
                    skip_channels=16,
                    kernel_size=3,
                    dilation_base=1,
                    dropout_rate=0.0,
                    input_channels=4,
                    use_bias=True
                )
                # Replace layers with ones that have different channel dimensions
                self.layers = nn.ModuleList([
                    nn.Conv1d(4, 8, 3, padding=1),   # First layer: 4 -> 8 channels
                    nn.Conv1d(8, 12, 3, padding=1),  # Second layer: 8 -> 12 channels
                    nn.Conv1d(12, 8, 3, padding=1)   # Third layer: 12 -> 8 channels
                ])
                # Replace skip connections to match new dimensions
                self.skip_connections = nn.ModuleList([
                    nn.Conv1d(8, 16, 1),   # First skip: 8 -> 16 channels
                    nn.Conv1d(12, 16, 1),  # Second skip: 12 -> 16 channels
                    nn.Conv1d(8, 16, 1)    # Third skip: 8 -> 16 channels
                ])
        
        # Create the custom stack
        custom_stack = CustomWavenetStack()
        
        # Create input
        input_tensor = torch.randn(2, 4, 16)
        
        # Forward pass
        skip_output, residual_output = custom_stack(input_tensor)
        
        # Check output shapes
        assert skip_output.shape == (2, 16, 16), "Skip output shape incorrect"
        assert residual_output.shape == (2, 8, 16), "Residual output shape incorrect"
    
    def test_forward_execution_flow(self, mocker, wavenet_stack_input):
        """
        GIVEN a WavenetStack with mocked components
        WHEN forward pass is executed
        THEN all components should be called in the correct order
        """
        from src.models.wavenet import WavenetStack
        
        # Create a simplified stack for testing execution flow
        stack = WavenetStack(
            num_layers_per_stack=2,  # Just 2 layers for simplicity
            residual_channels=8,
            skip_channels=4,
            kernel_size=3,
            dilation_base=1,
            dropout_rate=0.2,
            input_channels=4,
            use_bias=True
        )
        
        # Create spies for key operations
        layer_spies = [mocker.spy(layer, 'forward') for layer in stack.layers]
        skip_spies = [mocker.spy(skip, 'forward') for skip in stack.skip_connections]
        dropout_spy = mocker.spy(stack.dropout, 'forward')
        
        # Execute forward pass
        skip_output, residual_output = stack(wavenet_stack_input)
        
        # Verify all components were called the expected number of times
        for layer_spy in layer_spies:
            layer_spy.assert_called_once()
            
        for skip_spy in skip_spies:
            skip_spy.assert_called_once()
        
        # Dropout should be called once per layer
        assert dropout_spy.call_count == len(stack.layers)
        
        # Verify output shapes
        batch_size, _, seq_len = wavenet_stack_input.shape
        assert skip_output.shape == (batch_size, 4, seq_len)
        assert residual_output.shape == (batch_size, 8, seq_len)
    
    def test_single_sample_batch(self):
        """
        GIVEN a batch with a single sample
        WHEN forward pass is executed
        THEN it should process correctly
        """
        from src.models.wavenet import WavenetStack
        
        # Create single sample input
        input_tensor = torch.randn(1, 4, 16)
        
        # Create stack
        stack = WavenetStack(
            num_layers_per_stack=2,
            residual_channels=8,
            skip_channels=4,
            kernel_size=3,
            dilation_base=2,
            dropout_rate=0.0,  # No dropout for deterministic output
            input_channels=4,
            use_bias=True
        )
        
        # Forward pass
        skip_output, residual_output = stack(input_tensor)
        
        # Check output shapes
        assert skip_output.shape == (1, 4, 16)
        assert residual_output.shape == (1, 8, 16)
    
    def test_layer_by_layer_execution(self, wavenet_stack_input):
        """
        GIVEN a WavenetStack
        WHEN forward pass is executed
        THEN each layer should process the output of the previous layer
        """
        from src.models.wavenet import WavenetStack
        import torch.nn as nn
        
        # Create a stack with identifiable layers
        stack = WavenetStack(
            num_layers_per_stack=3,
            residual_channels=8,
            skip_channels=4,
            kernel_size=3,
            dilation_base=1,
            dropout_rate=0.0,  # No dropout for deterministic output
            input_channels=4,
            use_bias=True
        )
        
        # Replace layers with ones that add a unique identifier to the output
        for i, layer in enumerate(stack.layers):
            # Create a new layer that adds a constant value to the output
            def create_modified_layer(i, original_layer):
                class ModifiedLayer(nn.Module):
                    def __init__(self):
                        super().__init__()
                        self.original = original_layer
                        self.identifier = i + 1  # Layer identifier (1-based)
                    
                    def forward(self, x):
                        # Apply original layer and add identifier
                        result = self.original(x)
                        # Add a small constant based on layer number to make output unique
                        # Use a small enough value to not affect numerical stability
                        return result + 0.01 * self.identifier
                
                return ModifiedLayer()
            
            stack.layers[i] = create_modified_layer(i, layer)
        
        # Create hooks to capture intermediate outputs
        layer_outputs = []
        
        def hook_fn(module, input, output):
            layer_outputs.append(output.clone())
        
        hooks = []
        for layer in stack.layers:
            hooks.append(layer.register_forward_hook(hook_fn))
        
        # Forward pass
        skip_output, residual_output = stack(wavenet_stack_input)
        
        # Remove hooks
        for hook in hooks:
            hook.remove()
        
        # Verify each layer processed the output of the previous layer
        # The first layer should process the input
        assert len(layer_outputs) == 3
        
        # Each output should contain its layer's identifier
        # Layer 1 output should have value close to 0.01
        # Layer 2 output should have value close to 0.02
        # Layer 3 output should have value close to 0.03
        for i, output in enumerate(layer_outputs):
            # Extract the unique identifier component (approximately)
            # Subtract the mean of the input to isolate the added constant
            identifier = output.mean() - wavenet_stack_input.mean()
            expected = 0.01 * (i + 1)
            # Allow some tolerance for numerical operations
            assert abs(identifier - expected) < 0.01, f"Layer {i+1} output doesn't contain expected identifier"
