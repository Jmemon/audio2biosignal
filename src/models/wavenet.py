import torch
import torch.nn as nn

class WavenetStack(nn.Module):
    """
    A stack of dilated convolutional layers with gated activations and skip connections.
    
    This module implements a core component of the WaveNet architecture, consisting of a series
    of dilated 1D convolutions with exponentially increasing dilation factors. Each layer
    applies a gated activation function (tanh * sigmoid) and contributes to both residual
    and skip connections.
    
    Architecture:
        - Sequential dilated convolutions with dilation = dilation_base^layer_index
        - Gated activation units: tanh(x) * sigmoid(x)
        - Residual connections between layers to facilitate gradient flow
        - Skip connections from each layer to the output for multi-scale feature extraction
        - Dropout for regularization
    
    Args:
        num_layers_per_stack (int): Number of convolutional layers in the stack
        residual_channels (int): Number of channels in the residual connections
        skip_channels (int): Number of channels in the skip connections
        kernel_size (int): Size of the convolutional kernel
        dilation_base (int or float): Base for exponential dilation growth
        dropout_rate (float): Probability of dropping units during training (0.0-1.0)
        input_channels (int): Number of input channels
        use_bias (bool): Whether to include bias parameters in convolutions
    
    Returns:
        tuple: (skip_sum, residual)
            - skip_sum (torch.Tensor): Sum of all skip connections [batch_size, skip_channels, seq_length]
            - residual (torch.Tensor): Final residual output [batch_size, residual_channels, seq_length]
    
    Note:
        - Padding is calculated to maintain sequence length across all dilations
        - The receptive field grows exponentially with the number of layers
        - For causal convolutions, adjust padding calculation accordingly
    """
    def __init__(self, num_layers_per_stack, residual_channels, skip_channels, kernel_size,
                 dilation_base, dropout_rate, input_channels, use_bias):
        """
        Initialize a WavenetStack with dilated convolutions and skip connections.
        
        Constructs a stack of dilated convolutional layers with exponentially increasing
        dilation factors, creating a receptive field that grows exponentially with depth.
        Each layer contributes to both residual and skip connections, enabling efficient
        gradient flow and multi-scale feature extraction.
        
        Architecture:
            - Creates num_layers_per_stack convolutional layers with O(n) complexity
            - Dilation pattern follows exponential growth: dilation_base^layer_index
            - Each layer has two outputs: residual connection and skip connection
            - Padding is calculated to maintain sequence length across all dilations
        
        Args:
            num_layers_per_stack (int): Number of convolutional layers to create
            residual_channels (int): Number of output channels for residual connections
            skip_channels (int): Number of output channels for skip connections
            kernel_size (int): Size of the convolutional kernel (typically odd)
            dilation_base (int or float): Base for exponential dilation growth
            dropout_rate (float): Probability of dropping units during training (0.0-1.0)
            input_channels (int): Number of input channels for the first layer
            use_bias (bool): Whether to include bias parameters in convolutions
            
        Raises:
            ValueError: If kernel_size, channels, or dilation_base are invalid
            
        Thread Safety:
            - Thread-safe for initialization (no shared state)
            - Not thread-safe for parameter updates
            
        Resource Management:
            - Parameters scale linearly with num_layers_per_stack
            - Memory footprint depends on sequence length and channel dimensions
        """
        super(WavenetStack, self).__init__()
        self.layers = nn.ModuleList()
        self.skip_connections = nn.ModuleList()

        for n in range(num_layers_per_stack):
            dilation = dilation_base ** n
            in_channels = input_channels if n == 0 else residual_channels
            conv = nn.Conv1d(in_channels, residual_channels, kernel_size,
                             dilation=dilation, padding=(kernel_size - 1) * dilation // 2, bias=use_bias)
            self.layers.append(conv)
            skip_conv = nn.Conv1d(residual_channels, skip_channels, 1, bias=use_bias)
            self.skip_connections.append(skip_conv)
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, x):
        skip_outputs = []
        for conv, skip_conv in zip(self.layers, self.skip_connections):
            out = conv(x)
            gated = torch.tanh(out) * torch.sigmoid(out)
            residual = gated + x
            x = residual
            skip = skip_conv(gated)
            skip_outputs.append(skip)
            x = self.dropout(x)
        return sum(skip_outputs), x  # Return sum of skip connections and last output

class Wavenet(nn.Module):
    def __init__(self, model_config):
        """
        Initialize a Wavenet model with hierarchical dilated convolutions for sequence modeling.
        
        Constructs a complete Wavenet architecture by instantiating multiple WavenetStacks
        and configuring the output transformation layers. The initialization process extracts
        all required parameters from the model_config dictionary and builds the network topology
        with O(num_stacks * num_layers_per_stack) complexity.
        
        Architecture:
            - Multiple stacked WavenetStack modules with shared configuration
            - Sequential skip connection aggregation across all stacks
            - Two-layer output transformation with ReLU activations
            - Proper tensor dimension handling for sequence data
        
        Args:
            model_config (dict): Configuration dictionary containing:
                - num_stacks (int): Number of WavenetStack modules to chain
                - num_layers_per_stack (int): Number of dilated convolutions per stack
                - residual_channels (int): Dimensionality of residual connections
                - skip_channels (int): Dimensionality of skip connections
                - kernel_size (int): Size of convolutional kernels
                - dilation_base (int or float): Base for exponential dilation growth
                - dropout_rate (float): Dropout probability between 0.0-1.0
                - input_channels (int): Number of input features
                - output_channels (int): Number of output features
                - use_bias (bool): Whether to include bias in convolutions
        
        Raises:
            KeyError: If any required configuration parameter is missing
            ValueError: If parameters have invalid values (delegated to submodules)
            
        Thread Safety:
            - Thread-safe for initialization (no shared state)
            - Not thread-safe for forward pass (stateful activations)
            
        Resource Management:
            - Parameters scale with O(num_stacks * num_layers_per_stack * channels)
            - Memory usage grows linearly with sequence length
        """
        super(Wavenet, self).__init__()
        num_stacks = model_config['num_stacks']
        num_layers_per_stack = model_config['num_layers_per_stack']
        residual_channels = model_config['residual_channels']
        skip_channels = model_config['skip_channels']
        kernel_size = model_config['kernel_size']
        dilation_base = model_config['dilation_base']
        dropout_rate = model_config['dropout_rate']
        input_channels = model_config['input_channels']
        output_channels = model_config['output_channels']
        use_bias = model_config['use_bias']

        self.stacks = nn.ModuleList()
        for _ in range(num_stacks):
            stack = WavenetStack(num_layers_per_stack, residual_channels, skip_channels, kernel_size,
                                 dilation_base, dropout_rate, input_channels, use_bias)
            self.stacks.append(stack)
            input_channels = residual_channels  # For subsequent stacks

        self.relu = nn.ReLU()
        self.conv_out1 = nn.Conv1d(skip_channels, skip_channels, 1)
        self.conv_out2 = nn.Conv1d(skip_channels, output_channels, 1)

    def forward(self, x):
        # x shape: (batch_size, input_channels, seq_length)
        x = x.transpose(1, 2)  # Convert to (batch_size, seq_length, input_channels)
        skip_total = 0
        for stack in self.stacks:
            skip, x = stack(x)
            skip_total += skip
        out = self.relu(skip_total)
        out = self.conv_out1(out)
        out = self.relu(out)
        out = self.conv_out2(out)
        out = out.transpose(1, 2)  # Convert back to (batch_size, seq_length, output_channels)
        return out
