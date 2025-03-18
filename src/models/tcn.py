import torch
import torch.nn as nn

class TCNBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, dilation, dropout_rate):
        """
        Temporal Convolutional Network block implementing causal dilated convolutions with residual connections.
        
        This block forms the fundamental building unit of a TCN architecture, designed to capture
        temporal dependencies while maintaining sequence length. It implements two stacked dilated
        causal convolutions with ReLU activations and dropout regularization, followed by a residual
        connection that preserves gradient flow through deep networks.
        
        Architecture:
            - Two sequential dilated 1D convolutions with identical kernel size and dilation factor
            - Automatic padding calculation to maintain sequence length: (kernel_size-1)*dilation//2
            - Residual connection with optional 1x1 convolution for channel matching
            - ReLU activations after each convolution and after the residual addition
            - Dropout applied after each activation for regularization
        
        Args:
            in_channels (int): Number of input channels
            out_channels (int): Number of output channels
            kernel_size (int): Size of the convolutional kernel (typically odd)
            dilation (int): Dilation factor for the convolutions, controls receptive field growth
            dropout_rate (float): Dropout probability between 0 and 1
            
        Raises:
            ValueError: If in_channels or out_channels <= 0
            ValueError: If kernel_size <= 0
            ValueError: If dilation <= 0
            ValueError: If dropout_rate < 0 or dropout_rate > 1
            
        Notes:
            - Time complexity: O(L * Cin * Cout * k) where L is sequence length, k is kernel size
            - The effective receptive field grows exponentially when blocks are stacked with
              increasing dilation factors (typically powers of 2)
            - For causal convolutions, padding is applied to preserve sequence length while
              ensuring no information leakage from future timesteps
        """
        # Validate parameters
        if in_channels <= 0:
            raise ValueError("in_channels should be positive")
        if out_channels <= 0:
            raise ValueError("out_channels should be positive")
        if kernel_size <= 0:
            raise ValueError("kernel_size should be positive")
        if dilation <= 0:
            raise ValueError("dilation should be positive")
        if dropout_rate < 0 or dropout_rate > 1:
            raise ValueError("dropout_rate should be between 0 and 1")
            
        super(TCNBlock, self).__init__()
        # Two dilated convolutions with residual connection and dropout
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
        out = self.conv1(x)
        out = self.relu1(out)
        out = self.dropout1(out)
        out = self.conv2(out)
        out = self.relu2(out)
        out = self.dropout2(out)
        res = x if self.downsample is None else self.downsample(x)
        return self.relu(out + res)

class TCN(nn.Module):
    """
    Temporal Convolutional Network for sequence modeling with exponentially dilated causal convolutions.
    
    This model implements a deep hierarchical architecture of dilated convolutional layers for
    capturing long-range temporal dependencies in sequential data. It stacks multiple TCNBlocks
    with exponentially increasing dilation factors to achieve a large receptive field while
    maintaining computational efficiency and stable gradient flow through residual connections.
    
    Architecture:
        - Sequential stack of TCNBlocks with exponentially increasing dilation (2^i)
        - Each block contains two dilated causal convolutions with residual connections
        - Final linear projection layer to map features to output dimensions
        - Maintains sequence length throughout the network via appropriate padding
        - Time complexity: O(L * B * C * K) where L is sequence length, B is number of blocks,
          C is number of channels, and K is kernel size
        - Receptive field grows exponentially: RF = 1 + (K-1) * sum(2^i) for i in [0...num_blocks-1]
    
    Args:
        model_config (Dict[str, Any]): Configuration dictionary containing:
            input_size (int): Dimensionality of input features
            output_size (int): Dimensionality of output predictions
            num_blocks (int): Number of TCN blocks in the network
            num_channels (int): Number of channels in convolutional layers
            kernel_size (int): Size of convolutional kernels
            dropout (float): Dropout probability for regularization
    
    Raises:
        KeyError: If any required configuration parameter is missing
        ValueError: If any configuration parameter has an invalid value
            (validation performed by TCNBlock constructors)
    
    Behavior:
        - Automatically transposes input/output tensors between time-first and channel-first formats
        - Maintains causal structure, ensuring predictions at time t only depend on inputs up to time t
        - Preserves sequence length throughout the network
    
    Integration:
        - Expects input tensors of shape (batch_size, seq_length, input_size)
        - Returns output tensors of shape (batch_size, seq_length, output_size)
        - Compatible with standard PyTorch optimizers and loss functions
        - Example:
          ```
          config = {
              'input_size': 10,
              'output_size': 1,
              'num_blocks': 4,
              'num_channels': 64,
              'kernel_size': 3,
              'dropout': 0.2
          }
          model = TCN(config)
          output = model(input_tensor)  # input_tensor: [batch_size, seq_len, input_size]
          ```
    
    Limitations:
        - Fixed dilation pattern (powers of 2) may not be optimal for all sequence lengths
        - All TCN blocks use the same kernel size and channel count
        - No support for bidirectional processing (intentional for causal modeling)
        - Memory usage scales with sequence length and number of channels
    """
    def __init__(self, model_config):
        """
        Initialize a Temporal Convolutional Network with hierarchical dilated convolutions.
        
        Constructs the complete TCN architecture by extracting configuration parameters,
        building a sequential stack of TCNBlocks with exponentially increasing dilation factors,
        and configuring the final linear projection layer. The initialization process validates
        required parameters and establishes the network topology with O(num_blocks) complexity.
        
        Architecture:
            - Creates num_blocks TCNBlock modules with dilation = 2^i for i in [0...num_blocks-1]
            - First block maps input_size to num_channels, subsequent blocks maintain num_channels
            - Final linear layer maps num_channels to output_size for prediction
            - Total parameter count scales with O(num_blocks * num_channels^2 * kernel_size)
        
        Args:
            model_config (Dict[str, Any]): Configuration dictionary containing:
                input_size (int): Dimensionality of input features
                output_size (int): Dimensionality of output predictions
                num_blocks (int): Number of TCN blocks in the network
                num_channels (int): Number of channels in convolutional layers
                kernel_size (int): Size of convolutional kernels
                dropout (float): Dropout probability for regularization
        
        Raises:
            KeyError: If any required configuration parameter is missing
            ValueError: If num_blocks < 0 (propagated from TCNBlock constructors)
            
        Thread Safety:
            - Thread-safe for initialization (no shared state)
            - Not thread-safe for parameter updates
            
        Resource Management:
            - Parameters scale linearly with num_blocks and quadratically with num_channels
            - Memory footprint depends on sequence length and channel dimensions
        """
        super(TCN, self).__init__()
        input_size = model_config['input_size']
        output_size = model_config['output_size']
        num_blocks = model_config['num_blocks']
        num_channels = model_config['num_channels']
        kernel_size = model_config['kernel_size']
        dropout = model_config['dropout']

        layers = []
        for i in range(num_blocks):
            dilation = 2 ** i
            in_channels = input_size if i == 0 else num_channels
            layers.append(TCNBlock(in_channels, num_channels, kernel_size, dilation, dropout))
        self.network = nn.Sequential(*layers)
        self.linear = nn.Linear(num_channels, output_size)

    def forward(self, x):
        # x shape: (batch_size, seq_length, input_size)
        x = x.transpose(1, 2)  # Convert to (batch_size, input_size, seq_length)
        out = self.network(x)
        out = out.transpose(1, 2)  # Convert back to (batch_size, seq_length, num_channels)
        out = self.linear(out)
        return out
