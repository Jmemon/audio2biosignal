import math
import torch
from torch import nn
from pydantic import BaseModel

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

class ResidualUpsamplerBlock(nn.Module):
    """
    Residual Upsampler Block that performs upsampling with residual connections.
    
    This block applies:
    - Two 1D convolution layers each with kernel size 5, stride 1,
      maintaining the same number of channels.
    - An upsampling layer with the specified scale factor.
    - LeakyReLU activations with negative_slope=0.2.
    
    The input is added to the output after upsampling to form a residual connection.
    """

    def __init__(self, channels: int, upsampling_factor: float):
        """
        Initializes the ResidualUpsamplerBlock.

        Args:
            channels (int): Number of input and output channels.
            upsampling_factor (float): Factor by which to upsample the feature map.
        """
        super().__init__()
        self.conv1 = nn.Conv1d(channels, channels, kernel_size=5, stride=1, padding=2)
        self.conv2 = nn.Conv1d(channels, channels, kernel_size=5, stride=1, padding=2)
        self.upsample = nn.Upsample(scale_factor=upsampling_factor, mode='linear', align_corners=False)
        self.activation = nn.LeakyReLU(negative_slope=0.2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for the ResidualUpsamplerBlock.

        Args:
            x (torch.Tensor): Input tensor of shape (batch, channels, width).

        Returns:
            torch.Tensor: Upsampled tensor with residual connection.
        """
        residual = x
        out = self.activation(self.conv1(x))
        out = self.activation(self.conv2(out))
        out = self.upsample(out)
        # Adjust residual dimensions if necessary
        if out.shape != residual.shape:
            residual = self.upsample(residual)
        return out + residual

class ResidualUpsampler(nn.Module):
    """
    Residual Upsampler architecture for audio signal upsampling.

    This architecture consists of:
    - An initial projection layer transforming the input channels to hidden channels using
      a 1D convolution with kernel size 9 followed by LeakyReLU.
    - A sequence of ResidualUpsamplerBlocks:
        * Multiple blocks with 4x upsampling as many as possible, without exceeding the desired
          upsampling factor.
        * A final block to upsample by the remaining factor.
    - A final projection layer converting hidden channels to output channels using a 1D convolution
      with kernel size 3 followed by LeakyReLU.
    """

    def __init__(self, config: ResidualUpsamplerConfig):
        """
        Initializes the ResidualUpsampler network.

        Args:
            config (ResidualUpsamplerConfig): Configuration parameters for the upsampler.
        """
        super().__init__()
        self.config = config
        upsampling_factor = config.upsampling_factor
        in_channels = config.in_channels
        hidden_channels = config.hidden_channels
        out_channels = config.out_channels

        # Input projection: Convert input to hidden channel space.
        self.input_proj = nn.Sequential(
            nn.Conv1d(in_channels, hidden_channels, kernel_size=9, stride=1, padding=4),
            nn.LeakyReLU(negative_slope=0.2)
        )

        # Calculate number of 4x upsampling blocks.
        num_full_blocks = int(math.floor(math.log(upsampling_factor, 4))) if upsampling_factor >= 4 else 0
        self.blocks = nn.ModuleList()

        # Append full 4x upsampling blocks.
        for _ in range(num_full_blocks):
            self.blocks.append(ResidualUpsamplerBlock(hidden_channels, 4.0))

        # Compute and append block for the remaining upsampling factor.
        remaining_factor = upsampling_factor / (4 ** num_full_blocks) if num_full_blocks > 0 else upsampling_factor
        if remaining_factor != 1.0:
            self.blocks.append(ResidualUpsamplerBlock(hidden_channels, remaining_factor))

        # Final projection layers: Process intermediate features and produce output.
        self.final_layer = nn.Sequential(
            nn.Conv1d(hidden_channels, hidden_channels, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(negative_slope=0.2)
        )
        self.output_proj = nn.Sequential(
            nn.Conv1d(hidden_channels, out_channels, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(negative_slope=0.2)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Executes the forward pass of the ResidualUpsampler.

        Args:
            x (torch.Tensor): Input tensor of shape (batch, in_channels, width).

        Returns:
            torch.Tensor: Upsampled output tensor.
        """
        # Apply input projection.
        x = self.input_proj(x)
        # Pass through each upsampling block.
        for block in self.blocks:
            x = block(x)
        # Process with final layer.
        x = self.final_layer(x)
        # Apply output projection to obtain final output.
        x = self.output_proj(x)
        return x
