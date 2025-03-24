
## Configuration
Use pydantic to define the configuration for the upsampler.

class ResidualUpsamplerConfig(BaseModel):
    upsampling_factor: float
    in_channels: int
    hidden_channels: int
    out_channels: int

## Architecture – Residual Upsampling
Use PyTorch to define the architecture for the upsampler.

class ResidualUpsamplerBlock(nn.Module):
    def __init__(self, channels: int, upsampling_factor: float):
        2 1d conv layers with kernel size 5 and stride 1, in_channels = channels, out_channels = channels
        nn.Upsample(scale_factor=upsampling_factor)
        LeakyRelu(0.2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        Pass x through layers in init
        Add x to the output of the layers
        Return the output

class ResidualUpsampler(nn.Module):
    def __init__(self, config: ResidualUpsamplerConfig):
        self.upsampling_factor = config.upsampling_factor
        self.in_channels = config.in_channels
        self.hidden_channels = config.hidden_channels
        self.out_channels = config.out_channels

        Initialize an input projection layer: 1d conv(kernel_size = 9, stride=1, in_channels = in_channels, out_channels = hidden_channels), LeakyRelu(0.2)
        Calculate how many blocks of 4x upsampling gets us as close to upsampling factor as we can get without going over (ie floor(log_4(upsampling_factor))). Initialize that many ResidualUpsamplerBlocks with 4x upsampling.
        Include a final ResidualUpsamplerBlock to upsample by whatever is left over (upsampling_factor / 4^floor(log_4(upsampling_factor))) to get to the desired upsampling factor
        Initialize an output projection layer: 1d conv(kernel_size = 3, stride=1, in_channels = hidden_channels, out_channels = out_channels), LeakyRelu(0.2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        Pass x through the input projection layer
        Pass x through the ResidualUpsamplerBlocks (output of each block is passed in as input to the next block)
        Pass the output of the last ResidualUpsamplerBlock through the final layer
        Pass the output of the final layer through the output projection layer
        Return the final output
