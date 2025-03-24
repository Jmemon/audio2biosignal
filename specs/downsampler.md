
## Configuration
Use pydantic to define the configuration for the upsampler.

class DownsamplerConfig(BaseModel):
    downsampling_factor: float

## Architecture 
Use PyTorch to define the architecture for the downsampler.

class Downsampler(nn.Module):
