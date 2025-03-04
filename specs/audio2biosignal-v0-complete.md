# Complete audio2biosignal v0 (EDA training)

## High-Level Objective
Complete the initial version of the audio2biosignal project, entailing the training of a seq2seq model to predict EDA from audio.

## Mid-Level Objective
- Implement a TCN architecture and Wavenet architecture as options for the seq2seq model. (Both non-causal)
- Facilitate the use of datasets on S3.
- Implement the necessary methods for the datasets to be used in the EDA training pipeline.
- Complete the training kickoff script.

## Implementation Notes
- Comment everything
- IMPLEMENT EVERYTHING

## Context
### Beginning Context
src/configs.py
src/data/datasets/types.py
src/data/datasets/hku956.py
src/data/datasets/pmemo2019.py
src/data/dataloader.py
src/data/audio_preprocessing.py
src/data/eda_preprocessing.py
src/optimizer.py
scripts/train_audio2eda.py

### Ending Context
src/utilities.py
src/models/tcn.py
src/models/wavenet.py
src/configs.py
src/data/datasets/types.py
src/data/datasets/hku956.py
src/data/datasets/pmemo2019.py
src/data/dataloader.py
src/data/audio_preprocessing.py
src/data/eda_preprocessing.py
src/optimizer.py
scripts/train_audio2eda.py

## Low-Level Tasks
1. Move DatasetTypes enum.
UPDATE src/configs.py:
    MOVE DatasetTypes FROM src/data/datasets/types.py TO top of src/configs.py

DELETE src/data/datasets/types.py

2. Create utilities file with S3 file management subsystem.
CREATE src/utilities.py:
    CREATE S3FileManager class:
        - Implement as a singleton pattern
        - Support downloading files from S3 buckets
        - Implement prefetching mechanism to prepare files ahead of time
        - Use temporary storage to avoid persisting files
        - Include a cache with LRU (Least Recently Used) eviction policy
        - Support concurrent downloads with thread pool
        - Provide both blocking and non-blocking download methods
        - Include methods:
            - download_file(s3_path) -> Path: Blocking download
            - prefetch_files(s3_paths) -> None: Non-blocking batch prefetch
            - get_file(s3_path) -> Optional[Path]: Non-blocking cache check
            - clear_cache() -> None: Remove all cached files
        
    CREATE s3_download() -> Path:
        - Wrapper around S3FileManager.download_file for simple use cases
        
    IMPLEMENTATION DETAILS:
        - Use ThreadPoolExecutor for concurrent downloads
        - Use PriorityQueue for download scheduling
        - Store files in temporary directory that's cleaned up on exit
        - Track in-progress downloads to avoid duplicate requests
        - Support configurable cache size limits
        - Handle S3 path parsing (s3://bucket/key format)

3. Implement HKU956Dataset
   - Integrate with S3FileManager for efficient file loading
   - Implement prefetching in __getitem__ to prepare next batch
   - Use temporary files for audio and EDA data
   - Implement proper cleanup to avoid memory leaks
    UPDATE src/data/datasets/hku956.py:
        UPDATE __init__(dataset_config, audio_feature_config, eda_feature_config, split):
            Store dataset_config, audio_feature_config, eda_feature_config, split in self
            Build self.eda_files: List[Tuple[str, str]] and self.audio_files: Dict[str, str] by:
                Go through `2. original_song_audio.csv` on HKU956 s3 and add every {song_id: audio_s3_path} to self.audio_files
                Parse through HKU956 dataset on s3 and put every (subject_id, eda_s3_path) into self.eda_files for every eda file.
        UPDATE _load_audio_file: MIRROR HKU956Dataset._load_audio_file()
        UPDATE _load_eda_file: MIRROR HKU956Dataset._load_eda_file()
        UPDATE __len__: MIRROR HKU956Dataset.__len__()
        UPDATE __getitem__: MIRROR HKU956Dataset.__getitem__()

4. Implement PMEmoDataset methods
    UPDATE src/data/datasets/pmemo2019.py:
        - Integrate with S3FileManager for efficient file loading
        - Implement prefetching in __getitem__ to prepare next batch
        - Use temporary files for audio and EDA data
        - Implement proper cleanup to avoid memory leaks
        UPDATE PMEmo2019.__init__: 
            Build self.eda_files: List[Tuple[str, str]] and self.audio_files: Dict[str, str] 
            By first retrieving the file structure of the PMEmo2019 dataset on s3 and then parsing through the dataset on s3, then putting (subject_id, s3_file_path) into self.eda_files for every eda file and {song_id: s3_file_path} into self.audio_files for every audio file. Use the information in `metadata.csv` to build the audio_files. The column `musicId` contains the song_id, and the column `filename` then contains the name of the corresponding audio file, which will be in the `chorus` directory.
        UPDATE PMEmo2019._load_audio_file: MIRROR HKU956Dataset._load_audio_file()
        UPDATE PMEmo2019._load_eda_file: MIRROR HKU956Dataset._load_eda_file()
        UPDATE PMEmo2019.__len__: MIRROR HKU956Dataset.__len__()
        UPDATE PMEmo2019.__getitem__: MIRROR HKU956Dataset.__getitem__()

5. DataLoaderBuilder
   - Implement custom DataLoader with prefetching capabilities
   - Support configurable prefetch size
   - Integrate with S3FileManager for batch prefetching
   - Implement custom collate function for padding sequences

6. Instantiate scheduler
UPDATE src/optimizer.py:
    UPDATE OptimizerBuilder.build(optimizer_config, model_params) -> Tuple[Optimizer, Optional[_LRScheduler]]:
        Instantiate scheduler based on optimizer_config.scheduler_config. 
        Use `torch.optim` schedulers.

7. Delete current seq2seq model files.
DELETE src/models/*

8. Update ModelConfig to work with our two architectures: TCN and Wavenet.
UPDATE src/configs.py:
    UPDATE ModelConfig:
        DELETE all existing fields
        CREATE architecture: Literal["tcn", "wavenet"]
        CREATE params: Dict[str, Any]

9.  Create file for TCN (temporal convolutional network) seq2seq model.
CREATE src/models/tcn.py:
    CREATE class TCNBlock(nn.Module):
        CREATE def __init__(self, in_channels, out_channels, kernel_size, dilation, dropout_rate)
                Write a comment explaining the architecture of the block
                Composed of two dilated NON-causal convolutions, a residual connection, a ReLU activation, and a dropout layer
                Initialize all of these components, to then be called in the forward method
            def forward(self, x) -> Tensor:
                x is the input tensor of shape (batch_size, sequence_length, input_size)
                Returns the output tensor of shape (batch_size, sequence_length, output_size)

    CREATE class TCN(nn.Module):
        CREATE def __init__(self, model_config: Dict[str, Any])
                Initialize model architecture using the hyperparameters in model_config
                model_config is a dictionary with the following keys:
                    - input_size: int  # Number of input features
                    - output_size: int  # Number of output features
                    - num_blocks: int  # Number of blocks in the TCN
                    - num_channels: int  # Number of channels in each block
                    - kernel_size: int  # Kernel size for the convolutional layers
                    - dropout: float  # Dropout rate
                Pass the relevant hyperparameters to the TCNBlock class
                Initialize as many TCNBlocks as num_blocks

            def forward(self, x) -> Tensor:
                x is the input tensor of shape (batch_size, sequence_length, input_size)
                Returns the output tensor of shape (batch_size, sequence_length, output_size)

10. Create file for Wavenet seq2seq model.
CREATE src/models/wavenet.py:
    CREATE class WavenetStack(nn.Module):
        CREATE def __init__(self, num_layers_per_stack, residual_channels, skip_channels, kernel_size, dilation_base, dropout_rate, input_channels, output_channels, use_bias)
            Initialize the stack with the hyperparameters passed to this function
            Initialize List of dilated convolution layers, use exponential dilation growth for the dilated convolutions going from dilation=1 to dilation=dilation_base^(num_layers_per_stack-1)

            def forward(self, x) -> Tensor:
                x is the input tensor of shape (batch_size, sequence_length, input_size)
                For each dilated convolution layer:
                    Apply the dilated convolution layer to x
                    apply gated activation (tanh(x) * sigmoid(x))
                    apply residual connection
                    apply skip connection
                    apply dropout
                Sum all of the skip connections
                Apply a ReLU activation
                Apply a 1x1 conv.
                Apply a relu.
                Apply a 1x1 conv.
                Returns the output tensor of shape (batch_size, sequence_length, output_size)

    CREATE class Wavenet(nn.Module):
        CREATE def __init__(self, model_config: Dict[str, Any])
                Initialize model architecture using the hyperparameters in model_config
                model_config is a dictionary with the following keys:
                    - num_stacks: int  # Number of stacks of dilated convs                                                                                                                  
                    - num_layers_per_stack: int  # Layers per stack (with different dilations)                                                                                                        
                    - residual_channels: int  # Channels in residual connections                                                                                                                   
                    - skip_channels: int  # Channels in skip connections                                                                                                                       
                    - kernel_size: int  # Kernel size for dilated convolutions                                                                                                               
                    - dilation_base: int  # Base for exponential dilation growth                                                                                                               
                    - dropout_rate: float  # Dropout probability                                                                                                                                
                    - input_channels: int  # Input feature dimension (e.g., MFCC features)                                                                                                      
                    - output_channels: int  # Output dimension (EDA signal)                                                                                                                      
                    - use_bias: bool  # Whether to use bias in convolutions
                Verify that the model_config is valid
                Pass the relevant hyperparameters to the WavenetStack class
                Initialize as many WavenetStacks as num_stacks

            def forward(self, x) -> Tensor:
                x is the input tensor of shape (batch_size, sequence_length, input_size)
                Returns the output tensor of shape (batch_size, sequence_length, output_size)

11. Modify example configs to work with new model config structure.
UPDATE configs/example1.yml, configs/example2.yml, configs/example3.yml:
    UPDATE model:
        DELETE all fields
        CREATE architecture: "tcn" OR "wavenet"
        CREATE params: Sensible values for TCN or Wavenet that satisfy the requirements of the `TCNBlock` or `WavenetStack` class from `src/models/tcn.py` or `src/models/wavenet.py`.

12.   Update train_audio2eda to instantiate all components.
UPDATE src/train_audio2eda.py:
    UPDATE main():
        UPDATE model instantiation with new config structure.
        CREATE trainer object with huggingface trainer. 
