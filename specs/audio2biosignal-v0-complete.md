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
- IMPLEMENT EVERYTHING, Do not include any unimplemented methods, no `pass` statements.
- Add all the necessary imports. And add packages you decide to use to `pyproject.toml`.
- UNDER NO CIRCUMSTANCES SHOULD YOU WRITE ENV VARIABLES IN THE CODE.

## Context
### Beginning Context
pyproject.toml
src/**/*.py
configs/**/*.yml
scripts/train_audio2eda.py

### Ending Context
src/**/*.py
pyproject.toml
configs/**/*.yml
scripts/train_audio2eda.py

## Low-Level Tasks
1. Move DatasetTypes enum.
```aider
UPDATE src/configs.py, src/data/datasets/types.py:
    MOVE DatasetTypes FROM src/data/datasets/types.py TO top of src/configs.py
```

```aider
DELETE src/data/datasets/types.py
```

2. Create utilities file with S3 file management subsystem.
```aider
CREATE src/utilities.py:
    CREATE S3FileManager class:
        - Implement as a singleton pattern
        - Support downloading files from S3 buckets
        - Implement prefetching mechanism to prepare files ahead of time
        - Use temporary storage to avoid persisting files
        - Include a cache with LRU (Least Recently Used) eviction policy
        - Support concurrent downloads with thread pool
        - Provide both blocking and non-blocking download methods
        - Include methods: FULLY IMPLEMENT ALL OF THESE METHODS.
            - download_file(s3_path) -> Path: Blocking download
            - prefetch_files(s3_paths) -> None: Non-blocking batch prefetch
            - get_file(s3_path) -> Optional[Path]: Non-blocking cache check
            - clear_cache() -> None: Remove all cached files
        
    IMPLEMENTATION DETAILS:
        - Use ThreadPoolExecutor for concurrent downloads
        - Use PriorityQueue for download scheduling
        - Store files in temporary directory that's cleaned up on exit
        - Track in-progress downloads to avoid duplicate requests
        - Support configurable cache size limits
        - Handle S3 path parsing (s3://bucket/key format)
```

3. Merge AudioFeatureConfig and EDAFeatureConfig into AudioEDAFeatureConfig.
```aider
UPDATE src/configs.py:
    from typing_extensions import Annotated
    from pydantic import AfterValidator

    CREATE AudioEDAFeatureConfig:
        MERGE AudioFeatureConfig and EDAFeatureConfig into AudioEDAFeatureConfig (keeping the fields of both)
        ADD mutual_sample_rate: int = 200

    DELETE AudioFeatureConfig, EDAFeatureConfig

    UPDATE DataConfig:
        DELETE batch_size

    UPDATE ModelConfig:
        DELETE all existing fields
        CREATE field architecture: Literal["tcn", "wavenet"]
        CREATE field params: Dict[str, Any]

    CREATE def ModelConfigValidator(model_config: ModelConfig) -> ModelConfig:
        assert isinstance(model_config.architecture, str)
        assert isinstance(model_config.params, dict)
        if model_config.architecture == "tcn":
            if all(key in model_config.params for key in ["input_size", "output_size", "num_blocks", "num_channels", "kernel_size", "dropout"]):
                return model_config
            else:
                raise ValueError("Invalid TCN model parameters: " + str(model_config.params))
        elif model_config.architecture == "wavenet":
            if all(key in model_config.params for key in ["num_stacks", "num_layers_per_stack", "residual_channels", "skip_channels", "kernel_size", "dilation_base", "dropout_rate", "input_channels", "output_channels", "use_bias"]):
                return model_config
            else:
                raise ValueError("Invalid Wavenet model parameters: " + str(model_config.params))
        
        raise ValueError("Invalid model architecture: " + str(model_config.architecture))

    UPDATE TrainConfig:
        UPDATE model: Annotated[ModelConfig, AfterValidator(ModelConfigValidator)]
        CREATE field batch_size: int = 32
```

4. Change `preprocess_audio` and `preprocess_eda` to work with AudioEDAFeatureConfig.
```aider
UPDATE src/data/audio_preprocessing.py:
    UPDATE preprocess_audio(audio_file_path: Path, feature_config: AudioEDAFeatureConfig) -> torch.Tensor:
        Resample the audio file to the mutual sample rate.
        EVERYTHING ELSE REMAINS THE SAME.

UPDATE src/data/eda_preprocessing.py:
    UPDATE preprocess_eda(eda_file_path: Path, feature_config: AudioEDAFeatureConfig) -> torch.Tensor:
        Resample the eda file to the mutual sample rate.
        EVERYTHING ELSE REMAINS THE SAME.
```

5. Implement HKU956Dataset
```aider
UPDATE src/data/datasets/hku956.py:
    CREATE collate_fn(batch: List[Tuple[torch.Tensor, torch.Tensor]]) -> Tuple[torch.Tensor, torch.Tensor]:
        Find the maximum length of the tensors
        Left pad all tensors to this length with 0s
        The tensors that are in the first position of the tuples are audio tensors, and the tensors that are in the second position of the tuples are eda tensors.
        All the audio tensors should be combined into a single tensor of shape (batch_size, max_length, audio_feature_config.audio_n_mfcc)
        All the eda tensors should be combined into a single tensor of shape (batch_size, max_length)
        Return the tensors as a tuple

    UPDATE class HKU956Dataset:
        Integrate with S3FileManager for efficient file loading
        UPDATE __init__(dataset_config, audio_feature_config, eda_feature_config):
            Store dataset_config, audio_feature_config, eda_feature_config in self
            DELETE split from __init__
            CREATE self._audio_files: Dict[str, str]:
                Go through `s3://audio2biosignal-train-data/HKU956/2. original_song_audio.csv`, which contains two columns: `song_id` and `link`. 
                `song_id` contains ints and `link` contains URLs as strings. 
                Add every row as a key-value pair to self.audio_files as {song_id: link}.
            CREATE self._eda_files: Dict[Tuple[str, str], str]:
                Go through `s3://audio2biosignal-train-data/HKU956/1. physiological_signals/`, which contains a folder for every subject. Each subject is named `hku<id>`.
                Within each subject folder, there is a folder called `EDA`. Within this folder, every csv file is a file containing EDA recordings for one song. The filename is of the format `<num_we_dont_care_about>_<song_id>.csv`.
                An example of one of these paths: `s3://audio2biosignal-train-data/HKU956/1. physiological_signals/hku1904/EDA/0_1240700.csv`.
                Add every row as a key-value pair to self.eda_files as {(subject, song_id): eda_s3_path}. (Extract the song_id from the EDA filename, and the subject from the path).
            CREATE self.examples: List[Dict[Tuple[str, str], Tuple[str, str]]]:
                Create a list of dictionaries, where each dictionary contains a key-value pair of the form {(subject, song_id): (audio_s3_path, eda_s3_path)}.
                Iterate over self._eda_files, retrieve the corresponding audio_s3_path from self._audio_files, and add the pair to self.examples.
        UPDATE _load_audio_file(self, audio_file_link: str) -> torch.Tensor:
            Download the audio file using the mp3 link.
            Preprocess the audio file using `audio_processing.preprocess_audio` and self.audio_feature_config. 
            Return the preprocessed audio file as a torch.Tensor.
        UPDATE _load_eda_file(self, eda_file_path: str) -> torch.Tensor:
            Download the eda file using the s3 path.
            Preprocess the eda file using `eda_processing.preprocess_eda` and self.eda_feature_config, self.audio_feature_config.
            Return the preprocessed eda file as a torch.Tensor.
        UPDATE __len__(self) -> int:
            Return the length of self.examples.
        UPDATE __getitem__(self, index: int) -> Tuple[torch.Tensor, torch.Tensor]:
            Implement prefetching in __getitem__ to prepare next batch
            Use temporary files for audio and EDA data
            Implement proper cleanup to avoid memory leaks
            Return the audio and eda tensors at the given index.
```

6. Implement PMEmoDataset methods
```aider
UPDATE src/data/datasets/pmemo2019.py:
    CREATE collate_fn(batch: List[Tuple[torch.Tensor, torch.Tensor]]) -> Tuple[torch.Tensor, torch.Tensor]:
        MIRROR HKU956Dataset.collate_fn(...)

    UPDATE class PMEmo2019:
        Integrate with S3FileManager for efficient file loading
        UPDATE __init__(self, dataset_config, audio_feature_config, eda_feature_config): 
            STORE dataset_config, audio_feature_config, eda_feature_config in self
            DELETE split from __init__
            CREATE self._audio_files: Dict[str, str]:
                Download the metadata.csv file from `s3://audio2biosignal-train-data/PMEmo2019/metadata.csv`.
                Iterate over the rows of the csv files, focusing on the columns `musicId` and `fileName`.
                For every row, add the pair {musicId: `s3://audio2biosignal-train-data/PMEmo2019/chorus/<fileName>`} to self._audio_files.
            CREATE self._eda_files: Dict[Tuple[str, str], str]:
                All EDA files are located in `s3://audio2biosignal-train-data/PMEmo2019/EDA/`. Each file is named `EDA_<musicId>.csv` and contains EDA data for several subjects for one song. Each column in the csv file is a subject, except for the first column, which is the time. Besides the time columns, the header of each column is a subject id.
                Iterate over every file in `s3://audio2biosignal-train-data/PMEmo2019/EDA/`, extract every subject id from the csv headers, and add the pair {(musicId, subject_id): `s3://audio2biosignal-train-data/PMEmo2019/EDA/EDA_<musicId>.csv`} to self._eda_files.
            CREATE self.examples: List[Dict[Tuple[str, str], Tuple[str, str]]]:
                Create a list of dictionaries, where each dictionary contains a key-value pair of the form {(subject, song_id): (audio_s3_path, eda_s3_path)}.
                Iterate over self._audio_files, then retrieve the corresponding eda_s3_path from self._eda_files, and add the pair to self.examples.
        UPDATE _load_audio_file(self, audio_file_s3_path: str) -> torch.Tensor:
            Download the audio file using the s3 path.
            Preprocess the audio file using `audio_processing.preprocess_audio` and self.audio_feature_config.
            Return the preprocessed audio file as a torch.Tensor.
        UPDATE _load_eda_file(self, eda_file_s3_path: str) -> Dict[int, torch.Tensor]:
            Download the eda file using the s3 path. 
            It will contain a column called `time(s)`, which is the time in seconds.
            Every other column has a subject id as the header, so there will be multiple EDA recordings in this file, each for the same song, but for different subjects.
            Preprocess each EDA recording using `eda_processing.preprocess_eda` and self.eda_feature_config, self.audio_feature_config.
            Return the preprocessed eda file as a dictionary with the subject ids as keys and the preprocessed eda recordings as the values.
        UPDATE __len__(self) -> int:
            MIRROR HKU956Dataset.__len__()
        UPDATE __getitem__(self, index: int) -> Tuple[torch.Tensor, torch.Tensor]:
            MIRROR HKU956Dataset.__getitem__(...)
```

7. DataLoaderBuilder
```aider
UPDATE src/data/dataloader.py:
    UPDATE class DataLoaderBuilder:
        UPDATE staticmethod build(data_config: DataConfig, audio_feature_config: AudioFeatureConfig, eda_feature_config: EDAFeatureConfig, split: str) -> Tuple[List[DataLoader], List[DataLoader], List[DataLoader]]:
            Instantiate datasets by passing the data_config, audio_feature_config, eda_feature_config to the datasets.
            For each dataset CREATE train, val, and test dataloaders by:
                Take the corresponding splits from each dataset (eg first 80% for train, next 10% for val, last 10% for test)
                Use `<dataset_file>.collate_fn` as `collate_fn`.
                Pass this split, collate_fn, and from data_config: batch_size, num_workers, prefetch_size; to torch.utils.data.DataLoader.
            Return (train_dataloaders, val_dataloaders, test_dataloaders)
```

8. Instantiate scheduler
```aider
UPDATE src/optimizer.py:
    UPDATE OptimizerBuilder.build(optimizer_config, model_params) -> Tuple[Optimizer, Optional[_LRScheduler]]:
        Instantiate scheduler based on optimizer_config.scheduler_config. 
        Use `torch.optim` schedulers.
```

9. Delete current seq2seq model files.
```aider
DELETE src/models/*
```

10.   Create file for TCN (temporal convolutional network) seq2seq model.
```aider
CREATE src/models/tcn.py:
    CREATE class TCNBlock(nn.Module):
        CREATE def __init__(self, in_channels, out_channels, kernel_size, dilation, dropout_rate)
                Write a comment explaining the architecture of the block
                The architecture is composed of two dilated NON-causal convolutions, a residual connection, a ReLU activation, and a dropout layer
                Initialize all of these components, to then be called in the forward method
            def forward(self, x) -> Tensor:
                x is the input tensor of shape (batch_size, sequence_length, input_size)
                Call the sequence of modules initialized in __init__
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
                Call the sequence of modules initialized in __init__
                Returns the output tensor of shape (batch_size, sequence_length, output_size)
```

11.   Create file for Wavenet seq2seq model.
```aider
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
```

12.  Modify example configs to work with new model config structure.
```aider
UPDATE configs/example1.yml, configs/example2.yml, configs/example3.yml:
    UPDATE model:
        DELETE all fields
        CREATE architecture: str: Randomly choose "tcn" OR "wavenet"
        CREATE params: Dict[str, Any]: You choose sensible values for TCN or Wavenet (depending on the architecture you chose) that satisfy the requirements of the `TCNBlock` and `WavenetStack` classes from `src/models/tcn.py:TCN.__init__` and `src/models/wavenet.py:Wavenet.__init__`, respectively.

        UPDATE all fields corresponding to changes in low-level task 3.
```

13.  Update train_audio2eda to instantiate all components.
```aider
UPDATE src/train_audio2eda.py:
    UPDATE main():
        Load the .env file to get the wandb api key and the aws credentials.
        Use argparse to get retrieve the config file path.
        Use yaml to load the config file.
        Create the wandb logger using the values in the logging config in the config. Indicate if we need to pass any values for this step.
        Instantiate the model, optimizer, scheduler, loss, and train/val/test dataloaders based on the values in the config file.
        Create the checkpoint directory using the values in the checkpoint config in the config.
        Pass all of these components to the huggingface trainer. USE THE TRAINER FROM THE HUGGINFACE LIBRARY.
        Kick off the loop.
```
