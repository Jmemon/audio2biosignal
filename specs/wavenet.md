# Autoregressive Multimodal Wavenet for generating synthetic EDA signals from audio.

## High-Level Objective
Add a wavenet architecture to the project, including the model file, a config class and validation function, tests, and example configs.

## Mid-Level Objectives
- Create `src/models/wavenet.py` containing the model architecture.
- Create `pydantic WavenetConfig(BaseModel)` in `src/configs.py` where we can specify the architectural hyperparameters, and containing a function to validate the hyperparameters.
- Create `tests/unit/models/test_wavenet.py` containing unit tests for the model.
- Add to `tests/unit/test_configs.py` a test for the `WavenetConfig` class and validation function.
- Create some example configs in `configs/` to train an audio2EDA model using the Wavenet architecture.

## Implementation Notes
Use `pytorch` for the architecture implementation.
Use `pydantic` for the config class and validation function.
Use `adw/create_tests.py` to create the tests.
Use `adw/create_config.py` to create each example config.

## Context
### Beginning Context
src/configs.py
tests/unit/test_configs.py

### Ending Context
src/configs.py
tests/unit/test_configs.py
src/models/wavenet.py


## Architecture
Entire model (not just this module) will be passed MFCCs extracted from an audio with sample rate M, and the predicted EDA signal with sample rate N.




We need two ratios to parameterize this architecture: 
- audio_time_steps_per_spectral_frame = audio_window_size
- spectral_frames_per_eda_time_step = audio_sample_rate / (eda_sample_rate * audio_window_size)




It will be predicting an EDA signal, whose expected output has sample rate N.




MFCCs shape: (batch_size, channels, spectral_frames, n_mfccs)
Input EDA shape: (batch_size, ??)  
        ?? = number of eda steps I want to auto-regress over, which is related to receptive field size. 
        K input_eda_steps = K * spectral_frames_per_eda_step spectral frames for receptive field

Output EDA shape: (batch_size, output_size)




To get num_layers_per_stack and num_stacks, decide how many steps you want to use to predict each eda_step (in eda_step time units).
So then we need the value spectral_frames_per_eda_time_step * prediction_steps to determine the desired receptive field (cap out per stack receptive_field at 512 or something, then k stacks has receptive field 512^k).
Use a number of layers per stack and number of stacks that gets us desired_receptive_field / 2. Because the downsampling layer will combine a group of transformed spectral frames in a way that gets us the full desired receptive field.
The initial output_signal will be aligned with the spectral_frames. We will then use a learned down-sampling layer to combine each group of transformed spectral frames corresponding to one eda_step to convert them. 
Each group of transformd spectral frames will have information from the prior spectral_frames_per_eda_step * prediction_steps spectral frames.




Model Inputs:
- MFCCs: (batch_size, channels, spectral_frames, n_mfcc)
- EDA: (batch_size, time_steps)




Architecture Hyperparameters: 
- context_size_in_eda_steps
- num_mfccs
- audio_sample_rate
- eda_sample_rate
- mfcc_window_size
- mfcc_hop_size
- in_channels
- hidden_channels
- out_channels
- ???

Initialization:
    What needs to be computed from the passed-in hyperparameters?
    - spectral_frames_per_eda_step
    - audio_time_steps_per_spectral_frame
    - layers_per_stack
    - num_stacks
    - upsampling_factor
    - downsampling_factor

1-to-1 Conv:
    Hyperparameters: channels, num_features (for eda these are (1,1))
    For audio: kernels should have size (channels, num_mfccs) 
    For EDA: kernels should have size (1, 1) (since there's one channel and one feature per timestep)
    They should slide step-by-step along time/frame dimension.

class Downsampler(nn.Module):
    Input: (high_res_generated_pre_eda,)
    Output: Attention layer with context = spectral_frames_per_eda_time_step

class Upsampler(nn.Module):
    Hyperparameters: upsampling_factor
    Input: (eda_signal,)
    Output: 
        1d conv with kernel_size=upsampling_factor // ??? and stride = kernel_size // 3
        Linear (increasing dim) -> ReLU alternating until sequence length has increased by upsampling_factor

Dilated Convolution(i):
    Input: (input_signal,)
    Output: conv(
        input_signal, 
        in_channels=in_channels if i == 0 else hidden_channels, 
        out_channels=hidden_channels if i < layers_per_stack - 1 else out_channels, 
        dilation=2^i, 
        kernel_size=2
    )

class Gate(nn.Module):
    Input: (input_signal, upsampled_generated_eda)
    Output: element-wise-multiplication(
        tanh(
            add(
                1-to-1 conv(input_signal), 
                1-to-1 conv(upsampled_generated_eda)
            )
        ),
        sigmoid(
            add(
                1-to-1 conv(input_signal), 
                1-to-1 conv(upsampled_generated_eda)
            )
        )
    )

class WavenetStack(nn.Module):
    Each WavenetStack contains a sequence of layers (length L) containing the following. 
    For i in (0, 1, ..., L-1):
        Dilated convolution(dilation=2^i, kernel_size=2) (input_signal (transformed mfccs)) -> (intermediate_signal)
        Gating mechanism (intermediate_signal, upsampled_generated_eda) -> (gated_signal)
        1x1 conv (gated_signal) -> (skip_signal)
        Store skip_signal
        Residual (input_signal, skip_signal) -> (output_signal)
        This output_signal is then passed to the next layer in the stack as the input_signal.

class Wavenet(nn.Module):
    Upsample [learned upsampling layer] (generated_eda) -> (upsampled_generated_eda)
    Sequence of WavenetStacks (MFCCs, upsampled_generated_eda) -> (skip_signal_1, skip_signal_2, ...)
    Sum the skip signals (skip_signal_1, skip_signal_2, ...) -> (skip_signal_sum)
    1-to-1 conv (skip_signal_sum) -> (intermediate_signal_1)
    ReLU activation (intermediate_signal_1) -> (activation_signal_1)
    1-to-1 conv (activation_signal_1) -> (intermediate_signal_2)
    ReLU activation (intermediate_signal_2) -> (activation_signal_2)
    1-to-1 conv (activation_signal_2) -> (high_res_pre_eda)
    Downsampling layer, an attention layer with context = spectral_frames_per_eda_time_step. (high_res_pre_eda) -> (pre_eda)
    Softmax activation (pre_eda) -> (generated_eda)



## Low-Level Tasks
1. Create `src/models/wavenet.py` containing the model architecture.
```aider
CREATE src/models/wavenet.py:
    Create the wavenet model architecture as described in the architecture section.
```
2. Add `pydantic WavenetConfig(BaseModel)` to `src/configs.py` containing the architectural hyperparameters and a function to validate these hyperparameters.
```aider
UPDATE src/configs.py:
    CREATE pydantic WavenetConfig(BaseModel):
        context_size_in_seconds: int
        in_channels: int
        hidden_channels: int
        out_channels: int = 1
```
3. Create a shell command calling `adw/create_tests.py` to create `tests/unit/models/test_wavenet.py` containing unit tests for the model.
```aider
Write a shell command `python adw/create_tests.py src/models/wavenet.py
```
4. Create a shell command calling `adw/create_tests.py` to add tests for the `WavenetConfig` class and its validation function to `tests/unit/test_configs.py`.
5. Create shell commands calling `adw/create_config.py` to create some example configs in `configs/` to train an audio2EDA model using the Wavenet architecture.