# Autoregressive Multimodal Wavenet for generating synthetic EDA signals conditioned on audio.

## Configuration



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
- context_size_in_seconds: float
- num_mfccs: int
- audio_sample_rate: int
- eda_sample_rate: int
- mfcc_window_size: int
- mfcc_hop_size: int
- in_channels: int
- hidden_channels: int
- out_channels: int
- ???

Initialization:
    What needs to be computed from the passed-in hyperparameters?
    - spectral_frames_per_eda_step
    - audio_time_steps_per_spectral_frame
    - layers_per_stack
    - num_stacks
    - upsampling_factor
    - downsampling_factor



So the way I'm thinking right now:
Audio comes in as (batch_size, in_channels, spectral_frames, n_mfccs)
EDA comes in as (batch_size, in_channels, spectral_frames, 1)  
# we are expecting the upsampler to add the channel dimension as well as the additional time steps, add an assert to ensure this.

Clarification on what 1-to-1 Conv means:
    Our tensors will have shape (batch_size, channels, steps, num_features).
    1-to-1 means steps and num_features should be the same before and after the conv.
    However the number of channels can change. This is where feature processing happens.
    So this can be conv1d, conv2d, conv3d, whatever is needed to get the desired output shape.

Dilated Convolution(i):
    Input: (input_signal,)
    Output: conv(
        input_signal, 
        in_channels=in_channels if i == 0 else hidden_channels, 
        out_channels=hidden_channels, 
        dilation=2^i, 
        kernel_size=2
    )

class Gate(nn.Module):
    # in the future we can add arbitrary signals by adding 1-to-1 convs processing for each one and adding them to the filter and gate.
    Input: (input_signal; BxC_inXTxnum_mfccs, upsampled_generated_eda; BxC_inxTx1)
    Output: element-wise-multiplication(
        tanh(
            add(
                1-to-1 conv(input_signal; BxC_hxTxnum_mfccs -> BxC_hXtxnum_mfccs), 
                1-to-1 conv(upsampled_generated_eda; BxC_inxTx1 -> BxC_inxTx1)
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
        Dilated convolution(dilation=2^i, kernel_size=2) (input_signal; BxC_in(if i == 0 else C_h)XTxnum_mfccs) -> (intermediate_signal; BxC_hXTxnum_mfccs)
        Gating mechanism (intermediate_signal; BxC_hXTxnum_mfccs, upsampled_generated_eda; BxC_inxTx1) -> (gated_signal; BxC_hXTxnum_mfccs)
        1-to-1 conv (gated_signal) -> (skip_signal)
        Store skip_signal
        Residual (input_signal, skip_signal) -> (output_signal)
        This output_signal is then passed to the next layer in the stack as the input_signal.

class Wavenet(nn.Module):
    Sequence of WavenetStacks (MFCCs, upsampled_generated_eda) -> (skip_signal_1, skip_signal_2, ...)
    Sum the skip signals (skip_signal_1, skip_signal_2, ...) -> (skip_signal_sum)
    ReLU activation (skip_signal_sum) -> (activation_signal_1)
    1-to-1 conv (activation_signal_1) -> (feature_signal_1)
    ReLU activation (feature_signal_1) -> (activation_signal_2)
    1-to-1 conv (activation_signal_2) -> (feature_signal_2)
    Take final spectral_frames_per_eda_time_step spectral frames from feature_signal_2
    apply (???) to return spectral_frames_per_eda_time_step probability distributions each one corresponding to one high-res-eda-step.
