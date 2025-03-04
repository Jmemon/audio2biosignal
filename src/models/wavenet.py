import torch
import torch.nn as nn

class WavenetStack(nn.Module):
    def __init__(self, num_layers_per_stack, residual_channels, skip_channels, kernel_size,
                 dilation_base, dropout_rate, input_channels, use_bias):
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
