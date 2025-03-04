import torch
import torch.nn as nn

class TCNBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, dilation, dropout_rate):
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
    def __init__(self, model_config):
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
