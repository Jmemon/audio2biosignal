import torch
import torch.nn as nn
from torch import Tensor
from typing import Any, Dict

class TCNBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, dilation, dropout_rate):
        super(TCNBlock, self).__init__()
        padding = (kernel_size - 1) * dilation // 2

        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size,
                               padding=padding, dilation=dilation)
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(dropout_rate)

        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size,
                               padding=padding, dilation=dilation)
        self.relu2 = nn.ReLU()
        self.dropout2 = nn.Dropout(dropout_rate)

        self.downsample = nn.Conv1d(in_channels, out_channels, kernel_size=1) \
            if in_channels != out_channels else None

    def forward(self, x: Tensor) -> Tensor:
        out = self.conv1(x)
        out = self.relu1(out)
        out = self.dropout1(out)

        out = self.conv2(out)
        out = self.relu2(out)
        out = self.dropout2(out)

        residual = x if self.downsample is None else self.downsample(x)
        return out + residual

class TCN(nn.Module):
    def __init__(self, model_config: Dict[str, Any]):
        super(TCN, self).__init__()
        input_size = model_config['input_size']
        output_size = model_config['output_size']
        num_blocks = model_config['num_blocks']
        num_channels = model_config['num_channels']
        kernel_size = model_config['kernel_size']
        dropout = model_config['dropout']

        layers = []
        in_channels = input_size
        for i in range(num_blocks):
            out_channels = num_channels
            dilation = 2 ** i
            layers.append(TCNBlock(in_channels, out_channels, kernel_size, dilation, dropout))
            in_channels = out_channels

        self.network = nn.Sequential(*layers)
        self.linear = nn.Linear(num_channels, output_size)

    def forward(self, x: Tensor) -> Tensor:
        x = x.permute(0, 2, 1)
        out = self.network(x)
        out = out.permute(0, 2, 1)
        out = self.linear(out)
        return out
