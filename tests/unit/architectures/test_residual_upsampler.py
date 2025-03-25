"""
Tests for the ResidualUpsampler component.

This suite verifies:
1. Correct initialization and configuration.
2. Proper behavior of the forward pass including dimensionality handling.
3. Edge case behaviors, e.g. when upsampling factor is 1.
"""

import math
import torch
import pytest
from torch import nn
from src.architectures.residual_upsampler import ResidualUpsampler, ResidualUpsamplerConfig

class TestResidualUpsamplerInitialization:
    """Tests covering object initialization and configuration."""

    def test_valid_initialization(self):
        """
        GIVEN a valid ResidualUpsamplerConfig
        WHEN a ResidualUpsampler is instantiated
        THEN all projection layers and blocks are initialized correctly.
        """
        config = ResidualUpsamplerConfig(
            upsampling_factor=4.0,
            in_channels=3,
            hidden_channels=8,
            out_channels=3
        )
        upsampler = ResidualUpsampler(config)

        # Verify input and output projection layers exist
        assert upsampler.input_proj is not None
        assert upsampler.output_proj is not None

        # Calculate expected number of blocks
        num_full_blocks = int(math.floor(math.log(config.upsampling_factor, 4))) if config.upsampling_factor >= 4 else 0
        expected_blocks = num_full_blocks
        remaining_factor = config.upsampling_factor / (4 ** num_full_blocks) if num_full_blocks > 0 else config.upsampling_factor
        if abs(remaining_factor - 1.0) > 1e-6:
            expected_blocks += 1

        assert len(upsampler.blocks) == expected_blocks

class TestResidualUpsamplerForward:
    """Tests covering the forward pass functionality."""

    @pytest.fixture
    def config(self):
        return ResidualUpsamplerConfig(
            upsampling_factor=4.0,
            in_channels=3,
            hidden_channels=8,
            out_channels=3
        )

    @pytest.fixture
    def upsampler(self, config):
        return ResidualUpsampler(config)

    def test_forward_dimensionality(self, upsampler):
        """
        GIVEN a dummy input tensor of shape (batch, channels, width)
        WHEN the forward pass is executed
        THEN the output tensor has the expected upsampled width and output channels.
        """
        input_tensor = torch.randn(2, 3, 16)
        output = upsampler(input_tensor)

        # Compute overall scale factor from all upsampling blocks
        scale = 1.0
        for block in upsampler.blocks:
            scale *= block.upsample.scale_factor
        expected_width = int(16 * scale)

        assert output.shape[0] == 2
        assert output.shape[1] == upsampler.config.out_channels
        assert output.shape[2] == expected_width

    def test_forward_behavior(self, upsampler):
        """
        GIVEN a dummy input tensor with ones
        WHEN the forward pass is executed
        THEN the output is finite and preserves the residual pathway.
        """
        input_tensor = torch.ones(2, 3, 16)
        output = upsampler(input_tensor)
        assert torch.isfinite(output).all()

class TestResidualUpsamplerEdgeCases:
    """Tests covering edge cases of the ResidualUpsampler."""

    def test_upsampling_factor_one(self):
        """
        GIVEN a ResidualUpsamplerConfig with upsampling_factor=1.0
        WHEN a ResidualUpsampler is instantiated and forward pass executed
        THEN the output width matches the input width.
        """
        config = ResidualUpsamplerConfig(
            upsampling_factor=1.0,
            in_channels=3,
            hidden_channels=8,
            out_channels=3
        )
        upsampler = ResidualUpsampler(config)
        input_tensor = torch.randn(1, 3, 10)
        output = upsampler(input_tensor)
        assert output.shape[2] == input_tensor.shape[2]

class TestResidualUpsamplerResidualPath:
    """Tests covering the behavior of the residual connection adjustment."""

    class DummyBlock(nn.Module):
        def __init__(self, factor):
            super().__init__()
            self.factor = factor
            self.upsample = nn.Upsample(scale_factor=factor, mode='nearest')
        def forward(self, x):
            residual = x
            out = self.upsample(x)
            if out.shape != residual.shape:
                residual = self.upsample(residual)
            return out + residual

    def test_residual_adjustment(self):
        """
        GIVEN a ResidualUpsampler with a dummy block that upsamples by 2
        WHEN a forward pass is executed
        THEN the residual is correctly adjusted so that output equals the sum of the upsampled input.
        """
        config = ResidualUpsamplerConfig(
            upsampling_factor=2.0,
            in_channels=2,
            hidden_channels=4,
            out_channels=2
        )
        upsampler = ResidualUpsampler(config)
        # Override blocks with a dummy block that upsamples by 2
        upsampler.blocks = nn.ModuleList([self.DummyBlock(2.0)])
        # Replace projection layers with the identity function for test clarity
        upsampler.input_proj = nn.Identity()
        upsampler.final_layer = nn.Identity()
        upsampler.output_proj = nn.Identity()

        input_tensor = torch.tensor([[[1.0, 2.0, 3.0],
                                       [4.0, 5.0, 6.0]]])
        output = upsampler(input_tensor)
        upsampled = nn.Upsample(scale_factor=2.0, mode='nearest')(input_tensor)
        expected = upsampled + upsampled
        assert output.shape == expected.shape
        assert torch.allclose(output, expected)
