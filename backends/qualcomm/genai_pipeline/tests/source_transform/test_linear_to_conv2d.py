# Copyright (c) Qualcomm Innovation Center, Inc.
# All rights reserved
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Tests for the linear-to-conv2d module transforms."""

import sys
import types
import unittest
from unittest.mock import MagicMock

import torch

from executorch.backends.qualcomm.genai_pipeline.source_transform.linear_to_conv2d import (
    convert_linear_to_conv2d,
    prepare_conv_submodules,
)
from torch import nn


class _LinearAsConv2d(nn.Module):
    def __init__(self, linear):
        super().__init__()
        self.conv = nn.Conv2d(
            linear.in_features,
            linear.out_features,
            kernel_size=1,
            bias=linear.bias is not None,
        )
        self.conv.weight.data.copy_(
            linear.weight.data.reshape(linear.out_features, linear.in_features, 1, 1)
        )
        if linear.bias is not None:
            self.conv.bias.data.copy_(linear.bias.data)

    def forward(self, x):
        y = self.conv(x.transpose(-1, -2).unsqueeze(-1))
        return y.squeeze(-1).transpose(-1, -2)


def _convert_named_linears_to_conv2d(module):
    for name, child in list(module.named_children()):
        if isinstance(child, nn.Linear):
            setattr(module, name, _LinearAsConv2d(child))
    return module


def _fake_backend_utils():
    module = types.ModuleType("executorch.backends.qualcomm.utils.utils")
    module.convert_linear_to_conv2d = _convert_named_linears_to_conv2d
    return module


class TestPrepareConvSubmodules(unittest.TestCase):
    """Attention and feed-forward blocks pre-shape their weights for conv2d."""

    def test_calls_both_prepare_hooks_on_every_layer(self):
        layers = []
        for _ in range(2):
            layer = MagicMock()
            layer.attention.prepare_attention_conv = MagicMock()
            layer.feed_forward.prepare_feedforward_conv = MagicMock()
            layers.append(layer)
        module = MagicMock(layers=layers)

        result = prepare_conv_submodules(module)

        for layer in layers:
            layer.attention.prepare_attention_conv.assert_called_once_with()
            layer.feed_forward.prepare_feedforward_conv.assert_called_once_with()
        self.assertIs(result, module)

    def test_skips_layers_without_the_hooks(self):
        layer = MagicMock(spec=[])
        layer.attention = MagicMock(spec=[])
        layer.feed_forward = MagicMock(spec=[])
        module = MagicMock(layers=[layer])

        self.assertIs(prepare_conv_submodules(module), module)


class TestConvertLinearToConv2d(unittest.TestCase):
    """HTP runs conv2d faster than linear; the backend util rewrites them.

    The util walks named attributes, so the module under test mirrors the real
    decoder's shape (``self.wq = nn.Linear(...)``) rather than using
    ``nn.Sequential``, whose children it does not reach.
    """

    def test_replaces_named_linear_attributes(self):
        class Block(nn.Module):
            def __init__(self):
                super().__init__()
                self.wq = nn.Linear(4, 4, bias=False)

        module = Block()

        with unittest.mock.patch.dict(
            sys.modules,
            {"executorch.backends.qualcomm.utils.utils": _fake_backend_utils()},
        ):
            result = convert_linear_to_conv2d(module)

        self.assertFalse(
            any(isinstance(m, nn.Linear) for m in result.modules()),
            "nn.Linear should have been rewritten",
        )

    def test_preserves_forward_semantics(self):
        class Block(nn.Module):
            def __init__(self):
                super().__init__()
                self.wq = nn.Linear(4, 4, bias=False)

            def forward(self, x):
                return self.wq(x)

        module = Block().eval()
        x = torch.randn(1, 2, 4)
        with torch.no_grad():
            expected = module(x)
            with unittest.mock.patch.dict(
                sys.modules,
                {"executorch.backends.qualcomm.utils.utils": _fake_backend_utils()},
            ):
                actual = convert_linear_to_conv2d(module)(x)

        torch.testing.assert_close(actual, expected)


if __name__ == "__main__":
    unittest.main()
