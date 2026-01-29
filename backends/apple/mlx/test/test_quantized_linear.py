#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
Test for TorchAO int4 quantized nn.Linear using the MLX delegate.

Usage:
    # Run via run_all_tests (recommended):
    python -m executorch.backends.apple.mlx.test.run_all_tests quantized_linear

    # Run directly with custom args:
    python -m executorch.backends.apple.mlx.test.test_quantized_linear run --group-size 64
"""

from typing import List, Tuple

import torch
import torch.nn as nn

from .test_utils import OpTestCase, register_test, run_op_test_main


class QuantizedLinearModel(nn.Module):
    """Simple linear layer that will be quantized."""

    def __init__(
        self, in_features: int = 64, out_features: int = 128, bias: bool = True
    ):
        super().__init__()
        self.linear = nn.Linear(in_features, out_features, bias=bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear(x)


@register_test
class QuantizedLinearTest(OpTestCase):
    """Test case for TorchAO int4 quantized nn.Linear."""

    name = "quantized_linear"
    rtol = 0.1  # Higher tolerance for quantized ops
    atol = 0.1

    def __init__(
        self,
        in_features: int = 64,
        out_features: int = 128,
        batch_size: int = 2,
        seq_len: int = 16,
        bias: bool = True,
        group_size: int = 32,
        dtype: torch.dtype = torch.bfloat16,
    ):
        self.in_features = in_features
        self.out_features = out_features
        self.batch_size = batch_size
        self.seq_len = seq_len
        self.bias = bias
        self.group_size = group_size
        self.dtype = dtype

        # Build unique test name
        parts = ["quantized_linear", f"g{group_size}"]
        if not bias:
            parts.append("no_bias")
        self.name = "_".join(parts)

    @classmethod
    def get_test_configs(cls) -> List["QuantizedLinearTest"]:
        """Return all test configurations to run."""
        return [
            cls(),  # default (group_size=32, with bias)
        ]

    def create_model(self) -> nn.Module:
        model = QuantizedLinearModel(
            self.in_features, self.out_features, bias=self.bias
        )
        model = model.to(self.dtype)

        try:
            from torchao.quantization.granularity import PerGroup
            from torchao.quantization.quant_api import IntxWeightOnlyConfig, quantize_

            quantize_(
                model,
                IntxWeightOnlyConfig(
                    weight_dtype=torch.int4, granularity=PerGroup(self.group_size)
                ),
            )
        except ImportError:
            raise RuntimeError("TorchAO not installed. Run: pip install torchao")

        return model

    def create_inputs(self) -> Tuple[torch.Tensor, ...]:
        x = torch.randn(
            self.batch_size, self.seq_len, self.in_features, dtype=self.dtype
        )
        return (x,)

    def get_dynamic_shapes(self):
        return None


# Factory for CLI usage
def _create_from_args(args) -> QuantizedLinearTest:
    if args is None:
        return QuantizedLinearTest()
    return QuantizedLinearTest(
        in_features=args.in_features,
        out_features=args.out_features,
        batch_size=args.batch_size,
        seq_len=args.seq_len,
        bias=not args.no_bias,
        group_size=args.group_size,
    )


def _add_args(parser):
    parser.add_argument("--in-features", type=int, default=64, help="Input features")
    parser.add_argument("--out-features", type=int, default=128, help="Output features")
    parser.add_argument("--batch-size", type=int, default=2, help="Batch size")
    parser.add_argument("--seq-len", type=int, default=16, help="Sequence length")
    parser.add_argument("--no-bias", action="store_true", help="Test without bias")
    parser.add_argument(
        "--group-size", type=int, default=32, help="Quantization group size"
    )


if __name__ == "__main__":
    run_op_test_main(
        _create_from_args, "Test quantized nn.Linear on MLX delegate", _add_args
    )
