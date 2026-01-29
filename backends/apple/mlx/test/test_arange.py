#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
Test for torch.arange() op using the MLX delegate.

Arange creates a 1D tensor with evenly spaced values.

Usage:
    # Run via run_all_tests (recommended):
    python -m executorch.backends.apple.mlx.test.run_all_tests arange

    # Run directly:
    python -m executorch.backends.apple.mlx.test.test_arange run
"""

from typing import List, Tuple

import torch
import torch.nn as nn

from .test_utils import OpTestCase, register_test, run_op_test_main


class ArangeModel(nn.Module):
    """Model that creates a tensor using arange and multiplies with input."""

    def __init__(self, stop: int):
        super().__init__()
        self.stop = stop

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Create arange and multiply with input to ensure arange goes through graph
        indices = torch.arange(self.stop, dtype=x.dtype, device=x.device)
        return x * indices


class DynamicArangeModel(nn.Module):
    """Model that uses arange with dynamic start/stop from tensor.item().

    This tests the case where arange's start/stop are symbolic (SymInt),
    which happens when they come from item(). This is the pattern used in
    Whisper decoder for positional embeddings.
    """

    def __init__(self, length: int, vocab_size: int = 32):
        super().__init__()
        self.length = length
        self.embed = nn.Embedding(vocab_size, 16)

    def forward(self, pos: torch.Tensor) -> torch.Tensor:
        # Extract position as int (becomes SymInt during tracing)
        torch._check(pos.numel() == 1)
        pos_int = pos.item()
        torch._check_is_size(pos_int)

        # Create positions using arange with dynamic start/stop
        # This is the exact pattern from Whisper decoder
        positions = torch.arange(
            pos_int, pos_int + self.length, device=pos.device, dtype=torch.long
        )

        # Use the positions to index into embedding
        return self.embed(positions)


@register_test
class ArangeTest(OpTestCase):
    """Test case for torch.arange()."""

    name = "arange"
    rtol = 1e-4
    atol = 1e-4

    def __init__(
        self,
        stop: int = 10,
    ):
        self.stop = stop
        self.name = f"arange_{stop}"

    @classmethod
    def get_test_configs(cls) -> List["ArangeTest"]:
        """Return all test configurations to run."""
        return [
            cls(stop=10),
            cls(stop=32),
            cls(stop=100),
        ]

    def create_model(self) -> nn.Module:
        return ArangeModel(self.stop)

    def create_inputs(self) -> Tuple[torch.Tensor, ...]:
        x = torch.randn(self.stop)
        return (x,)

    def get_dynamic_shapes(self):
        return None


@register_test
class DynamicArangeTest(OpTestCase):
    """Test case for torch.arange() with dynamic start/stop.

    This tests the pattern used in Whisper decoder where arange's
    start/stop come from tensor.item() and are symbolic at trace time.
    """

    name = "arange_dynamic"
    rtol = 1e-4
    atol = 1e-4

    def __init__(
        self,
        position: int = 4,
        length: int = 4,
        vocab_size: int = 32,
    ):
        self.position = position
        self.length = length
        self.vocab_size = vocab_size
        self.name = f"arange_dynamic_pos{position}_len{length}"

    @classmethod
    def get_test_configs(cls) -> List["DynamicArangeTest"]:
        """Return all test configurations to run."""
        return [
            cls(position=0, length=4),
            cls(position=4, length=4),
            cls(position=10, length=8),
        ]

    def create_model(self) -> nn.Module:
        return DynamicArangeModel(length=self.length, vocab_size=self.vocab_size)

    def create_inputs(self) -> Tuple[torch.Tensor, ...]:
        pos = torch.tensor([self.position], dtype=torch.long)
        return (pos,)

    def get_dynamic_shapes(self):
        return None


# Factory for CLI usage
def _create_from_args(args) -> ArangeTest:
    if args is None:
        return ArangeTest()
    if args.dynamic:
        return DynamicArangeTest(
            position=args.position,
            length=args.length,
            vocab_size=args.vocab_size,
        )
    return ArangeTest(
        stop=args.stop,
    )


def _add_args(parser):
    parser.add_argument("--stop", type=int, default=10, help="Stop value for arange")
    parser.add_argument(
        "--dynamic",
        action="store_true",
        help="Use dynamic arange test (start/stop from item())",
    )
    parser.add_argument(
        "--position",
        type=int,
        default=4,
        help="Start position for dynamic arange (default: 4)",
    )
    parser.add_argument(
        "--length",
        type=int,
        default=4,
        help="Sequence length for dynamic arange (default: 4)",
    )
    parser.add_argument(
        "--vocab-size",
        type=int,
        default=32,
        help="Embedding vocabulary size for dynamic arange (default: 32)",
    )


if __name__ == "__main__":
    run_op_test_main(
        _create_from_args, "Test torch.arange() on MLX delegate", _add_args
    )
