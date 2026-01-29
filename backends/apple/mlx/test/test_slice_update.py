#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
Test for slice update operations using the MLX delegate.

This tests the SliceUpdate op directly (without transpose patterns),
verifying numerical correctness. Uses ExecutorTorch's [B, S, H, D] layout.

For pattern recognition tests (transpose → update_cache → transpose fusion),
see test_kv_cache_pattern.py.

Usage:
    # Run via run_all_tests (recommended):
    python -m executorch.backends.apple.mlx.test.run_all_tests slice_update

    # Run specific variant:
    python -m executorch.backends.apple.mlx.test.run_all_tests slice_update_dynamic_pos

    # Run directly with custom args:
    python -m executorch.backends.apple.mlx.test.test_slice_update run --variant dynamic_pos
"""

from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn

# Import custom ops to register llama.update_cache
from executorch.extension.llm.custom_ops import custom_ops  # noqa: F401
from torch.export import Dim

from .test_utils import OpTestCase, register_test, run_op_test_main


class SliceUpdateModel(nn.Module):
    """
    Slice update using llama.update_cache custom op.

    Cache is stored as [B, S, H, D] (ExecutorTorch convention).
    Input is [B, S_step, H, D] (ExecutorTorch convention).

    This model can be configured for:
    - Static: Fixed start_pos at export time
    - Dynamic pos: start_pos passed as input tensor
    - Fully dynamic: Both start_pos and seq_len are dynamic
    """

    def __init__(
        self,
        num_heads: int = 4,
        head_dim: int = 64,
        max_seq_len: int = 128,
        start_pos: int = 0,
        dynamic_pos: bool = False,
    ):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.max_seq_len = max_seq_len
        self.start_pos = start_pos
        self.dynamic_pos = dynamic_pos

        # KV cache buffers - [B, S, H, D] layout (ExecutorTorch convention)
        self.register_buffer(
            "k_cache", torch.zeros(1, max_seq_len, num_heads, head_dim)
        )
        self.register_buffer(
            "v_cache", torch.zeros(1, max_seq_len, num_heads, head_dim)
        )

    def forward(
        self,
        k_val: torch.Tensor,
        v_val: torch.Tensor,
        start_pos: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Update KV cache and return updated cache.

        Args:
            k_val: Key values [B, S_step, H, D]
            v_val: Value values [B, S_step, H, D]
            start_pos: Optional position tensor (for dynamic pos variants)
        """
        if self.dynamic_pos:
            pos = start_pos.item()
        else:
            pos = self.start_pos

        # Call update_cache custom op
        torch.ops.llama.update_cache(k_val, self.k_cache, pos)
        torch.ops.llama.update_cache(v_val, self.v_cache, pos)

        # Return cache with clone to avoid buffer mutation output issue
        return self.k_cache.clone(), self.v_cache.clone()


@register_test
class SliceUpdateTest(OpTestCase):
    """Test case for slice update operations.

    Tests the SliceUpdate/update_cache op directly without transposes.
    Uses ExecutorTorch's [B, S, H, D] layout.

    Variants:
    - static: Fixed start_pos at export time
    - dynamic_pos: start_pos passed as input tensor
    - fully_dynamic: Both start_pos and seq_len are dynamic
    """

    name = "slice_update"
    rtol = 1e-5
    atol = 1e-5

    def __init__(
        self,
        num_heads: int = 4,
        head_dim: int = 64,
        max_seq_len: int = 128,
        seq_step: int = 8,
        start_pos: int = 0,
        test_start_pos: int = 16,
        export_seq_step: int = 8,
        test_seq_step: int = 4,
        variant: str = "static",
    ):
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.max_seq_len = max_seq_len
        self.seq_step = seq_step
        self.start_pos = start_pos
        self.test_start_pos = test_start_pos
        self.export_seq_step = export_seq_step
        self.test_seq_step = test_seq_step
        self.variant = variant

        # Set name based on variant
        variant_names = {
            "static": "slice_update",
            "dynamic_pos": "slice_update_dynamic_pos",
            "fully_dynamic": "slice_update_fully_dynamic",
        }
        self.name = variant_names.get(variant, "slice_update")

        # Create dynamic dimension for fully_dynamic variant
        if variant == "fully_dynamic":
            self.seq_dim = Dim("seq_step", min=1, max=max_seq_len)
        else:
            self.seq_dim = None

    @classmethod
    def get_test_configs(cls) -> List["SliceUpdateTest"]:
        """Return all test configurations to run."""
        return [
            cls(variant="static"),
            cls(variant="dynamic_pos"),
            cls(variant="fully_dynamic"),
        ]

    def _has_dynamic_pos(self) -> bool:
        """Return True if this variant takes start_pos as input."""
        return self.variant in ("dynamic_pos", "fully_dynamic")

    def _has_dynamic_seq(self) -> bool:
        """Return True if this variant has dynamic sequence length."""
        return self.variant == "fully_dynamic"

    def create_model(self) -> nn.Module:
        return SliceUpdateModel(
            num_heads=self.num_heads,
            head_dim=self.head_dim,
            max_seq_len=self.max_seq_len,
            start_pos=self.start_pos,
            dynamic_pos=self._has_dynamic_pos(),
        )

    def create_inputs(self) -> Tuple[torch.Tensor, ...]:
        """Create inputs for export (tracing)."""
        seq_len = self.export_seq_step if self._has_dynamic_seq() else self.seq_step
        k_val = torch.randn(1, seq_len, self.num_heads, self.head_dim)
        v_val = torch.randn(1, seq_len, self.num_heads, self.head_dim)

        if self._has_dynamic_pos():
            start_pos = torch.tensor(0, dtype=torch.int64)
            return (k_val, v_val, start_pos)
        return (k_val, v_val)

    def create_test_inputs(self) -> Tuple[torch.Tensor, ...]:
        """Create inputs for testing."""
        seq_len = self.test_seq_step if self._has_dynamic_seq() else self.seq_step
        k_val = torch.randn(1, seq_len, self.num_heads, self.head_dim)
        v_val = torch.randn(1, seq_len, self.num_heads, self.head_dim)

        if self._has_dynamic_pos():
            start_pos = torch.tensor(self.test_start_pos, dtype=torch.int64)
            return (k_val, v_val, start_pos)
        return (k_val, v_val)

    def get_dynamic_shapes(self) -> Optional[Dict]:
        """Return dynamic shapes specification for torch.export."""
        if self.variant == "fully_dynamic":
            return {
                "k_val": {1: self.seq_dim},
                "v_val": {1: self.seq_dim},
                "start_pos": None,
            }
        return None


# Factory for CLI usage
def _create_from_args(args) -> SliceUpdateTest:
    if args is None:
        return SliceUpdateTest()

    return SliceUpdateTest(
        num_heads=args.num_heads,
        head_dim=args.head_dim,
        max_seq_len=args.max_seq_len,
        seq_step=args.seq_step,
        start_pos=args.start_pos,
        variant=args.variant,
    )


def _add_args(parser):
    parser.add_argument(
        "--variant",
        choices=["static", "dynamic_pos", "fully_dynamic"],
        default="static",
        help="Which test variant to run",
    )
    parser.add_argument(
        "--num-heads",
        type=int,
        default=4,
        help="Number of KV heads (default: 4)",
    )
    parser.add_argument(
        "--head-dim",
        type=int,
        default=64,
        help="Head dimension (default: 64)",
    )
    parser.add_argument(
        "--max-seq-len",
        type=int,
        default=128,
        help="Max sequence length for cache (default: 128)",
    )
    parser.add_argument(
        "--seq-step",
        type=int,
        default=8,
        help="Tokens per step (default: 8)",
    )
    parser.add_argument(
        "--start-pos",
        type=int,
        default=0,
        help="Start position (default: 0)",
    )


if __name__ == "__main__":
    run_op_test_main(_create_from_args, "Test slice update on MLX delegate", _add_args)
