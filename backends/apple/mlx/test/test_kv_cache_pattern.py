#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
Test for KV cache pattern recognition using the MLX delegate.

This tests the transpose → update_cache → transpose pattern fusion,
verifying that MLX correctly recognizes and optimizes this pattern
into a single SliceUpdateNode on axis=2.

Uses SDPA's [B, H, S, D] layout (heads before sequence).

NOTE: Output comparison is skipped for most pattern tests because
ExecutorTorch's llama.update_cache custom op has a bug where it doesn't
work correctly with non-contiguous (transposed view) tensors. The MLX
backend correctly implements the operation, so these tests verify:
1. Pattern recognition works (transposes are fused)
2. Runtime execution succeeds

For numerical correctness tests, see test_slice_update.py which tests
the SliceUpdate op directly without transposes.

Usage:
    # Run via run_all_tests (recommended):
    python -m executorch.backends.apple.mlx.test.run_all_tests kv_cache_pattern

    # Run specific variant:
    python -m executorch.backends.apple.mlx.test.run_all_tests kv_cache_pattern_verify

    # Run directly with custom args:
    python -m executorch.backends.apple.mlx.test.test_kv_cache_pattern run --variant verify
"""

from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn

# Import custom ops to register llama.update_cache
from executorch.extension.llm.custom_ops import custom_ops  # noqa: F401
from torch.export import Dim

from .test_utils import (
    export_model_to_pte,
    OpTestCase,
    register_test,
    run_op_test_main,
    save_tensors_to_bin,
)


class KVCachePatternModel(nn.Module):
    """
    KV cache update using the transpose → update_cache → transpose pattern.

    Cache is stored as [B, H, S, D] (SDPA convention).
    Input is [B, H, S_step, D] (SDPA convention).

    Both cache and input are transposed to [B, S, H, D] for update_cache,
    then the result is implicitly transposed back. The MLX handler fuses
    this pattern into a single SliceUpdateNode on axis=2.
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

        # KV cache buffers - [B, H, S, D] layout (SDPA convention)
        self.register_buffer(
            "k_cache", torch.zeros(1, num_heads, max_seq_len, head_dim)
        )
        self.register_buffer(
            "v_cache", torch.zeros(1, num_heads, max_seq_len, head_dim)
        )

    def forward(
        self,
        k_val: torch.Tensor,
        v_val: torch.Tensor,
        start_pos: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Update KV cache using the transpose pattern.

        Args:
            k_val: Key values [B, H, S_step, D]
            v_val: Value values [B, H, S_step, D]
            start_pos: Optional position tensor (for dynamic pos variants)
        """
        if self.dynamic_pos:
            pos = start_pos.item()
        else:
            pos = self.start_pos

        # Transpose inputs from [B, H, S_step, D] to [B, S_step, H, D]
        k_val_transposed = k_val.transpose(1, 2)
        v_val_transposed = v_val.transpose(1, 2)

        # Transpose cache views from [B, H, S, D] to [B, S, H, D]
        k_cache_view = self.k_cache.transpose(1, 2)
        v_cache_view = self.v_cache.transpose(1, 2)

        # Call update_cache custom op (mutates cache via transposed view)
        torch.ops.llama.update_cache(k_val_transposed, k_cache_view, pos)
        torch.ops.llama.update_cache(v_val_transposed, v_cache_view, pos)

        # Return cache directly - already [B, H, S, D]
        return self.k_cache.clone(), self.v_cache.clone()


class KVCachePatternVerifyModel(nn.Module):
    """
    KV cache update using direct slice assignment for verification.

    This model uses direct slice assignment instead of llama.update_cache
    to generate correct expected outputs. Used to verify that the pattern
    handler produces correct results.
    """

    def __init__(
        self,
        num_heads: int = 4,
        head_dim: int = 64,
        max_seq_len: int = 128,
        start_pos: int = 0,
    ):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.max_seq_len = max_seq_len
        self.start_pos = start_pos

        # KV cache buffers - [B, H, S, D] layout (SDPA convention)
        self.register_buffer(
            "k_cache", torch.zeros(1, num_heads, max_seq_len, head_dim)
        )
        self.register_buffer(
            "v_cache", torch.zeros(1, num_heads, max_seq_len, head_dim)
        )

    def forward(
        self,
        k_val: torch.Tensor,
        v_val: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Update cache using direct slice assignment (for verification)."""
        start_pos = self.start_pos
        seq_len = k_val.shape[2]

        # Direct slice update on axis=2
        self.k_cache[:, :, start_pos : start_pos + seq_len, :] = k_val
        self.v_cache[:, :, start_pos : start_pos + seq_len, :] = v_val

        return self.k_cache.clone(), self.v_cache.clone()


@register_test
class KVCachePatternTest(OpTestCase):
    """Test case for KV cache pattern recognition.

    Tests that MLX correctly recognizes the transpose → update_cache → transpose
    pattern and fuses it into a single SliceUpdateNode on axis=2.

    Variants:
    - pattern: Basic pattern test (skips output comparison)
    - verify: Uses direct slice assignment for expected outputs (verifies correctness)
    - fully_dynamic: Pattern with dynamic pos and seq_len (skips output comparison)
    """

    name = "kv_cache_pattern"
    rtol = 1e-5
    atol = 1e-5

    # ExecutorTorch bug explanation for skip_comparison
    _ET_BUG_REASON = (
        "ExecutorTorch's llama.update_cache doesn't work with transposed views"
    )

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
        variant: str = "pattern",
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
            "pattern": "kv_cache_pattern",
            "verify": "kv_cache_pattern_verify",
            "fully_dynamic": "kv_cache_pattern_fully_dynamic",
        }
        self.name = variant_names.get(variant, "kv_cache_pattern")

        # Skip comparison for pattern tests (except verify)
        if variant != "verify":
            self.skip_comparison = True
            self.skip_comparison_reason = self._ET_BUG_REASON

        # Create dynamic dimension for fully_dynamic variant
        if variant == "fully_dynamic":
            self.seq_dim = Dim("seq_step", min=1, max=max_seq_len)
        else:
            self.seq_dim = None

    @classmethod
    def get_test_configs(cls) -> List["KVCachePatternTest"]:
        """Return all test configurations to run."""
        return [
            cls(variant="pattern"),
            cls(variant="verify"),
            cls(variant="fully_dynamic"),
        ]

    def _has_dynamic_pos(self) -> bool:
        """Return True if this variant takes start_pos as input."""
        return self.variant == "fully_dynamic"

    def _has_dynamic_seq(self) -> bool:
        """Return True if this variant has dynamic sequence length."""
        return self.variant == "fully_dynamic"

    def create_model(self) -> nn.Module:
        return KVCachePatternModel(
            num_heads=self.num_heads,
            head_dim=self.head_dim,
            max_seq_len=self.max_seq_len,
            start_pos=self.start_pos,
            dynamic_pos=self._has_dynamic_pos(),
        )

    def create_inputs(self) -> Tuple[torch.Tensor, ...]:
        """Create inputs for export (tracing)."""
        seq_len = self.export_seq_step if self._has_dynamic_seq() else self.seq_step
        k_val = torch.randn(1, self.num_heads, seq_len, self.head_dim)
        v_val = torch.randn(1, self.num_heads, seq_len, self.head_dim)

        if self._has_dynamic_pos():
            start_pos = torch.tensor(0, dtype=torch.int64)
            return (k_val, v_val, start_pos)
        return (k_val, v_val)

    def create_test_inputs(self) -> Tuple[torch.Tensor, ...]:
        """Create inputs for testing."""
        seq_len = self.test_seq_step if self._has_dynamic_seq() else self.seq_step
        k_val = torch.randn(1, self.num_heads, seq_len, self.head_dim)
        v_val = torch.randn(1, self.num_heads, seq_len, self.head_dim)

        if self._has_dynamic_pos():
            start_pos = torch.tensor(self.test_start_pos, dtype=torch.int64)
            return (k_val, v_val, start_pos)
        return (k_val, v_val)

    def get_dynamic_shapes(self) -> Optional[Dict]:
        """Return dynamic shapes specification for torch.export."""
        if self.variant == "fully_dynamic":
            return {
                "k_val": {2: self.seq_dim},
                "v_val": {2: self.seq_dim},
                "start_pos": None,
            }
        return None

    def generate_test_files(self, verbose: bool = False) -> Tuple:
        """Generate test files with correct expected outputs for verify variant."""
        if self.variant != "verify":
            return super().generate_test_files(verbose=verbose)

        # Special handling for verify: use direct slice assignment for expected outputs
        test_dir = self.get_test_dir()

        pte_path = test_dir / "model.pte"
        input_path = test_dir / "input.bin"
        expected_path = test_dir / "expected_output.bin"

        # Set seed for reproducibility
        self._set_seed()

        # Create model and inputs
        model = self.create_model()
        export_inputs = self.create_inputs()

        # Set seed again before creating test inputs
        self._set_seed()
        test_inputs = self.create_test_inputs()

        # Get expected outputs using CORRECT method (direct slice assignment)
        verify_model = KVCachePatternVerifyModel(
            num_heads=self.num_heads,
            head_dim=self.head_dim,
            max_seq_len=self.max_seq_len,
            start_pos=self.start_pos,
        )
        verify_model.eval()
        with torch.no_grad():
            expected_outputs = list(verify_model(*test_inputs))

        # Export model with export inputs
        print(f"Exporting model to {pte_path}")

        export_model_to_pte(
            model,
            export_inputs,
            pte_path,
            use_fp16=self.use_fp16,
            dynamic_shapes=self.get_dynamic_shapes(),
            verbose=verbose,
        )

        # Save test inputs
        print(f"Saving inputs to {input_path}")
        save_tensors_to_bin(list(test_inputs), input_path)

        # Save expected outputs
        print(f"Saving expected outputs to {expected_path}")
        save_tensors_to_bin(expected_outputs, expected_path)

        return pte_path, input_path, expected_path


# Factory for CLI usage
def _create_from_args(args) -> KVCachePatternTest:
    if args is None:
        return KVCachePatternTest()

    return KVCachePatternTest(
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
        choices=["pattern", "verify", "fully_dynamic"],
        default="pattern",
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
    run_op_test_main(
        _create_from_args, "Test KV cache pattern on MLX delegate", _add_args
    )
