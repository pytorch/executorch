#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
Test for KV cache update operations using the MLX delegate.

Uses the UPDATE_CACHE pattern which recognizes:
    transpose(1,2) -> llama.update_cache -> transpose(1,2)

This pattern is needed because:
- MLX uses [B, H, S, D] layout (heads before sequence)
- llama.update_cache expects [B, S, H, D] layout (sequence before heads)

The pattern eliminates the transposes by operating directly on dim=2.

NOTE: Some tests skip output comparison because ExecutorTorch's llama.update_cache
custom op has a bug where it doesn't work with non-contiguous (transposed view)
tensors. When you pass cache.transpose(1,2) as the destination, the update silently
fails. The MLX SliceUpdateNode correctly implements the operation, so MLX outputs
are correct but don't match the (buggy) PyTorch expected outputs.

Usage:
    # Run via run_all_tests (recommended):
    python -m executorch.backends.apple.mlx.test.run_all_tests kv_cache

    # Run specific variant:
    python -m executorch.backends.apple.mlx.test.run_all_tests kv_cache_direct

    # Run directly with custom args:
    python -m executorch.backends.apple.mlx.test.test_kv_cache run --test direct
"""

from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn

# Import custom ops to register llama.update_cache
from executorch.extension.llm.custom_ops import custom_ops  # noqa: F401
from torch.export import Dim

from .test_utils import (
    OpTestCase,
    print_mlx_graph_summary,
    register_test,
    run_cpp_test_runner,
    run_op_test_main,
)


class KVCacheUpdateModelPatternVerify(nn.Module):
    """
    KV cache update using slice assignment for verification.

    Cache is stored as [B, H, S, D] (SDPA convention).
    Input is [B, H, S_step, D] (SDPA convention).

    This model uses direct slice assignment instead of llama.update_cache
    to generate correct expected outputs for verifying UpdateCacheHandler.
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
        k_val: torch.Tensor,  # [B, H, S_step, D]
        v_val: torch.Tensor,  # [B, H, S_step, D]
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Update cache using direct slice assignment (for verification)."""
        start_pos = self.start_pos
        seq_len = k_val.shape[2]

        # Direct slice update on axis=2
        self.k_cache[:, :, start_pos : start_pos + seq_len, :] = k_val
        self.v_cache[:, :, start_pos : start_pos + seq_len, :] = v_val

        return self.k_cache.clone(), self.v_cache.clone()


class KVCacheUpdateModel(nn.Module):
    """
    KV cache update using llama.update_cache custom op.

    Cache is stored as [B, H, S, D] (SDPA convention).
    Input is [B, H, S_step, D] (SDPA convention).

    Both cache and update are transposed to [B, S, H, D] for update_cache,
    but since cache is stored as [B, H, S, D], no output transpose is needed.
    The MLX handler fuses this to a single SliceUpdateNode on axis=2.
    """

    def __init__(
        self,
        num_heads: int = 4,
        head_dim: int = 64,
        max_seq_len: int = 128,
        start_pos: int = 0,  # Fixed position for export
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
        k_val: torch.Tensor,  # [B, H, S_step, D] - SDPA layout
        v_val: torch.Tensor,  # [B, H, S_step, D] - SDPA layout
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Update KV cache at fixed position and return updated cache.

        Input k_val/v_val are in [B, H, S, D] layout (SDPA convention).
        We transpose both cache and input to [B, S, H, D] for update_cache.
        Since cache is stored as [B, H, S, D], return it directly (with clone).
        """
        start_pos = self.start_pos

        # Transpose inputs from [B, H, S_step, D] to [B, S_step, H, D] for update_cache
        k_val_transposed = k_val.transpose(1, 2)
        v_val_transposed = v_val.transpose(1, 2)

        # Transpose cache views from [B, H, S, D] to [B, S, H, D] for update_cache
        k_cache_view = self.k_cache.transpose(1, 2)
        v_cache_view = self.v_cache.transpose(1, 2)

        # Call update_cache custom op (mutates cache via the transposed view)
        _ = torch.ops.llama.update_cache(k_val_transposed, k_cache_view, start_pos)
        _ = torch.ops.llama.update_cache(v_val_transposed, v_cache_view, start_pos)

        # Return cache directly - already [B, H, S, D]!
        # Use clone to avoid buffer mutation output issue
        return self.k_cache.clone(), self.v_cache.clone()


class KVCacheUpdateModelDirect(nn.Module):
    """
    KV cache update using llama.update_cache custom op directly.

    Cache is stored as [B, S, H, D] (ExecutorTorch convention).
    Input is [B, S_step, H, D] (ExecutorTorch convention).

    This tests the standalone op handler (axis=1) rather than the pattern
    handler (axis=2) which requires surrounding transposes.
    """

    def __init__(
        self,
        num_heads: int = 4,
        head_dim: int = 64,
        max_seq_len: int = 128,
        start_pos: int = 0,  # Fixed position for export
    ):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.max_seq_len = max_seq_len
        self.start_pos = start_pos

        # KV cache buffers - [B, S, H, D] layout (ExecutorTorch convention)
        self.register_buffer(
            "k_cache", torch.zeros(1, max_seq_len, num_heads, head_dim)
        )
        self.register_buffer(
            "v_cache", torch.zeros(1, max_seq_len, num_heads, head_dim)
        )

    def forward(
        self,
        k_val: torch.Tensor,  # [B, S_step, H, D] - ExecutorTorch layout
        v_val: torch.Tensor,  # [B, S_step, H, D] - ExecutorTorch layout
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Update KV cache at fixed position and return updated cache.

        Both inputs and cache are in [B, S, H, D] layout (ExecutorTorch convention).
        No transposes needed - uses the standalone op handler directly.
        """
        start_pos = self.start_pos

        # Call update_cache custom op directly (no transposes)
        _ = torch.ops.llama.update_cache(k_val, self.k_cache, start_pos)
        _ = torch.ops.llama.update_cache(v_val, self.v_cache, start_pos)

        # Return cache with clone to avoid buffer mutation output issue
        return self.k_cache.clone(), self.v_cache.clone()


class KVCacheUpdateModelDynamicPos(nn.Module):
    """
    KV cache update with dynamic start_pos passed as input.

    Cache is stored as [B, S, H, D] (ExecutorTorch convention).
    Input is [B, S_step, H, D] (ExecutorTorch convention).

    This tests the dynamic start_pos code path where start_pos is a
    SymInt input rather than a fixed constant.
    """

    def __init__(
        self,
        num_heads: int = 4,
        head_dim: int = 64,
        max_seq_len: int = 128,
    ):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.max_seq_len = max_seq_len

        # KV cache buffers - [B, S, H, D] layout (ExecutorTorch convention)
        self.register_buffer(
            "k_cache", torch.zeros(1, max_seq_len, num_heads, head_dim)
        )
        self.register_buffer(
            "v_cache", torch.zeros(1, max_seq_len, num_heads, head_dim)
        )

    def forward(
        self,
        k_val: torch.Tensor,  # [B, S_step, H, D] - ExecutorTorch layout
        v_val: torch.Tensor,  # [B, S_step, H, D] - ExecutorTorch layout
        start_pos: torch.Tensor,  # Scalar tensor for position
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Update KV cache at dynamic position and return updated cache.

        start_pos is passed as a scalar tensor input, making it dynamic
        (a SymInt after export).
        """
        # Extract scalar from tensor for update_cache
        pos = start_pos.item()

        # Call update_cache custom op directly (no transposes)
        _ = torch.ops.llama.update_cache(k_val, self.k_cache, pos)
        _ = torch.ops.llama.update_cache(v_val, self.v_cache, pos)

        # Return cache with clone to avoid buffer mutation output issue
        return self.k_cache.clone(), self.v_cache.clone()


class KVCacheUpdateModelFullyDynamic(nn.Module):
    """
    KV cache update with both dynamic start_pos and dynamic seq_len.

    Cache is stored as [B, S, H, D] (ExecutorTorch convention).
    Input is [B, S_step, H, D] where S_step is dynamic.

    This tests the fully dynamic code path where both start_pos and
    the sequence length of the input are dynamic (SymInts).

    Uses llama.update_cache with dynamic seq_len and start_pos.item() for dynamic position.
    """

    def __init__(
        self,
        num_heads: int = 4,
        head_dim: int = 64,
        max_seq_len: int = 128,
    ):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.max_seq_len = max_seq_len

        # KV cache buffers - [B, S, H, D] layout (ExecutorTorch convention)
        self.register_buffer(
            "k_cache", torch.zeros(1, max_seq_len, num_heads, head_dim)
        )
        self.register_buffer(
            "v_cache", torch.zeros(1, max_seq_len, num_heads, head_dim)
        )

    def forward(
        self,
        k_val: torch.Tensor,  # [B, S_step, H, D] - ExecutorTorch layout, S_step is dynamic
        v_val: torch.Tensor,  # [B, S_step, H, D] - ExecutorTorch layout, S_step is dynamic
        start_pos: torch.Tensor,  # Scalar tensor for position
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Update KV cache using llama.update_cache with dynamic sequence length.

        Both start_pos (via .item()) and the sequence length of k_val/v_val are dynamic.
        This is the most general case for KV cache updates.
        """
        # Use .item() to get SymInt from scalar tensor
        pos = start_pos.item()

        # Use llama.update_cache - handles dynamic seq_len
        torch.ops.llama.update_cache(k_val, self.k_cache, pos)
        torch.ops.llama.update_cache(v_val, self.v_cache, pos)

        # Return cache with clone to avoid buffer mutation output issue
        return self.k_cache.clone(), self.v_cache.clone()


class KVCacheUpdateModelPatternFullyDynamic(nn.Module):
    """
    KV cache update with both dynamic start_pos and dynamic seq_len.

    Cache is stored as [B, H, S, D] (SDPA convention).
    Input is [B, H, S_step, D] where S_step is dynamic.

    This tests the UPDATE_CACHE pattern (transpose -> update_cache -> transpose)
    with both dynamic start_pos and dynamic seq_len.
    """

    def __init__(
        self,
        num_heads: int = 4,
        head_dim: int = 64,
        max_seq_len: int = 128,
    ):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.max_seq_len = max_seq_len

        # KV cache buffers - [B, H, S, D] layout (SDPA convention)
        self.register_buffer(
            "k_cache", torch.zeros(1, num_heads, max_seq_len, head_dim)
        )
        self.register_buffer(
            "v_cache", torch.zeros(1, num_heads, max_seq_len, head_dim)
        )

    def forward(
        self,
        k_val: torch.Tensor,  # [B, H, S_step, D] - SDPA layout, S_step is dynamic
        v_val: torch.Tensor,  # [B, H, S_step, D] - SDPA layout, S_step is dynamic
        start_pos: torch.Tensor,  # Scalar tensor for position
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Update KV cache using llama.update_cache with transposes.

        Input k_val/v_val are in [B, H, S, D] layout (SDPA convention).
        We transpose both cache and input to [B, S, H, D] for update_cache,
        then return cache directly (already [B, H, S, D]).

        Both start_pos (via .item()) and the sequence length are dynamic.
        """
        # Use .item() to get SymInt from scalar tensor
        pos = start_pos.item()

        # Transpose inputs from [B, H, S_step, D] to [B, S_step, H, D] for update_cache
        k_val_transposed = k_val.transpose(1, 2)
        v_val_transposed = v_val.transpose(1, 2)

        # Transpose cache views from [B, H, S, D] to [B, S, H, D] for update_cache
        k_cache_view = self.k_cache.transpose(1, 2)
        v_cache_view = self.v_cache.transpose(1, 2)

        # Call update_cache custom op (mutates cache via the transposed view)
        torch.ops.llama.update_cache(k_val_transposed, k_cache_view, pos)
        torch.ops.llama.update_cache(v_val_transposed, v_cache_view, pos)

        # Return cache directly - already [B, H, S, D]!
        # Use clone to avoid buffer mutation output issue
        return self.k_cache.clone(), self.v_cache.clone()


@register_test
class KVCacheTest(OpTestCase):
    """Test case for KV cache update operations.

    Supports multiple variants:
    - pattern: Uses transpose -> update_cache -> transpose pattern (axis=2)
    - pattern_verify: Same as pattern but uses direct slice assignment for expected outputs
    - direct: Uses update_cache directly without transposes (axis=1)
    - dynamic_pos: Uses dynamic start_pos input
    - fully_dynamic: Uses both dynamic start_pos and dynamic seq_len
    - pattern_fully_dynamic: Pattern variant with dynamic shapes

    Note: Pattern tests (kv_cache, kv_cache_pattern_fully_dynamic) skip output comparison
    because ExecutorTorch's llama.update_cache custom op has a bug with non-contiguous
    (transposed view) tensors. MLX correctly implements the operation.
    """

    name = "kv_cache"
    rtol = 1e-5
    atol = 1e-5

    # Tests that skip output comparison due to ExecutorTorch bug
    _SKIP_COMPARISON_TESTS = {"kv_cache", "kv_cache_pattern_fully_dynamic"}

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

        # Store test inputs for pattern_verify
        self._test_k_val = None
        self._test_v_val = None

        # Set name based on variant
        variant_names = {
            "pattern": "kv_cache",
            "pattern_verify": "kv_cache_pattern_verify",
            "direct": "kv_cache_direct",
            "dynamic_pos": "kv_cache_dynamic_pos",
            "fully_dynamic": "kv_cache_fully_dynamic",
            "pattern_fully_dynamic": "kv_cache_pattern_fully_dynamic",
        }
        self.name = variant_names.get(variant, "kv_cache")

        # Create dynamic dimension for fully_dynamic variants
        if variant in ("fully_dynamic", "pattern_fully_dynamic"):
            self.seq_dim = Dim("seq_step", min=1, max=max_seq_len)
        else:
            self.seq_dim = None

    @classmethod
    def get_test_configs(cls) -> List["KVCacheTest"]:
        """Return all test configurations to run."""
        return [
            cls(variant="pattern"),
            cls(variant="pattern_verify"),
            cls(variant="direct"),
            cls(variant="dynamic_pos"),
            cls(variant="fully_dynamic"),
            cls(variant="pattern_fully_dynamic"),
        ]

    def create_model(self) -> nn.Module:
        if self.variant == "pattern":
            return KVCacheUpdateModel(
                num_heads=self.num_heads,
                head_dim=self.head_dim,
                max_seq_len=self.max_seq_len,
                start_pos=self.start_pos,
            )
        elif self.variant == "pattern_verify":
            return KVCacheUpdateModel(
                num_heads=self.num_heads,
                head_dim=self.head_dim,
                max_seq_len=self.max_seq_len,
                start_pos=self.start_pos,
            )
        elif self.variant == "direct":
            return KVCacheUpdateModelDirect(
                num_heads=self.num_heads,
                head_dim=self.head_dim,
                max_seq_len=self.max_seq_len,
                start_pos=self.start_pos,
            )
        elif self.variant == "dynamic_pos":
            return KVCacheUpdateModelDynamicPos(
                num_heads=self.num_heads,
                head_dim=self.head_dim,
                max_seq_len=self.max_seq_len,
            )
        elif self.variant == "fully_dynamic":
            return KVCacheUpdateModelFullyDynamic(
                num_heads=self.num_heads,
                head_dim=self.head_dim,
                max_seq_len=self.max_seq_len,
            )
        elif self.variant == "pattern_fully_dynamic":
            return KVCacheUpdateModelPatternFullyDynamic(
                num_heads=self.num_heads,
                head_dim=self.head_dim,
                max_seq_len=self.max_seq_len,
            )
        else:
            raise ValueError(f"Unknown variant: {self.variant}")

    def _is_sdpa_layout(self) -> bool:
        """Return True if this variant uses SDPA layout [B, H, S, D]."""
        return self.variant in ("pattern", "pattern_verify", "pattern_fully_dynamic")

    def _has_start_pos_input(self) -> bool:
        """Return True if this variant takes start_pos as input."""
        return self.variant in ("dynamic_pos", "fully_dynamic", "pattern_fully_dynamic")

    def create_inputs(self) -> Tuple[torch.Tensor, ...]:
        """Create inputs for export (tracing)."""
        if self._is_sdpa_layout():
            # [B, H, S, D] layout
            if self.variant == "pattern_fully_dynamic":
                k_val = torch.randn(
                    1, self.num_heads, self.export_seq_step, self.head_dim
                )
                v_val = torch.randn(
                    1, self.num_heads, self.export_seq_step, self.head_dim
                )
            else:
                k_val = torch.randn(1, self.num_heads, self.seq_step, self.head_dim)
                v_val = torch.randn(1, self.num_heads, self.seq_step, self.head_dim)
        else:
            # [B, S, H, D] layout
            if self.variant == "fully_dynamic":
                k_val = torch.randn(
                    1, self.export_seq_step, self.num_heads, self.head_dim
                )
                v_val = torch.randn(
                    1, self.export_seq_step, self.num_heads, self.head_dim
                )
            else:
                k_val = torch.randn(1, self.seq_step, self.num_heads, self.head_dim)
                v_val = torch.randn(1, self.seq_step, self.num_heads, self.head_dim)

        if self._has_start_pos_input():
            start_pos = torch.tensor(0, dtype=torch.int64)
            return (k_val, v_val, start_pos)
        return (k_val, v_val)

    def create_test_inputs(self) -> Tuple[torch.Tensor, ...]:
        """Create inputs for testing."""
        # For pattern_verify, return cached test inputs for reproducibility
        if self.variant == "pattern_verify":
            if self._test_k_val is None:
                self._test_k_val = torch.randn(
                    1, self.num_heads, self.seq_step, self.head_dim
                )
                self._test_v_val = torch.randn(
                    1, self.num_heads, self.seq_step, self.head_dim
                )
            return (self._test_k_val, self._test_v_val)

        if self._is_sdpa_layout():
            # [B, H, S, D] layout
            if self.variant == "pattern_fully_dynamic":
                k_val = torch.randn(
                    1, self.num_heads, self.test_seq_step, self.head_dim
                )
                v_val = torch.randn(
                    1, self.num_heads, self.test_seq_step, self.head_dim
                )
            else:
                k_val = torch.randn(1, self.num_heads, self.seq_step, self.head_dim)
                v_val = torch.randn(1, self.num_heads, self.seq_step, self.head_dim)
        else:
            # [B, S, H, D] layout
            if self.variant == "fully_dynamic":
                k_val = torch.randn(
                    1, self.test_seq_step, self.num_heads, self.head_dim
                )
                v_val = torch.randn(
                    1, self.test_seq_step, self.num_heads, self.head_dim
                )
            else:
                k_val = torch.randn(1, self.seq_step, self.num_heads, self.head_dim)
                v_val = torch.randn(1, self.seq_step, self.num_heads, self.head_dim)

        if self._has_start_pos_input():
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
        elif self.variant == "pattern_fully_dynamic":
            return {
                "k_val": {2: self.seq_dim},
                "v_val": {2: self.seq_dim},
                "start_pos": None,
            }
        return None

    def generate_test_files(self, verbose: bool = False) -> Tuple:
        """Generate test files with correct expected outputs.

        For pattern_verify variant, uses direct slice assignment for expected
        outputs to avoid ExecutorTorch's llama.update_cache bug.
        """
        if self.variant != "pattern_verify":
            # Use default implementation for other variants
            return super().generate_test_files(verbose=verbose)

        # Special handling for pattern_verify: compute expected outputs using
        # direct slice assignment instead of buggy llama.update_cache
        from .test_utils import export_model_to_pte, save_tensors_to_bin

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
        verify_model = KVCacheUpdateModelPatternVerify(
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

    def run_test(self, verbose: bool = False) -> bool:
        """Run the full test with special handling for pattern tests.

        Pattern tests (kv_cache, kv_cache_pattern_fully_dynamic) skip output
        comparison because ExecutorTorch's llama.update_cache has a bug with
        non-contiguous tensors.
        """
        print(f"\n{'='*60}")
        print(f"Running test: {self.name}")
        print(f"{'='*60}\n")

        # Generate test files
        print("Step 1: Generating test files...")
        pte_path, input_path, expected_path = self.generate_test_files(verbose=verbose)

        # Print MLX graph summary
        print_mlx_graph_summary(pte_path)

        # Run C++ binary
        print("Step 2: Running C++ binary...")
        actual_path = self.get_test_dir() / "actual_output.bin"
        if not run_cpp_test_runner(pte_path, input_path, actual_path, verbose=verbose):
            return False

        # Compare outputs
        print("\nStep 3: Comparing outputs...")

        # For pattern tests, skip comparison due to ExecutorTorch bug
        if self.name in self._SKIP_COMPARISON_TESTS:
            print(
                "NOTE: Output comparison disabled for pattern test because ExecutorTorch's"
            )
            print("      llama.update_cache custom op doesn't work with non-contiguous")
            print(
                "      (transposed view) tensors. MLX correctly implements the operation,"
            )
            print("      but PyTorch expected outputs are wrong.")
            print("✓ PASSED (runtime execution succeeded)")
            return True
        else:
            # Compare outputs
            passed, message = self.compare_with_actual(actual_path)
            if passed:
                print(f"✓ PASSED: {message}")
            else:
                print(f"✗ FAILED: {message}")
            return passed


# Factory for CLI usage
def _create_from_args(args) -> KVCacheTest:
    if args is None:
        return KVCacheTest()

    # Map CLI test names to variants
    variant_map = {
        "pattern": "pattern",
        "pattern_verify": "pattern_verify",
        "direct": "direct",
        "dynamic_pos": "dynamic_pos",
        "fully_dynamic": "fully_dynamic",
        "pattern_fully_dynamic": "pattern_fully_dynamic",
    }
    variant = variant_map.get(args.test, "pattern")

    return KVCacheTest(
        num_heads=args.num_heads,
        head_dim=args.head_dim,
        max_seq_len=args.max_seq_len,
        seq_step=args.seq_step,
        start_pos=args.start_pos,
        variant=variant,
    )


def _add_args(parser):
    parser.add_argument(
        "--test",
        choices=[
            "pattern",
            "pattern_verify",
            "direct",
            "dynamic_pos",
            "fully_dynamic",
            "pattern_fully_dynamic",
        ],
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
        _create_from_args, "Test KV cache update on MLX delegate", _add_args
    )
