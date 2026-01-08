# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import unittest

import torch

# Import the custom ops to ensure they are registered
from executorch.extension.llm.custom_ops import custom_ops  # noqa: F401

# Check CUDA availability once at module level
CUDA_AVAILABLE = torch.cuda.is_available()


class TestUpdateCrossAttnCache(unittest.TestCase):
    def test_update_cross_attn_cache(self):

        # Create tensors
        # Cache: [B=2, H=1, S_max=4, D=4]
        cache = torch.zeros(2, 1, 4, 4, dtype=torch.float32)
        # Value: [B=2, H=1, S=2, D=4] (S < S_max)
        value = torch.randn(2, 1, 2, 4, dtype=torch.float32)

        # Compile a function that uses the op
        @torch.compile
        def fn(v, c):
            return torch.ops.executorch.update_cross_attn_cache(v, c)

        # Run it
        out = fn(value, cache)

        # Check correctness
        # The first 2 elements in dim 2 (sequence dim) should match value
        torch.testing.assert_close(
            cache[:, :, :2, :], value, msg="Cache slice not updated correctly"
        )

        # Make sure out and cache are close. In eager they are the same objects.
        torch.testing.assert_close(
            out, cache, msg="Output and cache are different objects"
        )

        # The rest should be zeros
        torch.testing.assert_close(
            cache[:, :, 2:, :],
            torch.zeros_like(cache[:, :, 2:, :]),
            msg="Rest of cache was modified",
        )

    def test_update_cross_attn_cache_in_cond(self):
        # Create tensors

        # Value: [B=2, H=1, S=2, D=4]
        value = torch.randn(2, 1, 2, 4, dtype=torch.float32)
        # Alternative value for false branch
        value_alt = torch.randn(2, 1, 2, 4, dtype=torch.float32)

        # Define a function that uses the op inside torch.cond
        def fn_with_cond(pred, v1, v2, c):
            def true_fn(v1, v2, cache):
                return torch.ops.executorch.update_cross_attn_cache(v1, cache)

            def false_fn(v1, v2, cache):
                return torch.ops.executorch.update_cross_attn_cache(v2, cache)

            return torch.cond(pred, true_fn, false_fn, (v1, v2, c))

        # Test with true condition
        pred_true = torch.tensor(True)
        cache_true = torch.zeros(2, 1, 4, 4, dtype=torch.float32)

        # Compile the function
        @torch.compile
        def compiled_fn(pred, v1, v2, c):
            return fn_with_cond(pred, v1, v2, c)

        # Run with true condition
        compiled_fn(pred_true, value, value_alt, cache_true)

        # Check that the true branch was executed (value was used)
        torch.testing.assert_close(
            cache_true[:, :, :2, :],
            value,
            msg="Cache not updated correctly in true branch",
        )

        # Test with false condition
        pred_false = torch.tensor(False)
        cache_false = torch.zeros(2, 1, 4, 4, dtype=torch.float32)

        compiled_fn(pred_false, value, value_alt, cache_false)

        # Check that the false branch was executed (value_alt was used)
        torch.testing.assert_close(
            cache_false[:, :, :2, :],
            value_alt,
            msg="Cache not updated correctly in false branch",
        )

    def test_update_cross_attn_cache_export(self):

        # Create tensors
        # Cache: [B=2, H=1, S_max=4, D=4]
        cache = torch.zeros(2, 1, 4, 4, dtype=torch.float32)
        # Value: [B=2, H=1, S=2, D=4]
        value = torch.randn(2, 1, 2, 4, dtype=torch.float32)
        # Alternative value for false branch
        value_alt = torch.randn(2, 1, 2, 4, dtype=torch.float32)

        # Define a module that uses torch.cond with the op
        class UpdateCacheCondModule(torch.nn.Module):
            def forward(self, pred, v1, v2, c):
                def true_fn(v1, v2, cache):
                    return torch.ops.executorch.update_cross_attn_cache(v1, cache)

                def false_fn(v1, v2, cache):
                    return torch.ops.executorch.update_cross_attn_cache(v2, cache)

                return torch.cond(pred, true_fn, false_fn, (v1, v2, c))

        module = UpdateCacheCondModule()

        # Export the module with true condition
        pred_true = torch.tensor(True)
        exported_program = torch.export.export(
            module,
            (pred_true, value, value_alt, cache),
        )

        # Run the exported program with true condition
        cache_true = torch.zeros(2, 1, 4, 4, dtype=torch.float32)
        exported_program.module()(pred_true, value, value_alt, cache_true)

        # Check that the true branch was executed (value was used)
        torch.testing.assert_close(
            cache_true[:, :, :2, :],
            value,
            msg="Cache not updated correctly in true branch after export",
        )

        # Run the exported program with false condition
        pred_false = torch.tensor(False)
        cache_false = torch.zeros(2, 1, 4, 4, dtype=torch.float32)
        exported_program.module()(pred_false, value, value_alt, cache_false)

        # Check that the false branch was executed (value_alt was used)
        torch.testing.assert_close(
            cache_false[:, :, :2, :],
            value_alt,
            msg="Cache not updated correctly in false branch after export",
        )

    def test_update_cross_attn_cache_different_shapes(self):

        # Test with different batch sizes and sequence lengths
        test_cases = [
            # (B, H, S_max, S, D)
            (1, 2, 10, 5, 8),
            (4, 4, 8, 3, 16),
            (2, 1, 16, 10, 32),
        ]

        @torch.compile
        def fn(v, c):
            return torch.ops.executorch.update_cross_attn_cache(v, c)

        for B, H, S_max, S, D in test_cases:
            # Cache: [B, H, S_max, D], Value: [B, H, S, D]
            cache = torch.zeros(B, H, S_max, D, dtype=torch.float32)
            value = torch.randn(B, H, S, D, dtype=torch.float32)

            fn(value, cache)

            # Check that the first S positions in dim 2 are updated
            torch.testing.assert_close(
                cache[:, :, :S, :],
                value,
                msg=f"Failed for shape B={B}, H={H}, S_max={S_max}, S={S}, D={D}",
            )

            # Check that the rest remain zeros
            if S < S_max:
                torch.testing.assert_close(
                    cache[:, :, S:, :],
                    torch.zeros_like(cache[:, :, S:, :]),
                    msg=f"Remaining cache modified for shape B={B}, H={H}, S_max={S_max}, S={S}, D={D}",
                )

    def test_update_cross_attn_cache_full_sequence(self):

        # Cache: [B=2, H=1, S_max=4, D=4]
        cache = torch.zeros(2, 1, 4, 4, dtype=torch.float32)
        # Value: [B=2, H=1, S=4, D=4] (S == S_max)
        value = torch.randn(2, 1, 4, 4, dtype=torch.float32)

        @torch.compile
        def fn(v, c):
            return torch.ops.executorch.update_cross_attn_cache(v, c)

        fn(value, cache)

        # The entire cache should match value
        torch.testing.assert_close(
            cache, value, msg="Cache not fully updated when S == S_max"
        )

    @unittest.skipUnless(CUDA_AVAILABLE, "CUDA not available")
    def test_alias_and_update_cross_attn_cache_with_cond_triton(self):
        """Test combining alias and update_cross_attn_cache ops with torch.cond,
        lowered to Triton on CUDA. True branch uses alias, false branch uses
        update_cross_attn_cache."""

        # Create CUDA tensors
        # Value: [B=2, H=1, S=2, D=4]
        value = torch.randn(2, 1, 2, 4, dtype=torch.float32, device="cuda")
        # Extra tensor for alias op
        extra = torch.randn(2, 1, 4, 4, dtype=torch.float32, device="cuda")

        # Define a function that uses different ops in each branch
        def fn_with_cond(pred, v, extra_tensor, c):
            def true_fn(v, extra_tensor, cache):
                # True branch: use alias op only
                aliased_cache, aliased_extra = torch.ops.executorch.alias(
                    cache, extra_tensor
                )
                # Return sum of aliased tensors (no cache mutation)
                return aliased_cache + aliased_extra

            def false_fn(v, extra_tensor, cache):
                # False branch: use update_cross_attn_cache op only
                updated = torch.ops.executorch.update_cross_attn_cache(v, cache)
                return updated

            return torch.cond(pred, true_fn, false_fn, (v, extra_tensor, c))

        # Compile the function with Triton backend
        @torch.compile(backend="inductor")
        def compiled_fn(pred, v, extra_tensor, c):
            return fn_with_cond(pred, v, extra_tensor, c)

        # Test with true condition (alias branch)
        pred_true = torch.tensor(True, device="cuda")
        cache_true = torch.zeros(2, 1, 4, 4, dtype=torch.float32, device="cuda")

        result_true = compiled_fn(pred_true, value, extra, cache_true)

        # Check that the true branch was executed (alias: cache + extra)
        expected_true = cache_true + extra
        torch.testing.assert_close(
            result_true,
            expected_true,
            msg="Result incorrect in true branch (alias) with CUDA/Triton",
        )

        # Cache should remain unchanged in true branch (alias doesn't mutate)
        torch.testing.assert_close(
            cache_true,
            torch.zeros(2, 1, 4, 4, dtype=torch.float32, device="cuda"),
            msg="Cache should not be mutated in true branch (alias)",
        )

        # Test with false condition (update_cross_attn_cache branch)
        pred_false = torch.tensor(False, device="cuda")
        cache_false = torch.zeros(2, 1, 4, 4, dtype=torch.float32, device="cuda")

        compiled_fn(pred_false, value, extra, cache_false)

        # Check that the false branch was executed (update_cross_attn_cache)
        # The cache should be updated with value in the first S positions
        torch.testing.assert_close(
            cache_false[:, :, :2, :],
            value,
            msg="Cache not updated correctly in false branch with CUDA/Triton",
        )

        # The rest of the cache should remain zeros
        torch.testing.assert_close(
            cache_false[:, :, 2:, :],
            torch.zeros(2, 1, 2, 4, dtype=torch.float32, device="cuda"),
            msg="Rest of cache was modified in false branch",
        )
