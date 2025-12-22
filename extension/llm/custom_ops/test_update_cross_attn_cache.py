import unittest

import torch

# Import the custom ops to ensure they are registered
from executorch.extension.llm.custom_ops import custom_ops  # noqa: F401


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
        print("Testing executorch::update_cross_attn_cache with different shapes...")

        # Test with different batch sizes and sequence lengths
        test_cases = [
            # (B, H, S_max, S, D)
            (1, 2, 10, 5, 8),
            (4, 4, 8, 3, 16),
            (2, 1, 16, 10, 32),
        ]

        for B, H, S_max, S, D in test_cases:
            # Cache: [B, H, S_max, D], Value: [B, H, S, D]
            cache = torch.zeros(B, H, S_max, D, dtype=torch.float32)
            value = torch.randn(B, H, S, D, dtype=torch.float32)

            @torch.compile
            def fn(v, c):
                return torch.ops.executorch.update_cross_attn_cache(v, c)

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