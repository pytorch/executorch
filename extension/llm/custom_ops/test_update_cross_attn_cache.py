import unittest

import torch

# Import the custom ops to ensure they are registered
from executorch.extension.llm.custom_ops import custom_ops  # noqa: F401


class TestUpdateCrossAttnCache(unittest.TestCase):
    def test_update_cross_attn_cache(self):

        # Create tensors
        # Cache: [B=2, S_max=4, H=1, D=4]
        cache = torch.zeros(2, 4, 1, 4, dtype=torch.float32)
        # Value: [B=2, S=2, H=1, D=4] (S < S_max)
        value = torch.randn(2, 2, 1, 4, dtype=torch.float32)

        # Compile a function that uses the op
        @torch.compile
        def fn(v, c):
            return torch.ops.executorch.update_cross_attn_cache(v, c)

        # Run it
        out = fn(value, cache)

        # Check correctness
        # The first 2 elements in dim 1 should match value
        torch.testing.assert_close(
            cache[:, :2, :, :], value, msg="Cache slice not updated correctly"
        )

        # Make sure out and cache are close. In eager they are the same objects.
        torch.testing.assert_close(
            out, cache, msg="Output and cache are different objects"
        )

        # The rest should be zeros
        torch.testing.assert_close(
            cache[:, 2:, :, :],
            torch.zeros_like(cache[:, 2:, :, :]),
            msg="Rest of cache was modified",
        )

    def test_update_cross_attn_cache_in_cond(self):
        # Create tensors

        # Value: [B=2, S=2, H=1, D=4]
        value = torch.randn(2, 2, 1, 4, dtype=torch.float32)
        # Alternative value for false branch
        value_alt = torch.randn(2, 2, 1, 4, dtype=torch.float32)

        # Define a function that uses the op inside torch.cond
        def fn_with_cond(pred, v1, v2, c):
            def true_fn(v, cache):
                return torch.ops.executorch.update_cross_attn_cache(v, cache)

            def false_fn(v, cache):
                return torch.ops.executorch.update_cross_attn_cache(v, cache)

            return torch.cond(pred, true_fn, false_fn, (v1, c), (v2, c))

        # Test with true condition
        pred_true = torch.tensor(True)
        cache_true = torch.zeros(2, 4, 1, 4, dtype=torch.float32)

        # Compile the function
        @torch.compile
        def compiled_fn(pred, v1, v2, c):
            return fn_with_cond(pred, v1, v2, c)

        # Run with true condition
        compiled_fn(pred_true, value, value_alt, cache_true)

        # Check that the true branch was executed (value was used)
        torch.testing.assert_close(
            cache_true[:, :2, :, :],
            value,
            msg="Cache not updated correctly in true branch",
        )

        # Test with false condition
        pred_false = torch.tensor(False)
        cache_false = torch.zeros(2, 4, 1, 4, dtype=torch.float32)

        compiled_fn(pred_false, value, value_alt, cache_false)

        # Check that the false branch was executed (value_alt was used)
        torch.testing.assert_close(
            cache_false[:, :2, :, :],
            value_alt,
            msg="Cache not updated correctly in false branch",
        )

    def test_update_cross_attn_cache_export(self):

        # Create tensors
        # Cache: [B=2, S_max=4, H=1, D=4]
        cache = torch.zeros(2, 4, 1, 4, dtype=torch.float32)
        # Value: [B=2, S=2, H=1, D=4]
        value = torch.randn(2, 2, 1, 4, dtype=torch.float32)

        # Define a function that uses the op
        class UpdateCacheModule(torch.nn.Module):
            def forward(self, v, c):
                return torch.ops.executorch.update_cross_attn_cache(v, c)

        module = UpdateCacheModule()

        # Export the module
        exported_program = torch.export.export(
            module,
            (value, cache),
        )

        # Run the exported program
        cache_exported = torch.zeros(2, 4, 1, 4, dtype=torch.float32)
        exported_program.module()(value, cache_exported)

        # Check correctness
        torch.testing.assert_close(
            cache_exported[:, :2, :, :],
            value,
            msg="Cache not updated correctly after export",
        )

    def test_update_cross_attn_cache_different_shapes(self):
        print("Testing executorch::update_cross_attn_cache with different shapes...")

        # Test with different batch sizes and sequence lengths
        test_cases = [
            # (B, S_max, S, H, D)
            (1, 10, 5, 2, 8),
            (4, 8, 3, 4, 16),
            (2, 16, 10, 1, 32),
        ]

        for B, S_max, S, H, D in test_cases:
            cache = torch.zeros(B, S_max, H, D, dtype=torch.float32)
            value = torch.randn(B, S, H, D, dtype=torch.float32)

            @torch.compile
            def fn(v, c):
                return torch.ops.executorch.update_cross_attn_cache(v, c)

            fn(value, cache)

            # Check that the first S positions are updated
            torch.testing.assert_close(
                cache[:, :S, :, :],
                value,
                msg=f"Failed for shape B={B}, S_max={S_max}, S={S}, H={H}, D={D}",
            )

            # Check that the rest remain zeros
            if S < S_max:
                torch.testing.assert_close(
                    cache[:, S:, :, :],
                    torch.zeros_like(cache[:, S:, :, :]),
                    msg=f"Remaining cache modified for shape B={B}, S_max={S_max}, S={S}, H={H}, D={D}",
                )

    def test_update_cross_attn_cache_full_sequence(self):

        # Cache: [B=2, S_max=4, H=1, D=4]
        cache = torch.zeros(2, 4, 1, 4, dtype=torch.float32)
        # Value: [B=2, S=4, H=1, D=4] (S == S_max)
        value = torch.randn(2, 4, 1, 4, dtype=torch.float32)

        @torch.compile
        def fn(v, c):
            return torch.ops.executorch.update_cross_attn_cache(v, c)

        fn(value, cache)

        # The entire cache should match value
        torch.testing.assert_close(
            cache, value, msg="Cache not fully updated when S == S_max"
        )
