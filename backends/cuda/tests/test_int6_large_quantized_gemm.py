# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""INT6 W6A16 tile buckets M = 8, 16, 32, 64."""

import os
import unittest

import torch

from executorch.backends.cuda.autotune.launch_params import InvalidLaunchParam
from executorch.backends.cuda.quantize_op_dispatch.int6_dispatch import _unit_dq_mm_int6
from executorch.backends.cuda.tests.test_int6_dispatch import _make_int6_tensor
from executorch.backends.cuda.triton.kernels import int6_quantized_gemm as kernels
from executorch.backends.cuda.triton.kernels.int6_quantized_gemm import (
    INT6_QUANTIZED_GEMM,
    SUPPORTED_BUCKETS,
    TILE_SPLIT_K_CANDIDATES,
)
from executorch.backends.cuda.triton.kernels.quantized_gemm_utils import (
    tile_autotune_configs,
)

GROUP_SIZE = 16
LARGE_BUCKETS = (8, 16, 32, 64)


def packed_weights(n: int, k: int, seed: int):
    torch.manual_seed(seed)
    weight, _, _ = _make_int6_tensor(n, k, GROUP_SIZE)
    return tuple(
        tensor.cuda() for tensor in (weight.ql, weight.qh, weight.scale, weight.steps)
    )


def assert_close(test: unittest.TestCase, out: torch.Tensor, ref: torch.Tensor):
    out_f = out.float()
    ref_f = ref.float()
    test.assertTrue(torch.isfinite(out_f).all())
    torch.testing.assert_close(out_f, ref_f, rtol=0.02, atol=4.0)
    mean_rel = (out_f - ref_f).abs().mean() / ref_f.abs().mean().clamp_min(1.0e-6)
    test.assertLess(mean_rel.item(), 0.01)


def legal_configs(bucket: int, k: int, split_k: int = 1):
    return kernels._prune_tiles(
        tile_autotune_configs(bucket),
        {"K": k},
        SPLIT_K=split_k,
    )


# Configs checked per (bucket, N, K, split) by default: an evenly spaced sample
# keeps each test under the CI time limit. INT6_LARGE_TEST_ALL_CONFIGS=1 runs
# every legal config.
_SAMPLED_CONFIGS = 6


def _sampled(configs):
    if (
        os.environ.get("INT6_LARGE_TEST_ALL_CONFIGS") == "1"
        or len(configs) <= _SAMPLED_CONFIGS
    ):
        return configs
    step = len(configs) / _SAMPLED_CONFIGS
    return [configs[int(i * step)] for i in range(_SAMPLED_CONFIGS)]


class Int6LargeRulesTest(unittest.TestCase):
    def test_buckets_and_generic_candidate_counts(self):
        self.assertEqual(SUPPORTED_BUCKETS, (1, 2, 3, 4, 8, 16, 32, 64))
        expected_counts = {8: 18, 16: 36, 32: 54, 64: 72}
        for bucket, expected in expected_counts.items():
            configs = tile_autotune_configs(bucket)
            self.assertEqual(len(configs), expected)
            self.assertEqual(len(legal_configs(bucket, 256)), expected // 3)
            self.assertEqual(len(legal_configs(bucket, 5376)), expected)
            self.assertTrue(hasattr(torch.ops.triton, f"int6_quantized_gemm_m{bucket}"))

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA required")
    def test_static_rows_below_bucket_are_supported(self):
        weights = packed_weights(37, 256, seed=1)
        for bucket in LARGE_BUCKETS:
            x = torch.randn(bucket - 1, 256, dtype=torch.bfloat16, device="cuda")
            self.assertTrue(
                INT6_QUANTIZED_GEMM.supports(bucket, x, *weights, GROUP_SIZE)
            )


@unittest.skipUnless(torch.cuda.is_available(), "CUDA required")
class Int6LargeNumericsTest(unittest.TestCase):
    def _check_every_config(self, bucket):
        for n in (37, 257):
            for k in (256, 5376):
                weights = packed_weights(n, k, seed=n ^ k)
                widest = max(c for c in TILE_SPLIT_K_CANDIDATES if c <= k // 256)
                for m in (bucket, bucket - 1):
                    x = torch.randn(m, k, dtype=torch.bfloat16, device="cuda")
                    ref = _unit_dq_mm_int6(x, *weights, GROUP_SIZE)
                    for split_k in sorted({1, widest}):
                        for config in _sampled(legal_configs(bucket, k, split_k)):
                            with self.subTest(
                                n=n,
                                k=k,
                                bucket=bucket,
                                m=m,
                                split_k=split_k,
                                config=str(config),
                            ):
                                out = kernels._launch_w6a16(
                                    bucket,
                                    x,
                                    *weights,
                                    GROUP_SIZE,
                                    split_k=split_k,
                                    config=config,
                                )
                                self.assertEqual(out.shape, (m, n))
                                self.assertEqual(out.dtype, torch.bfloat16)
                                assert_close(self, out, ref)
                del weights
                torch.cuda.empty_cache()

    def test_bucket_8_configs(self):
        self._check_every_config(8)

    def test_bucket_16_configs(self):
        self._check_every_config(16)

    def test_bucket_32_configs(self):
        self._check_every_config(32)

    def test_bucket_64_configs(self):
        self._check_every_config(64)

    def test_raw_uint8_scale_codes_are_signed(self):
        """uint8 storage holds the same signed scale codes, negative ones included."""
        n, k, bucket, m = 37, 256, 8, 7
        ql, qh, scale, steps = packed_weights(n, k, seed=9)
        codes = scale.clone()
        codes[:, ::2] = -codes[:, ::2]
        raw = codes.view(torch.uint8)
        self.assertTrue((raw >= 128).any().item())
        x = torch.randn(m, k, dtype=torch.bfloat16, device="cuda")
        out = kernels._launch_w6a16(
            bucket,
            x,
            ql,
            qh,
            raw,
            steps,
            GROUP_SIZE,
            split_k=1,
            config=legal_configs(bucket, k)[0],
        )
        assert_close(self, out, _unit_dq_mm_int6(x, ql, qh, codes, steps, GROUP_SIZE))

    def test_sanitizer_smoke(self):
        n, k, bucket, m = 37, 256, 8, 7
        weights = packed_weights(n, k, seed=11)
        x = torch.randn(m, k, dtype=torch.bfloat16, device="cuda")
        config = legal_configs(bucket, k)[0]
        out = kernels._launch_w6a16(
            bucket,
            x,
            *weights,
            GROUP_SIZE,
            split_k=1,
            config=config,
        )
        assert_close(self, out, _unit_dq_mm_int6(x, *weights, GROUP_SIZE))

    def test_an_illegal_split_is_rejected(self):
        weights = packed_weights(37, 512, seed=13)
        x = torch.randn(8, 512, dtype=torch.bfloat16, device="cuda")
        with self.assertRaises(InvalidLaunchParam):
            kernels._launch_w6a16(8, x, *weights, GROUP_SIZE, split_k=4)


if __name__ == "__main__":
    unittest.main()
