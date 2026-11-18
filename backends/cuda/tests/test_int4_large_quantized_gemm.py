# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Exhaustive correctness tests for W4A16 INT4 buckets M=8,16,32,64."""

import os
import unittest

import torch

from executorch.backends.cuda.autotune.launch_params import InvalidLaunchParam
from executorch.backends.cuda.coalesced_int4_tensor import CudaCoalescedInt4Tensor
from executorch.backends.cuda.quantize_op_dispatch.int4_dispatch import _dequant_matmul
from executorch.backends.cuda.triton.kernels import int4_quantized_gemm as int4_kernel
from executorch.backends.cuda.triton.kernels.int4_quantized_gemm import (
    int4_autotune_configs,
    INT4_QUANTIZED_GEMM,
    SUPPORTED_BUCKETS,
    TILE_SPLIT_K_CANDIDATES,
)
from executorch.backends.cuda.triton.kernels.quantized_gemm_utils import (
    check_split_k,
    TILE_BLOCK_M_CHOICES,
    TILE_BLOCK_N_CHOICES,
    TILE_WARP_CHOICES,
)
from executorch.extension.llm.export.int4 import ExportableInt4Tensor
from executorch.extension.llm.export.quant.quantize import quantize_weight
from executorch.extension.llm.export.quant.recipe import QuantConfig

GROUP_SIZE = 32
LARGE_BUCKETS = (8, 16, 32, 64)
N_TAILS = (37, 257)
K_VALUES = (256, 512, 6656)
SMALLER_M = {8: 6, 16: 8, 32: 24, 64: 48}


def _packed(n: int, k: int, seed: int = 0):
    torch.manual_seed(seed)
    dense = torch.randn(n, k, dtype=torch.bfloat16)
    config = QuantConfig(
        bits=4, group_size=GROUP_SIZE, symmetric=False, method="min_max"
    )
    weight = CudaCoalescedInt4Tensor.from_exportable_int4_tensor(
        ExportableInt4Tensor.from_int4_tensor(quantize_weight(dense, config))
    )
    return tuple(
        tensor.cuda()
        for tensor in (
            weight.qdata,
            weight.scale,
            weight.scale_step,
            weight.zero_point,
            weight.zero_point_step,
        )
    )


def _check_close(test: unittest.TestCase, actual, reference):
    test.assertEqual(actual.dtype, torch.bfloat16)
    test.assertTrue(torch.isfinite(actual).all().item())
    actual_f = actual.float()
    reference_f = reference.float()
    torch.testing.assert_close(actual_f, reference_f, rtol=0.02, atol=4.0)
    mean_rel = (
        actual_f - reference_f
    ).abs().mean() / reference_f.abs().mean().clamp_min(1.0e-6)
    test.assertLess(mean_rel.item(), 0.01)


class Int4LargeRulesTest(unittest.TestCase):
    def test_bucket_set_and_registered_ops(self):
        self.assertEqual(SUPPORTED_BUCKETS, (1, 2, 3, 4, 8, 16, 32, 64))
        self.assertEqual(INT4_QUANTIZED_GEMM.buckets, SUPPORTED_BUCKETS)
        for bucket in LARGE_BUCKETS:
            self.assertTrue(hasattr(torch.ops.triton, f"int4_quantized_gemm_m{bucket}"))

    def test_generic_tile_space_and_candidate_counts(self):
        expected_counts = {8: 18, 16: 36, 32: 54, 64: 72}
        for bucket in LARGE_BUCKETS:
            configs = int4_autotune_configs(bucket)
            seen = {
                (
                    config.kwargs["BLOCK_M"],
                    config.kwargs["BLOCK_N"],
                    config.num_warps,
                    config.kwargs["PIPELINE_STAGES"],
                    config.num_stages,
                )
                for config in configs
            }
            expected = {
                (block_m, block_n, warps, stages, stages)
                for block_m in TILE_BLOCK_M_CHOICES
                if block_m <= bucket
                for block_n in TILE_BLOCK_N_CHOICES
                for warps in TILE_WARP_CHOICES
                for stages in (1, 2, 3)
            }
            with self.subTest(bucket=bucket):
                self.assertEqual(len(configs), expected_counts[bucket])
                self.assertEqual(seen, expected)

    def test_stage_pruning_for_all_required_k(self):
        expected_stages = {256: 1, 512: 2, 6656: 3}
        block_m_counts = {8: 1, 16: 2, 32: 3, 64: 4}
        for bucket in LARGE_BUCKETS:
            configs = int4_autotune_configs(bucket)
            for k, stages in expected_stages.items():
                kept = int4_kernel._prune_tiles(configs, {"K": k}, SPLIT_K=1)
                with self.subTest(bucket=bucket, k=k):
                    self.assertEqual(len(kept), block_m_counts[bucket] * 3 * 2 * stages)
                    self.assertEqual(
                        max(config.kwargs["PIPELINE_STAGES"] for config in kept),
                        stages,
                    )

    def test_tile_split_needs_one_super_block_per_split(self):
        for k in (256, 512, 5376, 6656):
            for split in TILE_SPLIT_K_CANDIDATES:
                weights_k = k
                with self.subTest(k=k, split=split):
                    if split <= weights_k // 256:
                        check_split_k(split, weights_k, 256)
                    else:
                        with self.assertRaises(InvalidLaunchParam):
                            check_split_k(split, weights_k, 256)


@unittest.skipUnless(torch.cuda.is_available(), "CUDA required")
class Int4LargeLegalityTest(unittest.TestCase):
    def _args(self, m=8, n=37, k=512):
        return [
            torch.randn(m, k, dtype=torch.bfloat16, device="cuda"),
            *_packed(n, k, seed=m + n + k),
            GROUP_SIZE,
        ]

    def _expect_unsupported(self, bucket, args, pattern):
        self.assertFalse(INT4_QUANTIZED_GEMM.supports(bucket, *args))
        with self.assertRaisesRegex(RuntimeError, pattern):
            INT4_QUANTIZED_GEMM.validate(bucket, *args)

    def test_equal_and_smaller_static_m_are_supported(self):
        for bucket in LARGE_BUCKETS:
            for m in (bucket, SMALLER_M[bucket], 1):
                args = self._args(m=m)
                with self.subTest(bucket=bucket, m=m):
                    self.assertTrue(INT4_QUANTIZED_GEMM.supports(bucket, *args))
                    INT4_QUANTIZED_GEMM.validate(bucket, *args)

    def test_static_m_must_be_positive_and_not_exceed_bucket(self):
        for bucket in LARGE_BUCKETS:
            for m in (0, bucket + 1):
                self._expect_unsupported(
                    bucket,
                    self._args(m=m),
                    rf"static M must be within \[1, {bucket}\]",
                )

    def test_large_bucket_uses_existing_int4_contract(self):
        cases = (
            (lambda args: args.__setitem__(0, args[0].half()), "activation must be"),
            (lambda args: args.__setitem__(1, args[1].float()), "qdata must be"),
            (lambda args: args.__setitem__(2, args[2].int()), "scale codes must be"),
            (lambda args: args.__setitem__(3, args[3].float()), "scale_step must be"),
            (lambda args: args.__setitem__(6, 64), "group_size must be"),
        )
        for mutate, pattern in cases:
            args = self._args()
            mutate(args)
            self._expect_unsupported(8, args, pattern)

    def test_invalid_k_and_weight_shapes(self):
        n, k = 37, 288
        args = [
            torch.randn(8, k, dtype=torch.bfloat16, device="cuda"),
            torch.zeros(n, k // 2, dtype=torch.uint8, device="cuda"),
            torch.zeros(n, k // 32, dtype=torch.uint8, device="cuda"),
            torch.zeros(n, k // 256, dtype=torch.float16, device="cuda"),
            torch.zeros(n, k // 32, dtype=torch.uint8, device="cuda"),
            torch.zeros(n, k // 256, dtype=torch.float16, device="cuda"),
            GROUP_SIZE,
        ]
        self._expect_unsupported(8, args, "K must be a multiple of 256")
        args = self._args()
        args[2] = args[2][:, :-1].contiguous()
        self._expect_unsupported(8, args, "scale/zero shape")


def _selected_ints(name, defaults):
    value = os.environ.get(name)
    if not value:
        return defaults
    return tuple(int(piece) for piece in value.split(","))


# Configs checked per (bucket, N, K, split) by default: an evenly spaced sample
# keeps each test under the CI time limit. INT4_LARGE_TEST_ALL_CONFIGS=1 runs
# every legal config.
_SAMPLED_CONFIGS = 6


def _sampled(configs):
    if (
        os.environ.get("INT4_LARGE_TEST_ALL_CONFIGS") == "1"
        or len(configs) <= _SAMPLED_CONFIGS
    ):
        return configs
    step = len(configs) / _SAMPLED_CONFIGS
    return [configs[int(i * step)] for i in range(_SAMPLED_CONFIGS)]


@unittest.skipUnless(torch.cuda.is_available(), "CUDA required")
class Int4LargeNumericsTest(unittest.TestCase):
    def _check_every_config(self, bucket):
        n_values = _selected_ints("INT4_LARGE_TEST_N", N_TAILS)
        k_values = _selected_ints("INT4_LARGE_TEST_K", K_VALUES)
        for n in n_values:
            for k in k_values:
                weights = _packed(n, k, seed=n + k)
                # The direct kernel and the split-K path with the most splits K allows.
                splits = sorted(
                    {1, max(s for s in TILE_SPLIT_K_CANDIDATES if s <= k // 256)}
                )
                for split_k in splits:
                    configs = int4_kernel._prune_tiles(
                        int4_autotune_configs(bucket),
                        {"K": k},
                        SPLIT_K=split_k,
                    )
                    configs = _sampled(configs)
                    for m in (bucket, SMALLER_M[bucket]):
                        x = torch.randn(m, k, dtype=torch.bfloat16, device="cuda")
                        reference = _dequant_matmul(x, *weights, GROUP_SIZE)
                        for config in configs:
                            with self.subTest(
                                n=n,
                                k=k,
                                bucket=bucket,
                                m=m,
                                split_k=split_k,
                                config=str(config),
                            ):
                                actual = int4_kernel._launch_w4a16(
                                    bucket,
                                    x,
                                    *weights,
                                    GROUP_SIZE,
                                    split_k=split_k,
                                    config=config,
                                )
                                self.assertEqual(actual.shape, (m, n))
                                _check_close(self, actual, reference)
                del weights
                torch.cuda.empty_cache()

    def test_bucket_8_configs_m_tail_n_tail_and_k(self):
        self._check_every_config(8)

    def test_bucket_16_configs_m_tail_n_tail_and_k(self):
        self._check_every_config(16)

    def test_bucket_32_configs_m_tail_n_tail_and_k(self):
        self._check_every_config(32)

    def test_bucket_64_configs_m_tail_n_tail_and_k(self):
        self._check_every_config(64)

    def test_production_ops_match_reference_for_n_tails(self):
        for n in N_TAILS:
            weights = _packed(n, 512, seed=700 + n)
            for bucket in LARGE_BUCKETS:
                m = SMALLER_M[bucket]
                x = torch.randn(m, 512, dtype=torch.bfloat16, device="cuda")
                actual = INT4_QUANTIZED_GEMM.op(bucket)(x, *weights, GROUP_SIZE)
                _check_close(
                    self,
                    actual,
                    _dequant_matmul(x, *weights, GROUP_SIZE),
                )

    def test_dynamic_m_compiled_for_buckets_8_and_64(self):
        for bucket, runtime_m in ((8, (6, 8)), (64, (48, 64))):
            torch._dynamo.reset()
            n, k = 257, 512
            weights = _packed(n, k, seed=900 + bucket)

            def linear(x, bucket=bucket, weights=weights):
                return INT4_QUANTIZED_GEMM.op(bucket)(x, *weights, GROUP_SIZE)

            compiled = torch.compile(linear, fullgraph=True)
            for m in runtime_m:
                with self.subTest(bucket=bucket, m=m):
                    x = torch.randn(m, k, dtype=torch.bfloat16, device="cuda")
                    torch._dynamo.mark_dynamic(x, 0, min=1, max=bucket)
                    actual = compiled(x)
                    self.assertEqual(actual.shape, (m, n))
                    _check_close(
                        self,
                        actual,
                        _dequant_matmul(x, *weights, GROUP_SIZE),
                    )

    def test_split_replay_is_deterministic(self):
        bucket, n, k = 64, 257, 6656
        weights = _packed(n, k, seed=111)
        x = torch.randn(bucket, k, dtype=torch.bfloat16, device="cuda")
        config = int4_kernel._prune_tiles(
            int4_autotune_configs(bucket), {"K": k}, SPLIT_K=8
        )[0]
        first = int4_kernel._launch_w4a16(
            bucket,
            x,
            *weights,
            GROUP_SIZE,
            split_k=8,
            config=config,
        )
        second = int4_kernel._launch_w4a16(
            bucket,
            x,
            *weights,
            GROUP_SIZE,
            split_k=8,
            config=config,
        )
        self.assertTrue(torch.equal(first, second))


if __name__ == "__main__":
    unittest.main()
