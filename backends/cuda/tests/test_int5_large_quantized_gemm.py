# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""INT5 BF16 tile, split reduction, and export routing regression checks."""

import unittest

import torch
import torch.nn.functional as F
import triton

from executorch.backends.cuda.autotune.launch_params import InvalidLaunchParam
from executorch.backends.cuda.dp4a_planar_int5_tensor import (
    CudaDp4aPlanarInt5Tensor,
    pack_int5,
)
from executorch.backends.cuda.quantize_op_dispatch.int5_dispatch import (
    _dequant_matmul_int5,
)
from executorch.backends.cuda.triton.kernels import int5_quantized_gemm as int5_kernel
from executorch.backends.cuda.triton.kernels.int5_quantized_gemm import (
    _launch_w5a16,
    int5_autotune_configs,
    INT5_QUANTIZED_GEMM,
    TILE_SPLIT_K_CANDIDATES,
)
from torch._library.triton import set_wrap_triton_enabled


@unittest.skipUnless(torch.cuda.is_available(), "CUDA required")
class Int5LargeTest(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(19)
        self.reduced_precision = (
            torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction
        )
        torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = False

    def tearDown(self):
        torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = (
            self.reduced_precision
        )

    def _weights(self, n, k, group_size=32):
        q = torch.randint(0, 32, (n, k), dtype=torch.uint8, device="cuda")
        ql, qh = pack_int5(q)
        scale = torch.randint(
            1, 255, (n, k // group_size), dtype=torch.uint8, device="cuda"
        )
        steps = torch.full((n, k // 256), 0.0001, dtype=torch.float16, device="cuda")
        return ql, qh, scale, steps, torch.full_like(scale, 16), torch.ones_like(steps)

    def test_direct_and_split_tiles_with_tail_dimensions(self):
        config = triton.Config(
            {"BLOCK_M": 16, "BLOCK_N": 64, "PIPELINE_STAGES": 1},
            num_warps=4,
            num_stages=1,
        )
        cases = (
            (7, 8, 129, 768, 1),
            (16, 16, 129, 768, 1),
            (16, 16, 129, 768, 2),
            (16, 16, 129, 1280, 4),
            (8, 8, 129, 4096, 16),
            (32, 32, 257, 512, 1),
            (64, 64, 257, 512, 2),
        )
        for m, bucket, n, k, split in cases:
            with self.subTest(m=m, bucket=bucket, n=n, k=k, split=split):
                weights = self._weights(n, k)
                x = torch.randn(m, k, device="cuda", dtype=torch.bfloat16)
                ref = _dequant_matmul_int5(x, *weights, 32)
                out = _launch_w5a16(
                    bucket, x, *weights, 32, split_k=split, config=config
                )
                torch.testing.assert_close(out, ref, rtol=0.012, atol=0.025)
                self.assertTrue(torch.isfinite(out).all())
                replay = _launch_w5a16(
                    bucket, x, *weights, 32, split_k=split, config=config
                )
                self.assertTrue(torch.equal(out, replay))

    def test_large_config_space_has_one_bucket64_k64_candidate(self):
        expected_counts = {8: 18, 16: 36, 32: 54, 64: 73}
        for bucket, expected in expected_counts.items():
            configs = int5_autotune_configs(bucket)
            self.assertEqual(len(configs), expected)
            k64 = [config for config in configs if config.kwargs["BLOCK_K"] == 64]
            self.assertEqual(len(k64), 1 if bucket == 64 else 0)
            self.assertTrue(
                all(
                    config.kwargs["BLOCK_K"] == 256
                    for config in configs
                    if config not in k64
                )
            )

    def test_large_stage_pruning_covers_k_and_split(self):
        for bucket in (8, 16, 32, 64):
            configs = int5_autotune_configs(bucket)
            for k in (256, 512, 6656):
                for split in TILE_SPLIT_K_CANDIDATES:
                    if split > k // 256:
                        continue
                    kept = int5_kernel._prune_tiles(configs, {"K": k}, SPLIT_K=split)
                    max_stages = max(1, triton.cdiv(k // 256, split))
                    self.assertTrue(
                        all(
                            config.kwargs["PIPELINE_STAGES"] <= max_stages
                            for config in kept
                        )
                    )

    def test_bucket64_k64_direct_and_split_are_exact_and_deterministic(self):
        n, k = 129, 1024
        weights = self._weights(n, k)
        x = torch.randn(64, k, device="cuda", dtype=torch.bfloat16)
        ref = _dequant_matmul_int5(x, *weights, 32)
        config = triton.Config(
            {
                "BLOCK_M": 64,
                "BLOCK_N": 64,
                "BLOCK_K": 64,
                "PIPELINE_STAGES": 1,
            },
            num_warps=4,
            num_stages=1,
        )
        for split in (1, 4):
            with self.subTest(split=split):
                out = _launch_w5a16(64, x, *weights, 32, split_k=split, config=config)
                torch.testing.assert_close(out, ref, rtol=0.012, atol=0.025)
                replay = _launch_w5a16(
                    64, x, *weights, 32, split_k=split, config=config
                )
                self.assertTrue(torch.equal(out, replay))

    def test_bucket64_k64_split_cuda_graph_replay_is_deterministic(self):
        n, k = 129, 1024
        weights = self._weights(n, k)
        x = torch.randn(64, k, device="cuda", dtype=torch.bfloat16)
        config = triton.Config(
            {
                "BLOCK_M": 64,
                "BLOCK_N": 64,
                "BLOCK_K": 64,
                "PIPELINE_STAGES": 1,
            },
            num_warps=4,
            num_stages=1,
        )

        def run():
            return _launch_w5a16(64, x, *weights, 32, split_k=4, config=config)

        outer_stream = torch.cuda.current_stream()
        stream = torch.cuda.Stream()
        stream.wait_stream(outer_stream)
        with torch.cuda.stream(stream), set_wrap_triton_enabled(False):
            run()
            stream.synchronize()
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph, stream=stream):
                captured = run()
            graph.replay()
            stream.synchronize()
            first = captured.clone()
            graph.replay()
            stream.synchronize()
            second = captured.clone()
        outer_stream.wait_stream(stream)
        self.assertTrue(torch.equal(first, second))

    def test_large_split_candidates_and_legality(self):
        self.assertEqual(TILE_SPLIT_K_CANDIDATES, (1, 2, 4, 8, 16))
        weights = self._weights(129, 2048)
        x = torch.randn(8, 2048, device="cuda", dtype=torch.bfloat16)
        with self.assertRaises(InvalidLaunchParam):
            _launch_w5a16(8, x, *weights, 32, split_k=16)

    def test_large_buckets_reject_group_size_64(self):
        x = torch.randn(16, 512, device="cuda", dtype=torch.bfloat16)
        self.assertFalse(
            INT5_QUANTIZED_GEMM.supports(16, x, *self._weights(129, 512, 64), 64)
        )
        self.assertTrue(
            INT5_QUANTIZED_GEMM.supports(16, x, *self._weights(129, 512), 32)
        )

    def test_lm_head_logits_top1_and_cosine(self):
        n, k = 202048, 6656
        generator = torch.Generator(device="cuda").manual_seed(4242)
        ql = torch.randint(
            0, 256, (n, k // 2), dtype=torch.uint8, device="cuda", generator=generator
        )
        qh = torch.randint(
            0, 256, (n, k // 8), dtype=torch.uint8, device="cuda", generator=generator
        )
        scale = torch.randint(
            1,
            128,
            (n, k // 32),
            dtype=torch.uint8,
            device="cuda",
            generator=generator,
        )
        zero = torch.randint(
            0,
            128,
            (n, k // 32),
            dtype=torch.uint8,
            device="cuda",
            generator=generator,
        )
        steps = torch.full((n, k // 256), 0.001, dtype=torch.float16, device="cuda")
        zero_steps = torch.full_like(steps, 0.25)
        weights = (ql, qh, scale, steps, zero, zero_steps)
        small_agreements = []
        large_agreements = []
        small_min_cosine = 1.0
        large_min_cosine = 1.0
        for outliers in (False, True):
            for m in (1, 2, 3, 4, 8, 16, 32, 64):
                x = torch.randn(
                    m, k, device="cuda", dtype=torch.bfloat16, generator=generator
                )
                if outliers:
                    channels = torch.tensor(
                        (0, k // 3, 2 * k // 3, k - 1), device="cuda"
                    )
                    x[:, channels] *= torch.tensor(
                        (50.0, 35.0, 25.0, 20.0),
                        device="cuda",
                        dtype=torch.bfloat16,
                    )
                actual = INT5_QUANTIZED_GEMM.op(m)(x, *weights, 32).float()
                reference = _dequant_matmul_int5(x, *weights, 32).float()
                agreements = actual.argmax(dim=1) == reference.argmax(dim=1)
                (small_agreements if m <= 4 else large_agreements).append(agreements)
                cosine = F.cosine_similarity(actual, reference, dim=1).min().item()
                if m <= 4:
                    small_min_cosine = min(small_min_cosine, cosine)
                else:
                    large_min_cosine = min(large_min_cosine, cosine)
        small_agreement = torch.cat(small_agreements).float().mean().item()
        large_agreement = torch.cat(large_agreements).float().mean().item()
        # W5A8 is the pre-existing activation-quantized path; the exact W5A16
        # changes in this file must meet the stricter large-bucket gate.
        self.assertGreaterEqual(small_agreement, 0.90)
        self.assertGreaterEqual(large_agreement, 0.999)
        self.assertGreaterEqual(small_min_cosine, 0.9998)
        self.assertGreaterEqual(large_min_cosine, 0.9999)

    def test_m16_export_uses_fused_op(self):
        n, k = 129, 512
        weights = self._weights(n, k)
        weight = CudaDp4aPlanarInt5Tensor(
            *weights, block_size=[1, 32], shape=torch.Size([n, k])
        )
        head = torch.nn.Linear(k, n, bias=False, device="meta")
        head.weight = torch.nn.Parameter(weight, requires_grad=False)
        x = torch.randn(1, 16, k, device="cuda", dtype=torch.bfloat16)
        exported = torch.export.export(head, (x,), strict=True)
        self.assertIn("int5_quantized_gemm_m16", str(exported.graph))


if __name__ == "__main__":
    unittest.main()
