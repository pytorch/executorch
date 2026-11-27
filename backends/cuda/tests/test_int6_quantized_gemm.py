# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Tests for the decode-sized INT6 Triton GEMM (triton/kernels/int6_quantized_gemm.py).

Every autotune candidate of every bucket must match the W6A16 dequant +
F.linear reference, statically and for a compiled dynamic M; every legality
rule makes ``supports`` False and ``validate`` raise.

    python -m pytest backends/cuda/tests/test_int6_quantized_gemm.py -v
"""

import unittest
from unittest import mock

import torch
import triton
from executorch.backends.cuda.autotune.launch_params import (
    autotune_launch_params,
    clear_launch_param_cache,
    InvalidLaunchParam,
)
from executorch.backends.cuda.quantize_op_dispatch.int6_dispatch import _unit_dq_mm_int6
from executorch.backends.cuda.tests.test_int6_dispatch import _make_int6_tensor
from executorch.backends.cuda.triton.kernels import int6_quantized_gemm as int6_kernel
from executorch.backends.cuda.triton.kernels.int6_quantized_gemm import (
    int6_autotune_configs,
    INT6_QUANTIZED_GEMM,
    SUPPORTED_BUCKETS,
)
from executorch.backends.cuda.triton.kernels.quantized_gemm_utils import (
    SPLIT_K_CANDIDATES,
)

GROUP_SIZE = 16
# The W6A8 DP4A decode buckets; 8..64 are the W6A16 tile buckets
# (test_int6_large_quantized_gemm).
W6A8_BUCKETS = (1, 2, 3, 4)


def _packed(n: int, k: int, seed: int = 0):
    torch.manual_seed(seed)
    weight, _, _ = _make_int6_tensor(n, k, GROUP_SIZE)
    return tuple(
        tensor.cuda() for tensor in (weight.ql, weight.qh, weight.scale, weight.steps)
    )


def _check_close(test: unittest.TestCase, out: torch.Tensor, ref: torch.Tensor) -> None:
    test.assertTrue(torch.isfinite(out).all())
    torch.testing.assert_close(out.float(), ref.float(), rtol=0.02, atol=4.0)
    mean_rel = (out.float() - ref.float()).abs().mean() / ref.float().abs().mean()
    test.assertLess(mean_rel.item(), 0.01)


def _single_config(config: triton.Config):
    return triton.autotune(configs=[config], key=["N", "K", "SPLIT_K"])(
        int6_kernel._int6_w6a8_bucket_kernel
    )


class Int6QuantizedGemmRulesTest(unittest.TestCase):
    def test_candidates_cover_the_generic_space(self) -> None:
        configs = int6_autotune_configs()
        seen = {
            (
                config.kwargs["K_TILE"],
                config.kwargs["BLOCK_N"],
                config.num_warps,
                config.kwargs["PIPELINE_STAGES"],
                config.num_stages,
            )
            for config in configs
        }
        self.assertEqual(len(configs), 24)
        self.assertEqual(
            seen,
            {
                (k_tile, warps, warps, stages, stages)
                for k_tile in (32, 16)
                for warps in (1, 2, 4, 8)
                for stages in (1, 2, 3)
            },
        )

    def test_prune_drops_stages_above_main_loop_trips(self) -> None:
        configs = int6_autotune_configs()
        cases = (
            (256, 1, 1, 8),
            (2048, 1, 3, 20),
            (2048, 2, 2, 12),
            (8192, 1, 3, 24),
            (8192, 8, 2, 12),
        )
        for k, split_k, max_stages, expected_count in cases:
            with self.subTest(k=k, split_k=split_k):
                kept = int6_kernel._prune(configs, {"K": k}, SPLIT_K=split_k)
                self.assertEqual(
                    max(c.kwargs["PIPELINE_STAGES"] for c in kept),
                    max_stages,
                )
                self.assertEqual(len(kept), expected_count)


class Int6QuantizedGemmLegalityTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        if not torch.cuda.is_available():
            raise unittest.SkipTest("CUDA required")

    def _args(self, m: int = 1, n: int = 64, k: int = 512):
        x = torch.randn(m, k, dtype=torch.bfloat16, device="cuda")
        return [x, *_packed(n, k, seed=3), GROUP_SIZE]

    def _expect_unsupported(self, bucket, args, pattern) -> None:
        self.assertFalse(INT6_QUANTIZED_GEMM.supports(bucket, *args))
        with self.assertRaisesRegex(RuntimeError, pattern):
            INT6_QUANTIZED_GEMM.validate(bucket, *args)

    def test_valid_inputs_and_raw_byte_dtypes_are_supported(self) -> None:
        for bucket in SUPPORTED_BUCKETS:
            args = self._args(m=bucket)
            self.assertTrue(INT6_QUANTIZED_GEMM.supports(bucket, *args))
            INT6_QUANTIZED_GEMM.validate(bucket, *args)

        args = self._args()
        args[1] = args[1].view(torch.int8)
        args[2] = args[2].view(torch.int8)
        args[3] = args[3].view(torch.uint8)
        self.assertTrue(INT6_QUANTIZED_GEMM.supports(1, *args))
        INT6_QUANTIZED_GEMM.validate(1, *args)

    def test_unsupported_bucket(self) -> None:
        args = self._args()
        self.assertFalse(INT6_QUANTIZED_GEMM.supports(5, *args))
        with self.assertRaisesRegex(
            RuntimeError, "unsupported int6_quantized_gemm bucket 5"
        ):
            INT6_QUANTIZED_GEMM.validate(5, *args)

    def test_each_dtype_rule(self) -> None:
        cases = {
            "activation": (0, torch.float16, "activation must be bfloat16"),
            "ql": (1, torch.float32, "ql must be"),
            "qh": (2, torch.float32, "qh must be"),
            "scale": (3, torch.int32, "scale codes must be"),
            "steps": (4, torch.float32, "steps must be float16"),
        }
        for name, (index, dtype, pattern) in cases.items():
            with self.subTest(rule=name):
                args = self._args()
                args[index] = args[index].to(dtype)
                self._expect_unsupported(1, args, pattern)

    def test_each_rank_rule(self) -> None:
        for index, name in enumerate(("x", "ql", "qh", "scale", "steps")):
            with self.subTest(tensor=name):
                args = self._args()
                args[index] = args[index].unsqueeze(0)
                self._expect_unsupported(1, args, "rank-2")

    def test_group_size_and_static_m_rules(self) -> None:
        args = self._args()
        args[5] = 32
        self._expect_unsupported(1, args, "group_size must be 16")

        args = self._args(m=2)
        self._expect_unsupported(1, args, "static M must be within")

    def test_k_must_be_positive_static_multiple_of_256(self) -> None:
        n, k = 32, 384
        args = [
            torch.randn(1, k, dtype=torch.bfloat16, device="cuda"),
            torch.zeros(n, k // 2, dtype=torch.uint8, device="cuda"),
            torch.zeros(n, k // 4, dtype=torch.uint8, device="cuda"),
            torch.zeros(n, k // GROUP_SIZE, dtype=torch.int8, device="cuda"),
            torch.zeros(n, k // 256, dtype=torch.float16, device="cuda"),
            GROUP_SIZE,
        ]
        self._expect_unsupported(1, args, "K must be a multiple of 256")

        args = [
            torch.empty(1, 0, dtype=torch.bfloat16, device="cuda"),
            torch.empty(n, 0, dtype=torch.uint8, device="cuda"),
            torch.empty(n, 0, dtype=torch.uint8, device="cuda"),
            torch.empty(n, 0, dtype=torch.int8, device="cuda"),
            torch.empty(n, 0, dtype=torch.float16, device="cuda"),
            GROUP_SIZE,
        ]
        self._expect_unsupported(1, args, "K must be positive")

    def test_symbolic_k_and_unprovable_m_are_unsupported(self) -> None:
        from torch._subclasses.fake_tensor import FakeTensorMode
        from torch.fx.experimental.symbolic_shapes import ShapeEnv

        shape_env = ShapeEnv()
        mode = FakeTensorMode(shape_env=shape_env)
        with mode:
            weights = [
                torch.empty(32, 256, dtype=torch.uint8, device="cuda"),
                torch.empty(32, 128, dtype=torch.uint8, device="cuda"),
                torch.empty(32, 32, dtype=torch.int8, device="cuda"),
                torch.empty(32, 2, dtype=torch.float16, device="cuda"),
            ]
            symbolic_k = shape_env.create_unbacked_symint()
            dynamic_k_args = [
                torch.empty(1, symbolic_k, dtype=torch.bfloat16, device="cuda"),
                *weights,
                GROUP_SIZE,
            ]
            self._expect_unsupported(1, dynamic_k_args, "K must be static")

            symbolic_m = shape_env.create_unbacked_symint()
            dynamic_m_args = [
                torch.empty(symbolic_m, 512, dtype=torch.bfloat16, device="cuda"),
                *weights,
                GROUP_SIZE,
            ]
            self._expect_unsupported(4, dynamic_m_args, "dynamic M is not provably")

    def test_each_shape_rule(self) -> None:
        cases = {
            "ql": (1, "ql K/2 mismatch"),
            "qh": (2, "qh shape"),
            "scale": (3, "scale shape"),
            "steps": (4, "steps shape"),
        }
        for name, (index, pattern) in cases.items():
            with self.subTest(tensor=name):
                args = self._args()
                args[index] = args[index][:, :-1].contiguous()
                self._expect_unsupported(1, args, pattern)

    def test_contiguous_and_device_rules(self) -> None:
        for index, name in enumerate(("x", "ql", "qh", "scale", "steps")):
            with self.subTest(contiguous=name):
                args = self._args()
                args[index] = torch.stack((args[index], args[index]), dim=-1)[..., 0]
                self.assertFalse(args[index].is_contiguous())
                self._expect_unsupported(1, args, "contiguous")

        args = self._args()
        args[3] = args[3].cpu()
        self._expect_unsupported(1, args, "same device")

        args = self._args()
        for index in range(5):
            args[index] = args[index].cpu()
        self._expect_unsupported(1, args, "CUDA device")

    def test_fake_inputs_are_supported(self) -> None:
        from torch._subclasses.fake_tensor import FakeTensorMode

        real = self._args()
        mode = FakeTensorMode()
        fake = [mode.from_tensor(t.cpu()) for t in real[:5]] + [GROUP_SIZE]
        self.assertTrue(INT6_QUANTIZED_GEMM.supports(1, *fake))
        INT6_QUANTIZED_GEMM.validate(1, *fake)

    def test_op_validates_before_launching(self) -> None:
        args = self._args(m=2)
        with self.assertRaisesRegex(RuntimeError, "static M must be within"):
            INT6_QUANTIZED_GEMM.op(1)(*args)


class Int6QuantizedGemmTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        if not torch.cuda.is_available():
            raise unittest.SkipTest("CUDA required")

    def test_ops_are_registered_per_bucket(self) -> None:
        self.assertEqual(SUPPORTED_BUCKETS, (1, 2, 3, 4, 8, 16, 32, 64))
        for bucket in SUPPORTED_BUCKETS:
            self.assertTrue(hasattr(torch.ops.triton, f"int6_quantized_gemm_m{bucket}"))

    def test_every_static_candidate_matches_w6a16_reference(self) -> None:
        shapes = ((37, 256), (53, 512), (37, 5376))
        for n, k in shapes:
            weights = _packed(n, k, seed=n + k)
            for bucket in W6A8_BUCKETS:
                x = torch.randn(bucket, k, dtype=torch.bfloat16, device="cuda")
                ref = _unit_dq_mm_int6(x, *weights, GROUP_SIZE)
                for config in int6_autotune_configs():
                    with self.subTest(
                        n=n,
                        k=k,
                        bucket=bucket,
                        config=str(config),
                    ):
                        single = _single_config(config)
                        with mock.patch.dict(
                            int6_kernel._BUCKET_KERNELS, {bucket: single}
                        ):
                            out = int6_kernel._launch(bucket, x, *weights, GROUP_SIZE)
                        self.assertEqual(out.shape, (bucket, n))
                        self.assertEqual(out.dtype, torch.bfloat16)
                        _check_close(self, out, ref)

    def test_every_candidate_serves_dynamic_m(self) -> None:
        for n, k in ((37, 256), (37, 512), (37, 5376)):
            weights = _packed(n, k, seed=71 + k)
            for bucket in (3, 4):
                for config in int6_autotune_configs():
                    with self.subTest(n=n, k=k, bucket=bucket, config=str(config)):
                        torch._dynamo.reset()
                        single = _single_config(config)
                        with mock.patch.dict(
                            int6_kernel._BUCKET_KERNELS, {bucket: single}
                        ):
                            compiled = torch.compile(
                                lambda x, bucket=bucket, weights=weights: INT6_QUANTIZED_GEMM.op(
                                    bucket
                                )(
                                    x, *weights, GROUP_SIZE
                                ),
                                fullgraph=True,
                            )
                            for m in range(2, bucket + 1):
                                x = torch.randn(
                                    m, k, dtype=torch.bfloat16, device="cuda"
                                )
                                torch._dynamo.mark_dynamic(x, 0, min=2, max=bucket)
                                out = compiled(x)
                                self.assertEqual(out.shape, (m, n))
                                _check_close(
                                    self,
                                    out,
                                    _unit_dq_mm_int6(x, *weights, GROUP_SIZE),
                                )

    def test_dynamic_bucket4_exact_rows_matches_reference(self) -> None:
        n, k = 37, 512
        weights = _packed(n, k, seed=79)
        config = next(
            config
            for config in int6_autotune_configs()
            if config.kwargs["K_TILE"] == 32
            and config.kwargs["BLOCK_N"] == 1
            and config.kwargs["PIPELINE_STAGES"] == 1
        )
        single = _single_config(config)
        with mock.patch.dict(int6_kernel._BUCKET_KERNELS, {4: single}):
            compiled = torch.compile(
                lambda x: INT6_QUANTIZED_GEMM.op(4)(x, *weights, GROUP_SIZE),
                fullgraph=True,
            )
            for m in (2, 3, 4):
                x = torch.randn(m, k, dtype=torch.bfloat16, device="cuda")
                torch._dynamo.mark_dynamic(x, 0, min=2, max=4)
                out = compiled(x)
                self.assertEqual(out.shape, (m, n))
                _check_close(
                    self,
                    out,
                    _unit_dq_mm_int6(x, *weights, GROUP_SIZE),
                )

    def test_raw_byte_storage_dtypes_execute_correctly(self) -> None:
        original = _packed(37, 512, seed=81)
        weights = list(original)
        weights[0] = weights[0].view(torch.int8)
        weights[1] = weights[1].view(torch.int8)
        weights[2] = weights[2].view(torch.uint8)
        x = torch.randn(1, 512, dtype=torch.bfloat16, device="cuda")
        out = INT6_QUANTIZED_GEMM.op(1)(x, *weights, GROUP_SIZE)
        ref = _unit_dq_mm_int6(x, *original, GROUP_SIZE)
        _check_close(self, out, ref)

    def test_dynamic_m_is_timed_on_the_dynamic_path(self) -> None:
        """Split-K timing for a dynamic M runs the DYNAMIC_M kernel, and is cached
        apart from a static M of the same size."""
        from torch._dynamo.source import ConstantSource
        from torch._subclasses.fake_tensor import FakeTensorMode
        from torch.fx.experimental.symbolic_shapes import (
            DimDynamic,
            ShapeEnv,
            StatelessSymbolicContext,
        )

        weights = _packed(37, 512, seed=34)
        real = int6_kernel.launch_split_k_gemm
        timed = []

        def recording(kernel, **kw):
            if isinstance(kw["m"], int):
                timed.append(kw["DYNAMIC_M"])
            return real(kernel, **kw)

        x = torch.randn(4, 512, dtype=torch.bfloat16, device="cuda")
        clear_launch_param_cache()
        with mock.patch.object(
            int6_kernel, "launch_split_k_gemm", side_effect=recording
        ), autotune_launch_params() as stats:
            with FakeTensorMode(shape_env=ShapeEnv()) as mode:
                fake_x = mode.from_tensor(
                    x,
                    source=ConstantSource("x"),
                    symbolic_context=StatelessSymbolicContext(
                        dynamic_sizes=[DimDynamic.DYNAMIC, DimDynamic.STATIC]
                    ),
                )
                self.assertNotIsInstance(fake_x.shape[0], int)
                int6_kernel._launch(
                    4, fake_x, *(mode.from_tensor(w) for w in weights), GROUP_SIZE
                )
            dynamic_measured = stats.measured
            int6_kernel._launch(4, x, *weights, GROUP_SIZE)
        self.assertEqual(dynamic_measured, 1)
        self.assertEqual(stats.measured, 2)
        self.assertEqual(timed[0], True)
        self.assertEqual(timed[-1], False)

    def test_every_split_matches_reference(self) -> None:
        weights = _packed(37, 768, seed=101)
        for bucket in W6A8_BUCKETS:
            x = torch.randn(bucket, 768, dtype=torch.bfloat16, device="cuda")
            ref = _unit_dq_mm_int6(x, *weights, GROUP_SIZE)
            for split in SPLIT_K_CANDIDATES:
                if split > 768 // 256:
                    with self.assertRaises(InvalidLaunchParam):
                        int6_kernel._launch(
                            bucket, x, *weights, GROUP_SIZE, split_k=split
                        )
                    continue
                with self.subTest(bucket=bucket, split=split):
                    out = int6_kernel._launch(
                        bucket, x, *weights, GROUP_SIZE, split_k=split
                    )
                    _check_close(self, out, ref)


if __name__ == "__main__":
    unittest.main()
