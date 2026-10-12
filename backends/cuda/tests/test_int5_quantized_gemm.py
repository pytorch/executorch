# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Tests for the decode-sized INT5 Triton GEMM.

Every autotune candidate of every bucket is checked against the exact stored
Q5_K dequantization, statically and for symbolic dynamic M. Every legality rule
must make ``supports`` false and ``validate`` raise.

    PYTHONPATH=$PWD/executorch/src python -m pytest test_int5_quantized_gemm.py -v
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
from executorch.backends.cuda.quantize_op_dispatch.int5_dispatch import (
    _dequant_matmul_int5,
)
from executorch.backends.cuda.tests.test_int5_dispatch import _make_int5_tensor
from executorch.backends.cuda.triton.kernels import int5_quantized_gemm as int5_kernel
from executorch.backends.cuda.triton.kernels.int5_quantized_gemm import (
    int5_autotune_configs,
    INT5_QUANTIZED_GEMM,
    SUPPORTED_BUCKETS,
)
from executorch.backends.cuda.triton.kernels.quantized_gemm_utils import (
    SPLIT_K_CANDIDATES,
)

GROUP_SIZE = 32
_WEIGHT_FIELDS = (
    "ql",
    "qh",
    "scale",
    "scale_step",
    "zero_point",
    "zero_point_step",
)


def _packed(n: int, k: int, seed: int = 0, group_size: int = GROUP_SIZE):
    torch.manual_seed(seed)
    weight, _ = _make_int5_tensor(n, k, group_size)
    return tuple(getattr(weight, name).cuda() for name in _WEIGHT_FIELDS)


def _shape_args(
    m: int = 1,
    n: int = 64,
    k: int = 512,
    group_size: int = GROUP_SIZE,
    device: str = "cuda",
):
    return [
        torch.randn(m, k, dtype=torch.bfloat16, device=device),
        torch.zeros(n, k // 2, dtype=torch.uint8, device=device),
        torch.zeros(n, k // 8, dtype=torch.uint8, device=device),
        torch.zeros(n, k // group_size, dtype=torch.uint8, device=device),
        torch.zeros(n, k // 256, dtype=torch.float16, device=device),
        torch.zeros(n, k // group_size, dtype=torch.uint8, device=device),
        torch.zeros(n, k // 256, dtype=torch.float16, device=device),
        group_size,
    ]


def _check_close(test: unittest.TestCase, out: torch.Tensor, ref: torch.Tensor) -> None:
    test.assertTrue(torch.isfinite(out).all())
    torch.testing.assert_close(out.float(), ref.float(), rtol=0.02, atol=4.0)
    mean_rel = (out.float() - ref.float()).abs().mean() / ref.float().abs().mean()
    test.assertLess(mean_rel.item(), 0.01)


def _single_config(config: triton.Config):
    return triton.autotune(configs=[config], key=["N", "K", "SPLIT_K"])(
        int5_kernel._int5_w5a8_bucket_kernel
    )


class Int5QuantizedGemmRulesTest(unittest.TestCase):
    def test_candidates_cover_exact_shared_autotune_space(self) -> None:
        expected_rows = {
            1: (0, 1, 2),
            2: (0, 2, 3),
            3: (0, 3, 4),
            4: (0, 4),
        }
        for bucket in SUPPORTED_BUCKETS:
            with self.subTest(bucket=bucket):
                configs = int5_autotune_configs(bucket)
                seen = {
                    (
                        config.kwargs["ROWS"],
                        config.kwargs["BLOCK_N"],
                        config.num_warps,
                        config.kwargs["PIPELINE_STAGES"],
                        config.num_stages,
                    )
                    for config in configs
                }
                expected = {
                    (rows, block_n, block_n, stages, stages)
                    for rows in expected_rows[bucket]
                    for block_n in (1, 2, 4, 8)
                    for stages in (1, 2, 3)
                }
                self.assertEqual(len(configs), len(expected))
                self.assertEqual(seen, expected)

    def test_prune_drops_stages_above_main_loop_trips(self) -> None:
        cases = (
            (256, 1, 1),
            (2048, 1, 2),
            (2048, 2, 1),
            (8192, 1, 3),
            (8192, 8, 1),
        )
        row_implementations = {1: 3, 2: 3, 3: 3, 4: 2}
        for bucket in SUPPORTED_BUCKETS:
            configs = int5_autotune_configs(bucket)
            for k, split_k, max_stages in cases:
                with self.subTest(bucket=bucket, k=k, split_k=split_k):
                    kept = int5_kernel._prune(configs, {"K": k}, SPLIT_K=split_k)
                    self.assertEqual(
                        max(c.kwargs["PIPELINE_STAGES"] for c in kept),
                        max_stages,
                    )
                    self.assertEqual(
                        len(kept), row_implementations[bucket] * 4 * max_stages
                    )
                    self.assertTrue(
                        all(c.kwargs["PIPELINE_STAGES"] <= max_stages for c in kept)
                    )

    def test_buckets_and_ops_are_registered(self) -> None:
        self.assertEqual(SUPPORTED_BUCKETS, (1, 2, 3, 4))
        self.assertEqual(INT5_QUANTIZED_GEMM.buckets, SUPPORTED_BUCKETS)
        for bucket in SUPPORTED_BUCKETS:
            self.assertTrue(
                hasattr(torch.ops.triton, f"int5_quantized_gemm_m{bucket}"),
                bucket,
            )
        with self.assertRaisesRegex(
            RuntimeError, "unsupported int5_quantized_gemm bucket 5"
        ):
            INT5_QUANTIZED_GEMM.op(5)


class Int5QuantizedGemmLegalityTest(unittest.TestCase):
    """Every invalid input makes supports false and validate raise."""

    @classmethod
    def setUpClass(cls) -> None:
        if not torch.cuda.is_available():
            raise unittest.SkipTest("CUDA required")

    def _args(self, m: int = 1, n: int = 64, k: int = 512):
        x = torch.randn(m, k, dtype=torch.bfloat16, device="cuda")
        return [x, *_packed(n, k, seed=3), GROUP_SIZE]

    def _expect_unsupported(self, bucket: int, args, pattern: str) -> None:
        self.assertFalse(INT5_QUANTIZED_GEMM.supports(bucket, *args))
        with self.assertRaisesRegex(RuntimeError, pattern):
            INT5_QUANTIZED_GEMM.validate(bucket, *args)

    def test_valid_cuda_inputs_and_byte_dtypes_are_supported(self) -> None:
        for bucket in SUPPORTED_BUCKETS:
            args = self._args(m=bucket)
            self.assertTrue(INT5_QUANTIZED_GEMM.supports(bucket, *args))
            INT5_QUANTIZED_GEMM.validate(bucket, *args)

        for ql_dtype in (torch.uint8, torch.int8):
            for qh_dtype in (torch.uint8, torch.int8):
                with self.subTest(ql_dtype=ql_dtype, qh_dtype=qh_dtype):
                    args = self._args()
                    args[1] = args[1].view(ql_dtype)
                    args[2] = args[2].view(qh_dtype)
                    self.assertTrue(INT5_QUANTIZED_GEMM.supports(1, *args))
                    INT5_QUANTIZED_GEMM.validate(1, *args)

    def test_unsupported_bucket(self) -> None:
        self._expect_unsupported(
            5,
            self._args(),
            "unsupported int5_quantized_gemm bucket 5",
        )

    def test_each_dtype_rule(self) -> None:
        cases = {
            "activation": (0, torch.float16, "activation must be bfloat16"),
            "ql": (1, torch.float32, "ql must be uint8/int8"),
            "qh": (2, torch.float32, "qh must be uint8/int8"),
            "scale": (3, torch.int8, "scale codes must be uint8"),
            "scale_step": (4, torch.float32, "scale_step must be float16"),
            "zero": (5, torch.int8, "zero codes must be uint8"),
            "zero_point_step": (
                6,
                torch.float32,
                "zero_point_step must be float16",
            ),
        }
        for name, (index, dtype, pattern) in cases.items():
            with self.subTest(rule=name):
                args = self._args()
                args[index] = args[index].to(dtype)
                self._expect_unsupported(1, args, pattern)

    def test_every_tensor_must_be_rank_two(self) -> None:
        names = (
            "x",
            "ql",
            "qh",
            "scale",
            "scale_step",
            "zero",
            "zero_point_step",
        )
        for index, name in enumerate(names):
            with self.subTest(tensor=name):
                args = self._args()
                args[index] = args[index].unsqueeze(0)
                self._expect_unsupported(1, args, "rank-2")

    def test_group_size_contract(self) -> None:
        for group_size in (32, 64, 128, 256, 512):
            with self.subTest(valid_group_size=group_size):
                args = _shape_args(group_size=group_size)
                self.assertTrue(INT5_QUANTIZED_GEMM.supports(1, *args))
                INT5_QUANTIZED_GEMM.validate(1, *args)

        for group_size in (0, -32, 16, 48, 96, 32.0, True):
            with self.subTest(invalid_group_size=group_size):
                args = self._args()
                args[7] = group_size
                self._expect_unsupported(
                    1,
                    args,
                    "group_size must be a positive power-of-two multiple of 32",
                )

        args = self._args()
        args[7] = 1024
        self._expect_unsupported(1, args, "group_size=1024 must divide K=512")

    def test_k_must_be_positive_static_multiple_of_256(self) -> None:
        self._expect_unsupported(
            1,
            _shape_args(k=384),
            "K must be a multiple of 256",
        )
        self._expect_unsupported(
            1,
            _shape_args(k=0),
            "K must be positive",
        )

        from torch._subclasses.fake_tensor import FakeTensorMode
        from torch.fx.experimental.symbolic_shapes import ShapeEnv

        shape_env = ShapeEnv()
        mode = FakeTensorMode(shape_env=shape_env)
        with mode:
            symbolic_k = shape_env.create_unbacked_symint()
            args = [
                torch.empty(
                    1,
                    symbolic_k,
                    dtype=torch.bfloat16,
                    device="cuda",
                ),
                torch.empty(32, 256, dtype=torch.uint8, device="cuda"),
                torch.empty(32, 64, dtype=torch.uint8, device="cuda"),
                torch.empty(32, 16, dtype=torch.uint8, device="cuda"),
                torch.empty(32, 2, dtype=torch.float16, device="cuda"),
                torch.empty(32, 16, dtype=torch.uint8, device="cuda"),
                torch.empty(32, 2, dtype=torch.float16, device="cuda"),
                GROUP_SIZE,
            ]
            self._expect_unsupported(1, args, "K must be static")

    def test_weight_shapes_must_be_static(self) -> None:
        from torch._subclasses.fake_tensor import FakeTensorMode
        from torch.fx.experimental.symbolic_shapes import ShapeEnv

        shape_env = ShapeEnv()
        mode = FakeTensorMode(shape_env=shape_env)
        with mode:
            symbolic_n = shape_env.create_unbacked_symint()
            args = [
                torch.empty(1, 512, dtype=torch.bfloat16, device="cuda"),
                torch.empty(symbolic_n, 256, dtype=torch.uint8, device="cuda"),
                torch.empty(32, 64, dtype=torch.uint8, device="cuda"),
                torch.empty(32, 16, dtype=torch.uint8, device="cuda"),
                torch.empty(32, 2, dtype=torch.float16, device="cuda"),
                torch.empty(32, 16, dtype=torch.uint8, device="cuda"),
                torch.empty(32, 2, dtype=torch.float16, device="cuda"),
                GROUP_SIZE,
            ]
            self._expect_unsupported(1, args, "weight shapes must be static")

    def test_check_rows_static_and_symbolic_behavior(self) -> None:
        for bucket in SUPPORTED_BUCKETS:
            with self.subTest(bucket=bucket, static_m=bucket + 1):
                self._expect_unsupported(
                    bucket,
                    self._args(m=bucket + 1),
                    "static M must be within",
                )

        from torch._subclasses.fake_tensor import FakeTensorMode
        from torch.fx.experimental.symbolic_shapes import ShapeEnv

        shape_env = ShapeEnv()
        mode = FakeTensorMode(shape_env=shape_env)
        with mode:
            weights = [
                torch.empty(32, 256, dtype=torch.uint8, device="cuda"),
                torch.empty(32, 64, dtype=torch.uint8, device="cuda"),
                torch.empty(32, 16, dtype=torch.uint8, device="cuda"),
                torch.empty(32, 2, dtype=torch.float16, device="cuda"),
                torch.empty(32, 16, dtype=torch.uint8, device="cuda"),
                torch.empty(32, 2, dtype=torch.float16, device="cuda"),
            ]
            unbounded_m = shape_env.create_unbacked_symint()
            unbounded_args = [
                torch.empty(
                    unbounded_m,
                    512,
                    dtype=torch.bfloat16,
                    device="cuda",
                ),
                *weights,
                GROUP_SIZE,
            ]
            self._expect_unsupported(
                4,
                unbounded_args,
                "dynamic M is not provably within",
            )

            bounded_m = shape_env.create_unbacked_symint()
            shape_env.constrain_symbol_range(
                bounded_m.node.expr,
                compiler_min=1,
                compiler_max=4,
            )
            bounded_args = [
                torch.empty(
                    bounded_m,
                    512,
                    dtype=torch.bfloat16,
                    device="cuda",
                ),
                *weights,
                GROUP_SIZE,
            ]
            self.assertTrue(INT5_QUANTIZED_GEMM.supports(4, *bounded_args))
            INT5_QUANTIZED_GEMM.validate(4, *bounded_args)

    def test_each_packed_shape_rule(self) -> None:
        cases = {
            "ql": (1, "ql K/2 mismatch"),
            "qh": (2, "qh shape does not match"),
            "scale": (3, "scale/zero shape does not match"),
            "scale_step": (4, "step shape does not match"),
            "zero": (5, "scale/zero shape does not match"),
            "zero_point_step": (6, "step shape does not match"),
        }
        for name, (index, pattern) in cases.items():
            with self.subTest(tensor=name):
                args = self._args()
                args[index] = args[index][:, :-1].contiguous()
                self._expect_unsupported(1, args, pattern)

    def test_every_tensor_must_be_contiguous(self) -> None:
        names = (
            "x",
            "ql",
            "qh",
            "scale",
            "scale_step",
            "zero",
            "zero_point_step",
        )
        for index, name in enumerate(names):
            with self.subTest(tensor=name):
                args = self._args()
                args[index] = torch.stack((args[index], args[index]), dim=-1)[..., 0]
                self.assertFalse(args[index].is_contiguous())
                self._expect_unsupported(1, args, "contiguous")

    def test_tensors_must_share_a_cuda_or_fake_device(self) -> None:
        args = self._args()
        args[3] = args[3].cpu()
        self._expect_unsupported(1, args, "same device")

        args = self._args()
        for index in range(7):
            args[index] = args[index].cpu()
        self._expect_unsupported(1, args, "CUDA device")

        from torch._subclasses.fake_tensor import FakeTensorMode

        real = self._args()
        mode = FakeTensorMode()
        fake = [mode.from_tensor(tensor.cpu()) for tensor in real[:7]] + [GROUP_SIZE]
        self.assertTrue(INT5_QUANTIZED_GEMM.supports(1, *fake))
        INT5_QUANTIZED_GEMM.validate(1, *fake)

    def test_op_validates_before_launching(self) -> None:
        with self.assertRaisesRegex(RuntimeError, "static M must be within"):
            INT5_QUANTIZED_GEMM.op(1)(*self._args(m=2))


class Int5QuantizedGemmTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        if not torch.cuda.is_available():
            raise unittest.SkipTest("CUDA required")

    def test_every_static_bucket_and_candidate_matches_reference(self) -> None:
        n = 37
        for k in (256, 512, 6656):
            weights = _packed(n, k, seed=n + k)
            for bucket in SUPPORTED_BUCKETS:
                x = torch.randn(
                    bucket,
                    k,
                    dtype=torch.bfloat16,
                    device="cuda",
                )
                ref = _dequant_matmul_int5(x, *weights, GROUP_SIZE)
                for config in int5_autotune_configs(bucket):
                    with self.subTest(
                        n=n,
                        k=k,
                        bucket=bucket,
                        config=str(config),
                    ):
                        single = _single_config(config)
                        with mock.patch.dict(
                            int5_kernel._BUCKET_KERNELS,
                            {bucket: single},
                        ):
                            out = int5_kernel._launch(
                                bucket,
                                x,
                                *weights,
                                GROUP_SIZE,
                            )
                        self.assertEqual(out.shape, (bucket, n))
                        self.assertEqual(out.dtype, torch.bfloat16)
                        _check_close(self, out, ref)

    def test_every_bucket4_candidate_serves_symbolic_dynamic_m(self) -> None:
        n = 37
        for k in (256, 512, 6656):
            weights = _packed(n, k, seed=71 + k)
            inputs = {
                m: torch.randn(m, k, dtype=torch.bfloat16, device="cuda")
                for m in (2, 3)
            }
            references = {
                m: _dequant_matmul_int5(x, *weights, GROUP_SIZE)
                for m, x in inputs.items()
            }
            for config in int5_autotune_configs(4):
                with self.subTest(n=n, k=k, config=str(config)):
                    torch._dynamo.reset()
                    single = _single_config(config)
                    with mock.patch.dict(
                        int5_kernel._BUCKET_KERNELS,
                        {4: single},
                    ):
                        compiled = torch.compile(
                            lambda x, weights=weights: INT5_QUANTIZED_GEMM.op(4)(
                                x,
                                *weights,
                                GROUP_SIZE,
                            ),
                            fullgraph=True,
                        )
                        for m, x in inputs.items():
                            torch._dynamo.mark_dynamic(x, 0, min=2, max=4)
                            out = compiled(x)
                            self.assertEqual(out.shape, (m, n))
                            self.assertEqual(out.dtype, torch.bfloat16)
                            _check_close(self, out, references[m])

    def test_raw_int8_ql_and_qh_execute_correctly(self) -> None:
        original = _packed(37, 512, seed=81)
        weights = list(original)
        weights[0] = weights[0].view(torch.int8)
        weights[1] = weights[1].view(torch.int8)
        for bucket in SUPPORTED_BUCKETS:
            with self.subTest(bucket=bucket):
                x = torch.randn(
                    bucket,
                    512,
                    dtype=torch.bfloat16,
                    device="cuda",
                )
                out = INT5_QUANTIZED_GEMM.op(bucket)(
                    x,
                    *weights,
                    GROUP_SIZE,
                )
                _check_close(
                    self,
                    out,
                    _dequant_matmul_int5(
                        x,
                        *original,
                        GROUP_SIZE,
                    ),
                )

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
        real = int5_kernel.launch_split_k_gemm
        timed = []

        def recording(kernel, **kw):
            if isinstance(kw["m"], int):
                timed.append(kw["DYNAMIC_M"])
            return real(kernel, **kw)

        x = torch.randn(4, 512, dtype=torch.bfloat16, device="cuda")
        clear_launch_param_cache()
        with mock.patch.object(
            int5_kernel, "launch_split_k_gemm", side_effect=recording
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
                int5_kernel._launch(
                    4, fake_x, *(mode.from_tensor(w) for w in weights), GROUP_SIZE
                )
            dynamic_measured = stats.measured
            int5_kernel._launch(4, x, *weights, GROUP_SIZE)
        self.assertEqual(dynamic_measured, 1)
        self.assertEqual(stats.measured, 2)
        self.assertEqual(timed[0], True)
        self.assertEqual(timed[-1], False)

    def test_every_split_matches_reference(self) -> None:
        weights = _packed(37, 2048, seed=101)
        for bucket in SUPPORTED_BUCKETS:
            x = torch.randn(bucket, 2048, dtype=torch.bfloat16, device="cuda")
            ref = _dequant_matmul_int5(x, *weights, GROUP_SIZE)
            for split in SPLIT_K_CANDIDATES:
                if split > 2048 // 256:
                    with self.assertRaises(InvalidLaunchParam):
                        int5_kernel._launch(
                            bucket, x, *weights, GROUP_SIZE, split_k=split
                        )
                    continue
                with self.subTest(bucket=bucket, split=split):
                    out = int5_kernel._launch(
                        bucket, x, *weights, GROUP_SIZE, split_k=split
                    )
                    _check_close(self, out, ref)

    def test_n_tail_and_static_extra_row_are_safe(self) -> None:
        n, k = 37, 512
        weights = _packed(n, k, seed=113)
        for bucket in SUPPORTED_BUCKETS:
            rows = min(bucket + 1, max(SUPPORTED_BUCKETS))
            config = next(
                config
                for config in int5_autotune_configs(bucket)
                if config.kwargs["ROWS"] == rows
                and config.kwargs["BLOCK_N"] == 8
                and config.kwargs["PIPELINE_STAGES"] == 1
            )
            with self.subTest(bucket=bucket, rows=rows):
                x = torch.randn(
                    bucket,
                    k,
                    dtype=torch.bfloat16,
                    device="cuda",
                )
                single = _single_config(config)
                with mock.patch.dict(
                    int5_kernel._BUCKET_KERNELS,
                    {bucket: single},
                ):
                    out = int5_kernel._launch(
                        bucket,
                        x,
                        *weights,
                        GROUP_SIZE,
                    )
                self.assertEqual(out.shape, (bucket, n))
                _check_close(
                    self,
                    out,
                    _dequant_matmul_int5(
                        x,
                        *weights,
                        GROUP_SIZE,
                    ),
                )


if __name__ == "__main__":
    unittest.main()
