# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Exhaustive tests for the decode-sized INT8 Triton GEMM.

The default run covers every autotune candidate. For focused development,
``INT8_TEST_K`` may select a comma-separated subset of 256, 512, and 4096, and
``INT8_TEST_CONFIG_LIMIT`` may select the first N candidates per bucket.

    PYTHONPATH=$PWD/executorch/src python -m pytest test_int8_quantized_gemm.py -v
"""

import unittest
import warnings
from unittest import mock

import sympy
import torch
import triton
from executorch.backends.cuda.autotune.launch_params import (
    autotune_launch_params,
    clear_launch_param_cache,
    InvalidLaunchParam,
)

from executorch.backends.cuda.quantize_op_dispatch.int8_dispatch import _unit_dq_mm_int8
from executorch.backends.cuda.triton.kernels import int8_quantized_gemm as int8_kernel
from executorch.backends.cuda.triton.kernels.int8_quantized_gemm import (
    int8_autotune_configs,
    INT8_QUANTIZED_GEMM,
    SUPPORTED_BUCKETS,
)
from executorch.backends.cuda.triton.kernels.quantized_gemm_utils import (
    SPLIT_K_CANDIDATES,
)
from torch._subclasses.fake_tensor import FakeTensorMode
from torch.fx.experimental.symbolic_shapes import ShapeEnv


_N = 37
_K_VALUES = (256, 512, 4096)
_GROUP_SIZES = (32, 128)
_EXPECTED_ROWS = {1: (0, 1, 2), 2: (0, 2, 3), 3: (0, 3, 4), 4: (0, 4)}


def _weights(n: int, k: int, group_size: int, seed: int):
    """Random CUDA INT8 weights with bounded dequantized values."""
    generator = torch.Generator(device="cuda").manual_seed(seed)
    qdata = torch.randint(
        -64, 64, (n, k), dtype=torch.int8, device="cuda", generator=generator
    )
    scale = (
        torch.rand(
            (n, k // group_size),
            dtype=torch.float32,
            device="cuda",
            generator=generator,
        )
        * 0.02
        + 0.005
    ).to(torch.bfloat16)
    zero = torch.randint(
        -8,
        9,
        (n, k // group_size),
        dtype=torch.int8,
        device="cuda",
        generator=generator,
    )
    return qdata, scale, zero


def _activation(m: int, k: int, seed: int) -> torch.Tensor:
    generator = torch.Generator(device="cuda").manual_seed(seed)
    return torch.randn((m, k), dtype=torch.bfloat16, device="cuda", generator=generator)


def _empty_args(m=1, k=512, n=_N, group_size=32, device="cpu"):
    return [
        torch.empty((m, k), dtype=torch.bfloat16, device=device),
        torch.empty((n, k), dtype=torch.int8, device=device),
        torch.empty((n, k // group_size), dtype=torch.bfloat16, device=device),
        torch.empty((n, k // group_size), dtype=torch.int8, device=device),
        group_size,
    ]


def _single_config_kernel(config: triton.Config):
    return triton.autotune(configs=[config], key=["N", "K", "SPLIT_K"])(
        int8_kernel._int8_w8a8_bucket_kernel
    )


def _check_close(test: unittest.TestCase, out: torch.Tensor, ref: torch.Tensor) -> None:
    test.assertTrue(torch.isfinite(out).all().item())
    torch.testing.assert_close(out.float(), ref.float(), rtol=0.02, atol=4.0)
    mean_abs_ref = ref.float().abs().mean()
    test.assertGreater(mean_abs_ref.item(), 0.0)
    mean_relative = (out.float() - ref.float()).abs().mean() / mean_abs_ref
    test.assertLess(mean_relative.item(), 0.01)


def _make_symbol(shape_env: ShapeEnv, name: str, hint: int, low: int, high: int):
    value = shape_env.create_symintnode(sympy.Symbol(name), hint=hint)
    shape_env.constrain_symbol_range(
        value.node.expr, compiler_min=low, compiler_max=high
    )
    return value


class Int8QuantizedGemmRulesTest(unittest.TestCase):
    def test_candidates_are_the_exact_cartesian_space(self) -> None:
        for bucket, rows in _EXPECTED_ROWS.items():
            configs = int8_autotune_configs(bucket)
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
                (row, warps, warps, stages, stages)
                for row in rows
                for warps in (1, 2, 4, 8)
                for stages in (1, 2, 3)
            }
            with self.subTest(bucket=bucket):
                self.assertEqual(len(configs), len(rows) * 12)
                self.assertEqual(seen, expected)

    def test_prune_drops_stages_above_main_loop_trips(self) -> None:
        cases = (
            (256, 1, 1),
            (2048, 1, 2),
            (2048, 2, 1),
            (4096, 1, 3),
            (8192, 8, 1),
        )
        for bucket, rows in _EXPECTED_ROWS.items():
            configs = int8_autotune_configs(bucket)
            for k, split_k, max_stages in cases:
                with self.subTest(
                    bucket=bucket, k=k, split_k=split_k, max_stages=max_stages
                ):
                    kept = int8_kernel._prune(configs, {"K": k}, SPLIT_K=split_k)
                    self.assertEqual(len(kept), len(rows) * 4 * max_stages)
                    self.assertEqual(
                        max(config.kwargs["PIPELINE_STAGES"] for config in kept),
                        max_stages,
                    )
                    self.assertTrue(
                        all(
                            config.kwargs["PIPELINE_STAGES"] <= max_stages
                            for config in kept
                        )
                    )


class Int8QuantizedGemmLegalityTest(unittest.TestCase):
    def _expect_unsupported(self, bucket, args, pattern) -> None:
        self.assertFalse(INT8_QUANTIZED_GEMM.supports(bucket, *args))
        with self.assertRaisesRegex(RuntimeError, pattern):
            INT8_QUANTIZED_GEMM.validate(bucket, *args)

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA required")
    def test_valid_cuda_inputs_are_supported(self) -> None:
        for bucket in SUPPORTED_BUCKETS:
            for group_size in (32, 64, 128, 256, 512):
                args = [
                    _activation(bucket, 512, bucket * 1000 + group_size),
                    *_weights(_N, 512, group_size, bucket * 2000 + group_size),
                    group_size,
                ]
                with self.subTest(bucket=bucket, group_size=group_size):
                    self.assertTrue(INT8_QUANTIZED_GEMM.supports(bucket, *args))
                    INT8_QUANTIZED_GEMM.validate(bucket, *args)

    def test_same_device_fake_cpu_and_cuda_inputs_are_supported(self) -> None:
        for device in ("cpu", "cuda"):
            with self.subTest(device=device), FakeTensorMode():
                args = _empty_args(m=4, device=device)
                self.assertTrue(INT8_QUANTIZED_GEMM.supports(4, *args))
                INT8_QUANTIZED_GEMM.validate(4, *args)
                out = INT8_QUANTIZED_GEMM.op(4)(*args)
                self.assertEqual(out.shape, (4, _N))
                self.assertEqual(out.dtype, torch.bfloat16)
                self.assertEqual(out.device, args[0].device)

    def test_dynamic_rows_must_be_provably_within_bucket(self) -> None:
        for bucket in (3, 4):
            shape_env = ShapeEnv()
            m = _make_symbol(shape_env, f"valid_m{bucket}", bucket, 2, bucket)
            with FakeTensorMode(shape_env=shape_env):
                args = _empty_args(m=m)
                self.assertTrue(INT8_QUANTIZED_GEMM.supports(bucket, *args))
                INT8_QUANTIZED_GEMM.validate(bucket, *args)

            shape_env = ShapeEnv()
            m = _make_symbol(shape_env, f"invalid_m{bucket}", bucket, 2, bucket + 1)
            with FakeTensorMode(shape_env=shape_env):
                args = _empty_args(m=m)
                self._expect_unsupported(
                    bucket, args, rf"dynamic M is not provably within \[1, {bucket}\]"
                )

    def test_k_must_be_static(self) -> None:
        shape_env = ShapeEnv()
        k = _make_symbol(shape_env, "dynamic_k", 512, 256, 1024)
        with FakeTensorMode(shape_env=shape_env):
            args = _empty_args(k=k)
            self._expect_unsupported(1, args, "K must be static")

    def test_weight_shapes_must_be_static(self) -> None:
        shape_env = ShapeEnv()
        n = _make_symbol(shape_env, "dynamic_n", _N, 1, 64)
        with FakeTensorMode(shape_env=shape_env):
            args = _empty_args(n=n)
            self._expect_unsupported(1, args, "weight shapes must be static")

    def _cuda_args(self, m=1, k=512, n=_N, group_size=32):
        return [
            _activation(m, k, m * 10000 + k + group_size),
            *_weights(n, k, group_size, m * 20000 + k + group_size),
            group_size,
        ]

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA required")
    def test_every_tensor_must_have_rank_two(self) -> None:
        for index, name in enumerate(("activation", "qdata", "scale", "zero")):
            args = self._cuda_args()
            args[index] = args[index].unsqueeze(0)
            with self.subTest(tensor=name):
                self._expect_unsupported(1, args, "expects rank-2")

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA required")
    def test_every_tensor_has_one_exact_dtype(self) -> None:
        cases = (
            (0, torch.float32, "activation must be bfloat16"),
            (1, torch.uint8, "qdata must be int8"),
            (2, torch.float32, "scale must be bfloat16"),
            (3, torch.int16, "zero must be int8"),
        )
        for index, dtype, pattern in cases:
            args = self._cuda_args()
            args[index] = args[index].to(dtype)
            with self.subTest(index=index, dtype=dtype):
                self._expect_unsupported(1, args, pattern)

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA required")
    def test_group_size_contract(self) -> None:
        invalid = (
            (None, "positive power-of-two multiple"),
            (32.0, "positive power-of-two multiple"),
            (0, "positive power-of-two multiple"),
            (-32, "positive power-of-two multiple"),
            (16, "positive power-of-two multiple"),
            (96, "positive power-of-two multiple"),
            (1024, "must divide K=512"),
        )
        for group_size, pattern in invalid:
            args = self._cuda_args()
            args[-1] = group_size
            with self.subTest(group_size=group_size):
                self._expect_unsupported(1, args, pattern)

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA required")
    def test_k_must_be_a_multiple_of_256(self) -> None:
        args = self._cuda_args(k=288)
        self._expect_unsupported(1, args, "K must be a multiple of 256")

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA required")
    def test_qdata_and_metadata_shapes_must_match(self) -> None:
        cases = (
            (
                "qdata K",
                lambda args: args.__setitem__(1, args[1][:, :-1].contiguous()),
                "qdata K mismatch",
            ),
            (
                "scale groups",
                lambda args: args.__setitem__(2, args[2][:, :-1].contiguous()),
                "scale/zero shape",
            ),
            (
                "scale N",
                lambda args: args.__setitem__(2, args[2][:-1, :].contiguous()),
                "scale/zero shape",
            ),
            (
                "zero groups",
                lambda args: args.__setitem__(3, args[3][:, :-1].contiguous()),
                "scale/zero shape",
            ),
            (
                "zero N",
                lambda args: args.__setitem__(3, args[3][:-1, :].contiguous()),
                "scale/zero shape",
            ),
        )
        for name, mutate, pattern in cases:
            args = self._cuda_args()
            mutate(args)
            with self.subTest(rule=name):
                self._expect_unsupported(1, args, pattern)

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA required")
    def test_every_tensor_must_be_contiguous(self) -> None:
        for index, name in enumerate(("activation", "qdata", "scale", "zero")):
            args = self._cuda_args(m=2)
            args[index] = args[index].t().contiguous().t()
            self.assertFalse(args[index].is_contiguous())
            with self.subTest(tensor=name):
                self._expect_unsupported(2, args, "inputs must be contiguous")

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA required")
    def test_qdata_must_be_four_byte_aligned(self) -> None:
        args = self._cuda_args()
        storage = torch.empty(
            args[1].numel() + 1, dtype=torch.int8, device=args[1].device
        )
        args[1] = storage[1:].view_as(args[1])
        self.assertTrue(args[1].is_contiguous())
        self.assertEqual(args[1].storage_offset(), 1)
        self._expect_unsupported(1, args, "qdata storage offset must be 4-byte aligned")

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA required")
    def test_real_inputs_must_share_one_cuda_device(self) -> None:
        args = self._cuda_args()
        args[1] = args[1].cpu()
        self._expect_unsupported(1, args, "same device")

        args = self._cuda_args()
        args[:4] = [tensor.cpu() for tensor in args[:4]]
        self._expect_unsupported(1, args, "activation must be on a CUDA device")

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA required")
    def test_static_rows_must_equal_bucket(self) -> None:
        args = self._cuda_args(m=2)
        self._expect_unsupported(1, args, "static M must equal the bucket 1")

    def test_ops_are_registered_per_bucket(self) -> None:
        self.assertEqual(SUPPORTED_BUCKETS, (1, 2, 3, 4))
        self.assertEqual(INT8_QUANTIZED_GEMM.buckets, SUPPORTED_BUCKETS)
        for bucket in SUPPORTED_BUCKETS:
            self.assertTrue(
                hasattr(torch.ops.triton, f"int8_quantized_gemm_m{bucket}"), bucket
            )
        with self.assertRaisesRegex(
            RuntimeError, "unsupported int8_quantized_gemm bucket 5"
        ):
            INT8_QUANTIZED_GEMM.op(5)

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA required")
    def test_registered_op_validates_before_launch(self) -> None:
        with self.assertRaisesRegex(RuntimeError, "static M must equal the bucket 1"):
            INT8_QUANTIZED_GEMM.op(1)(*self._cuda_args(m=2))


class Int8QuantizedGemmNumericsTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        if not torch.cuda.is_available():
            raise unittest.SkipTest("CUDA required")

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

        weights = _weights(_N, 512, 32, seed=34)
        real = int8_kernel.launch_split_k_gemm
        timed = []

        def recording(kernel, **kw):
            if isinstance(kw["m"], int):
                timed.append(kw["DYNAMIC_M"])
            return real(kernel, **kw)

        x = torch.randn(4, 512, dtype=torch.bfloat16, device="cuda")
        clear_launch_param_cache()
        with mock.patch.object(
            int8_kernel, "launch_split_k_gemm", side_effect=recording
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
                int8_kernel._launch(
                    4, fake_x, *(mode.from_tensor(w) for w in weights), 32
                )
            dynamic_measured = stats.measured
            int8_kernel._launch(4, x, *weights, 32)
        self.assertEqual(dynamic_measured, 1)
        self.assertEqual(stats.measured, 2)
        self.assertEqual(timed[0], True)
        self.assertEqual(timed[-1], False)

    def test_every_split_matches_reference(self) -> None:
        with torch.no_grad():
            weights = _weights(_N, 4096, 32, seed=424242)
            for bucket in SUPPORTED_BUCKETS:
                x = _activation(bucket, 4096, seed=4242 + bucket)
                ref = _unit_dq_mm_int8(x, *weights, 32)
                for split in SPLIT_K_CANDIDATES:
                    with self.subTest(bucket=bucket, split=split):
                        out = int8_kernel._launch(
                            bucket, x, *weights, 32, split_k=split
                        )
                        _check_close(self, out, ref)

    def test_an_illegal_split_is_rejected(self) -> None:
        weights = _weights(_N, 512, 32, seed=7)
        x = _activation(1, 512, seed=7)
        with self.assertRaises(InvalidLaunchParam):
            int8_kernel._launch(1, x, *weights, 32, split_k=4)

    def test_every_static_bucket_and_candidate(self) -> None:
        with torch.no_grad():
            for k in _K_VALUES:
                for group_size in _GROUP_SIZES:
                    weights = _weights(
                        _N, k, group_size, seed=100000 + k * 10 + group_size
                    )
                    for bucket in SUPPORTED_BUCKETS:
                        x = _activation(
                            bucket,
                            k,
                            seed=200000 + bucket * 10000 + k * 10 + group_size,
                        )
                        ref = _unit_dq_mm_int8(x, *weights, group_size)
                        checked = 0
                        for config_index, config in enumerate(
                            int8_autotune_configs(bucket)
                        ):
                            with self.subTest(
                                bucket=bucket,
                                n=_N,
                                k=k,
                                group_size=group_size,
                                config_index=config_index,
                                config=str(config),
                            ):
                                single = _single_config_kernel(config)
                                try:
                                    with mock.patch.dict(
                                        int8_kernel._BUCKET_KERNELS, {bucket: single}
                                    ):
                                        out = int8_kernel._launch(
                                            bucket, x, *weights, group_size
                                        )
                                except triton.runtime.errors.OutOfResources as e:
                                    # The autotuner drops a config that does not
                                    # fit the device (e.g. BLOCK_N=8 x 3 stages
                                    # at K=4096 on a 99 KiB A10G).
                                    warnings.warn(f"{config}: {e}", stacklevel=1)
                                    continue
                                checked += 1
                                self.assertEqual(out.shape, (bucket, _N))
                                self.assertEqual(out.dtype, torch.bfloat16)
                                _check_close(self, out, ref)
                        self.assertGreater(
                            checked,
                            0,
                            f"no candidate fits for {k=} {group_size=} {bucket=}",
                        )

    def test_every_dynamic_bucket_and_candidate(self) -> None:
        with torch.no_grad():
            for k in _K_VALUES:
                for group_size in _GROUP_SIZES:
                    weights = _weights(
                        _N, k, group_size, seed=300000 + k * 10 + group_size
                    )
                    for bucket in (3, 4):
                        for config_index, config in enumerate(
                            int8_autotune_configs(bucket)
                        ):
                            with self.subTest(
                                bucket=bucket,
                                n=_N,
                                k=k,
                                group_size=group_size,
                                config_index=config_index,
                                config=str(config),
                            ):
                                torch._dynamo.reset()
                                single = _single_config_kernel(config)
                                op = INT8_QUANTIZED_GEMM.op(bucket)

                                def linear(
                                    x, op=op, weights=weights, group_size=group_size
                                ):
                                    return op(x, *weights, group_size)

                                inputs = [
                                    _activation(
                                        m,
                                        k,
                                        seed=(
                                            400000
                                            + bucket * 10000
                                            + m * 1000
                                            + k * 10
                                            + group_size
                                        ),
                                    )
                                    for m in range(bucket, 1, -1)
                                ]
                                for x in inputs:
                                    torch._dynamo.mark_dynamic(x, 0, min=2, max=bucket)

                                with mock.patch.dict(
                                    int8_kernel._BUCKET_KERNELS, {bucket: single}
                                ):
                                    compiled = torch.compile(linear, fullgraph=True)
                                    for m, x in zip(range(bucket, 1, -1), inputs):
                                        out = compiled(x)
                                        ref = _unit_dq_mm_int8(x, *weights, group_size)
                                        self.assertEqual(out.shape, (m, _N))
                                        self.assertEqual(out.dtype, torch.bfloat16)
                                        _check_close(self, out, ref)


if __name__ == "__main__":
    unittest.main()
