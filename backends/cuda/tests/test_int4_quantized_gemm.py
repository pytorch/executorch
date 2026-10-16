# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Tests for the decode-sized INT4 Triton GEMM (triton/kernels/int4_quantized_gemm.py).

Every autotune candidate of every bucket must match the dequant + F.linear
reference on the stored CudaCoalescedInt4Tensor weights and stay in the W4A8
precision class against the W4A16 reference; the ops must compile with
Inductor, statically and for a dynamic M, and match eager; and the config
AOTInductor picks at compile time must be the fastest one even while the CPU
is saturated (on CUDA; see Int4AutotunePickTest for ROCm).

    python -m pytest backends/cuda/tests/test_int4_quantized_gemm.py -v
"""

import concurrent.futures
import contextlib
import os
import pickle
import statistics
import tempfile
import types
import unittest
from unittest import mock

import torch
import torch.nn.functional as F
import triton
from executorch.backends.cuda.autotune.cuda_graph_timing import cuda_graph_timing
from executorch.backends.cuda.autotune.launch_params import (
    autotune_launch_params,
    clear_launch_param_cache,
    InvalidLaunchParam,
)

from executorch.backends.cuda.coalesced_int4_tensor import CudaCoalescedInt4Tensor
from executorch.backends.cuda.quantize_op_dispatch.int4_dispatch import _dequant_matmul
from executorch.backends.cuda.tests.autotune_test_utils import (
    record_autotune_picks,
    SaturatedCpu,
    summarize,
)
from executorch.backends.cuda.triton.kernels import (
    int4_quantized_gemm as int4_kernel,
    quantized_gemm_family,
    quantized_gemm_utils,
)
from executorch.backends.cuda.triton.kernels.int4_quantized_gemm import (
    int4_autotune_configs,
    INT4_QUANTIZED_GEMM,
    SUPPORTED_BUCKETS,
)
from executorch.backends.cuda.triton.kernels.quantized_gemm_utils import (
    autotune_do_bench,
    check_split_k,
    SPLIT_K_CANDIDATES,
)
from executorch.extension.llm.export.int4 import ExportableInt4Tensor
from executorch.extension.llm.export.quant.quantize import quantize_weight
from executorch.extension.llm.export.quant.recipe import QuantConfig
from torch._library.triton import set_wrap_triton_enabled
from torch._subclasses.fake_tensor import FakeTensorMode

GROUP_SIZE = 32
MODEL_SHAPES = (
    (8448, 6656),
    (6656, 4096),
    (39936, 6656),
    (6656, 19968),
    (256, 6656),
)
SMALL_SHAPES = ((768, 256), (256, 768), (1536, 512))
W4A8_BUCKETS = (1, 2, 3, 4)


def _packed_with_dense(n: int, k: int, seed: int = 0):
    """Return the original BF16 weight and its production CUDA storage."""
    torch.manual_seed(seed)
    dense = torch.randn(n, k, dtype=torch.bfloat16)
    config = QuantConfig(
        bits=4, group_size=GROUP_SIZE, symmetric=False, method="min_max"
    )
    weight = CudaCoalescedInt4Tensor.from_exportable_int4_tensor(
        ExportableInt4Tensor.from_int4_tensor(quantize_weight(dense, config))
    )
    packed = tuple(
        t.cuda()
        for t in (
            weight.qdata,
            weight.scale,
            weight.scale_step,
            weight.zero_point,
            weight.zero_point_step,
        )
    )
    return dense.cuda(), packed


def _packed(n: int, k: int, seed: int = 0):
    """The stored tensors of a CudaCoalescedInt4Tensor, on CUDA."""
    _, packed = _packed_with_dense(n, k, seed)
    return packed


def _check_close(test, out, ref):
    test.assertTrue(torch.isfinite(out).all())
    torch.testing.assert_close(out.float(), ref.float(), rtol=0.03, atol=8.0)
    mean_rel = (out.float() - ref.float()).abs().mean() / ref.float().abs().mean()
    test.assertLess(mean_rel.item(), 0.02)


@unittest.skipIf(
    not quantized_gemm_family._parallel_compile_available(),
    "this Triton or Python cannot compile in parallel",
)
class Int4ParallelPrecompileTest(unittest.TestCase):
    """launch_split_k_gemm compiles an untuned shape's configs in parallel."""

    @classmethod
    def setUpClass(cls) -> None:
        if not torch.cuda.is_available():
            raise unittest.SkipTest("CUDA required")

    def _count_pooled(self):
        counts = {"pooled": 0}
        submit = quantized_gemm_family._async_compile.AsyncCompileMode.submit

        def counting_submit(mode, *args, **kwargs):
            counts["pooled"] += 1
            return submit(mode, *args, **kwargs)

        return counts, mock.patch.object(
            quantized_gemm_family._async_compile.AsyncCompileMode,
            "submit",
            counting_submit,
        )

    @staticmethod
    def _compiled(bucket: int) -> int:
        fn = int4_kernel._BUCKET_KERNELS[bucket].fn
        return len(fn.device_caches[torch.cuda.current_device()][0])

    def test_an_untuned_shape_compiles_every_config_in_the_pool_first(self) -> None:
        # A shape no other test tunes, so its key is not in the autotune cache.
        n, k, bucket = 384, 2304, 2
        weights = _packed(n, k, seed=7)
        x = torch.randn(bucket, k, dtype=torch.bfloat16, device="cuda")
        counts, patch = self._count_pooled()
        before = self._compiled(bucket)
        # Launch the bucket's autotuner directly, so its cache gets the tuning key.
        with patch, set_wrap_triton_enabled(False):
            out = int4_kernel._launch(bucket, x, *weights, GROUP_SIZE, split_k=1)
            pooled, compiled = counts["pooled"], self._compiled(bucket)
            again = int4_kernel._launch(bucket, x, *weights, GROUP_SIZE, split_k=1)
        self.assertGreater(pooled, 1)
        # Every kernel the benchmark ran was compiled in the pool (which may
        # merge identical requests): a compile outside it adds a kernel the pool
        # was never asked for.
        self.assertGreater(compiled, before)
        self.assertLessEqual(compiled - before, pooled)
        # A tuned key is not compiled again.
        self.assertEqual(counts["pooled"], pooled)
        self.assertEqual(self._compiled(bucket), compiled)
        _check_close(self, out, _dequant_matmul(x, *weights, GROUP_SIZE))
        self.assertTrue(torch.equal(out, again))

    def test_every_split_candidate_compiles_on_one_pool(self) -> None:
        # A shape no other test tunes; the split is timed, not given.
        n, k, bucket = 320, 2560, 2
        weights = _packed(n, k, seed=11)
        x = torch.randn(bucket, k, dtype=torch.bfloat16, device="cuda")
        compile_pool = quantized_gemm_family._compile_pool
        compiled_at_pool_exit: list[int] = []

        @contextlib.contextmanager
        def recording_pool():
            with compile_pool():
                yield
            compiled_at_pool_exit.append(self._compiled(bucket))

        counts, patch = self._count_pooled()
        before = self._compiled(bucket)
        with patch, mock.patch.object(
            quantized_gemm_family, "_compile_pool", recording_pool
        ), autotune_launch_params() as stats:
            out = int4_kernel._launch(bucket, x, *weights, GROUP_SIZE)
        self.assertEqual(stats.measured, 1)
        self.assertGreater(len(stats.picks[0].times_us), 1)
        # The first pool, which holds every split candidate, compiles every
        # kernel the timed launches then benchmark.
        self.assertGreater(compiled_at_pool_exit[0], before)
        self.assertEqual(self._compiled(bucket), compiled_at_pool_exit[0])
        self.assertLessEqual(self._compiled(bucket) - before, counts["pooled"])
        _check_close(self, out, _dequant_matmul(x, *weights, GROUP_SIZE))

    def _cache_hits(self, bucket: int):
        """Cache hits and misses of the bucket's autotuned kernel in this
        process (its helper kernels, such as the split-K reduce, are not
        autotuned and compile here as before)."""
        name = int4_kernel._BUCKET_KERNELS[bucket].fn.fn.__name__
        hits = {"hit": 0, "miss": 0}

        def listener(src, metadata, metadata_group, times, cache_hit):
            if src.name == name:
                hits["hit" if cache_hit else "miss"] += 1

        return hits, mock.patch.object(triton.knobs.compilation, "listener", listener)

    @unittest.skipIf(torch.version.hip is not None, "worker compiles are CUDA-only")
    def test_configs_compile_in_worker_processes(self) -> None:
        # A shape no other test tunes; with an empty Triton cache, every config
        # this process loads is one a worker compiled into the shared cache.
        n, k, bucket = 448, 1792, 2
        weights = _packed(n, k, seed=13)
        x = torch.randn(bucket, k, dtype=torch.bfloat16, device="cuda")
        hits, patch = self._cache_hits(bucket)
        with tempfile.TemporaryDirectory() as cache_dir, triton.knobs.cache.scope(), (
            patch
        ):
            triton.knobs.cache.dir = cache_dir
            out = int4_kernel._launch(bucket, x, *weights, GROUP_SIZE)
        self.assertIsNotNone(quantized_gemm_family._compile_process_pool())
        self.assertGreater(hits["hit"], 1)
        self.assertEqual(hits["miss"], 0)
        _check_close(self, out, _dequant_matmul(x, *weights, GROUP_SIZE))

    @unittest.skipIf(torch.version.hip is not None, "worker compiles are CUDA-only")
    def test_workers_that_cannot_import_the_kernel_fall_back_to_this_process(
        self,
    ) -> None:
        n, k, bucket = 448, 2816, 2
        weights = _packed(n, k, seed=17)
        x = torch.randn(bucket, k, dtype=torch.bfloat16, device="cuda")
        request = quantized_gemm_family._process_compile_request

        def unimportable(compile_fn):
            args = request(compile_fn)
            return None if args is None else (args[0], "no_such_module", *args[2:])

        hits, patch = self._cache_hits(bucket)
        # An empty Triton cache: what this process compiles shows up as misses.
        with tempfile.TemporaryDirectory() as cache_dir, triton.knobs.cache.scope(), mock.patch.object(
            quantized_gemm_family, "_process_compile_request", unimportable
        ), mock.patch.object(
            quantized_gemm_family, "_process_pool_failed", False
        ), mock.patch.object(
            quantized_gemm_family, "_process_pool", None
        ), patch:
            triton.knobs.cache.dir = cache_dir
            out = int4_kernel._launch(bucket, x, *weights, GROUP_SIZE)
            self.assertTrue(quantized_gemm_family._process_pool_failed)
        # Compiled here instead, and still correct.
        self.assertGreater(hits["miss"], 0)
        _check_close(self, out, _dequant_matmul(x, *weights, GROUP_SIZE))

    def test_only_unusable_workers_stop_process_compiles(self) -> None:
        from torch._inductor.compile_worker.subproc_pool import SubprocException

        failed = quantized_gemm_family._workers_failed
        # One config failing to compile in a worker keeps the workers.
        self.assertFalse(failed(error=SubprocException("ptxas error")))
        self.assertFalse(failed(error=pickle.PicklingError("local object")))
        self.assertFalse(failed(result=None))
        # The kernel cannot be imported there, or the pool closed or broke.
        self.assertTrue(failed(result=quantized_gemm_family._UNIMPORTABLE))
        self.assertTrue(failed(error=RuntimeError("SubprocPool closed")))
        self.assertTrue(
            failed(error=concurrent.futures.process.BrokenProcessPool("died"))
        )

    def test_a_worker_applies_the_triton_environment_of_its_request(self) -> None:
        with mock.patch.dict(
            os.environ, {"TRITON_STALE": "1"}
        ), triton.knobs.cache.scope():
            result = quantized_gemm_family._compile_in_process(
                tempfile.gettempdir(),
                "no_such_module",
                "kernel",
                "no_such_module",
                "Source",
                {},
                {},
                {},
                None,
                {},
                {},
                {"TRITON_DISABLE_LINE_INFO": "1"},
            )
            self.assertEqual(os.environ["TRITON_DISABLE_LINE_INFO"], "1")
            self.assertNotIn("TRITON_STALE", os.environ)
        self.assertEqual(result, quantized_gemm_family._UNIMPORTABLE)

    def test_a_pool_that_cannot_start_stops_process_compiles(self) -> None:
        n, k, bucket = 448, 3072, 2
        weights = _packed(n, k, seed=31)
        x = torch.randn(bucket, k, dtype=torch.bfloat16, device="cuda")
        with mock.patch(
            "torch._inductor.compile_worker.subproc_pool.SubprocPool",
            side_effect=OSError("out of file descriptors"),
        ), mock.patch.object(
            quantized_gemm_family, "_process_pool", None
        ), mock.patch.object(
            quantized_gemm_family, "_process_pool_failed", False
        ):
            out = int4_kernel._launch(bucket, x, *weights, GROUP_SIZE)
            self.assertTrue(quantized_gemm_family._process_pool_failed)
        _check_close(self, out, _dequant_matmul(x, *weights, GROUP_SIZE))

    def _through_pool(self, error: BaseException):
        future = concurrent.futures.Future()
        future.set_exception(error)
        pool = mock.Mock()
        pool.submit.return_value = future
        request = (
            "cache",
            "module",
            "qualname",
            "module",
            "Source",
            {},
            {},
            {},
            None,
            {},
            {},
            {},
        )
        compile_fn = mock.Mock()
        with mock.patch.object(
            quantized_gemm_family, "_process_compile_request", return_value=request
        ), mock.patch.object(
            quantized_gemm_family, "_compile_process_pool", return_value=pool
        ), mock.patch.object(
            quantized_gemm_family, "_process_pool_failed", False
        ), mock.patch.object(
            quantized_gemm_family, "_process_pool", None
        ):
            quantized_gemm_family._compiled_in_a_process_first(compile_fn)()
            failed = quantized_gemm_family._process_pool_failed
        compile_fn.assert_called_once_with()
        return failed

    def test_a_config_failing_in_a_worker_keeps_the_pool(self) -> None:
        from torch._inductor.compile_worker.subproc_pool import SubprocException

        self.assertFalse(self._through_pool(SubprocException("ptxas error")))

    def test_a_closed_pool_stops_process_compiles(self) -> None:
        self.assertTrue(self._through_pool(RuntimeError("SubprocPool closed")))

    def test_fake_tensors_are_not_precompiled(self) -> None:
        # A full launch of a shape no other test tunes, so only the fake-tensor
        # check can keep it off the pool: the same launch on real tensors uses it.
        n, k, bucket = 384, 1792, 1
        weights = _packed(n, k, seed=19)
        x = torch.randn(bucket, k, dtype=torch.bfloat16, device="cuda")
        mode = FakeTensorMode()
        fake = [mode.from_tensor(t) for t in (x, *weights)]
        counts, patch = self._count_pooled()
        with patch:
            with mode:
                int4_kernel._launch(bucket, *fake, GROUP_SIZE)
            self.assertEqual(counts["pooled"], 0)
            int4_kernel._launch(bucket, x, *weights, GROUP_SIZE)
        self.assertGreater(counts["pooled"], 0)

    def test_a_failing_precompile_does_not_block_the_launch(self) -> None:
        # A shape no other test tunes, so the launch tries to precompile it.
        n, k, bucket = 320, 1792, 1
        weights = _packed(n, k, seed=29)
        x = torch.randn(bucket, k, dtype=torch.bfloat16, device="cuda")
        autotuner = int4_kernel._BUCKET_KERNELS[bucket]
        with mock.patch.object(
            autotuner, "warmup", side_effect=RuntimeError("no precompile")
        ), self.assertLogs(quantized_gemm_family.logger, "WARNING") as logs:
            out = int4_kernel._launch(bucket, x, *weights, GROUP_SIZE)
        self.assertIn("no precompile", "\n".join(logs.output))
        _check_close(self, out, _dequant_matmul(x, *weights, GROUP_SIZE))

    def test_the_tuning_key_is_the_autotuners_cache_key(self) -> None:
        # _tuning_key mirrors Triton's Autotuner.run; check it against the key
        # Triton stored its pick under.
        n, k, bucket = 384, 1280, 1
        weights = _packed(n, k, seed=23)
        x = torch.randn(bucket, k, dtype=torch.bfloat16, device="cuda")
        autotuner = int4_kernel._BUCKET_KERNELS[bucket]
        launches = []
        wrap = quantized_gemm_family.wrap_triton

        def recording(kernel):
            launcher = wrap(kernel)

            class _Grid:
                def __getitem__(self, grid):
                    def call(*args, **kwargs):
                        launches.append((kernel, args, kwargs))
                        return launcher[grid](*args, **kwargs)

                    return call

            return _Grid()

        # Through the op, whose eager path runs the autotuner itself (wrap_triton's
        # eager path tunes a fresh copy on every call).
        with mock.patch.object(quantized_gemm_family, "wrap_triton", recording):
            INT4_QUANTIZED_GEMM.op(bucket)(x, *weights, GROUP_SIZE)
        _, args, kwargs = next(launch for launch in launches if launch[0] is autotuner)
        self.assertIn(
            quantized_gemm_family._tuning_key(autotuner, args, kwargs), autotuner.cache
        )


class ParallelCompileGateTest(unittest.TestCase):
    def _available(self, *, version: tuple) -> bool:
        fake_sys = types.SimpleNamespace(version_info=version)
        quantized_gemm_family._parallel_compile_available.cache_clear()
        self.addCleanup(quantized_gemm_family._parallel_compile_available.cache_clear)
        with mock.patch.object(
            quantized_gemm_family, "_async_compile", types.SimpleNamespace()
        ), mock.patch.object(quantized_gemm_family, "sys", fake_sys):
            return quantized_gemm_family._parallel_compile_available()

    def test_a_triton_without_async_compile_compiles_serially(self) -> None:
        self.addCleanup(quantized_gemm_family._parallel_compile_available.cache_clear)
        quantized_gemm_family._parallel_compile_available.cache_clear()
        with mock.patch.object(quantized_gemm_family, "_async_compile", None):
            self.assertFalse(quantized_gemm_family._parallel_compile_available())

    def test_cpython_before_the_ast_fix_compiles_serially(self) -> None:
        for version in (
            (3, 11, 4, "final", 0),
            (3, 11, 7, "final", 0),
            (3, 12, 1, "final", 0),
        ):
            with self.subTest(version=version):
                self.assertFalse(self._available(version=version))

    def test_cpython_with_the_ast_fix_compiles_in_parallel(self) -> None:
        for version in (
            (3, 11, 8, "final", 0),
            (3, 12, 2, "final", 0),
            (3, 13, 0, "final", 0),
        ):
            with self.subTest(version=version):
                self.assertTrue(self._available(version=version))

    def test_a_gated_off_prepass_neither_pools_nor_queues(self) -> None:
        with mock.patch.object(
            quantized_gemm_family, "_parallel_compile_available", return_value=False
        ), mock.patch.object(quantized_gemm_family, "_compile_pool") as pool:
            with quantized_gemm_family._compile_every_candidate_at_once():
                self.assertFalse(quantized_gemm_family._QUEUING_ONLY.get())
        pool.assert_not_called()

    def test_the_gate_decides_whether_an_eligible_launch_precompiles(self) -> None:
        for gate, precompiled in ((False, False), (True, True)):
            with self.subTest(gate=gate):
                kernel = mock.Mock(
                    spec=triton.runtime.Autotuner,
                    configs=[1, 2],
                    cache={},
                    arg_names=[],
                    keys=[],
                )
                with mock.patch.object(
                    quantized_gemm_family,
                    "_parallel_compile_available",
                    return_value=gate,
                ):
                    quantized_gemm_family._precompile_untuned_configs(kernel, (), {})
                self.assertEqual(kernel.warmup.called, precompiled)


class Int4QuantizedGemmRulesTest(unittest.TestCase):
    def test_split_k_needs_one_k_tile_per_split(self) -> None:
        for k in (256, 768, 4096):
            for split in SPLIT_K_CANDIDATES:
                with self.subTest(k=k, split=split):
                    if split <= k // 256:
                        check_split_k(split, k)
                    else:
                        with self.assertRaises(InvalidLaunchParam):
                            check_split_k(split, k)

    def test_candidates_cover_the_generic_space(self) -> None:
        for bucket, rows in (
            (1, (0, 1, 2)),
            (2, (0, 2, 3)),
            (3, (0, 3, 4)),
            (4, (0, 4)),
        ):
            configs = int4_autotune_configs(bucket)
            seen = {
                (
                    c.kwargs["ROWS"],
                    c.kwargs["BLOCK_N"],
                    c.num_warps,
                    c.kwargs["PIPELINE_STAGES"],
                    c.num_stages,
                )
                for c in configs
            }
            self.assertEqual(len(configs), len(rows) * 12, bucket)
            self.assertEqual(
                seen,
                {
                    (r, w, w, st, st)
                    for r in rows
                    for w in (1, 2, 4, 8)
                    for st in (1, 2, 3)
                },
            )

    def test_prune_drops_stages_above_the_main_loop_trips(self) -> None:
        configs = int4_autotune_configs(4)
        for k, split, max_stages in (
            (256, 1, 1),
            (2048, 1, 2),
            (2048, 2, 1),
            (8192, 1, 3),
            (8192, 8, 1),
        ):
            with self.subTest(k=k, split=split):
                kept = int4_kernel._prune(configs, {"K": k}, SPLIT_K=split)
                self.assertEqual(
                    max(c.kwargs["PIPELINE_STAGES"] for c in kept), max_stages
                )
                self.assertEqual(len(kept), 8 * max_stages)
                self.assertTrue(
                    all(c.kwargs["PIPELINE_STAGES"] <= max_stages for c in kept)
                )


class Int4QuantizedGemmLegalityTest(unittest.TestCase):
    """Every rule: ``supports`` is False (no exception) and ``validate`` raises."""

    @classmethod
    def setUpClass(cls) -> None:
        if not torch.cuda.is_available():
            raise unittest.SkipTest("CUDA required")

    def _args(self, m=1, k=512, n=64):
        x = torch.randn(m, k, dtype=torch.bfloat16, device="cuda")
        return [x, *_packed(n, k, seed=3), GROUP_SIZE]

    def _expect_unsupported(self, bucket, args, pattern):
        self.assertFalse(INT4_QUANTIZED_GEMM.supports(bucket, *args))
        with self.assertRaisesRegex(RuntimeError, pattern):
            INT4_QUANTIZED_GEMM.validate(bucket, *args)

    def test_valid_inputs_are_supported(self) -> None:
        for bucket in SUPPORTED_BUCKETS:
            args = self._args(m=bucket)
            self.assertTrue(INT4_QUANTIZED_GEMM.supports(bucket, *args))
            INT4_QUANTIZED_GEMM.validate(bucket, *args)

    def test_each_rule(self) -> None:
        cases = {
            "activation dtype": (
                lambda a: a.__setitem__(0, a[0].half()),
                "activation must be bfloat16",
            ),
            "qdata dtype": (lambda a: a.__setitem__(1, a[1].float()), "qdata must be"),
            "scale dtype": (
                lambda a: a.__setitem__(2, a[2].int()),
                "scale codes must be uint8",
            ),
            "step dtype": (
                lambda a: a.__setitem__(3, a[3].float()),
                "scale_step must be float16",
            ),
            "group size": (lambda a: a.__setitem__(6, 64), "group_size must be 32"),
            "static M": (
                lambda a: a.__setitem__(0, torch.cat([a[0], a[0]])),
                "static M must be within",
            ),
            "rank": (lambda a: a.__setitem__(0, a[0].unsqueeze(0)), "rank-2"),
            "contiguous": (
                lambda a: a.__setitem__(1, a[1].t().contiguous().t()),
                "contiguous",
            ),
            "device": (lambda a: a.__setitem__(2, a[2].cpu()), "same device"),
            "cpu activation": (
                lambda a: [a.__setitem__(i, a[i].cpu()) for i in range(6)],
                "CUDA device",
            ),
            "scale shape": (
                lambda a: a.__setitem__(2, a[2][:, :-1].contiguous()),
                "scale/zero shape",
            ),
            "step shape": (
                lambda a: a.__setitem__(3, a[3][:-1].contiguous()),
                "step shape",
            ),
        }
        for name, (mutate, pattern) in cases.items():
            with self.subTest(rule=name):
                args = self._args()
                mutate(args)
                self._expect_unsupported(1, args, pattern)

    def test_k_must_be_a_multiple_of_256(self) -> None:
        x = torch.randn(1, 288, dtype=torch.bfloat16, device="cuda")
        n = 64
        args = [
            x,
            torch.zeros(n, 144, dtype=torch.uint8, device="cuda"),
            torch.zeros(n, 9, dtype=torch.uint8, device="cuda"),
            torch.zeros(n, 1, dtype=torch.float16, device="cuda"),
            torch.zeros(n, 9, dtype=torch.uint8, device="cuda"),
            torch.zeros(n, 1, dtype=torch.float16, device="cuda"),
            GROUP_SIZE,
        ]
        self._expect_unsupported(1, args, "K must be a multiple of 256")

    def test_op_validates_before_launching(self) -> None:
        args = self._args(m=2)
        with self.assertRaisesRegex(RuntimeError, "static M must be within"):
            INT4_QUANTIZED_GEMM.op(1)(*args)


class Int4QuantizedGemmTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        if not torch.cuda.is_available():
            raise unittest.SkipTest("CUDA required")

    def test_every_candidate_matches_dequant_matmul(self) -> None:
        for n, k in ((768, 256), (257, 512), (512, 8192)):
            weights = _packed(n, k, seed=n + k)
            for bucket in W4A8_BUCKETS:
                x = torch.randn(bucket, k, dtype=torch.bfloat16, device="cuda")
                ref = _dequant_matmul(x, *weights, GROUP_SIZE)
                for config in int4_autotune_configs(bucket):
                    with self.subTest(n=n, k=k, bucket=bucket, config=str(config)):
                        single = triton.autotune(
                            configs=[config], key=["N", "K", "SPLIT_K"]
                        )(int4_kernel._int4_w4a8_bucket_kernel)
                        with mock.patch.dict(
                            int4_kernel._BUCKET_KERNELS, {bucket: single}
                        ):
                            out = int4_kernel._launch(bucket, x, *weights, GROUP_SIZE)
                        _check_close(self, out, ref)

    def test_every_candidate_serves_a_dynamic_m(self) -> None:
        """Compiled for a dynamic M in [2, bucket], every config (stages
        included) matches dequant + F.linear at each runtime M."""
        n, k = 257, 512
        weights = _packed(n, k, seed=41)
        for bucket in (3, 4):
            for config in int4_autotune_configs(bucket):
                with self.subTest(bucket=bucket, config=str(config)):
                    torch._dynamo.reset()
                    single = triton.autotune(
                        configs=[config], key=["N", "K", "SPLIT_K"]
                    )(int4_kernel._int4_w4a8_bucket_kernel)
                    with mock.patch.dict(int4_kernel._BUCKET_KERNELS, {bucket: single}):
                        compiled = torch.compile(
                            lambda x, bucket=bucket: INT4_QUANTIZED_GEMM.op(bucket)(
                                x, *weights, GROUP_SIZE
                            ),
                            fullgraph=True,
                        )
                        for m in range(bucket, 1, -1):
                            x = torch.randn(m, k, dtype=torch.bfloat16, device="cuda")
                            torch._dynamo.mark_dynamic(x, 0, min=2, max=bucket)
                            out = compiled(x)
                            self.assertEqual(out.shape, (m, n))
                            _check_close(
                                self, out, _dequant_matmul(x, *weights, GROUP_SIZE)
                            )

    def test_every_bucket_matches_dequant_matmul_on_small_shapes(self) -> None:
        for n, k in SMALL_SHAPES:
            weights = _packed(n, k, seed=n + k)
            for bucket in SUPPORTED_BUCKETS:
                with self.subTest(bucket=bucket, n=n, k=k):
                    x = torch.randn(bucket, k, dtype=torch.bfloat16, device="cuda")
                    out = INT4_QUANTIZED_GEMM.op(bucket)(x, *weights, GROUP_SIZE)
                    self.assertEqual(out.shape, (bucket, n))
                    self.assertEqual(out.dtype, torch.bfloat16)
                    _check_close(self, out, _dequant_matmul(x, *weights, GROUP_SIZE))

    def test_every_bucket_matches_dequant_matmul_on_model_shapes(self) -> None:
        # Buckets 8-64 are covered on model shapes by test_int4_large_quantized_gemm.
        for n, k in MODEL_SHAPES:
            weights = _packed(n, k, seed=n ^ k)
            for bucket in W4A8_BUCKETS:
                with self.subTest(bucket=bucket, n=n, k=k):
                    x = torch.randn(bucket, k, dtype=torch.bfloat16, device="cuda")
                    out = INT4_QUANTIZED_GEMM.op(bucket)(x, *weights, GROUP_SIZE)
                    _check_close(self, out, _dequant_matmul(x, *weights, GROUP_SIZE))
            del weights
            torch.cuda.empty_cache()

    def test_tail_shape(self) -> None:
        weights = _packed(257, 256, seed=7)
        for bucket in SUPPORTED_BUCKETS:
            with self.subTest(bucket=bucket):
                x = torch.randn(bucket, 256, dtype=torch.bfloat16, device="cuda")
                out = INT4_QUANTIZED_GEMM.op(bucket)(x, *weights, GROUP_SIZE)
                _check_close(self, out, _dequant_matmul(x, *weights, GROUP_SIZE))

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

        weights = _packed(768, 512, seed=34)
        real = int4_kernel.launch_split_k_gemm
        timed = []

        def recording(kernel, **kw):
            if isinstance(kw["m"], int):
                timed.append(kw["DYNAMIC_M"])
            return real(kernel, **kw)

        x = torch.randn(4, 512, dtype=torch.bfloat16, device="cuda")
        clear_launch_param_cache()
        with mock.patch.object(
            int4_kernel, "launch_split_k_gemm", side_effect=recording
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
                int4_kernel._launch(
                    4, fake_x, *(mode.from_tensor(w) for w in weights), GROUP_SIZE
                )
            dynamic_measured = stats.measured
            int4_kernel._launch(4, x, *weights, GROUP_SIZE)
        self.assertEqual(dynamic_measured, 1)
        self.assertEqual(stats.measured, 2)
        self.assertEqual(timed[0], True)
        self.assertEqual(timed[-1], False)

    def test_every_split_matches_dequant_matmul(self) -> None:
        weights = _packed(256, 4096, seed=91)
        for bucket in W4A8_BUCKETS:
            x = torch.randn(bucket, 4096, dtype=torch.bfloat16, device="cuda")
            ref = _dequant_matmul(x, *weights, GROUP_SIZE)
            for split in SPLIT_K_CANDIDATES:
                with self.subTest(bucket=bucket, split=split):
                    out = int4_kernel._launch(
                        bucket, x, *weights, GROUP_SIZE, split_k=split
                    )
                    _check_close(self, out, ref)

    def test_an_illegal_split_is_rejected(self) -> None:
        weights = _packed(256, 768)
        x = torch.randn(1, 768, dtype=torch.bfloat16, device="cuda")
        with self.assertRaises(InvalidLaunchParam):
            int4_kernel._launch(1, x, *weights, GROUP_SIZE, split_k=4)

    def test_ops_are_registered_per_bucket(self) -> None:
        self.assertEqual(SUPPORTED_BUCKETS, (1, 2, 3, 4, 8, 16, 32, 64))
        for bucket in SUPPORTED_BUCKETS:
            self.assertTrue(
                hasattr(torch.ops.triton, f"int4_quantized_gemm_m{bucket}"), bucket
            )
        with self.assertRaisesRegex(
            RuntimeError, "unsupported int4_quantized_gemm bucket 5"
        ):
            INT4_QUANTIZED_GEMM.op(5)

    def test_rejects_an_input_above_its_bucket(self) -> None:
        weights = _packed(256, 256)
        x = torch.randn(5, 256, dtype=torch.bfloat16, device="cuda")
        with self.assertRaisesRegex(RuntimeError, "static M must be within"):
            INT4_QUANTIZED_GEMM.op(4)(x, *weights, GROUP_SIZE)

    def test_rejects_other_group_sizes(self) -> None:
        weights = _packed(256, 256)
        x = torch.randn(1, 256, dtype=torch.bfloat16, device="cuda")
        with self.assertRaisesRegex(RuntimeError, "group_size"):
            INT4_QUANTIZED_GEMM.op(1)(x, *weights, 64)

    def test_cuda_graph_replay_matches_eager(self) -> None:
        weights = _packed(256, 768, seed=88)
        for bucket in SUPPORTED_BUCKETS:
            with self.subTest(bucket=bucket):
                x = torch.randn(bucket, 768, dtype=torch.bfloat16, device="cuda")
                eager = INT4_QUANTIZED_GEMM.op(bucket)(x, *weights, GROUP_SIZE)
                for _ in range(3):
                    INT4_QUANTIZED_GEMM.op(bucket)(x, *weights, GROUP_SIZE)
                torch.cuda.synchronize()
                graph = torch.cuda.CUDAGraph()
                with torch.cuda.graph(graph):
                    captured = INT4_QUANTIZED_GEMM.op(bucket)(x, *weights, GROUP_SIZE)
                graph.replay()
                torch.cuda.synchronize()
                self.assertTrue(torch.equal(captured, eager))

    # Compiled through torch.compile's Python wrapper: the same Inductor
    # handling of the autotuned triton_op as AOTI, without loading an AOTI .so.
    def test_compiled_buckets_match_eager(self) -> None:
        torch._dynamo.reset()
        weights = _packed(256, 256)
        inputs = tuple(
            torch.randn(bucket, 256, dtype=torch.bfloat16, device="cuda")
            for bucket in SUPPORTED_BUCKETS
        )

        def linear(*xs):
            return tuple(
                INT4_QUANTIZED_GEMM.op(bucket)(x, *weights, GROUP_SIZE)
                for bucket, x in zip(SUPPORTED_BUCKETS, xs)
            )

        with torch.no_grad():
            compiled = torch.compile(linear, fullgraph=True)(*inputs)
            eager = linear(*inputs)
        self.assertEqual(len(compiled), len(eager))
        for bucket, actual, expected in zip(SUPPORTED_BUCKETS, compiled, eager):
            with self.subTest(bucket=bucket):
                _check_close(self, actual, expected)

    def test_a_failing_cuda_graph_timing_falls_back_and_says_so_once(self) -> None:
        from executorch.backends.cuda.triton.kernels import quantized_gemm_utils

        x = torch.randn(1 << 16, device="cuda")

        def kernel_call():
            x.mul_(1.0)

        stream = torch.cuda.Stream()
        with torch.cuda.stream(stream), cuda_graph_timing(), mock.patch.object(
            quantized_gemm_utils,
            "time_with_cuda_graph_us",
            side_effect=RuntimeError("capture failed"),
        ), mock.patch.object(
            quantized_gemm_utils, "_graph_timing_fallback_warned", False
        ), self.assertLogs(
            quantized_gemm_utils.logger, "WARNING"
        ) as logs:
            for _ in range(2):
                # The quantiles triton.autotune asks for.
                ms = quantized_gemm_utils.autotune_do_bench(
                    kernel_call, (0.5, 0.2, 0.8)
                )
                self.assertGreater(ms[0], 0)
        self.assertEqual(len(logs.output), 1)
        self.assertIn("capture failed", logs.output[0])

    def test_a_tuned_shape_traces_only_its_picked_config(self) -> None:
        """Once the autotuner has picked a config for a shape, tracing hands
        Inductor that config alone, so AOTI does not tune it again."""
        torch._dynamo.reset()
        n, k, bucket = 832, 1536, 16
        weights = _packed(n, k, seed=41)
        x = torch.randn(bucket, k, dtype=torch.bfloat16, device="cuda")
        autotuner = int4_kernel._W4A16_DIRECT_KERNELS[bucket]
        self.assertGreater(len(autotuner.configs), 1)

        def linear(x):
            return INT4_QUANTIZED_GEMM.op(bucket)(x, *weights, GROUP_SIZE)

        with torch.no_grad():
            eager = linear(x)  # tunes this shape
            compiled, code = torch._inductor.utils.run_and_get_code(
                torch.compile(linear, fullgraph=True), x
            )
        source = "\n".join(code)
        name = autotuner.fn.fn.__name__
        self.assertIn(f"def {name}(", source)
        block = source[
            source.rindex("user_autotune(", 0, source.index(f"def {name}(")) :
        ]
        configs = block[: block.index("],")]
        self.assertEqual(configs.count("'num_warps'"), 1)
        _check_close(self, compiled, eager)

    def test_an_untuned_shape_traces_every_config(self) -> None:
        mode = FakeTensorMode()
        traced = []
        wrap = quantized_gemm_family.wrap_triton

        def recording(kernel):
            traced.append(kernel)
            return wrap(kernel)

        with mode:
            weights = tuple(
                torch.empty_like(t, device="cuda") for t in _packed(704, 1280, seed=43)
            )
            x = torch.empty(16, 1280, dtype=torch.bfloat16, device="cuda")
        with mock.patch.object(quantized_gemm_family, "wrap_triton", recording):
            with mode:
                int4_kernel._launch(16, x, *weights, GROUP_SIZE, split_k=1)
        self.assertIs(traced[0], int4_kernel._W4A16_DIRECT_KERNELS[16])

    def test_the_picked_config_kernel_is_one_object_per_config(self) -> None:
        n, k, bucket = 832, 1536, 16
        weights = _packed(n, k, seed=41)
        x = torch.randn(bucket, k, dtype=torch.bfloat16, device="cuda")
        traced = []
        wrap = quantized_gemm_family.wrap_triton

        def recording(kernel):
            traced.append(kernel)
            return wrap(kernel)

        # Through the op, whose eager path runs the autotuner itself, so the
        # first call tunes and caches its pick (wrap_triton's eager path tunes a
        # fresh copy on every call and would never reuse it).
        with mock.patch.object(quantized_gemm_family, "wrap_triton", recording):
            for _ in range(3):
                out = INT4_QUANTIZED_GEMM.op(bucket)(x, *weights, GROUP_SIZE)
        autotuner = int4_kernel._W4A16_DIRECT_KERNELS[bucket]
        tuned = [kernel for kernel in traced if kernel is not autotuner]
        self.assertGreaterEqual(len(tuned), 2)
        self.assertTrue(all(kernel is tuned[0] for kernel in tuned))
        self.assertEqual(len(tuned[0].configs), 1)
        self.assertIn(tuned[0].configs[0], autotuner.cache.values())
        _check_close(self, out, _dequant_matmul(x, *weights, GROUP_SIZE))

    def test_dynamic_m_compiled_bucket_serves_fewer_rows(self) -> None:
        """A bucket compiled for a dynamic M <= its size masks the extra rows."""
        torch._dynamo.reset()
        n, k = 768, 512
        weights = _packed(n, k, seed=33)

        def linear(x):
            return INT4_QUANTIZED_GEMM.op(4)(x, *weights, GROUP_SIZE)

        compiled = torch.compile(linear, fullgraph=True)
        with torch.no_grad():
            for m in (4, 3, 2):
                with self.subTest(m=m):
                    x = torch.randn(m, k, dtype=torch.bfloat16, device="cuda")
                    torch._dynamo.mark_dynamic(x, 0, min=2, max=4)
                    out = compiled(x)
                    self.assertEqual(out.shape, (m, n))
                    _check_close(self, out, _dequant_matmul(x, *weights, GROUP_SIZE))


class Int4QuantizedGemmPrecisionTest(unittest.TestCase):
    """W4A8 stays within the precision class of the W4A16 dequant reference."""

    @classmethod
    def setUpClass(cls) -> None:
        if not torch.cuda.is_available():
            raise unittest.SkipTest("CUDA required")

    @staticmethod
    def _activation(bucket: int, k: int, seed: int, outliers: bool) -> torch.Tensor:
        torch.cuda.manual_seed(seed)
        x = torch.randn(bucket, k, dtype=torch.bfloat16, device="cuda")
        if outliers:
            channels = torch.tensor((0, k // 3, 2 * k // 3, k - 1), device="cuda")
            factors = torch.tensor(
                (50.0, 35.0, 25.0, 20.0), device="cuda", dtype=torch.bfloat16
            )
            x[:, channels] *= factors
        return x

    def _check_precision_case(self, bucket, dense, weights, x, label) -> None:
        a8 = INT4_QUANTIZED_GEMM.op(bucket)(x, *weights, GROUP_SIZE).float()
        a16 = _dequant_matmul(x, *weights, GROUP_SIZE).float()
        dense_ref = F.linear(x.float(), dense.float())
        self.assertTrue(torch.isfinite(a8).all(), label)

        a8_dense = (a8 - dense_ref).abs()
        a16_dense = (a16 - dense_ref).abs()
        dense_ratio = a8_dense.mean() / a16_dense.mean().clamp_min(1.0e-6)
        direct_mean_rel = (a8 - a16).abs().mean() / a16.abs().mean().clamp_min(1e-6)
        max_allowance = max(2.0, 0.02 * dense_ref.abs().max().item())

        self.assertLessEqual(dense_ratio.item(), 1.10, label)
        self.assertLessEqual(direct_mean_rel.item(), 0.02, label)
        self.assertLessEqual(
            a8_dense.max().item(), a16_dense.max().item() + max_allowance, label
        )

    def test_precision_gaussian_and_outliers(self) -> None:
        previous_tf32 = torch.backends.cuda.matmul.allow_tf32
        torch.backends.cuda.matmul.allow_tf32 = False
        try:
            for shape_index, (n, k) in enumerate(MODEL_SHAPES + SMALL_SHAPES):
                dense, weights = _packed_with_dense(n, k, seed=1200 + shape_index)
                for bucket in W4A8_BUCKETS:
                    for outliers in (False, True):
                        label = (
                            f"M={bucket},N={n},K={k},"
                            f"distribution={'outlier' if outliers else 'gaussian'}"
                        )
                        with self.subTest(label=label):
                            x = self._activation(
                                bucket,
                                k,
                                seed=9100 + shape_index * 31 + bucket,
                                outliers=outliers,
                            )
                            self._check_precision_case(bucket, dense, weights, x, label)
                del dense, weights
                torch.cuda.empty_cache()
        finally:
            torch.backends.cuda.matmul.allow_tf32 = previous_tf32

    def test_logits_top1_and_cosine(self) -> None:
        n, k = 32768, 6656
        _, weights = _packed_with_dense(n, k, seed=4242)
        for bucket in W4A8_BUCKETS:
            with self.subTest(bucket=bucket):
                x = self._activation(bucket, k, seed=4300 + bucket, outliers=True)
                a8 = INT4_QUANTIZED_GEMM.op(bucket)(x, *weights, GROUP_SIZE).float()
                a16 = _dequant_matmul(x, *weights, GROUP_SIZE).float()
                agreement = (a8.argmax(dim=1) == a16.argmax(dim=1)).float().mean()
                cosine = F.cosine_similarity(a8, a16, dim=1)
                self.assertGreaterEqual(agreement.item(), 0.99)
                self.assertGreaterEqual(cosine.min().item(), 0.999)


# Every weight packs at least 6 MiB of INT4 data, so each decode kernel runs
# for many event ticks (one tick is ~0.5 us) and a pick's regret reflects the
# kernel rather than the timer.
_GUARD_SHAPES = ((8192, 2048), (6144, 2048), (2048, 8192), (32000, 2048))
_GUARD_BUCKETS = (1, 4)


class _Int4Decode(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        for i, (n, k) in enumerate(_GUARD_SHAPES):
            for j, tensor in enumerate(_packed(n, k, seed=500 + i)):
                self.register_buffer(f"w{i}_{j}", tensor)

    def forward(self, x1_2048, x4_2048, x1_8192, x4_8192):
        inputs = {
            (1, 2048): x1_2048,
            (4, 2048): x4_2048,
            (1, 8192): x1_8192,
            (4, 8192): x4_8192,
        }
        outs = []
        for i, (_, k) in enumerate(_GUARD_SHAPES):
            weights = tuple(getattr(self, f"w{i}_{j}") for j in range(5))
            for bucket in _GUARD_BUCKETS:
                outs.append(
                    INT4_QUANTIZED_GEMM.op(bucket)(
                        inputs[(bucket, k)], *weights, GROUP_SIZE
                    )
                )
        return tuple(outs)


class AutotuneDoBenchTest(unittest.TestCase):
    def _on_stream(self, cuda_stream: int):
        return mock.patch.object(
            torch.cuda,
            "current_stream",
            return_value=types.SimpleNamespace(cuda_stream=cuda_stream),
        )

    def test_inside_graph_timing_on_a_side_stream_times_in_a_graph(self) -> None:
        kernel_call = mock.Mock()
        with self._on_stream(1), mock.patch.object(
            quantized_gemm_utils, "graph_timing_active", return_value=True
        ), mock.patch.object(
            quantized_gemm_utils, "time_with_cuda_graph_us", return_value=1500.0
        ) as graph, mock.patch.object(
            triton.testing, "do_bench"
        ) as bench:
            self.assertEqual(autotune_do_bench(kernel_call, None), 1.5)
        graph.assert_called_once_with(kernel_call)
        bench.assert_not_called()

    def test_outside_graph_timing_uses_triton(self) -> None:
        kernel_call = mock.Mock()
        with self._on_stream(1), mock.patch.object(
            quantized_gemm_utils, "graph_timing_active", return_value=False
        ), mock.patch.object(
            quantized_gemm_utils, "time_with_cuda_graph_us"
        ) as graph, mock.patch.object(
            triton.testing, "do_bench", return_value=2.0
        ) as bench:
            self.assertEqual(autotune_do_bench(kernel_call, None), 2.0)
        graph.assert_not_called()
        bench.assert_called_once_with(
            kernel_call,
            warmup=quantized_gemm_utils.AUTOTUNE_WARMUP_MS,
            rep=quantized_gemm_utils.AUTOTUNE_REP_MS,
            quantiles=None,
        )

    def test_the_legacy_default_stream_uses_triton(self) -> None:
        kernel_call = mock.Mock()
        with self._on_stream(0), mock.patch.object(
            quantized_gemm_utils, "graph_timing_active", return_value=True
        ), mock.patch.object(
            quantized_gemm_utils, "time_with_cuda_graph_us"
        ) as graph, mock.patch.object(
            triton.testing, "do_bench", return_value=2.0
        ) as bench:
            autotune_do_bench(kernel_call, None)
        graph.assert_not_called()
        bench.assert_called_once()


class Int4AutotunePickTest(unittest.TestCase):
    """The config picked for each INT4 kernel while AOTInductor compiles is the
    fastest one, measured afterwards with the CPU idle, even though the compile
    ran with every core busy. Our triton.autotune picks it during the
    launch-parameter timing, and AOTInductor traces only that pick.

    On ROCm the CUDA backend times Triton configs without CUDA graphs (see
    cuda_graph_autotune_timing in cuda_backend.py), so their ranking under a
    busy CPU is not promised there: the test checks that every INT4 kernel is
    autotuned, with a candidate count the pruning allows, but not which config
    wins."""

    MAX_REGRET = 1.10
    MAX_MEAN_REGRET = 1.03

    @classmethod
    def setUpClass(cls) -> None:
        if not torch.cuda.is_available():
            raise unittest.SkipTest("CUDA required")

    def setUp(self) -> None:
        # Start from untuned kernels, so the compile picks every config itself
        # rather than reusing the picks other tests made.
        autotuners = [
            kernel
            for table in (
                int4_kernel._BUCKET_KERNELS,
                int4_kernel._W4A16_DIRECT_KERNELS,
                int4_kernel._W4A16_SPLIT_KERNELS,
            )
            for kernel in table.values()
        ]
        saved = [(kernel, dict(kernel.cache)) for kernel in autotuners]
        for kernel in autotuners:
            kernel.cache.clear()

        def restore() -> None:
            for kernel, cache in saved:
                kernel.cache.clear()
                kernel.cache.update(cache)

        self.addCleanup(restore)

    def test_compile_time_picks_are_the_fastest_under_cpu_load(self) -> None:
        from executorch.backends.cuda.cuda_backend import CudaBackend
        from executorch.backends.cuda.cuda_partitioner import CudaPartitioner
        from executorch.exir import EdgeCompileConfig, to_edge_transform_and_lower

        module = _Int4Decode().eval()
        inputs = tuple(
            torch.randn(bucket, k, dtype=torch.bfloat16, device="cuda")
            for k in (2048, 8192)
            for bucket in _GUARD_BUCKETS
        )
        inputs = (inputs[0], inputs[1], inputs[2], inputs[3])
        with torch.no_grad():
            program = torch.export.export(module, inputs, strict=True)
        picks = []
        measure_picks = torch.version.hip is None
        # The keys whose picked config AOTI traces: the configs it compiles in.
        traced = set()
        tuned_kernel = quantized_gemm_family._tuned_kernel

        def recording_tuned_kernel(kernel, args, kwargs):
            chosen = tuned_kernel(kernel, args, kwargs)
            # Traced calls pass fake or functional tensors; the timing's eager
            # calls (plain tensors) launch every split's pick and compile nothing in.
            tracing = any(
                isinstance(a, torch.Tensor) and type(a) is not torch.Tensor
                for a in args
            )
            if tracing and chosen is not kernel:
                traced.add(
                    (
                        id(kernel),
                        quantized_gemm_family._tuning_key(kernel, args, kwargs),
                    )
                )
            return chosen

        specs = [CudaBackend.generate_method_name_compile_spec("forward")]
        with torch.compiler.config.patch(
            force_disable_caches=True
        ), SaturatedCpu() as cpu, mock.patch.object(
            quantized_gemm_family, "_tuned_kernel", recording_tuned_kernel
        ):
            with record_autotune_picks(cpu, picks, measure=measure_picks):
                to_edge_transform_and_lower(
                    program,
                    partitioner=[CudaPartitioner(specs)],
                    compile_config=EdgeCompileConfig(_check_ir_validity=False),
                )
        tuned = [p for p in picks if p.kind == "custom" and "int4" in p.name]
        print("INT4 picks, every split timed:", summarize(tuned))
        # The decisions whose config was compiled in: one per (shape, bucket),
        # for the split the launch-parameter timing chose.
        int4 = [p for p in tuned if p.key in traced]
        print("INT4 picks compiled in:", summarize(int4))
        self.assertEqual(len(int4), len(_GUARD_SHAPES) * len(_GUARD_BUCKETS), picks)
        # prune_configs_by is applied under AOTI: each kernel was tuned over the
        # configs whose pipeline stages fit its main loop.
        # The split is timed during the compile, so each decision may have tuned
        # any legal split's pruned configs.
        for p in int4:
            allowed = {
                len(
                    int4_kernel._prune(
                        int4_autotune_configs(bucket), {"K": k}, SPLIT_K=split
                    )
                )
                for _, k in _GUARD_SHAPES
                for bucket in _GUARD_BUCKETS
                for split in SPLIT_K_CANDIDATES
                if split <= k // 256
            }
            self.assertIn(p.candidates, allowed, p)
        if measure_picks:
            worst = max(int4, key=lambda p: p.regret)
            self.assertLessEqual(worst.regret, self.MAX_REGRET, worst)
            self.assertLessEqual(
                statistics.mean(p.regret for p in int4), self.MAX_MEAN_REGRET, int4
            )


if __name__ == "__main__":
    unittest.main()
