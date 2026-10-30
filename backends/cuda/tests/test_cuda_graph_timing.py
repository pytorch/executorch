# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Tests for CUDA-graph autotune timing (autotune/cuda_graph_timing.py).

    python -m pytest backends/cuda/tests/test_cuda_graph_timing.py -v
"""

import statistics
import struct
import unittest
from unittest import mock

import torch
import torch.nn as nn
import torch.nn.functional as F
from executorch.backends.cuda.autotune import cuda_graph_timing as cgt
from executorch.backends.cuda.cuda_backend import (
    CUDA_GRAPH_AUTOTUNE_TIMING_COMPILE_SPEC,
    CudaBackend,
)
from executorch.backends.cuda.tests import autotune_test_utils
from executorch.backends.cuda.tests.autotune_test_utils import (
    record_autotune_picks,
    SaturatedCpu,
    summarize,
)
from executorch.exir.backend.compile_spec_schema import CompileSpec
from torch._inductor.runtime import benchmarking


def _require_cuda(test: unittest.TestCase) -> None:
    if not torch.cuda.is_available():
        test.skipTest("CUDA required")


class CudaGraphTimingTest(unittest.TestCase):
    def setUp(self) -> None:
        _require_cuda(self)

    def test_registers_only_inside_the_context(self) -> None:
        before = benchmarking._BENCHMARK_DISPATCH.get("cuda")
        x = torch.zeros(1 << 16, device="cuda")
        with cgt.cuda_graph_timing() as stats:
            self.assertIs(
                benchmarking._BENCHMARK_DISPATCH["cuda"],
                cgt._time_candidate_with_cuda_graph,
            )
            ms = benchmarking.benchmarker.benchmark(lambda: x.add_(1), device="cuda")
        self.assertIs(benchmarking._BENCHMARK_DISPATCH.get("cuda"), before)
        self.assertEqual(stats.captured, 1)
        self.assertGreater(ms, 0.0)

    def test_ranks_by_device_time(self) -> None:
        small = torch.zeros(1 << 14, device="cuda")
        large = torch.zeros(1 << 26, device="cuda")
        with cgt.cuda_graph_timing():
            t_small = benchmarking.benchmarker.benchmark(
                lambda: small.mul_(1.0001), device="cuda"
            )
            t_large = benchmarking.benchmarker.benchmark(
                lambda: large.mul_(1.0001), device="cuda"
            )
        self.assertGreater(t_large, 10 * t_small)

    def test_uncapturable_candidate_falls_back(self) -> None:
        x = torch.zeros(16, device="cuda")

        def syncs():
            x.add_(1)
            torch.cuda.synchronize()

        with cgt.cuda_graph_timing() as stats:
            ms = benchmarking.benchmarker.benchmark(syncs, device="cuda")
            # The stream is still usable after the failed capture.
            ok = benchmarking.benchmarker.benchmark(lambda: x.add_(1), device="cuda")
        self.assertEqual(stats.fallbacks, 1)
        self.assertEqual(stats.captured, 1)
        self.assertGreater(ms, 0.0)
        self.assertGreater(ok, 0.0)

    def test_oom_parks_offload_tensors_on_cpu_and_restores_them(self) -> None:
        weights = torch.arange(1 << 20, dtype=torch.float32, device="cuda")
        alias = weights[: 1 << 10]
        expected = weights.clone()
        sizes_during_capture = []
        real = cgt.time_with_cuda_graph_us

        def flaky(fn):
            if len(sizes_during_capture) < 2:
                sizes_during_capture.append(weights.untyped_storage().nbytes())
                raise torch.OutOfMemoryError("simulated")
            sizes_during_capture.append(weights.untyped_storage().nbytes())
            return real(fn)

        y = torch.zeros(16, device="cuda")
        with mock.patch.object(cgt, "time_with_cuda_graph_us", side_effect=flaky):
            with cgt.cuda_graph_timing(offload=lambda: [weights, alias]) as stats:
                benchmarking.benchmarker.benchmark(lambda: y.add_(1), device="cuda")
        self.assertEqual(stats.oom_retries, 2)
        self.assertEqual(stats.offloads, 1)
        self.assertEqual(stats.captured, 1)
        self.assertEqual(sizes_during_capture[-1], 0)
        self.assertTrue(weights.is_cuda)
        torch.testing.assert_close(weights, expected)
        self.assertEqual(alias.data_ptr(), weights.data_ptr())

    def test_a_failed_restore_stops_the_compile(self) -> None:
        for error in (torch.OutOfMemoryError("simulated"), RuntimeError("simulated")):
            with self.subTest(error=type(error).__name__):
                weights = torch.arange(1 << 20, dtype=torch.float32, device="cuda")
                other = torch.ones(1 << 10, device="cuda")
                expected = other.clone()
                real = cgt._restore_storage

                def restore(storage, host, error=error, weights=weights, real=real):
                    if host.nbytes() == weights.nelement() * weights.element_size():
                        raise error
                    real(storage, host)

                def oom(fn):
                    raise torch.OutOfMemoryError("simulated")

                y = torch.zeros(16, device="cuda")
                with mock.patch.object(
                    cgt, "time_with_cuda_graph_us", side_effect=oom
                ), mock.patch.object(cgt, "_restore_storage", side_effect=restore):
                    with self.assertRaises(cgt.ParkedTensorsNotRestored):
                        with cgt.cuda_graph_timing(
                            offload=lambda weights=weights, other=other: [
                                weights,
                                other,
                            ]
                        ):
                            benchmarking.benchmarker.benchmark(
                                lambda y=y: y.add_(1), device="cuda"
                            )
                # Every other parked tensor is still restored.
                torch.testing.assert_close(other, expected)

    def test_compile_spec_turns_it_off(self) -> None:
        before = benchmarking._BENCHMARK_DISPATCH.get("cuda")
        with CudaBackend.get_extra_aoti_compile_context_manager([]):
            self.assertIs(
                benchmarking._BENCHMARK_DISPATCH["cuda"],
                cgt._time_candidate_with_cuda_graph,
            )
        off = [CompileSpec(CUDA_GRAPH_AUTOTUNE_TIMING_COMPILE_SPEC, b"OFF")]
        with CudaBackend.get_extra_aoti_compile_context_manager(off):
            self.assertIs(benchmarking._BENCHMARK_DISPATCH.get("cuda"), before)
        self.assertIs(benchmarking._BENCHMARK_DISPATCH.get("cuda"), before)


# ---------------------------------------------------------------------------
# End to end: export while the CPU is saturated, and check every pick.
# ---------------------------------------------------------------------------

# Large enough that even the smallest kernels run for many event ticks (one
# tick is ~0.5 us), so a pick's regret reflects the kernels, not the timer.
_WIDTHS = (512, 1536, 3072)
_TOKENS = 4096
_HEADS = 4


class _ManyKernels(nn.Module):
    """Compiles into many distinct reductions, pointwise kernels, matmul
    templates and triton::sdpa calls, each of them autotuned."""

    def __init__(self) -> None:
        super().__init__()
        n = len(_WIDTHS)
        self.norms = nn.ModuleList(nn.RMSNorm(w, dtype=torch.bfloat16) for w in _WIDTHS)
        self.projs = nn.ModuleList(
            nn.Linear(
                _WIDTHS[i], _WIDTHS[(i + 1) % n], bias=False, dtype=torch.bfloat16
            )
            for i in range(n)
        )
        self.qkv = nn.Linear(
            _WIDTHS[0], 3 * _WIDTHS[0], bias=False, dtype=torch.bfloat16
        )

    def forward(self, x):
        h = x
        for norm, proj in zip(self.norms, self.projs):
            h = proj(F.silu(norm(h)) * torch.sigmoid(h))
        b, t, d = h.shape
        q, k, v = (
            y.reshape(b, t, _HEADS, d // _HEADS).transpose(1, 2)
            for y in self.qkv(h).chunk(3, dim=-1)
        )
        attn = F.scaled_dot_product_attention(q, k, v).transpose(1, 2).reshape(b, t, d)
        return h + attn


def _export_and_lower(extra_specs) -> None:
    from executorch.backends.cuda.cuda_partitioner import CudaPartitioner
    from executorch.exir import EdgeCompileConfig, to_edge_transform_and_lower

    torch.manual_seed(0)
    model = _ManyKernels().cuda().eval()
    x = torch.randn(1, _TOKENS, _WIDTHS[0], dtype=torch.bfloat16, device="cuda")
    with torch.no_grad():
        program = torch.export.export(model, (x,), strict=True)
    specs = [CudaBackend.generate_method_name_compile_spec("forward"), *extra_specs]
    to_edge_transform_and_lower(
        program,
        partitioner=[CudaPartitioner(specs)],
        compile_config=EdgeCompileConfig(_check_ir_validity=False),
    )


def _f32(ms: float) -> float:
    # Event times reach the autotuner as float32.
    return struct.unpack("f", struct.pack("f", ms))[0]


class RegretTest(unittest.TestCase):
    """Both backends' ticks are checked here: ROCm CI does not run this file."""

    def _regret_in_ticks(self, tick: float, best: int, pick: int) -> float:
        with mock.patch.object(autotune_test_utils, "_EVENT_TICK_MS", tick):
            return autotune_test_utils._regret(_f32(pick * tick), _f32(best * tick))

    def test_one_tick_apart_is_not_a_worse_pick(self) -> None:
        for tick in (0.512e-3, 1e-3):
            for best, pick in (
                (0, 0),
                (0, 1),
                (5, 5),
                (5, 6),
                (10, 11),
                (31250, 31251),
            ):
                with self.subTest(tick=tick, best=best, pick=pick):
                    self.assertEqual(self._regret_in_ticks(tick, best, pick), 1.0)

    def test_the_best_is_credited_one_tick(self) -> None:
        for tick in (0.512e-3, 1e-3):
            for best, pick in ((0, 2), (5, 7), (5, 8), (10, 12)):
                with self.subTest(tick=tick, best=best, pick=pick):
                    self.assertAlmostEqual(
                        self._regret_in_ticks(tick, best, pick),
                        pick / (best + 1),
                        places=2,
                    )

    def test_two_ticks_apart_count_at_any_scale(self) -> None:
        for tick in (0.512e-3, 1e-3):
            for best in (31250, 1_000_000):
                with self.subTest(tick=tick, best=best):
                    self.assertGreater(self._regret_in_ticks(tick, best, best + 2), 1.0)


class SaturatedCpuExportTest(unittest.TestCase):
    # Picks may differ from the paused-CPU reference within measurement noise.
    MAX_REGRET = 1.20
    MAX_MEAN_REGRET = 1.03
    MAX_DECISIONS_OVER_1_10 = 1

    def setUp(self) -> None:
        _require_cuda(self)

    def test_autotune_picks_stay_right_while_the_cpu_is_saturated(self) -> None:
        patched = []
        # Every export must autotune from scratch rather than reuse picks.
        with torch.compiler.config.patch(
            force_disable_caches=True
        ), SaturatedCpu() as cpu:
            with record_autotune_picks(cpu, patched):
                _export_and_lower([])
        print("CUDA-graph timing:", summarize(patched))
        kinds = {p.kind for p in patched}
        self.assertIn("inductor", kinds)
        self.assertIn("template", kinds)
        self.assertIn("custom", kinds)
        regrets = [p.regret for p in patched]
        worst = max(patched, key=lambda p: p.regret)
        self.assertLessEqual(worst.regret, self.MAX_REGRET, worst)
        self.assertLessEqual(statistics.mean(regrets), self.MAX_MEAN_REGRET)
        self.assertLessEqual(
            sum(r > 1.10 for r in regrets), self.MAX_DECISIONS_OVER_1_10, patched
        )


if __name__ == "__main__":
    unittest.main()
