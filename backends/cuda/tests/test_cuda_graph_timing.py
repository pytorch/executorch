# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Tests for CUDA-graph autotune timing (autotune/cuda_graph_timing.py).

    python -m pytest backends/cuda/tests/test_cuda_graph_timing.py -v
"""

import statistics
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

    @unittest.skipIf(
        torch.version.hip is not None,
        "a failed capture invalidates the HIP stream",
    )
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
        default = (
            before
            if torch.version.hip is not None
            else cgt._time_candidate_with_cuda_graph
        )
        with CudaBackend.get_extra_aoti_compile_context_manager([]):
            self.assertIs(benchmarking._BENCHMARK_DISPATCH.get("cuda"), default)
        off = [CompileSpec(CUDA_GRAPH_AUTOTUNE_TIMING_COMPILE_SPEC, b"OFF")]
        with CudaBackend.get_extra_aoti_compile_context_manager(off):
            self.assertIs(benchmarking._BENCHMARK_DISPATCH.get("cuda"), before)
        self.assertIs(benchmarking._BENCHMARK_DISPATCH.get("cuda"), before)


# ---------------------------------------------------------------------------
# End to end: export while the CPU is saturated, and check every pick.
# ---------------------------------------------------------------------------

_WIDTHS = (256, 768, 1536)
_TOKENS = 64
_HEADS = 4


class _ManyKernels(nn.Module):
    """Small, but compiles into many distinct reductions, pointwise kernels,
    matmul templates and triton::sdpa calls, each of them autotuned."""

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


@unittest.skipIf(
    torch.version.hip is not None, "CUDA-graph autotune timing is off on ROCm"
)
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
