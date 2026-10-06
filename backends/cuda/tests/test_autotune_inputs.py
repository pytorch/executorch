# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Tests for representative autotune values (autotune/inputs.py).

    python -m pytest backends/cuda/tests/test_autotune_inputs.py -v
"""

import contextlib
import importlib
import unittest
from unittest import mock

import torch
from executorch.backends.cuda.autotune import inputs as ai
from executorch.backends.cuda.autotune.inputs import (
    autotune_input_scenarios,
    autotune_inputs,
    scenario_arguments,
)


def _toy_kernel(X_ptr, N_ptr, n_elements, BLOCK: int):
    pass


class AutotuneInputsDeclarationTest(unittest.TestCase):
    def setUp(self) -> None:
        self._registry = dict(ai._REGISTRY)

    def tearDown(self) -> None:
        ai._REGISTRY.clear()
        ai._REGISTRY.update(self._registry)

    def test_returns_the_kernel_and_registers_it_by_name(self) -> None:
        provider = lambda args: [1, 2]  # noqa: E731
        self.assertIs(autotune_inputs(N_ptr=provider)(_toy_kernel), _toy_kernel)
        self.assertIs(ai._REGISTRY["_toy_kernel"].providers["N_ptr"], provider)

    def test_unwraps_autotuner_and_jit_wrappers(self) -> None:
        jit = mock.Mock(spec=["fn"], fn=_toy_kernel)
        autotuner = mock.Mock(spec=["fn"], fn=jit)
        self.assertIs(autotune_inputs(N_ptr=lambda a: [1])(autotuner), autotuner)
        self.assertIn("_toy_kernel", ai._REGISTRY)

    def test_rejects_unknown_parameters_and_empty_declarations(self) -> None:
        with self.assertRaisesRegex(ValueError, "no parameter.*KV_LEN"):
            autotune_inputs(KV_LEN=lambda a: [1])(_toy_kernel)
        with self.assertRaises(ValueError):
            autotune_inputs()

    def test_rejects_the_same_kernel_name_from_another_function(self) -> None:
        autotune_inputs(N_ptr=lambda a: [1])(_toy_kernel)

        def other(N_ptr):
            pass

        other.__name__ = "_toy_kernel"
        with self.assertRaisesRegex(ValueError, "declared by both"):
            autotune_inputs(N_ptr=lambda a: [1])(other)


class ScenarioArgumentsTest(unittest.TestCase):
    NAMES = ("X_ptr", "N_ptr", "n_elements")

    def setUp(self) -> None:
        self._registry = dict(ai._REGISTRY)
        autotune_inputs(N_ptr=lambda a: [a["n_elements"] // 2, a["n_elements"]])(_toy_kernel)

    def tearDown(self) -> None:
        ai._REGISTRY.clear()
        ai._REGISTRY.update(self._registry)

    def test_undeclared_kernel_is_untouched(self) -> None:
        self.assertIsNone(scenario_arguments("other", self.NAMES, (1, 2, 3), {}, {}))

    def test_numbers_fill_private_copies(self) -> None:
        x, n = torch.randn(8), torch.zeros(1, dtype=torch.int32)
        runs = scenario_arguments("_toy_kernel", self.NAMES, (x, n, 64), {}, {})
        self.assertEqual([r[0][1].item() for r in runs], [32, 64])
        for args, _ in runs:
            self.assertIs(args[0], x)
            self.assertIsNot(args[1], n)
            self.assertEqual(args[1].dtype, torch.int32)
        self.assertEqual(n.item(), 0)

    def test_providers_see_constexprs_and_kwargs(self) -> None:
        ai._REGISTRY.clear()
        autotune_inputs(N_ptr=lambda a: [a["BLOCK"]])(_toy_kernel)
        n = torch.zeros(1, dtype=torch.int32)
        runs = scenario_arguments(
            "_toy_kernel", self.NAMES, (torch.randn(8),), {"N_ptr": n}, {"BLOCK": 16}
        )
        self.assertEqual(runs[0][1]["N_ptr"].item(), 16)

    def test_non_tensor_argument_is_left_alone(self) -> None:
        self.assertIsNone(
            scenario_arguments("_toy_kernel", self.NAMES, (torch.randn(8), 0, 64), {}, {})
        )

    def test_tensor_values_must_match(self) -> None:
        ai._REGISTRY.clear()
        autotune_inputs(N_ptr=lambda a: [torch.zeros(2, dtype=torch.int32)])(_toy_kernel)
        n = torch.zeros(1, dtype=torch.int32)
        with self.assertRaisesRegex(ValueError, "must match"):
            scenario_arguments("_toy_kernel", self.NAMES, (torch.randn(8), n, 64), {}, {})

    def test_paired_declarations_need_equal_counts(self) -> None:
        ai._REGISTRY.clear()
        autotune_inputs(X_ptr=lambda a: [1.0], N_ptr=lambda a: [1, 2])(_toy_kernel)
        args = (torch.randn(8), torch.zeros(1, dtype=torch.int32), 64)
        with self.assertRaisesRegex(ValueError, "same, non-zero number"):
            scenario_arguments("_toy_kernel", self.NAMES, args, {}, {})

    def test_context_patches_bench_only_while_active(self) -> None:
        from torch._inductor.runtime.triton_heuristics import CachingAutotuner

        original = CachingAutotuner.bench
        with autotune_input_scenarios():
            with autotune_input_scenarios():
                self.assertIs(CachingAutotuner.bench, ai._bench_over_scenarios)
            self.assertIs(CachingAutotuner.bench, ai._bench_over_scenarios)
        self.assertIs(CachingAutotuner.bench, original)


_SDPA = importlib.import_module("executorch.backends.cuda.triton.kernels.sdpa")
_H_KV, _GROUPS, _HEAD_DIM, _CACHE = 2, 16, 128, 131072
_RUNTIME_KV_LENS = (2600, 32768)


class _Decode(torch.nn.Module):
    def forward(self, q, k, v, kv_len):
        return torch.ops.triton.sdpa_decode_splitk(q, k, v, kv_len=kv_len)


def _config_key(config) -> tuple:
    return (tuple(sorted(config.kwargs.items())), config.num_warps, config.num_stages)


@contextlib.contextmanager
def _record_decode_pick(picks: list):
    from torch._inductor.runtime.triton_heuristics import CachingAutotuner

    original = CachingAutotuner.benchmark_all_configs

    def record(self, *args, **kwargs):
        timings = original(self, *args, **kwargs)
        if getattr(self.fn, "__name__", "") == "_sdpa_decode_splitk_kernel":
            picks.append(_config_key(min(timings, key=timings.get).config))
        return timings

    CachingAutotuner.benchmark_all_configs = record
    try:
        yield
    finally:
        CachingAutotuner.benchmark_all_configs = original


def _decode_inputs():
    torch.manual_seed(0)
    q = torch.randn(1, _H_KV * _GROUPS, 1, _HEAD_DIM, dtype=torch.bfloat16, device="cuda")
    k = torch.randn(1, _H_KV, _CACHE, _HEAD_DIM, dtype=torch.bfloat16, device="cuda")
    v = torch.randn_like(k)
    return q, k, v


def _compiled_decode_pick() -> tuple:
    from executorch.backends.cuda.cuda_backend import CudaBackend
    from executorch.backends.cuda.cuda_partitioner import CudaPartitioner
    from executorch.exir import EdgeCompileConfig, to_edge_transform_and_lower

    q, k, v = _decode_inputs()
    kv_len = torch.tensor([_RUNTIME_KV_LENS[0]], dtype=torch.int64, device="cuda")
    with torch.no_grad():
        program = torch.export.export(_Decode(), (q, k, v, kv_len), strict=True)
    picks = []
    with torch.compiler.config.patch(force_disable_caches=True), _record_decode_pick(picks):
        to_edge_transform_and_lower(
            program,
            partitioner=[CudaPartitioner([CudaBackend.generate_method_name_compile_spec("forward")])],
            compile_config=EdgeCompileConfig(_check_ir_validity=False),
        )
    assert len(picks) == 1, picks
    return picks[0]


def _time_every_config(kv_len_value: int) -> dict:
    """Device time of each decode config at a real KV length (CUDA graphs)."""
    import triton
    from executorch.backends.cuda.autotune.cuda_graph_timing import time_with_cuda_graph_us

    q, k, v = _decode_inputs()
    kv_len = torch.tensor([kv_len_value], dtype=torch.int64, device="cuda")
    kernel = _SDPA._sdpa_decode_splitk_kernel
    times = {}
    with torch.cuda.stream(torch.cuda.Stream()):
        for config in kernel.configs:
            single = triton.autotune(configs=[config], key=kernel.keys)(kernel.fn)
            with mock.patch.object(_SDPA, "_sdpa_decode_splitk_kernel", single):
                try:
                    times[_config_key(config)] = time_with_cuda_graph_us(
                        lambda: torch.ops.triton.sdpa_decode_splitk(q, k, v, kv_len=kv_len)
                    )
                except Exception:  # noqa: BLE001 - configs that do not fit are skipped
                    continue
    return times


class SdpaKvLenAutotuneTest(unittest.TestCase):
    """The decode SDPA config AOTInductor compiles in is the fastest at real
    KV lengths, not at the empty cache autotuning would otherwise see."""

    MAX_REGRET = 1.10

    @classmethod
    def setUpClass(cls) -> None:
        if not torch.cuda.is_available():
            raise unittest.SkipTest("CUDA required")

    def test_sdpa_kernels_reading_kv_len_are_declared(self) -> None:
        for name in ("_sdpa_fwd_kernel", "_sdpa_decode_splitk_kernel", "_sdpa_small_query_splitk_kernel"):
            self.assertIn("KV_LEN_ptr", ai._REGISTRY[name].providers, name)

    def test_compiled_pick_is_fastest_at_runtime_kv_lengths(self) -> None:
        pick = _compiled_decode_pick()
        for kv_len in _RUNTIME_KV_LENS:
            times = _time_every_config(kv_len)
            best = min(times.values())
            regret = times[pick] / best
            print(f"kv_len={kv_len}: compiled pick {regret:.2f}x of the best")
            self.assertLessEqual(regret, self.MAX_REGRET, (kv_len, pick, times))


if __name__ == "__main__":
    unittest.main()
