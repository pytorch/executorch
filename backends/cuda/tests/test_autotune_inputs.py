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
        autotune_inputs(N_ptr=lambda a: [a["n_elements"] // 2, a["n_elements"]])(
            _toy_kernel
        )

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
            scenario_arguments(
                "_toy_kernel", self.NAMES, (torch.randn(8), 0, 64), {}, {}
            )
        )

    def test_tensor_values_must_match(self) -> None:
        ai._REGISTRY.clear()
        autotune_inputs(N_ptr=lambda a: [torch.zeros(2, dtype=torch.int32)])(
            _toy_kernel
        )
        n = torch.zeros(1, dtype=torch.int32)
        with self.assertRaisesRegex(ValueError, "must match"):
            scenario_arguments(
                "_toy_kernel", self.NAMES, (torch.randn(8), n, 64), {}, {}
            )

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
def _record_pick(kernel_name: str, picks: list):
    from torch._inductor.runtime.triton_heuristics import CachingAutotuner

    original = CachingAutotuner.benchmark_all_configs

    def record(self, *args, **kwargs):
        timings = original(self, *args, **kwargs)
        if getattr(self.fn, "__name__", "") == kernel_name:
            picks.append(_config_key(min(timings, key=timings.get).config))
        return timings

    CachingAutotuner.benchmark_all_configs = record
    try:
        yield
    finally:
        CachingAutotuner.benchmark_all_configs = original


def _compile_picks(module, inputs, kernel_name: str) -> list:
    from executorch.backends.cuda.cuda_backend import CudaBackend
    from executorch.backends.cuda.cuda_partitioner import CudaPartitioner
    from executorch.exir import EdgeCompileConfig, to_edge_transform_and_lower

    with torch.no_grad():
        program = torch.export.export(module, inputs, strict=True)
    picks = []
    with torch.compiler.config.patch(force_disable_caches=True), _record_pick(
        kernel_name, picks
    ):
        to_edge_transform_and_lower(
            program,
            partitioner=[
                CudaPartitioner(
                    [CudaBackend.generate_method_name_compile_spec("forward")]
                )
            ],
            compile_config=EdgeCompileConfig(_check_ir_validity=False),
        )
    return picks


def _decode_inputs():
    torch.manual_seed(0)
    q = torch.randn(
        1, _H_KV * _GROUPS, 1, _HEAD_DIM, dtype=torch.bfloat16, device="cuda"
    )
    k = torch.randn(1, _H_KV, _CACHE, _HEAD_DIM, dtype=torch.bfloat16, device="cuda")
    v = torch.randn_like(k)
    return q, k, v


def _compiled_decode_pick() -> tuple:
    q, k, v = _decode_inputs()
    kv_len = torch.tensor([_RUNTIME_KV_LENS[0]], dtype=torch.int64, device="cuda")
    picks = _compile_picks(_Decode(), (q, k, v, kv_len), "_sdpa_decode_splitk_kernel")
    assert len(picks) == 1, picks
    return picks[0]


def _time_every_config(kv_len_value: int) -> dict:
    """Device time of each decode config at a real KV length (CUDA graphs)."""
    import triton
    from executorch.backends.cuda.autotune.cuda_graph_timing import (
        time_with_cuda_graph_us,
    )

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
                        lambda: torch.ops.triton.sdpa_decode_splitk(
                            q, k, v, kv_len=kv_len
                        )
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
        for name in (
            "_sdpa_fwd_kernel",
            "_sdpa_decode_splitk_kernel",
            "_sdpa_small_query_splitk_kernel",
        ):
            self.assertIn("KV_LEN_ptr", ai._REGISTRY[name].providers, name)
            self.assertIn("Mask_ptr", ai._REGISTRY[name].providers, name)

    def test_visible_mask_pairs_with_every_kv_length(self) -> None:
        args = {"Lk": 65536, "Lq": 32}
        self.assertEqual(
            _SDPA._visible_mask_scenarios(args),
            [True] * len(_SDPA._kv_len_scenarios(args)),
        )

    def test_compiled_pick_is_fastest_at_runtime_kv_lengths(self) -> None:
        pick = _compiled_decode_pick()
        for kv_len in _RUNTIME_KV_LENS:
            times = _time_every_config(kv_len)
            best = min(times.values())
            regret = times[pick] / best
            print(f"kv_len={kv_len}: compiled pick {regret:.2f}x of the best")
            self.assertLessEqual(regret, self.MAX_REGRET, (kv_len, pick, times))


# A cell-layout step: T queries of T sequences over one pool, an explicit
# visibility mask and the live extent as kv_len (lower_offgraph_kv.cell_step).
_CELL_T, _CELL_H_Q, _CELL_POOL = 32, 32, 65536
_CELL_LIVE = (4650, 9300)


class _CellStep(torch.nn.Module):
    def forward(self, q, k, v, mask, kv_len):
        return torch.ops.triton.sdpa(q, k, v, mask, 0.0, False, 0.0, True, kv_len)


def _cell_inputs(live: int):
    """Each sequence's 128-cell prompt, then decode cells interleaved across
    the sequences, filling ``live`` cells."""
    torch.manual_seed(0)
    q = torch.randn(
        1, _CELL_H_Q, _CELL_T, _HEAD_DIM, dtype=torch.bfloat16, device="cuda"
    )
    k = torch.randn(
        1, _CELL_POOL, _H_KV, _HEAD_DIM, dtype=torch.bfloat16, device="cuda"
    ).transpose(1, 2)
    v = torch.randn_like(k)
    mask = torch.zeros(1, 1, _CELL_T, _CELL_POOL, dtype=torch.bool, device="cuda")
    prompt = 128
    for i in range(_CELL_T):
        mask[0, 0, i, i * prompt : (i + 1) * prompt] = True
    tail = torch.arange(_CELL_T * prompt, live, device="cuda")
    mask[0, 0, (tail - _CELL_T * prompt) % _CELL_T, tail] = True
    kv_len = torch.tensor([live], dtype=torch.int64, device="cuda")
    return q, k, v, mask, kv_len


def _time_every_cell_config(live: int) -> dict:
    import triton
    from executorch.backends.cuda.autotune.cuda_graph_timing import (
        time_with_cuda_graph_us,
    )

    q, k, v, mask, kv_len = _cell_inputs(live)
    kernel = _SDPA._sdpa_fwd_kernel
    configs = _SDPA._sdpa_fwd_prune(
        kernel.configs, None, HEAD_DIM=_HEAD_DIM, NUM_GROUPS=_CELL_H_Q // _H_KV
    )
    times = {}
    with torch.cuda.stream(torch.cuda.Stream()):
        for config in configs:
            single = triton.autotune(configs=[config], key=kernel.keys)(kernel.fn)
            with mock.patch.object(_SDPA, "_sdpa_fwd_kernel", single):
                try:
                    times[_config_key(config)] = time_with_cuda_graph_us(
                        lambda: _CellStep()(q, k, v, mask, kv_len)
                    )
                except Exception:  # noqa: BLE001 - configs that do not fit are skipped
                    continue
    return times


class SdpaCellStepAutotuneTest(unittest.TestCase):
    """The standard SDPA config (tile and pack GQA) compiled for a masked
    cell step is the fastest at live extents, not at the all-False mask
    autotuning would otherwise time."""

    MAX_REGRET = 1.10

    @classmethod
    def setUpClass(cls) -> None:
        if not torch.cuda.is_available():
            raise unittest.SkipTest("CUDA required")

    def test_compiled_pick_is_fastest_at_live_extents(self) -> None:
        picks = _compile_picks(
            _CellStep(), _cell_inputs(_CELL_LIVE[0]), "_sdpa_fwd_kernel"
        )
        self.assertEqual(len(picks), 1, picks)
        for live in _CELL_LIVE:
            times = _time_every_cell_config(live)
            best = min(times.values())
            regret = times[picks[0]] / best
            print(f"live={live}: compiled pick {regret:.2f}x of the best")
            self.assertLessEqual(regret, self.MAX_REGRET, (live, picks[0], times))


if __name__ == "__main__":
    unittest.main()
