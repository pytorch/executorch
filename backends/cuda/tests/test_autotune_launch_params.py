# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Tests for @autotune_launch_param (autotune/launch_params.py)."""

import unittest
from unittest import mock

import torch
import triton
import triton.language as tl
from executorch.backends.cuda.autotune import launch_params
from executorch.backends.cuda.autotune.launch_params import (
    autotune_launch_param,
    autotune_launch_params,
    clear_launch_param_cache,
    InvalidLaunchParam,
)
from torch._subclasses.fake_tensor import FakeTensorMode, is_fake
from torch.library import triton_op, wrap_triton

# Matmul repetitions per candidate: split_k=4 is clearly the fastest.
_COST = {1: 12, 2: 6, 4: 1, 8: 3, 16: 24}
_CALLS: list[int] = []


@autotune_launch_param("split_k", (1, 2, 4, 8, 16))
def _toy_launch(x: torch.Tensor, max_split: int, *, split_k: int) -> torch.Tensor:
    if split_k > max_split:
        raise InvalidLaunchParam(f"split_k={split_k} > {max_split}")
    _CALLS.append(split_k)
    out = x
    for _ in range(_COST[split_k]):
        out = out @ x
    return out


@autotune_launch_param("split_k", (8, 2, 4))
def _toy_tie(x: torch.Tensor, *, split_k: int) -> torch.Tensor:
    return x @ x


def _require_cuda(test: unittest.TestCase) -> None:
    if not torch.cuda.is_available():
        test.skipTest("CUDA required")


class AutotuneLaunchParamTest(unittest.TestCase):
    def setUp(self) -> None:
        _require_cuda(self)
        clear_launch_param_cache()
        _CALLS.clear()
        self.x = torch.randn(1024, 1024, device="cuda") / 32

    def test_outside_the_context_uses_the_first_candidate(self) -> None:
        _toy_launch(self.x, 16)
        self.assertEqual(_CALLS, [1])

    def test_picks_the_fastest_candidate_inside_the_context(self) -> None:
        with autotune_launch_params() as stats:
            _toy_launch(self.x, 16)
        self.assertEqual(stats.measured, 1)
        self.assertEqual(stats.picks[0].choice, 4)
        self.assertEqual(set(stats.picks[0].times_us), {1, 2, 4, 8, 16})
        self.assertEqual(_CALLS[-1], 4)

    def test_each_shape_is_timed_once(self) -> None:
        with autotune_launch_params() as stats:
            _toy_launch(self.x, 16)
            _toy_launch(self.x, 16)
            _toy_launch(torch.randn(512, 512, device="cuda") / 32, 16)
        self.assertEqual((stats.measured, stats.cache_hits), (2, 1))

    def test_an_explicit_value_is_used_as_is(self) -> None:
        with autotune_launch_params() as stats:
            _toy_launch(self.x, 16, split_k=8)
        self.assertEqual((stats.measured, _CALLS), (0, [8]))

    def test_illegal_candidates_are_skipped(self) -> None:
        with autotune_launch_params() as stats:
            _toy_launch(self.x, 2)
        pick = stats.picks[0]
        self.assertEqual(set(pick.times_us), {1, 2})
        self.assertEqual(set(pick.rejected), {4, 8, 16})
        self.assertEqual(pick.choice, 2)

    def test_no_legal_candidate_raises(self) -> None:
        with autotune_launch_params(), self.assertRaisesRegex(
            ValueError, "no legal split_k"
        ):
            _toy_launch(self.x, 0)

    def test_ties_go_to_the_smallest_candidate(self) -> None:
        with autotune_launch_params() as stats:
            _toy_tie(self.x)
        self.assertEqual(stats.picks[0].choice, 2)

    def test_fake_inputs_are_timed_on_real_tensors(self) -> None:
        with FakeTensorMode() as mode:
            fake = mode.from_tensor(self.x)
            with autotune_launch_params() as stats:
                out = _toy_launch(fake, 16)
        self.assertTrue(is_fake(out))
        self.assertEqual(tuple(out.shape), (1024, 1024))
        self.assertEqual(stats.picks[0].choice, 4)

    def test_out_of_memory_falls_back_to_the_first_candidate(self) -> None:
        with mock.patch.object(
            launch_params, "_time_candidates", side_effect=torch.OutOfMemoryError("oom")
        ) as timing, autotune_launch_params() as stats:
            _toy_launch(self.x, 16)
        self.assertEqual(timing.call_count, 2)
        self.assertEqual((stats.out_of_memory, stats.measured, _CALLS), (1, 0, [1]))

    def test_out_of_memory_retries_with_offload_tensors_parked(self) -> None:
        weight = torch.randn(1024, device="cuda")
        real = launch_params._time_candidates
        weight_bytes = []

        def timing(*args, **kwargs):
            weight_bytes.append(weight.untyped_storage().nbytes())
            if len(weight_bytes) < 3:
                raise torch.OutOfMemoryError("oom")
            return real(*args, **kwargs)

        with mock.patch.object(
            launch_params, "_time_candidates", side_effect=timing
        ), autotune_launch_params(offload=lambda: [weight]) as stats:
            _toy_launch(self.x, 16)
        # Parked (storage freed on the device) only for the third attempt.
        self.assertEqual(weight_bytes, [4096, 4096, 0])
        self.assertEqual(
            (stats.offloads, stats.out_of_memory, stats.picks[0].choice), (1, 0, 4)
        )
        self.assertEqual(weight.untyped_storage().nbytes(), 4096)

    def test_overlapping_input_layouts_are_timed(self) -> None:
        expanded = torch.randn(1, 1024, device="cuda").expand(1024, 1024)
        with autotune_launch_params() as stats:
            _toy_launch(expanded, 16)
        self.assertEqual(stats.measured, 1)

    def test_same_named_functions_are_cached_separately(self) -> None:
        def make(cost):
            @autotune_launch_param("split_k", (1, 2))
            def launch(x, *, split_k):
                out = x
                for _ in range(cost[split_k]):
                    out = out @ x
                return out

            return launch

        first, second = make({1: 1, 2: 8}), make({1: 8, 2: 1})
        with autotune_launch_params() as stats:
            first(self.x)
            second(self.x)
        self.assertEqual([p.choice for p in stats.picks], [1, 2])

    def test_parameter_must_exist(self) -> None:
        with self.assertRaisesRegex(TypeError, "has no parameter 'num_splits'"):

            @autotune_launch_param("num_splits", (1, 2))
            def _launch(x, *, split_k):
                return x


# CI guard: the CUDA backend's AOTI compile traces the measured-fastest value.
# Compute-bound (few elements, many dependent FMAs): time grows with REPS.
_REPS = {1: 4096, 2: 1024, 4: 16, 8: 512, 16: 8192}
_TRACED: list[tuple[bool, bool, int]] = []


@triton.autotune(
    configs=[
        triton.Config({"BLOCK": 256}, num_warps=4),
        triton.Config({"BLOCK": 1024}, num_warps=4),
    ],
    key=["N"],
)
@triton.jit
def _toy_kernel(x_ptr, out_ptr, N, REPS: tl.constexpr, BLOCK: tl.constexpr):
    offs = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    mask = offs < N
    v = tl.load(x_ptr + offs, mask=mask)
    for _ in range(REPS):
        v = v * 0.999 + 0.001
    tl.store(out_ptr + offs, v, mask=mask)


@autotune_launch_param("split_k", (1, 2, 4, 8, 16))
def _toy_kernel_launch(x: torch.Tensor, *, split_k: int) -> torch.Tensor:
    _TRACED.append((is_fake(x), launch_params._STATE.stats is not None, split_k))
    out = torch.empty_like(x)
    n = x.numel()
    wrap_triton(_toy_kernel)[lambda meta: (triton.cdiv(n, meta["BLOCK"]),)](
        x, out, n, REPS=_REPS[split_k]
    )
    return out


@triton_op("autotune_launch_param_test::toy", mutates_args={})
def _toy_op(x: torch.Tensor) -> torch.Tensor:
    return _toy_kernel_launch(x)


class _ToyModel(torch.nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return _toy_op(x) + 1


class CudaBackendExportTest(unittest.TestCase):
    def setUp(self) -> None:
        _require_cuda(self)
        clear_launch_param_cache()
        _TRACED.clear()

    def test_aoti_compile_traces_the_measured_fastest_value(self) -> None:
        from executorch.backends.cuda.cuda_backend import CudaBackend
        from executorch.backends.cuda.cuda_partitioner import CudaPartitioner
        from executorch.exir import EdgeCompileConfig, to_edge_transform_and_lower

        x = torch.randn(1 << 16, device="cuda")
        program = torch.export.export(_ToyModel(), (x,))
        to_edge_transform_and_lower(
            program,
            partitioner=[
                CudaPartitioner(
                    [CudaBackend.generate_method_name_compile_spec("forward")]
                )
            ],
            compile_config=EdgeCompileConfig(_check_ir_validity=False),
        )
        measured = [split for fake, active, split in _TRACED if not fake and active]
        traced_in_compile = [
            split for fake, active, split in _TRACED if fake and active
        ]
        self.assertEqual(sorted(set(measured)), [1, 2, 4, 8, 16], _TRACED)
        self.assertTrue(traced_in_compile, _TRACED)
        self.assertEqual(set(traced_in_compile), {4}, _TRACED)


if __name__ == "__main__":
    unittest.main()
