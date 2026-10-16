# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""triton_mutation_cache: repeats of PyTorch's Triton access analysis are reused."""

import re
import threading
import time
import unittest
from unittest import mock

import torch
import triton
import triton.language as tl
from executorch.backends.cuda.triton.kernels import triton_mutation_cache
from torch._higher_order_ops import triton_kernel_wrap
from torch._subclasses.fake_tensor import FakeTensorMode
from torch.library import wrap_triton


@triton.jit
def _scale_kernel(
    x_ptr, y_ptr, z_ptr, n, scale, WRITE_Z: tl.constexpr, BLOCK: tl.constexpr
):
    offsets = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    mask = offsets < n
    x = tl.load(x_ptr + offsets, mask=mask)
    tl.store(y_ptr + offsets, x * scale, mask=mask)
    if WRITE_Z:
        tl.store(z_ptr + offsets, x + scale, mask=mask)


@triton.jit
def _store_z_kernel(
    x_ptr, y_ptr, z_ptr, n, scale, WRITE_Z: tl.constexpr, BLOCK: tl.constexpr
):
    offsets = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    mask = offsets < n
    x = tl.load(x_ptr + offsets, mask=mask)
    tl.store(z_ptr + offsets, x * scale, mask=mask)


def _scale(x: torch.Tensor, scale: float) -> torch.Tensor:
    y = torch.empty_like(x)
    z = torch.empty_like(x)
    n = x.numel()
    wrap_triton(_scale_kernel)[(triton.cdiv(n, 256),)](
        x, y, z, n, scale, WRITE_Z=False, BLOCK=256
    )
    return y


class _Layers(torch.nn.Module):
    """Calls the kernel directly, so export traces (and analyzes) each call."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Like a model's layers: the same kernel, dtypes and scalars each time.
        for _ in range(4):
            x = _scale(x, 0.5)
        return x


def _written(analysis) -> set:
    return {dep.name for dep in analysis.read_writes.writes}


class TritonMutationCacheTest(unittest.TestCase):
    def setUp(self) -> None:
        if not torch.cuda.is_available():
            self.skipTest("CUDA required")
        self.original = triton_kernel_wrap.identify_accessed_tensors

    def _kwargs(self, mode, **overrides):
        with mode:
            x = torch.empty(1000, device="cuda")
            kwargs = {
                "x_ptr": x,
                "y_ptr": torch.empty_like(x),
                "z_ptr": torch.empty_like(x),
                "n": 1000,
                "scale": 2.0,
                "WRITE_Z": False,
                "BLOCK": 256,
            }
        kwargs.update(overrides)
        return kwargs

    def _counting_stand_in(self, delay: float = 0.0):
        """A stand-in analysis that counts its calls, to patch in for PyTorch's."""
        calls = []

        def analysis(kernel, kwargs, tma_descriptor_metadata):
            calls.append(1)
            time.sleep(delay)
            return {"writes": sorted(kwargs)}

        return (
            mock.patch.object(
                triton_kernel_wrap, "identify_accessed_tensors", analysis
            ),
            calls,
        )

    def test_the_context_patches_the_function_pytorch_calls_and_restores_it(
        self,
    ) -> None:
        with triton_mutation_cache.enabled():
            self.assertIsNot(
                triton_kernel_wrap.identify_accessed_tensors, self.original
            )
        self.assertIs(triton_kernel_wrap.identify_accessed_tensors, self.original)

    def test_first_call_goes_through_pytorch_and_repeats_reuse_it(self) -> None:
        mode = FakeTensorMode()
        with triton_mutation_cache.enabled() as counts:
            analyze = triton_kernel_wrap.identify_accessed_tensors
            first = analyze(_scale_kernel, self._kwargs(mode), {})
            # Other fake tensors of the same dtypes and the same scalars.
            for _ in range(3):
                again = analyze(_scale_kernel, self._kwargs(FakeTensorMode()), {})
                self.assertEqual(again, first)
                self.assertIsNot(again, first)
        self.assertEqual(counts, {"hits": 3, "misses": 1, "uncached": 0})
        self.assertEqual(first, self.original(_scale_kernel, self._kwargs(mode), {}))

    def test_what_changes_the_kernel_is_not_reused(self) -> None:
        mode = FakeTensorMode()
        variants = [
            {},
            {"WRITE_Z": True},
            {"BLOCK": 128},
            {"n": 1},
            {"scale": 3.0},
        ]
        with mode:
            half = torch.empty(1000, device="cuda", dtype=torch.float16)
        variants.append({"x_ptr": half})
        with triton_mutation_cache.enabled() as counts:
            analyze = triton_kernel_wrap.identify_accessed_tensors
            for overrides in variants:
                kwargs = self._kwargs(mode, **overrides)
                self.assertEqual(
                    analyze(_scale_kernel, kwargs, {}),
                    self.original(_scale_kernel, kwargs, {}),
                    overrides,
                )
        self.assertEqual(counts["misses"], len(variants))
        self.assertEqual(
            _written(
                self.original(_scale_kernel, self._kwargs(mode, WRITE_Z=True), {})
            ),
            {"y_ptr", "z_ptr"},
        )

    def test_kernels_with_the_same_signature_are_kept_apart(self) -> None:
        mode = FakeTensorMode()
        kwargs = self._kwargs(mode, WRITE_Z=True)
        with triton_mutation_cache.enabled():
            analyze = triton_kernel_wrap.identify_accessed_tensors
            scale = analyze(_scale_kernel, kwargs, {})
            store = analyze(_store_z_kernel, kwargs, {})
        self.assertEqual(_written(scale), {"y_ptr", "z_ptr"})
        self.assertEqual(_written(store), {"z_ptr"})

    def test_an_autotuners_first_config_is_part_of_the_key(self) -> None:
        tuned = triton.autotune(
            configs=[triton.Config({"WRITE_Z": False, "BLOCK": 256})], key=["n"]
        )(_scale_kernel)
        kwargs = self._kwargs(FakeTensorMode())
        with triton_mutation_cache.enabled():
            analyze = triton_kernel_wrap.identify_accessed_tensors
            before = analyze(tuned, kwargs, {})
            tuned.configs = [triton.Config({"WRITE_Z": True, "BLOCK": 256})]
            after = analyze(tuned, kwargs, {})
        self.assertEqual(_written(before), {"y_ptr"})
        self.assertEqual(_written(after), {"y_ptr", "z_ptr"})

    def test_fresh_autotuners_of_one_kernel_share_an_entry(self) -> None:
        mode = FakeTensorMode()
        with triton_mutation_cache.enabled() as counts:
            analyze = triton_kernel_wrap.identify_accessed_tensors
            for _ in range(2):
                tuned = triton.autotune(
                    configs=[
                        triton.Config({"WRITE_Z": False, "BLOCK": 256}, num_warps=4)
                    ],
                    key=["n"],
                )(_scale_kernel)
                analyze(tuned, self._kwargs(mode), {})
        self.assertEqual(counts["hits"], 1)
        self.assertEqual(counts["misses"], 1)

    def test_tma_descriptors_and_unhashable_values_always_go_through_pytorch(
        self,
    ) -> None:
        patch, calls = self._counting_stand_in()
        with patch, triton_mutation_cache.enabled() as counts:
            analyze = triton_kernel_wrap.identify_accessed_tensors
            for _ in range(2):
                analyze(_scale_kernel, {"x_ptr": 1}, {"x_ptr": ("stable", ([16],))})
                analyze(_scale_kernel, {"x_ptr": [1, 2]}, {})
        self.assertEqual(len(calls), 4)
        self.assertEqual(counts, {"hits": 0, "misses": 0, "uncached": 4})

    def test_concurrent_calls_of_a_key_analyze_once(self) -> None:
        patch, calls = self._counting_stand_in(delay=0.2)
        results = []
        with patch, triton_mutation_cache.enabled() as counts:
            analyze = triton_kernel_wrap.identify_accessed_tensors
            threads = [
                threading.Thread(
                    target=lambda: results.append(analyze(_scale_kernel, {"n": 7}, {}))
                )
                for _ in range(4)
            ]
            for thread in threads:
                thread.start()
            for thread in threads:
                thread.join()
        self.assertEqual(len(calls), 1)
        self.assertEqual(results, [{"writes": ["n"]}] * 4)
        self.assertEqual(counts, {"hits": 3, "misses": 1, "uncached": 0})

    def test_real_tensors_are_keyed_by_dtype_and_pointer_alignment(self) -> None:
        patch, calls = self._counting_stand_in()
        base = torch.empty(64, dtype=torch.float32, device="cuda")
        with patch, triton_mutation_cache.enabled() as counts:
            analyze = triton_kernel_wrap.identify_accessed_tensors
            # base[4:] starts 16 bytes in, so its pointer has base's alignment;
            # base[1:] starts 4 bytes in and does not.
            for x in (base, base[4:], base[1:], base[1:]):
                analyze(_scale_kernel, {"x_ptr": x}, {})
        self.assertEqual(len(calls), 2)
        self.assertEqual(counts, {"hits": 2, "misses": 2, "uncached": 0})

    def test_entries_do_not_outlive_the_context(self) -> None:
        patch, calls = self._counting_stand_in()
        with patch:
            for _ in range(2):
                with triton_mutation_cache.enabled():
                    analyze = triton_kernel_wrap.identify_accessed_tensors
                    analyze(_scale_kernel, {"n": 1}, {})
                    analyze(_scale_kernel, {"n": 1}, {})
        self.assertEqual(len(calls), 2)

    def test_tensor_shapes_and_strides_do_not_change_the_analysis(self) -> None:
        # The key leaves them out; PyTorch's own analysis must not depend on them.
        mode = FakeTensorMode()
        with mode:
            tensors = [
                torch.empty(1000, device="cuda"),
                torch.empty(64, 32, device="cuda").t(),
                torch.empty(2048, device="cuda")[1:],
            ]
        first = self.original(_scale_kernel, self._kwargs(mode), {})
        with triton_mutation_cache.enabled():
            analyze = triton_kernel_wrap.identify_accessed_tensors
            for x in tensors:
                kwargs = self._kwargs(mode, x_ptr=x, y_ptr=x, z_ptr=x)
                self.assertEqual(self.original(_scale_kernel, kwargs, {}), first)
                self.assertEqual(analyze(_scale_kernel, kwargs, {}), first)

    def test_export_traces_the_same_graph_and_analyzes_once(self) -> None:
        x = torch.randn(1000, device="cuda")

        def exported_code():
            program = torch.export.export(_Layers(), (x,)).run_decompositions({})
            # Side-table indices grow with every export; the rest must match.
            return re.sub(
                r"(kernel_idx|constant_args_idx) = \d+",
                r"\1 = N",
                program.graph_module.code,
            )

        with triton_mutation_cache.enabled() as counts:
            cached = exported_code()
        self.assertEqual(counts["misses"], 1)
        self.assertGreaterEqual(counts["hits"], 3)
        self.assertEqual(cached, exported_code())
