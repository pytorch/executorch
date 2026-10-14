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


def _uncached():
    return triton_kernel_wrap.identify_accessed_tensors.__wrapped_uncached__


class TritonMutationCacheTest(unittest.TestCase):
    def setUp(self) -> None:
        if not torch.cuda.is_available():
            self.skipTest("CUDA required")
        triton_mutation_cache.install()
        triton_mutation_cache.clear()

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

    def test_install_patches_the_function_pytorch_calls(self) -> None:
        self.assertTrue(
            hasattr(
                triton_kernel_wrap.identify_accessed_tensors, "__wrapped_uncached__"
            )
        )
        # Installing again does not wrap the wrapper.
        triton_mutation_cache.install()
        self.assertFalse(
            hasattr(
                triton_kernel_wrap.identify_accessed_tensors.__wrapped_uncached__,
                "__wrapped_uncached__",
            )
        )

    def test_first_call_goes_through_pytorch_and_repeats_reuse_it(self) -> None:
        mode = FakeTensorMode()
        analyze = triton_kernel_wrap.identify_accessed_tensors
        uncached = _uncached()
        first = analyze(_scale_kernel, self._kwargs(mode), {})
        # Other fake tensors of the same dtypes and the same scalars.
        for _ in range(3):
            again = analyze(_scale_kernel, self._kwargs(FakeTensorMode()), {})
            self.assertEqual(again, first)
            self.assertIsNot(again, first)
        self.assertEqual(
            triton_mutation_cache.stats(), {"hits": 3, "misses": 1, "uncached": 0}
        )
        self.assertEqual(first, uncached(_scale_kernel, self._kwargs(mode), {}))

    def test_what_changes_the_kernel_is_not_reused(self) -> None:
        analyze = triton_kernel_wrap.identify_accessed_tensors
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
        for overrides in variants:
            kwargs = self._kwargs(mode, **overrides)
            self.assertEqual(
                analyze(_scale_kernel, kwargs, {}),
                _uncached()(_scale_kernel, kwargs, {}),
                overrides,
            )
        self.assertEqual(triton_mutation_cache.stats()["misses"], len(variants))
        writes = {
            dep.name
            for dep in analyze(
                _scale_kernel, self._kwargs(mode, WRITE_Z=True), {}
            ).read_writes.writes
        }
        self.assertEqual(writes, {"y_ptr", "z_ptr"})

    def _counting_wrapper(self, delay: float = 0.0):
        """A cache over a stand-in analysis that counts its calls."""
        calls = []

        def analysis(kernel, kwargs, tma_descriptor_metadata):
            calls.append(1)
            time.sleep(delay)
            return {"writes": sorted(kwargs)}

        return triton_mutation_cache._cached(analysis), calls

    def test_tma_descriptors_and_unhashable_values_always_go_through_pytorch(
        self,
    ) -> None:
        wrapper, calls = self._counting_wrapper()
        for _ in range(2):
            wrapper(_scale_kernel, {"x_ptr": 1}, {"x_ptr": ("stable", ([16],))})
            wrapper(_scale_kernel, {"x_ptr": [1, 2]}, {})
        self.assertEqual(len(calls), 4)
        self.assertEqual(
            triton_mutation_cache.stats(), {"hits": 0, "misses": 0, "uncached": 4}
        )

    def test_the_cache_keeps_the_most_recently_used_entries(self) -> None:
        wrapper, calls = self._counting_wrapper()
        with mock.patch.object(triton_mutation_cache, "MAX_ENTRIES", 2):
            for n in (1, 2, 1, 3):  # 2 is the least recently used when 3 arrives
                wrapper(_scale_kernel, {"n": n}, {})
            self.assertEqual(len(triton_mutation_cache._CACHE), 2)
            wrapper(_scale_kernel, {"n": 1}, {})  # kept
            wrapper(_scale_kernel, {"n": 2}, {})  # dropped, analyzed again
        self.assertEqual(len(calls), 4)

    def test_concurrent_first_calls_of_a_key_analyze_once(self) -> None:
        wrapper, calls = self._counting_wrapper(delay=0.2)
        results = []
        threads = [
            threading.Thread(
                target=lambda: results.append(wrapper(_scale_kernel, {"n": 7}, {}))
            )
            for _ in range(4)
        ]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join()
        self.assertEqual(len(calls), 1)
        self.assertEqual(results, [{"writes": ["n"]}] * 4)
        self.assertEqual(
            triton_mutation_cache.stats(), {"hits": 3, "misses": 1, "uncached": 0}
        )

    def test_clear_drops_entries_and_resets_counters(self) -> None:
        wrapper, calls = self._counting_wrapper()
        wrapper(_scale_kernel, {"n": 1}, {})
        wrapper(_scale_kernel, {"n": 1}, {})
        snapshot = triton_mutation_cache.stats()
        snapshot["hits"] = 100  # a copy: the counters are unchanged
        self.assertEqual(triton_mutation_cache.stats()["hits"], 1)
        triton_mutation_cache.clear()
        self.assertEqual(
            triton_mutation_cache.stats(), {"hits": 0, "misses": 0, "uncached": 0}
        )
        self.assertEqual(len(triton_mutation_cache._CACHE), 0)
        wrapper(_scale_kernel, {"n": 1}, {})
        self.assertEqual(len(calls), 2)

    def test_tensor_shapes_and_strides_do_not_change_the_analysis(self) -> None:
        # The key leaves them out; PyTorch's own analysis must not depend on them.
        analyze = triton_kernel_wrap.identify_accessed_tensors
        mode = FakeTensorMode()
        with mode:
            tensors = [
                torch.empty(1000, device="cuda"),
                torch.empty(64, 32, device="cuda").t(),
                torch.empty(2048, device="cuda")[1:],
            ]
        first = analyze(_scale_kernel, self._kwargs(mode), {})
        for x in tensors:
            kwargs = self._kwargs(mode, x_ptr=x, y_ptr=x, z_ptr=x)
            self.assertEqual(_uncached()(_scale_kernel, kwargs, {}), first)
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

        cached = exported_code()
        stats = triton_mutation_cache.stats()
        self.assertEqual(stats["misses"], 1)
        self.assertGreaterEqual(stats["hits"], 3)
        with mock.patch.object(
            triton_kernel_wrap,
            "identify_accessed_tensors",
            _uncached(),
        ):
            uncached = exported_code()
        self.assertEqual(cached, uncached)
