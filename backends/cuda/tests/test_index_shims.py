# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Mask indexing, index_put and scatter_reduce in the CUDA backend's index shims.

A mask index makes a result size depend on the data, so Inductor leaves it,
and an index_put with a mask, to the backend's own index shims. Inductor also
leaves them every scatter_reduce it cannot write with an atomic add. These
tests lower real programs, run them through the Python runtime, and compare
with eager PyTorch on CUDA.
"""

import os
import tempfile
import unittest
from typing import Sequence, Tuple
from unittest.mock import patch

import torch
from executorch.backends.cuda.cuda_backend import CudaBackend
from executorch.backends.cuda.cuda_partitioner import CudaPartitioner
from executorch.backends.cuda.passes.replace_masked_copy_with_where import (
    ReplaceMaskedCopyWithWherePass,
)
from executorch.exir import EdgeCompileConfig, to_edge_transform_and_lower
from executorch.exir.dialects._ops import ops as exir_ops
from torch.export import export


@unittest.skipUnless(torch.cuda.is_available(), "requires CUDA")
@unittest.skipIf(
    torch.version.hip is not None,
    "loads through the Python runtime, which the ROCm build does not include",
)
class TestIndexShims(unittest.TestCase):
    def _run(
        self, model: torch.nn.Module, cases: Sequence[Tuple[torch.Tensor, ...]]
    ) -> None:
        """Lower model for cases[0], then compare every case with eager CUDA."""
        spec = CudaBackend.generate_method_name_compile_spec("forward")
        lowered = to_edge_transform_and_lower(
            export(model, cases[0]),
            partitioner=[CudaPartitioner([spec])],
            compile_config=EdgeCompileConfig(_check_ir_validity=False),
        )
        # Everything runs in the delegate, the index ops included.
        self.assertEqual(
            {
                n.target.__name__
                for n in lowered.exported_program().graph.nodes
                if n.op == "call_function"
            },
            {"executorch_call_delegate", "getitem"},
        )
        from executorch.runtime import Runtime

        with tempfile.TemporaryDirectory() as directory:
            path = os.path.join(directory, "model.pte")
            with open(path, "wb") as f:
                lowered.to_executorch().write_to_file(f)
            method = Runtime.get().load_program(path).load_method("forward")
            for inputs in cases:
                outputs = method.execute(list(inputs))
                expected = model(*(t.cuda() for t in inputs))
                for output, reference in zip(outputs, expected, strict=True):
                    torch.testing.assert_close(
                        output.cpu(), reference.cpu(), equal_nan=True
                    )

    def test_mask_index_and_index_put(self):
        """index_put with accumulate goes to the shim too: Inductor has no atomic
        add for int64."""

        class MaskIndex(torch.nn.Module):
            def forward(self, x, mask, column, idx, counts):
                out = x.clone()
                out[mask] = x[mask] * 2
                # The mask and column index dims 0 and 2, which are not adjacent.
                picked = x[mask, :, column].sum(0)
                # A transposed input gives a result that is not row-major.
                rows = x.transpose(1, 2)[mask] * 3
                ones = torch.ones_like(idx)
                counts = counts.index_put((idx,), ones, accumulate=True)
                wide = x.double()
                wide = wide.index_put((mask,), wide[mask], accumulate=True)
                return out, picked, rows, counts, wide

        x = torch.randn(4, 5, 6)
        column = torch.tensor([2])
        idx = torch.tensor([0, 3, 3, 1, 3])
        # Nonzero counts check that repeated writes add to the old values.
        counts = torch.arange(8) * 5
        masks = (
            torch.tensor([True, False, True, True]),
            torch.zeros(4, dtype=torch.bool),
            torch.ones(4, dtype=torch.bool),
        )
        self._run(MaskIndex(), [(x, mask, column, idx, counts) for mask in masks])

    def test_two_dim_mask(self):
        class TwoDimMask(torch.nn.Module):
            def forward(self, x, mask):
                return (x[mask] * 2,)

        torch.manual_seed(0)
        x = torch.randn(4, 5, 6)
        masks = (torch.rand(4, 5) > 0.5, torch.zeros(4, 5, dtype=torch.bool))
        self._run(TwoDimMask(), [(x, mask) for mask in masks])

    def test_masked_copy_lowers_to_where(self):
        """Outputs alone cannot show the rewrite: a masked copy that keeps its
        index still gives the right values through the shims."""

        class Copy(torch.nn.Module):
            def forward(self, x, mask):
                out = torch.zeros_like(x)
                out[:, mask.view(-1)] = x[:, mask.view(-1)]
                return (out,)

        rewritten = []
        call = ReplaceMaskedCopyWithWherePass.call

        def record(self, graph_module):
            result = call(self, graph_module)
            rewritten.extend(
                n.target
                for n in result.graph_module.graph.nodes
                if n.op == "call_function"
            )
            return result

        x = torch.randn(2, 5, 3)
        masks = (
            torch.tensor([True, False, True, True, False]),
            torch.zeros(5, dtype=torch.bool),
            torch.ones(5, dtype=torch.bool),
        )
        with patch.object(ReplaceMaskedCopyWithWherePass, "call", record):
            self._run(Copy(), [(x, mask) for mask in masks])
        self.assertIn(exir_ops.edge.aten.where.self, rewritten)
        self.assertNotIn(exir_ops.edge.aten.index.Tensor, rewritten)

    def test_index_put_with_repeated_indices(self):
        """Deterministic mode sends half, bfloat16 and int32-index writes to the shim."""

        class Put(torch.nn.Module):
            def forward(self, x, idx, values):
                summed = x.index_put((idx,), values, accumulate=True)
                # Without accumulate, the last of the repeated writes wins.
                written = x.index_put((idx.int(),), values)
                return summed, written

        idx = torch.tensor([0, 3, 3, 1, 3])
        deterministic = torch.are_deterministic_algorithms_enabled()
        warn_only = torch.is_deterministic_algorithms_warn_only_enabled()
        torch.use_deterministic_algorithms(True)
        try:
            for dtype in (torch.float16, torch.bfloat16):
                with self.subTest(dtype=dtype):
                    x = torch.arange(8, dtype=dtype)
                    values = torch.arange(1, 6, dtype=dtype)
                    self._run(Put(), [(x, idx, values)])
        finally:
            torch.use_deterministic_algorithms(deterministic, warn_only=warn_only)

    def test_scatter_reduce(self):
        """Inductor adds atomically but has no atomic max, min or product, so
        those reductions, and the mean, go to the shim. So does an int64 sum,
        which Inductor cannot add atomically."""

        class ScatterReduce(torch.nn.Module):
            def forward(self, x, index, src):
                return tuple(
                    x.scatter_reduce(1, index, src, reduce, include_self=include)
                    for reduce in ("sum", "amax", "amin", "prod", "mean")
                    for include in (True, False)
                )

        # Smaller than src, with repeats, and column 4 is never written.
        index = torch.tensor([[0, 2, 2, 5], [1, 1, 1, 0], [5, 3, 0, 5]])
        torch.manual_seed(0)
        for dtype in (torch.float32, torch.float16, torch.bfloat16, torch.int64):
            with self.subTest(dtype=dtype):
                # Small integers keep every sum and product exact in any order.
                x = torch.randint(-4, 5, (5, 6)).to(dtype)
                src = torch.randint(-4, 5, (4, 5)).to(dtype)
                if dtype.is_floating_point:
                    # Lands on x[1, 1]; every reduction keeps the NaN.
                    src[1, 2] = float("nan")
                self._run(ScatterReduce(), [(x, index, src)])

    def test_scatter_mean_counts_in_its_dtype(self):
        """ATen counts a mean's terms in the tensor's own dtype, so 128 int8
        terms count as -128."""

        class Mean(torch.nn.Module):
            def forward(self, x, index, src):
                return tuple(
                    x.scatter_reduce(0, index, src, "mean", include_self=include)
                    for include in (True, False)
                )

        x = torch.zeros(2, dtype=torch.int8)
        index = torch.zeros(128, dtype=torch.long)
        self._run(Mean(), [(x, index, torch.ones(128, dtype=torch.int8))])

    def test_float64_sort(self):
        class Sort(torch.nn.Module):
            def forward(self, x):
                # Rows this long go to the sort shim, not to Triton.
                return torch.sort(x, stable=True)

        torch.manual_seed(0)
        self._run(Sort(), [(torch.randn(4, 1000, dtype=torch.float64),)])
