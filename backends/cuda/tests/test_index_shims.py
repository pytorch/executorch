# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Ops that Inductor does not lower by itself in the CUDA backend.

A mask index makes a result size depend on the data, so Inductor leaves it,
and an index_put with a mask, to the backend's own index shims. Inductor also
leaves them every scatter and scatter_reduce it cannot write with an atomic
add. varlen_attn and _reshape_copy have no Inductor lowering, so the backend
decomposes them. These tests lower real programs, run them through the Python
runtime, and compare with eager PyTorch on CUDA.
"""

import contextlib
import os
import tempfile
import unittest
from typing import Iterator, Sequence, Tuple
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
from torch.nn.attention.varlen import varlen_attn


@contextlib.contextmanager
def _deterministic() -> Iterator[None]:
    """Deterministic mode, which sends more scatters and index_puts to the
    shims."""
    enabled = torch.are_deterministic_algorithms_enabled()
    warn_only = torch.is_deterministic_algorithms_warn_only_enabled()
    torch.use_deterministic_algorithms(True)
    try:
        yield
    finally:
        torch.use_deterministic_algorithms(enabled, warn_only=warn_only)


@unittest.skipUnless(torch.cuda.is_available(), "requires CUDA")
@unittest.skipIf(
    torch.version.hip is not None,
    "loads through the Python runtime, which the ROCm build does not include",
)
class TestIndexShims(unittest.TestCase):
    def _run(
        self,
        model: torch.nn.Module,
        cases: Sequence[Tuple[torch.Tensor, ...]],
        **tolerance: float,
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
                        output.cpu(), reference.cpu(), equal_nan=True, **tolerance
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
        # Negative and int32: the shims wrap the index and read it as int32.
        column = torch.tensor([-4], dtype=torch.int32)
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
        with _deterministic():
            for dtype in (torch.float16, torch.bfloat16):
                with self.subTest(dtype=dtype):
                    x = torch.arange(8, dtype=dtype)
                    values = torch.arange(1, 6, dtype=dtype)
                    self._run(Put(), [(x, idx, values)])

    def test_scatter_reduce(self):
        """Inductor adds atomically but has no atomic max, min or product, so
        those reductions, and the mean, go to the shim. So does a sum it cannot
        add atomically: int64, or bfloat16 below sm_90."""

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

    def test_scatter_with_tensor_source(self):
        """Inductor writes a scatter from a tensor itself only in a dtype it can
        add atomically, so int64, bool and, below sm_90, bfloat16 go to the
        shim. Of repeated writes the last in index order wins, as in eager's
        deterministic mode, which also sends every dtype to the shim. As in
        ATen, the index can be int32."""

        class Scatter(torch.nn.Module):
            def forward(self, x, index, src):
                return (x.scatter(1, index, src),)

        # Smaller than src, with repeats, and column 4 is never written.
        index = torch.tensor([[0, 2, 2, 5], [1, 1, 1, 0], [5, 3, 0, 5]])
        x = torch.full((5, 6), -1)
        # Repeated writes differ from the write before them, also as bool.
        src = torch.arange(20).reshape(4, 5)
        with _deterministic():
            for dtype, index_dtype in (
                (torch.bfloat16, torch.int64),
                (torch.int64, torch.int32),
                (torch.bool, torch.int64),
                (torch.float32, torch.int64),
                (torch.float16, torch.int64),
            ):
                with self.subTest(dtype=dtype, index_dtype=index_dtype):
                    values = src % 2 if dtype == torch.bool else src
                    inputs = (x.to(dtype), index.to(index_dtype), values.to(dtype))
                    self._run(Scatter(), [inputs])

    def test_scatter_strides(self):
        """Scatters along the middle dim of a 3-D tensor, given as a negative
        dim, from a strided source and from transposed operands, so the shims
        walk strides that differ between their tensors."""

        class Strided(torch.nn.Module):
            def forward(self, x, index, src):
                picked = src[:, ::2]
                return (
                    x.scatter(-2, index, picked),
                    x.scatter_reduce(-2, index, picked, "amax"),
                    x.transpose(0, 2).scatter(
                        1, index.transpose(0, 2), picked.transpose(0, 2)
                    ),
                )

        torch.manual_seed(0)
        x = torch.randn(4, 6, 5)
        index = torch.randint(0, 6, (3, 4, 5))
        src = torch.randn(3, 8, 5)
        with _deterministic():
            self._run(Strided(), [(x, index, src)])

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
        # With self, the second element counts 256, which wraps to 0.
        index = torch.tensor([0] * 128 + [1] * 255)
        self._run(Mean(), [(x, index, torch.ones(383, dtype=torch.int8))])

    def test_scatter_reduce_edge_cases(self):
        """A bool sum, products that overflow int32 and half, and an empty
        index."""

        class EdgeCases(torch.nn.Module):
            def forward(self, flags, ints, index, empty):
                halves = ints.half()
                return (
                    flags.scatter_reduce(0, index, flags, "sum"),
                    # As in ATen, int32 wraps and half overflows to infinity.
                    ints.scatter_reduce(0, index, ints, "prod"),
                    halves.scatter_reduce(0, index, halves, "prod"),
                    halves.scatter_reduce(0, empty, halves[:0], "mean"),
                )

        flags = torch.tensor([True, False, False, True, False])
        ints = torch.tensor([300, -7, 46341, 300, 5], dtype=torch.int32)
        index = torch.tensor([0, 2, 2, 0, 1])
        empty = torch.zeros(0, dtype=torch.long)
        self._run(EdgeCases(), [(flags, ints, index, empty)])

    def test_scatter_reduce_sum_adds_in_float(self):
        """Deterministic mode sends half and bfloat16 sums to the shim, which
        adds in float, as ATen does there, so many small terms add up."""

        class Sum(torch.nn.Module):
            def forward(self, index, src):
                return tuple(
                    torch.zeros(3, dtype=dtype, device=src.device).scatter_reduce(
                        0, index, src.to(dtype), "sum"
                    )
                    for dtype in (torch.float16, torch.bfloat16)
                )

        with _deterministic():
            self._run(Sum(), [(torch.arange(1000) % 3, torch.full((1000,), 0.1))])

    def test_float64_and_float16_values(self):
        class Values(torch.nn.Module):
            def forward(self, x, scale):
                # Rows this long go to the sort shim, not to Triton.
                values, indices = torch.sort(x, stable=True)
                return values * scale.item(), indices

        torch.manual_seed(0)
        for dtype in (torch.float64, torch.float16):
            with self.subTest(dtype=dtype):
                x = torch.randn(4, 1000, dtype=dtype)
                # A fraction, which a float passed to Triton as an integer
                # loses.
                scale = torch.tensor(1.5, dtype=dtype)
                self._run(Values(), [(x, scale)])

    def test_reshape_copy(self):
        """Torch-TensorRT hands back the views it leaves to PyTorch as
        _reshape_copy."""

        class ReshapeCopy(torch.nn.Module):
            def forward(self, x):
                reshape = torch.ops.aten._reshape_copy
                return (
                    reshape(x.transpose(0, 1), [-1, 4]),
                    reshape(x[:, 1:3], [8, -1]),
                    reshape(x, [4, 32]),
                )

        self._run(ReshapeCopy(), [(torch.randn(8, 4, 4),)])

    def test_varlen_attn(self):
        """varlen_attn calls Flash attention, which the CUDA shims lack."""

        class Attention(torch.nn.Module):
            def forward(self, q, k, v, cu_seq):
                # Flattening views each output, which needs the op's layout.
                return tuple(
                    varlen_attn(
                        q, k, v, cu_seq, cu_seq, 16, 16, window_size=window
                    ).flatten(1)
                    for window in ((-1, -1), (-1, 0), (8, 0), (4, 2))
                )

        torch.manual_seed(0)
        q, k, v = (torch.randn(28, 4, 64, dtype=torch.bfloat16) for _ in range(3))
        # Equal scores so large that float32 rounds their log-sum-exp.
        big = torch.full_like(q, 1024)
        # The sequence boundaries come only at run time.
        cases = [
            (q, k, v, torch.tensor([0, 12, 28], dtype=torch.int32)),
            (q, k, v, torch.tensor([0, 16, 28], dtype=torch.int32)),
            (big, big, v, torch.tensor([0, 12, 28], dtype=torch.int32)),
        ]
        self._run(Attention(), cases, atol=1e-2, rtol=1e-2)

    def test_masked_select(self):
        """A direct masked_select call goes to the shim, unlike a mask index,
        which Inductor reads as index.Tensor. The mask broadcasts over rows and a
        transposed input checks that the shim reads strides."""

        class MaskedSelect(torch.nn.Module):
            def forward(self, x, row_mask):
                return (
                    torch.masked_select(x, x > 0),
                    torch.masked_select(x, row_mask),
                    torch.masked_select(x.transpose(0, 1), x.transpose(0, 1) < 0),
                )

        torch.manual_seed(0)
        x = torch.randn(4, 3)
        cases = [
            (x, torch.tensor([True, False, True])),
            (torch.zeros(4, 3), torch.zeros(3, dtype=torch.bool)),
            (torch.randn(4, 3), torch.ones(3, dtype=torch.bool)),
        ]
        self._run(MaskedSelect(), cases)

    def test_nonzero(self):
        """A direct nonzero call goes to the shim, unlike a mask index, which
        Inductor reads as index.Tensor. A transposed input checks that the shim
        reads strides, and the sum checks the result layout Inductor expects."""

        class Nonzero(torch.nn.Module):
            def forward(self, x):
                return (
                    torch.nonzero(x > 0),
                    torch.nonzero(x.transpose(0, 1)),
                    torch.nonzero(x > 0).sum(0),
                )

        x = torch.tensor([[1.0, -2.0, 3.0], [-4.0, 5.0, -0.0]])
        cases = [(x,), (torch.zeros(2, 3),), (torch.randn(2, 3),)]
        self._run(Nonzero(), cases)
