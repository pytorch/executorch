# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Tests for the Core AI delegate-boundary compatibility check.

* :class:`IoMismatchesTest`: unit tests for the ``io_mismatches`` utility.
* :class:`BoundaryLoweringTest`: e2e lowering of models with various input and
  output types through the full flow (``to_edge_transform_and_lower`` ->
  ``to_executorch``), asserting the expected Core AI delegation and no leftover
  ops. The boundary check runs inside ``preprocess``.
"""

import unittest

import torch
import torch.nn as nn

from executorch.backends.apple.coreai import (
    get_default_compile_config,
    get_default_passes,
)
from executorch.backends.apple.coreai.compiler.io_compat import io_mismatches
from executorch.backends.apple.coreai.partition.partitioner import CoreAIPartitioner
from executorch.exir import to_edge_transform_and_lower
from executorch.exir.lowered_backend_module import executorch_call_delegate
from torch.export import Dim


class _Sym:
    """Stand-in for a symbolic (dynamic) edge dim."""


class IoMismatchesTest(unittest.TestCase):
    def test_io_mismatches(self):
        # (name, coreai_io, edge_io, expected substring or None if compatible)
        cases = [
            ("compatible", [("f32", (2, 8))], [(torch.float32, (2, 8))], None),
            ("count", [("f32", (2, 8))], [], "count mismatch"),
            (
                "float dtype",
                [("f16", (2, 8))],
                [(torch.float32, (2, 8))],
                "dtype mismatch",
            ),
            ("int exact", [("si8", (4,))], [(torch.int8, (4,))], None),
            (
                "int64 narrowed",
                [("si32", (4,))],
                [(torch.int64, (4,))],
                "dtype mismatch",
            ),
            ("rank", [("f32", (2, 8))], [(torch.float32, (2,))], "rank mismatch"),
            (
                "static shape",
                [("f32", (2, 8))],
                [(torch.float32, (2, 16))],
                "static shape mismatch",
            ),
            (
                "symbolic dim skipped",
                [("f32", (2, 8))],
                [(torch.float32, (_Sym(), 8))],
                None,
            ),
            ("rank-0 compatible", [("f32", ())], [(torch.float32, ())], None),
            (
                "rank-0 vs rank-1",
                [("f32", ())],
                [(torch.float32, (1,))],
                "rank mismatch",
            ),
            ("both non-tensor", [("int", None)], [("int", None)], None),
            (
                "tensor vs non-tensor",
                [("!scalar", None)],
                [(torch.float32, (2, 8))],
                "non-tensor",
            ),
        ]
        for name, coreai_io, edge_io, expected in cases:
            with self.subTest(name):
                errs = io_mismatches(coreai_io, edge_io, "input")
                if expected is None:
                    self.assertEqual(errs, [], f"{name}: {errs}")
                else:
                    self.assertTrue(any(expected in e for e in errs), f"{name}: {errs}")


def _lower(
    module,
    example_inputs,
    dynamic_shapes=None,
    *,
    expect_delegates=1,
    allow_leftover=False,
):
    """Full AOT flow through Core AI; returns the lowered program.

    ``_skip_dim_order`` keeps ExecuTorch on ``aten._to_copy`` (which coreai
    supports) instead of emitting ``dim_order_ops._to_dim_order_copy``.

    Args:
        expect_delegates: assert this exact number of Core AI delegates; pass
            ``None`` to skip the count check.
        allow_leftover: if ``False`` (default), assert no ops remain outside the
            delegate (ignoring getitem and boundary ``_to_copy`` casts).  Set
            ``True`` for models Core AI only partially delegates (symint / i64
            producing ops).
    """
    ep = torch.export.export(
        module.eval(), example_inputs, dynamic_shapes=dynamic_shapes
    )
    lowered = to_edge_transform_and_lower(
        ep,
        partitioner=[CoreAIPartitioner()],
        transform_passes=get_default_passes(),
        compile_config=get_default_compile_config(),
    )
    gm = lowered.exported_program().graph_module
    delegates = [
        n
        for n in gm.graph.nodes
        if n.op == "call_function" and n.target is executorch_call_delegate
    ]
    leftover = [
        str(n.target)
        for n in gm.graph.nodes
        if n.op == "call_function"
        and n.target is not executorch_call_delegate
        and "getitem" not in str(n.target)
        and "_to_copy" not in str(n.target)  # boundary dtype-narrowing casts
    ]
    if expect_delegates is not None:
        assert (
            len(delegates) == expect_delegates
        ), f"expected {expect_delegates} Core AI delegate(s), got {len(delegates)}"
    if not allow_leftover:
        assert not leftover, f"not fully delegated; leftover ops: {leftover}"
    lowered.to_executorch()
    return lowered


class _Add(nn.Module):
    def forward(self, x):
        return x + x


class _And(nn.Module):
    def forward(self, x):
        return torch.logical_and(x, x)


class _AddScalar(nn.Module):
    def forward(self, x, n):
        return x + n


class _ConstIntOut(nn.Module):
    def forward(self, x):
        return x + x, 2


class _ConstFloatOut(nn.Module):
    def forward(self, x):
        return x + x, 4.0


class _Embedding(nn.Module):
    def __init__(self):
        super().__init__()
        self.emb = nn.Embedding(16, 4)

    def forward(self, tok):
        return self.emb(tok)


class _Rank0Out(nn.Module):
    def forward(self, x):
        return (x + x).sum()  # rank-0 (0-d) tensor output


class _ProducedInt64Out(nn.Module):
    def forward(self, x):
        # produced (non-const-folded) int64 output alongside a delegated tensor
        return x + x, (x > 0).to(torch.int64)


class BoundaryLoweringTest(unittest.TestCase):
    def test_supported_input_dtypes(self):
        # dtypes Core AI carries directly (no 64-bit narrowing); all fully
        # delegate.  int64/float64 (narrowed) are covered separately below.
        cases = {
            "float32": (_Add(), torch.randn(2, 8, dtype=torch.float32)),
            "float16": (_Add(), torch.randn(2, 8).to(torch.float16)),
            "bfloat16": (_Add(), torch.randn(2, 8).to(torch.bfloat16)),
            "int8": (_Add(), torch.randint(-8, 8, (2, 8), dtype=torch.int8)),
            "int16": (_Add(), torch.randint(-8, 8, (2, 8), dtype=torch.int16)),
            "int32": (_Add(), torch.randint(-8, 8, (2, 8), dtype=torch.int32)),
            "uint8": (_Add(), torch.randint(0, 8, (2, 8), dtype=torch.uint8)),
            "bool": (_And(), torch.randint(0, 2, (2, 8)).bool()),
        }
        for name, (module, x) in cases.items():
            with self.subTest(dtype=name):
                _lower(module, (x,))

    # 64-bit narrowing (input cast + output widen via the default pass).
    def test_float64(self):
        _lower(_Add(), (torch.randn(2, 8, dtype=torch.float64),))

    def test_int64(self):
        _lower(_Add(), (torch.randint(-8, 8, (2, 8), dtype=torch.int64),))

    def test_embedding_int64_index(self):
        # Real int64 case: token indices stay int64 at the model boundary; the
        # default narrow pass casts them to int32 for the delegated embedding.
        _lower(_Embedding(), (torch.randint(0, 16, (2, 8), dtype=torch.int64),))

    def test_symint_input(self):
        # A genuine dynamic scalar (SymInt) can't be a Core AI graph input; the
        # symint-consuming op is left outside the delegate, so the model lowers
        # (partial delegation) without crashing.
        _lower(
            _AddScalar(),
            (torch.randn(2, 8), 3),
            dynamic_shapes={"x": None, "n": Dim.DYNAMIC},
            allow_leftover=True,
        )

    # Non-tensor (const) outputs.
    def test_const_int_output(self):
        _lower(_ConstIntOut(), (torch.randn(2, 8),))

    def test_const_float_output(self):
        _lower(_ConstFloatOut(), (torch.randn(2, 8),))

    # Rank-0 (0-d) output.
    def test_rank0_output(self):
        _lower(_Rank0Out(), (torch.randn(2, 8),))

    # Produced (non-const-folded) int64 output.
    def test_produced_int64_output(self):
        # coreai narrows i64, so the i64-producing op stays outside the delegate
        # (guard) and runs portable; the model still lowers and the int64 output
        # dtype is preserved.
        lowered = _lower(
            _ProducedInt64Out(),
            (torch.randn(2, 8),),
            expect_delegates=None,  # partial: only the f32 part is delegated
            allow_leftover=True,
        )
        out_dtypes = [
            a.meta["val"].dtype
            for a in lowered.exported_program().graph_module.graph.output_node().args[0]
            if isinstance(a, torch.fx.Node)
            and isinstance(a.meta.get("val"), torch.Tensor)
        ]
        self.assertIn(torch.int64, out_dtypes)  # int64 output preserved


class _MutatedBuffer(nn.Module):
    """In-place buffer mutation, as a KV cache does."""

    def __init__(self):
        super().__init__()
        self.register_buffer("count", torch.zeros(8))

    def forward(self, x):
        self.count.add_(1.0)
        return x + self.count


class MutableBufferBoundaryTest(unittest.TestCase):
    """A mutated buffer crosses the boundary as an input as well as an output.

    coreai gives it a graph argument and a result, so both sides of the
    compatibility check have to count it.
    """

    def test_mutated_buffer_lowers(self):
        _lower(_MutatedBuffer(), (torch.randn(2, 8),))


if __name__ == "__main__":
    unittest.main()
