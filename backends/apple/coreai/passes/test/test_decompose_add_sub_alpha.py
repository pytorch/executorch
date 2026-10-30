# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Unit tests for :class:`DecomposeAddSubAlphaPass`."""

import unittest

import torch
import torch.nn as nn

from executorch.backends.apple.coreai import (
    get_default_compile_config,
    get_default_passes,
)
from executorch.backends.apple.coreai.partition.partitioner import CoreAIPartitioner
from executorch.backends.apple.coreai.passes import (
    DecomposeAddSubAlphaPass,
    NarrowToCoreAIDtypesPass,
)
from executorch.exir import to_edge, to_edge_transform_and_lower
from executorch.exir.dialects._ops import ops as exir_ops
from torch.export.graph_signature import InputKind


class _AddAlpha(nn.Module):
    def __init__(self, alpha):
        super().__init__()
        self.alpha = alpha

    def forward(self, x, y):
        return torch.add(x, y, alpha=self.alpha)


class _SubAlpha(_AddAlpha):
    def forward(self, x, y):
        return torch.sub(x, y, alpha=self.alpha)


class _Add(nn.Module):
    def forward(self, x, y):
        return x + y


class _AddScalarOther(nn.Module):
    def forward(self, x):
        return x.add(2, alpha=3)


class _AddFloatScalarOther(nn.Module):
    def forward(self, x):
        return x.add(0.5, alpha=3)


def _aten_gm(module, inputs):
    return torch.export.export(module.eval(), inputs).graph_module


def _edge_gm(module, inputs):
    ep = torch.export.export(module.eval(), inputs)
    edge = to_edge(ep, compile_config=get_default_compile_config())
    return edge.exported_program().graph_module


def _call_nodes(gm, *targets):
    return [
        n for n in gm.graph.nodes if n.op == "call_function" and n.target in targets
    ]


class DecomposeAddSubAlphaPassTest(unittest.TestCase):
    def _assert_decomposed(self, gm, module, inputs, add_sub, mul):
        expected = module(*inputs)
        result = DecomposeAddSubAlphaPass()(gm)
        self.assertTrue(result.modified)

        (node,) = _call_nodes(gm, *add_sub)
        self.assertNotIn("alpha", node.kwargs)
        scaled = node.args[1]
        self.assertIs(scaled.target, mul)
        alpha = scaled.args[1]
        self.assertEqual(alpha.op, "get_attr")
        self.assertEqual(getattr(gm, alpha.target).shape, torch.Size([]))

        (actual,) = gm(*inputs)
        self.assertEqual(actual.dtype, expected.dtype)
        self.assertEqual(actual.shape, expected.shape)
        self.assertEqual(node.meta["val"].shape, expected.shape)
        torch.testing.assert_close(actual, expected)
        return node

    def test_aten_add_and_sub(self):
        aten = torch.ops.aten
        inputs = (torch.randn(2, 3), torch.randn(2, 3))
        for module in (_AddAlpha(2.5), _SubAlpha(-0.5)):
            with self.subTest(module=type(module).__name__):
                self._assert_decomposed(
                    _aten_gm(module, inputs),
                    module,
                    inputs,
                    (aten.add.Tensor, aten.sub.Tensor),
                    aten.mul.Tensor,
                )

    def test_edge_add_and_sub(self):
        edge = exir_ops.edge.aten
        inputs = (torch.randn(2, 3), torch.randn(2, 3))
        for module in (_AddAlpha(2.5), _SubAlpha(-0.5)):
            with self.subTest(module=type(module).__name__):
                self._assert_decomposed(
                    _edge_gm(module, inputs),
                    module,
                    inputs,
                    (edge.add.Tensor, edge.sub.Tensor),
                    edge.mul.Tensor,
                )

    def test_shapes_and_dtypes(self):
        edge = exir_ops.edge.aten
        cases = {
            "broadcast": (torch.randn(2, 1, 4), torch.randn(3, 1)),
            "scalar_output": (torch.randn(()), torch.randn(())),
            "int32": (
                torch.randint(-8, 8, (2, 3), dtype=torch.int32),
                torch.randint(-8, 8, (3,), dtype=torch.int32),
            ),
            "int_other_float_result": (
                torch.randn(2, 3),
                torch.randint(-8, 8, (2, 3), dtype=torch.int32),
            ),
            "f16_other_f32_result": (torch.randn(2, 3), torch.randn(3).half()),
            "f32_0dim_other_f16_result": (
                torch.randn(2, 3).half(),
                torch.randn(()),
            ),
        }
        for name, inputs in cases.items():
            alpha = 3 if inputs[0].dtype == torch.int32 else 1.75
            module = _SubAlpha(alpha)
            with self.subTest(case=name):
                self._assert_decomposed(
                    _edge_gm(module, inputs),
                    module,
                    inputs,
                    (edge.sub.Tensor,),
                    edge.mul.Tensor,
                )

    def test_casts_lower_precision_other_to_result_dtype(self):
        # f16 other against an f32 result: multiplying by a 0-dim f32 alpha
        # alone would stay in f16, so other must be cast to f32 first.
        inputs = (torch.randn(2, 3), torch.randn(3).half())
        gm = _aten_gm(_AddAlpha(1.75), inputs)
        DecomposeAddSubAlphaPass()(gm)
        (mul,) = _call_nodes(gm, torch.ops.aten.mul.Tensor)
        cast = mul.args[0]
        self.assertIs(cast.target, torch.ops.aten._to_copy.default)
        self.assertEqual(cast.kwargs["dtype"], torch.float32)
        self.assertEqual(mul.meta["val"].dtype, torch.float32)

    def test_unit_and_default_alpha_unchanged(self):
        inputs = (torch.randn(2, 3), torch.randn(2, 3))
        for module in (_Add(), _AddAlpha(1), _AddAlpha(1.0), _SubAlpha(1)):
            for make_gm in (_aten_gm, _edge_gm):
                with self.subTest(module=type(module).__name__, gm=make_gm.__name__):
                    gm = make_gm(module, inputs)
                    before = str(gm.graph)
                    result = DecomposeAddSubAlphaPass()(gm)
                    self.assertFalse(result.modified)
                    self.assertEqual(str(gm.graph), before)

    def test_scalar_other_folded_into_scalar(self):
        inputs = (torch.randn(2, 3),)
        gm = _aten_gm(_AddScalarOther(), inputs)
        result = DecomposeAddSubAlphaPass()(gm)
        self.assertTrue(result.modified)
        (add,) = _call_nodes(gm, torch.ops.aten.add.Tensor)
        self.assertEqual(add.args[1], 6)
        self.assertEqual(add.kwargs, {})
        self.assertFalse(_call_nodes(gm, torch.ops.aten.mul.Tensor))
        torch.testing.assert_close(gm(*inputs)[0], _AddScalarOther()(*inputs))

    def test_fp16_operands_share_result_dtype(self):
        # Each operand of the rewritten mul and add/sub must already be fp16, so
        # Core AI never sees a mixed-dtype binary op. In the edge graph a scalar
        # other is lifted as an f32/i64 constant, which must be cast to fp16.
        edge = exir_ops.edge.aten
        x, y = torch.randn(2, 3).half(), torch.randn(3).half()
        cases = {
            "tensor_other": (_SubAlpha(0.3), (x, y)),
            "float_scalar_other": (_AddFloatScalarOther(), (x,)),
            "int_scalar_other": (_AddScalarOther(), (x,)),
        }
        for name, (module, inputs) in cases.items():
            with self.subTest(case=name):
                ep = to_edge_transform_and_lower(
                    torch.export.export(module, inputs),
                    transform_passes=get_default_passes(),
                    compile_config=get_default_compile_config(),
                ).exported_program()

                (mul,) = _call_nodes(ep.graph_module, edge.mul.Tensor)
                (add_sub,) = _call_nodes(
                    ep.graph_module, edge.add.Tensor, edge.sub.Tensor
                )
                for node in (mul, add_sub):
                    dtypes = {a.meta["val"].dtype for a in node.args}
                    self.assertEqual(dtypes, {torch.float16})
                    self.assertEqual(node.meta["val"].dtype, torch.float16)

                alpha = ep.state_dict[mul.args[1].target]
                self.assertEqual(alpha.dtype, torch.float16)
                self.assertEqual(alpha.shape, torch.Size([]))
                torch.testing.assert_close(
                    ep.module()(*inputs), module(*inputs), rtol=2e-3, atol=2e-3
                )

    def test_registered_after_dtype_narrowing(self):
        passes = get_default_passes()
        types = [type(p) for p in passes]
        self.assertIn(DecomposeAddSubAlphaPass, types)
        self.assertLess(
            types.index(NarrowToCoreAIDtypesPass),
            types.index(DecomposeAddSubAlphaPass),
        )

    def test_alpha_lifted_to_constant_buffer(self):
        module = _AddAlpha(2.5)
        inputs = (torch.randn(2, 3), torch.randn(3))
        ep = to_edge_transform_and_lower(
            torch.export.export(module, inputs),
            transform_passes=get_default_passes(),
            compile_config=get_default_compile_config(),
        ).exported_program()

        specs = ep.graph_signature.input_specs
        user_inputs = [s for s in specs if s.kind == InputKind.USER_INPUT]
        self.assertEqual(len(user_inputs), 2)
        (buffer,) = [s for s in specs if s.kind == InputKind.BUFFER]
        torch.testing.assert_close(ep.state_dict[buffer.target], torch.tensor(2.5))
        self.assertFalse([n for n in ep.graph.nodes if n.op == "get_attr"])
        torch.testing.assert_close(ep.module()(*inputs), module(*inputs))

    def test_fully_delegated_to_coreai(self):
        for dtype in (torch.float32, torch.float16):
            with self.subTest(dtype=dtype):
                inputs = (torch.randn(2, 3, dtype=dtype), torch.randn(3, dtype=dtype))
                ep = to_edge_transform_and_lower(
                    torch.export.export(_SubAlpha(2.5), inputs),
                    partitioner=[CoreAIPartitioner()],
                    transform_passes=get_default_passes(),
                    compile_config=get_default_compile_config(),
                ).exported_program()
                ops = {
                    str(n.target) for n in ep.graph.nodes if n.op == "call_function"
                } - {"<built-in function getitem>"}
                self.assertEqual(ops, {"executorch_call_delegate"})


if __name__ == "__main__":
    unittest.main()
