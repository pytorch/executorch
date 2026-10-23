# Copyright 2026 Arm Limited and/or its affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import pytest
import torch
from executorch.backends.arm._passes.exir_to_tosa_pass import ExirToTosaPass
from executorch.backends.arm._passes.quant_args import QuantArgs
from executorch.backends.arm._passes.rewrite_argmax_pass import RewriteArgmaxPass
from executorch.backends.arm.tosa.specification import (
    TosaLoweringContext,
    TosaSpecification,
)
from executorch.exir import to_edge
from executorch.exir.dialects._ops import ops as exir_ops


class ArgmaxToInt32(torch.nn.Module):
    def forward(self, tensor):
        return tensor.argmax(dim=1).to(torch.int32)


class CastToInt32(torch.nn.Module):
    def forward(self, tensor):
        return tensor.to(torch.int32)


class ArgmaxCastChain(torch.nn.Module):
    def forward(self, tensor):
        return tensor.argmax(dim=1).to(torch.int32).to(torch.float32)


class CastChain(torch.nn.Module):
    def forward(self, tensor):
        return tensor.to(torch.int32).to(torch.float32)


class PointwiseOperands(torch.nn.Module):
    def forward(self, tensor, other):
        return tensor * other, -tensor


@pytest.mark.parametrize(
    "module,expected_casts,expected_dtype",
    [
        (ArgmaxToInt32(), 0, torch.int32),
        (CastToInt32(), 1, torch.int32),
        (ArgmaxCastChain(), 1, torch.float32),
        (CastChain(), 2, torch.float32),
    ],
)
def test_lowering_preserves_remaining_dimension_order_copies(
    module, expected_casts, expected_dtype
):
    inputs = (torch.randn(2, 3),)
    program = to_edge(torch.export.export(module, inputs)).exported_program()
    with TosaLoweringContext(TosaSpecification.create_from_string("TOSA-1.0+FP")):
        rewritten = RewriteArgmaxPass()(program.graph_module)
        assert rewritten is not None
        result = ExirToTosaPass(program)(rewritten.graph_module)
    assert result is not None
    assert (
        sum(
            node.target == exir_ops.edge.dim_order_ops._to_dim_order_copy.default
            for node in result.graph_module.graph.nodes
        )
        == expected_casts
    )
    assert not any(
        node.target == exir_ops.backend.tosa.CAST.default
        for node in result.graph_module.graph.nodes
    )
    output = result.graph_module.graph.output_node().args[0][0]
    assert output.meta["val"].dtype == expected_dtype


@pytest.mark.parametrize("dtype", (torch.float32, torch.int8, torch.int16, torch.int32))
def test_pointwise_rewrites_materialize_explicit_operands(dtype):
    inputs = (torch.ones(2, 3, dtype=dtype), torch.ones(2, 3, dtype=dtype))
    program = to_edge(
        torch.export.export(PointwiseOperands(), inputs)
    ).exported_program()
    input_zp, output_zp = 0, 0
    if dtype == torch.int8:
        input_zp, output_zp = 3, -4
        negate = next(
            node
            for node in program.graph.nodes
            if node.target == exir_ops.edge.aten.neg.default
        )
        negate.meta["input_qparams"] = {0: QuantArgs(0.5, input_zp, -128, 127, dtype)}
        negate.meta["output_qparams"] = {0: QuantArgs(0.5, output_zp, -128, 127, dtype)}

    with TosaLoweringContext(TosaSpecification.create_from_string("TOSA-1.0+INT+FP")):
        result = ExirToTosaPass(program)(program.graph_module)
    assert result is not None
    result.graph_module.graph.lint()
    multiply = next(
        node
        for node in result.graph_module.graph.nodes
        if node.target == exir_ops.backend.tosa.MUL.default
    )
    assert len(multiply.args) == 3
    shift = multiply.args[2]
    assert shift.target == exir_ops.backend.tosa.CONST.default
    assert shift.args == ([0], [1], torch.int8)

    negate = next(
        node
        for node in result.graph_module.graph.nodes
        if node.target == exir_ops.backend.tosa.NEGATE.default
    )
    assert len(negate.args) == 3
    assert negate.args[1].args == ([input_zp], [1], dtype)
    assert negate.args[2].args == ([output_zp], [1], dtype)
