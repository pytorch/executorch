# Copyright 2026 Arm Limited and/or its affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import sympy  # type: ignore
import torch
from executorch.backends.arm._passes import DecomposeSelectPass
from executorch.backends.test.program_builder import ProgramBuilder
from executorch.exir.dialects._ops import ops as exir_ops
from torch._subclasses.fake_tensor import FakeTensorMode
from torch.fx.experimental.symbolic_shapes import ShapeEnv


def _make_symint(
    shape_env: ShapeEnv, symbol: str, hint: int, min: int = 1, max: int = 64
) -> torch.SymInt:
    symint = shape_env.create_symintnode(sympy.Symbol(symbol), hint=hint)
    assert isinstance(symint, torch.SymInt)
    shape_env.constrain_symbol_range(
        symint.node.expr, compiler_min=min, compiler_max=max
    )
    return symint


def test_decompose_select_negative_symbolic_index_uses_symbolic_sub() -> None:
    shape_env = ShapeEnv()
    seq = _make_symint(shape_env, "seq", hint=4)

    with FakeTensorMode(shape_env=shape_env) as mode:
        builder = ProgramBuilder(fake_tensor_mode=mode)
        x = builder.placeholder("x", mode.from_tensor(torch.empty(size=(1, seq, 576))))
        h = builder.call_operator(exir_ops.edge.aten.add.Tensor, (x, x))
        select = builder.call_operator(exir_ops.edge.aten.select_copy.int, (h, 1, -1))
        builder.output([select])

        result = DecomposeSelectPass()(builder.get_program().graph_module)

    assert result is not None

    select_nodes = [
        node
        for node in result.graph_module.graph.nodes
        if node.op == "call_function"
        and node.target == exir_ops.edge.aten.select_copy.int
    ]
    slice_nodes = [
        node
        for node in result.graph_module.graph.nodes
        if node.op == "call_function"
        and node.target == exir_ops.edge.aten.slice_copy.Tensor
    ]
    squeeze_nodes = [
        node
        for node in result.graph_module.graph.nodes
        if node.op == "call_function"
        and node.target == exir_ops.edge.aten.squeeze_copy.dims
    ]

    assert not select_nodes
    assert len(slice_nodes) == 1
    assert len(squeeze_nodes) == 1

    slice_node = slice_nodes[0]
    assert slice_node.args[1] == 1
    # Start/end are now FX nodes (materialized from SymInts) rather than raw
    # SymInts, since Graph.create_node rejects raw symbolic leaves in
    # call_function args. The original SymInt is preserved in meta['val'].
    start_arg, end_arg = slice_node.args[2], slice_node.args[3]
    assert isinstance(start_arg, torch.fx.Node)
    assert isinstance(end_arg, torch.fx.Node)
    start_val = start_arg.meta["val"]
    end_val = end_arg.meta["val"]
    assert isinstance(start_val, torch.SymInt)
    assert isinstance(end_val, torch.SymInt)
    assert str(start_val).endswith(" - 1")
    assert str(end_val) in str(start_val)
    assert squeeze_nodes[0].args == (slice_node, [1])

    result.graph_module.graph.lint()
