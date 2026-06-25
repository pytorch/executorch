# Copyright 2026 Arm Limited and/or its affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import sympy  # type: ignore[import-untyped]
import torch
from executorch.backends.arm._passes.arm_pass_utils import refresh_node_meta
from torch._subclasses.fake_tensor import FakeTensorMode
from torch.fx.experimental.symbolic_shapes import ShapeEnv


def test_refresh_node_meta_preserves_symbolic_shape() -> None:
    shape_env = ShapeEnv()
    batch = shape_env.create_symintnode(sympy.Symbol("batch"), hint=2)
    assert isinstance(batch, torch.SymInt)
    shape_env.constrain_symbol_range(batch.node.expr, compiler_min=1, compiler_max=8)
    with FakeTensorMode(shape_env=shape_env, allow_non_fake_inputs=True):
        x_val = torch.empty((batch, 3))
        y_val = torch.empty((1, 3))

    graph = torch.fx.Graph()
    x = graph.placeholder("x")
    x.meta["val"] = x_val
    y = graph.placeholder("y")
    y.meta["val"] = y_val
    add = graph.call_function(torch.ops.aten.add.Tensor, args=(x, y))
    add.meta["val"] = torch.empty((2, 3))

    refresh_node_meta(add)

    assert isinstance(add.meta["val"].shape[0], torch.SymInt)
    assert sympy.simplify(add.meta["val"].shape[0].node.expr - batch.node.expr) == 0


def test_refresh_node_meta_leaves_value_when_input_meta_is_missing() -> None:
    graph = torch.fx.Graph()
    x = graph.placeholder("x")
    relu = graph.call_function(torch.ops.aten.relu.default, args=(x,))
    original_val = torch.empty((2, 3))
    relu.meta["val"] = original_val

    refresh_node_meta(relu)

    assert relu.meta["val"] is original_val
