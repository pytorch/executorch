# Copyright 2026 Arm Limited and/or its affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from typing import cast

import torch
from executorch.backends.arm._passes.match_arg_ranks_pass import MatchArgRanksPass
from executorch.backends.arm.tosa.specification import (
    TosaLoweringContext,
    TosaSpecification,
)
from executorch.exir import ExportedProgram
from executorch.exir.dialects._ops import ops as exir_ops
from torch._subclasses.fake_tensor import FakeTensorMode


def test_match_arg_ranks_handles_tosa_binary_ops() -> None:
    fake_mode = FakeTensorMode()
    graph = torch.fx.Graph()
    x = graph.placeholder("x")
    x.meta["val"] = fake_mode.from_tensor(torch.empty((1, 270, 480, 3, 2, 2)))
    y = graph.placeholder("y")
    y.meta["val"] = fake_mode.from_tensor(torch.empty((1, 1, 1, 1)))
    add = graph.call_function(exir_ops.backend.tosa.ADD.default, args=(x, y))
    add.meta["val"] = fake_mode.from_tensor(torch.empty((1, 270, 480, 3, 2, 2)))
    graph.output(add)
    graph_module = torch.fx.GraphModule(torch.nn.Module(), graph)

    with TosaLoweringContext(TosaSpecification.create_from_string("TOSA-1.0+FP")):
        result = MatchArgRanksPass(cast(ExportedProgram, None)).call(graph_module)
    call_nodes = [
        node for node in result.graph_module.graph.nodes if node.op == "call_function"
    ]
    view = next(
        node
        for node in call_nodes
        if node.target == exir_ops.edge.aten.view_copy.default
    )
    add = next(
        node for node in call_nodes if node.target == exir_ops.backend.tosa.ADD.default
    )

    assert result.modified
    assert view.all_input_nodes[0].target == "y"
    assert view.args[1] == [1, 1, 1, 1, 1, 1]
    assert add.all_input_nodes[0].target == "x"
    assert add.all_input_nodes[1] is view
