# Copyright 2026 Arm Limited and/or its affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import math
from typing import Tuple

import pytest
import torch
from executorch.backends.arm._passes import DecomposeLinearPass
from executorch.exir import to_edge
from executorch.exir.capture._config import EdgeCompileConfig
from executorch.exir.dialects._ops import ops as exir_ops
from torch.export import Dim, export


class Linear(torch.nn.Module):

    def __init__(self) -> None:
        super(Linear, self).__init__()
        self.linear = torch.nn.Linear(16, 16)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear(x)

    def get_example_inputs(self) -> Tuple[torch.Tensor]:
        return (torch.randn(1, 3, 16),)


def test_decompose_linear_dynamic() -> None:
    module = Linear()
    example_inputs = module.get_example_inputs()
    ep = export(
        module,
        example_inputs,
        strict=True,
        dynamic_shapes={
            "x": {
                1: Dim("batch_1", min=1, max=16),
            }
        },
    )
    edge_model = to_edge(
        ep,
        compile_config=EdgeCompileConfig(
            _core_aten_ops_exception_list=[torch.ops.aten.linear.default],
            preserve_ops=[torch.ops.aten.linear.default],
        ),
    )
    edge_model = edge_model.transform([DecomposeLinearPass()])
    output_node = edge_model.exported_program().graph_module.graph.output_node()
    assert output_node.op == "output"
    outputs = output_node.args[0]
    # Make sure the output shape has symbolic dimensions after the pass
    for output in outputs:
        assert any(isinstance(dim, torch.SymInt) for dim in output.meta["val"].shape)
    conv = next(
        node
        for node in edge_model.exported_program().graph.nodes
        if node.target == exir_ops.edge.aten.convolution.default
    )
    conv_shape = conv.args[0].meta["val"].shape
    assert tuple(conv_shape[:3]) == (1, 16, 1)
    assert isinstance(conv_shape[3], torch.SymInt)
    for rows in (1, 7):
        x = torch.randn(1, rows, 16)
        torch.testing.assert_close(edge_model.exported_program().module()(x), module(x))


@pytest.mark.parametrize("shape", [(7,), (5, 7), (2, 3, 7), (2, 3, 5, 7)])
@pytest.mark.parametrize("bias", [False, True])
def test_decompose_linear_spatial_rows(shape, bias):
    module = torch.nn.Linear(shape[-1], 11, bias=bias)
    x = torch.randn(*shape[:-1], 2 * shape[-1])[..., ::2]
    edge_model = to_edge(
        export(module, (x,), strict=True),
        compile_config=EdgeCompileConfig(
            _core_aten_ops_exception_list=[torch.ops.aten.linear.default],
            preserve_ops=[torch.ops.aten.linear.default],
        ),
    ).transform([DecomposeLinearPass()])
    ep = edge_model.exported_program()
    conv = next(
        node
        for node in ep.graph.nodes
        if node.target == exir_ops.edge.aten.convolution.default
    )
    rows = math.prod(shape[:-1])
    assert tuple(conv.args[0].meta["val"].shape) == (1, shape[-1], 1, rows)
    assert tuple(conv.args[1].meta["val"].shape) == (11, shape[-1], 1, 1)
    assert tuple(conv.meta["val"].shape) == (1, 11, 1, rows)
    torch.testing.assert_close(ep.module()(x), module(x), atol=1e-6, rtol=1e-5)
