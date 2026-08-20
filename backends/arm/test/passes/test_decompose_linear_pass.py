# Copyright 2026 Arm Limited and/or its affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from typing import Tuple

import torch
from executorch.backends.arm._passes import DecomposeLinearPass
from executorch.exir import to_edge
from executorch.exir.capture._config import EdgeCompileConfig
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
