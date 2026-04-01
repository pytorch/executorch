# Copyright 2026 Arm Limited and/or its affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import torch
from executorch.backends.arm._passes.insert_dynamic_padding import (
    InsertDynamicPaddingPass,
)
from executorch.backends.arm._passes.rewrite_conv_pass import RewriteConvPass
from executorch.backends.arm.tosa.specification import (
    TosaLoweringContext,
    TosaSpecification,
)
from executorch.exir import to_edge
from executorch.exir.dialects._ops import ops as exir_ops
from torch.export import Dim, export


class ConvModule(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(3, 16, kernel_size=2, stride=3, padding=2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)


def test_insert_dynamic_padding_no_target():
    model = ConvModule()
    example_inputs = (torch.randn(1, 3, 8, 8),)
    ep = export(
        model,
        example_inputs,
        dynamic_shapes={
            "x": {2: Dim("height", min=4, max=10), 3: Dim("width", min=4, max=10)}
        },
    )
    edge_model = to_edge(ep)
    with TosaLoweringContext(TosaSpecification.create_from_string("TOSA-1.1+FP+shape")):
        edge_model = edge_model.transform(
            [RewriteConvPass(edge_model.exported_program())]
        )
        nodes = edge_model.exported_program().graph.nodes
        conv_node = next(
            n for n in nodes if n.target == exir_ops.backend.tosa.CONV2D.default
        )
        initial_padding = conv_node.args[4]
        assert any(isinstance(p, torch.SymInt) for p in initial_padding)

        edge_model = edge_model.transform(
            [
                InsertDynamicPaddingPass(),
            ]
        )
        nodes = edge_model.exported_program().graph.nodes
        conv_node = next(
            n for n in nodes if n.target == exir_ops.backend.tosa.CONV2D.default
        )
        padding = conv_node.args[4]
        assert padding == [0, 0, 0, 0]
        padding_node = next(
            n for n in nodes if n.target == exir_ops.backend.tosa.PAD.default
        )
        assert padding_node is not None
        pad_list = padding_node.args[1].meta["val"]
        assert len(pad_list) == 8
        assert pad_list[:4] == [0, 0, 0, 0]  # NC-padding
        assert pad_list[4:] == initial_padding  # HW-padding
