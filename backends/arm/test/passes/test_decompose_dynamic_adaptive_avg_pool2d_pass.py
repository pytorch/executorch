# Copyright 2026 Arm Limited and/or its affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from typing import Tuple

import torch
from executorch.backends.arm._passes.decompose_dynamic_adaptive_avg_pool2d_pass import (
    DecomposeDynamicAdaptiveAvgPool2dPass,
)
from executorch.backends.arm.tosa.specification import (
    TosaLoweringContext,
    TosaSpecification,
)
from executorch.exir import to_edge
from executorch.exir.dialects._ops import ops as exir_ops
from torch._export.utils import _get_shape_env_from_gm
from torch.export import Dim, export

input_t = Tuple[torch.Tensor]


class AdaptiveAvgPoolDynamic(torch.nn.Module):
    def __init__(self, output_size: tuple[int | None, int | None] = (4, 4)):
        super().__init__()
        self.output_size = output_size

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.nn.functional.adaptive_avg_pool2d(x, self.output_size)


def _run_dynamic_decomposition(dynamic_shapes, output_size=(4, 4)):
    module = AdaptiveAvgPoolDynamic(output_size)
    example_inputs = (torch.rand(1, 3, 8, 8),)
    ep = export(module, example_inputs, dynamic_shapes=dynamic_shapes)
    edge_model = to_edge(ep)
    shape_env = _get_shape_env_from_gm(edge_model.exported_program().graph_module)

    with TosaLoweringContext(
        TosaSpecification.create_from_string("TOSA-1.1+FP+shape"), shape_env=shape_env
    ):
        edge_model = edge_model.transform([DecomposeDynamicAdaptiveAvgPool2dPass()])
    return list(edge_model.exported_program().graph.nodes)


def test_decompose_dynamic_adaptive_avg_pool2d_irregular_uses_tosa_adaptive():
    nodes = _run_dynamic_decomposition(
        {
            "x": {
                2: Dim("height", min=4, max=10),
                3: Dim("width", min=4, max=10),
            }
        }
    )

    assert not any(
        n.target == exir_ops.edge.aten._adaptive_avg_pool2d.default for n in nodes
    )
    assert (
        sum(
            n.target == exir_ops.backend.tosa.AVG_POOL2D_ADAPTIVE.default for n in nodes
        )
        == 16
    )
    assert sum(n.target == exir_ops.backend.tosa.SLICE.default for n in nodes) == 16
    assert sum(n.target == exir_ops.edge.aten.permute_copy.default for n in nodes) == 32
    assert any(n.target == exir_ops.backend.tosa.DIM.default for n in nodes)
    assert any(n.target == exir_ops.backend.tosa.DIV_FLOOR_SHAPE.default for n in nodes)
    assert any(n.target == exir_ops.backend.tosa.SUB_SHAPE.default for n in nodes)
    assert any(n.target == exir_ops.backend.tosa.CONCAT_SHAPE.default for n in nodes)


def test_rewrite_adaptive_avg_pool2d_does_not_require_dynamic_decompose_pass():
    from executorch.backends.arm._passes.rewrite_adaptive_avg_pool2d import (
        RewriteAdaptiveAvgPool2dPass,
    )

    assert (
        DecomposeDynamicAdaptiveAvgPool2dPass
        not in RewriteAdaptiveAvgPool2dPass._passes_required_after
    )


def test_decompose_dynamic_adaptive_avg_pool2d_requires_rewrite_adaptive_avg_pool2d():
    from executorch.backends.arm._passes.rewrite_adaptive_avg_pool2d import (
        RewriteAdaptiveAvgPool2dPass,
    )

    assert (
        RewriteAdaptiveAvgPool2dPass
        in DecomposeDynamicAdaptiveAvgPool2dPass._passes_required_after
    )
