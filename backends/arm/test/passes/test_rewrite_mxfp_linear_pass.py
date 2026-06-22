# Copyright 2026 Arm Limited and/or its affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import operator

import executorch.backends.arm.tosa.dialect  # noqa: F401
import torch
from executorch.backends.arm._passes.rewrite_mxfp_linear import RewriteMXFPLinearPass
from executorch.backends.arm.ao_ext import MXFPOpConfig, to_mxfp
from executorch.backends.arm.tosa.specification import (
    TosaLoweringContext,
    TosaSpecification,
)
from executorch.exir.dialects._ops import ops as exir_ops
from torch.export import export


class _LinearModule(torch.nn.Module):
    def __init__(self, bias: bool = True) -> None:
        super().__init__()
        self.linear = torch.nn.Linear(32, 8, bias=bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear(x)


class _DualLinearModule(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.linear = torch.nn.Linear(32, 8, bias=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear(x) + self.linear(x)


def _is_linear(module: torch.nn.Module, _fqn: str) -> bool:
    return isinstance(module, torch.nn.Linear)


def _get_nodes_from_target(
    graph_module: torch.fx.GraphModule, target_op
) -> list[torch.fx.Node]:
    return [
        node
        for node in graph_module.graph.nodes
        if node.op == "call_function" and node.target == target_op
    ]


def test_rewrite_mxfp_linear_replaces_custom_op() -> None:
    model = _LinearModule(bias=True).eval()
    to_mxfp(model, MXFPOpConfig(), filter_fn=_is_linear)
    exported = export(model, (torch.randn(4, 5, 32),), strict=False)
    tosa_spec = TosaSpecification.create_from_string("TOSA-1.1+FP+mxfp")

    with TosaLoweringContext(tosa_spec):
        graph_module = (
            RewriteMXFPLinearPass(exported).call(exported.graph_module).graph_module
        )

    cast_nodes = _get_nodes_from_target(
        graph_module, exir_ops.backend.tosa.CAST_TO_BLOCK_SCALED.default
    )
    matmul_nodes = _get_nodes_from_target(
        graph_module, exir_ops.backend.tosa.MATMUL_T_BLOCK_SCALED.default
    )

    assert (
        len(_get_nodes_from_target(graph_module, torch.ops.tosa_mxfp.linear.default))
        == 0
    )
    assert len(cast_nodes) == 1
    assert len(matmul_nodes) == 1
    assert len(_get_nodes_from_target(graph_module, exir_ops.edge.aten.add.Tensor)) == 1
    # One getitem for each of the two outputs of CAST_TO_BLOCK_SCALED
    assert len(_get_nodes_from_target(graph_module, operator.getitem)) == 2

    cast_node = cast_nodes[0]
    assert tuple(cast_node.meta["val"][0].shape) == (1, 4 * 5, 32)  # Output data vector
    assert tuple(cast_node.meta["val"][1].shape) == (1, 4 * 5, 1)  # Output scale vector

    matmul_node = matmul_nodes[0]
    assert tuple(matmul_node.meta["val"].shape) == (1, 4 * 5, 8)

    output_node = graph_module.graph.output_node()
    assert tuple(output_node.meta["val"][0].shape) == (4, 5, 8)


def test_rewrite_mxfp_dual_linear() -> None:
    model = _DualLinearModule().eval()
    to_mxfp(model, MXFPOpConfig(), filter_fn=_is_linear)
    exported = export(model, (torch.randn(4, 32),), strict=False)
    tosa_spec = TosaSpecification.create_from_string("TOSA-1.1+FP+mxfp")

    with TosaLoweringContext(tosa_spec):
        graph_module = (
            RewriteMXFPLinearPass(exported).call(exported.graph_module).graph_module
        )

    assert (
        len(_get_nodes_from_target(graph_module, torch.ops.tosa_mxfp.linear.default))
        == 0
    )
    assert (
        len(
            _get_nodes_from_target(
                graph_module, exir_ops.backend.tosa.CAST_TO_BLOCK_SCALED.default
            )
        )
        == 2
    )
    assert (
        len(
            _get_nodes_from_target(
                graph_module, exir_ops.backend.tosa.MATMUL_T_BLOCK_SCALED.default
            )
        )
        == 2
    )
