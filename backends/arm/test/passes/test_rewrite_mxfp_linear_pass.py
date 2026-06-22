# Copyright 2026 Arm Limited and/or its affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import operator

import executorch.backends.arm.tosa.dialect  # noqa: F401
import torch
from executorch.backends.arm._passes.rewrite_mxfp_linear import RewriteMXFPLinearPass
from executorch.backends.arm.ao_ext import MXFPOpConfig, to_mxfp
from executorch.backends.arm.ao_ext.mxfp import mxfp_dtype_to_str
from executorch.backends.arm.tosa.mapping import TosaSpecialDtype
from executorch.backends.arm.tosa.specification import (
    TosaLoweringContext,
    TosaSpecification,
)
from executorch.exir.dialects._ops import ops as exir_ops
from torch.export import export
from torchao.prototype.mx_formats.mx_tensor import DTYPE_FP6_E2M3


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


def _rewrite_linear_module(
    config: MXFPOpConfig,
) -> tuple[torch.fx.GraphModule, list[torch.fx.Node], list[torch.fx.Node]]:
    model = _LinearModule(bias=True).eval()
    to_mxfp(model, config, filter_fn=_is_linear)
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
    return graph_module, cast_nodes, matmul_nodes


def test_rewrite_mxfp_linear_replaces_custom_op() -> None:
    graph_module, cast_nodes, matmul_nodes = _rewrite_linear_module(MXFPOpConfig())

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


def test_rewrite_mxfp6_linear_marks_payload_dtype() -> None:
    graph_module, cast_nodes, matmul_nodes = _rewrite_linear_module(
        MXFPOpConfig(weight_dtype=DTYPE_FP6_E2M3)
    )
    cast_node = cast_nodes[0]
    matmul_node = matmul_nodes[0]
    input_qdata_node = next(
        node
        for node in graph_module.graph.nodes
        if node.op == "call_function"
        and node.target == operator.getitem
        and node.args[0] == cast_node
        and node.args[1] == 0
    )
    weight_qdata_node = matmul_node.args[2]
    assert isinstance(weight_qdata_node, torch.fx.Node)

    assert cast_node.kwargs["output_dtype"] == mxfp_dtype_to_str(DTYPE_FP6_E2M3)
    assert matmul_node.kwargs["payload_dtype"] == mxfp_dtype_to_str(DTYPE_FP6_E2M3)
    assert tuple(cast_node.meta["val"][0].shape) == (1, 4 * 5, 32)
    assert (
        input_qdata_node.meta[TosaSpecialDtype.meta_key()] == TosaSpecialDtype.FP6E2M3
    )
    assert (
        weight_qdata_node.meta[TosaSpecialDtype.meta_key()] == TosaSpecialDtype.FP6E2M3
    )


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
