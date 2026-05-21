# Copyright 2026 Arm Limited and/or its affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import operator

import executorch.backends.arm.tosa.dialect  # noqa: F401
import torch
from executorch.backends.arm._passes.rewrite_mxfp_conv2d import RewriteMXFPConv2dPass
from executorch.backends.arm.ao_ext import MXFPOpConfig, to_mxfp
from executorch.backends.arm.tosa.mapping import TosaSpecialDtype
from executorch.backends.arm.tosa.specification import (
    TosaLoweringContext,
    TosaSpecification,
)
from executorch.exir.dialects._ops import ops as exir_ops
from torch.export import export


class _Conv2dModule(torch.nn.Module):
    def __init__(self, bias: bool = True) -> None:
        super().__init__()
        self.conv = torch.nn.Conv2d(
            32,
            8,
            kernel_size=3,
            stride=2,
            padding=1,
            bias=bias,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)


def _is_conv2d(module: torch.nn.Module, _fqn: str) -> bool:
    return isinstance(module, torch.nn.Conv2d)


def _targets(graph_module: torch.fx.GraphModule) -> list[object]:
    return [
        node.target for node in graph_module.graph.nodes if node.op == "call_function"
    ]


def _nodes_from_target(
    graph_module: torch.fx.GraphModule, target_op
) -> list[torch.fx.Node]:
    return [
        node
        for node in graph_module.graph.nodes
        if node.op == "call_function" and node.target == target_op
    ]


def test_rewrite_mxfp_conv2d_replaces_custom_op() -> None:
    model = _Conv2dModule(bias=True).eval()
    to_mxfp(model, MXFPOpConfig(), filter_fn=_is_conv2d)
    exported = export(model, (torch.randn(1, 32, 10, 12),), strict=False)
    tosa_spec = TosaSpecification.create_from_string("TOSA-1.1+FP+mxfp")

    with TosaLoweringContext(tosa_spec):
        graph_module = (
            RewriteMXFPConv2dPass(exported).call(exported.graph_module).graph_module
        )

    targets = _targets(graph_module)

    assert torch.ops.tosa_mxfp.conv2d.default not in targets
    assert exir_ops.backend.tosa.CAST_TO_BLOCK_SCALED.default in targets
    assert exir_ops.backend.tosa.CONV2D_BLOCK_SCALED.default in targets
    assert exir_ops.edge.aten.permute_copy.default in targets
    assert operator.getitem in targets


def test_rewrite_mxfp_conv2d_restores_output_shape() -> None:
    model = _Conv2dModule(bias=False).eval()
    to_mxfp(model, MXFPOpConfig(), filter_fn=_is_conv2d)
    exported = export(model, (torch.randn(1, 32, 10, 12),), strict=False)
    tosa_spec = TosaSpecification.create_from_string("TOSA-1.1+FP+mxfp")

    with TosaLoweringContext(tosa_spec):
        graph_module = (
            RewriteMXFPConv2dPass(exported).call(exported.graph_module).graph_module
        )

    conv_node = next(
        node
        for node in graph_module.graph.nodes
        if node.op == "call_function"
        and node.target == exir_ops.backend.tosa.CONV2D_BLOCK_SCALED.default
    )
    output_node = next(
        node
        for node in reversed(tuple(graph_module.graph.nodes))
        if node.op == "call_function"
        and node.target == exir_ops.edge.aten.permute_copy.default
    )

    assert tuple(conv_node.meta["val"].shape) == (1, 5, 6, 8)
    assert tuple(output_node.meta["val"].shape) == (1, 8, 5, 6)


def test_rewrite_mxfp4_conv2d_marks_payloads() -> None:
    model = _Conv2dModule(bias=True).eval()
    to_mxfp(
        model,
        MXFPOpConfig(weight_dtype=torch.float4_e2m1fn_x2),
        filter_fn=_is_conv2d,
    )
    exported = export(model, (torch.randn(1, 32, 10, 12),), strict=False)
    tosa_spec = TosaSpecification.create_from_string("TOSA-1.1+FP+mxfp")

    with TosaLoweringContext(tosa_spec):
        graph_module = (
            RewriteMXFPConv2dPass(exported).call(exported.graph_module).graph_module
        )

    conv_node = _nodes_from_target(
        graph_module, exir_ops.backend.tosa.CONV2D_BLOCK_SCALED.default
    )[0]
    input_qdata_node = conv_node.args[0]
    weight_qdata_node = conv_node.args[2]

    assert isinstance(input_qdata_node, torch.fx.Node)
    assert isinstance(weight_qdata_node, torch.fx.Node)
    assert tuple(input_qdata_node.meta["val"].shape) == (1, 10, 12, 16)
    assert tuple(weight_qdata_node.meta["val"].shape) == (8, 3, 3, 16)
    assert (
        input_qdata_node.meta[TosaSpecialDtype.meta_key()] == TosaSpecialDtype.FP4E2M1
    )
    assert (
        weight_qdata_node.meta[TosaSpecialDtype.meta_key()] == TosaSpecialDtype.FP4E2M1
    )
    assert tuple(conv_node.meta["val"].shape) == (1, 5, 6, 8)
