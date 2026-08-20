# Copyright 2026 Arm Limited and/or its affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import operator

import executorch.backends.arm.tosa.dialect  # noqa: F401
import torch
from executorch.backends.arm._passes.rewrite_mxfp_conv2d import RewriteMXFPConv2dPass
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


def _rewrite_conv2d_module(
    config: MXFPOpConfig,
    bias: bool = True,
    model_dtype: torch.dtype = torch.float32,
) -> tuple[torch.fx.GraphModule, list[torch.fx.Node], list[torch.fx.Node]]:
    model = _Conv2dModule(bias=bias).eval().to(model_dtype)
    to_mxfp(model, config, filter_fn=_is_conv2d)
    exported = export(
        model,
        (torch.randn(1, 32, 10, 12, dtype=model_dtype),),
        strict=False,
    )
    tosa_spec = TosaSpecification.create_from_string("TOSA-1.1+FP+mxfp")

    with TosaLoweringContext(tosa_spec):
        graph_module = (
            RewriteMXFPConv2dPass(exported).call(exported.graph_module).graph_module
        )

    cast_nodes = _nodes_from_target(
        graph_module, exir_ops.backend.tosa.CAST_TO_BLOCK_SCALED.default
    )
    conv_nodes = _nodes_from_target(
        graph_module, exir_ops.backend.tosa.CONV2D_BLOCK_SCALED.default
    )
    return graph_module, cast_nodes, conv_nodes


def test_rewrite_mxfp_conv2d_replaces_custom_op() -> None:
    graph_module, cast_nodes, conv_nodes = _rewrite_conv2d_module(MXFPOpConfig())
    targets = _targets(graph_module)

    assert torch.ops.tosa_mxfp.conv2d.default not in targets
    assert len(cast_nodes) == 1
    assert len(conv_nodes) == 1
    assert exir_ops.edge.aten.permute_copy.default in targets
    assert operator.getitem in targets

    cast_node = cast_nodes[0]
    assert tuple(cast_node.meta["val"][0].shape) == (1, 10, 12, 32)
    assert tuple(cast_node.meta["val"][1].shape) == (1, 10, 12, 1)

    conv_node = conv_nodes[0]
    assert tuple(conv_node.meta["val"].shape) == (1, 5, 6, 8)


def test_rewrite_mxfp_conv2d_restores_output_shape() -> None:
    graph_module, _cast_nodes, conv_nodes = _rewrite_conv2d_module(
        MXFPOpConfig(), bias=False
    )
    conv_node = conv_nodes[0]
    output_node = next(
        node
        for node in reversed(tuple(graph_module.graph.nodes))
        if node.op == "call_function"
        and node.target == exir_ops.edge.aten.permute_copy.default
    )

    assert tuple(conv_node.meta["val"].shape) == (1, 5, 6, 8)
    assert tuple(output_node.meta["val"].shape) == (1, 8, 5, 6)


def test_rewrite_mxfp_conv2d_preserves_inferred_bfloat16_output_cast() -> None:
    graph_module, _, conv_nodes = _rewrite_conv2d_module(
        MXFPOpConfig(),
        model_dtype=torch.bfloat16,
    )

    output_node = graph_module.graph.output_node()

    assert len(conv_nodes) == 1
    assert conv_nodes[0].meta["val"].dtype == torch.float32
    assert output_node.meta["val"][0].dtype == torch.bfloat16


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


def test_rewrite_mxfp6_conv2d_marks_payload_dtype() -> None:
    graph_module, cast_nodes, conv_nodes = _rewrite_conv2d_module(
        MXFPOpConfig(weight_dtype=DTYPE_FP6_E2M3)
    )
    cast_node = cast_nodes[0]
    conv_node = conv_nodes[0]
    input_qdata_node = next(
        node
        for node in graph_module.graph.nodes
        if node.op == "call_function"
        and node.target == operator.getitem
        and node.args[0] == cast_node
        and node.args[1] == 0
    )
    weight_qdata_node = conv_node.args[2]

    assert isinstance(input_qdata_node, torch.fx.Node)
    assert isinstance(weight_qdata_node, torch.fx.Node)
    assert cast_node.kwargs["output_dtype"] == mxfp_dtype_to_str(DTYPE_FP6_E2M3)
    assert conv_node.kwargs["payload_dtype"] == mxfp_dtype_to_str(DTYPE_FP6_E2M3)
    assert tuple(cast_node.meta["val"][0].shape) == (1, 10, 12, 32)
    assert (
        input_qdata_node.meta[TosaSpecialDtype.meta_key()] == TosaSpecialDtype.FP6E2M3
    )
    assert (
        weight_qdata_node.meta[TosaSpecialDtype.meta_key()] == TosaSpecialDtype.FP6E2M3
    )
    assert tuple(conv_node.meta["val"].shape) == (1, 5, 6, 8)
