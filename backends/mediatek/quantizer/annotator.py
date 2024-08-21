# Copyright (c) 2024 MediaTek Inc.
#
# Licensed under the BSD License (the "License"); you may not use this file
# except in compliance with the License. See the license file in the root
# directory of this source tree for more details.

from typing import Callable, List

import torch

from torch._export import capture_pre_autograd_graph
from torch._ops import OpOverload
from torch._subclasses import FakeTensor

from torch.ao.quantization.quantizer import QuantizationAnnotation
from torch.ao.quantization.quantizer.utils import (
    _annotate_input_qspec_map,
    _annotate_output_qspec,
)
from torch.fx import Graph, Node
from torch.fx.passes.utils.matcher_with_name_node_map_utils import (
    SubgraphMatcherWithNameNodeMap,
)

from .qconfig import QuantizationConfig


OP_TO_ANNOTATOR = {}


def annotate(graph: Graph, quant_config: QuantizationConfig) -> None:
    # Pattern annotation
    _annotate_rmsnorm_pattern(graph, quant_config)
    _annotate_fused_activation_pattern(graph, quant_config)

    # Per-op annotation
    for node in graph.nodes:
        if node.op == "placeholder":
            annotate_placeholder(node, quant_config)
        elif node.op == "call_function":
            annotate_func = OP_TO_ANNOTATOR.get(node.target, None)
            if annotate_func is not None:
                annotate_func(node, quant_config)


def register_annotator(ops: List[OpOverload]):

    def decorator(annotator_fn: Callable):
        for op in ops:
            OP_TO_ANNOTATOR[op] = annotator_fn

    return decorator


def _is_annotated(node: Node):
    """
    Given a list of nodes (that represents an operator pattern),
    return True if any of the node
    is annotated, otherwise return False
    """
    KEY = "quantization_annotation"
    return KEY in node.meta and node.meta[KEY]._annotated


def _mark_as_annotated(nodes: List[Node]):
    KEY = "quantization_annotation"
    for node in nodes:
        if KEY not in node.meta:
            node.meta[KEY] = QuantizationAnnotation()
        node.meta[KEY]._annotated = True


def _is_float_activation_tensor(node: Node):
    if not isinstance(node, Node):
        return False
    if "val" not in node.meta:
        return False
    if not isinstance(node.meta["val"], FakeTensor):
        return False
    return node.meta["val"].dtype == torch.float32


def _annotate_fused_activation_pattern(
    graph: Graph, quant_config: QuantizationConfig
) -> None:
    for relu_node in graph.nodes:
        # Check relu/relu6 node
        if relu_node.op != "call_function":
            continue
        if relu_node.target not in [
            torch.ops.aten.relu.default,
            torch.ops.aten.relu_.default,
            torch.ops.aten.relu6.default,
        ]:
            continue

        producer_node = relu_node.args[0]
        if not isinstance(producer_node, Node):
            continue
        if producer_node.op != "call_function":
            continue
        if len(producer_node.users) != 1:
            continue

        # Handle affine + relu fusion
        if producer_node.target in [
            torch.ops.aten.conv1d.default,
            torch.ops.aten.conv2d.default,
            torch.ops.aten.linear.default,
        ]:
            weight_node = producer_node.args[1]
            _annotate_input_qspec_map(
                producer_node,
                weight_node,
                quant_config.weight,
            )
            _annotate_output_qspec(relu_node, quant_config.activation)
            _mark_as_annotated([producer_node, weight_node, relu_node])
            continue

        # Handle arithmetic + relu fusion
        if producer_node.target in [
            torch.ops.aten.add.Scalar,
            torch.ops.aten.add.Tensor,
            torch.ops.aten.add_.Scalar,
            torch.ops.aten.add_.Tensor,
            torch.ops.aten.div.Scalar,
            torch.ops.aten.div.Tensor,
            torch.ops.aten.div_.Scalar,
            torch.ops.aten.div_.Tensor,
            torch.ops.aten.divide.Scalar,
            torch.ops.aten.divide.Tensor,
            torch.ops.aten.mul.Scalar,
            torch.ops.aten.mul.Tensor,
            torch.ops.aten.mul_.Scalar,
            torch.ops.aten.mul_.Tensor,
            torch.ops.aten.rsub.Scalar,
            torch.ops.aten.rsub.Tensor,
            torch.ops.aten.sub.Scalar,
            torch.ops.aten.sub.Tensor,
            torch.ops.aten.sub_.Scalar,
            torch.ops.aten.sub_.Tensor,
        ]:
            _annotate_output_qspec(relu_node, quant_config.activation)
            _mark_as_annotated([producer_node, relu_node])
            continue


def _annotate_rmsnorm_pattern(graph: Graph, quant_config: QuantizationConfig) -> None:

    class ExecuTorchPattern(torch.nn.Module):
        def forward(self, x):
            norm = x * torch.rsqrt((x * x).mean(-1, keepdim=True) + 1e-6)
            return norm, {}

    class MTKPattern(torch.nn.Module):
        def forward(self, x):
            norm = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + 1e-6)
            return norm, {}

    for pattern_cls in (ExecuTorchPattern, MTKPattern):
        pattern_gm = capture_pre_autograd_graph(pattern_cls(), (torch.randn(3, 3),))
        matcher = SubgraphMatcherWithNameNodeMap(
            pattern_gm, ignore_literals=True, remove_overlapping_matches=False
        )
        matches = matcher.match(graph)
        for match in matches:
            target_nodes = []
            for node in match.nodes_map.values():
                if node in match.placeholder_nodes:
                    continue
                if node.op == "call_function" and node.target in OP_TO_ANNOTATOR:
                    target_nodes.append(node)

            if any(_is_annotated(node) for node in target_nodes):
                continue
            _mark_as_annotated(target_nodes)
            for node in match.returning_nodes:
                _annotate_output_qspec(node, quant_config.activation)


def annotate_placeholder(node: Node, quant_config: QuantizationConfig) -> None:
    if _is_annotated(node):
        return

    if _is_float_activation_tensor(node):
        _annotate_output_qspec(node, quant_config.activation)

    _mark_as_annotated([node])


@register_annotator(
    [
        torch.ops.aten.conv1d.default,
        torch.ops.aten.conv2d.default,
        torch.ops.aten.linear.default,
    ]
)
def annotate_affine_ops(node: Node, quant_config: QuantizationConfig) -> None:
    if _is_annotated(node):
        return

    weight_node = node.args[1]
    _annotate_input_qspec_map(
        node,
        weight_node,
        quant_config.weight,
    )
    _annotate_output_qspec(node, quant_config.activation)

    # Make weight as annotated because it is a constant node
    _mark_as_annotated([node, weight_node])


@register_annotator(
    [
        torch.ops.aten.add.Scalar,
        torch.ops.aten.add.Tensor,
        torch.ops.aten.add_.Scalar,
        torch.ops.aten.add_.Tensor,
        torch.ops.aten.bmm.default,
        torch.ops.aten.div.Scalar,
        torch.ops.aten.div.Tensor,
        torch.ops.aten.div_.Scalar,
        torch.ops.aten.div_.Tensor,
        torch.ops.aten.divide.Scalar,
        torch.ops.aten.divide.Tensor,
        torch.ops.aten.gelu.default,
        torch.ops.aten.group_norm.default,
        torch.ops.aten.layer_norm.default,
        torch.ops.aten.leaky_relu.default,
        torch.ops.aten.matmul.default,
        torch.ops.aten.mul.Scalar,
        torch.ops.aten.mul.Tensor,
        torch.ops.aten.mul_.Scalar,
        torch.ops.aten.mul_.Tensor,
        torch.ops.aten.pow.Scalar,
        torch.ops.aten.pow.Tensor_Scalar,
        torch.ops.aten.pow.Tensor_Tensor,
        torch.ops.aten.prelu.default,
        torch.ops.aten.rsub.Scalar,
        torch.ops.aten.rsub.Tensor,
        torch.ops.aten.silu.default,
        torch.ops.aten.sub.Scalar,
        torch.ops.aten.sub.Tensor,
        torch.ops.aten.sub_.Scalar,
        torch.ops.aten.sub_.Tensor,
    ]
)
def annotate_output_qspec(node: Node, quant_config: QuantizationConfig) -> None:
    if _is_annotated(node):
        return
    _annotate_output_qspec(node, quant_config.activation)
    _mark_as_annotated([node])


@register_annotator([torch.ops.aten.embedding.default])
def annotate_embedding_op(node: Node, quant_config: QuantizationConfig) -> None:
    if _is_annotated(node):
        return

    wgt_node = node.args[0]
    _annotate_input_qspec_map(node, wgt_node, quant_config.activation)
    _mark_as_annotated([node])
