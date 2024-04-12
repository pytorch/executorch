# Copyright (c) Qualcomm Innovation Center, Inc.
# All rights reserved
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.
from dataclasses import dataclass
from typing import Callable, Dict, List, Optional, Sequence

import torch

from torch._ops import OpOverload
from torch._subclasses import FakeTensor

from torch.ao.quantization.quantizer import (
    QuantizationAnnotation,
    QuantizationSpec,
    SharedQuantizationSpec,
)

from torch.ao.quantization.quantizer.utils import (
    _annotate_input_qspec_map,
    _annotate_output_qspec,
)
from torch.fx import Node

QUANT_ANNOTATION_KEY = "quantization_annotation"
OP_ANNOTATOR: Dict[OpOverload, Callable] = {}


@dataclass(eq=True, frozen=True)
class QuantizationConfig:
    input_activation: Optional[QuantizationSpec]
    output_activation: Optional[QuantizationSpec]
    weight: Optional[QuantizationSpec]
    bias: Optional[QuantizationSpec | Callable]


def register_annotator(ops: List[OpOverload]):
    def decorator(annotator: Callable):
        for op in ops:
            OP_ANNOTATOR[op] = annotator

    return decorator


def _is_input_float_tensor(node: Node):
    """Check if the input is not a float tensor, so that we can skip quantization for the node
    since observers only works with float Tensors
    """
    if (
        not isinstance(node, Node)
        or "val" not in node.meta
        or not isinstance(node.meta["val"], FakeTensor)
    ):
        return False
    return node.meta["val"].dtype == torch.float32


def _is_annotated(nodes: List[Node]):
    """
    Given a list of nodes (that represents an operator pattern),
    return True if any of the node
    is annotated, otherwise return False
    """
    annotated = False
    for node in nodes:
        annotated = annotated or (
            QUANT_ANNOTATION_KEY in node.meta
            and node.meta[QUANT_ANNOTATION_KEY]._annotated
        )
    return annotated


def _mark_nodes_as_annotated(nodes: List[Node]):
    for node in nodes:
        if QUANT_ANNOTATION_KEY not in node.meta:
            node.meta[QUANT_ANNOTATION_KEY] = QuantizationAnnotation()
        node.meta[QUANT_ANNOTATION_KEY]._annotated = True


def annotate_in_out_obs_sharing_op(
    node: Node, quantization_config: QuantizationConfig
) -> None:
    if _is_annotated([node]):
        return

    input_act = node.args[0]
    assert isinstance(input_act, Node)

    # only annotate input output sharing operator
    # when the output of the input node is annotated
    if (
        QUANT_ANNOTATION_KEY not in input_act.meta
        or not input_act.meta[QUANT_ANNOTATION_KEY]._annotated
        or input_act.meta[QUANT_ANNOTATION_KEY].output_qspec is None
    ):
        return

    act_qspec = SharedQuantizationSpec(input_act)
    node.meta[QUANT_ANNOTATION_KEY] = QuantizationAnnotation(
        input_qspec_map={
            input_act: act_qspec,
        },
        output_qspec=act_qspec,
        _annotated=True,
    )


def annotate_single_in_single_out(
    node: Node, quantization_config: QuantizationConfig
) -> None:
    if _is_annotated([node]):
        return

    input_qspec_map = {}
    input_act = node.args[0]
    assert isinstance(input_act, Node)
    input_qspec_map[input_act] = quantization_config.input_activation

    node_tensor = node.meta.get("val")
    if torch.is_tensor(node_tensor) and node_tensor.dtype != torch.float32:
        return

    node.meta[QUANT_ANNOTATION_KEY] = QuantizationAnnotation(
        input_qspec_map=input_qspec_map,
        output_qspec=quantization_config.output_activation,
        _annotated=True,
    )


def annotate_binary(node: Node, quantization_config: QuantizationConfig) -> None:
    if _is_annotated([node]):
        return

    input_act_qspec = quantization_config.input_activation
    output_act_qspec = quantization_config.output_activation

    input_qspec_map = {}
    input_act0 = node.args[0]
    if _is_input_float_tensor(input_act0):
        input_qspec_map[input_act0] = input_act_qspec

    input_act1 = node.args[1]
    if _is_input_float_tensor(input_act1):
        input_qspec_map[input_act1] = input_act_qspec

    node.meta[QUANT_ANNOTATION_KEY] = QuantizationAnnotation(
        input_qspec_map=input_qspec_map,
        output_qspec=output_act_qspec,
        _annotated=True,
    )


@register_annotator([torch.ops.aten.add.Tensor])
def annotate_add(node: Node, quantization_config: QuantizationConfig) -> None:
    annotate_binary(node, quantization_config)


@register_annotator([torch.ops.aten.sub.Tensor])
def annotate_sub(node: Node, quantization_config: QuantizationConfig) -> None:
    annotate_binary(node, quantization_config)


@register_annotator([torch.ops.aten.mul.Tensor, torch.ops.aten.mul.Scalar])
def annotate_mul(node: Node, quantization_config: QuantizationConfig) -> None:
    annotate_binary(node, quantization_config)


@register_annotator([torch.ops.aten.div.Tensor, torch.ops.aten.divide.Tensor])
def annotate_div(node: Node, quantization_config: QuantizationConfig) -> None:
    annotate_binary(node, quantization_config)


@register_annotator([torch.ops.aten.rsub.Scalar])
def annotate_rsub(node: Node, quantization_config: QuantizationConfig) -> None:
    annotate_binary(node, quantization_config)


@register_annotator([torch.ops.aten.ceil.default])
def annotate_ceil(node: Node, quantization_config: QuantizationConfig) -> None:
    annotate_single_in_single_out(node, quantization_config)


@register_annotator([torch.ops.aten.clamp.default])
def annotate_clamp(node: Node, quantization_config: QuantizationConfig) -> None:
    annotate_single_in_single_out(node, quantization_config)


@register_annotator([torch.ops.aten.relu.default, torch.ops.aten.relu_.default])
def annotate_relu(node: Node, quantization_config: QuantizationConfig) -> None:
    annotate_single_in_single_out(node, quantization_config)


@register_annotator([torch.ops.aten.tanh.default])
def annotate_tanh(node: Node, quantization_config: QuantizationConfig) -> None:
    annotate_single_in_single_out(node, quantization_config)


@register_annotator(
    [torch.ops.aten.hardswish.default, torch.ops.aten.hardswish_.default]
)
def annotate_hardswish(node: Node, quantization_config: QuantizationConfig) -> None:
    annotate_single_in_single_out(node, quantization_config)


@register_annotator(
    [torch.ops.aten.hardsigmoid.default, torch.ops.aten.hardsigmoid_.default]
)
def annotate_hardsigmoid(node: Node, quantization_config: QuantizationConfig) -> None:
    annotate_single_in_single_out(node, quantization_config)


@register_annotator([torch.ops.aten.hardtanh.default, torch.ops.aten.hardtanh_.default])
def annotate_hardtanh(node: Node, quantization_config: QuantizationConfig) -> None:
    annotate_single_in_single_out(node, quantization_config)


@register_annotator([torch.ops.aten.mean.default])
def annotate_mean(node: Node, quantization_config: QuantizationConfig) -> None:
    annotate_single_in_single_out(node, quantization_config)


@register_annotator([torch.ops.aten.max_pool2d.default])
def annotate_max_pool2d(node: Node, quantization_config: QuantizationConfig) -> None:
    annotate_single_in_single_out(node, quantization_config)


@register_annotator([torch.ops.aten.max_pool2d_with_indices.default])
def annotate_max_pool2d_with_indices(
    node: Node, quantization_config: QuantizationConfig
) -> None:
    annotate_single_in_single_out(node, quantization_config)


@register_annotator([torch.ops.aten.adaptive_avg_pool2d.default])
def annotate_adaptive_avgpool2d(
    node: Node, quantization_config: QuantizationConfig
) -> None:
    annotate_single_in_single_out(node, quantization_config)


@register_annotator([torch.ops.aten.avg_pool2d.default])
def annotate_avgpool2d(node: Node, quantization_config: QuantizationConfig) -> None:
    annotate_single_in_single_out(node, quantization_config)


@register_annotator([torch.ops.aten.permute.default])
def annotate_permute(node: Node, quantization_config: QuantizationConfig) -> None:
    annotate_single_in_single_out(node, quantization_config)


@register_annotator([torch.ops.aten.view.default])
def annotate_view(node: Node, quantization_config: QuantizationConfig) -> None:
    annotate_single_in_single_out(node, quantization_config)


@register_annotator([torch.ops.aten.pixel_shuffle.default])
def annotate_pixel_shuffle_default(
    node: Node, quantization_config: QuantizationConfig
) -> None:
    annotate_single_in_single_out(node, quantization_config)


@register_annotator([torch.ops.aten.upsample_bilinear2d.vec])
def annotate_upsample_bilinear2d(
    node: Node, quantization_config: QuantizationConfig
) -> None:
    annotate_single_in_single_out(node, quantization_config)


@register_annotator([torch.ops.aten.softmax.int, torch.ops.aten._softmax.default])
def annotate_softmax(node: Node, quantization_config: QuantizationConfig) -> None:
    annotate_single_in_single_out(node, quantization_config)


@register_annotator([torch.ops.aten.log_softmax.int])
def annotate_log_softmax(node: Node, quantization_config: QuantizationConfig) -> None:
    annotate_single_in_single_out(node, quantization_config)


@register_annotator([torch.ops.aten.pad.default])
def annotate_pad(node: Node, quantization_config: QuantizationConfig) -> None:
    annotate_single_in_single_out(node, quantization_config)


@register_annotator([torch.ops.aten.reshape.default])
def annotate_reshape(node: Node, quantization_config: QuantizationConfig) -> None:
    annotate_single_in_single_out(node, quantization_config)


@register_annotator([torch.ops.aten.select.int])
def annotate_select(node: Node, quantization_config: QuantizationConfig) -> None:
    annotate_single_in_single_out(node, quantization_config)


@register_annotator([torch.ops.aten.mean.dim])
def annotate_mean_dim(node: Node, quantization_config: QuantizationConfig) -> None:
    annotate_single_in_single_out(node, quantization_config)


@register_annotator([torch.ops.aten.slice.Tensor])
def annotate_slice(node: Node, quantization_config: QuantizationConfig) -> None:
    annotate_single_in_single_out(node, quantization_config)


@register_annotator([torch.ops.aten.gelu.default])
def annotate_gelu(node: Node, quantization_config: QuantizationConfig) -> None:
    annotate_single_in_single_out(node, quantization_config)


@register_annotator([torch.ops.aten.scaled_dot_product_attention.default])
def annotate_scaled_dot_product_attention(
    node: Node, quantization_config: QuantizationConfig
) -> None:
    annotate_single_in_single_out(node, quantization_config)


@register_annotator(
    [
        torch.ops.aten.squeeze.default,
        torch.ops.aten.squeeze.dim,
        torch.ops.aten.squeeze_copy.dims,
    ]
)
def annotate_squeeze(node: Node, quantization_config: QuantizationConfig) -> None:
    annotate_in_out_obs_sharing_op(node, quantization_config)
    if not _is_annotated([node]):
        annotate_single_in_single_out(node, quantization_config)


@register_annotator([torch.ops.aten.rsqrt.default])
def annotate_rsqrt(node: Node, quantization_config: QuantizationConfig) -> None:
    annotate_single_in_single_out(node, quantization_config)


@register_annotator([torch.ops.aten.sigmoid, torch.ops.aten.sigmoid.default])
def annotate_sigmoid(node: Node, quantization_config: QuantizationConfig) -> None:
    annotate_single_in_single_out(node, quantization_config)


@register_annotator([torch.ops.aten.pow.Tensor_Scalar])
def annotate_pow(node: Node, quantization_config: QuantizationConfig) -> None:
    annotate_single_in_single_out(node, quantization_config)


@register_annotator([torch.ops.aten.unsqueeze.default])
def annotate_unsqueeze(node: Node, quantization_config: QuantizationConfig) -> None:
    annotate_in_out_obs_sharing_op(node, quantization_config)
    if not _is_annotated([node]):
        annotate_single_in_single_out(node, quantization_config)


@register_annotator(
    [
        torch.ops.aten.unsqueeze_copy.default,
    ]
)
def annotate_unsqueeze_copy(
    node: Node, quantization_config: QuantizationConfig
) -> None:
    annotate_in_out_obs_sharing_op(node, quantization_config)
    if not _is_annotated([node]):
        annotate_single_in_single_out(node, quantization_config)


@register_annotator([torch.ops.aten.transpose.int])
def annotate_transpose(node: Node, quantization_config: QuantizationConfig) -> None:
    annotate_in_out_obs_sharing_op(node, quantization_config)
    if not _is_annotated([node]):
        annotate_single_in_single_out(node, quantization_config)


@register_annotator([torch.ops.aten.embedding.default])
def annotate_embedding(node: Node, quantization_config: QuantizationConfig) -> None:
    weight = node.args[0]

    input_qspec_map = {}
    input_qspec_map[weight] = quantization_config.input_activation

    node.meta[QUANT_ANNOTATION_KEY] = QuantizationAnnotation(
        input_qspec_map=input_qspec_map,
        output_qspec=SharedQuantizationSpec((weight, node)),
        _annotated=True,
    )


@register_annotator([torch.ops.aten.expand.default])
def annotate_expand(node: Node, quantization_config: QuantizationConfig) -> None:
    annotate_in_out_obs_sharing_op(node, quantization_config)
    if not _is_annotated([node]):
        annotate_single_in_single_out(node, quantization_config)


@register_annotator([torch.ops.aten.flatten.using_ints])
def annotate_flatten(node: Node, quantization_config: QuantizationConfig) -> None:
    annotate_in_out_obs_sharing_op(node, quantization_config)
    if not _is_annotated([node]):
        annotate_single_in_single_out(node, quantization_config)


@register_annotator([torch.ops.aten.stack.default])
def annotate_stack(node: Node, quantization_config: QuantizationConfig) -> None:
    input_qspec_map = {}
    for input_act in node.args[0]:
        assert isinstance(input_act, Node)
        input_qspec_map[input_act] = quantization_config.input_activation

        node_tensor = node.meta.get("val")
        if torch.is_tensor(node_tensor) and node_tensor.dtype == torch.int64:
            continue

    node.meta[QUANT_ANNOTATION_KEY] = QuantizationAnnotation(
        input_qspec_map=input_qspec_map,
        output_qspec=quantization_config.output_activation,
        _annotated=True,
    )


@register_annotator([torch.ops.aten.matmul.default])
def annotate_matmul(node: Node, quantization_config: QuantizationConfig) -> None:
    if _is_annotated([node]):
        return

    input_act_qspec = quantization_config.input_activation
    output_act_qspec = quantization_config.output_activation

    input_qspec_map = {}
    input_act0 = node.args[0]
    if isinstance(input_act0, Node):
        input_qspec_map[input_act0] = input_act_qspec

    input_act1 = node.args[1]
    if isinstance(input_act1, Node):
        # In matmul, QNN_DATATYPE_SFIXED_POINT_16 Input1 must have QNN_DATATYPE_UFIXED_POINT_16 Input0 and must be symmetric quantized.
        if input_act_qspec.dtype == torch.int32:
            input_qspec_map[input_act1] = quantization_config.weight
            quantization_annotation = input_act1.meta.get(QUANT_ANNOTATION_KEY, None)
            if quantization_annotation:
                quantization_annotation.output_qspec = quantization_config.weight
        else:
            input_qspec_map[input_act1] = input_act_qspec

    node.meta[QUANT_ANNOTATION_KEY] = QuantizationAnnotation(
        input_qspec_map=input_qspec_map,
        output_qspec=output_act_qspec,
        _annotated=True,
    )


@register_annotator([torch.ops.aten.bmm.default])
def annotate_bmm(node: Node, quantization_config: QuantizationConfig) -> None:
    if _is_annotated([node]):
        return

    input_act_qspec = quantization_config.input_activation
    output_act_qspec = quantization_config.output_activation

    input_qspec_map = {}
    input_act0 = node.args[0]
    if isinstance(input_act0, Node):
        input_qspec_map[input_act0] = input_act_qspec

    input_act1 = node.args[1]
    if isinstance(input_act1, Node):
        # In bmm, QNN_DATATYPE_SFIXED_POINT_16 Input1 must have QNN_DATATYPE_UFIXED_POINT_16 Input0 and must be symmetric quantized.
        if input_act_qspec.dtype == torch.int32:
            input_qspec_map[input_act1] = quantization_config.weight
            quantization_annotation = input_act1.meta.get(QUANT_ANNOTATION_KEY, None)
            if quantization_annotation:
                quantization_annotation.output_qspec = quantization_config.weight
        else:
            input_qspec_map[input_act1] = input_act_qspec

    node.meta[QUANT_ANNOTATION_KEY] = QuantizationAnnotation(
        input_qspec_map=input_qspec_map,
        output_qspec=output_act_qspec,
        _annotated=True,
    )

    # We use get_source_partition in pass, but it is the same source for MultiheadAttention, so we need to change its source_fn_stack.
    node.meta["source_fn_stack"] = [(node, torch.bmm)]


@register_annotator([torch.ops.aten.conv2d.default, torch.ops.aten.conv1d.default])
def annotate_conv2d(node: Node, quantization_config: QuantizationConfig) -> None:
    if _is_annotated([node]):
        return

    input_qspec_map = {}
    input_act = node.args[0]
    assert isinstance(input_act, Node)
    input_spec = quantization_config.input_activation
    input_qspec_map[input_act] = input_spec

    weight = node.args[1]
    assert isinstance(weight, Node)
    input_qspec_map[weight] = quantization_config.weight

    if len(node.args) > 2:
        bias = node.args[2]
        if isinstance(bias, Node):
            if callable(quantization_config.bias):
                input_qspec_map[bias] = quantization_config.bias(node)
            else:
                input_qspec_map[bias] = quantization_config.bias

    node.meta[QUANT_ANNOTATION_KEY] = QuantizationAnnotation(
        input_qspec_map=input_qspec_map,
        output_qspec=quantization_config.output_activation,
        _annotated=True,
    )


@register_annotator([torch.ops.aten.linear.default])
def annotate_linear(node: Node, quantization_config: QuantizationConfig) -> None:
    act_node = node.args[0]
    weight_node = node.args[1]
    bias_node = None
    if len(node.args) > 2:
        bias_node = node.args[2]

    if _is_annotated([node]):
        return

    _annotate_input_qspec_map(
        node,
        act_node,
        quantization_config.input_activation,
    )
    _annotate_input_qspec_map(
        node,
        weight_node,
        quantization_config.weight,
    )
    nodes_to_mark_annotated = [node, weight_node]
    if bias_node:
        if callable(quantization_config.bias):
            bias_config = quantization_config.bias(node)
        else:
            bias_config = quantization_config.bias
        _annotate_input_qspec_map(node, bias_node, bias_config)
        nodes_to_mark_annotated.append(bias_node)
    _annotate_output_qspec(node, quantization_config.output_activation)
    _mark_nodes_as_annotated(nodes_to_mark_annotated)

    # We use get_source_partition in pass, but it is the same source for MultiheadAttention, so we need to change its source_fn_stack.
    node.meta["source_fn_stack"] = [(node, torch.nn.Linear)]


@register_annotator([torch.ops.aten.layer_norm.default])
def annotate_layer_norm(node: Node, quantization_config: QuantizationConfig) -> None:
    act_node = node.args[0]
    weight_node = node.args[2]
    bias_node = None
    if len(node.args) > 2:
        bias_node = node.args[3]

    if _is_annotated([node]):
        return

    _annotate_input_qspec_map(
        node,
        act_node,
        quantization_config.input_activation,
    )
    _annotate_input_qspec_map(
        node,
        weight_node,
        quantization_config.input_activation,
    )
    nodes_to_mark_annotated = [node, weight_node]
    if bias_node:
        _annotate_input_qspec_map(
            node,
            bias_node,
            quantization_config.bias,
        )
        nodes_to_mark_annotated.append(bias_node)
    _annotate_output_qspec(node, quantization_config.output_activation)
    _mark_nodes_as_annotated(nodes_to_mark_annotated)


@register_annotator([torch.ops.aten.cat.default, torch.ops.aten.concat.default])
def annotate_cat(node: Node, quantization_config: QuantizationConfig) -> None:
    input_nodes = node.args[0]
    if _is_annotated([node]):
        return

    assert isinstance(input_nodes, Sequence)

    first_input_node = input_nodes[0]
    input_qspec_map = {}
    assert isinstance(first_input_node, Node)
    assert isinstance(node, Node)
    input_qspec_map[first_input_node] = quantization_config.input_activation
    share_qparams_with_input_act0_qspec = SharedQuantizationSpec(
        (first_input_node, node)
    )

    for input_node in input_nodes[1:]:
        if input_node not in input_qspec_map:
            assert isinstance(input_node, Node)
            input_qspec_map[input_node] = share_qparams_with_input_act0_qspec

    node.meta[QUANT_ANNOTATION_KEY] = QuantizationAnnotation(
        input_qspec_map=input_qspec_map,
        output_qspec=share_qparams_with_input_act0_qspec,
        _annotated=True,
    )


@register_annotator([torch.ops.aten.unbind.int])
def annotate_unbind(node: Node, quantization_config: QuantizationConfig) -> None:
    if _is_annotated([node]):
        return

    input_qspec_map = {}
    input_act = node.args[0]
    assert isinstance(input_act, Node)
    input_qspec_map[input_act] = quantization_config.input_activation

    node_tensor = node.meta.get("val")
    if torch.is_tensor(node_tensor) and node_tensor.dtype == torch.int64:
        return

    node.meta[QUANT_ANNOTATION_KEY] = QuantizationAnnotation(
        input_qspec_map=input_qspec_map,
        _annotated=True,
    )
