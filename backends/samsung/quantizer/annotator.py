# Copyright (c) Qualcomm Innovation Center, Inc
# Copyright (c) 2025 Samsung Electronics Co. LTD
# All rights reserved
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from typing import Callable, Dict, List

import torch
from torch._ops import OpOverload
from torch._subclasses import FakeTensor

from torch.fx import Graph, Node

from torchao.quantization.pt2e import FixedQParamsObserver
from torchao.quantization.pt2e.quantizer import (
    annotate_output_qspec,
    QuantizationAnnotation,
    QuantizationSpec,
    SharedQuantizationSpec,
)

from .qconfig import QuantizationConfig

OP_ANNOTATOR: Dict[OpOverload, Callable] = {}

ADD_OPS = [
    torch.ops.aten.add,
    torch.ops.aten.add.Tensor,
    torch.ops.aten.add_.Tensor,
]


def register_annotator(ops: List[OpOverload]):
    def decorator(annotator: Callable):
        for op in ops:
            OP_ANNOTATOR[op] = annotator

    return decorator


def annotate(graph: Graph, quant_config: QuantizationConfig) -> None:
    # Pattern annotation
    _annotate_fused_activation_pattern(graph, quant_config)

    # Per-op annotation
    for node in graph.nodes:
        if node.op == "placeholder":
            annotate_placeholder(node, quant_config)
        elif node.op == "call_function":
            annotate_func = OP_ANNOTATOR.get(node.target, None)
            if annotate_func is not None:
                annotate_func(node, quant_config)


def _is_annotated(nodes: List[Node]):
    """
    Given a list of nodes (that represents an operator pattern),
    return True if any of the node
    is annotated, otherwise return False
    """
    annotated = False
    for node in nodes:
        annotated = annotated or (
            "quantization_annotation" in node.meta
            and node.meta["quantization_annotation"]._annotated
        )
    return annotated


def _is_fake_tensor(node: Node):
    if (
        isinstance(node, Node)
        and "val" in node.meta
        and isinstance(node.meta["val"], FakeTensor)
    ):
        return True
    return False


def _is_float_tensor(node: Node):
    """Check if the node's tensor is a float tensor,
    so that we can skip quantization for the node
    since observers only works with float Tensors
    """
    if not _is_fake_tensor(node):
        return False
    return node.meta["val"].dtype in [torch.float32, torch.float16]


def _mark_nodes_as_annotated(nodes: List[Node]):
    for node in nodes:
        if "quantization_annotation" not in node.meta:
            node.meta["quantization_annotation"] = QuantizationAnnotation()
        node.meta["quantization_annotation"]._annotated = True


# for nodes whose targets ars placehold (not call_function)
def annotate_placeholder(node: Node, quant_config: QuantizationConfig) -> None:
    if _is_annotated([node]):
        return

    if _is_float_tensor(node):
        annotate_output_qspec(node, quant_config.output_activation)

    _mark_nodes_as_annotated([node])


# CASE 1: fused_activation case (ex. Conv2D + ReLU)
def _is_hardtanh_for_relux(relu_node: torch.fx.node.Node):
    if relu_node.target in [
        torch.ops.aten.hardtanh.default,
        torch.ops.aten.hardtanh_.default,
    ]:
        # checking if hardtanh is convertable to ReLU6
        # ReLU1 is not supported now
        if not relu_node.args[1] == 0.0:
            return False
        if relu_node.args[2] == 6.0:  # for ReLU6
            return True
    return True


def _annotate_fused_activation_pattern(
    graph: Graph, quant_config: QuantizationConfig
) -> None:
    for relu_node in graph.nodes:
        # Check relu/relu6 node
        if relu_node.op != "call_function":
            continue
        if relu_node.target not in [
            # The strategy of ReLU and ReLU6 is fold_activation in ENNQuant
            torch.ops.aten.relu.default,
            torch.ops.aten.relu_.default,
            torch.ops.aten.relu6.default,
            torch.ops.aten.relu6_.default,
            torch.ops.aten.hardtanh.default,
            torch.ops.aten.hardtanh_.default,
        ]:
            continue

        if not _is_hardtanh_for_relux(relu_node):
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
            # input & weight (or bias) setting for Conv node(producer_node)
            quantization_annotation = producer_node.meta.get(
                "quantization_annotation", QuantizationAnnotation()
            )
            if quantization_annotation.input_qspec_map is None:
                quantization_annotation.input_qspec_map = {}

            input = producer_node.args[0]
            quantization_annotation.input_qspec_map[input] = (
                quant_config.input_activation
            )

            quantization_annotation.input_qspec_map[producer_node.args[1]] = (
                quant_config.weight
            )
            if len(producer_node.args) > 2 and quant_config.bias is not None:
                quantization_annotation.input_qspec_map[producer_node.args[2]] = (
                    quant_config.bias
                )

            producer_node.meta["quantization_annotation"] = quantization_annotation
            producer_node.meta["quantization_annotation"]._annotated = True
            # out setting for activation node (relu_node)
            quantization_annotation = relu_node.meta.get(
                "quantization_annotation", QuantizationAnnotation()
            )
            quantization_annotation.output_qspec = quant_config.output_activation

            relu_node.meta["quantization_annotation"] = quantization_annotation
            relu_node.meta["quantization_annotation"]._annotated = True
            continue


# CASE 2-1: two input case without Shared Quant
@register_annotator(
    [
        torch.ops.aten.div,
        torch.ops.aten.div.Tensor,
        torch.ops.aten.divide.Tensor,
        torch.ops.aten.matmul.default,
        torch.ops.aten.bmm.default,
        torch.ops.aten.sum.dim_IntList,
    ]
)
def annotate_2in1out(node: Node, quant_config: QuantizationConfig) -> None:
    input_act0 = node.args[0]
    input_act1 = node.args[1]
    # skipping quantization if 1st input is not float.
    if _is_annotated([node]) or not _is_float_tensor(input_act0):
        return

    input_act_qspec = quant_config.input_activation
    output_act_qspec = (
        quant_config.output_activation if _is_float_tensor(node) else None
    )

    input_qspec_map = {}
    if _is_float_tensor(input_act0):
        input_qspec_map[input_act0] = input_act_qspec

    if _is_float_tensor(input_act1):
        input_qspec_map[input_act1] = input_act_qspec

    node.meta["quantization_annotation"] = QuantizationAnnotation(
        input_qspec_map=input_qspec_map,
        output_qspec=output_act_qspec,
        _annotated=True,
    )


# getting QuantAnnot though the first input
def _get_quantization_annotation(node: Node):
    if node.op == "placeholder":
        return False
    elif "quantization_annotation" in node.meta:
        return node
    elif node.args == ():
        return False
    elif isinstance(node.args[0], Node):
        return _get_quantization_annotation(node.args[0])
    elif isinstance(node.args[0], list):
        # for cat, concatenate and stack
        if isinstance(node.args[0][0], Node):
            return _get_quantization_annotation(node.args[0][0])
        else:
            return False
    else:
        return False


# CASE 2-2: two input case with Shared Quant
# ops.add / ops.add_ are processed by another annotator
@register_annotator(
    [
        torch.ops.aten.sub,
        torch.ops.aten.mul,
        torch.ops.aten.sub.Tensor,
        torch.ops.aten.mul.Tensor,
        torch.ops.aten.sub_.Tensor,
        torch.ops.aten.mul_.Tensor,
        torch.ops.aten.rsub.Scalar,
        torch.ops.aten.mul.Scalar,
    ]
)
def annotate_2in1out_with_SharedQuant(
    node: Node, quant_config: QuantizationConfig
) -> None:

    input_qspec_map = {}
    input0 = node.args[0]
    input1 = node.args[1]

    # skipping quantization if 1st input is not float.
    if _is_annotated([node]) or not _is_float_tensor(input0):
        return
    if (
        isinstance(input0, Node)
        and isinstance(input1, float)
        and not _get_quantization_annotation(input0)
    ):
        return
    if (
        isinstance(input0, float)
        and isinstance(input1, Node)
        and not _get_quantization_annotation(input1)
    ):
        return
    if isinstance(input0, Node) and isinstance(input1, Node):
        shared_qspec = SharedQuantizationSpec((input0, node))
        input_qspec_map[input0] = quant_config.input_activation
        input_qspec_map[input1] = shared_qspec

        node.meta["quantization_annotation"] = QuantizationAnnotation(
            input_qspec_map=input_qspec_map,
            output_qspec=shared_qspec,
            _annotated=True,
        )

    else:
        input_act_qspec = quant_config.input_activation
        output_act_qspec = (
            quant_config.output_activation if _is_float_tensor(node) else None
        )

        input_qspec_map = {}
        input_act0 = node.args[0]
        if _is_float_tensor(input_act0):
            input_qspec_map[input_act0] = input_act_qspec

        input_act1 = node.args[1]
        if _is_float_tensor(input_act1):
            input_qspec_map[input_act1] = input_act_qspec

        node.meta["quantization_annotation"] = QuantizationAnnotation(
            input_qspec_map=input_qspec_map,
            output_qspec=output_act_qspec,
            _annotated=True,
        )


# CASE 2-3: only for add ops
@register_annotator(ADD_OPS)
def annotate_add_ops_with_SharedQuant(
    node: Node, quant_config: QuantizationConfig
) -> None:

    input_qspec_map = {}
    input0 = node.args[0]
    input1 = node.args[1]

    # skipping quantization if 1st input is not float.
    if _is_annotated([node]) or not _is_float_tensor(input0):
        return

    if isinstance(input0, Node) and isinstance(input1, Node):
        NonQuantShare_ops_for_add = [torch.ops.aten.dropout.default] + ADD_OPS
        if (
            input0.op == "call_function" and input0.target in NonQuantShare_ops_for_add
        ) or (
            input1.op == "call_function" and input1.target in NonQuantShare_ops_for_add
        ):
            input_act_qspec = quant_config.input_activation
            output_act_qspec = (
                quant_config.output_activation if _is_float_tensor(node) else None
            )

            input_qspec_map = {}
            input_act0 = node.args[0]
            if _is_float_tensor(input_act0):
                input_qspec_map[input_act0] = input_act_qspec

            input_act1 = node.args[1]
            if _is_float_tensor(input_act1):
                input_qspec_map[input_act1] = input_act_qspec

            node.meta["quantization_annotation"] = QuantizationAnnotation(
                input_qspec_map=input_qspec_map,
                output_qspec=output_act_qspec,
                _annotated=True,
            )
        else:
            shared_qspec = SharedQuantizationSpec((input0, node))
            input_qspec_map[input0] = quant_config.input_activation
            input_qspec_map[input1] = shared_qspec

            node.meta["quantization_annotation"] = QuantizationAnnotation(
                input_qspec_map=input_qspec_map,
                output_qspec=shared_qspec,
                _annotated=True,
            )
    elif (
        isinstance(input0, Node)
        and isinstance(input1, float)
        and not _get_quantization_annotation(input0)
    ):
        pass
    elif (
        isinstance(input0, float)
        and isinstance(input1, Node)
        and not _get_quantization_annotation(input1)
    ):
        pass
    else:
        input_act_qspec = quant_config.input_activation
        output_act_qspec = (
            quant_config.output_activation if _is_float_tensor(node) else None
        )

        input_qspec_map = {}
        input_act0 = node.args[0]
        if _is_float_tensor(input_act0):
            input_qspec_map[input_act0] = input_act_qspec

        input_act1 = node.args[1]
        if _is_float_tensor(input_act1):
            input_qspec_map[input_act1] = input_act_qspec

        node.meta["quantization_annotation"] = QuantizationAnnotation(
            input_qspec_map=input_qspec_map,
            output_qspec=output_act_qspec,
            _annotated=True,
        )


# CASE 3-1: Single input + Single Out case without Shared Quant
@register_annotator(
    [
        torch.ops.aten.ceil.default,
        torch.ops.aten.clamp.default,
        torch.ops.aten.relu.default,
        torch.ops.aten.relu_.default,
        torch.ops.aten.relu6.default,
        torch.ops.aten.relu6_.default,
        torch.ops.aten.cos.default,
        torch.ops.aten.sin.default,
        torch.ops.aten.tanh.default,
        torch.ops.aten.hardswish.default,
        torch.ops.aten.hardswish_.default,
        torch.ops.aten.hardsigmoid.default,
        torch.ops.aten.hardsigmoid_.default,
        torch.ops.aten.hardtanh.default,
        torch.ops.aten.hardtanh_.default,
        torch.ops.aten.mean.default,
        torch.ops.aten.adaptive_avg_pool2d.default,
        torch.ops.aten.avg_pool2d.default,
        torch.ops.aten.leaky_relu.default,
        torch.ops.aten.leaky_relu_.default,
        torch.ops.aten.prelu.default,
        torch.ops.aten.upsample_bilinear2d.vec,
        torch.ops.aten.upsample_nearest2d.vec,
        torch.ops.aten.mean.dim,
        torch.ops.aten.sqrt.default,
        torch.ops.aten.gelu.default,
        torch.ops.aten.scaled_dot_product_attention.default,
        torch.ops.aten.rsqrt.default,
        torch.ops.aten.pow.Tensor_Scalar,
        torch.ops.aten.topk.default,
    ]
)
def annotate_1in1out(node: Node, quant_config: QuantizationConfig) -> None:
    # skipping quantization if input is not float.
    if _is_annotated([node]) or not _is_float_tensor(node.args[0]):
        return

    quantization_annotation = node.meta.get(
        "quantization_annotation", QuantizationAnnotation()
    )
    if quantization_annotation.input_qspec_map is None:
        quantization_annotation.input_qspec_map = {}

    # one inputs + one output case.
    input_act_qspec = quant_config.input_activation
    quantization_annotation.input_qspec_map[node.args[0]] = input_act_qspec
    quantization_annotation.output_qspec = quant_config.output_activation

    node.meta["quantization_annotation"] = quantization_annotation
    node.meta["quantization_annotation"]._annotated = True


# CASE 3-2: Single input + Single Out case with Shared Quant
@register_annotator(
    [
        torch.ops.aten.permute.default,
        torch.ops.aten.view.default,
        torch.ops.aten._unsafe_view.default,
        torch.ops.aten.squeeze.default,
        torch.ops.aten.squeeze.dim,
        torch.ops.aten.squeeze_copy.dims,
        torch.ops.aten.unsqueeze.default,
        torch.ops.aten.unsqueeze_copy.default,
        torch.ops.aten.transpose.int,
        torch.ops.aten.expand.default,
        torch.ops.aten.max_pool2d.default,
        torch.ops.aten.max_pool2d_with_indices.default,
        torch.ops.aten.reshape.default,
        torch.ops.aten.select.int,
        torch.ops.aten.flatten.using_ints,
        torch.ops.aten.pad.default,
        torch.ops.aten.slice.Tensor,
        torch.ops.aten.to.dtype,
    ]
)
def annotate_1in1out_with_SharedQuant(
    node: Node, quant_config: QuantizationConfig
) -> None:
    input_qspec_map = {}
    input = node.args[0]
    assert isinstance(input, Node)
    if _is_annotated([node]) or not _is_float_tensor(input):
        return

    shared_qspec = SharedQuantizationSpec((input, node))

    # get QuantAnnot from the input path
    shared_quant_node = _get_quantization_annotation(input)
    if shared_quant_node:
        input_qspec_map[shared_quant_node] = SharedQuantizationSpec(shared_quant_node)
        shared_qspec = SharedQuantizationSpec((shared_quant_node, node))
    else:
        # if no QuantAnnot in the input path
        input_qspec_map[input] = quant_config.input_activation
        shared_qspec = SharedQuantizationSpec((input, node))

    node.meta["quantization_annotation"] = QuantizationAnnotation(
        input_qspec_map=input_qspec_map,
        output_qspec=shared_qspec,
        _annotated=True,
    )


# CASE 3-3: Single input + Single Out case with FP
@register_annotator(
    [
        torch.ops.aten.softmax.int,
        torch.ops.aten._softmax.default,
        torch.ops.aten._safe_softmax.default,
        torch.ops.aten.log_softmax.int,
    ]
)
def annotate_1in1out_with_SharedQuant_for_FP(
    node: Node, quant_config: QuantizationConfig
) -> None:
    input_qspec_map = {}
    input = node.args[0]
    assert isinstance(input, Node)

    if _is_annotated([node]) or not _is_float_tensor(input):
        return

    if input.target in ADD_OPS and _is_annotated([input]):
        del input.meta["quantization_annotation"]

    # get QuantAnnot from the input path
    shared_quant_node = _get_quantization_annotation(input)
    if shared_quant_node:
        # if QuantAnnot in the input path, input_qspec is shared, but output_qspec is not.
        input_qspec_map[shared_quant_node] = SharedQuantizationSpec(shared_quant_node)

        node.meta["quantization_annotation"] = QuantizationAnnotation(
            input_qspec_map=input_qspec_map,
            output_qspec=quant_config.output_activation,
            _annotated=True,
        )
    else:
        # if no QuantAnnot in the input path
        node.meta["quantization_annotation"] = QuantizationAnnotation(
            output_qspec=quant_config.output_activation,
            _annotated=True,
        )


# CASE 4: One value input + one index input with Shared Quant
@register_annotator([torch.ops.aten.index.Tensor])
def annotate_index(node: Node, quant_config: QuantizationConfig) -> None:
    input_qspec_map = {}
    input = node.args[0]
    assert isinstance(input, Node)

    if _is_annotated([node]) or not _is_float_tensor(input):
        return

    # get QuantAnnt from the input path
    shared_quant_node = _get_quantization_annotation(input)
    if shared_quant_node:
        shared_qspec = SharedQuantizationSpec((shared_quant_node, node))
        input_qspec_map[input] = quant_config.input_activation

        # sharing QuantAnnot with the parent
        node.meta["quantization_annotation"] = QuantizationAnnotation(
            input_qspec_map=input_qspec_map,
            output_qspec=shared_qspec,
            _annotated=True,
        )


# CASE 5 input + index + value & output with Shared Quant
@register_annotator(
    [torch.ops.aten.index_put.default, torch.ops.aten.index_put_.default]
)
def annotate_index_put(node: Node, quant_config: QuantizationConfig) -> None:
    input_qspec_map = {}
    input = node.args[0]  # from KVCache in LLAMA
    value = node.args[2]  # from linear projection layer
    assert isinstance(input, Node)
    assert isinstance(value, Node)

    if _is_annotated([node]) or not _is_float_tensor(input):
        return

    # get QuantAnnot from input path
    shared_quant_node = _get_quantization_annotation(input)
    if shared_quant_node:
        shared_qspec = SharedQuantizationSpec((shared_quant_node, node))
        input_qspec_map[input] = shared_qspec
        input_qspec_map[value] = shared_qspec
        output_qspec = shared_qspec
    else:
        # if no QuantAnnot in input path, asign the default QuantAnnot from quant_config.
        input_qspec_map[input] = quant_config.input_activation
        input_qspec_map[value] = SharedQuantizationSpec((input, node))
        output_qspec = SharedQuantizationSpec((input, node))

    node.meta["quantization_annotation"] = QuantizationAnnotation(
        input_qspec_map=input_qspec_map,
        output_qspec=output_qspec,
        _annotated=True,
    )


# CASE 6 unbind + getitem case
# (inputQuant--unbinde--no Qunat) --> (no Qunat--getitem--outputQuant)
@register_annotator([torch.ops.aten.unbind.int])
def annotate_unbind(node: Node, quant_config: QuantizationConfig) -> None:
    input_qspec_map = {}
    input = node.args[0]
    assert isinstance(input, Node)

    if _is_annotated([node]) or not _is_float_tensor(input):
        return

    # get QuantAnnot from input path
    shared_quant_node = _get_quantization_annotation(input)
    if shared_quant_node:
        input_qspec_map[input] = quant_config.input_activation
        shared_qspec = SharedQuantizationSpec((shared_quant_node, node))
    else:
        # if no QuantAnnot in input path, asign the default QuantAnnot from quant_config.
        input_qspec_map[input] = quant_config.input_activation
        shared_qspec = SharedQuantizationSpec((input, node))

    node.meta["quantization_annotation"] = QuantizationAnnotation(
        input_qspec_map=input_qspec_map,
        output_qspec=shared_qspec,
        _annotated=True,
    )

    for users_node in node.users:
        users_node.meta["quantization_annotation"] = QuantizationAnnotation(
            output_qspec=shared_qspec,
            _annotated=True,
        )


# CASE 7: stand-alone Conv2d and Conv1d
@register_annotator(
    [
        torch.ops.aten.conv2d.default,
        torch.ops.aten.conv1d.default,
        torch.ops.aten.linear.default,
    ]
)
def annotate_conv2d(node: Node, quant_config: QuantizationConfig) -> None:
    # skipping quantization if weights are not float
    if _is_annotated([node]) or not _is_float_tensor(node.args[1]):
        return

    input = node.args[0]
    # input & weight (or bias) setting for Conv node(producer_node)
    quantization_annotation = node.meta.get(
        "quantization_annotation", QuantizationAnnotation()
    )
    if quantization_annotation.input_qspec_map is None:
        quantization_annotation.input_qspec_map = {}

    shared_quant_node = _get_quantization_annotation(input)
    if shared_quant_node:
        quantization_annotation.input_qspec_map[input] = SharedQuantizationSpec(
            shared_quant_node
        )
    else:
        quantization_annotation.input_qspec_map[input] = quant_config.input_activation
    quantization_annotation.input_qspec_map[node.args[1]] = quant_config.weight
    if len(node.args) > 2 and quant_config.bias is not None:
        quantization_annotation.input_qspec_map[node.args[2]] = quant_config.bias
    quantization_annotation.output_qspec = quant_config.output_activation

    node.meta["quantization_annotation"] = quantization_annotation
    node.meta["quantization_annotation"]._annotated = True


# CASE 8: embedding
@register_annotator([torch.ops.aten.embedding.default])
def annotate_embedding(node: Node, quant_config: QuantizationConfig) -> None:
    input_qspec_map = {}
    weight = node.args[0]
    if _is_annotated([node]) or not _is_float_tensor(weight):
        return

    input_qspec_map[weight] = quant_config.input_activation

    node.meta["quantization_annotation"] = QuantizationAnnotation(
        input_qspec_map=input_qspec_map,
        output_qspec=quant_config.output_activation,
        _annotated=True,
    )


# CASE 9: Concat & Stack
@register_annotator(
    [
        torch.ops.aten.cat.default,
        torch.ops.aten.concat.default,
        torch.ops.aten.stack.default,
    ]
)
def annotate_cat(node: Node, quant_config: QuantizationConfig) -> None:
    inputs = node.args[0]
    first_input = inputs[0]
    assert isinstance(inputs, list)
    assert isinstance(first_input, Node)

    if _is_annotated([node]) or not _is_float_tensor(first_input):
        return

    input_qspec_map = {}
    shared_qspec = SharedQuantizationSpec((first_input, node))
    for input in inputs:
        if input == first_input:
            input_qspec_map[input] = quant_config.input_activation
        else:
            input_qspec_map[input] = shared_qspec

    node.meta["quantization_annotation"] = QuantizationAnnotation(
        input_qspec_map=input_qspec_map,
        output_qspec=shared_qspec,
        _annotated=True,
    )


# CASE 10: various normalizations
@register_annotator([torch.ops.aten.rms_norm.default])
def annotate_rms_norm(node: Node, quant_config: QuantizationConfig) -> None:
    if _is_annotated([node]):
        return

    quantization_annotation = node.meta.get(
        "quantization_annotation", QuantizationAnnotation()
    )
    if quantization_annotation.input_qspec_map is None:
        quantization_annotation.input_qspec_map = {}

    quantization_annotation.input_qspec_map[node.args[0]] = (
        quant_config.input_activation
    )  # active
    quantization_annotation.input_qspec_map[node.args[2]] = (
        quant_config.input_activation
    )  # weight
    quantization_annotation.output_qspec = quant_config.output_activation
    node.meta["quantization_annotation"] = quantization_annotation
    node.meta["quantization_annotation"]._annotated = True


@register_annotator([torch.ops.aten.group_norm.default])
def annotate_group_norm(node: Node, quant_config: QuantizationConfig) -> None:
    if _is_annotated([node]):
        return

    quantization_annotation = node.meta.get(
        "quantization_annotation", QuantizationAnnotation()
    )
    if quantization_annotation.input_qspec_map is None:
        quantization_annotation.input_qspec_map = {}

    quantization_annotation.input_qspec_map[node.args[0]] = (
        quant_config.input_activation
    )  # active
    quantization_annotation.input_qspec_map[node.args[2]] = (
        quant_config.weight
    )  # weight
    quantization_annotation.output_qspec = quant_config.output_activation

    node.meta["quantization_annotation"] = quantization_annotation
    node.meta["quantization_annotation"]._annotated = True


@register_annotator([torch.ops.aten.layer_norm.default])
def annotate_layer_norm(node: Node, quant_config: QuantizationConfig) -> None:
    if _is_annotated([node]):
        return

    quantization_annotation = node.meta.get(
        "quantization_annotation", QuantizationAnnotation()
    )
    if quantization_annotation.input_qspec_map is None:
        quantization_annotation.input_qspec_map = {}

    quantization_annotation.input_qspec_map[node.args[0]] = (
        quant_config.input_activation
    )  # active
    quantization_annotation.input_qspec_map[node.args[2]] = (
        quant_config.input_activation
    )  # weight
    quantization_annotation.output_qspec = quant_config.output_activation

    node.meta["quantization_annotation"] = quantization_annotation
    node.meta["quantization_annotation"]._annotated = True


@register_annotator([torch.ops.aten._native_batch_norm_legit_no_training.default])
def annotate_batch_norm(node: Node, quant_config: QuantizationConfig) -> None:
    if _is_annotated([node]):
        return

    quantization_annotation = node.meta.get(
        "quantization_annotation", QuantizationAnnotation()
    )
    if quantization_annotation.input_qspec_map is None:
        quantization_annotation.input_qspec_map = {}

    quantization_annotation.input_qspec_map[node.args[0]] = (
        quant_config.input_activation
    )  # active

    quantization_annotation.input_qspec_map[node.args[1]] = (
        quant_config.input_activation
    )  # weight
    quantization_annotation.output_qspec = quant_config.output_activation

    node.meta["quantization_annotation"] = quantization_annotation
    node.meta["quantization_annotation"]._annotated = True


# CASE 11: Sigmoid
@register_annotator([torch.ops.aten.sigmoid, torch.ops.aten.sigmoid.default])
def annotate_sigmoid(node: Node, quant_config: QuantizationConfig) -> None:
    if _is_annotated([node]):
        return

    input_qspec_map = {}
    input_act = node.args[0]
    input_qspec_map[input_act] = quant_config.input_activation

    assert isinstance(input_act, Node)
    out_qconf = quant_config.output_activation

    q_max = (
        torch.iinfo(out_qconf.dtype).max
        if out_qconf.quant_max is None
        else out_qconf.quant_max
    )
    q_min = (
        torch.iinfo(out_qconf.dtype).min
        if out_qconf.quant_min is None
        else out_qconf.quant_min
    )

    scale = 1 / (q_max - q_min + 1)

    bias_obs_ctr = FixedQParamsObserver.with_args(
        scale=scale,
        zero_point=0,
        dtype=quant_config.output_activation.dtype,
        qscheme=torch.torch.per_tensor_affine,
        quant_max=q_max,
        quant_min=q_min,
    )

    # make sigmoid map to the range between 0~1
    out_act_quantization_spec = QuantizationSpec(
        dtype=quant_config.output_activation.dtype,
        quant_max=q_max,
        quant_min=q_min,
        observer_or_fake_quant_ctr=bias_obs_ctr,
        qscheme=torch.torch.per_tensor_affine,
    )

    if _is_float_tensor(node):
        node.meta["quantization_annotation"] = QuantizationAnnotation(
            input_qspec_map=input_qspec_map,
            output_qspec=out_act_quantization_spec,
            _annotated=True,
        )
