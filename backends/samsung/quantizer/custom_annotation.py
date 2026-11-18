# Copyright (c) 2026 Samsung Electronics Co. LTD
# All rights reserved
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from typing import Optional

import torch
from executorch.backends.samsung.quantizer.annotator import (
    _is_annotated,
    _is_float_tensor,
    OP_ANNOTATOR,
)

from executorch.backends.samsung.quantizer.qconfig import (
    _get_eps,
    get_a16w2_enn_quant_config,
    get_a16w4_enn_quant_config,
    get_a16w8_enn_quant_config,
    get_a8w4_enn_quant_config,
    get_a8w8_enn_quant_config,
    QuantizationConfig,
)

from torch.fx import Node
from torchao.quantization.pt2e import HistogramObserver
from torchao.quantization.pt2e.quantizer import (
    QuantizationAnnotation,
    SharedQuantizationSpec,
)


def annotate_2in1out_for_mul(node: Node, quant_config: QuantizationConfig) -> None:
    # matmul's 2nd input reuses the default annotator's shared quantization.
    annotate_2_binary_inputs(node, quant_config, share_input1_with_default=True)


def annotate_2_binary_inputs(
    node: Node,
    quant_config: QuantizationConfig,
    share_input1_with_default: bool = False,
) -> None:
    """Annotate a 2-input op with per-input activation qspecs.

    Some inputs (e.g. the weight of a matmul) must reuse the default
    annotator's shared quantization, while others (e.g. the inputs of a binary
    sub/mul) need fully independent qspecs.
    """
    input0 = node.args[0]
    input1 = node.args[1]
    # skipping quantization if 1st input is not float.
    if not _is_float_tensor(input0):
        return

    if "quantization_annotation" in node.meta:
        del node.meta["quantization_annotation"]

    input_act_qspec = quant_config.input_activation
    output_act_qspec = (
        quant_config.output_activation if _is_float_tensor(node) else None
    )

    input_qspec_map = {}
    if isinstance(input0, Node) and _is_float_tensor(input0):
        input_qspec_map[input0] = input_act_qspec

    if isinstance(input1, Node) and _is_float_tensor(input1):
        input_qspec_map[input1] = (
            SharedQuantizationSpec(input1)
            if share_input1_with_default
            else input_act_qspec
        )

    node.meta["quantization_annotation"] = QuantizationAnnotation(
        input_qspec_map=input_qspec_map,
        output_qspec=output_act_qspec,
        _annotated=True,
    )


# Binary ops whose quantization params must NOT be shared (override the
# SharedQuantizationSpec-based annotation from the default annotator).
NON_SHARED_BINARY_OPS = [
    torch.ops.aten.sub.Tensor,
    torch.ops.aten.mul.Tensor,
    torch.ops.aten.sub_.Tensor,
    torch.ops.aten.mul_.Tensor,
    torch.ops.aten.rsub.Scalar,
    torch.ops.aten.mul.Scalar,
]


def annotate_2in1out_wo_SharedQuant(
    node: Node, quant_config: QuantizationConfig
) -> None:
    annotate_2_binary_inputs(node, quant_config)


def annotate_embedding(node: Node, quant_config: QuantizationConfig) -> None:
    input_qspec_map = {}
    weight = node.args[0]

    input_qspec_map[weight] = quant_config.weight

    output_act_qspec = (
        quant_config.output_activation if _is_float_tensor(node) else None
    )

    node.meta["quantization_annotation"] = QuantizationAnnotation(
        input_qspec_map=input_qspec_map,
        output_qspec=output_act_qspec,
        _annotated=True,
    )


def annotate_cat(node: Node, quant_config: QuantizationConfig) -> None:
    inputs = node.args[0]
    assert isinstance(inputs, (list, tuple)), "cat inputs must be a list/tuple"
    inputs = list(inputs)
    assert inputs, "cat inputs must not be empty"
    first_input = inputs[0]
    assert isinstance(first_input, Node), "first cat input must be a node"

    # Keep what the default annotator already set on the producer (e.g. a
    # conv's input and weight specs); only give its output the cat's qspec.
    first_annotation = first_input.meta.get("quantization_annotation")
    if first_annotation is None:
        first_input.meta["quantization_annotation"] = QuantizationAnnotation(
            output_qspec=quant_config.input_activation, _annotated=True
        )
    elif first_annotation.output_qspec is None:
        first_annotation.output_qspec = quant_config.input_activation

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

    second_input = inputs[1]
    if "quantization_annotation" in second_input.meta:
        second_input.meta["quantization_annotation"].output_qspec = (
            quant_config.input_activation
        )


def annotate_index_put(node: Node, quant_config: QuantizationConfig) -> None:
    value = node.args[2]
    input_qspec_map = {}
    input_qspec_map[value] = quant_config.input_activation
    node.meta["quantization_annotation"] = QuantizationAnnotation(
        input_qspec_map=input_qspec_map,
        output_qspec=SharedQuantizationSpec((value, node)),
        _annotated=True,
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

    quantization_annotation.input_qspec_map[input] = quant_config.input_activation
    quantization_annotation.input_qspec_map[node.args[1]] = quant_config.weight
    if len(node.args) > 2 and quant_config.bias is not None:
        quantization_annotation.input_qspec_map[node.args[2]] = quant_config.bias
    quantization_annotation.output_qspec = quant_config.output_activation

    node.meta["quantization_annotation"] = quantization_annotation
    node.meta["quantization_annotation"]._annotated = True


def annotate_from_index_put_to_matmul_input(
    node: Node, quant_config: QuantizationConfig
):
    if node.target in [
        torch.ops.aten.index_put.default,
        torch.ops.aten.index_put_.default,
    ]:
        annotate_index_put(node, quant_config)
    elif node.target in [
        torch.ops.aten.cat.default,
    ]:
        annotate_cat(node, quant_config)
    else:
        # The path of matmul's 2nd input in Llama model has
        #    transpose.int, repeat_interleave.self_int and index_put_.default
        # Even though repeat_interleave.self_int has 3 args,
        #    only args[0] needs quantization.
        # So, annotate_1in1out can cover all ops in the path
        annotate_1in1out(node, quant_config)
        arg0 = node.args[0]
        # Stop the walk at the leaf (e.g. a matmul whose 2nd input is an
        # activation rather than a KV-cache index_put path).
        if isinstance(arg0, Node) and arg0.op == "call_function":
            annotate_from_index_put_to_matmul_input(arg0, quant_config)


def _is_matmul_with_call_function_inputs(node: Node) -> bool:
    """True if ``node`` is a matmul whose two inputs are both call_function nodes."""
    return (
        node.op == "call_function"
        and node.target == torch.ops.aten.matmul.default
        and all(
            isinstance(arg, Node) and arg.op == "call_function"
            for arg in (node.args[0], node.args[1])
        )
    )


def _find_lm_head(gm: torch.fx.GraphModule) -> Optional[Node]:
    """The lm_head is the last linear. Its weight cannot be matched by name: Gemma3
    ties it to the embedding, so the exported weight is named after the embedding."""
    linears = [
        n
        for n in gm.graph.nodes
        if n.op == "call_function" and n.target == torch.ops.aten.linear.default
    ]
    return linears[-1] if linears else None


def annotate_linear_with_config(node: Node, quant_config: QuantizationConfig) -> None:
    annotate_conv2d(node, quant_config)
    ifm = node.args[0]
    if "quantization_annotation" in ifm.meta:
        ifm.meta["quantization_annotation"].output_qspec = quant_config.input_activation

    for user in node.users:
        if "quantization_annotation" not in user.meta:
            continue
        qa = user.meta["quantization_annotation"]
        if qa.input_qspec_map is None:
            qa.input_qspec_map = {}
        qa.input_qspec_map[node] = quant_config.output_activation


def custom_annotate_gemma4_text_decoder(gm: torch.fx.GraphModule) -> None:
    # Annotate 16a8w for matmul op to get better performance
    quant_config_a16w8 = get_a16w8_enn_quant_config(
        is_per_channel=False,
        is_qat=False,
        act_symmetric=False,
        wei_symmetric=True,
        eps=_get_eps,
    )
    # Annotate 8a8w for second input of matmul until past_kv_cache
    quant_config_a8w8 = get_a8w8_enn_quant_config(
        is_per_channel=False,
        is_qat=False,
        act_symmetric=True,
        wei_symmetric=True,
        eps=_get_eps,
    )
    # Annotate 16a8w for lm_head op to get better performance
    quant_config_a16w8_linear = get_a16w8_enn_quant_config(
        is_per_channel=True,
        is_qat=False,
        act_symmetric=False,
        wei_symmetric=True,
        eps=_get_eps,
    )

    # Annotate 16a8w for ple op to get better performance
    quant_config_a8w8_ple = get_a8w8_enn_quant_config(
        is_per_channel=True,
        is_qat=False,
        act_symmetric=True,
        wei_symmetric=True,
        eps=_get_eps,
    )

    quant_config_a8w4_linear = get_a8w4_enn_quant_config(
        is_per_channel=True,
        is_qat=False,
        act_symmetric=True,
        wei_symmetric=True,
        eps=_get_eps,
    )
    # Annotate 16a4w for embedding
    quant_config_a16w4_embedding = get_a16w4_enn_quant_config(
        is_per_channel=True,
        is_qat=False,
        act_symmetric=False,
        wei_symmetric=True,
        eps=_get_eps,
    )

    quant_config_a16w2_embedding = get_a16w2_enn_quant_config(
        is_per_channel=True,
        is_qat=False,
        act_symmetric=False,
        wei_symmetric=True,
        eps=_get_eps,
    )

    # KV cache
    for node in gm.graph.nodes:
        # Changing Quant from index_put to input of Matmul to support Mixed precision.
        if (
            node.op == "call_function"
            and node.target == torch.ops.aten.matmul.default
            and (
                node.args[0].op == "call_function"
                and node.args[1].op == "call_function"
            )
        ):
            annotate_2in1out_for_mul(node, quant_config_a16w8)  # for matmul op
            annotate_from_index_put_to_matmul_input(node.args[1], quant_config_a8w8)

        # Key linear op is A16W4 --> A16W8
        if node.op == "call_function" and node.target == torch.ops.aten.linear.default:
            if "quantization_annotation" in node.meta:
                del node.meta["quantization_annotation"]
            if "per_layer_projection_weight" in str(
                node.args[1]
            ) or "per_layer_input_gate_weight" in str(node.args[1]):
                annotate_linear_with_config(node, quant_config_a8w8_ple)
            elif "lm_head_weight" in str(
                node.args[1]
            ) or "per_layer_model_projection_weight" in str(node.args[1]):
                OP_ANNOTATOR[node.target](node, quant_config_a16w8_linear)
            else:
                annotate_linear_with_config(node, quant_config_a8w4_linear)

        # Per-ch (4 bits) for embedding weight
        if (
            node.op == "call_function"
            and node.target == torch.ops.aten.embedding.default
        ):
            if "quantization_annotation" in node.meta:
                del node.meta["quantization_annotation"]
            if "self_decoder_embed_tokens_weight" in node.args[0].name:
                annotate_embedding(node, quant_config_a16w2_embedding)
            else:
                annotate_embedding(node, quant_config_a16w4_embedding)


def custom_annotate_llama_a16w4(gm: torch.fx.GraphModule) -> None:
    # Annotate 16a8w for matmul op to get better performance
    quant_config_a16w8 = get_a16w8_enn_quant_config(
        is_per_channel=False, is_qat=False, act_symmetric=False, wei_symmetric=False
    )
    # Annotate 8a8w for second input of matmul until past_kv_cache
    quant_config_a8w8 = get_a8w8_enn_quant_config(
        is_per_channel=False, is_qat=False, act_symmetric=True, wei_symmetric=True
    )

    for node in gm.graph.nodes:
        # Binary ops (sub/mul) must use independent quantization params.
        if node.op == "call_function" and node.target in NON_SHARED_BINARY_OPS:
            annotate_2in1out_wo_SharedQuant(node, quant_config_a16w8)

        # Changing Quant from index_put to input of Matmul to support Mixed precision.
        if _is_matmul_with_call_function_inputs(node):
            annotate_2in1out_for_mul(node, quant_config_a16w8)  # for matmul op
            annotate_from_index_put_to_matmul_input(node.args[1], quant_config_a8w8)


def custom_annotate_gemma3_a16w4(gm: torch.fx.GraphModule) -> None:
    # Annotate 16a8w for matmul op to get better performance
    observer = HistogramObserver
    quant_config_a16w8 = get_a16w8_enn_quant_config(
        is_per_channel=False,
        is_qat=False,
        act_symmetric=False,
        wei_symmetric=False,
        act_observer_cls=observer,
    )
    # Annotate 8a8w for second input of matmul until past_kv_cache
    quant_config_a8w8 = get_a8w8_enn_quant_config(
        is_per_channel=False,
        is_qat=False,
        act_symmetric=True,
        wei_symmetric=True,
        act_observer_cls=observer,
    )
    # Annotate 16a8w for lm_head op to get better performance
    quant_config_a16w8_linear = get_a16w8_enn_quant_config(
        is_per_channel=True,
        is_qat=False,
        act_symmetric=False,
        wei_symmetric=True,
        act_observer_cls=observer,
    )

    lm_head = _find_lm_head(gm)
    for node in gm.graph.nodes:
        # Binary ops (sub/mul) must use independent quantization params.
        if node.op == "call_function" and node.target in NON_SHARED_BINARY_OPS:
            annotate_2in1out_wo_SharedQuant(node, quant_config_a16w8)

        # Changing Quant from index_put to input of Matmul to support Mixed precision.
        if _is_matmul_with_call_function_inputs(node):
            annotate_2in1out_for_mul(node, quant_config_a16w8)  # for matmul op
            annotate_from_index_put_to_matmul_input(node.args[1], quant_config_a8w8)

        # Annotate the lm_head linear with the A16W8 (high-precision) config.
        if node is lm_head:
            if "quantization_annotation" in node.meta:
                del node.meta["quantization_annotation"]
            OP_ANNOTATOR[node.target](node, quant_config_a16w8_linear)


def custom_annotate_gemma_a16w4(gm: torch.fx.GraphModule) -> None:
    # Annotate 16a8w for matmul op to get better performance
    quant_config_a16w8 = get_a16w8_enn_quant_config(
        is_per_channel=False,
        is_qat=False,
        act_symmetric=False,
        wei_symmetric=False,
        act_observer_cls=HistogramObserver,
    )
    # Annotate 8a8w for second input of matmul until past_kv_cache
    quant_config_a8w8 = get_a8w8_enn_quant_config(
        is_per_channel=False,
        is_qat=False,
        act_symmetric=True,
        wei_symmetric=True,
        act_observer_cls=HistogramObserver,
    )
    # Annotate 16a8w for lm_head op to get better performance
    quant_config_a16w8_linear = get_a16w8_enn_quant_config(
        is_per_channel=True,
        is_qat=False,
        act_symmetric=False,
        wei_symmetric=True,
        act_observer_cls=HistogramObserver,
    )

    for node in gm.graph.nodes:
        # Changing Quant from index_put to input of Matmul to support Mixed precision.
        if (
            node.op == "call_function"
            and node.target == torch.ops.aten.matmul.default
            and (
                node.args[0].op == "call_function"
                and node.args[1].op == "call_function"
            )
        ):
            annotate_2in1out_for_mul(node, quant_config_a16w8)  # for matmul op
            annotate_from_index_put_to_matmul_input(node.args[1], quant_config_a8w8)

        # added for Gemma3-1b <----------------------------------------------------
        # last linear op is A16W4 --> A16W8 <--------------------------------------
        if node.op == "call_function" and node.target == torch.ops.aten.linear.default:
            name = node.args[1]
            if str(name) == "lm_head_weight":
                if "quantization_annotation" in node.meta:
                    del node.meta["quantization_annotation"]
                OP_ANNOTATOR[node.target](node, quant_config_a16w8_linear)
