# Copyright (c) Qualcomm Innovation Center, Inc.
# All rights reserved
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from typing import Sequence

import torch
from executorch.backends.qualcomm.quantizer.annotators import (
    _is_float_tensor,
    Q_ANNOTATION_KEY,
)
from executorch.backends.qualcomm.quantizer.quantizer import (
    get_16a8w_qnn_ptq_config,
    get_16a8w_qnn_qat_config,
    get_8a8w_qnn_ptq_config,
    get_8a8w_qnn_qat_config,
    get_ptq_per_channel_quant_config,
    QuantizationConfig,
)
from executorch.exir.dialects._ops import ops as exir_ops
from torch.fx import Node
from torchao.quantization.pt2e import FixedQParamsObserver, MinMaxObserver
from torchao.quantization.pt2e.quantizer import (
    annotate_input_qspec_map,
    annotate_output_qspec,
    QuantizationAnnotation,
    QuantizationSpec,
    SharedQuantizationSpec,
)


def annotate_eurobert(gm: torch.fx.GraphModule):
    """
    QNN does not support int32 -> signed 16bit quant
    We need to first annotate this to_fp node as 8bit quant, so it will perform requantize
    Final graph should look like: int32 -> convert -> cast -> matmul.args[1]

    """
    quantization_config_8a8w = get_8a8w_qnn_ptq_config()
    for node in gm.graph.nodes:
        # A little tricky here. This matmul node is wrapped inside a submodule after 1st torch.export.
        # There are actually 2 'to' op that is redundant.
        # It will look like: int64 -> to_fp -> to_fp -> matmul.args[1]
        # Draw out the graph after the 1st export will help visualize the submodule.

        if node.target == torch.ops.aten.matmul.default and node.args[1].args[0].args[
            0
        ].meta["val"].dtype in [torch.int64, torch.int32]:
            to_node = node.args[1]
            input_qspec_map = {}
            assert isinstance(to_node, Node)
            input_spec = quantization_config_8a8w.input_activation
            input_qspec_map[to_node] = input_spec
            to_node.meta[Q_ANNOTATION_KEY] = QuantizationAnnotation(
                input_qspec_map=input_qspec_map,
                output_qspec=quantization_config_8a8w.output_activation,
                _annotated=True,
            )


def annotate_mimi_decoder(gm: torch.fx.GraphModule):
    """
    The 1st transpose conv in mimi decoder is really sensitive to scale/offset in 16a8w, which causes execution failure.
    Annotate 1st transpose conv as 8a8w to prevent execution failure.
    """
    quantization_config_8a8w = get_8a8w_qnn_ptq_config()
    for node in gm.graph.nodes:
        if not _is_float_tensor(node):
            continue
        elif node.target == torch.ops.aten.conv_transpose1d.default:
            input_qspec_map = {}
            input_act = node.args[0]
            assert isinstance(input_act, Node)
            input_spec = quantization_config_8a8w.input_activation
            input_qspec_map[input_act] = input_spec

            weight = node.args[1]
            assert isinstance(weight, Node)
            input_qspec_map[weight] = quantization_config_8a8w.weight

            if len(node.args) > 2 and isinstance(node.args[2], Node):
                bias = node.args[2]
                input_qspec_map[bias] = quantization_config_8a8w.bias

            node.meta[Q_ANNOTATION_KEY] = QuantizationAnnotation(
                input_qspec_map=input_qspec_map,
                output_qspec=quantization_config_8a8w.output_activation,
                _annotated=True,
            )
            break


def annotate_prefill_kv_output(gm: torch.fx.GraphModule, kv_quant_attrs: dict):
    for node in gm.graph.nodes:
        if node.op == "output":
            for index, prefill_output in enumerate(node.args[0]):
                kv_quant_attr = kv_quant_attrs[index]
                fixed_observer = FixedQParamsObserver.with_args(
                    scale=kv_quant_attr[0],
                    zero_point=kv_quant_attr[1],
                    quant_min=kv_quant_attr[2],
                    quant_max=kv_quant_attr[3],
                    dtype=kv_quant_attr[4],
                    qscheme=torch.torch.per_tensor_affine,
                )

                fixed_output_spec = QuantizationSpec(
                    quant_min=kv_quant_attr[2],
                    quant_max=kv_quant_attr[3],
                    dtype=kv_quant_attr[4],
                    ch_axis=0,
                    observer_or_fake_quant_ctr=fixed_observer,
                )

                input_qspec_map = {}
                for input in prefill_output.args:
                    if isinstance(input, Node):
                        input_qspec_map[input] = fixed_output_spec

                prefill_output.meta[Q_ANNOTATION_KEY] = QuantizationAnnotation(
                    input_qspec_map=input_qspec_map,
                    output_qspec=fixed_output_spec,
                    _annotated=True,
                )


def annotate_kv_8bit(  # noqa: C901
    gm: torch.fx.GraphModule,
    is_qat=False,
) -> None:
    """
    This function is for static LLM models.
    This function is specific for matmul op 16a8w.
    For k, we will tag such as the below, and
    for v, we will tag 8a until conv op.
                                                              q (16 bits) ──┬─> matmul op (16 bits)
                                       past k (8 bits) ┬─> cat op (8 bits) ─┘
    rotatary add (16 bits) ─┬> cat op (new k) (8 bits) ┘
    rotatary sub (16 bits) ─┘
    """

    def annotate_matmul(node: Node, quantization_config: QuantizationConfig):
        input_qspec_map = {}
        input_act = node.args[0]
        input_spec = quantization_config.input_activation
        input_qspec_map[input_act] = input_spec
        input_act1 = node.args[1]
        input_spec1 = quantization_config.weight
        input_qspec_map[input_act1] = input_spec1

        node.meta[Q_ANNOTATION_KEY] = QuantizationAnnotation(
            input_qspec_map=input_qspec_map,
            output_qspec=quantization_config.output_activation,
            _annotated=True,
        )

    def annotate_cat(node: Node, quantization_config: QuantizationConfig):
        input_nodes = node.args[0]

        first_input_node = input_nodes[0]
        input_qspec_map = {}
        input_qspec_map[first_input_node] = quantization_config.input_activation
        share_qparams_with_input_act0_qspec = SharedQuantizationSpec(
            (first_input_node, node)
        )

        for input_node in input_nodes[1:]:
            if input_node not in input_qspec_map:
                input_qspec_map[input_node] = share_qparams_with_input_act0_qspec

        node.meta[Q_ANNOTATION_KEY] = QuantizationAnnotation(
            input_qspec_map=input_qspec_map,
            output_qspec=share_qparams_with_input_act0_qspec,
            _annotated=True,
        )

    def annotate_rms_norm(node: Node, quantization_config: QuantizationConfig) -> None:
        act_node = node.args[0]
        weight_node = node.args[2]

        # TODO current only support 16a16w
        annotate_input_qspec_map(
            node,
            act_node,
            quantization_config.input_activation,
        )

        annotate_input_qspec_map(
            node,
            weight_node,
            quantization_config.input_activation,
        )
        annotate_output_qspec(node, quantization_config.output_activation)

    def annotate_single_in_single_out(
        node: Node, quantization_config: QuantizationConfig
    ) -> None:
        input_qspec_map = {}
        input_act = node.args[0]
        input_qspec_map[input_act] = quantization_config.input_activation

        node.meta[Q_ANNOTATION_KEY] = QuantizationAnnotation(
            input_qspec_map=input_qspec_map,
            output_qspec=quantization_config.output_activation,
            _annotated=True,
        )

    def annotate_single_in_share_out(
        node: Node, quantization_config: QuantizationConfig
    ) -> None:
        input_qspec_map = {}
        input_act = node.args[0]
        input_qspec_map[input_act] = quantization_config.input_activation

        node.meta[Q_ANNOTATION_KEY] = QuantizationAnnotation(
            input_qspec_map=input_qspec_map,
            output_qspec=SharedQuantizationSpec((input_act, node)),
            _annotated=True,
        )

    def annotate_stack(node: Node, quantization_config: QuantizationConfig) -> None:
        input_nodes = node.args[0]

        first_input_node = input_nodes[0]
        input_qspec_map = {}
        input_qspec_map[first_input_node] = quantization_config.input_activation
        share_qparams_with_input_act0_qspec = SharedQuantizationSpec(
            (first_input_node, node)
        )

        for input_node in input_nodes[1:]:
            if input_node not in input_qspec_map:
                input_qspec_map[input_node] = share_qparams_with_input_act0_qspec

        node.meta[Q_ANNOTATION_KEY] = QuantizationAnnotation(
            input_qspec_map=input_qspec_map,
            output_qspec=share_qparams_with_input_act0_qspec,
            _annotated=True,
        )

    def annotate_matmul_input1(node: Node, is_qat: str):
        if is_qat:
            quantization_config_8a8w = get_8a8w_qnn_qat_config(
                act_symmetric=True, act_observer=MinMaxObserver
            )
        else:
            quantization_config_8a8w = get_8a8w_qnn_ptq_config(
                act_symmetric=True, act_observer=MinMaxObserver
            )
        while isinstance(node, Node) and node.op == "call_function":
            if node.target in [
                torch.ops.aten.permute.default,
                torch.ops.aten.squeeze.dim,
                torch.ops.aten.transpose.int,
                torch.ops.aten.view.default,
                torch.ops.aten.reshape.default,
                torch.ops.aten.select.int,
                torch.ops.aten.slice.Tensor,
                torch.ops.aten.expand.default,
                torch.ops.aten.unsqueeze.default,
            ]:
                annotate_single_in_single_out(node, quantization_config_8a8w)
                node = node.args[0]
            elif node.target == torch.ops.aten.stack.default:
                annotate_stack(node, quantization_config_8a8w)
                node = node.args[0]
            elif node.target == torch.ops.aten.flatten.using_ints:
                annotate_single_in_share_out(node, quantization_config_8a8w)
                node = node.args[0]
            elif node.target == torch.ops.aten.rms_norm.default:
                annotate_rms_norm(node, quantization_config_8a8w)
                node = node.args[0]
            elif node.target == torch.ops.aten.cat.default:
                annotate_cat(node, quantization_config_8a8w)
                # For v, we tag 8a until conv op.
                # For k, we tag 8a until add or sub op (rotatary embedding).
                # The arguments of cat op: (the past kv cache, the new kv cache)
                node = node.args[0][1]
            elif node.target in [
                torch.ops.aten.add.Tensor,
                torch.ops.aten.sub.Tensor,
                torch.ops.aten.matmul.default,
                torch.ops.aten.conv2d.default,
            ]:
                break
            else:
                print(f"The node ({node}) is not expected in the input1 of the matmul")
                node = node.args[0]

    if is_qat:
        quantization_config_16a8w = get_16a8w_qnn_qat_config(
            act_observer=MinMaxObserver
        )
    else:
        quantization_config_16a8w = get_16a8w_qnn_ptq_config(
            act_observer=MinMaxObserver
        )

    for node in gm.graph.nodes:
        if (
            node.op == "call_function"
            and node.target == torch.ops.aten.matmul.default
            and all(arg.op == "call_function" for arg in node.args)
        ):
            # Only apply custom annotation on Q @ K^T @ V
            annotate_matmul(node, quantization_config_16a8w)
            annotate_matmul_input1(node.args[1], is_qat=is_qat)


def custom_annotate_llama_matmul_16a8w(gm: torch.fx.GraphModule) -> None:  # noqa: C901
    """
    This function is specific for llama matmul op 16a8w.
    """

    def annotate_matmul(node: Node, quantization_config: QuantizationConfig):
        input_qspec_map = {}
        input_act = node.args[0]
        input_spec = quantization_config.input_activation
        input_qspec_map[input_act] = input_spec
        input_act1 = node.args[1]
        input_spec1 = quantization_config.weight
        input_qspec_map[input_act1] = input_spec1
        node.meta[Q_ANNOTATION_KEY] = QuantizationAnnotation(
            input_qspec_map=input_qspec_map,
            output_qspec=quantization_config.output_activation,
            _annotated=True,
        )

    def annotate_index_put(node: Node, quantization_config: QuantizationConfig) -> None:
        # Avoid annotating the input node because mutable buffers will be folded during the convert_pt2e process.
        value = node.args[2]

        input_qspec_map = {}
        input_qspec_map[value] = quantization_config.input_activation

        node.meta[Q_ANNOTATION_KEY] = QuantizationAnnotation(
            input_qspec_map=input_qspec_map,
            output_qspec=SharedQuantizationSpec((value, node)),
            _annotated=True,
        )

    def annotate_single_in_single_out(
        node: Node, quantization_config: QuantizationConfig
    ) -> None:
        input_qspec_map = {}
        input_act = node.args[0]
        input_qspec_map[input_act] = quantization_config.input_activation
        node.meta[Q_ANNOTATION_KEY] = QuantizationAnnotation(
            input_qspec_map=input_qspec_map,
            output_qspec=quantization_config.output_activation,
            _annotated=True,
        )

    def annotate_cat(node: Node, quantization_config: QuantizationConfig):
        input_nodes = node.args[0]
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
        node.meta[Q_ANNOTATION_KEY] = QuantizationAnnotation(
            input_qspec_map=input_qspec_map,
            output_qspec=share_qparams_with_input_act0_qspec,
            _annotated=True,
        )

    def is_edge_condition(node: Node):
        if not isinstance(node, Node) or node.op != "call_function":
            return True
        return False

    def annotate_matmul_input1(node: Node, quantization_config: QuantizationConfig):
        if is_edge_condition(node):
            return
        if node.target in [
            torch.ops.aten.index_put.default,
            torch.ops.aten.index_put_.default,
        ]:
            annotate_index_put(node, quantization_config)
            annotate_matmul_input1(node.args[0], quantization_config)
        elif node.target == torch.ops.aten.cat.default:
            annotate_cat(node, quantization_config)
            # Expect that the inputs of the cat op are select ops
            for arg in node.args[0]:
                annotate_matmul_input1(arg, quantization_config)
        else:
            annotate_single_in_single_out(node, quantization_config)
            annotate_matmul_input1(node.args[0], quantization_config)

    # Annotate 16a8w for matmul op to get better performance
    quantization_config_16a8w = get_16a8w_qnn_ptq_config()
    # Annotate 8a8w for second input of matmul until past_kv_cache
    quantization_config_8a8w = get_8a8w_qnn_ptq_config(act_symmetric=True)
    for node in gm.graph.nodes:
        if node.op == "call_function" and node.target == torch.ops.aten.matmul.default:
            if "nn_module_stack" in node.meta:
                module_values_list = list(node.meta["nn_module_stack"].values())
                full_qualified_name = module_values_list[-1][0]
                if "SDPA" in full_qualified_name:
                    annotate_matmul(node, quantization_config_16a8w)
                    annotate_matmul_input1(node.args[1], quantization_config_8a8w)


def custom_annotate_llama_last_conv_16a8w(gm: torch.fx.GraphModule) -> None:
    def annotate_conv2d(node: Node, quantization_config: QuantizationConfig) -> None:
        input_qspec_map = {}
        input_act = node.args[0]
        input_spec = quantization_config.input_activation
        input_qspec_map[input_act] = input_spec

        weight = node.args[1]
        input_qspec_map[weight] = quantization_config.weight

        node.meta[Q_ANNOTATION_KEY] = QuantizationAnnotation(
            input_qspec_map=input_qspec_map,
            output_qspec=quantization_config.output_activation,
            _annotated=True,
        )

    quantization_config_16a8w_per_channel = get_ptq_per_channel_quant_config(
        torch.uint16, weight_dtype=torch.int8
    )
    for node in gm.graph.nodes:
        if node.op == "call_function" and node.target == torch.ops.aten.conv2d.default:
            if "nn_module_stack" in node.meta:
                module_values_list = list(node.meta["nn_module_stack"].values())
                full_qualified_name = module_values_list[0][0]
                if full_qualified_name == "L['self'].llama.output":
                    annotate_conv2d(
                        node, quantization_config=quantization_config_16a8w_per_channel
                    )


def custom_annotate_matmul_16a8w(gm: torch.fx.GraphModule):
    """
    Annotate matmul op with 16a8w quantization config
    """

    def annotate_matmul(node: Node, quantization_config: QuantizationConfig):
        input_qspec_map = {}
        input_act = node.args[0]
        input_spec = quantization_config.input_activation
        input_qspec_map[input_act] = input_spec
        input_act1 = node.args[1]
        input_spec1 = quantization_config.weight
        input_qspec_map[input_act1] = input_spec1
        node.meta[Q_ANNOTATION_KEY] = QuantizationAnnotation(
            input_qspec_map=input_qspec_map,
            output_qspec=quantization_config.output_activation,
            _annotated=True,
        )

    # Annotate 16a8w for matmul op to get better performance
    quantization_config_16a8w = get_16a8w_qnn_ptq_config()
    for node in gm.graph.nodes:
        if node.op == "call_function" and node.target == torch.ops.aten.matmul.default:
            annotate_matmul(node, quantization_config_16a8w)


def get_custom_quant_ios_dtype(
    cache_shape: torch.Size,
    node: torch.fx.Node,
    kv_dtype=torch.uint8,
    sharding_dtype=torch.uint16,
):
    """
    This function is specific for llama inputs and outputs
    """
    if node.op == "placeholder" and "attention_kv_cache_past_" in node.name:
        return kv_dtype

    # Tag index put node before copy node, because copy is a skipped node in qnn
    if (
        exir_ops.edge.aten.index_put.default == node.target
        and node.meta["val"].shape == cache_shape
    ):
        return kv_dtype

    # Tag sharding io
    if exir_ops.edge.llama.fallback.default in [
        u.target for u in list(node.users.keys())
    ] + [node.target]:
        return sharding_dtype

    # Tag index op as quantized tensors. It is caused by sharding
    if exir_ops.edge.aten.index.Tensor in [
        u.target for u in list(node.users.keys())
    ] + [node.target]:
        return sharding_dtype
