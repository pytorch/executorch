# Copyright (c) Qualcomm Innovation Center, Inc.
# All rights reserved
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.
from typing import Sequence

import torch
from executorch.backends.qualcomm.quantizer.quantizer import (
    get_16a8w_qnn_ptq_config,
    get_default_8bit_qnn_ptq_config,
    QuantizationConfig,
)
from executorch.backends.qualcomm.quantizer.utils import QUANT_ANNOTATION_KEY
from torch.ao.quantization.quantizer import (
    QuantizationAnnotation,
    SharedQuantizationSpec,
)
from torch.fx import Node


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
        node.meta[QUANT_ANNOTATION_KEY] = QuantizationAnnotation(
            input_qspec_map=input_qspec_map,
            output_qspec=quantization_config.output_activation,
            _annotated=True,
        )

    def annotate_index_put(node: Node, quantization_config: QuantizationConfig) -> None:
        input = node.args[0]
        value = node.args[2]
        input_qspec_map = {}
        input_qspec_map[input] = quantization_config.input_activation
        input_qspec_map[value] = SharedQuantizationSpec((input, node))
        node.meta[QUANT_ANNOTATION_KEY] = QuantizationAnnotation(
            input_qspec_map=input_qspec_map,
            output_qspec=SharedQuantizationSpec((input, node)),
            _annotated=True,
        )

    def annotate_single_in_single_out(
        node: Node, quantization_config: QuantizationConfig
    ) -> None:
        input_qspec_map = {}
        input_act = node.args[0]
        input_qspec_map[input_act] = quantization_config.input_activation
        node.meta[QUANT_ANNOTATION_KEY] = QuantizationAnnotation(
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
        node.meta[QUANT_ANNOTATION_KEY] = QuantizationAnnotation(
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
    quantization_config_8a8w = get_default_8bit_qnn_ptq_config(act_symmetric=True)
    for node in gm.graph.nodes:
        if node.op == "call_function" and node.target == torch.ops.aten.matmul.default:
            if "nn_module_stack" in node.meta:
                module_values_list = list(node.meta["nn_module_stack"].values())
                full_qualified_name = module_values_list[-1][0]
                if "SDPA" in full_qualified_name:
                    annotate_matmul(node, quantization_config_16a8w)
                    annotate_matmul_input1(node.args[1], quantization_config_8a8w)
