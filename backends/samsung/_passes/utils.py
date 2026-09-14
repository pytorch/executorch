# Copyright (c) Qualcomm Innovation Center, Inc.
# Copyright (c) 2025 Samsung Electronics Co. LTD
# All rights reserved
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from typing import Callable, Dict

import torch


def none_quant_tensor_quant_meta():
    return {
        "quant_dtype": torch.float32,
        "scales": 1,
        "zero_points": 0,
    }


def copy_nn_module_stack(src, target):
    """
    Copy meta["nn_module_stack"] from src node to target node if existing.
    """
    if value := src.meta.get("nn_module_stack"):
        target.meta["nn_module_stack"] = value


def merge_decomposed_graph(
    remap: Dict[str, torch.fx.Node],
    target_node: torch.fx.Node,
    target_graph: torch.fx.GraphModule,
    decomposed_graph_module: torch.fx.GraphModule,
    predicate: Callable[[torch.fx.Node], None] = None,
    # target_node, decomposed_output_node, remap
    output_processor: Callable[
        [torch.fx.Node, torch.fx.Node, Dict[str, torch.fx.Node]], None
    ] = None,
) -> None:
    def default_output_process(node):
        for user in node.users.copy():
            # remap
            user.replace_input_with(
                node,
                remap[decomposed_node.args[0][0]],
            )

    for decomposed_node in decomposed_graph_module.graph.nodes:
        copy_nn_module_stack(target_node, decomposed_node)
        if predicate is None or predicate(decomposed_node):
            # no need to copy existent 'output'
            if decomposed_node.op == "output":
                if output_processor is None:
                    default_output_process(target_node)
                else:
                    output_processor(target_node, decomposed_node, remap)
            # no need to copy existent placeholders
            elif decomposed_node.op == "placeholder":
                # replace node map from string to graph node
                remap[decomposed_node] = remap.pop(decomposed_node.name)
            else:
                remap[decomposed_node] = target_graph.node_copy(
                    decomposed_node,
                    arg_transform=lambda x, remap=remap: remap[x],
                )
