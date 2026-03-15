import logging

import torch

logger = logging.getLogger(__name__)

from torch_tensorrt.dynamo.lowering.passes.pass_utils import (
    clean_up_graph_after_modifications,
    find_complex_nodes,
)

from ._replace_complex_placeholder_to_tuple import replace_complex_placeholder_to_tuple


def tensorrt_complex_mul(args0, args1):
    args0_real, args0_imag = torch.ops.aten.split.Tensor(args0, 1, -1)
    args1_real, args1_imag = torch.ops.aten.split.Tensor(args1, 1, -1)

    args0_real = torch.ops.aten.squeeze.dim(args0_real, -1)
    args0_imag = torch.ops.aten.squeeze.dim(args0_imag, -1)
    args1_real = torch.ops.aten.squeeze.dim(args1_real, -1)
    args1_imag = torch.ops.aten.squeeze.dim(args1_imag, -1)

    complex_mul_real = torch.ops.aten.sub(
        torch.ops.aten.mul(args0_real, args1_real),
        torch.ops.aten.mul(args0_imag, args1_imag),
    )
    complex_mul_imag = torch.ops.aten.add(
        torch.ops.aten.mul(args0_real, args1_imag),
        torch.ops.aten.mul(args0_imag, args1_real),
    )

    return torch.ops.aten.stack((complex_mul_real, complex_mul_imag), -1)


def remove_complex_real_view_nodes(gm: torch.fx.GraphModule):
    modified_graph = False
    nodes_to_remove = []
    for node in gm.graph.nodes:
        if "view_as_complex" in node.name or "view_as_real" in node.name:
            nodes_to_remove.append(node)

    for node in nodes_to_remove:
        input_node = node.args[0] if node.args else None

        for other_node in gm.graph.nodes:
            new_args = tuple(
                input_node if arg is node else arg for arg in other_node.args
            )
            other_node.args = new_args

        gm.graph.erase_node(node)
        modified_graph = True

    if modified_graph:
        gm = clean_up_graph_after_modifications(gm)
        logger.debug(
            f"Graph after removing view_as_complex nodes and view_as_real nodes:\n{gm.graph}"
        )


def modify_reshape_nodes(gm: torch.fx.GraphModule, complex_nodes):
    for node in gm.graph.nodes:
        if node in complex_nodes:
            # slice and transpose will remain same
            if "reshape" in node.name:
                new_shape = list(node.args[1]) + [2]
                node.args = (node.args[0], tuple(new_shape))


def modify_mul_nodes(gm: torch.fx.GraphModule, complex_nodes):
    modified_graph = False
    for node in gm.graph.nodes:
        if node in complex_nodes:
            if "mul" in node.name:
                complex_mul_args = (node.args[0], node.args[1])
                with gm.graph.inserting_after(node):
                    replacement_node = gm.graph.create_node(
                        op="call_function",
                        target=tensorrt_complex_mul,
                        args=complex_mul_args,
                    )
                node.replace_all_uses_with(replacement_node)
                replacement_node.meta.update(node.meta)
                modified_graph = True
                gm.graph.erase_node(node)

    if modified_graph:
        gm = clean_up_graph_after_modifications(gm)
        logger.debug(
            f"Graph after custom complex mul nodes is applied to the graph:\n{gm.graph}"
        )


def modify_complex_nodes(gm: torch.fx.GraphModule, complex_nodes):
    modify_reshape_nodes(gm, complex_nodes)
    remove_complex_real_view_nodes(gm)
    modify_mul_nodes(gm, complex_nodes)


def modify_reshape_complex_nodes(gm: torch.fx.GraphModule, complexInputIndices):
    complex_nodes = find_complex_nodes(gm)
    if complex_nodes:
        replace_complex_placeholder_to_tuple(gm, complexInputIndices)
        modify_complex_nodes(gm, complex_nodes)
