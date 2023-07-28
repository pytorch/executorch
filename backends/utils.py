from typing import Iterable, List, Tuple

import torch
from executorch.exir.dialects._ops import ops as exir_ops

from executorch.exir.lowered_backend_module import create_submodule_from_nodes
from torch.fx.passes.utils.source_matcher_utils import SourcePartition

T_QuantPerTensor = exir_ops.edge.quantized_decomposed.quantize_per_tensor.default
T_DQuantPerTensor = exir_ops.edge.quantized_decomposed.dequantize_per_tensor.default


def is_same_node(
    node_left: Iterable[torch.fx.Node],
    node_right: Iterable[torch.fx.Node],
) -> bool:
    # two nodes are the same if they have the same target and op
    # same for their args
    if isinstance(node_left, torch.fx.Node) and isinstance(node_right, torch.fx.Node):
        if not (
            (node_left.target == node_right.target)
            and (node_left.op == node_right.op)
            and (len(node_left.all_input_nodes) == len(node_right.all_input_nodes))
            and all(
                [
                    is_same_node(arg_left, arg_right)
                    for arg_left, arg_right in zip(
                        node_left.all_input_nodes, node_right.all_input_nodes
                    )
                ]
            )
        ):
            return False
    else:
        if len(list(node_left)) != len(list(node_right)):
            return False
        for n_left, n_right in zip(node_left, node_right):
            # pyre-fixme[6]: For 1st argument expected `Iterable[Node]` but got `Node`.
            # pyre-fixme[6]: For 2nd argument expected `Iterable[Node]` but got `Node`.
            if not is_same_node(n_left, n_right):
                return False
    return True


def is_identical_graph(
    graph_left: torch.fx.GraphModule, graph_right: torch.fx.GraphModule
) -> bool:
    # two graph are the same if they have the same nodes and op. The order of nodes also
    # matters in this function is more strict. Two graph are not considered as the same
    # if the topological order of the nodes is the same in this function but the order of nodes
    # is not the same.
    if len(list(graph_left.graph.nodes)) != len(list(graph_right.graph.nodes)):
        return False
    for node_left, node_right in zip(graph_left.graph.nodes, graph_right.graph.nodes):
        if not (is_same_node(node_left, node_right)):
            return False
    return True


def remove_first_quant_and_last_dequant(
    graph_module: torch.fx.GraphModule,
) -> None:
    for node in graph_module.graph.nodes:
        if node.target == T_QuantPerTensor:
            if node.args[0].op == "placeholder":
                node_users = list(node.users.keys())
                for dequant_node in node_users:
                    # point the dequant arg to the placeholder
                    dequant_node.args = (node.args[0],) + dequant_node.args[1:]
        elif node.target == T_DQuantPerTensor:
            node_users = list(node.users.keys())
            if node_users[0].op == "output":
                # point the output arg to the quant node
                output_node = node_users[0]
                output_node.args = ([node.args[0]],)
    # Remove the quant/dequant nodes as they don't have users
    graph_module.graph.eliminate_dead_code()
    graph_module.recompile()


# TODO - use edge ops
def replace_quantized_partition_with_op(
    graph_module: torch.fx.GraphModule,
    partition: SourcePartition,
    replacement_op: torch._ops.OpOverloadPacket,
) -> Tuple[torch.fx.Node, List[torch.fx.Node], List[torch.fx.Node]]:

    """
    Replaces partition with the op specified by replacement_op. It's also expected that
    the nodes contained in partition are sourced from a quantized module as this function
    searches for the quantization pattern to consume along with the nodes in the partition,
    to be then replaced by replacement_op.

    Args:
        graph_module: The graph module from which this partition was sourced.
        partition: Partition to be replaced.
        replacement_op: The op to replace paritition with.
    Returns:
        Tuple: First element in the tuple is the new replaced module. The second and third
        node lists in the returned tuple consist of the dq and q nodes that were consumed
        along with this partition to be replaced by the replacement_op.
    """

    dequant_nodes = []
    quant_nodes = []
    input_nodes = []
    output_nodes = []

    partition_nodes = [node for node in partition.nodes if node not in partition.params]

    # We recreate our input nodes and output nodes list instead of using partition.input_nodes
    # and partition.output_nodes as the ordering of the nodes in those lists is not deterministic,
    # whereas for the quant fusion pass we expect deterministic ordering.
    for node in partition.nodes:
        for arg in node.args:
            if isinstance(arg, torch.fx.Node) and (arg not in partition.nodes):
                input_nodes.append(arg)

        for user in node.users.keys():
            if user not in partition.nodes:
                output_nodes.append(node)

    # Try to find all the dq nodes that are feeding into this module.
    for node in input_nodes:
        if node.target == T_DQuantPerTensor:
            dequant_nodes += [node]

    # Try to find all the q nodes that this module is feeding out into.
    for node in output_nodes:
        for user in node.users.keys():
            if user.target == T_QuantPerTensor:
                quant_nodes += [user]

    assert len(dequant_nodes) >= 1, "Dequant nodes missing in node list to be replaced."
    assert len(quant_nodes) >= 1, "Quant nodes missing in node list to be replaced."

    # After this, node list will essentially contain all the nodes in the
    # dq->op->q pattern that we will want to replace with a custom backend op.
    node_list = dequant_nodes + partition_nodes + quant_nodes

    submodule, call_module_node = create_submodule_from_nodes(
        graph_module, node_list, "to_be_replaced", skip_legalize_graph=True
    )

    # Update the replaced op so that we have all the latest args and kwargs.
    with graph_module.graph.inserting_before(call_module_node):
        replaced_op = graph_module.graph.call_function(
            replacement_op,
            call_module_node.args,
            kwargs=call_module_node.kwargs,
        )
        call_module_node.replace_all_uses_with(replaced_op)
        graph_module.graph.erase_node(call_module_node)
        replaced_op.meta = call_module_node.meta
    graph_module.recompile()

    return (replaced_op, dequant_nodes, quant_nodes)
