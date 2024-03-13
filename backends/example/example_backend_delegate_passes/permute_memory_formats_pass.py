# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from itertools import chain

import torch
from executorch.backends.example.example_operators.ops import module_to_annotator
from executorch.exir.dialects._ops import ops as exir_ops
from executorch.exir.dim_order_utils import get_dim_order
from executorch.exir.pass_base import ExportPass, PassResult
from torch.ao.quantization.pt2e.graph_utils import find_sequential_partitions


class PermuteMemoryFormatsPass(ExportPass):
    """
    This pass will insert to_dim ops to the pattern if satisfis requirement, like pattern_op.permuate_memory_format is set as True.
    Example 1:
        before pass: x -> conv -> out
        after pass: x -> to_dim(channel_last) -> conv -> to_dim_(contiguous) -> out

        before pass: x -> conv -> conv -> out
        after pass: x -> to_dim(channel_last) -> conv -> to_dim_(contiguous) -> to_dim(channel_last) -> conv -> to_dim_(contiguous) -> out

        before pass: x -> conv -> linear -> out
        after pass: x -> to_dim(channel_last) -> conv -> to_dim_(contiguous) -> to_dim(channel_last) -> linear -> to_dim_(contiguous) -> out
    """

    def call(  # noqa: suprress function is too complex (13)
        self, graph_module: torch.fx.GraphModule
    ) -> PassResult:
        for pattern in list(module_to_annotator.keys()):
            pattern_op = module_to_annotator[pattern]
            if pattern_op.permuate_memory_format:
                partitions = find_sequential_partitions(
                    graph_module,
                    pattern,
                )
                for partition in partitions:
                    # Some unpacking logic to get a flatten exit nodes list
                    output_nodes = [
                        node
                        for node in partition[0].output_nodes
                        if node.op != "placeholder"
                    ]
                    exit_nodes = [output_node.users for output_node in output_nodes]
                    exit_nodes = list(chain.from_iterable(exit_nodes))

                    """
                    # Step 1. Insert to_dim op when exit the pattern
                    # for example, if the pattern is conv, x -> conv -> out will become x -> conv -> to_dim(contiguous) -> out when permute memory format
                    # for x -> conv -> conv -> out, it will become x -> conv -> to_dim(contiguous) -> conv -> to_dim(contiguous) -> out
                    """
                    for exit_node in exit_nodes:
                        with graph_module.graph.inserting_before(exit_node):
                            # Handle the case when the pattern output is also the graph output,
                            # like, x -> conv -> out will become x -> conv -> to_dim(contiguous) -> out
                            if exit_node.op == "output":
                                exit_node_args = exit_node.args[0]
                                exit_to_dim_op = graph_module.graph.call_function(
                                    exir_ops.edge.dim_order_ops._to_dim_order_copy.default,
                                    args=exit_node_args,
                                    kwargs={
                                        "dtype": torch.float64,
                                        "dim_order": get_dim_order(
                                            torch.contiguous_format, 4
                                        ),
                                    },
                                )
                                # Insert to_dim op and it'll be the return op
                                _ = graph_module.graph.output((exit_to_dim_op,))
                                # Remove the old return op.
                                graph_module.graph.erase_node(exit_node)
                            # Handle the case when the pattern output is intermediate output,
                            # like, x -> conv -> conv -> out will become x -> conv -> to_dim(contiguous) -> conv -> out
                            elif exit_node.op == "call_function":
                                exit_node_args = []
                                for exit_node_arg in exit_node.args:
                                    if (
                                        isinstance(exit_node_arg, torch.fx.Node)
                                        and exit_node_arg.op != "placeholder"
                                    ):
                                        exit_to_dim_op = graph_module.graph.call_function(
                                            exir_ops.edge.dim_order_ops._to_dim_order_copy.default,
                                            args=(exit_node_arg,),
                                            kwargs={
                                                "dtype": torch.float64,
                                                "dim_order": get_dim_order(
                                                    torch.contiguous_format, 4
                                                ),
                                            },
                                        )
                                        exit_node_args.append(exit_to_dim_op)
                                    else:
                                        exit_node_args.append(exit_node_arg)
                                exit_node.args = list(exit_node_args)

                    """
                    # Step 2. Insert to_dim op when enter the pattern. After the first step, we already have to_dim(default) when exiting the pattern.
                    # Now we need to insert to_dim(channel_last) when enter the pattern.
                    # for example, if the pattern is conv, x -> conv -> to_dim(contiguous) -> out will become x -> to_dim(channel_last) -> conv -> to_dim(contiguous) -> out
                    # for x -> conv -> to_dim(contiguous) -> conv -> to_dim(contiguous) -> out, it will become x -> to_dim(channel_last) -> conv -> to_dim(contiguous) -> to_dim(channel_last) -> conv -> to_dim(contiguous) -> out
                    """
                    # create the input_node and the to_dim_op map
                    # for example, if the pattern is conv, x -> conv -> out, node
                    input_node_map = {}  # key: input_node, value: to_dim_op
                    to_dim_op_set = set()
                    for input_node in partition[0].input_nodes:
                        with graph_module.graph.inserting_after(input_node):
                            to_dim_op = graph_module.graph.call_function(
                                # Insert the to_dim op and update input_node_map
                                exir_ops.edge.dim_order_ops._to_dim_order_copy.default,
                                args=(input_node,),
                                kwargs={
                                    "dtype": torch.float64,
                                    "dim_order": get_dim_order(torch.channels_last, 4),
                                },
                            )
                            input_node_map[input_node] = to_dim_op
                            to_dim_op_set.add(to_dim_op)

                    # Update the args to the new to_dim op, skip if it's already set
                    for input_node in partition[0].input_nodes:
                        for user in list(input_node.users):
                            # if user is in to_dim_op_set, it means the user's arg is already set to_dim op
                            if user not in to_dim_op_set:
                                user_new_arg = [
                                    (
                                        input_node_map[user_arg]
                                        if user_arg in input_node_map
                                        else user_arg
                                    )
                                    for user_arg in user.args
                                ]
                                # Update input node's users arg
                                user.args = tuple(user_new_arg)

        # Ensure the graph is still valid
        graph_module.graph.lint()
        graph_module.recompile()
        return PassResult(graph_module, True)
