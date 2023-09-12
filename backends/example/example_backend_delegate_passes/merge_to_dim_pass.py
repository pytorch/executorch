# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.


import torch
from executorch.exir.dialects._ops import ops as exir_ops
from executorch.exir.dim_order_utils import get_dim_order
from executorch.exir.pass_base import ExportPass, PassResult


class MergeToDimPass(ExportPass):
    """
    This pass will insert to_dim ops to the pattern if satisfis requirement, like pattern_op.permuate_memory_format is set as True.
    Example:
        # Done for 1 to 1
        before pass: x -> to_dim(channel_last) -> conv -> to_dim_(contiguous) -> to_dim(channel_last) -> conv -> to_dim_(contiguous) -> out
        after pass: x -> to_dim(channel_last) -> conv -> conv -> to_dim_(contiguous) -> out

        # Not Done for 1 to N
        before pass: x -> to_dim(channel_last) -> conv -> to_dim_(contiguous) -> to_dim(channel_last) -> conv -> to_dim_(contiguous) -> out
                                                                 |-------------> to_dim(channel_last) -> conv -> to_dim_(contiguous) -> out
        after pass: x -> to_dim(channel_last) -> conv -> to_dim_(contiguous) -> to_dim(channel_last) -> conv -> to_dim_(contiguous) -> out
                                                               |--------------> to_dim(channel_last) -> conv -> to_dim_(contiguous) -> out

        # Not Done for N to 1
        before pass: x -> to_dim(channel_last) -> conv -> to_dim_(contiguous) -> to_dim(channel_last) -> conv -> to_dim_(contiguous) -> out
                     y -> to_dim(channel_last) -> conv -> to_dim_(contiguous) ---------|
        after pass:  x -> to_dim(channel_last) -> conv -> conv -> to_dim_(contiguous) -> out
                     y -> to_dim(channel_last) -> conv-----|

        # Not Done for N to N
    """

    def call(self, graph_module: torch.fx.GraphModule) -> PassResult:
        for node in graph_module.graph.nodes:
            if node.target == exir_ops.edge.dim_order_ops._to_dim_order_copy.default:
                # print(node, node.args, list(node.users), list(list(node.users)[0].args))
                if len(node.users) == 1 and len(list(node.users)[0].args) == 2:
                    args_map = {}
                    node_kwargs = node.args[-1]
                    node_users = list(node.users)

                    in_to_dim_node_dim_order = node_kwargs["dim_order"]
                    in_to_dim_node_dtype = node_kwargs["dtype"]
                    out_to_dim_node = node_users[0]
                    out_to_dim_node_kwargs = out_to_dim_node.args[-1]
                    out_to_dim_node_dim_order = out_to_dim_node_kwargs["dim_order"]
                    out_to_dim_node_dtype = out_to_dim_node_kwargs["dtype"]

                    if (
                        in_to_dim_node_dtype == out_to_dim_node_dtype
                        and in_to_dim_node_dim_order
                        == get_dim_order(torch.channels_last, 4)
                        and out_to_dim_node_dim_order
                        == get_dim_order(torch.contiguous_format, 4)
                    ):

                        out_to_dim_node_users = list(out_to_dim_node.users)
                        assert len(out_to_dim_node_users) == 1
                        out_to_dim_node_user = out_to_dim_node_users[0]
                        args_map[out_to_dim_node] = node.args[0]
                        out_to_dim_node_user_new_args = [
                            args_map[out_to_dim_node] if arg in args_map else arg
                            for arg in out_to_dim_node_user.args
                        ]
                        print("out_to_dim_node_user.args: ", out_to_dim_node_user.args)
                        print(
                            "out_to_dim_node_user_new_args: ",
                            out_to_dim_node_user_new_args,
                        )
                        out_to_dim_node_user.args = tuple(out_to_dim_node_user_new_args)

                        graph_module.erase_node(out_to_dim_node)
                        graph_module.erase_node(node)
            # TODO: Handle other merging rules, including 1->N, N->1, N->N
        return PassResult(graph_module, True)
