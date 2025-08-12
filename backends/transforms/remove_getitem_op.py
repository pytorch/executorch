# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import torch
from executorch.exir.dialects._ops import ops as exir_ops

from executorch.exir.pass_base import ExportPass, PassResult


class RemoveGetItemPass(ExportPass):
    """
    This remove item is used to remove getitem operator for max_pool2d_with_indices.default operator, and replace it with a single operator,
    that exratacts the first output. More speciafially, we are only getting the first output from aten::maxpool2d operator.
    Before Pass:
        MaxPool2d ---> GetItem[max_values, max_indexes]
    After Pass:
        MaxPool2d -> max_values
    """

    def call(self, graph_module: torch.fx.GraphModule):
        mdule = graph_module
        for node in mdule.graph.nodes:
            if node.op == "call_function":
                if (
                    node.target.__name__ == "aten.max_pool2d_with_indices.default"
                    or node.target.__name__ == "aten.max.dim"
                ):
                    users = list(node.users.keys())

                    if len(users) != 1:
                        if len(users) == 2 and node.target.__name__ == "aten.max.dim":
                            # Two users is allowed for max.dim. For that case,
                            # rather than removing the getitem node in this
                            # pass, we handle the getitem nodes in the op's
                            # visitor when serializing
                            continue
                        else:
                            raise AssertionError(
                                f"Invalid number of users for {node.target.__name__ }: {len(users)}"
                            )

                    getitem_node = list(node.users.keys())[0]

                    if getitem_node.target.__name__ != "getitem":
                        raise AssertionError(
                            f"Expected max node's user to be getitem, got {getitem_node.target.__name__}"
                        )

                    getitem_index = getitem_node.args[1]

                    with mdule.graph.inserting_before(node):
                        if (
                            node.target.__name__
                            == "aten.max_pool2d_with_indices.default"
                        ):
                            if getitem_index != 0:
                                raise AssertionError(
                                    f"Expected second argument of getitem node for {node.target.__name__ } to be 0, got {getitem_index}. XNNPACK delegate currently only supports getting just the max values from the op but not getting the corresponding indices."
                                )
                            new_max_wd = mdule.graph.create_node(
                                "call_function",
                                exir_ops.edge.aten.max_pool2d.default,
                                args=node.args,
                                kwargs=node.kwargs,
                            )
                        else:
                            if getitem_index != 0:
                                raise AssertionError(
                                    f"Expected second argument of getitem node for {node.target.__name__ } to be 0, got {getitem_index}. XNNPACK delegate currently only supports getting just the max values or getting both the max values and their corresponding indices from the op, but not getting the indices alone."
                                )
                            new_max_wd = mdule.graph.create_node(
                                "call_function",
                                exir_ops.edge.aten.amax.default,
                                args=node.args,
                                kwargs=node.kwargs,
                            )

                    getitem_node.replace_all_uses_with(new_max_wd)

                    mdule.graph.erase_node(getitem_node)
                    mdule.graph.erase_node(node)

        graph_module.recompile()
        # Propagate metadata and retrace module
        graph_module = super().call(graph_module).graph_module

        return PassResult(graph_module, True)
