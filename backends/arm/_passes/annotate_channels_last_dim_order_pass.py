# Copyright 2024 Arm Limited and/or its affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-unsafe

from typing import cast

import torch
from executorch.backends.arm.tosa_quant_utils import dq_op
from executorch.backends.arm.tosa_utils import is_consumer_node_depthwise_conv2d
from executorch.exir.pass_base import ExportPass, PassResult


class AnnotateChannelsLastDimOrder(ExportPass):
    """
    Annotates each node with a tosa_dim_order. tosa_dim_order can be seen as a channels-last dim-order
    that in most cases will be (0, 2, 3, 1) for nodes with 4D-shapes.
    The annotated tosa_dim_order is used to permute the node's shape such that it
    gives a TOSA-compliant shape.
    """

    def is_weight_node_for_depthwise_conv2d(self, node: torch.fx.Node):
        """
        returns True for dq and w in the following sequences;
        w -> depthwise_conv2d -> ...
        w -> dq -> depthwise_conv2d -> ...
        """
        if node.op == "call_function":
            if node.target != dq_op:
                return False
            prev_node = node.args[0]
            if cast(torch.fx.Node, prev_node).op != "placeholder":
                return False
            if is_consumer_node_depthwise_conv2d(node):
                consumer_node = list(node.users)[0]
                return consumer_node.args[1] == node
        elif node.op == "placeholder":
            # node is an input, weight or bias node
            consumer_node = list(node.users)[0]
            if self.is_weight_node_for_depthwise_conv2d(consumer_node):
                return True
            if is_consumer_node_depthwise_conv2d(node):
                # Check that node is the weight-argument and not input or bias
                return consumer_node.args[1] == node

        return False

    def call(self, graph_module: torch.fx.GraphModule):
        NHWC_Order = (0, 2, 3, 1)
        HWCM_Order = (2, 3, 0, 1)
        for node in graph_module.graph.nodes:
            if isinstance(
                node.meta["val"], (tuple, torch.fx.immutable_collections.immutable_list)
            ):
                node_data = node.meta["val"][0].data
            else:
                node_data = node.meta["val"].data

            if len(node_data.shape) == 4:
                dim_order = NHWC_Order
                if self.is_weight_node_for_depthwise_conv2d(node):
                    # The weights of TOSA DEPTHWISE_CONV2D have shape (H, W, C, M) which corresponds to
                    # dim_order = (2, 3, 0, 1) (https://www.mlplatform.org/tosa/tosa_spec.html#_depthwise_conv2d).
                    dim_order = HWCM_Order
            else:
                dim_order = tuple(range(node_data.dim()))
            node.meta["tosa_dim_order"] = dim_order
        graph_module.recompile()
        return PassResult(graph_module, True)
