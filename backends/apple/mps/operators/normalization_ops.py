#
#  Copyright (c) 2023 Apple Inc. All rights reserved.
#  Provided subject to the LICENSE file in the top level directory.
#

from typing import cast, List

import torch
from executorch.backends.apple.mps.operators.node_visitor import (
    NodeVisitor,
    register_node_visitor,
)

from executorch.backends.apple.mps.serialization.mps_graph_schema import (
    MPSBatchNorm,
    MPSGraph,
    MPSLayerNorm,
    MPSNode,
)
from executorch.backends.apple.mps.utils.mps_utils import get_input_node, get_scalar_val
from executorch.exir.sym_util import eval_shape


@register_node_visitor
class BatchNorm(NodeVisitor):
    target = "aten._native_batch_norm_legit_no_training.default"

    def __init__(self, *args) -> None:
        super().__init__(*args)

    def define_node(
        self,
        node: torch.fx.Node,
        mps_graph: MPSGraph,
    ) -> None:

        input_id = self.define_tensor(get_input_node(node, 0), mps_graph)
        weight_id = self.define_tensor(get_input_node(node, 1), mps_graph)
        bias_id = self.define_tensor(get_input_node(node, 2), mps_graph)
        mean_id = self.define_tensor(get_input_node(node, 3), mps_graph)
        var_id = self.define_tensor(get_input_node(node, 4), mps_graph)
        momentum: float = get_scalar_val(node, 5)
        epsilon: float = get_scalar_val(node, 6)

        output1_id, output2_id, output3_id = self.define_tensor_list(node, mps_graph)

        mps_node = MPSNode(
            mpsnode_union=MPSBatchNorm(
                input_id=input_id,
                mean_id=mean_id,
                var_id=var_id,
                weight_id=weight_id,
                bias_id=bias_id,
                momentum=momentum,
                epsilon=epsilon,
                output1_id=output1_id,
                output2_id=output2_id,
                output3_id=output3_id,
            )
        )
        mps_graph.mps_nodes.append(mps_node)


@register_node_visitor
class LayerNorm(NodeVisitor):
    target = "aten.native_layer_norm.default"

    def __init__(self, *args) -> None:
        super().__init__(*args)

    def define_node(
        self,
        node: torch.fx.Node,
        mps_graph: MPSGraph,
    ) -> None:

        input1_id = self.define_tensor(get_input_node(node, 0), mps_graph)
        normalized_shape = eval_shape(cast(List[torch.SymInt], node.args[1]))
        weight_id = self.define_tensor(get_input_node(node, 2), mps_graph)
        bias_id = self.define_tensor(get_input_node(node, 3), mps_graph)
        epsilon: float = get_scalar_val(node, 4)
        output1_id, output2_id, output3_id = self.define_tensor_list(node, mps_graph)

        mps_graph.mps_nodes.append(
            MPSNode(
                mpsnode_union=MPSLayerNorm(
                    input1_id=input1_id,
                    normalized_shape=normalized_shape,
                    weight_id=weight_id,
                    bias_id=bias_id,
                    eps=epsilon,
                    output1_id=output1_id,
                    output2_id=output2_id,
                    output3_id=output3_id,
                )
            )
        )
