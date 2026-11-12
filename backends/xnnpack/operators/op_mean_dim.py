# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
# Copyright 2026 Arm Limited and/or its affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from typing import Dict

import torch
from executorch.backends.xnnpack._passes.channels_last_tagged_reshape_pass import (
    ChannelsLastTaggedReshapePass,
)
from executorch.backends.xnnpack.operators.node_visitor import (
    check_or_raise,
    get_tensor_value,
    NodeVisitor,
    register_node_visitor,
)
from executorch.backends.xnnpack.serialization.xnnpack_graph_schema import (
    XNNGlobalAvgPooling2d,
    XNNGraph,
    XNNStaticMean,
    XNode,
)
from executorch.backends.xnnpack.utils.utils import (
    get_input_node,
    is_global_avg_pool_mean,
    normalize_mean_dims,
)
from executorch.backends.xnnpack.utils.xnnpack_constants import XNN_FLAG_KEEP_DIMS


@register_node_visitor
class MeanDim(NodeVisitor):
    """
    A mean.dim that is not bf16, over the innermost dimensions of a 4D tensor
    with keepdim=True (dims normalizing to [2, 3]), is lowered to Global
    Average Pooling on an NHWC input. Any other mean.dim is lowered to a
    static mean reduction on the NCHW input, which the runtime computes in
    fp32 for bf16.
    """

    target = "aten.mean.dim"

    def __init__(self, *args) -> None:
        super().__init__(*args)

    def define_node(
        self,
        node: torch.fx.Node,
        xnn_graph: XNNGraph,
        vals_to_ids: Dict[torch.fx.Node, int],
        debug_handle: int,
    ) -> None:
        check_or_raise(
            node.kwargs.get("dtype") is None,
            "XNNPACK does not support mean.dim with dtype",
        )

        global_avg_pool = is_global_avg_pool_mean(node)
        self.define_nodes_tensor_inputs_outputs(
            node, xnn_graph, vals_to_ids, convert_to_nhwc=global_avg_pool
        )
        input_node = get_input_node(node, 0)
        input_id = vals_to_ids[input_node]
        output_id = vals_to_ids[node]

        if global_avg_pool:
            xnode_union = XNNGlobalAvgPooling2d(
                input_id=input_id, output_id=output_id, flags=XNN_FLAG_KEEP_DIMS
            )
        else:
            check_or_raise(
                not input_node.meta.get(
                    ChannelsLastTaggedReshapePass.XNN_NHWC_NODE, False
                ),
                "Internal layout error: ChannelsLastTaggedReshapePass gave an NHWC "
                "input to a mean.dim that reduces over NCHW dims",
            )
            input_rank = len(get_tensor_value(xnn_graph.xvalues[input_id]).dims)
            keepdim = len(node.args) >= 3 and bool(node.args[2])
            xnode_union = XNNStaticMean(
                reduction_axes=sorted(normalize_mean_dims(node.args[1], input_rank)),
                input_id=input_id,
                output_id=output_id,
                flags=XNN_FLAG_KEEP_DIMS if keepdim else 0,
            )

        ser_node = XNode(xnode_union=xnode_union, debug_handle=debug_handle)
        xnn_graph.xnodes.append(ser_node)
