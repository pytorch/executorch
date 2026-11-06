# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
# Copyright 2026 Arm Limited and/or its affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from typing import cast, Dict

import torch
from executorch.backends.xnnpack.operators.node_visitor import (
    check_or_raise,
    get_tensor_value,
    NodeVisitor,
    register_node_visitor,
)
from executorch.backends.xnnpack.serialization.xnnpack_graph_schema import (
    XNNConvert,
    XNNDatatype,
    XNNGlobalAvgPooling2d,
    XNNGraph,
    XNNStaticMean,
    XNNTensorValue,
    XNode,
    XValue,
)
from executorch.backends.xnnpack.utils.utils import (
    get_input_node,
    is_global_avg_pool_mean,
    normalize_mean_dims,
)
from executorch.backends.xnnpack.utils.xnnpack_constants import (
    XNN_FLAG_KEEP_DIMS,
    XNN_INVALID_VALUE_ID,
)


@register_node_visitor
class MeanDim(NodeVisitor):
    """
    An fp16 or fp32 mean.dim over the innermost dimensions of a 4D tensor
    with keepdim=True (dims normalizing to [2, 3]) is lowered to Global
    Average Pooling on an NHWC input. Any other mean.dim is lowered to a
    static mean reduction on the NCHW input, computed in fp32 for a bf16
    input.
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
        input_id = vals_to_ids[cast(torch.fx.Node, node.args[0])]
        output_id = vals_to_ids[node]

        if global_avg_pool:
            xnode_unions = [
                XNNGlobalAvgPooling2d(
                    input_id=input_id, output_id=output_id, flags=XNN_FLAG_KEEP_DIMS
                )
            ]
        else:
            input_val = get_input_node(node, 0).meta["val"]
            keepdim = len(node.args) >= 3 and bool(node.args[2])
            mean = XNNStaticMean(
                reduction_axes=sorted(
                    normalize_mean_dims(node.args[1], input_val.dim())
                ),
                input_id=input_id,
                output_id=output_id,
                flags=XNN_FLAG_KEEP_DIMS if keepdim else 0,
            )
            xnode_unions = [mean]
            if input_val.dtype == torch.bfloat16:
                mean.input_id = self.define_fp32_like(xnn_graph, input_id)
                mean.output_id = self.define_fp32_like(xnn_graph, output_id)
                xnode_unions = [
                    XNNConvert(input_id=input_id, output_id=mean.input_id, flags=0),
                    mean,
                    XNNConvert(input_id=mean.output_id, output_id=output_id, flags=0),
                ]

        for xnode_union in xnode_unions:
            xnn_graph.xnodes.append(
                XNode(xnode_union=xnode_union, debug_handle=debug_handle)
            )

    @staticmethod
    def define_fp32_like(xnn_graph: XNNGraph, tensor_id: int) -> int:
        """Defines an fp32 intermediate with the dims of the given tensor."""
        dims = list(get_tensor_value(xnn_graph.xvalues[tensor_id]).dims)
        fp32_id = len(xnn_graph.xvalues)
        xnn_graph.xvalues.append(
            XValue(
                xvalue_union=XNNTensorValue(
                    datatype=XNNDatatype.xnn_datatype_fp32,
                    num_dims=len(dims),
                    dims=dims,
                    external_id=XNN_INVALID_VALUE_ID,
                    constant_buffer_idx=0,
                    flags=0,
                    id_out=fp32_id,
                )
            )
        )
        return fp32_id
