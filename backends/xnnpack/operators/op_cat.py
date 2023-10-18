# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from typing import cast, Dict, List

import torch
from executorch.backends.transforms import get_shape
from executorch.backends.xnnpack.operators.node_visitor import (
    NodeVisitor,
    register_node_visitor,
)
from executorch.backends.xnnpack.operators.quant_params import QuantParams
from executorch.backends.xnnpack.serialization.xnnpack_graph_schema import (
    XNNConcatenate2,
    XNNConcatenate3,
    XNNConcatenate4,
    XNNGraph,
    XNode,
)
from executorch.backends.xnnpack.utils.utils import PERM_NHWC_TO_NCHW
from executorch.backends.xnnpack.utils.xnnpack_constants import XNN_INVALID_VALUE_ID


@register_node_visitor
class CatVisitor(NodeVisitor):
    target = "aten.cat.default"

    def __init__(self, *args) -> None:
        super().__init__(*args)

    def define_node(
        self,
        node: torch.fx.Node,
        xnn_graph: XNNGraph,
        vals_to_ids: Dict[torch.fx.Node, int],
        debug_handle: int,
    ) -> None:
        axis = 0
        list_of_tensors = cast(List[torch.fx.Node], node.args[0])
        num_tensors_to_cat = len(list_of_tensors)

        for tensor_input in list_of_tensors:
            self.define_tensor(
                tensor_input,
                xnn_graph,
                vals_to_ids,
                quant_params=QuantParams.from_inputs(
                    tensor_input, self._exported_program
                ),
            )

        self.define_tensor(
            node, xnn_graph, vals_to_ids, quant_params=QuantParams.from_outputs(node)
        )

        if len(node.args) > 1:
            axis = cast(int, node.args[1])
            if axis < 0 and len(list_of_tensors) > 0:
                axis += len(get_shape(list_of_tensors[0]))

        if "XNN_NHWC_NODE" in node.meta:
            axis = PERM_NHWC_TO_NCHW[axis]

        if num_tensors_to_cat == 2:
            xnode = XNNConcatenate2(
                axis=axis,
                input1_id=vals_to_ids[list_of_tensors[0]],
                input2_id=vals_to_ids[list_of_tensors[1]],
                input3_id=XNN_INVALID_VALUE_ID,
                input4_id=XNN_INVALID_VALUE_ID,
                output_id=vals_to_ids[node],
                flags=0,
            )
        elif num_tensors_to_cat == 3:
            xnode = XNNConcatenate3(
                axis=axis,
                input1_id=vals_to_ids[list_of_tensors[0]],
                input2_id=vals_to_ids[list_of_tensors[1]],
                input3_id=vals_to_ids[list_of_tensors[2]],
                input4_id=XNN_INVALID_VALUE_ID,
                output_id=vals_to_ids[node],
                flags=0,
            )
        elif num_tensors_to_cat == 4:
            xnode = XNNConcatenate4(
                axis=axis,
                input1_id=vals_to_ids[list_of_tensors[0]],
                input2_id=vals_to_ids[list_of_tensors[1]],
                input3_id=vals_to_ids[list_of_tensors[2]],
                input4_id=vals_to_ids[list_of_tensors[3]],
                output_id=vals_to_ids[node],
                flags=0,
            )
        else:
            raise ValueError("XNNPACK Unsupported number of tensors for concatenation")

        ser_node = XNode(
            xnode_union=xnode,
            debug_handle=debug_handle,
        )
        xnn_graph.xnodes.append(ser_node)
