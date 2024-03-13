# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from typing import Dict

import torch
from executorch.backends.transforms import get_shape
from executorch.backends.xnnpack.operators.node_visitor import (
    get_tensor_value,
    NodeVisitor,
    register_node_visitor,
)
from executorch.backends.xnnpack.operators.quant_params import QuantParams

from executorch.backends.xnnpack.serialization.xnnpack_graph_schema import (
    XNNGraph,
    XNNStaticTranspose,
    XNode,
)
from executorch.backends.xnnpack.utils.utils import (
    check_or_raise,
    get_input_node,
    PERM_NCHW_TO_NHWC,
    PERM_NHWC_TO_NCHW,
)


@register_node_visitor
class ConvertMemoryFormat(NodeVisitor):
    target = "aten._to_copy.default"

    def __init__(self, *args) -> None:
        super().__init__(*args)

    def define_node(
        self,
        node: torch.fx.Node,
        xnn_graph: XNNGraph,
        vals_to_ids: Dict[torch.fx.Node, int],
        debug_handle: int,
    ) -> None:
        memory_format_target = node.kwargs.get("memory_format", torch.contiguous_format)
        to_channels_last = bool(memory_format_target == torch.channels_last)
        to_contiguous = bool(memory_format_target == torch.contiguous_format)
        check_or_raise(
            to_channels_last or to_contiguous,
            "Unsupported Memory Format for XNNPACK",
        )

        input_node = get_input_node(node, 0)
        input_quant_params = QuantParams.from_inputs(input_node, self._exported_program)
        output_quant_params = QuantParams.from_outputs(node)

        permute_order = PERM_NCHW_TO_NHWC if to_channels_last else PERM_NHWC_TO_NCHW

        self.define_tensor(
            input_node,
            xnn_graph,
            vals_to_ids,
            quant_params=input_quant_params,
            convert_to_nhwc=(
                (not to_channels_last) and len(get_shape(input_node)) == 4
            ),  # input is contiguous if converting out of channels last
        )

        self.define_tensor(
            node,
            xnn_graph,
            vals_to_ids,
            quant_params=output_quant_params,
            convert_to_nhwc=to_channels_last,  # output is channels last if converting into channels last
        )

        input_id = vals_to_ids[get_input_node(node, 0)]
        output_id = vals_to_ids[node]
        new_shape = get_tensor_value(xnn_graph.xvalues[output_id]).dims

        ser_node = XNode(
            xnode_union=XNNStaticTranspose(
                num_dims=len(new_shape),
                perm=permute_order,
                input_id=input_id,
                output_id=output_id,
                flags=0,
            ),
            debug_handle=debug_handle,
        )
        xnn_graph.xnodes.append(ser_node)
