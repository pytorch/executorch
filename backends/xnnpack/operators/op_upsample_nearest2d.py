# Copyright (c) 2026 Software Mansion
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from typing import Dict, List, Optional

import torch
from executorch.backends.transforms import get_shape
from executorch.backends.xnnpack.operators.node_visitor import (
    NodeVisitor,
    register_node_visitor,
)
from executorch.backends.xnnpack.operators.quant_params import QuantParams
from executorch.backends.xnnpack.serialization.xnnpack_graph_schema import (
    XNNConcatenate4,
    XNNConvert,
    XNNDepthToSpace2d,
    XNNGraph,
    XNNQuantizedTensorValue,
    XNNTensorValue,
    XNode,
    XValue,
)
from executorch.backends.xnnpack.utils.utils import check_or_raise, get_input_node
from executorch.backends.xnnpack.utils.xnnpack_constants import XNN_INVALID_VALUE_ID


@register_node_visitor
class UpsampleNearest2dVisitor(NodeVisitor):
    """
    XNNPACK has no nearest resize, so x2 upsampling is lowered as four copies
    of the input concatenated along channels, [N, H, W, 4C], followed by
    depth_to_space_2d(block_size=2). Neither op changes values, so the
    intermediates carry the input's quantization parameters, and an output
    quantized differently is requantized with a convert.
    """

    target = "aten.upsample_nearest2d.vec"

    def __init__(self, *args) -> None:
        super().__init__(*args)

    def define_node(
        self,
        node: torch.fx.Node,
        xnn_graph: XNNGraph,
        vals_to_ids: Dict[torch.fx.Node, int],
        debug_handle: int,
    ) -> None:
        input_node = get_input_node(node, 0)
        input_quant_params = QuantParams.from_inputs(input_node, self._exported_program)
        output_quant_params = QuantParams.from_outputs(node)
        self.define_tensor(
            input_node,
            xnn_graph,
            vals_to_ids,
            convert_to_nhwc=True,
            quant_params=input_quant_params,
        )
        self.define_tensor(
            node,
            xnn_graph,
            vals_to_ids,
            convert_to_nhwc=True,
            quant_params=output_quant_params,
        )
        input_id = vals_to_ids[input_node]
        output_id = vals_to_ids[node]

        in_shape = get_shape(input_node)
        out_shape = get_shape(node)
        check_or_raise(len(in_shape) == 4, "upsample_nearest2d expects a 4D input")
        n, c, h, w = in_shape
        check_or_raise(
            out_shape[-2] == 2 * h and out_shape[-1] == 2 * w,
            f"only x2 nearest upsampling is supported, got {in_shape} -> {out_shape}",
        )

        concat_id = self.define_intermediate(
            xnn_graph, node, [n, h, w, 4 * c], input_quant_params
        )
        if self.quant_params_match(input_quant_params, output_quant_params):
            d2s_output_id = output_id
        else:
            d2s_output_id = self.define_intermediate(
                xnn_graph, node, [n, 2 * h, 2 * w, c], input_quant_params
            )

        xnn_graph.xnodes.append(
            XNode(
                xnode_union=XNNConcatenate4(
                    axis=3,
                    input1_id=input_id,
                    input2_id=input_id,
                    input3_id=input_id,
                    input4_id=input_id,
                    output_id=concat_id,
                    flags=0,
                    input5_id=XNN_INVALID_VALUE_ID,
                ),
                debug_handle=debug_handle,
            )
        )
        xnn_graph.xnodes.append(
            XNode(
                xnode_union=XNNDepthToSpace2d(
                    block_size=2,
                    input_id=concat_id,
                    output_id=d2s_output_id,
                    flags=0,
                ),
                debug_handle=debug_handle,
            )
        )
        if d2s_output_id != output_id:
            xnn_graph.xnodes.append(
                XNode(
                    xnode_union=XNNConvert(
                        input_id=d2s_output_id, output_id=output_id, flags=0
                    ),
                    debug_handle=debug_handle,
                )
            )

    def define_intermediate(
        self,
        xnn_graph: XNNGraph,
        node: torch.fx.Node,
        dims: List[int],
        quant_params: Optional[QuantParams],
    ) -> int:
        tensor_id = len(xnn_graph.xvalues)
        tensor_value = XNNTensorValue(
            datatype=self.get_serialized_dtype(quant_params, node),
            num_dims=len(dims),
            dims=dims,
            external_id=XNN_INVALID_VALUE_ID,
            constant_buffer_idx=0,
            flags=0,
            id_out=tensor_id,
        )
        xnn_graph.xvalues.append(
            XValue(xvalue_union=tensor_value)
            if quant_params is None
            else XValue(
                xvalue_union=XNNQuantizedTensorValue(
                    tensor_value=tensor_value,
                    quant_params=self.get_quant_params(quant_params, xnn_graph),
                )
            )
        )
        return tensor_id

    @staticmethod
    def quant_params_match(a: Optional[QuantParams], b: Optional[QuantParams]) -> bool:
        if a is None or b is None:
            return a is b
        check_or_raise(
            not a.per_channel and not b.per_channel,
            "upsample_nearest2d expects per-tensor quantized activations",
        )
        return a.dtype == b.dtype and a.scale == b.scale and a.zp == b.zp
