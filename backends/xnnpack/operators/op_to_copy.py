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
    XNNConvert,
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
        input_node = get_input_node(node, 0)
        input_dtype = input_node.meta["val"].dtype
        output_dtype = node.meta["val"].dtype

        if input_dtype != output_dtype:
            # Mixed dtype + layout is not supported by xnn_define_convert;
            # such nodes are rejected by the partitioner and should never be
            # serialized as a convert.
            check_or_raise(
                list(input_node.meta["val"].dim_order())
                == list(node.meta["val"].dim_order()),
                "Combined dtype and layout _to_copy is not supported for XNNPACK convert",
            )
            self._define_dtype_convert(
                node, input_node, xnn_graph, vals_to_ids, debug_handle
            )
            return

        memory_format_target = node.kwargs.get("memory_format", torch.contiguous_format)
        to_channels_last = bool(memory_format_target == torch.channels_last)
        to_contiguous = bool(memory_format_target == torch.contiguous_format)
        check_or_raise(
            to_channels_last or to_contiguous,
            "Unsupported Memory Format for XNNPACK",
        )

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

        input_id = vals_to_ids[input_node]
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

    def _define_dtype_convert(
        self,
        node: torch.fx.Node,
        input_node: torch.fx.Node,
        xnn_graph: XNNGraph,
        vals_to_ids: Dict[torch.fx.Node, int],
        debug_handle: int,
    ) -> None:
        # Input and output tensors keep their own (differing) dtypes; the
        # convert node bridges them via xnn_define_convert at runtime.
        # xnn_define_convert does not support quantized tensors; ensure both
        # sides are unquantized (null QuantParams) before emitting the node.
        input_quant = QuantParams.from_inputs(input_node, self._exported_program)
        output_quant = QuantParams.from_outputs(node)
        check_or_raise(
            input_quant is None and output_quant is None,
            "xnn_define_convert only supports non-quantized dtype conversion",
        )
        self.define_tensor(
            input_node,
            xnn_graph,
            vals_to_ids,
            quant_params=input_quant,
        )
        self.define_tensor(
            node,
            xnn_graph,
            vals_to_ids,
            quant_params=output_quant,
        )

        ser_node = XNode(
            xnode_union=XNNConvert(
                input_id=vals_to_ids[input_node],
                output_id=vals_to_ids[node],
                flags=0,
            ),
            debug_handle=debug_handle,
        )
        xnn_graph.xnodes.append(ser_node)
