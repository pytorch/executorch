# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
# Copyright 2026 Arm Limited and/or its affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from enum import auto, Enum
from typing import Dict, List

import torch
from executorch.backends.xnnpack._passes.channels_last_tagged_reshape_pass import (
    ChannelsLastTaggedReshapePass,
)
from executorch.backends.xnnpack.operators.node_visitor import (
    get_tensor_value,
    NodeVisitor,
    register_node_visitor,
)
from executorch.backends.xnnpack.operators.quant_params import QuantParams

from executorch.backends.xnnpack.serialization.xnnpack_graph_schema import (
    XNNConvert,
    XNNCopy,
    XNNDatatype,
    XNNGraph,
    XNNStaticTranspose,
    XNNTensorValue,
    XNode,
    XValue,
)
from executorch.backends.xnnpack.utils.utils import (
    get_input_node,
    PERM_NCHW_TO_NHWC,
    PERM_NHWC_TO_NCHW,
)
from executorch.backends.xnnpack.utils.xnnpack_constants import XNN_INVALID_VALUE_ID


class ToCopyOperation(Enum):
    TRANSPOSE = auto()
    CAST = auto()
    COPY = auto()


def sort_decomposed_operations(
    decomposed_operations: List[ToCopyOperation],
    input_dtype: torch.dtype,
    output_dtype: torch.dtype,
) -> None:
    """Sort transpose+cast so transpose runs on the smaller dtype."""
    if (
        ToCopyOperation.TRANSPOSE not in decomposed_operations
        or ToCopyOperation.CAST not in decomposed_operations
    ):
        return

    input_element_size = torch.empty((), dtype=input_dtype).element_size()
    output_element_size = torch.empty((), dtype=output_dtype).element_size()
    if input_element_size == output_element_size:
        return

    first_operation = (
        ToCopyOperation.TRANSPOSE
        if output_element_size > input_element_size
        else ToCopyOperation.CAST
    )
    decomposed_operations.sort(key=lambda operation: operation != first_operation)


@register_node_visitor
class ToCopy(NodeVisitor):
    # _to_copy lowers dtype changes to XNNConvert, 4D contiguous/channels_last
    # memory-format changes to XNNStaticTranspose, and no-op copies to XNNCopy.
    target = "aten._to_copy.default"

    def _resolve_input_output_layouts(
        self, input_node: torch.fx.Node, node: torch.fx.Node
    ) -> tuple[bool, bool]:
        # ChannelsLastTaggedReshapePass is the source of truth in the normal
        # XNNPACK pipeline. Fall back to memory_format only for untagged nodes,
        # such as direct visitor tests or use before that pass has run.
        if ChannelsLastTaggedReshapePass.XNN_NHWC_NODE in node.meta:
            return (
                ChannelsLastTaggedReshapePass.is_nhwc_node(input_node),
                ChannelsLastTaggedReshapePass.is_nhwc_node(node),
            )

        input_val = input_node.meta["val"]
        memory_format = node.kwargs.get("memory_format", torch.preserve_format)

        if memory_format == torch.channels_last:
            return False, True
        elif memory_format == torch.contiguous_format:
            return input_val.dim() == 4, False
        elif memory_format in (None, torch.preserve_format):
            input_is_channels_last = (
                input_val.dim() == 4
                and list(input_val.dim_order()) == PERM_NCHW_TO_NHWC
            )
            return input_is_channels_last, input_is_channels_last

        return False, False

    def _define_intermediate_tensor(
        self,
        xnn_graph: XNNGraph,
        dims: List[int],
        dtype: XNNDatatype,
    ) -> int:
        value_id = len(xnn_graph.xvalues)
        xnn_graph.xvalues.append(
            XValue(
                xvalue_union=XNNTensorValue(
                    datatype=dtype,
                    num_dims=len(dims),
                    dims=dims,
                    constant_buffer_idx=0,
                    external_id=XNN_INVALID_VALUE_ID,
                    flags=0,
                    id_out=value_id,
                )
            )
        )
        return value_id

    @staticmethod
    def _append_copy_node(
        xnn_graph: XNNGraph,
        input_id: int,
        output_id: int,
        debug_handle: int,
    ) -> None:
        xnn_graph.xnodes.append(
            XNode(
                xnode_union=XNNCopy(
                    input_id=input_id,
                    output_id=output_id,
                    flags=0,
                ),
                debug_handle=debug_handle,
            )
        )

    @staticmethod
    def _append_convert_node(
        xnn_graph: XNNGraph,
        input_id: int,
        output_id: int,
        debug_handle: int,
    ) -> None:
        xnn_graph.xnodes.append(
            XNode(
                xnode_union=XNNConvert(
                    input_id=input_id,
                    output_id=output_id,
                    flags=0,
                ),
                debug_handle=debug_handle,
            )
        )

    def _append_output_tensor(
        self,
        node: torch.fx.Node,
        xnn_graph: XNNGraph,
        vals_to_ids: Dict[torch.fx.Node, int],
        operation: ToCopyOperation,
        current_dims: List[int],
        input_dtype: XNNDatatype,
        output_dtype: XNNDatatype,
        output_is_channels_last: bool,
        output_quant_params: QuantParams | None,
        convert_to_nhwc: bool,
        last_operation: bool,
    ) -> tuple[int, List[int], XNNDatatype, List[int] | None]:
        """Compute properties of output tensor and add it to the graph."""
        output_dims = current_dims
        output_dtype_for_operation = input_dtype
        permute_order = None

        if operation == ToCopyOperation.TRANSPOSE:
            permute_order = (
                PERM_NCHW_TO_NHWC if output_is_channels_last else PERM_NHWC_TO_NCHW
            )
            output_dims = [current_dims[i] for i in permute_order]
        elif operation == ToCopyOperation.CAST:
            output_dtype_for_operation = output_dtype

        if last_operation:
            self.define_tensor(
                node,
                xnn_graph,
                vals_to_ids,
                quant_params=output_quant_params,
                convert_to_nhwc=convert_to_nhwc,
            )
            return (
                vals_to_ids[node],
                output_dims,
                output_dtype_for_operation,
                permute_order,
            )

        return (
            self._define_intermediate_tensor(
                xnn_graph, output_dims, output_dtype_for_operation
            ),
            output_dims,
            output_dtype_for_operation,
            permute_order,
        )

    @staticmethod
    def _append_transpose_node(
        xnn_graph: XNNGraph,
        permute_order: List[int],
        input_id: int,
        output_id: int,
        debug_handle: int,
    ) -> None:
        output_shape = get_tensor_value(xnn_graph.xvalues[output_id]).dims
        xnn_graph.xnodes.append(
            XNode(
                xnode_union=XNNStaticTranspose(
                    num_dims=len(output_shape),
                    perm=permute_order,
                    input_id=input_id,
                    output_id=output_id,
                    flags=0,
                ),
                debug_handle=debug_handle,
            )
        )

    def define_node(
        self,
        node: torch.fx.Node,
        xnn_graph: XNNGraph,
        vals_to_ids: Dict[torch.fx.Node, int],
        debug_handle: int,
    ) -> None:
        input_node = get_input_node(node, 0)
        input_val = input_node.meta["val"]
        output_val = node.meta["val"]
        input_is_channels_last, output_is_channels_last = (
            self._resolve_input_output_layouts(input_node, node)
        )
        changes_dtype = input_val.dtype != output_val.dtype
        changes_dim_order = input_is_channels_last != output_is_channels_last

        input_quant_params = QuantParams.from_inputs(input_node, self._exported_program)
        output_quant_params = QuantParams.from_outputs(node)

        self.define_tensor(
            input_node,
            xnn_graph,
            vals_to_ids,
            quant_params=input_quant_params,
            convert_to_nhwc=input_is_channels_last,
        )

        decomposed_operations = []
        if changes_dim_order:
            decomposed_operations.append(ToCopyOperation.TRANSPOSE)
        if changes_dtype:
            decomposed_operations.append(ToCopyOperation.CAST)
        if len(decomposed_operations) == 0:
            decomposed_operations.append(ToCopyOperation.COPY)
        else:
            sort_decomposed_operations(
                decomposed_operations, input_val.dtype, output_val.dtype
            )

        input_id = vals_to_ids[input_node]
        input_dtype = self.get_serialized_dtype(input_quant_params, input_node)
        output_dtype = self.get_serialized_dtype(output_quant_params, node)
        current_dims = get_tensor_value(xnn_graph.xvalues[input_id]).dims

        for operation in decomposed_operations:
            last_operation = operation == decomposed_operations[-1]
            (
                output_id,
                output_dims,
                output_dtype_for_operation,
                permute_order,
            ) = self._append_output_tensor(
                node,
                xnn_graph,
                vals_to_ids,
                operation,
                current_dims,
                input_dtype,
                output_dtype,
                output_is_channels_last,
                output_quant_params,
                output_is_channels_last,
                last_operation,
            )
            match operation:
                case ToCopyOperation.COPY:
                    self._append_copy_node(xnn_graph, input_id, output_id, debug_handle)
                case ToCopyOperation.CAST:
                    self._append_convert_node(
                        xnn_graph, input_id, output_id, debug_handle
                    )
                case ToCopyOperation.TRANSPOSE:
                    assert permute_order is not None
                    self._append_transpose_node(
                        xnn_graph,
                        permute_order,
                        input_id,
                        output_id,
                        debug_handle,
                    )
            input_id = output_id
            current_dims = output_dims
            input_dtype = output_dtype_for_operation
