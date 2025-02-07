# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from typing import Dict

import torch
from executorch.backends.xnnpack.operators.node_visitor import (
    NodeVisitor,
    register_node_visitor,
)
from executorch.backends.xnnpack.serialization.xnnpack_graph_schema import XNNGraph
from executorch.backends.xnnpack.utils.quant_utils import (
    is_per_channel_group,
    is_per_token,
)
from executorch.backends.xnnpack.utils.utils import (
    check_or_raise,
    get_input_node,
    is_param_node,
)


@register_node_visitor
class OpDynamicDequantizePerTensor(NodeVisitor):
    """
    Dequantize Per Tensor Node visitor
    """

    target = "quantized_decomposed.dequantize_per_tensor.tensor"

    def __init__(self, *args) -> None:
        super().__init__(*args)

    def define_node(
        self,
        node: torch.fx.Node,
        xnn_graph: XNNGraph,
        vals_to_ids: Dict[torch.fx.Node, int],
        debug_handle: int,
    ) -> None:
        """
        We always skip this node because we know it is implicit
        """
        dq_input = get_input_node(node, 0)
        if dq_input in vals_to_ids:
            vals_to_ids[node] = vals_to_ids[dq_input]


@register_node_visitor
class OpDynamicDequantizePerToken(NodeVisitor):
    """
    Dequantize Per Token Node visitor
    """

    target = "quantized_decomposed.dequantize_per_token.default"

    def __init__(self, *args) -> None:
        super().__init__(*args)

    def define_node(
        self,
        node: torch.fx.Node,
        xnn_graph: XNNGraph,
        vals_to_ids: Dict[torch.fx.Node, int],
        debug_handle: int,
    ) -> None:
        """
        We always skip this node because we know it is implicit
        """
        dq_input = get_input_node(node, 0)
        if dq_input in vals_to_ids:
            vals_to_ids[node] = vals_to_ids[dq_input]


@register_node_visitor
class OpDequantizeAffine(NodeVisitor):
    target = "quant.dequantize_affine.default"

    def __init__(self, *args) -> None:
        super().__init__(*args)

    def define_node(
        self,
        node: torch.fx.Node,
        xnn_graph: XNNGraph,
        vals_to_ids: Dict[torch.fx.Node, int],
        debug_handle: int,
    ) -> None:
        """
        We always define dequantize affine nodes because they are always explicit
        """
        if is_per_channel_group(node):
            check_or_raise(
                is_param_node(self._exported_program, node.all_input_nodes[0]),
                f"Expected quantize affine node with per-token semantics to be used "
                f"in front of a weight node, but found node {node.all_input_nodes[0]}",
            )
            # Affine dequantize was recognized as per channel group which means that it should
            # be skipped as this means it is used in front of a weight node
            return

        check_or_raise(
            is_per_token(node),
            "Expecting Affine Dequantized Op to have per-token semantics",
        )
        # This must be a per-token affine dequantized node, so let us serialize as such
        dq_input = get_input_node(node, 0)
        if dq_input in vals_to_ids:
            vals_to_ids[node] = vals_to_ids[dq_input]
