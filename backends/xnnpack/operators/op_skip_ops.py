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


class OpSkipOps(NodeVisitor):
    """
    Parent Class for handling Skip Ops
    """

    def __init__(self, *args) -> None:
        super().__init__(*args)

    def define_node(
        self,
        node: torch.fx.Node,
        xnn_graph: XNNGraph,
        vals_to_ids: Dict[torch.fx.Node, int],
        debug_handle: int,
    ) -> None:
        return


@register_node_visitor
class OpChooseQparamsTensor(OpSkipOps):
    """
    do nothing if node is choose_qparams.tensor
    """

    target = "quantized_decomposed.choose_qparams.tensor"


@register_node_visitor
class OpDequantizePerChannelDefault(OpSkipOps):
    """
    do nothing if node is dequantize_per_channel.default
    """

    target = "quantized_decomposed.dequantize_per_channel.default"


@register_node_visitor
class OpGetItem(OpSkipOps):
    """
    do nothing if node is getitem
    """

    target = "getitem"


@register_node_visitor
class OpQuantizePerChannelDefault(OpSkipOps):
    """
    do nothing if node is quantize_per_channel.default
    """

    target = "quantized_decomposed.quantize_per_channel.default"


@register_node_visitor
class OpTCopyDefault(OpSkipOps):
    """
    do nothing if node is t_copy.default
    """

    target = "aten.t_copy.default"


@register_node_visitor
class OpViewCopyDefault(OpSkipOps):
    """
    currently, do nothing if node is view_copy.default
    need to handle this later on, currently view it as one of skip ops
    """

    target = "aten.view_copy.default"


@register_node_visitor
class OpSymSizeInt(OpSkipOps):
    """
    currently, do nothing if node is sym_size.int
    need to handle this later on, currently view it as one of skip ops
    """

    target = "sym_size.int"


@register_node_visitor
class OpChooseQparamsAffine(OpSkipOps):
    """
    do nothing if node is choose_qparams_affine.default
    """

    target = "quant.choose_qparams_affine.default"


@register_node_visitor
class OpChooseQparamsToken(OpSkipOps):
    """
    do nothing if node is choose_qparams_per_token_asymmetric.tensor
    """

    target = "quantized_decomposed.choose_qparams_per_token_asymmetric.default"


@register_node_visitor
class OpQuantizePerChannelGroupDefault(OpSkipOps):
    """
    do nothing if node is quantize_per_channel_group.default
    """

    target = "quantized_decomposed.quantize_per_channel_group.default"


@register_node_visitor
class OpDequantizePerChannelGroupDefault(OpSkipOps):
    """
    do nothing if node is dequantize_per_channel_group.default
    """

    target = "quantized_decomposed.dequantize_per_channel_group.default"
