# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from typing import cast, List

import torch
from executorch.backends.xnnpack._passes.xnnpack_pass import XNNPACKPass
from executorch.exir.dialects._ops import ops as exir_ops
from executorch.exir.pass_base import PassResult


class FoldConvPadPass(XNNPACKPass):
    """
    Folds a constant_pad_nd(value=0) feeding a convolution's activation input into
    the convolution's native (asymmetric) XNNPACK input padding. torch.export
    decomposes an even-kernel 'same'-padding conv into
    dequant -> constant_pad_nd -> convolution. Since XNNPACK pads a quantized conv
    input with the input zero_point (== float 0), folding a zero-valued spatial pad
    into the conv padding is numerically exact. Folded padding is stored on the conv
    as node.meta["xnnpack_input_padding"] = [top, right, bottom, left].
    """

    def call(self, graph_module: torch.fx.GraphModule):
        graph = graph_module.graph
        for conv in list(graph.nodes):
            if (
                conv.op != "call_function"
                or conv.target != exir_ops.edge.aten.convolution.default
            ):
                continue

            pad = conv.args[0]
            if (
                not isinstance(pad, torch.fx.Node)
                or pad.op != "call_function"
                or pad.target != exir_ops.edge.aten.constant_pad_nd.default
                or len(pad.users) != 1
            ):
                continue

            pad_value = pad.args[2] if len(pad.args) > 2 else 0
            if pad_value != 0:
                continue

            # constant_pad_nd amounts apply from the last dim backwards in pairs:
            # [W_left, W_right, H_top, H_bottom, C_*, C_*, N_*, N_*]. Only the two
            # spatial dims (H, W) of the NCHW conv input may be padded.
            pad_amounts = cast(List[int], pad.args[1])
            if len(pad_amounts) % 2 != 0 or any(a != 0 for a in pad_amounts[4:]):
                continue

            w_left = pad_amounts[0] if len(pad_amounts) > 0 else 0
            w_right = pad_amounts[1] if len(pad_amounts) > 1 else 0
            h_top = pad_amounts[2] if len(pad_amounts) > 2 else 0
            h_bottom = pad_amounts[3] if len(pad_amounts) > 3 else 0

            conv_padding = cast(List[int], conv.args[4])
            pad_h = conv_padding[0]
            pad_w = conv_padding[1] if len(conv_padding) > 1 else conv_padding[0]

            conv.meta["xnnpack_input_padding"] = [
                pad_h + h_top,
                pad_w + w_right,
                pad_h + h_bottom,
                pad_w + w_left,
            ]

            conv.replace_input_with(pad, pad.args[0])
            graph.erase_node(pad)

        graph_module.recompile()
        return PassResult(graph_module, True)
