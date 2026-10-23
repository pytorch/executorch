# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from typing import cast, List, Tuple

import torch
from executorch.backends.xnnpack._passes.xnnpack_pass import XNNPACKPass
from executorch.backends.xnnpack.utils.quant_utils import is_dequant, is_quant
from executorch.exir.dialects._ops import ops as exir_ops
from executorch.exir.pass_base import PassResult


class FusePadIntoConvPass(XNNPACKPass):
    """
    Folds a zero-valued constant_pad_nd on the two spatial dims into the
    convolution that consumes it. XNNPACK convolutions take independent
    top / right / bottom / left padding, so asymmetric pads such as
    F.pad(x, (2, 1, 2, 1)) before a 4x4 filter fold as well, and the padded
    copy of the activation (a full extra pass over the tensor, noticeably slow
    in NHWC) disappears.

    The graph is left intact so shape metadata stays valid: the pad node is
    tagged FOLDED_PAD_TAG and its visitor aliases the pad output to the pad
    input instead of emitting a node, and the convolution is tagged with the
    combined (top, right, bottom, left) padding that its visitor serializes.

    Must run after ChannelsLastTaggedReshapePass: if that pass had to insert
    a layout conversion between the pad and the convolution, the pad's only
    user is no longer the convolution and nothing is fused.
    """

    FOLDED_PAD_TAG = "XNN_FOLDED_PAD"
    FUSED_PADDING_TAG = "XNN_CONV_PADDING_TRBL"

    @staticmethod
    def get_fused_padding(node: torch.fx.Node):
        return node.meta.get(FusePadIntoConvPass.FUSED_PADDING_TAG, None)

    @staticmethod
    def is_folded_pad(node: torch.fx.Node) -> bool:
        return node.meta.get(FusePadIntoConvPass.FOLDED_PAD_TAG, False)

    def _foldable(self, pad: torch.fx.Node) -> Tuple[bool, List[int]]:
        """Returns (ok, [left, right, top, bottom]) for a 4D spatial zero pad."""
        if len(pad.users) != 1:
            return False, [0, 0, 0, 0]
        if len(pad.args) > 2 and cast(float, pad.args[2]) != 0.0:
            return False, [0, 0, 0, 0]
        src = pad.args[0]
        if not isinstance(src, torch.fx.Node) or "val" not in src.meta:
            return False, [0, 0, 0, 0]
        if src.meta["val"].dim() != 4:
            return False, [0, 0, 0, 0]
        amounts = list(cast(List[int], pad.args[1]))
        if len(amounts) % 2 or len(amounts) > 4 or any(a < 0 for a in amounts):
            return False, [0, 0, 0, 0]
        # Quantized graphs route the pad through q/dq pairs (InsertPadQDQPass);
        # leave those alone.
        if is_dequant(src) or any(is_quant(u) for u in pad.users):
            return False, [0, 0, 0, 0]
        amounts = amounts + [0] * (4 - len(amounts))  # [left, right, top, bottom]
        return True, amounts

    def call(self, graph_module: torch.fx.GraphModule):
        modified = False
        for pad in graph_module.graph.nodes:
            if (
                pad.op != "call_function"
                or pad.target != exir_ops.edge.aten.constant_pad_nd.default
            ):
                continue
            ok, (left, right, top, bottom) = self._foldable(pad)
            if not ok:
                continue
            conv = next(iter(pad.users))
            if (
                conv.op != "call_function"
                or conv.target != exir_ops.edge.aten.convolution.default
                or conv.args[0] is not pad
                or conv.args[6]  # transposed convolution pads differently
                or self.get_fused_padding(conv) is not None
            ):
                continue
            conv_padding = list(cast(List[int], conv.args[4]))
            if len(conv_padding) == 1:
                conv_padding = conv_padding + conv_padding
            if len(conv_padding) != 2:
                continue
            conv.meta[self.FUSED_PADDING_TAG] = (
                conv_padding[0] + top,
                conv_padding[1] + right,
                conv_padding[0] + bottom,
                conv_padding[1] + left,
            )
            pad.meta[self.FOLDED_PAD_TAG] = True
            modified = True
        # Tags only, no graph change: the metadata stays as traced.
        return PassResult(graph_module, modified)
