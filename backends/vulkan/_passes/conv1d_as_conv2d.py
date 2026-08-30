# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

from typing import List, Optional

import torch

from executorch.backends.transforms.utils import (
    get_param_tensor,
    is_param_node,
    set_param_tensor,
)
from executorch.exir.dialects._ops import ops as exir_ops
from executorch.exir.pass_base import ExportPass, PassResult

from torch.export import ExportedProgram

# Mirrors kIm2colMinCOut in Convolution.cpp. At or above this output channel
# count, should_use_conv2d_im2col() selects the im2col + GEMM path on every
# vendor, so a rewritten conv is guaranteed to land there rather than on the
# direct conv2d shader, which has not been compared against conv1d.
_IM2COL_MIN_C_OUT = 128


def _reshape_fake(val: torch.Tensor, sizes: List[int]) -> torch.Tensor:
    fake_mode = getattr(val, "fake_mode", None)
    if fake_mode is None:
        return val.reshape(sizes)
    with fake_mode:
        return val.reshape(sizes)


class Conv1dAsConv2dPass(ExportPass):
    """
    Rewrite eligible 1-D convolutions into a 2-D convolution over a singleton
    height dimension.

    conv1d.glsl computes one output element per invocation with no tiling and no
    reuse between invocations, and its global work grid packs texels along the
    batch dim, wasting three of every four lanes at batch 1. It measures around
    30x slower than the conv2d im2col + GEMM path for the same MAC count.
    Expressing the convolution as 2-D lets the existing conv2d machinery
    (im2col selection, memory layout tagging, weight prepacking) handle it with
    no runtime changes.

    Only convolutions that are certain to reach the im2col path are rewritten,
    so nothing silently moves onto a shader that has not been compared against
    conv1d. The weight is reshaped in place rather than through a view node
    because the conv2d implementation reads it as a constant TensorRef.
    """

    def __init__(self) -> None:
        super().__init__()
        self._exported_program: Optional[ExportedProgram] = None

    def _eligible(self, node: torch.fx.Node) -> bool:
        assert self._exported_program is not None
        in_node, weight_node = node.args[0], node.args[1]
        if not isinstance(in_node, torch.fx.Node):
            return False
        if not isinstance(weight_node, torch.fx.Node):
            return False
        in_shape = in_node.meta["val"].shape
        w_shape = weight_node.meta["val"].shape
        if len(in_shape) != 3 or len(w_shape) != 3:
            return False
        # The weight has to be a constant so it can be reshaped in place.
        if not is_param_node(self._exported_program, weight_node):
            return False
        transposed, groups = node.args[6], node.args[8]
        if transposed or groups != 1:
            return False
        # conv2d does not support batched input at all, so a rewrite there
        # would turn a working conv1d into a throw.
        if in_shape[0] != 1:
            return False
        # Depthwise and pointwise 1-D convs keep their own shaders.
        if w_shape[0] < _IM2COL_MIN_C_OUT or w_shape[2] == 1:
            return False
        # im2col requires unit dilation.
        return list(node.args[5]) == [1]

    def _rewrite(self, graph: torch.fx.Graph, node: torch.fx.Node) -> None:
        assert self._exported_program is not None
        view = exir_ops.edge.aten.view_copy.default
        in_node, weight_node = node.args[0], node.args[1]

        in_shape = list(in_node.meta["val"].shape)
        w_shape = list(weight_node.meta["val"].shape)
        out_shape = list(node.meta["val"].shape)
        in_4d_shape = [in_shape[0], in_shape[1], 1, in_shape[2]]
        w_4d_shape = [w_shape[0], w_shape[1], 1, w_shape[2]]
        out_4d_shape = [out_shape[0], out_shape[1], 1, out_shape[2]]

        # The weight tensor is contiguous, so inserting a singleton dim is a
        # pure metadata change. A weight shared by several convs is only
        # reshaped once; the rank check in _eligible() skips it afterwards.
        weight = get_param_tensor(self._exported_program, weight_node)
        assert weight is not None
        set_param_tensor(
            self._exported_program, weight_node, weight.reshape(w_4d_shape)
        )
        weight_node.meta["val"] = _reshape_fake(
            weight_node.meta["val"], w_4d_shape
        )

        with graph.inserting_before(node):
            in_4d = graph.call_function(view, args=(in_node, in_4d_shape))
        in_4d.meta = dict(in_node.meta)
        in_4d.meta["val"] = _reshape_fake(in_node.meta["val"], in_4d_shape)

        def pad(arg: List[int], lead: int) -> List[int]:
            return [lead, arg[0]]

        node.args = (
            in_4d,
            weight_node,
            node.args[2],
            pad(node.args[3], 1),  # stride
            pad(node.args[4], 0),  # padding
            pad(node.args[5], 1),  # dilation
            node.args[6],
            pad(node.args[7], 0),  # output_padding
            node.args[8],
        )

        with graph.inserting_after(node):
            out_3d = graph.call_function(view, args=(node, out_shape))
        out_3d.meta = dict(node.meta)
        node.replace_all_uses_with(out_3d)
        out_3d.args = (node, out_shape)
        node.meta["val"] = _reshape_fake(node.meta["val"], out_4d_shape)

    def call(self, graph_module: torch.fx.GraphModule) -> PassResult:
        assert self._exported_program is not None

        modified = False
        for node in list(graph_module.graph.nodes):
            if node.op != "call_function":
                continue
            if node.target != exir_ops.edge.aten.convolution.default:
                continue
            if not self._eligible(node):
                continue
            self._rewrite(graph_module.graph, node)
            modified = True

        if not modified:
            return PassResult(graph_module, False)

        graph_module.recompile()
        return PassResult(graph_module, True)
