# Copyright (c) Qualcomm Innovation Center, Inc.
# All rights reserved
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.
import os
from operator import attrgetter

import torch

# Also registers torch.ops.qnn_custom.hadamard_transform.
from executorch.backends.qualcomm.builders.custom_ops import _hadamard_matrix
from executorch.backends.qualcomm.utils.check_qnn_version import (
    is_qnn_sdk_version_less_than,
)
from executorch.exir.pass_base import ExportPass, PassResult
from executorch.exir.passes import dead_code_elimination_pass

from .utils import copy_meta


def _is_power_of_2_sqare_matrix(weight: torch.Tensor) -> bool:
    dim = weight.shape[0]
    # Shape gate: non-square / non-2D / non-power-of-2 weight can never match.
    return (
        weight.dim() != 2 or weight.shape[0] != weight.shape[1] or dim & (dim - 1) != 0
    )


def _match_hadamard_weight(weight: torch.Tensor) -> bool:
    # Returns True if `weight == _hadamard_matrix(dim) * s` for some scale s.
    # A linear/matmul with such a weight is equivalent to a QNN HadamardTransform.
    if _is_power_of_2_sqare_matrix(weight):
        return False

    w = weight.detach().to(torch.float64)
    nonzero = w[w != 0]
    if nonzero.numel() == 0:
        return False
    # The Hadamard weight is H * s for a single global scale s; infer s from any
    # nonzero entry (all |H_ij| == 1). For per-channel quant this only matches
    # when every channel's dequantized scale reconstructs the same H * s.
    scale = float(nonzero.flatten()[0].abs())
    hadamard = _hadamard_matrix(w.shape[0], w.device, w.dtype) * scale
    return torch.allclose(w, hadamard, rtol=0, atol=1e-4)


class RecomposeHadamard(ExportPass):
    """
    Rewrite a bias-less linear / matmul / 1x1 conv whose weight is a Hadamard
    matrix into a single qnn_custom.hadamard_transform op, so it is annotated and
    lowered as a first-class HadamardTransform instead of being detected late in
    the builder and validated as FullyConnected / MatMul / Conv.

    Runs in the annotation pipeline (before quantization), where the weight is a
    real tensor and can be inspected. hadamard_transform acts on the last dim, so
    linear / matmul rewrite directly, while conv (which mixes the channel dim) is
    wrapped in permutes that move the channel to the last dim and back.
    """

    def __init__(self):
        super().__init__()
        self.hadamard_target = torch.ops.qnn_custom.hadamard_transform.default

    def _is_pointwise_conv(self, node, weight: torch.Tensor) -> bool:
        # Only a 1x1, stride-1, no-pad, dilation-1, groups-1 conv is a pure
        # channel-mixing matmul equivalent to a Hadamard transform. conv2d args:
        # (input, weight, bias, stride, padding, dilation, groups) with defaults.
        stride = node.args[3] if len(node.args) > 3 else [1, 1]
        padding = node.args[4] if len(node.args) > 4 else [0, 0]
        dilation = node.args[5] if len(node.args) > 5 else [1, 1]
        groups = node.args[6] if len(node.args) > 6 else 1
        return (
            weight.dim() == 4
            and all(k == 1 for k in weight.shape[2:])
            and all(s == 1 for s in stride)
            and all(p == 0 for p in padding)
            and all(d == 1 for d in dilation)
            and groups == 1
        )

    def _get_hadamard_scale(self, weight: torch.Tensor) -> float:
        # weight == H * s (all |H_ij| == 1); linear/matmul(x) = x @ H. The op
        # applies the orthonormal H / sqrt(dim), so fold the remaining factor
        # s * sqrt(dim) into the op's scale (== 1 for an orthonormal Hadamard).
        dim = weight.shape[0]
        return float(weight.detach().abs().flatten()[0]) * (dim**0.5)

    def _rewrite_last_dim(self, graph, node, scale):
        # linear / matmul already transform the last dim: replace in place.
        with graph.inserting_before(node):
            hadamard_node = graph.create_node(
                "call_function",
                self.hadamard_target,
                (node.args[0], scale),
            )
        hadamard_node.meta = copy_meta(node.meta)
        for user in node.users.copy():
            user.replace_input_with(node, hadamard_node)

    def _rewrite_channel_dim(self, graph, node, scale):
        # conv mixes the channel dim (dim 1). Move it to the last dim, run the
        # transform there, then move it back.
        input_node = node.args[0]
        input_val = input_node.meta["val"]
        rank = input_val.dim()
        to_last = [0, *range(2, rank), 1]
        from_last = [0, rank - 1, *range(1, rank - 1)]
        with graph.inserting_before(node):
            pre = graph.create_node(
                "call_function", torch.ops.aten.permute.default, (input_node, to_last)
            )
            pre.meta = copy_meta(node.meta)
            pre.meta["val"] = input_val.permute(to_last)
            hadamard_node = graph.create_node(
                "call_function", self.hadamard_target, (pre, scale)
            )
            hadamard_node.meta = copy_meta(node.meta)
            hadamard_node.meta["val"] = pre.meta["val"]
            post = graph.create_node(
                "call_function",
                torch.ops.aten.permute.default,
                (hadamard_node, from_last),
            )
        post.meta = copy_meta(node.meta)
        for user in node.users.copy():
            user.replace_input_with(node, post)

    def _is_hadamard_transform(self, graph_module, node):
        if node.op != "call_function":
            return False

        is_conv = node.target == torch.ops.aten.conv2d.default
        is_last_dim = node.target in (
            torch.ops.aten.linear.default,
            torch.ops.aten.matmul.default,
        )
        if not (is_conv or is_last_dim):
            return False

        # linear/conv carry an optional bias in args[2]; matmul never does.
        has_bias = len(node.args) >= 3 and node.args[2] is not None
        if has_bias:
            return False

        weight_node = node.args[1]
        if weight_node.op != "get_attr":
            return False
        weight = attrgetter(weight_node.target)(graph_module)
        if is_conv and not self._is_pointwise_conv(node, weight):
            return False
        # A 1x1 conv filter is [out, in, 1, 1]; squeeze to [out, in] to match.
        squeezed = weight.reshape(weight.shape[:2]) if is_conv else weight
        if not _match_hadamard_weight(squeezed):
            return False
        return True

    def call(self, graph_module: torch.fx.GraphModule):
        try:
            if not os.environ.get("QNN_SDK_ROOT") or is_qnn_sdk_version_less_than(
                "2.47"
            ):
                return PassResult(graph_module, False)
        except Exception:
            return PassResult(graph_module, False)

        graph = graph_module.graph
        modified = False
        for node in graph.nodes:
            if not self._is_hadamard_transform(graph_module, node):
                continue
            weight_node = node.args[1]
            weight = attrgetter(weight_node.target)(graph_module)
            is_conv = node.target == torch.ops.aten.conv2d.default
            squeezed = weight.reshape(weight.shape[:2]) if is_conv else weight
            scale = self._get_hadamard_scale(squeezed)
            if is_conv:
                self._rewrite_channel_dim(graph, node, scale)
            else:
                self._rewrite_last_dim(graph, node, scale)
            modified = True

        if modified:
            dead_code_elimination_pass(graph_module)
        return PassResult(graph_module, modified)
