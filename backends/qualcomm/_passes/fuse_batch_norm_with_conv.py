# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import operator
from typing import Optional

import torch
from executorch.backends.transforms.utils import (
    create_constant_placeholder,
    delete_constant_placeholder,
    get_param_tensor,
    is_param_node,
)
from executorch.exir.pass_base import ExportPass, PassResult
from torch.export.graph_signature import InputKind
from torch.nn.utils.fusion import fuse_conv_bn_weights

# The export pipeline sees pre-dispatch ops; a program exported otherwise
# carries the core ATen forms.
_CONV_OPS = {
    torch.ops.aten.conv1d.default,
    torch.ops.aten.conv2d.default,
    torch.ops.aten.conv3d.default,
    torch.ops.aten.convolution.default,
}
_BATCH_NORM = torch.ops.aten.batch_norm.default
_BATCH_NORM_NO_TRAINING = torch.ops.aten._native_batch_norm_legit_no_training.default


class FuseBatchNormWithConv(ExportPass):
    """
    Fold a BatchNorm that follows a convolution into the convolution's weight
    and bias, in fp32, before lowering. Runs in the export pipeline, on the aten
    program, so the fused parameters are registered on the program it rewrites.

    Left in place, the BatchNorm becomes a standalone QNN op. On HTP in fp16 its
    per-channel scale, which reaches ~100 after depthwise convolutions, multiplies
    the fp16 rounding error of the convolution output. Quantized graphs are
    unaffected: PTQ already folds these pairs before this pass runs.
    """

    def __init__(self, edge_program: torch.export.ExportedProgram):
        super().__init__()
        self.edge_program = edge_program

    def _is_param(self, node) -> bool:
        return isinstance(node, torch.fx.Node) and is_param_node(
            self.edge_program, node
        )

    def _tensor(self, node) -> Optional[torch.Tensor]:
        return None if node is None else get_param_tensor(self.edge_program, node)

    def _match_bn(self, conv: torch.fx.Node) -> Optional[torch.fx.Node]:
        """The inference-mode BatchNorm consuming `conv`, if it can be folded."""
        if conv.target not in _CONV_OPS or len(conv.users) != 1:
            return None
        # fuse_conv_bn_weights assumes a regular (non-transposed) layout
        if conv.target == torch.ops.aten.convolution.default and conv.args[6]:
            return None
        bn = next(iter(conv.users))
        if bn.target == _BATCH_NORM:
            # batch_norm(input, weight, bias, mean, var, training, momentum, eps, ...)
            matched = not bn.args[5]
        elif bn.target == _BATCH_NORM_NO_TRAINING:
            # Only the normalized output (getitem 0) may be used.
            matched = all(
                user.target == operator.getitem and user.args[1] == 0
                for user in bn.users
            )
        else:
            matched = False
        if not matched:
            return None

        bias = conv.args[2] if len(conv.args) > 2 else None
        bn_weight, bn_bias, running_mean, running_var = bn.args[1:5]
        required = (conv.args[1], running_mean, running_var)
        optional = (bias, bn_weight, bn_bias)
        if all(self._is_param(n) for n in required) and all(
            n is None or self._is_param(n) for n in optional
        ):
            return bn
        return None

    def _fold(self, graph: torch.fx.Graph, conv: torch.fx.Node, bn: torch.fx.Node):
        weight = conv.args[1]
        bias = conv.args[2] if len(conv.args) > 2 else None
        bn_weight, bn_bias, running_mean, running_var = bn.args[1:5]
        eps = bn.args[7] if bn.target == _BATCH_NORM else bn.args[-1]
        fused_weight, fused_bias = fuse_conv_bn_weights(
            self._tensor(weight),
            self._tensor(bias),
            self._tensor(running_mean),
            self._tensor(running_var),
            eps,
            self._tensor(bn_weight),
            self._tensor(bn_bias),
        )

        first_placeholder = next(n for n in graph.nodes if n.op == "placeholder")
        with graph.inserting_before(first_placeholder):
            fused = [
                create_constant_placeholder(
                    self.edge_program,
                    graph,
                    f"{conv.name}_fused_bn_{kind}",
                    InputKind.PARAMETER,
                    tensor.detach(),
                )
                for kind, tensor in (("weight", fused_weight), ("bias", fused_bias))
            ]
        args = list(conv.args) + [None] * (3 - len(conv.args))
        args[1], args[2] = fused
        conv.args = tuple(args)

        if bn.target == _BATCH_NORM:
            bn.replace_all_uses_with(conv)
        else:
            for user in list(bn.users):
                user.replace_all_uses_with(conv)
                graph.erase_node(user)
        graph.erase_node(bn)

        for node in {weight, bias, bn_weight, bn_bias, running_mean, running_var}:
            if isinstance(node, torch.fx.Node) and len(node.users) == 0:
                delete_constant_placeholder(self.edge_program, node)

    def call(self, graph_module: torch.fx.GraphModule):
        graph = graph_module.graph
        modified = False
        for conv in list(graph.nodes):
            bn = self._match_bn(conv)
            if bn is not None:
                self._fold(graph, conv, bn)
                modified = True
        if modified:
            graph_module.recompile()
        return PassResult(graph_module, modified)
