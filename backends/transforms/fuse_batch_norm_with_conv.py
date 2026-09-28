# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import operator
from typing import Optional, Tuple

import torch

from executorch.backends.transforms.utils import get_param_tensor, is_param_node
from executorch.exir import ExportedProgram
from executorch.exir.dialects._ops import ops as exir_ops
from executorch.exir.pass_base import ExportPass, PassResult

from torch.nn.utils.fusion import fuse_conv_bn_weights


class FuseBatchNormWithConvPass(ExportPass):
    """
    Batch Norm can be implemented using 1x1 Depthwise Convolution. However doing so will increase
    memory usage since we serialize new weights to represent the convolution. In most cases,
    Batch norm is used after convolution. The 1x1 depthwise convolution can then be fused
    with the previous convolution
    """

    def __init__(self, exported_program: ExportedProgram):
        super().__init__()
        self.exported_program = exported_program

    def call(self, graph_module: torch.fx.GraphModule):
        graph = graph_module.graph
        counter = 0
        for conv in graph.nodes:
            # We want to discover a chain of conv -> batch_norm.
            # Only proceed if the current node is a conv node, and has a single
            # user/successor.
            if (
                conv.target != exir_ops.edge.aten.convolution.default
                or len(conv.users) != 1
            ):
                continue

            # The single user of conv op must be batch_norm. If not, bail.
            bn = list(conv.users.keys())[0]
            if (
                bn.target != exir_ops.edge.aten.native_batch_norm.default
                and bn.target
                != exir_ops.edge.aten._native_batch_norm_legit_no_training.default
            ):
                continue

            if not self.can_fuse(conv, bn, self.exported_program):
                continue

            # Get the parameters from conv op
            assert len(conv.args) == 9

            conv_weight = get_param_tensor(self.exported_program, conv.args[1])
            assert conv_weight is not None

            conv_bias = get_param_tensor(self.exported_program, conv.args[2])

            # Get the parameters from the batchnorm op
            assert (
                bn.target == exir_ops.edge.aten.native_batch_norm.default
                and len(bn.args) == 8
            ) or (
                bn.target
                == exir_ops.edge.aten._native_batch_norm_legit_no_training.default
                and len(bn.args) == 7
            )
            bn_weight = get_param_tensor(self.exported_program, bn.args[1])
            bn_bias = get_param_tensor(self.exported_program, bn.args[2])

            running_mean = get_param_tensor(self.exported_program, bn.args[3])
            assert running_mean is not None

            running_var = get_param_tensor(self.exported_program, bn.args[4])
            assert running_var is not None

            # args[7] for native_batch_norm, but args[6] for
            # _native_batch_norm_legit_no_training (which doesn't have training
            # as an arg)
            eps = bn.args[-1]

            # Compute the updated weight and bias after fusing conv op
            # with batchnorm op.
            fused_weight, fused_bias = self._fuse_conv_bn_weights(
                conv_weight,
                conv_bias,
                running_mean,
                running_var,
                eps,
                bn_weight,
                bn_bias,
                transposed=bool(conv.args[6]),
                groups=conv.args[8],
            )

            # Modify the graph by updating the weight and bias of conv op
            # with the fused weight and bias params, and replacing all the users
            # of getitem(batchnorm) with the conv op.
            with graph.inserting_before(conv):
                fused_weight_name = f"_fused_with_bn_weight_{counter}"
                graph_module.register_parameter(fused_weight_name, fused_weight)
                fused_weight_node = graph.get_attr(fused_weight_name)
                fused_bias_name = f"_fused_with_bn_bias_{counter}"
                graph_module.register_parameter(fused_bias_name, fused_bias)
                fused_bias_node = graph.get_attr(fused_bias_name)

            # Update the weight and bias of conv op
            conv_args = list(conv.args) + ([None] if len(conv.args) == 2 else [])
            conv_args[1] = fused_weight_node
            conv_args[2] = fused_bias_node
            conv.args = tuple(conv_args)
            # Remove any use of batchnorm from the graph
            for user in bn.users.copy():
                assert user.target == operator.getitem
                user.replace_all_uses_with(conv)
                graph.erase_node(user)

            graph.erase_node(bn)

            counter += 1

        graph_module.recompile()
        # To Regenerate meta data and shape information, retrace module
        graph_module = super().call(graph_module).graph_module

        return PassResult(graph_module, True)

    @staticmethod
    def _fuse_conv_bn_weights(
        conv_weight: torch.Tensor,
        conv_bias: Optional[torch.Tensor],
        running_mean: torch.Tensor,
        running_var: torch.Tensor,
        eps: float,
        bn_weight: Optional[torch.Tensor],
        bn_bias: Optional[torch.Tensor],
        transposed: bool,
        groups: int,
    ) -> Tuple[torch.nn.Parameter, torch.nn.Parameter]:
        """
        A transposed conv weight is [in, out/groups, *kernel], so its output
        channels sit on dim 1, split into one block per group along dim 0.
        Regroup it to [in/groups, out, *kernel] so dim 1 holds every output
        channel, fold with transpose=True, then restore the original layout.
        """
        if not transposed or groups == 1:
            return fuse_conv_bn_weights(
                conv_weight,
                conv_bias,
                running_mean,
                running_var,
                eps,
                bn_weight,
                bn_bias,
                transpose=transposed,
            )

        in_channels, out_per_group, *kernel = conv_weight.shape
        grouped_shape = (groups, in_channels // groups, out_per_group, *kernel)
        regrouped = (
            conv_weight.reshape(grouped_shape)
            .transpose(0, 1)
            .reshape(in_channels // groups, groups * out_per_group, *kernel)
        )
        fused_weight, fused_bias = fuse_conv_bn_weights(
            regrouped,
            conv_bias,
            running_mean,
            running_var,
            eps,
            bn_weight,
            bn_bias,
            transpose=True,
        )
        fused_weight = torch.nn.Parameter(
            fused_weight.detach()
            .reshape(in_channels // groups, groups, out_per_group, *kernel)
            .transpose(0, 1)
            .reshape(conv_weight.shape),
            fused_weight.requires_grad,
        )
        return fused_weight, fused_bias

    @staticmethod
    def can_fuse(
        conv: torch.fx.Node, bn: torch.fx.Node, program: ExportedProgram
    ) -> bool:
        """
        Determine whether a batch norm node can be fused with a preceding conv node.
        """

        # All the users of batchnorm node must be getitem ops. batchnorm
        # returns a 3-element tuple. Each user must only access the first
        # element of the tuple.
        if [
            (user.target == operator.getitem and user.args[1] == 0) for user in bn.users
        ].count(False):
            return False

        conv_weights = conv.args[1]
        bn_weights = bn.args[1]

        # Check that the weights for conv and batchnorm are both params
        if not isinstance(conv_weights, torch.fx.Node) or not isinstance(
            bn_weights, torch.fx.Node
        ):
            return False

        if [is_param_node(program, node) for node in {conv_weights, bn_weights}].count(
            False
        ):
            return False

        return True
