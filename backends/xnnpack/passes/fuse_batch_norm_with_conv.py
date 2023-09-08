# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import operator

import torch

from executorch.backends.xnnpack.passes.xnnpack_pass import XNNPACKPass

from executorch.backends.xnnpack.utils.utils import get_param_tensor, is_param_node
from executorch.exir.dialects._ops import ops as exir_ops
from executorch.exir.pass_base import PassResult

from torch.nn.utils.fusion import fuse_conv_bn_weights


class FuseBatchNormWithConvPass(XNNPACKPass):
    """
    Batch Norm can be implemented using 1x1 Depthwise Convolution. However doing so will increase
    memory usage since we serialize new weights to represent the convolution. In most cases,
    Batch norm is used after convoluution. The 1x1 depthwise convolution can then be fused
    with the previous convolution

    """

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

            # All the users of batchnorm node must be getitem ops. batchnorm
            # returns a 3-element tuple. Each user must only access the first
            # element of the tuple.
            if [
                (user.target == operator.getitem and user.args[1] == 0)
                for user in bn.users
            ].count(False):
                continue

            # Check that the weights for conv and batchnorm are both params
            if [
                is_param_node(self.exported_program, node)
                for node in {conv.args[1], bn.args[1]}
            ].count(False):
                continue

            # Get the parameters from conv op
            assert len(conv.args) == 9
            conv_weight = get_param_tensor(self.exported_program, conv.args[1])
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
            running_var = get_param_tensor(self.exported_program, bn.args[4])

            # args[7] for native_batch_norm, but args[6] for
            # _native_batch_norm_legit_no_training (which doesn't have training
            # as an arg)
            eps = bn.args[-1]

            # Compute the updated weight and bias after fusing conv op
            # with batchnorm op.
            fused_weight, fused_bias = fuse_conv_bn_weights(
                conv_weight,
                conv_bias,
                running_mean,
                running_var,
                eps,
                bn_weight,
                bn_bias,
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
