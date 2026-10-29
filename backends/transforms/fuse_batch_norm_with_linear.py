# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import operator

import torch

from executorch.backends.transforms.utils import get_param_tensor, is_param_node
from executorch.exir import ExportedProgram
from executorch.exir.dialects._ops import ops as exir_ops
from executorch.exir.pass_base import ExportPass, PassResult

from torch.nn.utils.fusion import fuse_linear_bn_weights


class FuseBatchNormWithLinearPass(ExportPass):
    """
    Fold an inference batch norm into the preceding linear's weight and bias.
    """

    def __init__(self, exported_program: ExportedProgram):
        super().__init__()
        self.exported_program = exported_program

    def call(self, graph_module: torch.fx.GraphModule):
        graph = graph_module.graph
        counter = 0
        for linear in graph.nodes:
            # We want to discover a chain of linear -> batch_norm.
            # Only proceed if the current node is a linear node, and has a
            # single user/successor.
            if (
                linear.target != exir_ops.edge.aten.linear.default
                or len(linear.users) != 1
            ):
                continue

            # The single user of linear op must be batch_norm. If not, bail.
            bn = list(linear.users.keys())[0]
            if (
                bn.target != exir_ops.edge.aten.native_batch_norm.default
                and bn.target
                != exir_ops.edge.aten._native_batch_norm_legit_no_training.default
            ):
                continue

            if not self.can_fuse(linear, bn, self.exported_program):
                continue

            # Linear has 2 or 3 args: (input, weight, bias).
            assert len(linear.args) in (2, 3)

            linear_weight = get_param_tensor(self.exported_program, linear.args[1])
            assert linear_weight is not None

            linear_bias = (
                get_param_tensor(self.exported_program, linear.args[2])
                if len(linear.args) == 3
                else None
            )

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

            if bn_bias is None:
                bn_bias = torch.zeros_like(running_mean)

            # args[7] for native_batch_norm, but args[6] for
            # _native_batch_norm_legit_no_training (which doesn't have training
            # as an arg)
            eps = bn.args[-1]

            # Compute the updated weight and bias after fusing linear op
            # with batchnorm op.
            fused_weight, fused_bias = fuse_linear_bn_weights(
                linear_weight,
                linear_bias,
                running_mean,
                running_var,
                eps,
                bn_weight,
                bn_bias,
            )

            # Modify the graph by updating the weight and bias of linear op
            # with the fused weight and bias params, and replacing all the users
            # of getitem(batchnorm) with the linear op.
            with graph.inserting_before(linear):
                fused_weight_name = f"_fused_with_bn_weight_{counter}"
                graph_module.register_parameter(fused_weight_name, fused_weight)
                fused_weight_node = graph.get_attr(fused_weight_name)
                fused_bias_name = f"_fused_with_bn_bias_{counter}"
                graph_module.register_parameter(fused_bias_name, fused_bias)
                fused_bias_node = graph.get_attr(fused_bias_name)

            # Update the weight and bias of linear op
            linear_args = list(linear.args) + ([None] if len(linear.args) == 2 else [])
            linear_args[1] = fused_weight_node
            linear_args[2] = fused_bias_node
            linear.args = tuple(linear_args)
            # Remove any use of batchnorm from the graph
            for user in bn.users.copy():
                assert user.target == operator.getitem
                user.replace_all_uses_with(linear)
                graph.erase_node(user)

            graph.erase_node(bn)

            counter += 1

        if counter == 0:
            return PassResult(graph_module, False)

        graph_module.recompile()
        # To Regenerate meta data and shape information, retrace module
        graph_module = super().call(graph_module).graph_module

        return PassResult(graph_module, True)

    @staticmethod
    def can_fuse(
        linear: torch.fx.Node, bn: torch.fx.Node, program: ExportedProgram
    ) -> bool:
        """
        Determine whether a batch norm node can be fused with a preceding linear node.
        """

        # All the users of batchnorm node must be getitem ops. batchnorm
        # returns a 3-element tuple. Each user must only access the first
        # element of the tuple.
        if [
            (user.target == operator.getitem and user.args[1] == 0) for user in bn.users
        ].count(False):
            return False

        if bn.target == exir_ops.edge.aten.native_batch_norm.default and bn.args[5]:
            return False

        # Linear and batch norm share their channel axis only for rank-2 outputs.
        linear_output = linear.meta.get("val")
        if not isinstance(linear_output, torch.Tensor) or linear_output.dim() != 2:
            return False

        params = (linear.args[1], bn.args[1], bn.args[3], bn.args[4])
        params += tuple(
            param for param in linear.args[2:3] + bn.args[2:3] if param is not None
        )
        return all(
            isinstance(param, torch.fx.Node) and is_param_node(program, param)
            for param in params
        )
