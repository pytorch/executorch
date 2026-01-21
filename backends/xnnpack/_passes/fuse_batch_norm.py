# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import operator

import torch
from executorch.backends.transforms.utils import (
    create_constant_placeholder,
    delete_constant_placeholder,
)
from executorch.backends.xnnpack._passes.xnnpack_pass import XNNPACKPass
from executorch.backends.xnnpack.utils.utils import (
    get_param_tensor,
    get_tensor_name,
    is_param_node,
)
from executorch.exir import ExportedProgram
from executorch.exir.backend.utils import WhyNoPartition
from executorch.exir.dialects._ops import ops as exir_ops
from executorch.exir.pass_base import PassResult
from torch.export.graph_signature import InputKind
from torch.nn.utils.fusion import fuse_conv_bn_weights, fuse_linear_bn_weights


class FuseBatchNormPass(XNNPACKPass):
    """
    BatchNorm can be implemented using 1x1 Depthwise Convolution. However, doing so will increase
    memory usage since we serialize new weights to represent the convolution. In most cases,
    BatchNorm is used after convolution or linear. The 1x1 depthwise convolution can then be fused
    with the previous convolution. For linear cases, BatchNorm can be folded into the previous linear layer.
    """

    def call(self, graph_module: torch.fx.GraphModule):
        graph = graph_module.graph
        constant_placeholders_to_delete = set()
        for input_node in graph.nodes:
            # We want to discover a chain of conv -> batch_norm or linear -> batch_norm.
            # Only proceed if the current node is a conv or linear, and has a single user/successor.
            is_conv = input_node.target == exir_ops.edge.aten.convolution.default
            is_linear = input_node.target == exir_ops.edge.aten.linear.default

            if not (is_conv or is_linear) or len(input_node.users) != 1:
                continue

            # The single user of the conv or linear node must be batch_norm. If not, bail.
            bn = list(input_node.users.keys())[0]
            if (
                bn.target != exir_ops.edge.aten.native_batch_norm.default
                and bn.target
                != exir_ops.edge.aten._native_batch_norm_legit_no_training.default
            ):
                continue

            if not self.can_fuse(input_node, bn, self.exported_program):
                continue

            self._fuse_ops(
                graph_module,
                graph,
                input_node,
                bn,
                is_conv,
                constant_placeholders_to_delete,
            )

        if len(constant_placeholders_to_delete) > 0:
            graph_module.graph.eliminate_dead_code()
            for node in constant_placeholders_to_delete:
                if (node is not None) and (len(node.users) == 0):
                    delete_constant_placeholder(self.exported_program, node)

        graph_module.recompile()
        # To regenerate metadata and shape information, retrace module.
        graph_module = super().call(graph_module).graph_module

        return PassResult(graph_module, True)

    @staticmethod
    def can_fuse(  # noqa: C901
        input_node: torch.fx.Node,
        bn: torch.fx.Node,
        program: ExportedProgram,
        why: WhyNoPartition | None = None,
    ) -> bool:
        """
        Determine whether a BatchNorm node can be fused with the preceding convolution or linear node.
        """

        if input_node.op != "call_function":
            return False

        if input_node.target not in (
            exir_ops.edge.aten.convolution.default,
            exir_ops.edge.aten.linear.default,
        ):
            if why:
                why("Input node must be a convolution or linear op.")
            return False

        is_conv = input_node.target == exir_ops.edge.aten.convolution.default

        # All users of the batch_norm node must be getitem ops.
        # batch_norm returns a 3-element tuple.
        # Each user must only access the first element of the tuple.
        if [
            (user.target == operator.getitem and user.args[1] == 0) for user in bn.users
        ].count(False):
            if why:
                why("Batch norm users must only access the output tensor.")
            return False

        input_node_weights = input_node.args[1]
        bn_weights = bn.args[1]

        # Check that the weights for conv or linear and batch_norm are both params.
        if not isinstance(input_node_weights, torch.fx.Node) or not isinstance(
            bn_weights, torch.fx.Node
        ):
            if why:
                why("Input node weights must be parameters.")
            return False

        if [
            is_param_node(program, node) for node in {input_node_weights, bn_weights}
        ].count(False):
            if why:
                why("Node weights must be static.")
            return False

        # Check the rank of the convolutution input - only Conv1d and 2d are supported.
        if is_conv:
            conv_input = input_node.args[0]
            if (
                not isinstance(conv_input, torch.fx.Node)
                or "val" not in conv_input.meta
                or len(conv_input.meta["val"].shape) not in (3, 4)
            ):
                if why:
                    why("Convolution input must be rank 3 or 4.")
                return False

        return True

    def _fuse_ops(
        self,
        graph_module: torch.fx.GraphModule,
        graph: torch.fx.Graph,
        input_node: torch.fx.Node,
        bn: torch.fx.Node,
        is_conv: bool,
        constant_placeholders_to_delete: set,
    ) -> None:
        """
        Fuse a BatchNorm node into the preceding convolution or linear node.
        Update the fused node's weight and bias, rewire users of the BatchNorm output,
        and remove the BatchNorm node.
        """

        if is_conv:
            assert len(input_node.args) == 9
            has_bias_arg = True
        else:
            # Otherwise, this is a linear node.
            # Linear has 2 or 3 args depending on whether bias is used: (input, weight, bias).
            assert len(input_node.args) in (2, 3)
            has_bias_arg = len(input_node.args) == 3

        # Get the weight and bias parameters from the conv or linear op.
        input_node_weight = get_param_tensor(self.exported_program, input_node.args[1])
        input_node_weight_name = get_tensor_name(
            self.exported_program, input_node.args[1]
        )
        assert input_node_weight is not None

        if has_bias_arg:
            input_node_bias = get_param_tensor(
                self.exported_program, input_node.args[2]
            )
            input_node_bias_name = get_tensor_name(
                self.exported_program, input_node.args[2]
            )
        else:
            input_node_bias = None
            input_node_bias_name = ""

        # Get the parameters from the batch_norm op.
        assert (
            bn.target == exir_ops.edge.aten.native_batch_norm.default
            and len(bn.args) == 8
        ) or (
            bn.target == exir_ops.edge.aten._native_batch_norm_legit_no_training.default
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
        # as an arg).
        eps = bn.args[-1]

        # Compute the updated weight and bias after fusing the conv or linear op with the batch_norm op.
        fuse_args = (
            input_node_weight,
            input_node_bias,
            running_mean,
            running_var,
            eps,
            bn_weight,
            bn_bias,
        )

        if is_conv:
            is_transpose = input_node.args[6]
            fused_weight, fused_bias = fuse_conv_bn_weights(*fuse_args, is_transpose)
        else:
            # Otherwise, this is a linear node.
            fused_weight, fused_bias = fuse_linear_bn_weights(*fuse_args)

        fused_weight_name = (input_node_weight_name + "_fused_bn").replace(".", "_")
        if input_node_bias_name == "":
            fused_bias_name = (input_node_weight_name + "_bias_fused_bn").replace(
                ".", "_"
            )
        else:
            fused_bias_name = (input_node_bias_name + "_fused_bn").replace(".", "_")

        # Modify the graph by updating the weight and bias of the conv or linear op
        # with the fused weight and bias params, and replacing all the users
        # of getitem(batch_norm) with the conv or linear op.
        with graph.inserting_before(input_node.args[1]):
            fused_op_weight_node = create_constant_placeholder(
                exp_program=self.exported_program,
                graph=graph_module.graph,
                kind=InputKind.PARAMETER,
                name=fused_weight_name,
                data=fused_weight,
            )
            if fused_bias is not None:
                fused_op_bias_node = create_constant_placeholder(
                    exp_program=self.exported_program,
                    graph=graph_module.graph,
                    kind=InputKind.PARAMETER,
                    name=fused_bias_name,
                    data=fused_bias,
                )
            else:
                fused_op_bias_node = None

            # Replace the original weight and bias with the fused batch_norm values.
            args = list(input_node.args)
            args[1] = fused_op_weight_node

            if has_bias_arg:
                # Overwrite original bias with the fused bias.
                args[2] = fused_op_bias_node
            elif fused_op_bias_node is not None:
                # Add the fused bias as a new argument if no bias had originally existed in the input_node.
                args.append(fused_op_bias_node)

            input_node.args = tuple(args)

            # Remove any use of batch_norm from the graph.
            for user in bn.users.copy():
                assert user.target == operator.getitem
                user.replace_all_uses_with(input_node)
                graph.erase_node(user)

            graph.erase_node(bn)
            constant_placeholders_to_delete.update(input_node.args[1:3] + bn.args[1:5])
