# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import logging
import operator

import torch
from executorch.backends.transforms.utils import create_constant_placeholder
from executorch.backends.xnnpack._passes.xnnpack_pass import XNNPACKPass
from executorch.backends.xnnpack.utils.utils import (
    check_or_raise,
    get_param_tensor,
    get_tensor_name,
    is_param_node,
)
from executorch.exir.backend.utils import WhyNoPartition
from executorch.exir.dialects._ops import ops as exir_ops
from torch.export.graph_signature import InputKind
from torch.fx.passes.infra.pass_base import PassResult

logger = logging.getLogger(__name__)
logger.setLevel(logging.WARNING)


class DecomposeBatchNorm(XNNPACKPass):
    """
    Decompose batchnorm operators into 1x1 depthwise convolution.
    """

    BATCH_NORM_OPS = {
        exir_ops.edge.aten.native_batch_norm.default,
        exir_ops.edge.aten._native_batch_norm_legit_no_training.default,
    }

    @staticmethod
    def can_decompose_batch_norm(  # noqa: C901
        node: torch.fx.Node,
        exported_program: torch.export.ExportedProgram,
        why: WhyNoPartition | None = None,
    ) -> bool:
        """
        Determine whether the given batch norm node can be decomposed by this pass.
        """

        if (
            node.op != "call_function"
            or node.target not in DecomposeBatchNorm.BATCH_NORM_OPS
        ):
            return False

        input_meta = node.args[0].meta["val"]

        # Since we're converting to conv and XNNPACK doesn't support conv3d, we can't
        # handle BatchNorm3d. Validate the input dimension. We'll take NC, NCL, or NCHW.
        if input_meta.dim() not in (2, 3, 4):
            if why:
                why(
                    node,
                    f"Unsupported input rank {input_meta.dim()} for XNN batch norm operator.",
                )
            return False

        # The batch norm node returns a tuple of output and other stuff we don't care about.
        # All users must be getitem nodes that fetch the output (index 0).
        # The partitioner should enforce this, but we'll check it here too.
        for user in node.users:
            if user.target != operator.getitem or user.args[1] != 0:
                if why:
                    why(node, "Batch norm users must only access the output tensor.")
                return False

        # Channel dimension and non-input args must be statically known.
        if not isinstance(input_meta.shape[1], int):
            if why:
                why(
                    node,
                    f"Channel dimension must be statically known, but was {input_meta.shape[1]}.",
                )
            return False

        if node.args[1] is not None and not is_param_node(
            exported_program, node.args[1]
        ):
            if why:
                why(node, "Batch norm affine weight must be static.")
            return False

        if node.args[2] is not None and not is_param_node(
            exported_program, node.args[2]
        ):
            if why:
                why(node, "Batch norm affine bias must be static.")
            return False

        if not is_param_node(exported_program, node.args[3]) or not is_param_node(
            exported_program, node.args[4]
        ):
            if why:
                why(node, "Batch norm running mean and variance must be static.")
            return False

        if isinstance(node.args[-1], torch.fx.Node):
            if why:
                why(node, "Batch norm epsilon must be static.")
            return False

        if (
            node.target == exir_ops.edge.aten.native_batch_norm.default
            and node.args[5] is not False
        ):
            if why:
                why(node, "Training batch norm is not supported.")
            return False

        return True

    @staticmethod
    def compute_w_and_b(
        eps: float,
        running_mean: torch.Tensor,  # [C]
        running_var: torch.Tensor,  # [C]
        gamma: torch.Tensor,  # [C], learned weight
        beta: torch.Tensor,  # [C], learned bias
    ) -> (torch.Tensor, torch.Tensor):
        """
        Compute equivalent per-channel weight and bias to match the batch norm
        computation with frozen values.
        """

        # See https://docs.pytorch.org/docs/stable/generated/torch.nn.BatchNorm1d.html

        # Do the math in double precision and convert back to the original dtype at the
        # end. ATen kernels do this math in increased precision for float16. Note that
        # all of the parameter dtypes must match, as per the ATen behavior.

        # Also note that gamma and beta can be None if affine=False. This is equivalent
        # to gamma = 1 and beta = 0.
        gamma_f64 = gamma.double() if gamma is not None else torch.Tensor([1]).double()
        beta_f64 = beta.double() if beta is not None else torch.Tensor([0]).double()
        running_mean_f64 = running_mean.double()
        running_var_f64 = running_var.double()

        denom = torch.sqrt(running_var_f64 + torch.Tensor([eps]))
        new_weight = gamma_f64 / denom
        new_bias = -running_mean_f64 * gamma_f64 / denom + beta_f64

        return new_weight.to(running_mean.dtype), new_bias.to(running_mean.dtype)

    def replace_bn_node_with_conv(
        self,
        bn_node: torch.fx.Node,
        graph_module: torch.fx.GraphModule,
    ) -> torch.fx.Node:
        """
        Replace a BatchNorm with NCL or NCHW input with an equivalent depthwise
        convolution.
        """

        # Compute the equivalent per-channel weights and biases.
        # Note that the batch norm node args are
        #   (input, gamma, beta, running_mean, running_var, [training], momentum, eps).
        # The training arg is not present in the _no_training variant.
        weight, bias = DecomposeBatchNorm.compute_w_and_b(
            eps=bn_node.args[-1],
            running_mean=get_param_tensor(self.exported_program, bn_node.args[3]),
            running_var=get_param_tensor(self.exported_program, bn_node.args[4]),
            gamma=get_param_tensor(self.exported_program, bn_node.args[1]),
            beta=get_param_tensor(self.exported_program, bn_node.args[2]),
        )

        # Conv weights have shape [out_c, in_c/g, spatial...].
        # For dw, in_c = g. The kernel is also 1x1 (or just 1, for 1d).
        #
        # BatchNorm weights have shape [in_c].
        # So we just need to unsqueeze the [in_c] to to [in_c, 1, 1, [1]].
        input_meta = bn_node.args[0].meta["val"]
        channel_count = input_meta.shape[1]
        spatial_dims = max(
            input_meta.dim() - 2, 1
        )  # Min of 1 since 1d can be NC or NCL.
        new_weight_shape = [weight.shape[0], 1] + [1] * spatial_dims
        weight = weight.reshape(new_weight_shape)

        # Generate names for the new weight and bias parameters based on the original
        # batch norm gamma parameter name.
        gamma_name = get_tensor_name(self.exported_program, bn_node.args[1])
        weight_name = (gamma_name + "_decomposed_bn_weight").replace(".", "_")
        bias_name = (gamma_name + "_decomposed_bn_bias").replace(".", "_")

        # Insert the new weight and bias as constant placeholders in the graph.
        with graph_module.graph.inserting_before(bn_node.args[1]):
            weight_node = create_constant_placeholder(
                exp_program=self.exported_program,
                graph=graph_module.graph,
                kind=InputKind.PARAMETER,
                name=weight_name,
                data=weight,
            )
            bias_node = create_constant_placeholder(
                exp_program=self.exported_program,
                graph=graph_module.graph,
                kind=InputKind.PARAMETER,
                name=bias_name,
                data=bias,
            )

        with graph_module.graph.inserting_after(bn_node):
            conv_node = graph_module.graph.call_function(
                exir_ops.edge.aten.convolution.default,
                args=(
                    bn_node.args[0],  # Input
                    weight_node,  # Weight
                    bias_node,  # Bias
                    [1] * spatial_dims,  # Stride
                    [0] * spatial_dims,  # Padding
                    [1] * spatial_dims,  # Dilation
                    False,  # Transposed
                    [0] * spatial_dims,  # Output_padding
                    channel_count,  # Groups (depthwise, so groups=in_channels)
                ),
            )

            # Find the getitem user nodes and replace them with the conv node.
            # The decomp checks above enforce that the node is only used by getitem[0].
            users = list(bn_node.users)
            for user in users:
                user.replace_all_uses_with(conv_node)
                graph_module.graph.erase_node(user)

            graph_module.graph.erase_node(bn_node)
            return conv_node

    def decompose_node(
        self, node: torch.fx.Node, graph_module: torch.fx.GraphModule
    ) -> None:
        input_meta = node.args[0].meta["val"]

        # These should be checked by the partitioner and calling node,
        # so we should never fail these checks.
        check_or_raise(
            node.op == "call_function"
            and node.target in DecomposeBatchNorm.BATCH_NORM_OPS,
            f"Invalid batch norm operator {node.op}.",
        )

        check_or_raise(
            input_meta.dim() in (2, 3, 4),
            f"Unsupported input rank {input_meta.dim()} for XNN batch norm operator.",
        )

        channel_count = input_meta.shape[1]
        check_or_raise(
            isinstance(channel_count, int),
            f"Channel dimension must be statically known, but was {channel_count}.",
        )

        # Create the convolution node.
        conv_node = self.replace_bn_node_with_conv(node, graph_module)

        # BatchNorm1d can be NC or NCL. Conv1d requies the L dim, so unsqueeze NC -> NCL.
        if input_meta.dim() == 2:
            with graph_module.graph.inserting_before(conv_node):
                # Insert unsqueeze node before.
                unsqueeze_node = graph_module.graph.call_function(
                    exir_ops.edge.aten.unsqueeze_copy.default,
                    args=(conv_node.args[0], 2),
                )
                conv_node.args = (unsqueeze_node, *conv_node.args[1:])

            with graph_module.graph.inserting_after(conv_node):
                # Insert squeeze node after.
                squeeze_node = graph_module.graph.call_function(
                    exir_ops.edge.aten.squeeze_copy.dim, args=(conv_node, 2)
                )
                conv_node.replace_all_uses_with(squeeze_node)
                # This gets overwritten by replace_all_uses_with. Maybe there's
                # a better solution?
                squeeze_node.args = (conv_node, *squeeze_node.args[1:])

    # override
    def call(self, graph_module: torch.fx.GraphModule):
        # Find and transform all eligible batch norm nodes.
        for node in graph_module.graph.nodes:
            if node.op == "call_function" and node.target in self.BATCH_NORM_OPS:
                if self.can_decompose_batch_norm(node, self.exported_program):
                    self.decompose_node(node, graph_module)

        graph_module.recompile()

        # Propagate metadata and retrace module
        graph_module = super().call(graph_module).graph_module

        return PassResult(graph_module, True)
