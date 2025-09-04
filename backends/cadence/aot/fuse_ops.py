# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-unsafe


# This file contains all the functions that fuse ops in the fx graph.

import logging
import math
import operator
from collections import deque
from numbers import Number
from typing import Any, Callable, cast

# Import these for the cadence function signatures.
import executorch.backends.cadence.aot.ops_registrations  # noqa: F401

import torch
import torch.fx
from executorch.backends.cadence.aot.compiler_utils import (
    broadcastable,
    get_cascaded_ops,
    get_permuted_dims,
    get_scale,
    get_shape,
    get_tensor_from_attr,
    get_transposed_dims,
    get_zero_point,
)
from executorch.backends.cadence.aot.pass_utils import (
    CadencePassAttribute,
    register_cadence_pass,
)
from executorch.backends.cadence.aot.utils import get_edge_overload_packet
from executorch.exir.dialects._ops import ops as exir_ops
from executorch.exir.dialects.edge._ops import EdgeOpOverload, EdgeOpOverloadPacket
from executorch.exir.pass_base import ExportPass, NodeMetadata, PassResult, ProxyValue
from executorch.exir.passes import dead_code_elimination_pass
from executorch.exir.passes.spec_prop_pass import SpecPropPass
from torch.fx.node import Argument
from torch.nn.utils.fusion import fuse_conv_bn_weights


@register_cadence_pass(CadencePassAttribute(opt_level=1))
class FuseMMWithAdd(ExportPass):
    # Return true if the node is a view node.

    def is_view_node(self, node: torch.fx.Node):
        return node.target == exir_ops.edge.aten.view_copy.default

    def fuse_mm_with_add(self, graph_module: torch.fx.GraphModule):
        """
        Given a graph of the form:
        X = aten.mm(A, B)
        Y = aten.add(X, C)
        Fuse X and Y into a single addmm node, after making sure that we can
        broadcast C into X.
        There could be view node that takes a view of X, and feeds that
        to the aten.add node:
        X = aten.mm(A, B)
        Y = X.view()
        Z = aten.add(Y, C)
        Handle this case as well. There are a few conditions for the
        optimization to be valid:
        1. There should be a single user of the mm node, otherwise we cannot
        remove it.
        2. There should be a single user of the add node, otherwise we cannot
        fuse it with mm.
        """
        graph = graph_module.graph
        for node in graph.find_nodes(
            op="call_function", target=exir_ops.edge.aten.mm.default
        ):
            # We want to discover a chain of mm -> add, or mm -> view -> add.
            # Only proceed if the current node is an mm node, and has only one
            # user/successor.
            if len(node.users) != 1:
                continue

            # Our addmm implementation computes (mat1 * mat2 + bias). So the
            # addmm node in the graph should have three args. We collectively
            # term mat1 and mat2 as mm_arg since they are the args of mm node,
            # and bias as bias_arg.
            # Since we already have discovered the mm node, we can get mat1 and
            # mat2 by iterating over its args. So the current node is mm_arg.
            # bias_arg can be found once we discover the add op that consumes
            # the output of this mm node. Our next step is to find the add op.
            mm_arg = node
            user = list(node.users.keys())[0]
            # intermediate_view is True when the fusion case is mm -> view -> add
            intermediate_view = False
            # Check if the single user of the mm node is a view op. If so, our
            # graph could potentially have mm -> view -> add. We need to skip
            # the view op, and check if its successor is the add op. One condition
            # we need to verify is that the view op must have only a single user
            # (the add op).
            if self.is_view_node(user) and len(user.users) == 1:
                # We want to maintain two invariants:
                # (1) 'user' is a potential add op that will get fused with the
                #     mm node;
                # (2) 'node' is the single predecessor of 'user' that is either
                #     the mm node, or the current view node;
                # To maintain the invariant, we must mark this view op as 'node',
                # and its single successor as 'user'.
                intermediate_view = True
                node = user
                user = list(node.users.keys())[0]

            # Thanks to the invariant, we can now simply check if 'user' is an
            # add op. We also want to ensure that the add op has only one user,
            # otherwise we will get not be able to eliminate add op post fusion.
            if user.target != exir_ops.edge.aten.add.Tensor or len(user.users) != 1:
                continue

            # At this point, we have found an mm and an add node that we can
            # fuse together. One arg of the add op is 'node' (thanks to the
            # invariant). Find the other arg, and tag it as bias_arg.
            assert len(user.args) == 2
            bias_arg = user.args[1] if user.args[0] == node else user.args[0]

            # As a last check, make sure that we can broadcast the bias tensor
            # to the output of mm.
            mm_arg_shape = get_shape(graph_module, mm_arg)
            bias_arg_shape = get_shape(graph_module, bias_arg)
            if (
                mm_arg_shape is None
                or bias_arg_shape is None
                or not broadcastable(mm_arg_shape, bias_arg_shape)
                or len(bias_arg_shape) > 2
            ):
                continue

            # Create a new addmm node, and insert it before add node. DCE should
            # take care of removing the dead mm and/or view node. Based on the
            # invariant, add node corresponds to 'user'.
            with graph.inserting_before(user):
                addmm_node = graph.call_function(
                    exir_ops.edge.aten.addmm.default,
                    args=(bias_arg, mm_arg.args[0], mm_arg.args[1]),
                )
            # Replace all the uses of add node with addmm node, and remove add
            # node from the graph.
            user.replace_all_uses_with(addmm_node)
            graph.erase_node(user)

            # As a finishing step, we want to ensure that the output of addmm is
            # in the expected shape. For example, Let us assume the following
            # input, where A, B are (4, 4) sized tensors, and C is (1, 4) sized
            # tensor.
            # T1 = torch.mm(A, B)
            # T2 = T1.view((2, 2, 4))
            # return torch.add(T2, C)
            # Here, the expectation is to get an output of size (2, 2, 4), which
            # is the shape out of view node T2. However, the fused addmm will
            # return an output of shape (4, 4). In a nutshell, we need to take
            # care of the output shape when the following two conditions are met:
            # 1. The fusion case is mm -> view -> add (i.e., intermediate_view
            #    is True)
            # 2. The single successor of addmm is not a view op.
            addmm_user = list(addmm_node.users.keys())[0]
            if intermediate_view and not self.is_view_node(addmm_user):
                # Create a view node that correctly reshapes the output of addmm
                # (i.e., 'user') to match the output shape of the add node.
                # Thanks to our invariant, we know that the correct shape is held
                # by 'node', which points to the view op in mm -> view -> add chain.
                # We create its copy, and insert it just before addmm_user.
                with graph.inserting_before(addmm_user):
                    view_copy_node = graph_module.graph.node_copy(node)
                # Any uses of addmm are replaced with this view_copy node.
                addmm_node.replace_all_uses_with(view_copy_node)
                # Now we massage the args of the view_copy node, so that it takes
                # view of addmm node.
                view_args = list(view_copy_node.args)
                view_args[0] = addmm_node
                view_copy_node.args = tuple(view_args)

        graph_module.recompile()

    def call(self, graph_module: torch.fx.GraphModule) -> PassResult:
        # Compute the spec prop pass before we begin the fusion pipeline
        result = SpecPropPass()(graph_module)
        assert result is not None
        self.fuse_mm_with_add(result.graph_module)
        result = super().call(result.graph_module)
        return result


@register_cadence_pass(CadencePassAttribute(opt_level=1))
class FuseBatchNormWithConv(ExportPass):
    """
    This pass fuses a conv op with batchnorm if the following two conditions
    are met:
    1. The only user of conv op should be batchnorm;
    2. Only the first element from the batchnorm output tuple should be used
    in the graph.
    """

    def fuse_batch_norm_with_conv(self, graph_module: torch.fx.GraphModule) -> None:
        graph = graph_module.graph
        for conv in graph.nodes:
            # We want to discover a chain of conv1d -> batch_norm.
            # Only proceed if the current node is a conv1d node, and has a single
            # user/successor.
            if (
                conv.target != exir_ops.edge.aten.convolution.default
                or len(conv.users) != 1
            ):
                continue

            # The single user of conv op must be batch_norm. If not, bail.
            bn = list(conv.users.keys())[0]
            if bn.target != exir_ops.edge.aten.native_batch_norm.default:
                continue

            # All the users of batchnorm node must be getitem ops. batchnorm
            # returns a 3-element tuple. Each user must only access the first
            # element of the tuple.
            if [
                (user.target == operator.getitem and user.args[1] == 0)
                for user in bn.users
            ].count(False):
                continue

            # Check that the weights for conv1d and batchnorm are both params
            if [node.op == "get_attr" for node in {conv.args[1], bn.args[1]}].count(
                False
            ):
                continue

            # Get the parameters from conv op
            assert len(conv.args) == 9
            conv_weight = get_tensor_from_attr(graph_module, conv.args[1])
            assert isinstance(conv_weight, torch.Tensor)
            conv_bias = get_tensor_from_attr(graph_module, conv.args[2])
            transpose = conv.args[6]

            # Get the parameters from the batchnorm op
            assert len(bn.args) == 8
            bn_weight = get_tensor_from_attr(graph_module, bn.args[1])
            bn_bias = get_tensor_from_attr(graph_module, bn.args[2])
            running_mean = get_tensor_from_attr(graph_module, bn.args[3])
            assert isinstance(running_mean, torch.Tensor)
            running_var = get_tensor_from_attr(graph_module, bn.args[4])
            assert isinstance(running_var, torch.Tensor)
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
                transpose,
            )

            # Modify the graph by updating the weight and bias of conv op
            # with the fused weight and bias params, and replacing all the users
            # of getitem(batchnorm) with the conv op.
            with graph.inserting_before(conv):
                fused_weight_name = f"_fused_with_bn_weight_{self.counter}"
                graph_module.register_parameter(fused_weight_name, fused_weight)
                fused_weight_node = graph.get_attr(fused_weight_name)
                fused_bias_name = f"_fused_with_bn_bias_{self.counter}"
                graph_module.register_parameter(fused_bias_name, fused_bias)
                fused_bias_node = graph.get_attr(fused_bias_name)

            # Update the weight and bias of conv op
            conv_args = list(conv.args)
            conv_args[1] = fused_weight_node
            conv_args[2] = fused_bias_node
            conv.args = tuple(conv_args)
            # Remove any use of batchnorm from the graph
            for user in bn.users:
                assert user.target == operator.getitem
                user.replace_all_uses_with(conv)
            self.counter += 1

        graph_module.recompile()

    def __init__(self):
        super().__init__()
        self.counter = 0

    def call(self, graph_module: torch.fx.GraphModule) -> PassResult:
        self.fuse_batch_norm_with_conv(graph_module)
        result = super().call(graph_module)
        return result


@register_cadence_pass(CadencePassAttribute(opt_level=1))
class FuseQuantizedBatchNormWithConv(ExportPass):
    """
    This pass fuses a quantized::conv op with quantized::batchnorm if the
    following two conditions are met:
    1. The only user of quantized::conv op should be quantized::batchnorm;
    2. The outputs of both ops are quantized with same scale and zero_point
    """

    def fuse_quantized_batch_norm_with_conv(
        self, graph_module: torch.fx.GraphModule
    ) -> None:
        graph = graph_module.graph
        for conv in graph.nodes:
            # We want to discover a chain of quantized::conv1d ->
            # quantized::batch_norm. Only proceed if the current node is a
            # quantized::conv node, and has a single user/successor.
            if (
                conv.target
                not in {
                    exir_ops.edge.quantized.conv1d.default,
                    exir_ops.edge.quantized.conv2d.new,
                }
                or len(conv.users) != 1
            ):
                continue

            # The single user of conv op must be batch_norm. If not, bail.
            bn = list(conv.users.keys())[0]
            if bn.target not in {
                exir_ops.edge.quantized.batch_norm1d.default,
                exir_ops.edge.quantized.batch_norm2d.default,
            }:
                continue

            # The outputs of conv and bn must both have same scale and zero_point
            if not math.isclose(
                conv.args[-2], bn.args[-2], rel_tol=1e-05, abs_tol=1e-05
            ):
                continue
            if conv.args[-1] != bn.args[-1]:
                continue

            # The weight and bias of quantized::conv op are packed in the second
            # arg. Unpack them.
            assert conv.args[1].op == "get_attr"
            packed_args = getattr(graph_module, conv.args[1].target)
            conv_weight_tensor, conv_bias_tensor = packed_args.unpack()
            # Assert that we have discovered the conv op's weight and bias tensors
            assert isinstance(conv_weight_tensor, torch.Tensor)
            assert conv_bias_tensor is None or isinstance(
                conv_bias_tensor, torch.Tensor
            )

            # Get the scale, zero_point, and dtype of convolution weight
            assert conv_weight_tensor.is_quantized
            per_tensor_quantization = (
                conv_weight_tensor.qscheme() == torch.per_tensor_affine
            )
            weight_dtype = conv_weight_tensor.dtype
            weight_scale = get_scale(conv_weight_tensor)
            weight_zero_point = get_zero_point(conv_weight_tensor, reduce=False)
            weight_axis = (
                0
                if per_tensor_quantization
                else conv_weight_tensor.q_per_channel_axis()
            )
            # Dequantize the convolution weight
            conv_weight_tensor = conv_weight_tensor.dequantize()

            # Get the parameters from the batchnorm op
            assert len(bn.args) == 8
            (bn_weight, bn_bias, running_mean, running_var, eps) = bn.args[1:6]
            # Get the tensors from the batchnorm args
            bn_weight_tensor = get_tensor_from_attr(graph_module, bn_weight)
            bn_bias_tensor = get_tensor_from_attr(graph_module, bn_bias)
            running_mean_tensor = get_tensor_from_attr(graph_module, running_mean)
            running_var_tensor = get_tensor_from_attr(graph_module, running_var)

            # Assert that we have discovered the batch_norm op's tensors
            assert bn_weight_tensor is None or isinstance(
                bn_weight_tensor, torch.Tensor
            )
            assert bn_bias_tensor is None or isinstance(bn_bias_tensor, torch.Tensor)
            assert isinstance(running_mean_tensor, torch.Tensor)
            assert isinstance(running_var_tensor, torch.Tensor)

            # Get the fused weights and bias
            fused_weight, fused_bias = fuse_conv_bn_weights(
                conv_weight_tensor,
                conv_bias_tensor,
                running_mean_tensor,
                running_var_tensor,
                eps,
                bn_weight_tensor,
                bn_bias_tensor,
                transpose=False,
            )

            # Requantize the fused weight with the scale and zero point of the
            # quantized::conv's weight
            if per_tensor_quantization:
                fused_weight = torch.quantize_per_tensor(
                    fused_weight,
                    weight_scale.item(),
                    cast(int, weight_zero_point.item()),
                    weight_dtype,
                )
            else:
                fused_weight = torch.quantize_per_channel(
                    fused_weight,
                    weight_scale,
                    weight_zero_point,
                    weight_axis,
                    weight_dtype,
                )

            # Now that we have the fused weight and bias, pack them for the
            # quantized::conv.
            stride = packed_args.stride()
            padding = packed_args.padding()
            dilation = packed_args.dilation()
            groups = packed_args.groups()
            args = (fused_weight, fused_bias, stride, padding, dilation, groups)
            packed_args = (
                exir_ops.edge.quantized.conv1d_prepack(*args)
                if conv.target == exir_ops.edge.quantized.conv1d.default
                else exir_ops.edge.quantized.conv2d_prepack(*args)
            )

            # Modify the graph by updating the weight and bias of conv op
            # with the fused weight and bias params, and replacing all the users
            # of batchnorm with the conv op.
            conv_args = list(conv.args)
            conv_args[1] = packed_args
            conv.args = tuple(conv_args)
            bn.replace_all_uses_with(conv)
            graph.erase_node(bn)
            self.counter += 1

        # Note: there is a quantized.conv2d.new operator in the resulting graph
        # that takes a torch.classes.quantized.Conv2dPackedParamsBase as one of the input
        # this prevents us to directly call graph_module.recompile().
        # pyre-fixme[16]: `GraphModule` has no attribute `_code`.
        # pyre-fixme[16]: Item `Tensor` of `Tensor | Module` has no attribute
        #  `python_code`.
        graph_module._code = graph_module._graph.python_code(root_module="self").src

    def __init__(self):
        super().__init__()
        self.counter = 0

    def call(self, graph_module: torch.fx.GraphModule) -> PassResult:
        self.fuse_quantized_batch_norm_with_conv(graph_module)
        result = super().call(graph_module)
        return result


@register_cadence_pass(CadencePassAttribute(opt_level=1))
class FuseCascadedTransposeOrPermuteOps(ExportPass):
    """
    Fuse a cascaded chain of transpose and permute ops
    """

    transpose_or_permute_target = {
        exir_ops.edge.aten.transpose_copy.int,
        exir_ops.edge.aten.permute_copy.default,
    }

    # Find a chain of transpose or permute ops, and fuse them into a single permute op.

    def fuse_cascaded_transpose_or_permute_ops(
        self, graph_module: torch.fx.GraphModule
    ):
        graph = graph_module.graph
        for node in graph.nodes:
            # We are only interested in permute/transpose ops
            if node.target not in self.transpose_or_permute_target:
                continue
            # Get the cascaded chain of transpose/permute ops starting at node
            cascaded_transpose_or_permute_ops = get_cascaded_ops(
                [node], self.transpose_or_permute_target
            )
            # The chain must have more than 1 node
            if len(cascaded_transpose_or_permute_ops) == 1:
                continue

            out_shape = get_shape(graph_module, node)
            assert out_shape is not None
            out_dims = len(out_shape)
            # This is the trivial dimension order
            dims = list(range(out_dims))
            # Compute the effect of the chain on dims
            for tp in cascaded_transpose_or_permute_ops:
                dims = (
                    get_transposed_dims(tp, dims)
                    if tp.target == exir_ops.edge.aten.transpose_copy.int
                    else get_permuted_dims(tp, dims)
                )

            # In case the permute chain cancelled each other, the final dims will
            # be the same as the initial order. In that case, the chain was nop.
            # Otherwise create a new permute op that encompasses the effect of the
            # chain.
            if dims == list(range(out_dims)):
                cascaded_transpose_or_permute_ops[-1].replace_all_uses_with(
                    node.args[0]
                )
            else:
                with graph.inserting_before(cascaded_transpose_or_permute_ops[-1]):
                    new_permute = graph.call_function(
                        exir_ops.edge.aten.permute_copy.default,
                        args=(node.args[0], dims),
                    )
                cascaded_transpose_or_permute_ops[-1].replace_all_uses_with(new_permute)

            # Now erase the chain
            for tp in reversed(cascaded_transpose_or_permute_ops):
                graph.erase_node(tp)

        graph_module.recompile()

    def call(self, graph_module: torch.fx.GraphModule) -> PassResult:
        self.fuse_cascaded_transpose_or_permute_ops(graph_module)
        result = super().call(graph_module)
        return result


@register_cadence_pass(CadencePassAttribute(opt_level=1))
class FuseCascadedViewOps(ExportPass):
    """
    Fuse a cascaded chain of view ops
    """

    def fuse_cascaded_view_ops(self, graph_module: torch.fx.GraphModule):
        view_target = exir_ops.edge.aten.view_copy.default
        for view_node in graph_module.graph.find_nodes(
            op="call_function", target=view_target, sort=True
        ):
            input_view = view_node.args[0]
            if input_view.op != "call_function" or input_view.target != view_target:
                continue

            view_node.replace_input_with(input_view, input_view.args[0])

        graph_module.recompile()

    def call(self, graph_module: torch.fx.GraphModule) -> PassResult:
        self.fuse_cascaded_view_ops(graph_module)
        dead_code_elimination_pass(graph_module)
        result = super().call(graph_module)
        return result


class FuseOpPairsAcrossBranchesPass(ExportPass):
    def check_ok_to_fuse(
        self,
        producer: torch.fx.Node,
        consumers: list[torch.fx.Node],
    ) -> bool:
        # Always ok to replace / remove.
        return True

    def can_fuse_for_chain(
        self,
        producer: torch.fx.Node,
        consumer: torch.fx.Node,
        consumer_op_packets: set[EdgeOpOverloadPacket],
    ) -> bool:
        """
        Returns true if producer and consumer can be fused for a single chain
        (-> producer -> ops -> consumer ->) to (-> ops -> fused_op)
        """
        if (
            isinstance(consumer.target, EdgeOpOverload)
            and get_edge_overload_packet(consumer.target) in consumer_op_packets
        ):
            return True
        return False

    def get_fuse_candidates(
        self,
        producer: torch.fx.Node,
        consumer_op_packets: set[EdgeOpOverloadPacket],
        bypass_ops: set[EdgeOpOverload],
    ) -> list[torch.fx.Node]:
        # Start by iterating over all the users of this node, and check
        # if they are have their target in consumer_op_packets.
        users = deque(producer.users.keys())
        # This holds the list of the user ops that directly (or transitively
        # via view/slice) consume this producer_op_packets, and hence can be removed.
        removal_candidates = []
        while users:
            user = users.popleft()

            # If the user is a bypass op, we bypass it, and examine
            # its users instead for consumer_op_packets.
            if user.target in bypass_ops:
                users.extend(list(user.users.keys()))
            elif self.can_fuse_for_chain(producer, user, consumer_op_packets):
                removal_candidates.append(user)
            else:
                removal_candidates.clear()
                break
        return removal_candidates

    def find_and_fuse(
        self,
        graph_module: torch.fx.GraphModule,
        producer_op_packets: set[EdgeOpOverloadPacket],
        consumer_op_packets: set[EdgeOpOverloadPacket],
        bypass_ops: set[EdgeOpOverload],
    ) -> None:
        for node in graph_module.graph.nodes:
            # We are only interested in ops that have overload target in
            # producer_op.
            if not (
                isinstance(node.target, EdgeOpOverload)
                and get_edge_overload_packet(node.target) in producer_op_packets
            ):
                continue

            removal_candidates = self.get_fuse_candidates(
                node, consumer_op_packets, bypass_ops
            )

            if len(removal_candidates) == 0:
                # No candidates found.
                continue

            if not self.check_ok_to_fuse(node, removal_candidates):
                # Not ok to remove quant-dequant pairs or replace with requantize.
                continue

            self.fuse(node, removal_candidates, graph_module)

        graph_module.recompile()

    def get_fused_node(
        self,
        producer: torch.fx.Node,
        consumer: torch.fx.Node,
        graph_module: torch.fx.GraphModule,
    ) -> torch.fx.Node:
        return consumer

    def fuse(
        self,
        node: torch.fx.Node,
        removal_candidates: list[torch.fx.Node],
        graph_module: torch.fx.GraphModule,
    ) -> None:
        # Replace all the uses of the producer op with it's input.
        node.replace_all_uses_with(cast(torch.fx.Node, node.args[0]))
        graph_module.graph.erase_node(node)

        # Iterate over all the removal candidates (quantize op users) and generate replacements.
        for rnode in removal_candidates:
            rnode.replace_all_uses_with(self.get_fused_node(node, rnode, graph_module))
            graph_module.graph.erase_node(rnode)


@register_cadence_pass(CadencePassAttribute(opt_level=1))
class FuseQuantDequantToRequantizePass(FuseOpPairsAcrossBranchesPass):
    """
    Fuse dequantize-quantize op pairs to a single requantize op.
    For the special case where quant params match, this will remove
    both dequant and quant ops.
    """

    # A list of ops that can be bypassed when looking for a
    # dequantize->quantize chain
    bypass_ops: set[EdgeOpOverload] = {
        exir_ops.edge.aten.slice_copy.Tensor,
        exir_ops.edge.aten.view_copy.default,
        exir_ops.edge.aten.clone.default,
        exir_ops.edge.aten.transpose_copy.int,
        exir_ops.edge.aten.permute_copy.default,
    }

    quantize_op_packets: set[EdgeOpOverloadPacket] = {
        exir_ops.edge.cadence.quantize_per_tensor,
        exir_ops.edge.quantized_decomposed.quantize_per_tensor,
    }
    dequantize_op_packets: set[EdgeOpOverloadPacket] = {
        exir_ops.edge.cadence.dequantize_per_tensor,
        exir_ops.edge.quantized_decomposed.dequantize_per_tensor,
    }

    def __init__(
        self, allow_requantize: bool = True, force_quant_dequant_fusion: bool = False
    ) -> None:
        super().__init__()
        self.allow_requantize: bool = allow_requantize
        self.force_quant_dequant_fusion: bool = force_quant_dequant_fusion

    def _pkg_name_match(self, node1: torch.fx.Node, node2: torch.fx.Node) -> bool:
        # pyre-ignore[16]: Item `typing.Callable` has no attribute `_op`
        return node1.target._op.namespace == node2.target._op.namespace

    def can_fuse_for_chain(
        self,
        producer: torch.fx.Node,
        consumer: torch.fx.Node,
        consumer_op_packets: set[EdgeOpOverloadPacket],
    ) -> bool:
        return super().can_fuse_for_chain(
            producer, consumer, consumer_op_packets
        ) and self._pkg_name_match(producer, consumer)

    def _create_requantize_node(
        self,
        in_tensor: torch.fx.Node,
        in_scale: float,
        in_zero_point: int,
        out_scale: float,
        out_zero_point: int,
        out_dtype: torch.dtype,
        graph: torch.fx.Graph,
    ) -> torch.fx.Node:
        return graph.call_function(
            exir_ops.edge.cadence.requantize.per_tensor,
            args=(
                in_tensor,
                in_scale,
                in_zero_point,
                out_scale,
                out_zero_point,
                out_dtype,
            ),
        )

    def _quant_params_match(self, node1: torch.fx.Node, node2: torch.fx.Node) -> bool:
        return node1.args[1:] == node2.args[1:]

    def check_ok_to_fuse(
        self,
        producer: torch.fx.Node,
        consumers: list[torch.fx.Node],
    ) -> bool:
        """Check if all node-user pairs are nops or are ok to replace with requant."""
        for rnode in consumers:
            if self.allow_requantize or self._quant_params_match(producer, rnode):
                # Cannot remove quant-dequant pair if quant params don't match and requantize
                # is not allowed.
                continue
            return False
        return True

    def get_fused_node(
        self,
        producer: torch.fx.Node,
        consumer: torch.fx.Node,
        graph_module: torch.fx.GraphModule,
    ) -> torch.fx.Node:
        in_scale, in_zero_point = producer.args[1:3]
        in_tensor, out_scale, out_zero_point, _, _, out_dtype = consumer.args
        if in_scale == out_scale and in_zero_point == out_zero_point:
            # If the quant params match, we can remove both dequantize-quantize ops.
            return cast(torch.fx.Node, consumer.args[0])

        assert (
            self.allow_requantize
        ), f"Found {producer=} {in_scale=} {in_zero_point=} | {consumer=} {out_scale=} {out_zero_point=}"

        with graph_module.graph.inserting_before(consumer):
            requantize_node = self._create_requantize_node(
                in_tensor=cast(torch.fx.Node, consumer.args[0]),
                in_scale=cast(float, in_scale),
                in_zero_point=cast(int, in_zero_point),
                out_scale=cast(float, out_scale),
                out_zero_point=cast(int, out_zero_point),
                out_dtype=cast(torch.dtype, out_dtype),
                graph=graph_module.graph,
            )
        return requantize_node

    def call(self, graph_module: torch.fx.GraphModule) -> PassResult:
        # Remove any dequantize op that has only quantize ops as its users.
        self.find_and_fuse(
            graph_module,
            producer_op_packets=self.dequantize_op_packets,
            consumer_op_packets=self.quantize_op_packets,
            bypass_ops=self.bypass_ops,
        )
        # Remove any quantize op that has only dequantze ops as its users.
        self.find_and_fuse(
            graph_module,
            producer_op_packets=self.quantize_op_packets,
            consumer_op_packets=self.dequantize_op_packets,
            # Do not requantize for quantize-dequantize pairs as this is not guaranteed
            # to be better for performance/memory.
            # Only fuse if all users of quant are dequant.
            bypass_ops=(
                self.bypass_ops
                if self.force_quant_dequant_fusion
                else {exir_ops.edge.aten.view_copy.default}
            ),
        )
        result = super().call(graph_module)
        return result


@register_cadence_pass(CadencePassAttribute(opt_level=1))
class FuseMulScalarIntoDequantPass(ExportPass):
    """
    Looks for the pattern where aten.mul.Scalar is multiplying the
     outputs of dequantize. If found, updates the dequant scale
    to reflect the multiplication and removes the mul node.
    """

    def attempt_fusion(
        self, graph_module: torch.fx.GraphModule, node: torch.fx.Node
    ) -> None:
        if node.target not in {
            exir_ops.edge.quantized_decomposed.dequantize_per_tensor.default,
            exir_ops.edge.cadence.dequantize_per_tensor.default,
        }:
            return

        # ensure that the single user of dequant is aten.mul.Scalar
        user = list(node.users.keys())[0]
        if len(node.users) != 1 or user.target != exir_ops.edge.aten.mul.Scalar:
            return

        # ensure that the other arg to mul is a node (i.e. not a constant)
        if len(user.args) > 1 and isinstance(user.args[1], torch.fx.Node):
            return

        new_deq_args = list(node.args)
        assert isinstance(node.args[1], Number)
        assert isinstance(user.args[1], Number)
        # pyre-ignore[58]: Unsupported operand *
        new_deq_args[1] = node.args[1] * user.args[1]

        logging.debug(
            f"Fused {node} and {user} into {node}. Updated scale from {node.args[1]} to {new_deq_args[1]}"
        )

        user.replace_all_uses_with(node)
        node.args = tuple(new_deq_args)

        graph_module.graph.erase_node(user)

        graph_module.recompile()

    def call(self, graph_module: torch.fx.GraphModule) -> PassResult:
        for node in graph_module.graph.nodes:
            self.attempt_fusion(graph_module, node)
        result = super().call(graph_module)
        return result


@register_cadence_pass(CadencePassAttribute(opt_level=1))
class FuseMulTensorIntoQuantPass(ExportPass):
    """
    Looks for the pattern where aten.mul.Tensor is followed by quant node.
    If found, updates the quant scale to reflect the multiplication and
    removes the mul node.
    """

    def attempt_fusion(
        self, graph_module: torch.fx.GraphModule, mul_node: torch.fx.Node
    ) -> None:
        if len(mul_node.args) != 2 or len(mul_node.users) != 1:
            return

        first_arg = cast(torch.fx.Node, mul_node.args[0])
        second_arg = cast(torch.fx.Node, mul_node.args[1])

        input_node = first_arg
        full_node = second_arg
        if second_arg.target == exir_ops.edge.aten.full.default:
            # Most common case, nothing to change.
            pass
        elif first_arg.target == exir_ops.edge.aten.full.default:
            # Input and full nodes are swapped.
            full_node = first_arg
            input_node = second_arg
        else:
            # Full node is not found, skip.
            return

        # Ensure that the mul op does not do any broadcasting.
        if input_node.meta["val"].shape != mul_node.meta["val"].shape:
            return

        mul_user = list(mul_node.users.keys())[0]

        # Ensure only the expected quant ops are using the current mul op.
        if mul_user.target not in {
            exir_ops.edge.quantized_decomposed.quantize_per_tensor.default,
            exir_ops.edge.cadence.quantize_per_tensor.default,
        }:
            return

        quant_node = mul_user

        # Calculate the new scale value.
        old_scale = quant_node.args[1]
        assert isinstance(old_scale, (int, float))
        mul_scalar = full_node.args[1]
        assert isinstance(mul_scalar, (int, float))
        """ The reason why we divide old scale by the mul value to get a new scale:
            y = x * mul_scalar
            q = zp + y / old_scale
            q = zp + x * mul_scalar / old_scale
            new_scale = old_scale / mul_scalar
            q = zp + x / new_scale
        """
        new_scale = float(old_scale) / float(mul_scalar)

        logging.debug(
            f"Fused {mul_node} and {full_node} into {quant_node}. Updated scale from {quant_node.args[1]} to {new_scale}"
        )

        # Update quant node input and scale.
        old_quant_input = cast(torch.fx.Node, quant_node.args[0])
        new_quant_input = cast(torch.fx.Node, mul_node.args[0])
        quant_node.replace_input_with(old_quant_input, new_quant_input)
        quant_node.update_arg(1, new_scale)

    def call(self, graph_module: torch.fx.GraphModule) -> PassResult:
        for node in graph_module.graph.find_nodes(
            op="call_function", target=exir_ops.edge.aten.mul.Tensor
        ):
            self.attempt_fusion(graph_module, node)
        graph_module.graph.eliminate_dead_code()
        return super().call(graph_module)


@register_cadence_pass(CadencePassAttribute(opt_level=1))
class FuseMulTensorIntoDequantPass(ExportPass):
    """
    Looks for the pattern where aten.mul is multiplying the outputs of dequantize
    and aten.full, or vice versa. If found, updates the dequant scale to reflect
    the multiplication and removes the full and mul nodes.
    """

    def attempt_fusion(
        self, graph_module: torch.fx.GraphModule, node: torch.fx.Node
    ) -> None:
        if node.target != exir_ops.edge.aten.mul.Tensor:
            return

        # ensure that one of the args to mul is dequantize and the other is aten.full
        dequant_nodes = [
            arg
            for arg in node.args
            if isinstance(arg, torch.fx.Node)
            and isinstance(arg.target, EdgeOpOverload)
            and get_edge_overload_packet(arg.target)
            in (
                exir_ops.edge.quantized_decomposed.dequantize_per_tensor,
                exir_ops.edge.cadence.dequantize_per_tensor,
            )
        ]
        multiplier_nodes = [
            arg
            for arg in node.args
            if isinstance(arg, torch.fx.Node)
            and arg.target == exir_ops.edge.aten.full.default
        ]

        if len(dequant_nodes) != 1 or len(multiplier_nodes) != 1:
            return

        deq_node = dequant_nodes[0]
        mplier_node = multiplier_nodes[0]

        # ensure that dequant and full don't have any other users
        if len(deq_node.users) > 1 or len(mplier_node.users) > 1:
            return

        new_deq_args = list(deq_node.args)
        assert isinstance(deq_node.args[1], Number)
        assert isinstance(mplier_node.args[1], Number)
        # pyre-ignore[58]: Unsupported operand *
        new_deq_args[1] = deq_node.args[1] * mplier_node.args[1]

        logging.debug(
            f"Fused {node} and {mplier_node} into {deq_node}. Updated scale from {deq_node.args[1]} to {new_deq_args[1]}"
        )

        node.replace_all_uses_with(deq_node)
        deq_node.args = tuple(new_deq_args)

        graph_module.graph.erase_node(node)
        graph_module.graph.erase_node(mplier_node)
        graph_module.recompile()

    def call(self, graph_module: torch.fx.GraphModule) -> PassResult:
        for node in graph_module.graph.nodes:
            self.attempt_fusion(graph_module, node)
        result = super().call(graph_module)
        return result


@register_cadence_pass(CadencePassAttribute(opt_level=1))
class FuseTransposeOrPermuteOpPairsPass(FuseOpPairsAcrossBranchesPass):
    """
    Fuse transpose or permute op pairs to a single view op.
    (transpose or permutation) -> (quant or dequant) -> (transpose or permutation)
    This happens when op2(op1) == identity, modulo unitary dimensions.
    'unitary dimensions' example: a tensor of shape [1, 5, 30] is equivalent (in memory) to [5, 1, 30]
    so transpose(1, 2) then transpose(0, 2) is a pseudo identity and should be fused.
    """

    # A list of ops that can be bypassed when looking for a
    # dequantize->quantize chain
    bypass_ops: set[EdgeOpOverload] = {
        exir_ops.edge.cadence.quantize_per_tensor.default,
        exir_ops.edge.quantized_decomposed.quantize_per_tensor.default,
        exir_ops.edge.quantized_decomposed.quantize_per_channel.default,
        exir_ops.edge.cadence.dequantize_per_tensor.default,
        exir_ops.edge.quantized_decomposed.dequantize_per_tensor.default,
        exir_ops.edge.quantized_decomposed.dequantize_per_channel.default,
        exir_ops.edge.cadence.quantized_relu.per_tensor,
    }

    def can_fuse_for_chain(
        self,
        producer: torch.fx.Node,
        consumer: torch.fx.Node,
        consumer_op_packets: set[EdgeOpOverloadPacket],
    ) -> bool:
        if not super().can_fuse_for_chain(producer, consumer, consumer_op_packets):
            return False

        # checking that permut2(permut1(identity)) == identity, modulo unitary dimensions
        input_shape = cast(torch.fx.Node, producer.args[0]).meta["val"].shape
        ident_dims = list(range(len(input_shape)))
        # this mapping helps to handle both transpose and permutations
        f: dict[Any, Callable] = {
            exir_ops.edge.aten.transpose_copy.int: get_transposed_dims,
            exir_ops.edge.aten.permute_copy.default: get_permuted_dims,
        }
        in_dims = f[producer.target](producer, ident_dims)
        out_dims = f[consumer.target](consumer, in_dims)
        # Filtering out unitary dimensions
        non_unit_ident_dims = [dim for dim in ident_dims if input_shape[dim] != 1]
        non_unit_out_dims = [dim for dim in out_dims if input_shape[dim] != 1]
        return non_unit_out_dims == non_unit_ident_dims

    def get_fused_node(
        self,
        producer: torch.fx.Node,
        consumer: torch.fx.Node,
        graph_module: torch.fx.GraphModule,
    ) -> torch.fx.Node:
        # This step is important because of how we can fuse transpositions that are not perfectly
        # reverse one of another but will be fused if there are unitary dimensions.
        # The fused operation must have the same output shape as the consumer.
        output_shape = consumer.meta["val"].shape
        with graph_module.graph.inserting_after(consumer):
            view = graph_module.graph.call_function(
                exir_ops.edge.aten.view_copy.default,
                (consumer.args[0], output_shape),
                {},
            )
        return view

    def call(self, graph_module: torch.fx.GraphModule) -> PassResult:
        # Remove any transpose/permutation op pair that cancel each other.
        self.find_and_fuse(
            graph_module,
            producer_op_packets={
                exir_ops.edge.aten.transpose_copy,
                exir_ops.edge.aten.permute_copy,
            },
            consumer_op_packets={
                exir_ops.edge.aten.transpose_copy,
                exir_ops.edge.aten.permute_copy,
            },
            bypass_ops=self.bypass_ops,
        )
        result = super().call(graph_module)
        return result


@register_cadence_pass(CadencePassAttribute(opt_level=1))
class FuseFullThenReshapePass(ExportPass):
    """
    A pass that fuses a chain of full and reshape-like operations into a single full operation.
    """

    fusion_candidates: set[EdgeOpOverload] = {
        exir_ops.edge.aten.transpose_copy.int,
        exir_ops.edge.aten.permute_copy.default,
        exir_ops.edge.aten.view_copy.default,
    }

    def call_operator(
        self,
        op,
        args: tuple[Argument, ...],
        kwargs: dict[str, Argument],
        meta: NodeMetadata,
    ) -> ProxyValue:
        if op not in self.fusion_candidates:
            return super().call_operator(op, args, kwargs, meta)

        full_node = cast(ProxyValue, args[0]).node
        if not (
            full_node.op == "call_function"
            and full_node.target == exir_ops.edge.aten.full.default
        ):
            # full -> self.fusion_candidates.
            return super().call_operator(op, args, kwargs, meta)

        fill_value = full_node.args[1]
        return super().call_operator(
            exir_ops.edge.aten.full.default,
            (
                meta["val"].shape,
                fill_value,
            ),
            {"dtype": meta["val"].dtype},
            meta,
        )

    def call(self, graph_module: torch.fx.GraphModule) -> PassResult:
        graph_module = super().call(graph_module).graph_module
        graph_module.graph.eliminate_dead_code()
        return PassResult(graph_module, True)


class CadenceFuseOpsInGraph:
    passes = [
        FuseMMWithAdd,
        FuseBatchNormWithConv,
        FuseQuantizedBatchNormWithConv,
        FuseCascadedTransposeOrPermuteOps,
        FuseCascadedViewOps,
        FuseQuantDequantToRequantizePass,
        FuseMulTensorIntoQuantPass,
        FuseMulTensorIntoDequantPass,
        FuseMulScalarIntoDequantPass,
        FuseFullThenReshapePass,
        FuseTransposeOrPermuteOpPairsPass,
    ]
