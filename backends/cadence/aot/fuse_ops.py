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
from typing import Any, Callable, cast, Optional

# Import these for the cadence function signatures.
import executorch.backends.cadence.aot.ops_registrations  # noqa: F401
import torch
import torch.fx
from executorch.backends.cadence.aot.compiler_utils import (
    broadcastable,
    get_cascaded_ops,
    get_permuted_dims,
    get_scale,
    get_tensor_from_attr,
    get_transposed_dims,
    get_zero_point,
)
from executorch.backends.cadence.aot.pass_utils import (
    CadencePassAttribute,
    get_arg,
    register_cadence_pass,
    RemoveOrReplacePassInterface,
)
from executorch.backends.cadence.aot.utils import get_edge_overload_packet
from executorch.exir.dialects._ops import ops as exir_ops
from executorch.exir.dialects.edge._ops import EdgeOpOverload, EdgeOpOverloadPacket
from executorch.exir.pass_base import ExportPass, PassResult
from torch.nn.utils.fusion import fuse_conv_bn_weights


def get_tensor_arg(node: torch.fx.Node, arg_name: str) -> torch.Tensor:
    graph_module = node.graph.owning_module
    tensor = get_tensor_from_attr(
        graph_module, get_arg(node, arg_name, torch.fx.Node)
    )
    assert isinstance(tensor, torch.Tensor), f"{arg_name} must be present"
    return tensor


@register_cadence_pass(CadencePassAttribute(opt_level=1))
class FuseMMWithAdd(RemoveOrReplacePassInterface):
    """
    Fuses mm -> add patterns into addmm.

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

    Handle this case as well.
    """

    @property
    def targets(self) -> list[EdgeOpOverload]:
        return [exir_ops.edge.aten.mm.default]

    def _is_view_node(self, node: torch.fx.Node) -> bool:
        """Return true if the node is a view node."""
        return node.target == exir_ops.edge.aten.view_copy.default

    def maybe_remove_or_replace(self, node: torch.fx.Node) -> bool:
        """
        Try to fuse this mm node with a following add node.

        Returns True if fusion was performed, False otherwise.
        """
        # We want to discover a chain of mm -> add, or mm -> view -> add.
        # Only proceed if the current node is an mm node, and has only one
        # user/successor.
        if len(node.users) != 1:
            return False

        # Our addmm implementation computes (mat1 * mat2 + bias). So the
        # addmm node in the graph should have three args. We collectively
        # term mat1 and mat2 as mm_arg since they are the args of mm node,
        # and bias as bias_arg.
        # Since we already have discovered the mm node, we can get mat1 and
        # mat2 by iterating over its args. So the current node is mm_arg.
        # bias_arg can be found once we discover the add op that consumes
        # the output of this mm node. Our next step is to find the add op.
        mm_node = node
        user = list(node.users.keys())[0]

        # intermediate_view is True when the fusion case is mm -> view -> add
        intermediate_view = False
        view_node = None

        # Check if the single user of the mm node is a view op. If so, our
        # graph could potentially have mm -> view -> add. We need to skip
        # the view op, and check if its successor is the add op. One condition
        # we need to verify is that the view op must have only a single user
        # (the add op).
        if self._is_view_node(user) and len(user.users) == 1:
            # We want to maintain two invariants:
            # (1) 'user' is a potential add op that will get fused with the
            #     mm node;
            # (2) 'view_node' is the intermediate view node (if present)
            intermediate_view = True
            view_node = user
            user = list(view_node.users.keys())[0]

        # Check if 'user' is an add op. We also want to ensure that the add op
        # has only one user, otherwise we will not be able to eliminate add op
        # post fusion.
        if user.target != exir_ops.edge.aten.add.Tensor or len(user.users) != 1:
            return False

        # At this point, we have found an mm and an add node that we can
        # fuse together. One arg of the add op is either mm_node or view_node.
        # Find the other arg, and tag it as bias_arg.
        assert len(user.args) == 2
        add_input = view_node if intermediate_view else mm_node
        bias_arg = user.args[1] if user.args[0] == add_input else user.args[0]

        # As a last check, make sure that we can broadcast the bias tensor
        # to the output of mm.
        mm_shape = mm_node.meta.get("val")
        bias_shape = (
            bias_arg.meta.get("val") if isinstance(bias_arg, torch.fx.Node) else None
        )

        if mm_shape is None or bias_shape is None:
            return False

        mm_arg_shape = mm_shape.shape
        bias_arg_shape = bias_shape.shape

        if not broadcastable(mm_arg_shape, bias_arg_shape) or len(bias_arg_shape) > 2:
            return False

        graph = node.graph

        # Create a new addmm node, and insert it before add node.
        with graph.inserting_before(user):
            addmm_node = graph.call_function(
                exir_ops.edge.aten.addmm.default,
                args=(bias_arg, mm_node.args[0], mm_node.args[1]),
            )
            addmm_node.meta = user.meta

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
        if len(addmm_node.users) == 0:
            return False

        addmm_user = list(addmm_node.users.keys())[0]
        if intermediate_view and not self._is_view_node(addmm_user):
            assert view_node is not None
            # Create a view node that correctly reshapes the output of addmm
            # to match the output shape of the add node.
            # The correct shape is held by 'view_node', which points to the
            # view op in mm -> view -> add chain.
            with graph.inserting_before(addmm_user):
                view_copy_node = graph.node_copy(view_node)
            # Any uses of addmm are replaced with this view_copy node.
            addmm_node.replace_all_uses_with(view_copy_node)
            # Now we massage the args of the view_copy node, so that it takes
            # view of addmm node.
            view_args = list(view_copy_node.args)
            view_args[0] = addmm_node
            view_copy_node.args = tuple(view_args)

        return True


@register_cadence_pass(CadencePassAttribute(opt_level=1))
class FuseBatchNormWithConv(RemoveOrReplacePassInterface):
    """
    This pass fuses a conv op with batchnorm if the following two conditions
    are met:
    1. The only user of conv op should be batchnorm;
    2. Only the first element from the batchnorm output tuple should be used
    in the graph.
    """

    @property
    def targets(self) -> list[EdgeOpOverload]:
        return [exir_ops.edge.aten.convolution.default]

    def __init__(self) -> None:
        super().__init__()
        self.counter = 0

    def _get_batchnorm_user(self, conv_node: torch.fx.Node) -> Optional[torch.fx.Node]:
        """
        Check if conv has a single user that is batch_norm, and all batch_norm
        users only access the first tuple element. Returns the bn node or None.
        """
        if len(conv_node.users) != 1:
            return None

        bn = list(conv_node.users.keys())[0]
        if bn.target != exir_ops.edge.aten.native_batch_norm.default:
            return None

        # All the users of batchnorm node must be getitem ops accessing
        # the first element of the tuple.
        if [(user.target == operator.getitem and user.args[1] == 0) for user in bn.users
        ].count(False):
            return None

        return bn

    def _weights_are_params(
        self, conv_node: torch.fx.Node, bn_node: torch.fx.Node
    ) -> bool:
        """Check that the weights for conv and batchnorm are both get_attr nodes."""
        conv_weight_node = get_arg(conv_node, "weight", torch.fx.Node)
        bn_weight_node = get_arg(bn_node, "weight", torch.fx.Node)
        return all(arg.op == "get_attr" for arg in {conv_weight_node, bn_weight_node})

    def _extract_conv_params(
        self, conv_node: torch.fx.Node
    ) -> tuple[torch.Tensor, Optional[torch.Tensor], bool]:
        """Extract weight, bias, and transpose flag from conv node."""
        conv_weight = get_tensor_arg(conv_node, "weight")
        # conv_bias is truly optional - fusion function handles None
        graph_module = conv_node.graph.owning_module
        conv_bias = get_tensor_from_attr(
            graph_module, cast(Optional[torch.fx.Node], get_arg(conv_node, "bias"))
        )
        transpose = get_arg(conv_node, "transposed", bool)
        return conv_weight, conv_bias, transpose

    def _extract_batchnorm_params(
        self,
        bn_node: torch.fx.Node,
    ) -> tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        float,
    ]:
        """Extract weight, bias, running_mean, running_var, and eps from batchnorm node."""
        assert len(bn_node.args) == 8
        bn_weight = get_tensor_arg(bn_node, "weight")
        bn_bias = get_tensor_arg(bn_node, "bias")
        running_mean = get_tensor_arg(bn_node, "running_mean")
        running_var = get_tensor_arg(bn_node, "running_var")
        eps = get_arg(bn_node, "eps", float)
        return bn_weight, bn_bias, running_mean, running_var, eps

    def _update_graph_with_fused_params(
        self,
        graph_module: torch.fx.GraphModule,
        graph: torch.fx.Graph,
        conv_node: torch.fx.Node,
        bn_node: torch.fx.Node,
        fused_weight: torch.nn.Parameter,
        fused_bias: torch.nn.Parameter,
    ) -> None:
        """Register fused params and update the graph to use them."""
        with graph.inserting_before(conv_node):
            fused_weight_name = f"_fused_with_bn_weight_{self.counter}"
            graph_module.register_parameter(fused_weight_name, fused_weight)
            fused_weight_node = graph.get_attr(fused_weight_name)
            fused_bias_name = f"_fused_with_bn_bias_{self.counter}"
            graph_module.register_parameter(fused_bias_name, fused_bias)
            fused_bias_node = graph.get_attr(fused_bias_name)

        # Update the weight and bias of conv op
        conv_args = list(conv_node.args)
        conv_args[1] = fused_weight_node
        conv_args[2] = fused_bias_node
        conv_node.args = tuple(conv_args)

        # Remove any use of batchnorm from the graph
        for user in bn_node.users:
            assert user.target == operator.getitem
            user.replace_all_uses_with(conv_node)

    def maybe_remove_or_replace(self, node: torch.fx.Node) -> bool:
        graph_module = node.graph.owning_module
        assert graph_module is not None
        graph = node.graph

        # Validate conv-bn pattern
        bn = self._get_batchnorm_user(node)
        if bn is None:
            return False

        if not self._weights_are_params(node, bn):
            return False

        # Extract conv parameters
        conv_weight, conv_bias, transpose = self._extract_conv_params(node)

        # Extract batchnorm parameters
        bn_weight, bn_bias, running_mean, running_var, eps = (
            self._extract_batchnorm_params(bn)
        )

        # Compute fused weights
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

        # Update the graph
        self._update_graph_with_fused_params(
            graph_module, graph, node, bn, fused_weight, fused_bias
        )
        self.counter += 1
        return True


@register_cadence_pass(CadencePassAttribute(opt_level=1))
class FuseQuantizedBatchNormWithConv(RemoveOrReplacePassInterface):
    """
    This pass fuses a quantized::conv op with quantized::batchnorm if the
    following two conditions are met:
    1. The only user of quantized::conv op should be quantized::batchnorm;
    2. The outputs of both ops are quantized with same scale and zero_point
    """

    _CONV_TARGETS = {
        exir_ops.edge.quantized.conv1d.default,
        exir_ops.edge.quantized.conv2d.new,
    }
    _BN_TARGETS = {
        exir_ops.edge.quantized.batch_norm1d.default,
        exir_ops.edge.quantized.batch_norm2d.default,
    }

    def __init__(self) -> None:
        super().__init__()
        self.counter = 0

    @property
    def targets(self) -> list[EdgeOpOverload]:
        return self._CONV_TARGETS

    def _get_batchnorm_user(self, conv_node: torch.fx.Node) -> Optional[torch.fx.Node]:
        """
        Check if conv has a single user that is quantized batch_norm with
        matching output scale/zero_point. Returns the bn node or None.
        """
        if len(conv_node.users) != 1:
            return None

        bn = list(conv_node.users.keys())[0]
        if bn.target not in self._BN_TARGETS:
            return None

        # The outputs of conv and bn must both have same scale and zero_point
        if not math.isclose(
            cast(float, get_arg(conv_node, "output_scale")),
            cast(float, get_arg(bn, "output_scale")),
            rel_tol=1e-05,
            abs_tol=1e-05,
        ):
            return None
        if get_arg(conv_node, "output_zero_point") != get_arg(bn, "output_zero_point"):
            return None

        return bn

    def _unpack_conv_weights(
        self, graph_module: torch.fx.GraphModule, conv_node: torch.fx.Node
    ) -> tuple[torch.Tensor, Optional[torch.Tensor], Any]:
        """
        Unpack quantized conv's packed arguments.
        Returns (weight_tensor, bias_tensor, packed_args).
        """
        conv_packed_arg_node = get_arg(conv_node, "packed_weight", torch.fx.Node)
        assert conv_packed_arg_node.op == "get_attr"
        packed_args = getattr(graph_module, conv_packed_arg_node.target)
        weight_tensor, bias_tensor = packed_args.unpack()

        assert isinstance(weight_tensor, torch.Tensor)
        return weight_tensor, bias_tensor, packed_args
        assert bias_tensor is None or isinstance(bias_tensor, torch.Tensor)

        return weight_tensor, bias_tensor, packed_args

    def _get_weight_quant_params(
        self, weight_tensor: torch.Tensor
    ) -> tuple[bool, Any, torch.Tensor, torch.Tensor, int]:
        """
        Extract quantization parameters from the weight tensor.
        Returns (per_tensor_quantization, dtype, scale, zero_point, axis).
        """
        assert weight_tensor.is_quantized
        per_tensor_quantization = weight_tensor.qscheme() == torch.per_tensor_affine
        dtype = weight_tensor.dtype
        scale = get_scale(weight_tensor)
        zero_point = get_zero_point(weight_tensor, reduce=False)
        axis = 0 if per_tensor_quantization else weight_tensor.q_per_channel_axis()
        return per_tensor_quantization, dtype, scale, zero_point, axis

    def _extract_batchnorm_params(
        self,
        bn_node: torch.fx.Node,
    ) -> tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        float,
    ]:
        """
        Extract weight, bias, mean, var, and eps from quantized batchnorm node.

        Expected schema:
        quantized::batch_norm1d(Tensor qx, Tensor? weight, Tensor? bias,
                               Tensor mean, Tensor var, float eps,
                               float output_scale, int output_zero_point) -> Tensor
        """
        assert len(bn_node.args) == 8

        from executorch.exir.dialects.edge._ops import EdgeOpOverload

        assert isinstance(
            bn_node.target, EdgeOpOverload
        ), f"Expected EdgeOpOverload, got {type(bn_node.target)}"

        # Extract parameters by name (not by index for maintainability)
        # The get_arg function handles normalization of positional args to kwargs
        bn_weight = get_tensor_arg(bn_node, "weight")
        bn_bias = get_tensor_arg(bn_node, "bias")
        running_mean = get_tensor_arg(bn_node, "mean")
        running_var = get_tensor_arg(bn_node, "var")
        eps = get_arg(bn_node, "eps", float)

        return bn_weight, bn_bias, running_mean, running_var, eps

    def _requantize_fused_weight(
        self,
        fused_weight: torch.Tensor,
        per_tensor_quantization: bool,
        dtype: Any,
        scale: torch.Tensor,
        zero_point: torch.Tensor,
        axis: int,
    ) -> torch.Tensor:
        """Requantize the fused weight with the original quantization params."""
        if per_tensor_quantization:
            return torch.quantize_per_tensor(
                fused_weight,
                scale.item(),
                cast(int, zero_point.item()),
                dtype,
            )
        else:
            return torch.quantize_per_channel(
                fused_weight,
                scale,
                zero_point,
                axis,
                dtype,
            )

    def _pack_and_update_graph(
        self,
        graph: torch.fx.Graph,
        conv_node: torch.fx.Node,
        bn_node: torch.fx.Node,
        fused_weight: torch.Tensor,
        fused_bias: torch.Tensor,
        packed_args: Any,
    ) -> None:
        """Pack the fused weights and update the graph."""
        stride = packed_args.stride()
        padding = packed_args.padding()
        dilation = packed_args.dilation()
        groups = packed_args.groups()
        args = (fused_weight, fused_bias, stride, padding, dilation, groups)

        new_packed_args = (
            exir_ops.edge.quantized.conv1d_prepack(*args)
            if conv_node.target == exir_ops.edge.quantized.conv1d.default
            else exir_ops.edge.quantized.conv2d_prepack(*args)
        )

        conv_args = list(conv_node.args)
        conv_args[1] = new_packed_args
        conv_node.args = tuple(conv_args)
        bn_node.replace_all_uses_with(conv_node)
        graph.erase_node(bn_node)

    def maybe_remove_or_replace(self, node: torch.fx.Node) -> bool:
        graph_module = node.graph.owning_module
        assert graph_module is not None
        graph = node.graph

        # Validate quantized conv-bn pattern
        bn = self._get_batchnorm_user(node)
        if bn is None:
            return False

        # Unpack quantized conv weights and get quantization params
        conv_weight, conv_bias, packed_args = self._unpack_conv_weights(
            graph_module, node
        )
        per_tensor_quant, weight_dtype, weight_scale, weight_zero_point, weight_axis = (
            self._get_weight_quant_params(conv_weight)
        )
        conv_weight_dequant = conv_weight.dequantize()

        # Extract batchnorm parameters
        bn_weight, bn_bias, running_mean, running_var, eps = (
            self._extract_batchnorm_params(bn)
        )

        # Compute fused weights
        fused_weight, fused_bias = fuse_conv_bn_weights(
            conv_weight_dequant,
            conv_bias,
            running_mean,
            running_var,
            eps,
            bn_weight,
            bn_bias,
            transpose=False,
        )

        # Requantize fused weight
        fused_weight = self._requantize_fused_weight(
            fused_weight,
            per_tensor_quant,
            weight_dtype,
            weight_scale,
            weight_zero_point,
            weight_axis,
        )

        # Pack and update the graph
        self._pack_and_update_graph(
            graph, node, bn, fused_weight, fused_bias, packed_args
        )
        self.counter += 1
        return True


@register_cadence_pass(CadencePassAttribute(opt_level=1))
class FuseCascadedTransposeOrPermuteOps(RemoveOrReplacePassInterface):
    """
    Fuse a cascaded chain of transpose and permute ops
    """

    transpose_or_permute_target = {
        exir_ops.edge.aten.transpose_copy.int,
        exir_ops.edge.aten.permute_copy.default,
    }

    @property
    def targets(self) -> list[EdgeOpOverload]:
        return list(self.transpose_or_permute_target)

    def maybe_remove_or_replace(self, node: torch.fx.Node) -> bool:
        # Get the cascaded chain of transpose/permute ops starting at node
        cascaded_transpose_or_permute_ops = get_cascaded_ops(
            [node], self.transpose_or_permute_target
        )
        # The chain must have more than 1 node
        if len(cascaded_transpose_or_permute_ops) == 1:
            return False

        # Get shape from node metadata
        val = node.meta.get("val")
        if val is None:
            return False
        out_shape = val.shape
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

        graph = node.graph

        # In case the permute chain cancelled each other, the final dims will
        # be the same as the initial order. In that case, the chain was nop.
        # Otherwise create a new permute op that encompasses the effect of the
        # chain.
        if dims == list(range(out_dims)):
            cascaded_transpose_or_permute_ops[-1].replace_all_uses_with(
                cast(torch.fx.Node, node.args[0])
            )
        else:
            with graph.inserting_before(cascaded_transpose_or_permute_ops[-1]):
                new_permute = graph.call_function(
                    exir_ops.edge.aten.permute_copy.default,
                    args=(node.args[0], dims),
                )
                new_permute.meta = cascaded_transpose_or_permute_ops[-1].meta
            cascaded_transpose_or_permute_ops[-1].replace_all_uses_with(new_permute)

        # Now erase the chain (except the first node which will be handled by the interface)
        for tp in reversed(cascaded_transpose_or_permute_ops[1:]):
            graph.erase_node(tp)

        # Return True to indicate the first node in the chain should be removed
        return True


@register_cadence_pass(CadencePassAttribute(opt_level=1))
class FuseCascadedViewOps(RemoveOrReplacePassInterface):
    """
    Fuse a cascaded chain of view ops
    """

    @property
    def targets(self) -> list[EdgeOpOverload]:
        return [exir_ops.edge.aten.view_copy.default]

    def maybe_remove_or_replace(self, node: torch.fx.Node) -> bool:
        # Check if the input to this view node is also a view node
        input_view = node.args[0]
        if not isinstance(input_view, torch.fx.Node):
            return False

        if (
            input_view.op != "call_function"
            or input_view.target != exir_ops.edge.aten.view_copy.default
        ):
            return False

        # Replace the input of this view node with the input of the cascaded view
        # This effectively "skips" the intermediate view node
        node.replace_input_with(input_view, cast(torch.fx.Node, input_view.args[0]))
        return True


class FuseOpPairsAcrossBranchesPass(ExportPass):
    """
    Base class for passes that fuse op pairs across branches.
    Provides common functionality for finding and fusing producer-consumer chains.
    """

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
    ) -> bool:
        """
        Find and fuse producer-consumer op pairs.

        Returns True if any fusion was performed, False otherwise.
        """
        modified = False
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
            modified = True

        if modified:
            graph_module.recompile()

        return modified

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
        modified = self.find_and_fuse(
            graph_module,
            producer_op_packets=self.dequantize_op_packets,
            consumer_op_packets=self.quantize_op_packets,
            bypass_ops=self.bypass_ops,
        )
        # Remove any quantize op that has only dequantze ops as its users.
        modified |= self.find_and_fuse(
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
        if modified:
            return super().call(graph_module)
        return PassResult(graph_module, False)


@register_cadence_pass(CadencePassAttribute(opt_level=1))
class FuseMulScalarIntoDequantPass(RemoveOrReplacePassInterface):
    """
    Looks for the pattern where aten.mul.Scalar is multiplying the
     outputs of dequantize. If found, updates the dequant scale
    to reflect the multiplication and removes the mul node.
    """

    @property
    def targets(self) -> list[EdgeOpOverload]:
        return [exir_ops.edge.aten.mul.Scalar]

    def maybe_remove_or_replace(self, node: torch.fx.Node) -> bool:
        # Ensure that the single user of dequant is aten.mul.Scalar
        mul_node = node
        input_nodes = mul_node.all_input_nodes
        if len(input_nodes) != 1 or len(input_nodes[0].users) != 1:
            return False

        dequant_node = input_nodes[0]

        if dequant_node.target not in [
            exir_ops.edge.quantized_decomposed.dequantize_per_tensor.default,
            exir_ops.edge.cadence.dequantize_per_tensor.default,
        ]:
            return False

        if len(mul_node.args) <= 1 or isinstance(mul_node.args[1], torch.fx.Node):
            return False

        new_deq_args = list(dequant_node.args)
        assert isinstance(dequant_node.args[1], Number)
        assert isinstance(mul_node.args[1], Number)
        # pyre-ignore[58]: Unsupported operand *
        new_deq_args[1] = dequant_node.args[1] * mul_node.args[1]

        # Replace all uses of mul with the dequant node
        mul_node.replace_all_uses_with(dequant_node)
        # Update the dequant node's args with the new scale
        dequant_node.args = tuple(new_deq_args)

        # Erase the mul node
        mul_node.graph.erase_node(mul_node)

        logging.debug(
            f"Fused {dequant_node} and {mul_node} into {dequant_node}. Updated scale from {dequant_node.args[1]} to {new_deq_args[1]}"
        )
        return True


@register_cadence_pass(CadencePassAttribute(opt_level=1))
class FuseMulTensorIntoQuantPass(RemoveOrReplacePassInterface):
    """
    Looks for the pattern where aten.mul.Tensor is followed by quant node.
    If found, updates the quant scale to reflect the multiplication and
    removes the mul node.
    """

    @property
    def targets(self) -> list[EdgeOpOverload]:
        return [exir_ops.edge.aten.mul.Tensor]

    def maybe_remove_or_replace(self, node: torch.fx.Node) -> bool:

        mul_node = node
        if len(mul_node.users) != 1:
            return False

        user = next(iter(mul_node.users))
        user_input_nodes = user.all_input_nodes
        if len(user_input_nodes) != 1:
            return False

        if user.target not in [
            exir_ops.edge.quantized_decomposed.quantize_per_tensor.default,
            exir_ops.edge.cadence.quantize_per_tensor.default,
        ]:
            return False

        # Alias for readability.
        quant_node = user

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
            return False

        # Ensure that the mul op does not do any broadcasting.
        if input_node.meta["val"].shape != node.meta["val"].shape:
            return False

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

        # Cannot fuse if either value is zero:
        # - mul_scalar == 0 would cause division by zero computing new_scale
        # - old_scale == 0 would result in new_scale = 0, causing division by zero during quantization
        if mul_scalar == 0 or old_scale == 0:
            return False
        new_scale = float(old_scale) / float(mul_scalar)

        logging.debug(
            f"Fused {node} and {full_node} into {quant_node}. Updated scale from {quant_node.args[1]} to {new_scale}"
        )

        # Update quant node input and scale.
        old_quant_input = cast(torch.fx.Node, quant_node.args[0])
        new_quant_input = input_node
        quant_node.replace_input_with(old_quant_input, new_quant_input)
        quant_node.update_arg(1, new_scale)

        return True


@register_cadence_pass(CadencePassAttribute(opt_level=1))
class FuseMulTensorIntoDequantPass(RemoveOrReplacePassInterface):
    """
    Looks for the pattern where aten.mul is multiplying the outputs of dequantize
    and aten.full, or vice versa. If found, updates the dequant scale to reflect
    the multiplication and removes the full and mul nodes.
    """

    @property
    def targets(self) -> list[EdgeOpOverload]:
        return [exir_ops.edge.aten.mul.Tensor]

    def maybe_remove_or_replace(self, node: torch.fx.Node) -> bool:
        # Ensure that one of the args to mul is dequantize and the other is aten.full
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
            return False

        deq_node = dequant_nodes[0]
        mplier_node = multiplier_nodes[0]

        # Ensure that dequant and full don't have any other users
        if len(deq_node.users) > 1 or len(mplier_node.users) > 1:
            return False

        new_deq_args = list(deq_node.args)
        assert isinstance(deq_node.args[1], Number)
        assert isinstance(mplier_node.args[1], Number)
        # pyre-ignore[58]: Unsupported operand *
        new_deq_args[1] = deq_node.args[1] * mplier_node.args[1]

        logging.debug(
            f"Fused {node} and {mplier_node} into {deq_node}. Updated scale from {deq_node.args[1]} to {new_deq_args[1]}"
        )

        # Replace all uses of the mul node with the dequant node
        node.replace_all_uses_with(deq_node)
        # Update the dequant node's args with the new scale
        deq_node.args = tuple(new_deq_args)

        # Erase the mul and full nodes
        node.graph.erase_node(node)
        node.graph.erase_node(mplier_node)

        return True


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
        modified = self.find_and_fuse(
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
        if modified:
            return super().call(graph_module)
        return PassResult(graph_module, False)


@register_cadence_pass(CadencePassAttribute(opt_level=1))
class FuseFullThenReshapePass(RemoveOrReplacePassInterface):
    """
    A pass that fuses a chain of full and reshape-like operations into a single full operation.
    """

    @property
    def targets(self) -> list[EdgeOpOverload]:
        return [
            exir_ops.edge.aten.transpose_copy.int,
            exir_ops.edge.aten.permute_copy.default,
            exir_ops.edge.aten.view_copy.default,
        ]

    def maybe_remove_or_replace(self, node: torch.fx.Node) -> bool:
        # Check if the input to this reshape-like node is a full node
        full_node = node.args[0]
        if not isinstance(full_node, torch.fx.Node):
            return False

        if not (
            full_node.op == "call_function"
            and full_node.target == exir_ops.edge.aten.full.default
        ):
            return False

        # Get the fill value from the full node
        fill_value = full_node.args[1]

        # Get the output shape and dtype from this node's metadata
        val = node.meta.get("val")
        if val is None:
            return False

        output_shape = val.shape
        output_dtype = val.dtype

        graph = node.graph

        # Create a new full node with the final shape
        with graph.inserting_before(node):
            new_full_node = graph.call_function(
                exir_ops.edge.aten.full.default,
                args=(output_shape, fill_value),
                kwargs={"dtype": output_dtype},
            )
            new_full_node.meta = node.meta

        # Replace all uses of the reshape node with the new full node
        node.replace_all_uses_with(new_full_node)

        return True


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
