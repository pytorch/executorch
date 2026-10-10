# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-unsafe


# This file contains all the functions that reorder ops in the graph module.


# The Cadence pass subclasses below reference torch.ops.cadence.* at class
# definition, so the op library must be registered first.
import executorch.backends.cadence.aot.ops_registrations  # noqa: F401
import torch
import torch.fx
from executorch.backends.cadence.aot.compiler_utils import get_placeholders
from executorch.backends.cadence.aot.pass_utils import RemoveOrReplacePassInterface
from executorch.backends.transforms.advance_quantize_op_above_def_chain import (
    AdvanceQuantizeOpAboveDefChainPass as _SharedAdvanceQuantizeOpAboveDefChainPass,
)
from executorch.backends.transforms.advance_quantize_op_above_def_in_branch import (
    AdvanceQuantizeOpAboveDefInBranchPass as _SharedAdvanceQuantizeOpAboveDefInBranchPass,
)
from executorch.backends.transforms.move_permute_after_concat import (
    MovePermuteAfterConcat as _SharedMovePermuteAfterConcat,
)
from executorch.backends.transforms.move_slice_before_permute import (
    MoveSliceBeforePermutePass as _SharedMoveSliceBeforePermutePass,
)
from executorch.backends.transforms.move_slice_before_view import (
    MoveSliceBeforeViewPass as _SharedMoveSliceBeforeViewPass,
)
from executorch.backends.transforms.postpone_dequantize_op_below_use_chain import (
    PostponeDequantizeOpBelowUseChainPass as _SharedPostponeDequantizeOpBelowUseChainPass,
)
from executorch.backends.transforms.postpone_permute_below_squeeze_view import (
    PostponePermuteOpBelowSqueezeOrUnsqueezeLikeView as _SharedPostponePermuteOpBelowSqueezeOrUnsqueezeLikeView,
)
from executorch.backends.transforms.propagate_slice import (
    PropagateSlice as _SharedPropagateSlice,
)
from executorch.backends.transforms.split_dequantized_cat import (
    SplitDequantizedCatPass as _SharedSplitDequantizedCatPass,
)
from executorch.exir.dialects._ops import ops as exir_ops
from executorch.exir.dialects.edge._ops import EdgeOpOverload


class AdvanceQuantizeOpAboveDefInBranchPass(
    _SharedAdvanceQuantizeOpAboveDefInBranchPass
):
    quantize_op_packets: set[object] = {
        torch.ops.quantized_decomposed.quantize_per_tensor,
        exir_ops.edge.quantized_decomposed.quantize_per_tensor,
        torch.ops.cadence.quantize_per_tensor,
        exir_ops.edge.cadence.quantize_per_tensor,
    }


class AdvanceQuantizeOpAboveDefChainPass(_SharedAdvanceQuantizeOpAboveDefChainPass):
    quantize_op_packets: set[object] = {
        exir_ops.edge.quantized_decomposed.quantize_per_tensor,
        torch.ops.quantized_decomposed.quantize_per_tensor,
        exir_ops.edge.cadence.quantize_per_tensor,
        torch.ops.cadence.quantize_per_tensor,
    }


class PostponeDequantizeOpBelowUseChainPass(
    _SharedPostponeDequantizeOpBelowUseChainPass
):
    quantize_op_packets: set[object] = {
        exir_ops.edge.quantized_decomposed.quantize_per_tensor,
        exir_ops.edge.quantized_decomposed.quantize_per_channel,
        exir_ops.edge.cadence.quantize_per_tensor,
    }
    dequantize_packet_to_overload: dict[object, str] = {
        exir_ops.edge.quantized_decomposed.dequantize_per_tensor: "default",
        exir_ops.edge.quantized_decomposed.dequantize_per_channel: "default",
        exir_ops.edge.cadence.dequantize_per_tensor: "default",
    }


class SinkOpsCloserToUsePass(RemoveOrReplacePassInterface):
    """
    Assume that the dequantize op D = dequantize(I) has only a single user.
    If the current graph looks like
    I = ...;
    D = dequantize(I);
    ...
    Y = use(D);
    then we can postpone the dequantize op closer to its use, and convert the
    graph to:
    I = ...;
    ...
    D = dequantize(I);
    Y = use(D);

    The transformation is valid since D had a single user. The benfit comes from
    the fact that now we have I in the live range instead of D, which has a
    much smaller size.
    """

    @property
    def targets(self) -> list[EdgeOpOverload]:
        return [
            exir_ops.edge.aten.dequantize,
            exir_ops.edge.quantized_decomposed.dequantize_per_tensor.default,
            exir_ops.edge.quantized_decomposed.dequantize_per_channel.default,
            exir_ops.edge.cadence.dequantize_per_tensor.default,
        ]

    def maybe_remove_or_replace(self, node: torch.fx.Node) -> bool:
        # The sinkable node must have a single user
        users = list(node.users.keys())
        if len(users) != 1:
            return False

        # Insert the dequant node just before its user
        with node.graph.inserting_before(users[0]):
            # Target is guaranteed to be a callable since it's from our targets list
            target_callable = node.target
            assert callable(target_callable), "Target must be callable"
            new_node = node.graph.call_function(
                target_callable, args=node.args, kwargs=node.kwargs
            )
            new_node.meta = node.meta
        node.replace_all_uses_with(new_node)
        node.graph.erase_node(node)

        return True


class HoistOpsCloserToDefPass(RemoveOrReplacePassInterface):
    """
    Assume that the input I to a quantize op Q = quantize(I) has only a single
    use, the quantize node itself.
    If the current graph looks like
    I = ...;
    ...
    Q = quantize(I);
    X = use(Q);
    then we can hoist the quantize op closer to its def, and convert the
    graph to:
    I = ...;
    Q = quantize(I);
    ...
    X = use(Q);

    The transformation is valid since I had a single user. The benefit comes from
    the fact that now we have Q in the live range instead of I, which has a
    much smaller size. The same transformation also applies to slice/select op.
    """

    @property
    def targets(self) -> list[EdgeOpOverload]:
        return [
            exir_ops.edge.quantized_decomposed.quantize_per_tensor.default,
            exir_ops.edge.cadence.quantize_per_tensor.default,
            exir_ops.edge.aten.slice_copy.Tensor,
            exir_ops.edge.aten.select_copy.int,
        ]

    def maybe_remove_or_replace(self, node: torch.fx.Node) -> bool:
        def_node = node.args[0]
        if not isinstance(def_node, torch.fx.Node):
            return False

        # The def node must have a single user
        users = list(def_node.users.keys())
        if len(users) != 1:
            return False

        # Get the node args as list
        args = list(node.args)

        # If the graph has placeholders, we do not want to hoist above the
        # last placeholder. Otherwise we will shrink the live range of the
        # def_node considerably, which could lead to reuse of input memory.
        insertion_point = (
            get_placeholders(node.graph)[-1]
            if def_node.op == "placeholder"
            else def_node
        )

        # If the node is quantize_per_channel, we need to hoist the scale
        # and zero_point tensors as well.
        if (
            node.target
            == exir_ops.edge.quantized_decomposed.quantize_per_channel.default
        ):
            scale, zero_point = args[1], args[2]
            if not isinstance(scale, torch.fx.Node) or not isinstance(
                zero_point, torch.fx.Node
            ):
                return False
            with node.graph.inserting_after(insertion_point):
                zero_point_copy = node.graph.node_copy(zero_point)
                scale_copy = node.graph.node_copy(scale)
                args[1], args[2] = scale_copy, zero_point_copy
                insertion_point = zero_point_copy

        # Insert the quant node just after insertion_point
        with node.graph.inserting_after(insertion_point):
            # Target is guaranteed to be a callable since it's from our targets list
            target_callable = node.target
            assert callable(target_callable), "Target must be callable"
            new_node = node.graph.call_function(
                target_callable, args=tuple(args), kwargs=node.kwargs
            )
            new_node.meta = node.meta
        node.replace_all_uses_with(new_node)
        node.graph.erase_node(node)

        return True


class PostponePermuteOpBelowSqueezeOrUnsqueezeLikeView(
    _SharedPostponePermuteOpBelowSqueezeOrUnsqueezeLikeView
):
    pass


class MovePermuteAfterConcat(_SharedMovePermuteAfterConcat):
    pass


class MoveSliceBeforePermutePass(_SharedMoveSliceBeforePermutePass):
    pass


class MoveSliceBeforeViewPass(_SharedMoveSliceBeforeViewPass):
    pass


class PropagateSlice(_SharedPropagateSlice):
    quant_unary_targets: list[EdgeOpOverload] = [
        exir_ops.edge.quantized_decomposed.quantize_per_tensor.default,
        exir_ops.edge.cadence.quantize_per_tensor.default,
        exir_ops.edge.quantized_decomposed.dequantize_per_tensor.default,
        exir_ops.edge.cadence.dequantize_per_tensor.default,
    ]


class SplitDequantizedCatPass(_SharedSplitDequantizedCatPass):
    quantize_op_packets: set[object] = {
        exir_ops.edge.quantized_decomposed.quantize_per_tensor,
        exir_ops.edge.cadence.quantize_per_tensor,
    }
    dequantize_op_packets: set[object] = {
        exir_ops.edge.quantized_decomposed.dequantize_per_tensor,
        exir_ops.edge.cadence.dequantize_per_tensor,
    }


# The following class consolidates functions to reoder ops (i.e., either hoist
# or sink some ops in the graph).
class CadenceReorderOpsInGraph:
    passes = [
        # Hoist/sink nodes closer to their SSA def/use
        HoistOpsCloserToDefPass,
        SinkOpsCloserToUsePass,
        MovePermuteAfterConcat,
        # For quantize/dequantize ops, move them above/below their def chain.
        # This is a more aggressive optimization than just hoisting/sinking
        # nodes closer to their def/use.
        AdvanceQuantizeOpAboveDefChainPass,
        PostponeDequantizeOpBelowUseChainPass,
        # These passes work on branches instead of linear chains to advance
        # quantize op beyond their def.
        AdvanceQuantizeOpAboveDefInBranchPass,
    ]
