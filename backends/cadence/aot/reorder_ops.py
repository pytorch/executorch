# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-unsafe


# This file contains all the functions that reorder ops in the graph module.

from collections import defaultdict
from math import prod
from typing import DefaultDict, List, Tuple

import torch
import torch.fx
from executorch.backends.cadence.aot.compiler_utils import get_placeholders, get_shape
from executorch.backends.cadence.aot.pass_utils import (
    CadencePassAttribute,
    get_arg,
    get_overload_packet,
    register_cadence_pass,
    RemoveOrReplacePassInterface,
)
from executorch.backends.cadence.aot.utils import get_edge_overload_packet
from executorch.backends.transforms.postpone_permute_below_squeeze_view import (
    PostponePermuteOpBelowSqueezeOrUnsqueezeLikeView as _SharedPostponePermuteOpBelowSqueezeOrUnsqueezeLikeView,
)
from executorch.exir.dialects._ops import ops as exir_ops
from executorch.exir.dialects.edge._ops import EdgeOpOverload
from executorch.exir.pass_base import ExportPass, PassResult
from executorch.exir.tensor import num_bytes_from_shape_and_dtype

# A list of ops that can be trivially quantized
trivially_quantizable_ops_overloadpkt = {
    exir_ops.edge.aten.chunk,
    exir_ops.edge.aten.clone,
    exir_ops.edge.aten.contiguous,
    exir_ops.edge.aten.expand_copy,
    exir_ops.edge.aten.permute_copy,
    exir_ops.edge.aten.select_copy,
    exir_ops.edge.aten.slice_copy,
    exir_ops.edge.aten.squeeze_copy,
    exir_ops.edge.aten.transpose_copy,
    exir_ops.edge.aten.unfold_copy,
    exir_ops.edge.aten.unsqueeze_copy,
    exir_ops.edge.aten.view_copy,
    torch.ops.aten.chunk,
    torch.ops.aten.clone,
    torch.ops.aten.contiguous,
    torch.ops.aten.expand_copy,
    torch.ops.aten.permute,
    torch.ops.aten.permute_copy,
    torch.ops.aten.select_copy,
    torch.ops.aten.slice,
    torch.ops.aten.slice_copy,
    torch.ops.aten.squeeze,
    torch.ops.aten.squeeze_copy,
    torch.ops.aten.transpose,
    torch.ops.aten.transpose_copy,
    torch.ops.aten.unsqueeze,
    torch.ops.aten.unsqueeze_copy,
    torch.ops.aten.view,
    torch.ops.aten.view_copy,
}

# slice-equivalent ops
slice_or_select_overloadpkt = {
    torch.ops.aten.slice_copy,
    torch.ops.aten.select_copy,
    exir_ops.edge.aten.slice_copy,
    exir_ops.edge.aten.select_copy,
}


@register_cadence_pass(CadencePassAttribute(opt_level=2))
class AdvanceQuantizeOpAboveDefInBranchPass(ExportPass):
    """
    If the graph is branched with the following pattern:
    I = ...
    S1 = slice(I)
    Q1 = quantize(S1)
    S2 = slice(I)
    Q2 = quantize(S2)
    S3 = slice(I)
    Q3 = quantize(S3)
    ...
    such that the elements in the slices S1 + S2 + S3 is greater than I,
    we can advance the quantize above their defs (i.e., all the slice nodes),
    and reorder the pattern to the following:
    I = ...
    Q1 = quantize(I)
    S1 = slice(Q1)
    Q1 = requantize(S1)
    S2 = slice(Q1)
    Q2 = requantize(S2)
    S3 = slice(Q1)
    Q3 = requantize(S3)
    ...
    Note that the other passes won't do this transformation because they expect
    a linear chain of def-use, which is not true here; the uses of I are
    branched.
    """

    def __init__(self):
        super().__init__()
        self.graph_module = None

    # Starting at node, iterate through its successors, bypassing any trivially
    # quantizable op. If all the descendents are quantize ops, return them.
    def get_descendent_quant_ops(self, node: torch.fx.Node) -> List[torch.fx.Node]:
        # The list of quant ops that are descendents of node, such that the only
        # nodes in the path from node --> quant are trivially quantizable ops.
        descendent_quant_ops = []
        # The list of trivially quantizable ops in the path from node --> quant op.
        trivial_quantized_ops = []

        users = list(node.users.keys())
        while users:
            user = users.pop(0)
            user_target = get_overload_packet(user.target)
            # Record a quant op successor
            if user_target in {
                torch.ops.quantized_decomposed.quantize_per_tensor,
                exir_ops.edge.quantized_decomposed.quantize_per_tensor,
                torch.ops.cadence.quantize_per_tensor,
                exir_ops.edge.cadence.quantize_per_tensor,
            }:
                descendent_quant_ops.append(user)
            # If the successor is a trivially quantizable op, consider its users
            # instead.
            elif user_target in trivially_quantizable_ops_overloadpkt:
                trivial_quantized_ops.append(user)
                users.extend(list(user.users.keys()))
            # Otherwise all successors of node are not quant op, so break the loop.
            else:
                descendent_quant_ops.clear()
                break

        # If all the nodes in trivial_quantize_ops of the node were slice ops,
        # ensure that the advance is still profitable.
        if descendent_quant_ops and all(
            get_overload_packet(x.target) in slice_or_select_overloadpkt
            for x in trivial_quantized_ops
        ):
            # Profitability metric: the sum of all the output slices must be at
            # least half the input node slice.
            slice_sizes = [
                prod(list(y))
                for x in trivial_quantized_ops
                if (y := get_shape(self.graph_module, x)) is not None
            ]
            node_shape = get_shape(self.graph_module, node)
            node_size = prod(list(node_shape)) if node_shape is not None else 0
            if node_size > 2 * sum(slice_sizes):
                descendent_quant_ops.clear()

        return descendent_quant_ops

    def advance_quantize_op(self, graph_module: torch.fx.GraphModule):
        graph = graph_module.graph
        for node in graph.nodes:
            # We are only interested in call functions and placeholders
            if node.op not in {"placeholder", "call_function"}:
                continue
            # If the node is trivially quantizable, skip it
            if (
                get_overload_packet(node.target)
                in trivially_quantizable_ops_overloadpkt
            ):
                continue
            # Get the descendent quant ops that are connected to the current
            # node via trivially quantizable ops.
            descendent_quant_ops = self.get_descendent_quant_ops(node)
            if not descendent_quant_ops:
                continue

            # Get the insertion point below which we need to insert anything.
            # if node is a placeholder, we will only insert a new node after
            # all the placeholders in the graph.
            insertion_pt = (
                get_placeholders(graph)[-1] if node.op == "placeholder" else node
            )

            # If the node only has a single quant op as descendent, we can
            # simply hoist the quant op below the current node as its single
            # child.
            if len(descendent_quant_ops) == 1:
                quant_node = descendent_quant_ops.pop()
                # Replace the uses of quant node with its predecessor
                quant_node.replace_all_uses_with(quant_node.args[0])  # pyre-fixme[6]
                # Hoist the quant node after the current node. Make sure that
                # the insertion is after placeholders
                with graph.inserting_after(insertion_pt):
                    dom_quant_args = (node,) + quant_node.args[1:]
                    dom_quant_node = graph.call_function(
                        exir_ops.edge.quantized_decomposed.quantize_per_tensor.default
                    )
                    dom_quant_node.meta = node.meta
                    node.replace_all_uses_with(dom_quant_node)
                    dom_quant_node.args = dom_quant_args
                graph.erase_node(quant_node)
                continue

            # Otherwise we have the quant descendents. Cluster them into sets
            # that have the same scale, zero_point, and dtype. We use quant_dict
            # for the clustering
            quant_dict: DefaultDict[Tuple, int] = defaultdict(int)
            for quant_node in descendent_quant_ops:
                quant_dict[quant_node.args[1:]] += 1
            rep_args = sorted(quant_dict.keys(), key=lambda x: x[1]).pop()

            # Create a new quant node that dominates all the nodes in
            # descendent_quant_ops. Make sure that the insertion is after
            # all the placeholders.
            with graph.inserting_after(insertion_pt):
                dom_quant_args = (node,) + rep_args
                dom_quant_node = graph.call_function(
                    exir_ops.edge.quantized_decomposed.quantize_per_tensor.default
                )
                dom_quant_node.meta = node.meta
                node.replace_all_uses_with(dom_quant_node)
                dom_quant_node.args = dom_quant_args

            # Finally, convert each of the quant node to a dequant/quant pair that
            # requantizes the data flowing through dom_quant_node.
            # TODO: Once requantize is implemented for PT2, replace the
            # dequant/quant pair here with a single requantize node
            for quant_node in descendent_quant_ops:
                with graph.inserting_before(quant_node):
                    dequant_node = graph.call_function(
                        exir_ops.edge.quantized_decomposed.dequantize_per_tensor.default
                    )
                    dequant_node.args = (quant_node.args[0],) + rep_args
                    quant_node.args = (dequant_node,) + quant_node.args[1:]

        graph_module.recompile()
        graph_module.graph.eliminate_dead_code()

    def call(self, graph_module: torch.fx.GraphModule) -> PassResult:
        self.graph_module = graph_module
        self.advance_quantize_op(graph_module)
        result = super().call(graph_module)
        return result


@register_cadence_pass(CadencePassAttribute(opt_level=1))
class AdvanceQuantizeOpAboveDefChainPass(ExportPass):
    """
    If the input to quantize op is linear chain of view, transpose, permute, or
    slice ops that are trivially quantized, we can convert the pattern
    view/transpose/permute/slice(fp32) -> quantize(int8/uint8) to
    quantize(int8/uint8) -> view/transpose/permute/slice(int8/uint8).
    The benefit of such reordering is that the view/transpose/permute/slice
    will move far less data.
    """

    def __init__(self):
        super().__init__()
        self.graph_module = None

    # Return true if advancing the quantize node is feasible
    def advancing_feasible(self, quant_node: torch.fx.Node):
        assert quant_node.op == "call_function" and len(quant_node.args) >= 1
        # Get the input of the quant node. Only proceed if it's a torch node.
        inp = quant_node.args[0]
        if not isinstance(inp, torch.fx.Node):
            return False

        # Return false if the input to the quantize node is (1) not trivially
        # quantizable, or (2) has more than one user.
        inp_users = list(inp.users.keys())
        inp_overloadpkt = None
        if isinstance(inp.target, EdgeOpOverload):
            inp_overloadpkt = get_edge_overload_packet(inp.target)
        else:
            inp_overloadpkt = get_overload_packet(inp.target)

        if (
            inp_overloadpkt not in trivially_quantizable_ops_overloadpkt
            or len(inp_users) != 1
        ):
            return False

        # Advancing quantize op above slice nodes is tricky. If we advance the
        # quantize node above slice, then we will quantize the input to the slice
        # op, which can be expensive. We only bypass nop slice at present.
        if inp_overloadpkt in slice_or_select_overloadpkt:
            sliced_tensor = inp.args[0]
            assert isinstance(sliced_tensor, torch.fx.Node)
            slice_input_shape = get_shape(self.graph_module, sliced_tensor)
            slice_output_shape = get_shape(self.graph_module, inp)
            # If we could not glean the shapes, or the slice op is a nop, bail
            if (
                slice_output_shape is None
                or slice_input_shape is None
                or prod(list(slice_output_shape)) < prod(list(slice_input_shape))
            ):
                return False

        # All the conditions satisfied, we advance.
        return True

    def advance_quantize_op(self, graph_module: torch.fx.GraphModule) -> bool:
        graph = graph_module.graph
        modified = False
        for node in reversed(graph.nodes):
            if get_overload_packet(node.target) not in (
                exir_ops.edge.quantized_decomposed.quantize_per_tensor,
                torch.ops.quantized_decomposed.quantize_per_tensor,
                exir_ops.edge.cadence.quantize_per_tensor,
                torch.ops.cadence.quantize_per_tensor,
            ):
                continue

            if not self.advancing_feasible(node):
                continue

            trivially_quantizable_op = node.args[0]
            # The input to the quant node must now be the input to the trivially
            # quantizable op.
            quant_args = list(node.args)
            quant_args[0] = trivially_quantizable_op.args[0]

            # Insert the new quant node with updated args before the current
            # quant node.
            with graph.inserting_before(node):
                quant_node = graph.call_function(node.target, args=tuple(quant_args))
                quant_node.meta = node.meta
            # Move the trivially quantizable node after the quant node
            with graph.inserting_after(node):
                tq_args = list(trivially_quantizable_op.args)
                tq_args[0] = quant_node
                tq_node = graph.call_function(
                    trivially_quantizable_op.target,
                    args=tuple(tq_args),
                    kwargs=trivially_quantizable_op.kwargs,
                )
                tq_node.meta = trivially_quantizable_op.meta
            # Replace all uses of node with newly created tq_node
            node.replace_all_uses_with(tq_node)
            # We can safely remove the quant node and trivially quantizable op
            graph.erase_node(node)
            graph.erase_node(trivially_quantizable_op)
            modified = True

        return modified

    def call(self, graph_module: torch.fx.GraphModule) -> PassResult:
        self.graph_module = graph_module
        modified = self.advance_quantize_op(graph_module)
        if modified:
            graph_module.recompile()
            graph_module.graph.eliminate_dead_code()
            return super().call(graph_module)

        return PassResult(graph_module, False)


@register_cadence_pass(CadencePassAttribute(opt_level=1))
class PostponeDequantizeOpBelowUseChainPass(ExportPass):
    """
    If the consumer of dequantize is a linear chain of view, transpose, permute,
    or slice ops that are trivially quantized, we can convert the pattern
    dequantize(int8/uint8) -> view/transpose/permute/slice(fp32) to
    view/transpose/permute/slice(int8/uint8) -> dequantize(int8/uint8)
    The benefit of such reordering is that the view/transpose/permute/slice
    will move far less data.
    """

    def __init__(self):
        super().__init__()
        self.graph_module = None

    # Return true if postponing the dequantize node is feasible
    def postponing_feasible(self, dequant_node: torch.fx.Node):
        users = list(dequant_node.users.keys())
        # Check if the dequantize op has a single user, and that user is
        # trivially quantizable.
        trivially_quantizable_users = all(
            get_overload_packet(user.target) in trivially_quantizable_ops_overloadpkt
            for user in users
        )
        if len(users) == 1:
            return trivially_quantizable_users

        # Otherwise check if all the users are slice op
        if not all(
            get_overload_packet(user.target) in slice_or_select_overloadpkt
            for user in users
        ):
            return False

        dequant_shape = get_shape(self.graph_module, dequant_node)
        slice_shapes = [
            shape
            for user in users
            if (shape := get_shape(self.graph_module, user))
            and (
                # skip slices that are the size of the sliced tensor itself.
                # They should technically get removed in the later passes as nop.
                shape is None
                or dequant_shape is None
                or prod(list(shape)) != prod(list(dequant_shape))
            )
        ]

        if dequant_shape is not None and all(
            shape is not None for shape in slice_shapes
        ):
            dequant_bytes = num_bytes_from_shape_and_dtype(dequant_shape, torch.float32)
            slice_bytes = sum(
                [
                    num_bytes_from_shape_and_dtype(shape, torch.float32)
                    for shape in slice_shapes
                ]
            )
            if slice_bytes <= dequant_bytes:
                return True

        # If the users of each slice op is quantize op, then we can postpone
        # dequantize, and convert slice -> dequantize -> quantize to
        # slice -> requantize.
        users = [x for y in users for x in y.users if x.op != "output"]
        return all(
            get_overload_packet(x.target)
            in {
                exir_ops.edge.quantized_decomposed.quantize_per_tensor,
                exir_ops.edge.quantized_decomposed.quantize_per_channel,
                exir_ops.edge.cadence.quantize_per_tensor,
            }
            for x in users
        )

    def postpone_dequantize_op(self, graph_module: torch.fx.GraphModule) -> bool:
        # Different supported dequant ops have their own default variants
        packet_to_overload_map = {
            exir_ops.edge.quantized_decomposed.dequantize_per_tensor: "default",
            exir_ops.edge.quantized_decomposed.dequantize_per_channel: "default",
            exir_ops.edge.cadence.dequantize_per_tensor: "default",
        }
        graph = graph_module.graph
        modified = False
        for node in graph.nodes:
            overload_packet = get_overload_packet(node.target)
            if (
                overload_packet not in packet_to_overload_map.keys()
                or not self.postponing_feasible(node)
            ):
                continue

            for user in node.users:
                with graph.inserting_after(user):
                    dequant_node = graph.call_function(
                        getattr(
                            overload_packet, packet_to_overload_map[overload_packet]
                        ),
                        args=(user, *node.args[1:]),
                    )
                    dequant_node.meta = user.meta.copy()
                    # Remove meta["debug_handle"] on new node if it exists.
                    # Reassign it at the caller level by calling generate_missing_debug_handles
                    dequant_node.meta.pop("debug_handle", None)
                    user.replace_all_uses_with(dequant_node)
                    dequant_node.args = (user, *node.args[1:])

            pred = node.args[0]
            node.replace_all_uses_with(pred)
            graph.erase_node(node)
            modified = True

        graph_module.recompile()
        return modified

    def call(self, graph_module: torch.fx.GraphModule) -> PassResult:
        # The logic in postpone_dequantize_op that handles branching checks the shape
        # of the dequant node, which isn't available if that node was already postponed
        # in the same pass invokation. The shape information is recreated by tracing in
        # super().call(), meaning that every branch in the graph that we wish to postpone
        # dequant past requires retracing. We iterate the pass until it no longer modifies
        # the graph (up to 3 times max, to avoid potential infinite loops)
        self.graph_module = graph_module
        iter_count = 0
        local_modified = False
        overall_modified = False

        while local_modified or iter_count == 0:
            local_modified = self.postpone_dequantize_op(self.graph_module)
            overall_modified |= local_modified

            if local_modified:
                self.graph_module = super().call(self.graph_module).graph_module

            iter_count += 1
            if iter_count == 3:
                break

        return PassResult(self.graph_module, overall_modified)


@register_cadence_pass(CadencePassAttribute(opt_level=1))
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


@register_cadence_pass(CadencePassAttribute(opt_level=1))
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


@register_cadence_pass(CadencePassAttribute(opt_level=1))
class PostponePermuteOpBelowSqueezeOrUnsqueezeLikeView(
    _SharedPostponePermuteOpBelowSqueezeOrUnsqueezeLikeView
):
    pass


@register_cadence_pass(CadencePassAttribute(opt_level=1))
class MoveSliceBeforePermutePass(RemoveOrReplacePassInterface):
    """Move slice_copy ops before permute_copy to reduce permute data volume.

    Rewrites permute(input, perm) -> slice(dim=D) into
    slice(input, dim=perm[D]) -> permute(sliced, perm), so the permute
    operates on a smaller tensor.
    """

    @property
    def targets(self) -> list[EdgeOpOverload]:
        return [exir_ops.edge.aten.permute_copy.default]

    def maybe_remove_or_replace(self, node: torch.fx.Node) -> bool:
        perm = cast(list[int], node.args[1])
        permute_input = node.args[0]
        assert isinstance(permute_input, torch.fx.Node)

        if len(node.users) != 1:
            return False

        user = next(iter(node.users))
        if user.target != exir_ops.edge.aten.slice_copy.Tensor:
            return False

        slice_users = [user]

        graph = node.graph
        modified = False
        for slice_node in slice_users:
            slice_dim = get_arg(slice_node, "dim", int)
            new_dim = perm[slice_dim]

            with graph.inserting_before(node):
                new_slice = graph.create_node(
                    "call_function",
                    exir_ops.edge.aten.slice_copy.Tensor,
                    args=(
                        permute_input,
                        new_dim,
                        get_arg(slice_node, "start"),
                        get_arg(slice_node, "end"),
                        get_arg(slice_node, "step", int),
                    ),
                )
                new_permute = graph.create_node(
                    "call_function",
                    exir_ops.edge.aten.permute_copy.default,
                    args=(new_slice, perm),
                )

            slice_node.replace_all_uses_with(new_permute)
            modified = True

        return modified


# The following class consolidates functions to reoder ops (i.e., either hoist
# or sink some ops in the graph).
class CadenceReorderOpsInGraph:
    passes = [
        # Hoist/sink nodes closer to their SSA def/use
        HoistOpsCloserToDefPass,
        SinkOpsCloserToUsePass,
        # For quantize/dequantize ops, move them above/below their def chain.
        # This is a more aggressive optimization than just hoisting/sinking
        # nodes closer to their def/use.
        AdvanceQuantizeOpAboveDefChainPass,
        PostponeDequantizeOpBelowUseChainPass,
        # These passes work on branches instead of linear chains to advance
        # quantize op beyond their def.
        AdvanceQuantizeOpAboveDefInBranchPass,
    ]
