# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-unsafe


# This file contains all the functions that reorder ops in the graph module.

import copy
from collections import defaultdict
from math import prod
from typing import cast, DefaultDict, List, Set, Tuple

import torch
import torch.fx
from executorch.backends.cadence.aot.compiler_utils import get_placeholders, get_shape
from executorch.backends.cadence.aot.pass_utils import (
    CadencePassAttribute,
    get_overload_packet,
    register_cadence_pass,
)
from executorch.backends.cadence.aot.utils import get_edge_overload_packet
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
                quant_node.replace_all_uses_with(quant_node.args[0])  # type: ignore[arg-type]
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
            inp_overloadpkt = get_overload_packet(inp.target)  # type: ignore[assignment]

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

    def advance_quantize_op(self, graph_module: torch.fx.GraphModule):
        graph = graph_module.graph
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

        graph_module.recompile()
        graph_module.graph.eliminate_dead_code()

    def call(self, graph_module: torch.fx.GraphModule) -> PassResult:
        self.graph_module = graph_module
        self.advance_quantize_op(graph_module)
        result = super().call(graph_module)
        return result


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
        modified = True

        while modified and iter_count < 3:
            modified = self.postpone_dequantize_op(self.graph_module)
            self.graph_module = super().call(self.graph_module).graph_module
            iter_count += 1

        return super().call(self.graph_module)


@register_cadence_pass(CadencePassAttribute(opt_level=1))
class SinkOpsCloserToUsePass(ExportPass):
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

    sinkable_ops: Set[EdgeOpOverload] = {
        exir_ops.edge.aten.dequantize,
        exir_ops.edge.quantized_decomposed.dequantize_per_tensor,
        exir_ops.edge.quantized_decomposed.dequantize_per_channel,
        exir_ops.edge.cadence.dequantize_per_tensor,
    }

    def sink_ops_closer_to_use(self, graph_module: torch.fx.GraphModule):
        graph = graph_module.graph
        # We are only interested in sinkable nodes
        sinkable_nodes = [
            node
            for node in graph.nodes
            if isinstance(node.target, EdgeOpOverload)
            and get_edge_overload_packet(node.target) in self.sinkable_ops
        ]
        for node in sinkable_nodes:
            # The sinkable node must have a single user
            users = list(node.users.keys())
            if len(users) != 1:
                continue

            # Insert the dequant node just before its user
            with graph.inserting_before(users[0]):
                new_node = graph.call_function(
                    node.target, args=node.args, kwargs=node.kwargs
                )
                new_node.meta = node.meta
            node.replace_all_uses_with(new_node)
            graph.erase_node(node)

        graph_module.recompile()

    def call(self, graph_module: torch.fx.GraphModule) -> PassResult:
        self.sink_ops_closer_to_use(graph_module)
        result = super().call(graph_module)
        return result


@register_cadence_pass(CadencePassAttribute(opt_level=1))
class HoistOpsCloserToDefPass(ExportPass):
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

    hoistable_ops: Set[EdgeOpOverload] = {
        exir_ops.edge.quantized_decomposed.quantize_per_tensor,
        exir_ops.edge.cadence.quantize_per_tensor,
        exir_ops.edge.aten.slice_copy,
        exir_ops.edge.aten.select_copy,
    }

    def hoist_ops_closer_to_def(self, graph_module: torch.fx.GraphModule):
        graph = graph_module.graph
        # We are only interested in hoistable nodes
        hoistable_nodes = [
            node
            for node in graph.nodes
            if isinstance(node.target, EdgeOpOverload)
            and get_edge_overload_packet(node.target) in self.hoistable_ops
        ]
        for node in hoistable_nodes:
            def_node = node.args[0]
            if not isinstance(def_node, torch.fx.Node):
                continue
            # The def node must have a single user
            users = list(def_node.users.keys())
            if len(users) != 1:
                continue

            # Get the node args as list
            args = list(node.args)

            # If the graph has placeholders, we do not want to hoist above the
            # last placeholder. Otherwise we will shrink the live range of the
            # def_node considerably, which could lead to reuse of input memory.
            def_node = (
                get_placeholders(graph)[-1]
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
                with graph.inserting_after(def_node):
                    zero_point_copy = graph.node_copy(zero_point)
                    scale_copy = graph.node_copy(scale)
                    args[1], args[2] = scale_copy, zero_point_copy
                    def_node = zero_point_copy

            # Insert the quant node just after def_node
            with graph.inserting_after(def_node):
                new_node = graph.call_function(
                    node.target, args=tuple(args), kwargs=node.kwargs
                )
                new_node.meta = node.meta
            node.replace_all_uses_with(new_node)
            graph.erase_node(node)

        # Eliminate dead code
        graph_module.recompile()
        graph_module.graph.eliminate_dead_code()

    def call(self, graph_module: torch.fx.GraphModule) -> PassResult:
        self.hoist_ops_closer_to_def(graph_module)
        result = super().call(graph_module)
        return result


@register_cadence_pass(CadencePassAttribute(opt_level=1))
class PostponePermuteOpBelowSqueezeOrUnsqueezeLikeView(ExportPass):
    """
    A common pattern seen in transformer models.  If the consumer of permute
    is a view op, swap their order so permute is below view.
    Change "permute -> view" to "view -> permute"
    This is to optimize a chain of view->permute->view->permute...
    so that the chain will be become view->v...->view->permute->p...->permute.
    The chain can be optimized by FuseCascadedTransposeOrPermuteOps() and
    FuseCascadedViewOps().
    Notice the class name has ViewSqueeze to indicate the View is
    functionally the same as a squeeze or unsqueeze. It does not necessarily
    mean the view_copy is normalized from squeeze or unsqueeze.
    """

    def __init__(self):
        super().__init__()
        self.graph_module = None

    # If list1 and list2 are same (same values and in same order) except
    # list1 has one more element with value of 1. Return index of the extra 1.
    # Otherwise return -1.
    def check_if_shapes_differ_in_single_dim_of_size_1(self, list1, list2) -> int:
        if len(list1) != len(list2) + 1:
            return -1
        for i in range(len(list2)):
            if list1[i] != list2[i]:
                # Return index of the extra 1 if the remaining parts are the same
                if list1[i] == 1 and list2[i:] == list1[i + 1 :]:
                    return i
                else:
                    return -1
        # If no difference was found, the extra element is at the end
        if list1[-1] == 1:
            return len(list2)
        else:
            return -1

    def insert_nodes(
        self,
        graph: torch.fx.Graph,
        pred: torch.fx.Node,
        permute_node: torch.fx.Node,
        view_node: torch.fx.Node,
        new_view_shape: List,
        new_permute_dims: List,
    ):
        with graph.inserting_after(view_node):
            new_view_node = graph.call_function(
                view_node.target,  # type: ignore[arg-type]
                args=(pred, new_view_shape),
            )

        with graph.inserting_after(new_view_node):
            new_permute_node = graph.call_function(
                permute_node.target,  # type: ignore[arg-type]
                args=(new_view_node, new_permute_dims),
            )
            new_permute_node.meta = view_node.meta
            view_node.replace_all_uses_with(new_permute_node)

        # view_node is user of permute_node, so must erase view_node first
        graph.erase_node(view_node)
        graph.erase_node(permute_node)

    # flake8: noqa 'PostponePermuteOpBelowSqueezeOrUnsqueezeLikeView.postpone_permute_op' is too complex (13)
    def postpone_permute_op(self, graph_module: torch.fx.GraphModule):
        packet_to_overload_map = {
            exir_ops.edge.aten.permute_copy: "default",
        }
        graph = graph_module.graph
        changed = True
        modified = False
        # Loop iteratively until no more changes are made
        while changed:
            changed = False
            for permute_node in graph.nodes:
                permute_overload_packet = get_overload_packet(permute_node.target)
                if permute_overload_packet not in packet_to_overload_map.keys():
                    continue

                users = list(permute_node.users.keys())
                # Transform only for pattern permute_copy->view_copy, and
                # view_copy op is the only user of permute_copy.
                if len(users) == 1 and users[0].target in (
                    exir_ops.edge.aten.view_copy.default,
                    exir_ops.edge.aten.view.default,
                ):
                    # If the permute_node/view_node was newly added to the
                    # graph, it may not have the meta["val"] FakeTensor.
                    # Skip in this case.
                    if permute_node.meta.get("val") is None:
                        continue
                    permute_node_shape = [
                        *cast(list, get_shape(graph_module, permute_node))
                    ]
                    permute_dims = permute_node.args[1]
                    view_node = users[0]
                    if view_node.meta.get("val") is None:
                        continue
                    view_node_shape = [*cast(list, get_shape(graph_module, view_node))]
                    pred = permute_node.args[0]
                    if pred.meta.get("val") is None:
                        continue
                    pred_shape = [*cast(list, get_shape(graph_module, pred))]
                    # Handle two cases
                    # 1. view_node_shape is almost same as permute_node_shape
                    #    except the view_node has one more dim somewhere
                    #    and the extra dim has value of 1.
                    # 2. view_node_shape is almost same as permute_node_shape
                    #    except permute_node_shape has one more dim somewhere
                    #    and the extra dim has value of 1.
                    # 3. view_node_shape is the same as permute_node_shape.
                    if len(permute_node_shape) + 1 == len(view_node_shape):
                        index = self.check_if_shapes_differ_in_single_dim_of_size_1(
                            view_node_shape, permute_node_shape
                        )
                        if index != -1:
                            # view_node_shape is almost same as permute_node_shape
                            # except it has one more dim somewhere
                            # and the extra dim has value of 1.
                            new_view_shape = copy.deepcopy(pred_shape)
                            new_view_shape.insert(index, 1)
                            new_permute_dims = [
                                x + 1 if x >= index else x for x in permute_dims
                            ]
                            new_permute_dims.insert(index, index)
                            self.insert_nodes(
                                graph,
                                pred,
                                permute_node,
                                view_node,
                                new_view_shape,
                                new_permute_dims,
                            )
                            changed = True
                            modified = True
                    elif len(view_node_shape) + 1 == len(permute_node_shape):
                        index = self.check_if_shapes_differ_in_single_dim_of_size_1(
                            permute_node_shape, view_node_shape
                        )
                        if index != -1:
                            # view_node_shape is almost same as permute_node_shape
                            # except permute_node_shape has one more dim somewhere
                            # and the extra dim has value of 1.
                            index_to_remove = permute_dims[index]
                            new_view_shape = copy.deepcopy(pred_shape)
                            del new_view_shape[index_to_remove]
                            new_permute_dims = [
                                x - 1 if x > index_to_remove else x
                                for x in permute_dims
                            ]
                            del new_permute_dims[index]
                            self.insert_nodes(
                                graph,
                                pred,
                                permute_node,
                                view_node,
                                new_view_shape,
                                new_permute_dims,
                            )
                            changed = True
                            modified = True
                    elif permute_node_shape == view_node_shape:
                        # view_node_shape is the same as permute_node_shape
                        # Replace the uses of view_node with permute_node
                        view_node.replace_all_uses_with(permute_node)
                        changed = True
                        modified = True

        graph_module.recompile()
        return modified

    def call(self, graph_module: torch.fx.GraphModule) -> PassResult:
        self.graph_module = graph_module
        iter_count = 0
        modified = True

        while modified and iter_count <= 3:
            modified = self.postpone_permute_op(self.graph_module)
            self.graph_module = super().call(self.graph_module).graph_module
            iter_count += 1

        return super().call(self.graph_module)


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
