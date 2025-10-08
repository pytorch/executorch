# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict


import logging
from dataclasses import dataclass, field
from typing import cast, List, Optional, Sequence, Set, Type

import torch
import torch.fx
from executorch.backends.cadence.aot.pass_utils import (
    CadencePassAttribute,
    get_arg,
    register_cadence_pass,
    set_arg,
)

from executorch.backends.cadence.aot.simplify_ops import SimplifySliceOpPass
from executorch.backends.cadence.aot.utils import get_edge_overload_packet
from executorch.backends.transforms.remove_clone_ops import RemoveCloneOpsTransform
from executorch.exir.dialects._ops import ops as exir_ops
from executorch.exir.dialects.edge._ops import EdgeOpOverload, EdgeOpOverloadPacket
from executorch.exir.pass_base import ExportPass, NodeMetadata, PassResult, ProxyValue
from executorch.exir.pass_manager import PassManager, PassType
from executorch.exir.passes import dead_code_elimination_pass
from executorch.exir.passes.spec_prop_pass import SpecPropPass
from torch.fx.node import Argument, Node


@register_cadence_pass(CadencePassAttribute(opt_level=0))
class RemoveCloneOpsTransformImported(ExportPass):
    def call(self, graph_module: torch.fx.GraphModule) -> PassResult:
        finalize_passes: List[PassType] = [
            RemoveCloneOpsTransform(),
        ]
        result = PassManager(passes=finalize_passes)(graph_module)
        dead_code_elimination_pass(result.graph_module)
        return result


@register_cadence_pass(CadencePassAttribute(opt_level=0))
class RemoveDetachCopyPass(ExportPass):
    def call_operator(
        self,
        op,  # pyre-ignore
        args: tuple[Argument, ...],
        kwargs: dict[str, Argument],
        meta: NodeMetadata,
    ) -> ProxyValue:
        if op != exir_ops.edge.aten.detach_copy.default:
            return super().call_operator(op, args, kwargs, meta)

        assert len(args) == 1
        return cast(ProxyValue, args[0])


# The following class consolidates passes to remove ops that are redundant:
# either by the virtue of the operation they perform, or redundant in the
# context of inference.
class RemoveRedundantOps:
    passes = [
        RemoveDetachCopyPass,
    ]


@register_cadence_pass(CadencePassAttribute(opt_level=0))
class RemoveZeroSizedCatArgsPass(ExportPass):
    def call_operator(
        self,
        op,  # pyre-ignore
        args: tuple[Argument, ...],
        kwargs: dict[str, Argument],
        meta: NodeMetadata,
    ) -> ProxyValue:
        if op != exir_ops.edge.aten.cat.default:
            return super().call_operator(op, args, kwargs, meta)

        # Remove any zero-sized tensor arg to form a new args list.
        cat_inputs: list[ProxyValue] = []
        for arg in cast(Sequence[ProxyValue], args[0]):
            if arg.to_tensor().numel() > 0:
                cat_inputs.append(arg)

        # If all the tensors were empty, we just return an empty tensor with
        # the right shape.
        if not cat_inputs:
            empty_shape = meta["val"].shape
            dtype = meta["val"].dtype
            return super().call_operator(
                exir_ops.edge.aten.full.default,
                (tuple(empty_shape), 0),
                {"dtype": dtype},
                meta,
            )

        # If there was only one tensor in the cat_inputs list,
        # we can safely erase this cat op.
        if len(cat_inputs) == 1:
            return cat_inputs[0]

        # Otherwise, we replace args[0] with cat_inputs.
        new_args = list(args)
        # pyre error introduced after D66937105
        new_args[0] = cat_inputs  # pyre-ignore[6]
        return super().call_operator(op, tuple(new_args), kwargs, meta)


@register_cadence_pass(CadencePassAttribute(opt_level=0))
class RemoveNopExpandOpPass(ExportPass):
    """
    For an expand op, if the operator shape matches the expand shape, then the
    expand is a nop.
    """

    def call_operator(
        self,
        op,  # pyre-ignore
        args: tuple[Argument, ...],
        kwargs: dict[str, Argument],
        meta: NodeMetadata,
    ) -> ProxyValue:
        if get_edge_overload_packet(op) not in {
            exir_ops.edge.aten.expand_copy,
            exir_ops.edge.aten.expand,
        }:
            return super().call_operator(op, args, kwargs, meta)

        # Parse the args, and check for nop condition
        arg0 = cast(ProxyValue, args[0])
        arg1 = cast(Sequence[int], args[1])
        in_tensor = arg0.to_tensor()
        if list(in_tensor.shape) == list(arg1):
            return arg0

        return super().call_operator(op, args, kwargs, meta)


@register_cadence_pass(CadencePassAttribute(opt_level=0))
class RemoveToOpsPass(ExportPass):
    # aten.to.* as of now are all nops
    def call_operator(
        self,
        op,  # pyre-ignore
        args: tuple[Argument, ...],
        kwargs: dict[str, Argument],
        meta: NodeMetadata,
    ) -> ProxyValue:
        if op not in (
            exir_ops.edge.aten.to.dtype,
            exir_ops.edge.aten.to.dtype_layout,
        ):
            return super().call_operator(op, args, kwargs, meta)

        logging.debug(f"Erasing to.dtype node (target = {op})")
        return cast(ProxyValue, args[0])


@register_cadence_pass(CadencePassAttribute(opt_level=1))
class RemoveZeroSizedConstantPadNd(ExportPass):
    def call_operator(
        self,
        op,  # pyre-ignore
        args: tuple[ProxyValue, tuple[int, ...], Argument],
        kwargs: dict[str, Argument],
        meta: NodeMetadata,
    ) -> ProxyValue:
        if op != exir_ops.edge.aten.constant_pad_nd.default:
            return super().call_operator(op, args, kwargs, meta)

        input_tensor = args[0]
        padding = args[1]

        if any(x != 0 for x in padding):
            return super().call_operator(op, args, kwargs, meta)

        logging.debug(f"Erasing 0 sized constant pad nd node with {input_tensor}")
        return input_tensor


@register_cadence_pass(CadencePassAttribute(opt_level=1))
class RemoveNopSliceOrViewOpPass(ExportPass):
    """
    Remove slice ops that are more like views, and view ops that do not change the shape
    """

    def call_operator(
        self,
        op,  # pyre-ignore
        args: tuple[Argument, ...],
        kwargs: dict[str, Argument],
        meta: NodeMetadata,
    ) -> ProxyValue:
        if op not in {
            exir_ops.edge.aten.slice_copy.Tensor,
            exir_ops.edge.aten.view_copy.default,
        }:
            return super().call_operator(op, args, kwargs, meta)

        arg0 = cast(ProxyValue, args[0])
        out_shape = meta["val"].shape

        # If both arg_shape and out_shape are the same, this slice is a nop
        return (
            arg0
            if arg0.to_tensor().shape == out_shape
            else super().call_operator(op, args, kwargs, meta)
        )


@register_cadence_pass(CadencePassAttribute(opt_level=1))
class RemoveNopLinalgVectorNormOpPass(ExportPass):
    """
    If the norm is applied over a dimension that is size 1, it can be eliminated.
    """

    def call_operator(
        self,
        op,  # pyre-ignore
        args: tuple[Argument, ...],
        kwargs: dict[str, Argument],
        meta: NodeMetadata,
    ) -> ProxyValue:
        if op is not exir_ops.edge.aten.linalg_vector_norm.default:
            return super().call_operator(op, args, kwargs, meta)

        # If the op has three args or less, it can't be a nop
        if len(args) <= 3:
            return super().call_operator(op, args, kwargs, meta)
        # If dim is None, or keepdim is False, it is not a nop
        dim = cast(Optional[tuple[int, ...]], args[2])
        keepdim = cast(bool, args[3])
        if dim is None or not keepdim:
            return super().call_operator(op, args, kwargs, meta)

        # If the norm has 4 args and keepdim is True, check if dim is not None
        # and if the dimensions in dim are size 1. If not, the norm is not a nop.
        t = cast(ProxyValue, args[0])
        shape = t.to_tensor().shape
        if len(args) < 4:
            for d in dim:
                if shape[d] != 1:
                    return super().call_operator(op, args, kwargs, meta)

        return t


@register_cadence_pass(CadencePassAttribute(opt_level=1))
class RemoveNopSelectOpPass(ExportPass):
    """
    A select op that selects from a dimension that is size 1 can be eliminated
    in a few cases. For example,
    ```
    x = view (x, [1, 3, 16])
    y = select(x, 0, 0)
    z = add(m, y)
    ```
    The special thing about this pattern is the add op, which allows
    broadcasting. So adding an operand with shape [3, 16] is the same as
    adding an operand with shape [1, 3, 16]. Therefore, if m has the same
    shape as x, then this select op is a nop, and can be eliminated:
    ```
    x = view (x, [1, 3, 16])
    z = add(x, m)
    ```
    """

    # A set of binary operators that could require broadcasting, and are
    # critical to this transformation if their operand is select op.
    binary_broadcast_ops: set[EdgeOpOverload] = {
        exir_ops.edge.aten.add.Tensor,
        exir_ops.edge.aten.mul.Tensor,
        exir_ops.edge.aten.div.Tensor,
    }

    def __init__(self) -> None:
        super().__init__()
        self.op_sizes: dict[str, tuple[torch.Size, torch.Size]] = {}

    # For select, view, or any op in binary_broadcast_ops, record the shapes of
    # input and output tensors.
    def call_operator(
        self,
        op,  # pyre-ignore
        args: tuple[Argument, ...],
        kwargs: dict[str, Argument],
        meta: NodeMetadata,
    ) -> ProxyValue:
        res = super().call_operator(op, args, kwargs, meta)
        # Unary ops: input and output
        if op in {
            exir_ops.edge.aten.select_copy.int,
            exir_ops.edge.aten.view_copy.default,
        }:
            arg0 = cast(ProxyValue, args[0])
            self.op_sizes[res.node.name] = (arg0.to_tensor().shape, meta["val"].shape)
        # Binary ops: two inputs, output shape can be inferred
        elif op in self.binary_broadcast_ops:
            arg0 = cast(ProxyValue, args[0])
            arg1 = cast(ProxyValue, args[1])
            self.op_sizes[res.node.name] = (
                arg0.to_tensor().shape,
                arg1.to_tensor().shape,
            )
        return res

    # Eliminate nop select ops. We begin by inspecting the binary_broadcast_ops,
    # and check if their arg is a select op.
    def eliminate_nop_select_op(self, graph_module: torch.fx.GraphModule) -> None:
        for sel_node in graph_module.graph.nodes:
            # We are only interested in select ops
            if sel_node.target != exir_ops.edge.aten.select_copy.int:
                continue
            # The shape of the input/output operands for this select op should
            # have been precomputed.
            assert sel_node.name in self.op_sizes
            (sel_in_shape, sel_out_shape) = self.op_sizes[sel_node.name]
            # Get the select dimension
            sel_dim = (
                sel_node.args[1]
                if sel_node.args[1] >= 0
                else sel_node.args[1] + len(sel_in_shape)
            )
            # If the input size along select dimension is not 1, bail.
            if sel_in_shape[sel_dim] != 1:
                continue

            # Get all the users of the select op that are either view, or
            # binary_broadcast_ops.
            users = [x for x in list(sel_node.users.keys()) if x.name in self.op_sizes]
            sel_in = sel_node.args[0]

            # Iterate over the users of select op, and remove the use of the
            # select op in the user if feasible.
            for node in users:
                args = list(node.args)
                for idx, sel_arg in enumerate(args):
                    # Check if the arg is the select op
                    if sel_arg != sel_node:
                        continue
                    # If the input of select has the same shape as the other arg
                    # of the binary op, the select op can be bypassed.
                    if sel_in_shape == self.op_sizes[node.name][(idx + 1) % 2]:
                        args[idx] = sel_in
                # update the node's args
                node.args = tuple(args)

        graph_module.recompile()
        graph_module.graph.eliminate_dead_code()

    def call(self, graph_module: torch.fx.GraphModule) -> PassResult:
        result = SpecPropPass()(graph_module)
        assert result is not None
        result = super().call(result.graph_module)
        self.eliminate_nop_select_op(result.graph_module)
        return result


@register_cadence_pass(CadencePassAttribute(opt_level=1))
class RemoveCloneOpPass(ExportPass):
    # If the op is a clone op, return the input and eliminate the op
    def call_operator(
        self,
        op,  # pyre-ignore
        args: tuple[ProxyValue],
        kwargs: dict[str, Argument],
        meta: NodeMetadata,
    ) -> ProxyValue:
        if op != exir_ops.edge.aten.clone.default:
            return super().call_operator(op, args, kwargs, meta)

        return args[0]


@register_cadence_pass(CadencePassAttribute(opt_level=1))
class RemoveContiguousOpPass(ExportPass):
    """
    This is based on the assumption that all tensors are contiguous in ExecuTorch
    and after cadence passes, and we should revisit this if that assumption is no longer true.
    This causes the model to not be runnable with the arguments given to the
    original graph module.
    """

    def call_operator(
        self,
        op,  # pyre-ignore
        args: tuple[Argument, ...],
        kwargs: dict[str, Argument],
        meta: NodeMetadata,
    ) -> ProxyValue:
        if op != exir_ops.edge.aten.contiguous.default:
            return super().call_operator(op, args, kwargs, meta)

        assert len(args) == 1
        return cast(ProxyValue, args[0])


@register_cadence_pass(CadencePassAttribute(opt_level=0))
class RemoveAliasCopyOpPass(ExportPass):
    """

    alias_copy is a no-op and can be removed.
    """

    def call_operator(
        self,
        op,  # pyre-ignore
        args: tuple[Argument, ...],
        kwargs: dict[str, Argument],
        meta: NodeMetadata,
    ) -> ProxyValue:
        if op != exir_ops.edge.aten.alias_copy.default:
            return super().call_operator(op, args, kwargs, meta)

        assert len(args) == 1
        return cast(ProxyValue, args[0])


@register_cadence_pass(CadencePassAttribute(opt_level=1))
class RemoveNopRequantizeOpPass(ExportPass):
    """
    For a requantize op, if the following three conditions are satisfied:
    1. the in_scale matches the out_scale
    2. the in_zero_point matches the out_zero_point
    3. the dtypes of the input and output tensors are the same
    then the requantize op is redundant, and can be eliminated
    """

    def call_operator(
        self,
        op,  # pyre-ignore
        args: tuple[Argument, ...],
        kwargs: dict[str, Argument],
        meta: NodeMetadata,
    ) -> ProxyValue:
        if op != exir_ops.edge.cadence.requantize.per_tensor:
            return super().call_operator(op, args, kwargs, meta)

        # Parse the args
        (X, in_scale, in_zero_point, out_scale, out_zero_point, out_dtype) = cast(
            tuple[ProxyValue, int, float, int, float, torch.dtype], args
        )
        in_dtype = X.to_tensor().dtype
        # Check the three conditions
        if (
            in_scale == out_scale
            and in_zero_point == out_zero_point
            and in_dtype == out_dtype
        ):
            return cast(ProxyValue, args[0])

        return super().call_operator(op, args, kwargs, meta)


@register_cadence_pass(CadencePassAttribute(opt_level=1))
class RemoveNopMulOpPass(ExportPass):
    """
    If a mul op is multiplying two tensors with the same shape and one
    of those tensors is all zeros, return the zero tensor instead.
    """

    def call_operator(
        self,
        op,  # pyre-ignore
        args: tuple[Argument, ...],
        kwargs: dict[str, Argument],
        meta: NodeMetadata,
    ) -> ProxyValue:
        if op != exir_ops.edge.aten.mul.Tensor:
            return super().call_operator(op, args, kwargs, meta)

        # Parse the args
        (input1, input2) = cast(tuple[ProxyValue, ProxyValue], args)

        # Check if both inputs have the same shape
        if input1.to_tensor().shape != input2.to_tensor().shape:
            return super().call_operator(op, args, kwargs, meta)

        # Check if one of the inputs is a zero tensor
        if input1.node.target == exir_ops.edge.aten.full.default:
            if input1.node.args[1] == 0:
                return input1
        elif input2.node.target == exir_ops.edge.aten.full.default:
            if input2.node.args[1] == 0:
                return input2

        return super().call_operator(op, args, kwargs, meta)


@register_cadence_pass(CadencePassAttribute(opt_level=1))
class RemoveNopAddOpPass(ExportPass):
    """
    If an add op is adding two tensors with the same shape and one
    of those tensors is all zeros, return the other tensor instead.
    """

    def call_operator(
        self,
        op,  # pyre-ignore
        args: tuple[Argument, ...],
        kwargs: dict[str, Argument],
        meta: NodeMetadata,
    ) -> ProxyValue:
        if op != exir_ops.edge.aten.add.Tensor:
            return super().call_operator(op, args, kwargs, meta)

        # Parse the args
        (input1, input2) = cast(tuple[ProxyValue, ProxyValue], args)

        # Check if both inputs have the same shape
        if input1.to_tensor().shape != input2.to_tensor().shape:
            return super().call_operator(op, args, kwargs, meta)

        # Check if one of the inputs is a zero tensor
        if input1.node.target == exir_ops.edge.aten.full.default:
            if input1.node.args[1] == 0:
                return input2
        elif input2.node.target == exir_ops.edge.aten.full.default:
            if input2.node.args[1] == 0:
                return input1

        return super().call_operator(op, args, kwargs, meta)


@register_cadence_pass(CadencePassAttribute(opt_level=2))
class RemovePermutesAroundElementwiseOps(ExportPass):
    """
    Looks for subgraphs of elementwise ops sandwiched between permutes and removes those
    permutes if possible.
    Allows special handling for certain non-elementwise ops that can be easily updated
    based on the permute's parameter such as mean, cat, and slice.
    """

    @dataclass()
    class Subgraph:
        start_permute: list[int]
        end_permute: list[int]
        # Nodes in the subgraph, does not include permutes.
        nodes: set[torch.fx.Node] = field(default_factory=set)
        # Incoming edges to the subgraph from permute nodes.
        edges_in: set[tuple[torch.fx.Node, torch.fx.Node]] = field(default_factory=set)
        # Outgoing edges of the subgraph to permute nodes.
        edges_out: set[tuple[torch.fx.Node, torch.fx.Node]] = field(default_factory=set)

    permutable_ops: set[EdgeOpOverload] = {
        exir_ops.edge.aten.add.Tensor,
        exir_ops.edge.aten.mul.Tensor,
        exir_ops.edge.aten.hardtanh.default,
        exir_ops.edge.quantized_decomposed.quantize_per_tensor.default,
        exir_ops.edge.quantized_decomposed.dequantize_per_tensor.default,
        exir_ops.edge.cadence.quantize_per_tensor.default,
        exir_ops.edge.cadence.dequantize_per_tensor.default,
        # Ops that require special handling.
        exir_ops.edge.aten.cat.default,
        exir_ops.edge.aten.mean.dim,
        exir_ops.edge.aten.slice_copy.Tensor,
    }

    def call(self, graph_module: torch.fx.GraphModule) -> PassResult:
        subgraphs_found: list[RemovePermutesAroundElementwiseOps.Subgraph] = []
        processed_nodes: set[torch.fx.Node] = set()
        for node in graph_module.graph.nodes:
            if node.target != exir_ops.edge.aten.permute_copy.default:
                continue

            start_permute = self.get_permutation(node)
            # Expected end permutation for the subgraph.
            end_permute = [start_permute.index(i) for i in range(len(start_permute))]

            for user in node.users:
                if user.target not in self.permutable_ops:
                    continue
                # Create a separate subgraph for each user since there may be cases
                # where only a portion of the users are permutable.
                subgraph = self.Subgraph(start_permute, end_permute)
                if self.visit(user, subgraph, processed_nodes):
                    subgraphs_found.append(subgraph)
                    for node in subgraph.nodes:
                        processed_nodes.add(node)

        for subgraph in subgraphs_found:
            self.permute_subgraph(subgraph)

        graph_module.graph.eliminate_dead_code()
        graph_module.recompile()

        return super().call(graph_module)

    def visit(
        self,
        node: torch.fx.Node,
        subgraph: Subgraph,
        processed_nodes: set[torch.fx.Node],
    ) -> bool:
        if node in subgraph.nodes:
            return True
        if node in processed_nodes or not self.is_node_permutable(node):
            return False
        subgraph.nodes.add(node)

        # Traverse downstream:
        for user in node.users:
            # Output should either go to a matching permute or another permutable op.
            if user.target == exir_ops.edge.aten.permute_copy.default:
                if self.get_permutation(user) != subgraph.end_permute:
                    return False
                subgraph.edges_out.add((node, user))
            elif not self.visit(user, subgraph, processed_nodes):
                return False

        # Traverse upstream:
        for inp in node.all_input_nodes:
            # Input should either come from a matching permute or another permutable op.
            if inp.target == exir_ops.edge.aten.permute_copy.default:
                if self.get_permutation(inp) != subgraph.start_permute:
                    return False
                subgraph.edges_in.add((inp, node))
            elif not self.visit(inp, subgraph, processed_nodes):
                return False

        return True

    def is_node_permutable(self, node: torch.fx.Node) -> bool:
        if node.target not in self.permutable_ops:
            return False
        if node.target == exir_ops.edge.aten.mean.dim:
            # keepdim should be True.
            if len(node.args) >= 3:
                if not node.args[2]:
                    return False
            elif "keepdim" in node.kwargs:
                if not node.kwargs["keepdim"]:
                    return False
            else:
                # Default keepdim is False.
                return False
        return True

    def permute_subgraph(self, subgraph: Subgraph) -> None:
        # Skip incoming permutes.
        for inp, out in subgraph.edges_in:
            assert inp.target == exir_ops.edge.aten.permute_copy.default
            if len(inp.args) >= 1:
                out.replace_input_with(inp, cast(torch.fx.Node, inp.args[0]))
            else:
                out.replace_input_with(inp, cast(torch.fx.Node, inp.kwargs["input"]))

        # Skip outgoing permutes.
        for inp, out in subgraph.edges_out:
            assert out.target == exir_ops.edge.aten.permute_copy.default
            out.replace_all_uses_with(inp)

        # Handle dimension related node arguments.
        for node in subgraph.nodes:
            if node.target == exir_ops.edge.aten.cat.default:
                self.update_cat(node, subgraph.start_permute)
            elif node.target == exir_ops.edge.aten.mean.dim:
                self.update_mean_dim(node, subgraph.start_permute)
            elif node.target == exir_ops.edge.aten.slice_copy.Tensor:
                self.update_slice_copy(node, subgraph.start_permute)

    def update_cat(self, node: torch.fx.Node, start_permute: list[int]) -> None:
        if len(node.args) >= 2:
            node.update_arg(1, start_permute[cast(int, node.args[1])])
        elif "dim" in node.kwargs:
            node.update_kwarg("dim", start_permute[cast(int, node.kwargs["dim"])])
        else:
            # Default cat dim is 0.
            node.update_kwarg("dim", start_permute[0])

    def update_mean_dim(self, node: torch.fx.Node, start_permute: list[int]) -> None:
        if len(node.args) >= 2:
            node.update_arg(
                1, [start_permute[dim] for dim in cast(list[int], node.args[1])]
            )
        else:
            node.update_kwarg(
                "dim",
                [start_permute[dim] for dim in cast(list[int], node.kwargs["dim"])],
            )

    def update_slice_copy(self, node: torch.fx.Node, start_permute: list[int]) -> None:
        if len(node.args) >= 2:
            node.update_arg(1, start_permute[cast(int, node.args[1])])
        else:
            node.update_kwarg("dim", start_permute[cast(int, node.kwargs["dim"])])

    def get_permutation(self, permute_node: torch.fx.Node) -> list[int]:
        assert permute_node.target == exir_ops.edge.aten.permute_copy.default
        if len(permute_node.args) >= 2:
            return cast(list[int], permute_node.args[1])
        assert "dim" in permute_node.kwargs
        return cast(list[int], permute_node.kwargs["dim"])


@register_cadence_pass(CadencePassAttribute(opt_level=2))
class RemoveSqueezeViewBeforeElementwiseOps(ExportPass):
    """
    Looks for subgraphs of the form:
    squeeze -> [elementwise ops] -> view
    and removes the squeeze node by reshaping the intermediate ops. If the final view
    is a corresponding unsqueeze it should also get eliminated by noop view elimination
    later. Only handles simple chain of intermediates now.

    The pass works on view ops instead of squeeze directly, thus it should be run after
    the squeeze/unsqueeze->view lowering.
    """

    intermediate_ops: set[EdgeOpOverload] = {
        exir_ops.edge.quantized_decomposed.quantize_per_tensor.default,
        exir_ops.edge.quantized_decomposed.dequantize_per_tensor.default,
        exir_ops.edge.cadence.quantize_per_tensor.default,
        exir_ops.edge.cadence.dequantize_per_tensor.default,
        # Ops that require special handling:
        exir_ops.edge.aten.slice_copy.Tensor,
    }

    def get_squeeze_indices(self, view_node: Node) -> List[int]:
        """
        Returns the indices of the input dimensions that are squeezed in the output if
        view node is a squeeze. Returns an empty list otherwise.
        """
        input_node = cast(Node, get_arg(view_node, "input"))
        input_shape = input_node.meta["val"].shape
        output_shape = view_node.meta["val"].shape

        if len(input_shape) <= len(output_shape):
            return []

        squeeze_indices = []
        out_idx = 0
        for idx, dim in enumerate(input_shape):
            if out_idx >= len(output_shape):
                return []
            if dim == output_shape[out_idx]:
                out_idx += 1
            else:
                # If there's a mismatch between the input and output dimensions, input
                # dimension has to be 1.
                if dim == 1:
                    squeeze_indices.append(idx)
                else:
                    return []

        # Check if all the output dimensions are consumed.
        if out_idx != len(output_shape):
            return []

        return squeeze_indices

    def handle_squeeze(self, view_node: Node, visited_view_nodes: Set[Node]) -> None:
        if view_node in visited_view_nodes:
            return

        squeeze_indices = self.get_squeeze_indices(view_node)
        if not squeeze_indices:
            return

        # Only handle simple chains for now.
        if len(view_node.users) != 1:
            return
        node = next(iter(view_node.users))

        # Traverse down from the node until finding another view op.
        intermediate_slices = []
        while node.target != exir_ops.edge.aten.view_copy.default:
            # Only handle simple chains for now
            if len(node.users) != 1:
                return
            if node.target not in self.intermediate_ops:
                return
            if node.target == exir_ops.edge.aten.slice_copy.Tensor:
                intermediate_slices.append(node)
            node = next(iter(node.users))

        # View node found. We can't optimize this view_node again since the
        # input shape is invalid now so add it to the visited set.
        visited_view_nodes.add(node)

        # Update the intermediate slices.
        for slice_node in intermediate_slices:
            slice_rank = len(slice_node.meta["val"].shape)
            slice_dim = cast(int, get_arg(slice_node, "dim"))
            if slice_dim < 0:
                slice_dim += slice_rank
            for squeeze_dim in squeeze_indices:
                if slice_dim >= squeeze_dim:
                    slice_dim += 1
            set_arg(slice_node, "dim", slice_dim)

        # Skip the initial view node.
        input_node = cast(Node, get_arg(view_node, "input"))
        view_node.replace_all_uses_with(input_node)

    def call(self, graph_module: torch.fx.GraphModule) -> PassResult:
        visited_view_nodes = set()
        for view_node in graph_module.graph.find_nodes(
            op="call_function", target=exir_ops.edge.aten.view_copy.default, sort=True
        ):
            self.handle_squeeze(view_node, visited_view_nodes)

        graph_module.graph.eliminate_dead_code()
        graph_module.recompile()

        return super().call(graph_module)


@register_cadence_pass(CadencePassAttribute(opt_level=1))
class RemoveBranchedQuantDequant(ExportPass):
    """
    This pass looks for adjacent quant and dequant nodes with identical
    parameters, where the quant node has other users in addition to the
    dequant. The quant and dequant pair would be removed by the
    FuseQuantDequantToRequantizePass if not for the multiple users. This pass
    removes just the dequant node by connecting it to the quant's parent node
    """

    quantize_op_packets: set[EdgeOpOverloadPacket] = {
        exir_ops.edge.cadence.quantize_per_tensor,
        exir_ops.edge.quantized_decomposed.quantize_per_tensor,
    }
    dequantize_op_packets: set[EdgeOpOverloadPacket] = {
        exir_ops.edge.cadence.dequantize_per_tensor,
        exir_ops.edge.quantized_decomposed.dequantize_per_tensor,
    }

    def call(self, graph_module: torch.fx.GraphModule) -> PassResult:
        self.remove_branched(
            graph_module, self.quantize_op_packets, self.dequantize_op_packets
        )
        self.remove_branched(
            graph_module, self.dequantize_op_packets, self.quantize_op_packets
        )

        graph_module.graph.eliminate_dead_code()
        result = super().call(graph_module)
        return result

    def remove_branched(
        self,
        graph_module: torch.fx.GraphModule,
        producer_pkts: set[EdgeOpOverloadPacket],
        consumer_pkts: set[EdgeOpOverloadPacket],
    ) -> None:
        for node in graph_module.graph.nodes:
            if (
                node.op != "call_function"
                or not isinstance(node.target, EdgeOpOverload)
                or get_edge_overload_packet(node.target) not in producer_pkts
            ):
                continue

            if len(node.users) < 2:
                continue

            for user in node.users:
                if (
                    not isinstance(user.target, EdgeOpOverload)
                    or get_edge_overload_packet(user.target) not in consumer_pkts
                ):
                    continue

                # check qparams match
                if node.args[1:] != user.args[1:]:
                    continue

                user.replace_all_uses_with(node.args[0])


class RemoveCatFromSliceCopyPass(ExportPass):
    """
    Simplifies cat->slice_copy chains where one of the cat inputs can be directly passed
    to the slice_copy.
    """

    def _remove_unused_cat(self, graph_module: torch.fx.GraphModule) -> None:
        for slice_copy_node in graph_module.graph.find_nodes(
            op="call_function", target=exir_ops.edge.aten.slice_copy.Tensor
        ):
            cat_node = cast(Node, get_arg(slice_copy_node, "input"))
            slice_dim = cast(int, get_arg(slice_copy_node, "dim"))
            start_idx = cast(int, get_arg(slice_copy_node, "start"))
            end_idx = cast(int, get_arg(slice_copy_node, "end"))
            step = cast(int, get_arg(slice_copy_node, "step"))

            if cat_node.target != exir_ops.edge.aten.cat.default or step != 1:
                continue

            # Make sure cat and slice happens on the same dimension.
            cat_dim = cast(Node, get_arg(cat_node, "dim"))
            if cat_dim != slice_dim:
                continue

            # Canonicalize slice indices.
            cat_output_shape = cat_node.meta["val"].shape
            if start_idx is None:
                start_idx = 0
            elif start_idx < 0:
                start_idx += cat_output_shape[cat_dim]
            if end_idx is None or end_idx > cat_output_shape[cat_dim]:
                end_idx = cat_output_shape[cat_dim]
            elif end_idx < 0:
                end_idx += cat_output_shape[cat_dim]

            offset = 0
            for cat_input_node in cast(List[Node], get_arg(cat_node, "tensors")):
                cat_input_shape = cat_input_node.meta["val"].shape

                # Check if the slice range overlaps with the cat input range.
                if offset <= start_idx and end_idx <= offset + cat_input_shape[cat_dim]:
                    slice_copy_node.replace_input_with(cat_node, cat_input_node)
                    set_arg(slice_copy_node, "start", start_idx - offset)
                    set_arg(slice_copy_node, "end", end_idx - offset)
                    break

                offset += cat_input_shape[cat_dim]

    def call(self, graph_module: torch.fx.GraphModule) -> PassResult:
        self._remove_unused_cat(graph_module)
        graph_module.recompile()
        graph_module.graph.eliminate_dead_code()
        return super().call(graph_module)


class CommonRemovePasses:
    passes: List[Type[ExportPass]] = [
        RemoveCloneOpPass,
        RemoveAliasCopyOpPass,
        RemoveNopExpandOpPass,
        RemoveNopSliceOrViewOpPass,
        RemoveNopSelectOpPass,
        RemoveToOpsPass,
        RemoveZeroSizedCatArgsPass,
    ]


class CadenceRemoveNops:
    passes: List[Type[ExportPass]] = CommonRemovePasses.passes + [
        SimplifySliceOpPass,
        RemoveCloneOpsTransformImported,
        RemoveNopRequantizeOpPass,
        RemoveZeroSizedConstantPadNd,
        RemoveContiguousOpPass,
        RemoveNopMulOpPass,
        RemoveNopAddOpPass,
        RemoveNopLinalgVectorNormOpPass,
        RemoveBranchedQuantDequant,
    ]
