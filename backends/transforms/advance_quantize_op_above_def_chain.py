# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-unsafe

from math import prod

import torch
import torch.fx
from executorch.backends.transforms.permute_pass_utils import (
    get_arg,
    get_edge_overload_packet,
    get_overload_packet,
    get_shape,
)
from executorch.backends.transforms.quantize_reorder_utils import (
    slice_or_select_overloadpkt,
    trivially_quantizable_ops_overloadpkt,
)
from executorch.exir.dialects._ops import ops as exir_ops
from executorch.exir.dialects.edge._ops import EdgeOpOverload
from executorch.exir.pass_base import ExportPass, PassResult


class AdvanceQuantizeOpAboveDefChainPass(ExportPass):
    """
    Advances a quantize op above data-movement ops to reduce data volume.

    Handles two cases:

    1. Linear chain: if the input to a quantize op is a chain of trivially
       quantizable ops (view, transpose, permute, slice), rewrite
       data_movement(fp32) -> quantize to quantize -> data_movement(quantized)
       so the data movement operates on smaller quantized tensors.

    2. Cat: if the input to a quantize op is a cat with a single user (the
       quantize), advance the quantize above the cat by quantizing each cat
       input individually.  A later pass can clean up any redundant
       dequant-quant pairs on the inputs.

    3. Caller-supplied ops: advance the quantize above a value-preserving op by
       quantizing selected direct floating-point tensor inputs. The caller is
       responsible for supplying only inputs that support the quantized dtype.

    For the cat case, SplitDequantizedCatPass should run first to ensure
    each cat has at most one quantize consumer.
    """

    quantize_op_packets: set[object] = {
        exir_ops.edge.quantized_decomposed.quantize_per_tensor,
        torch.ops.quantized_decomposed.quantize_per_tensor,
    }

    def __init__(
        self,
        extra_quantizable_ops: dict[EdgeOpOverload, tuple[int, ...]] | None = None,
    ) -> None:
        super().__init__()
        self.graph_module = None
        self._extra_quantizable_ops = extra_quantizable_ops or {}

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

    def _advance_above_cat(
        self, quant_node: torch.fx.Node, cat_node: torch.fx.Node
    ) -> None:
        """Advance a quantize op above a cat by quantizing each cat input."""
        graph = quant_node.graph
        quant_params = quant_node.args[1:]

        cat_inputs = cat_node.args[0]
        assert isinstance(cat_inputs, (list, tuple))

        new_inputs: list[torch.fx.Node] = []
        for inp in cat_inputs:
            # cat concatenates tensors, so every input must be a node.
            assert isinstance(inp, torch.fx.Node)

            with graph.inserting_before(cat_node):
                new_quant = graph.call_function(
                    # pyre-ignore[6]
                    quant_node.target,
                    args=(inp, *quant_params),
                )
                # This copies the fp32 input's meta, so meta["val"] keeps the
                # fp32 dtype rather than the quantized output dtype. That's fine:
                # nothing in this pass reads dtype from meta (only shape, which
                # is correct), and call() re-runs super().call() to re-propagate
                # fake tensors, making meta dtype-consistent before we return.
                new_quant.meta = inp.meta.copy()
            new_inputs.append(new_quant)

        dim = get_arg(cat_node, "dim", int)
        with graph.inserting_before(quant_node):
            new_cat = graph.call_function(
                # pyre-ignore[6]
                cat_node.target,
                args=(new_inputs, dim),
            )
            new_cat.meta = quant_node.meta.copy()

        quant_node.replace_all_uses_with(new_cat)
        graph.erase_node(quant_node)

    def _advance_above_extra_quantizable_op(
        self,
        quant_node: torch.fx.Node,
        op_node: torch.fx.Node,
        quantizable_input_indices: tuple[int, ...],
    ) -> bool:
        graph = quant_node.graph
        new_op_args = list(op_node.args)
        quantized_input = False

        for index in quantizable_input_indices:
            assert 0 <= index < len(op_node.args)
            arg = op_node.args[index]
            assert isinstance(arg, torch.fx.Node)
            value = arg.meta["val"]
            assert isinstance(value, torch.Tensor)
            if not value.dtype.is_floating_point:
                continue

            quant_args = list(quant_node.args)
            quant_args[0] = arg
            with graph.inserting_before(op_node):
                new_quant = graph.call_function(
                    # pyre-ignore[6]
                    quant_node.target,
                    args=tuple(quant_args),
                    kwargs=quant_node.kwargs,
                )
                # We will correct the dtype when we run
                # ExportPass call at the end.
                new_quant.meta = arg.meta.copy()
            new_op_args[index] = new_quant
            quantized_input = True

        if not quantized_input:
            return False

        with graph.inserting_before(quant_node):
            new_op = graph.call_function(
                # pyre-ignore[6]
                op_node.target,
                args=tuple(new_op_args),
                kwargs=op_node.kwargs,
            )
            new_op.meta = quant_node.meta.copy()

        quant_node.replace_all_uses_with(new_op)
        graph.erase_node(quant_node)
        return True

    def advance_quantize_op(self, graph_module: torch.fx.GraphModule) -> bool:
        graph = graph_module.graph
        modified = False
        for node in reversed(graph.nodes):
            if get_overload_packet(node.target) not in self.quantize_op_packets:
                continue

            inp = node.args[0]
            if (
                isinstance(inp, torch.fx.Node)
                and get_overload_packet(inp.target)
                in (exir_ops.edge.aten.cat, torch.ops.aten.cat)
                and len(inp.users) == 1
            ):
                self._advance_above_cat(node, inp)
                modified = True
                continue

            if (
                isinstance(inp, torch.fx.Node)
                and inp.target in self._extra_quantizable_ops
                and len(inp.users) == 1
                and self._advance_above_extra_quantizable_op(
                    node,
                    inp,
                    self._extra_quantizable_ops[inp.target],
                )
            ):
                modified = True
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
