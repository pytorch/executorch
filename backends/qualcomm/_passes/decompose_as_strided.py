# Copyright (c) Qualcomm Innovation Center, Inc.
# All rights reserved
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import math

import torch
from executorch.exir.dialects._ops import ops as exir_ops
from executorch.exir.dialects.edge._ops import EdgeOpOverload
from executorch.exir.pass_base import ExportPass, PassResult

from .utils import copy_meta, merge_decomposed_graph


class AsStridedDecomposed(torch.nn.Module):
    """Decompose as_strided into flatten + index_select + reshape.

    as_strided reinterprets memory layout using size/stride/storage_offset.
    We precompute the linear indices and use index_select on the flattened input.
    """

    def __init__(self, output_size, stride, storage_offset):
        super().__init__()
        indices = self._compute_indices(output_size, stride, storage_offset)
        self.register_buffer("indices", indices)
        self.output_size = output_size

    def _compute_indices(self, size, stride, storage_offset):
        """Compute flat linear indices for as_strided access pattern.

        indices[i0, i1, ..., in] = storage_offset + sum(ij * stride[j])
        Use int32 so the buffer is compatible with QNN (which does not support int64).
        """
        indices = torch.zeros(size, dtype=torch.int32)
        for dim in range(len(size)):
            shape = [1] * len(size)
            shape[dim] = size[dim]
            indices = (
                indices
                + torch.arange(size[dim], dtype=torch.int32).reshape(shape)
                * stride[dim]
            )
        indices = indices + storage_offset
        return indices.flatten()

    def forward(self, x):
        flat = x.reshape(-1)
        gathered = torch.index_select(flat, 0, self.indices)
        return gathered.reshape(self.output_size)


class DecomposeAsStrided(ExportPass):
    """
    Decompose aten.as_strided into supported QNN primitives.

    as_strided creates a view of a tensor with specified size, stride, and
    storage_offset. Since QNN has no native as_strided op, we decompose it:

    Fast path (pure view):
        If strides are contiguous for the output size and storage_offset == 0,
        this is equivalent to a reshape/view_copy.

    General path (via module export + merge_decomposed_graph):
        1. Flatten input to 1D
        2. index_select with precomputed linear indices (int32 constant)
        3. Reshape to target output size

    The general path materializes one int32 index per output element and is
    intended for modest output sizes.
    """

    def __init__(self) -> None:
        super().__init__()
        self._targets = {
            torch.ops.aten.as_strided.default,
            torch.ops.aten.as_strided_copy.default,
            exir_ops.edge.aten.as_strided_copy.default,
        }

    @staticmethod
    def _is_contiguous_strides(size, stride):
        """Check if strides correspond to a contiguous layout for the given size."""
        if not size:
            return True
        expected_stride = 1
        for i in range(len(size) - 1, -1, -1):
            if stride[i] != expected_stride:
                return False
            expected_stride *= size[i]
        return True

    @staticmethod
    def _max_index(size, stride, storage_offset):
        """Compute maximum linear index that as_strided would access."""
        if not size or math.prod(size) == 0:
            return -1
        return storage_offset + sum((s - 1) * st for s, st in zip(size, stride))

    def _decompose_as_view(self, node, graph, input_node, output_size):
        """Fast path: replace as_strided with a view/reshape."""
        input_val = input_node.meta["val"]
        if list(input_val.shape) != output_size:
            is_edge = isinstance(node.target, EdgeOpOverload)
            view_op = (
                exir_ops.edge.aten.view_copy.default
                if is_edge
                else torch.ops.aten.view.default
            )
            with graph.inserting_before(node):
                view_node = graph.create_node(
                    "call_function",
                    view_op,
                    (input_node, output_size),
                )
                view_node.meta = copy_meta(node.meta)
            for user in node.users.copy():
                user.replace_input_with(node, view_node)
        else:
            for user in node.users.copy():
                user.replace_input_with(node, input_node)

    def _decompose_with_gather(  # noqa: C901
        self, node, graph, graph_module, input_node, output_size, stride, storage_offset
    ):
        """General path: decompose via flatten + index_select + reshape."""
        model = AsStridedDecomposed(output_size, stride, storage_offset)

        input_fake = input_node.meta["val"]
        decomposed_module = torch.export.export(
            model,
            (input_fake,),
            strict=True,
        ).module()

        # Register the indices buffer with a unique name
        buffer_name = f"_as_strided_indices_{node.name}"
        graph_module.register_buffer(buffer_name, model.indices)

        # Rename get_attr nodes in decomposed graph to use the unique buffer name
        for d_node in decomposed_module.graph.nodes:
            if d_node.op == "get_attr" and d_node.target == "indices":
                d_node.target = buffer_name

        # Set attribute on decomposed module so node_copy works
        setattr(decomposed_module, buffer_name, model.indices)

        with graph.inserting_before(node):
            remap = {"x": input_node}
            merge_decomposed_graph(
                remap=remap,
                target_node=node,
                target_graph=graph,
                decomposed_graph_module=decomposed_module,
            )
            graph.erase_node(node)

    def call(self, graph_module: torch.fx.GraphModule) -> PassResult:
        graph = graph_module.graph
        modified = False

        for node in list(graph.nodes):
            if node.op != "call_function" or node.target not in self._targets:
                continue

            input_node = node.args[0]
            size_arg = node.args[1]
            stride_arg = node.args[2]
            storage_offset = node.args[3] if len(node.args) > 3 else 0
            if storage_offset is None:
                storage_offset = 0

            # Guard: skip SymInt (dynamic shapes)
            if not all(isinstance(s, int) for s in size_arg):
                continue
            if not all(isinstance(s, int) for s in stride_arg):
                continue

            output_size = list(size_arg)
            stride = list(stride_arg)

            # Guard: skip empty output
            if math.prod(output_size) == 0:
                continue

            input_val = input_node.meta["val"]

            # Both decomposition paths interpret flattened contiguous storage.
            # Leave non-contiguous inputs unchanged rather than alter their
            # storage semantics.
            if not input_val.is_contiguous():
                continue

            # Guard: validate index bounds
            max_idx = self._max_index(output_size, stride, storage_offset)
            assert max_idx < input_val.numel(), (
                f"DecomposeAsStrided: indices out of bounds. "
                f"max_index={max_idx} >= input_numel={input_val.numel()} "
                f"(size={output_size}, stride={stride}, offset={storage_offset})"
            )

            if (
                storage_offset == 0
                and self._is_contiguous_strides(output_size, stride)
                and math.prod(output_size) == input_val.numel()
            ):
                self._decompose_as_view(node, graph, input_node, output_size)
            else:
                self._decompose_with_gather(
                    node,
                    graph,
                    graph_module,
                    input_node,
                    output_size,
                    stride,
                    storage_offset,
                )
            modified = True

        if not modified:
            return PassResult(graph_module, False)

        graph.eliminate_dead_code()
        graph_module.recompile()
        return PassResult(graph_module, True)
