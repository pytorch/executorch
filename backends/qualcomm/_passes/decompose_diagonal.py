# Copyright (c) Qualcomm Innovation Center, Inc.
# All rights reserved
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import torch
from executorch.exir.dialects._ops import ops as exir_ops
from executorch.exir.dialects.edge._ops import EdgeOpOverload
from executorch.exir.pass_base import ExportPass, PassResult

from .utils import copy_meta


class DecomposeDiagonal(ExportPass):
    """
    Decompose diagonal operation into permute + view + arange + index_select.

    torch.diagonal(input, offset=0, dim1=0, dim2=1) extracts diagonal elements from the 2D submatrix defined by dim1 and dim2.

    Decomposition strategy:
        1. Permute input so dim1 and dim2 are the last two dimensions.
        2. Reshape (view) to flatten the last two dims: [..., M*N]
        3. Compute flat diagonal indices via arange(start, end, stride).
        4. Use index_select on the last dim with the computed indices.
    """

    def __init__(self) -> None:
        super().__init__()
        self._edge_targets = {
            exir_ops.edge.aten.diagonal_copy.default,
        }
        self._aten_targets = {
            torch.ops.aten.diagonal.default,
        }

    def _get_ops(self, is_edge):
        if is_edge:
            return {
                "permute": exir_ops.edge.aten.permute_copy.default,
                "view": exir_ops.edge.aten.view_copy.default,
                "arange": exir_ops.edge.aten.arange.start_step,
                "index_select": exir_ops.edge.aten.index_select.default,
                "full": exir_ops.edge.aten.full.default,
            }
        return {
            "permute": torch.ops.aten.permute.default,
            "view": torch.ops.aten.view.default,
            "arange": torch.ops.aten.arange.start_step,
            "index_select": torch.ops.aten.index_select.default,
            "full": torch.ops.aten.full.default,
        }

    def _compute_diag_params(self, M, N, offset):
        """Compute diagonal size, start offset, and stride from matrix dims and offset."""
        if offset >= 0:
            diag_size = min(M, N - offset)
            start_offset = offset
        else:
            diag_size = min(M + offset, N)
            start_offset = (-offset) * N
        stride = N + 1
        return diag_size, start_offset, stride

    def _compute_layout(self, input_shape, dim1, dim2):
        """Compute permutation order, remaining dims, and flattened shape."""
        ndim = len(input_shape)
        M, N = input_shape[dim1], input_shape[dim2]
        remaining_dims = [i for i in range(ndim) if i != dim1 and i != dim2]
        perm = remaining_dims + [dim1, dim2]
        remaining_shapes = [input_shape[i] for i in remaining_dims]
        flattened_shape = remaining_shapes + [M * N]
        need_permute = perm != list(range(ndim))
        return remaining_dims, perm, flattened_shape, need_permute

    def _build_diagonal_graph(
        self,
        node,
        graph,
        ops,
        input_node,
        input_val,
        perm,
        flattened_shape,
        need_permute,
        start_offset,
        diag_size,
        stride,
    ):
        """Build the decomposed graph: permute → view → arange → index_select."""
        meta = node.meta
        fake_mode = meta["val"].fake_mode

        with graph.inserting_before(node):
            # Step 1: Permute dim1, dim2 to last positions (if needed)
            if need_permute:
                permute_node = graph.create_node(
                    "call_function", ops["permute"], (input_node, perm)
                )
                permute_node.meta = copy_meta(
                    meta, lambda m: {**m, "val": input_val.permute(perm)}
                )
                reshape_input = permute_node
            else:
                reshape_input = input_node

            # Step 2: Reshape [..., M, N] -> [..., M*N]
            view_node = graph.create_node(
                "call_function", ops["view"], (reshape_input, flattened_shape)
            )
            if need_permute:
                view_val = input_val.permute(perm).contiguous().view(flattened_shape)
            else:
                view_val = input_val.contiguous().view(flattened_shape)
            view_node.meta = copy_meta(meta, lambda m: {**m, "val": view_val})

            # Step 3: Compute flat diagonal indices
            arange_end = start_offset + diag_size * stride
            arange_node = graph.create_node(
                "call_function",
                ops["arange"],
                (start_offset, arange_end, stride),
                {
                    "dtype": torch.int32,
                    "layout": torch.strided,
                    "device": torch.device("cpu"),
                    "pin_memory": False,
                },
            )
            arange_node.meta = copy_meta(
                meta,
                lambda m: {
                    **m,
                    "val": fake_mode.from_tensor(
                        torch.arange(
                            start_offset, arange_end, stride, dtype=torch.int32
                        )
                    ),
                },
            )

            # Step 4: index_select on last dim
            last_dim = len(flattened_shape) - 1
            index_select_node = graph.create_node(
                "call_function",
                ops["index_select"],
                (view_node, last_dim, arange_node),
            )
            index_select_node.meta = copy_meta(meta)

        return index_select_node

    def _decompose_diagonal(self, node, graph):
        input_node = node.args[0]
        is_edge = isinstance(node.target, EdgeOpOverload)
        ops = self._get_ops(is_edge)

        # Parse diagonal args: diagonal(input, offset=0, dim1=0, dim2=1)
        offset = node.args[1] if len(node.args) > 1 else 0
        dim1 = node.args[2] if len(node.args) > 2 else 0
        dim2 = node.args[3] if len(node.args) > 3 else 1

        # Get input shape from meta
        input_val = input_node.meta["val"]
        input_shape = list(input_val.shape)
        ndim = len(input_shape)

        # Normalize negative dims
        if dim1 < 0:
            dim1 += ndim
        if dim2 < 0:
            dim2 += ndim

        M, N = input_shape[dim1], input_shape[dim2]

        # Compute diagonal parameters
        diag_size, start_offset, stride = self._compute_diag_params(M, N, offset)

        if diag_size <= 0:
            # Match PyTorch behavior: return empty tensor when offset exceeds dims
            remaining_dims = [i for i in range(ndim) if i != dim1 and i != dim2]
            empty_shape = [input_shape[i] for i in remaining_dims] + [0]
            with graph.inserting_before(node):
                empty_node = graph.create_node(
                    "call_function", ops["full"], (empty_shape, 0.0)
                )
                empty_node.meta = copy_meta(node.meta)
            for user in node.users.copy():
                user.replace_input_with(node, empty_node)
            return

        # Compute layout
        remaining_dims, perm, flattened_shape, need_permute = self._compute_layout(
            input_shape, dim1, dim2
        )

        # Build decomposed graph
        result_node = self._build_diagonal_graph(
            node,
            graph,
            ops,
            input_node,
            input_val,
            perm,
            flattened_shape,
            need_permute,
            start_offset,
            diag_size,
            stride,
        )

        # Replace original node
        for user in node.users.copy():
            user.replace_input_with(node, result_node)

    def call(self, graph_module: torch.fx.GraphModule) -> PassResult:
        graph = graph_module.graph

        all_targets = self._edge_targets | self._aten_targets
        nodes_to_decompose = [
            n
            for n in graph.nodes
            if n.op == "call_function" and n.target in all_targets
        ]

        if not nodes_to_decompose:
            return PassResult(graph_module, False)

        for node in nodes_to_decompose:
            self._decompose_diagonal(node, graph)

        graph.eliminate_dead_code()
        graph_module.recompile()
        return PassResult(graph_module, True)
