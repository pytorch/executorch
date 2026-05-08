# Copyright 2026 Arm Limited and/or its affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from typing import Any, Set, Type

import torch
from executorch.backends.arm._passes import ArmPass
from executorch.backends.arm._passes.arm_pass_utils import (
    create_node,
    get_first_fake_tensor,
    is_param_node,
)
from executorch.exir import ExportedProgram
from executorch.exir.dialects._ops import ops as exir_ops
from executorch.exir.pass_base import ExportPass, PassResult


class NormalizeDelegateIOLayoutPass(ArmPass):
    """Adjust delegated boundary tensor shapes and insert permutes at I/O."""

    _passes_required_after: Set[Type[ExportPass]] = set()

    def __init__(self, exported_program: ExportedProgram, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.exported_program = exported_program

    @staticmethod
    def _inverse_permutation(perm: tuple[int, ...]) -> tuple[int, ...]:
        inverse = [0] * len(perm)
        for idx, axis in enumerate(perm):
            inverse[axis] = idx
        return tuple(inverse)

    @staticmethod
    def _permute_shape(shape: torch.Size, perm: tuple[int, ...]) -> tuple[int, ...]:
        return tuple(shape[axis] for axis in perm)

    @staticmethod
    def _is_identity_dim_order(dim_order: tuple[int, ...]) -> bool:
        return dim_order == tuple(range(len(dim_order)))

    def _normalize_input_layout(self, graph_module: torch.fx.GraphModule) -> bool:
        modified = False
        for node in graph_module.graph.nodes:
            if node.op != "placeholder" or is_param_node(self.exported_program, node):
                continue

            input_fake = get_first_fake_tensor(node)
            dim_order = input_fake.dim_order()
            if self._is_identity_dim_order(dim_order):
                continue

            boundary_shape = self._permute_shape(input_fake.shape, dim_order)
            node.meta["val"] = input_fake.reshape(boundary_shape)

            transpose_perm = self._inverse_permutation(dim_order)
            with graph_module.graph.inserting_after(node):
                permute_node = create_node(
                    graph_module.graph,
                    exir_ops.edge.aten.permute_copy.default,
                    args=(node, list(transpose_perm)),
                    from_node=node,
                )
                permute_node.meta["val"] = exir_ops.edge.aten.permute_copy.default(
                    node.meta["val"], list(transpose_perm)
                )

            users = [user for user in node.users if user != permute_node]
            for user in users:
                user.replace_input_with(node, permute_node)

            modified = True

        return modified

    def _rewrite_output_arg(
        self, arg: Any, graph_module: torch.fx.GraphModule
    ) -> tuple[Any, bool]:
        if isinstance(arg, torch.fx.Node):
            output_fake = get_first_fake_tensor(arg)
            dim_order = output_fake.dim_order()
            if self._is_identity_dim_order(dim_order):
                return arg, False

            with graph_module.graph.inserting_after(arg):
                permute_node = create_node(
                    graph_module.graph,
                    exir_ops.edge.aten.permute_copy.default,
                    args=(arg, list(dim_order)),
                    from_node=arg,
                )
                permute_node.meta["val"] = exir_ops.edge.aten.permute_copy.default(
                    output_fake, list(dim_order)
                )

            return permute_node, True

        if isinstance(arg, tuple):
            modified = False
            rewritten = []
            for item in arg:
                new_item, item_modified = self._rewrite_output_arg(item, graph_module)
                rewritten.append(new_item)
                modified = modified or item_modified
            return tuple(rewritten), modified

        if isinstance(arg, list):
            modified = False
            rewritten = []
            for item in arg:
                new_item, item_modified = self._rewrite_output_arg(item, graph_module)
                rewritten.append(new_item)
                modified = modified or item_modified
            return rewritten, modified

        return arg, False

    def _normalize_output_layout(self, graph_module: torch.fx.GraphModule) -> bool:
        output_node = graph_module.graph.output_node()
        rewritten_outputs, modified = self._rewrite_output_arg(
            output_node.args[0], graph_module
        )
        if modified:
            output_node.args = (rewritten_outputs,)
        return modified

    def call(self, graph_module: torch.fx.GraphModule) -> PassResult:
        modified = self._normalize_input_layout(graph_module)
        modified = self._normalize_output_layout(graph_module) or modified

        if modified:
            graph_module.recompile()
            graph_module = super().call(graph_module).graph_module

        return PassResult(graph_module, modified)
