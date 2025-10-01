# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

from typing import Iterable

import torch
from executorch.exir.dialects._ops import ops
from executorch.exir.pass_base import ExportPass, PassResult
from torch import fx


_SLICE_COPY_TARGETS = (
    torch.ops.aten.slice_copy.Tensor,
    ops.edge.aten.slice_copy.Tensor,
)

_SLICE_TARGETS = {
    torch.ops.aten.slice_copy.Tensor: torch.ops.aten.slice.Tensor,
    ops.edge.aten.slice_copy.Tensor: ops.edge.aten.slice.Tensor,
}


class ReplaceSliceCopyWithSlicePass(ExportPass):
    """Replace non-mutated ``slice_copy`` results with ``slice`` views."""

    def call(self, graph_module: fx.GraphModule) -> PassResult:
        graph_changed = False

        for node in graph_module.graph.nodes:
            if node.op != "call_function" or node.target not in _SLICE_COPY_TARGETS:
                continue

            if self._has_blocking_user(node, node.users.keys()):
                continue

            node.target = _SLICE_TARGETS[node.target]
            graph_changed = True

        if graph_changed:
            graph_module.graph.lint()
            graph_module.recompile()

        return PassResult(graph_module, graph_changed)

    def _has_blocking_user(self, node: fx.Node, users: Iterable[fx.Node]) -> bool:
        for user in users:
            if self._is_mutating_user(node, user) or self._is_view_user(node, user):
                return True
        return False

    def _is_mutating_user(self, node: fx.Node, user: fx.Node) -> bool:
        if user.op == "call_method":
            # Treat in-place tensor methods conservatively as mutations only when the
            # method name ends with ``_`` which is the PyTorch convention for mutation.
            return isinstance(user.target, str) and user.target.endswith("_")

        if user.op != "call_function":
            return False

        target = user.target
        if not hasattr(target, "_schema"):
            return False

        schema = target._schema  # pyre-ignore[16]
        # Positional arguments
        for index, arg in enumerate(user.args):
            if arg is node and self._argument_mutates(schema, index):
                return True

        # Keyword arguments
        for name, arg in user.kwargs.items():
            if arg is node and self._argument_mutates(schema, name):
                return True

        return False

    def _is_view_user(self, node: fx.Node, user: fx.Node) -> bool:
        if user.op == "call_method":
            # Treat tensor methods conservatively and assume they may be view-producing.
            return True

        if user.op != "call_function":
            return False

        target = user.target
        if getattr(target, "is_view", False):
            for arg in user.args:
                if arg is node:
                    return True
            for arg in user.kwargs.values():
                if arg is node:
                    return True

        return False

    def _argument_mutates(self, schema: torch._C.FunctionSchema, key) -> bool:  # pyre-ignore[11]
        arguments = schema.arguments
        if isinstance(key, int):
            if key >= len(arguments):
                return False
            argument = arguments[key]
        else:
            argument = next((arg for arg in arguments if arg.name == key), None)
            if argument is None:
                return False

        alias_info = argument.alias_info
        return bool(alias_info and alias_info.is_write)
