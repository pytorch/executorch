# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# This pass replaces view_copy ops with view ops. This is different than
# exir/passes/replace_view_copy_with_view.py and exir/passes/reinplace.py
# because this should only be used in the AOTInductor backend, as it
# has less restrictions on whether the tensor memory is densely packed,

from typing import Dict, Iterable

import torch
from executorch.exir.dialects._ops import ops
from executorch.exir.dialects.edge._ops import EdgeOpOverload
from executorch.exir.pass_base import ExportPass, PassResult
from torch import fx


_VIEW_TARGETS: Dict[
    torch._ops.OpOverload | EdgeOpOverload, torch._ops.OpOverload | EdgeOpOverload
] = {
    torch.ops.aten.slice_copy.Tensor: torch.ops.aten.slice.Tensor,
    ops.edge.aten.slice_copy.Tensor: ops.edge.aten.slice.Tensor,
    torch.ops.aten.select_copy.int: torch.ops.aten.select.int,
    ops.edge.aten.select_copy.int: ops.edge.aten.select.int,
    # ``split_copy`` has no c-shim and falls back to the proxy executor, which
    # the AOTI runtime does not support. Its view form ``split`` (same arg
    # signature, list-of-views return consumed by ``getitem``) is codegen'd
    # natively by inductor, so map it here like the other view-copy ops.
    torch.ops.aten.split_copy.Tensor: torch.ops.aten.split.Tensor,
    ops.edge.aten.split_copy.Tensor: ops.edge.aten.split.Tensor,
    # Edge turns ``detach`` into ``detach_copy``, which has no c-shim either.
    # Without autograd it is ``alias_copy``, which Inductor can decompose,
    # unlike ``detach_copy``. A view would let two outputs share one handle,
    # and ``clone`` fails Edge verification in dim order mode.
    torch.ops.aten.detach_copy.default: torch.ops.aten.alias_copy.default,
    ops.edge.aten.detach_copy.default: ops.edge.aten.alias_copy.default,
}


class ReplaceViewCopyWithViewPass(ExportPass):
    """Replace non-mutated ``view_copy`` type of ops with ``view`` ops."""

    def call(self, graph_module: fx.GraphModule) -> PassResult:
        graph_changed = False

        # Control flow branches, such as those of torch.cond, are submodules.
        for module in graph_module.modules():
            if not isinstance(module, fx.GraphModule):
                continue

            module_changed = False
            for node in module.graph.nodes:
                if node.op != "call_function" or node.target not in _VIEW_TARGETS:
                    continue

                if self._has_blocking_user(node, node.users.keys()):
                    continue

                node.target = _VIEW_TARGETS[node.target]
                module_changed = True

            if module_changed:
                module.graph.lint()
                module.recompile()
                graph_changed = True

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

    def _argument_mutates(
        self, schema: torch._C.FunctionSchema, key: int | str
    ) -> bool:
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
