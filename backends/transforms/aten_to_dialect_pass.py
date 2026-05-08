# Copyright 2026 Arm Limited and/or its affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.


import traceback
from collections.abc import Callable
from dataclasses import dataclass
from typing import ClassVar, TypeAlias

import torch

from executorch.backends.xnnpack._passes.xnnpack_pass import ExportPass

from executorch.exir import ExportedProgram
from torch.fx.node import Target
from torch.fx.passes.infra.pass_manager import PassResult


# Expected type to be returned by substitution functions.
@dataclass
class DialectNodeSpec:
    op: Target
    args: tuple
    kwargs: dict = None


# Expected type to be used for substitution functions
SubstitutionFn: TypeAlias = Callable[
    [torch.fx.Node, torch.export.ExportedProgram], DialectNodeSpec | None
]


class AtenToDialectPass(ExportPass):
    """
    General pass to convert ops 1-1 from ATen to a specific dialect.

    Usage:
        1. Subclass the pass for a specific dialect
        2. For each ATen target to be substituted, implement a function returning a DialectNodeSpec defining the
           corresponding dialect op, or None if the substitution does not apply.
        3. Register each substitution function for the subclass using the decorator register_dialect_substitution

    Only one substitution function can be registered for a given target.

    The pass must be initialized with an exported_program to allow substitution functions to modify placeholders,
    e.g. if the dialect ops require additional scratch buffers.
    """

    _DIALECT_SUBSTITUTIONS: ClassVar[dict[Target, SubstitutionFn]] = {}

    def __init__(self, exported_program: ExportedProgram):
        super().__init__()
        self.exported_program: ExportedProgram = exported_program

    # Ensure each subclass has its own substitution registry.
    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)
        cls._DIALECT_SUBSTITUTIONS = {}

    @classmethod
    def register_dialect_substitution(
        cls, target: Target
    ) -> Callable[[SubstitutionFn], SubstitutionFn]:

        def decorator(func: SubstitutionFn) -> SubstitutionFn:
            if target in cls._DIALECT_SUBSTITUTIONS:
                raise RuntimeError(
                    f"Multiple substitutions registered for the same target in {cls.__name__} are not allowed."
                )
            else:
                cls._DIALECT_SUBSTITUTIONS[target] = func
            return func

        return decorator

    def call(self, graph_module: torch.fx.GraphModule) -> PassResult:
        modified = False

        for node in graph_module.graph.nodes:
            if node.op != "call_function":
                continue

            substitution_func = self._DIALECT_SUBSTITUTIONS.get(node.target, None)
            if substitution_func is None:
                continue

            dialect_node_spec = substitution_func(node, self.exported_program)
            if dialect_node_spec is None:
                continue

            modified = True
            with graph_module.graph.inserting_before(node):
                dialect_node = graph_module.graph.create_node(
                    "call_function",
                    target=dialect_node_spec.op,
                    args=dialect_node_spec.args,
                    kwargs=dialect_node_spec.kwargs or {},
                )

                node.replace_all_uses_with(dialect_node)

                # Keep same meta dict for new node and append new trace
                dialect_node.meta = node.meta
                old_stack_trace = dialect_node.meta.get("stack_trace", "")
                dialect_node.meta["stack_trace"] = (
                    f"{old_stack_trace}\n{traceback.format_stack()[-2]}"
                )

                graph_module.graph.erase_node(node)

        if modified:
            graph_module.graph.eliminate_dead_code()
            graph_module.recompile()
            graph_module = super().call(graph_module).graph_module

        return PassResult(graph_module, modified)
