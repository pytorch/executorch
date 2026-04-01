# Copyright 2025 Arm Limited and/or its affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

from typing import cast, Sequence, Set, Type

import torch

from executorch.backends.arm._passes.arm_pass import ArmPass
from executorch.exir.dialects._ops import ops as exir_ops
from executorch.exir.pass_base import ExportPass
from torch.fx import GraphModule, Node
from torch.fx.passes.infra.pass_base import PassResult


class NormalizeWhileInitialArgsPass(ArmPass):
    """
    Normalize ``torch.ops.higher_order.while_loop`` by moving additional_args to carried_args,
    making the number of outputs equal to the number of inputs which is required by the TOSA specification.
    Example:
            def cond(val):
                return val.sum() < 10

            def body(val):
                return (val * 2,)
            while_loop(cond, body, (val,), additional_args= (buffer,))
       becomes:
            def cond(val, buffer):
                return val.sum() < 10

            def body(val, buffer):
                return (val * 2, buffer.clone())
            while_loop(cond, body, (val, buffer), ())

    The clone is neccessary to avoid issues with aliasing.
    """

    def __init__(self, use_exir_clone: bool, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        if use_exir_clone:
            self.clone_op = exir_ops.edge.aten.alias_copy.default
        else:
            self.clone_op = torch.ops.aten.clone.default

    _passes_required_after: Set[Type[ExportPass]] = set()

    def _connect_to_output(
        self, body_module: GraphModule, placeholders: Sequence[Node]
    ) -> list[Node]:
        if not placeholders:
            return []

        cloned_placeholders = []
        with body_module.graph.inserting_after(placeholders[-1]):
            for placeholder in placeholders:
                clone = body_module.graph.create_node(
                    "call_function",
                    self.clone_op,
                    (placeholder,),
                )
                cloned_placeholders.append(clone)
                clone.meta = placeholder.meta
        output_node = body_module.graph.output_node()
        output_values = output_node.args[0]
        if not isinstance(output_values, tuple):
            raise RuntimeError("Output of a while should be a tuple.")

        output_node.update_arg(0, output_values + tuple(cloned_placeholders))
        body_module.recompile()
        return list(cloned_placeholders)

    def _normalize_node(self, graph_module: GraphModule, node: Node) -> bool:
        additional_inputs = list(cast(Sequence[Node], node.args[3]))

        if not additional_inputs:
            return False

        carried_inputs = list(cast(Sequence[Node], node.args[2]))
        new_carried = tuple(carried_inputs + additional_inputs)
        node.update_arg(2, new_carried)
        node.update_arg(3, ())

        body_module_name = str(cast(Node, node.args[1]).target)
        body_module = cast(GraphModule, graph_module.get_submodule(body_module_name))  # type: ignore
        placeholders = [n for n in body_module.graph.nodes if n.op == "placeholder"]
        num_inputs = len(placeholders)
        old_num_inputs = len(carried_inputs)
        if num_inputs != len(new_carried):
            raise RuntimeError(
                f"Length of loop placeholders {placeholders} is not equal length of carried inputs {new_carried}"
            )

        missing_placeholders = placeholders[old_num_inputs:]
        self._connect_to_output(body_module, missing_placeholders)

        return True

    def call(self, graph_module: GraphModule) -> PassResult:
        modified = False
        for node in graph_module.graph.nodes:
            if node.op != "call_function":
                continue
            if node.target != torch.ops.higher_order.while_loop:
                continue
            modified |= self._normalize_node(graph_module, node)

        if modified:
            graph_module.recompile()

        return PassResult(graph_module, modified)
