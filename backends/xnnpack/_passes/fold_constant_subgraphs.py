# Copyright 2026 Arm Limited and/or its affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import operator

import torch

from executorch.exir.dialects._ops import ops as exir_ops
from executorch.exir.passes.constant_prop_pass import constant_prop_pass
from torch.export import ExportedProgram

_CONSTANT_SUBGRAPH_OPS = {
    torch.ops.aten.cat.default,
    torch.ops.aten.chunk.default,
    torch.ops.aten.split.Tensor,
    torch.ops.aten.split_with_sizes.default,
    operator.getitem,
    torch.ops.aten._weight_norm.default,
}


def _has_unsafe_tuple_consumer(exported_program: ExportedProgram) -> bool:
    """Avoid propagating tuple-valued chunk/split nodes, which
    constant_prop_pass cannot replace."""
    tuple_producer_targets = {
        torch.ops.aten.chunk.default,
        torch.ops.aten.split.Tensor,
        torch.ops.aten.split_with_sizes.default,
    }
    for node in exported_program.graph.nodes:
        if (
            node.op == "call_function"
            and node.target in tuple_producer_targets
            and any(
                user.op != "call_function" or user.target is not operator.getitem
                for user in node.users
            )
        ):
            return True
    return False


def _has_unsafe_mutation(node: torch.fx.Node, schema) -> bool:
    for index, argument in enumerate(schema.arguments):
        if argument.alias_info is None or not argument.alias_info.is_write:
            continue
        mutated = (
            node.args[index]
            if index < len(node.args)
            else node.kwargs.get(argument.name)
        )
        producer = mutated if isinstance(mutated, torch.fx.Node) else None
        producer_schema = (
            getattr(producer.target, "_schema", None)
            if producer is not None and producer.op == "call_function"
            else None
        )
        if (
            producer_schema is None
            or producer_schema.is_mutable
            or producer.target in _CONSTANT_SUBGRAPH_OPS
            or any(ret.alias_info is not None for ret in producer_schema.returns)
        ):
            return True
    return False


def _has_unsafe_node_effect(node: torch.fx.Node) -> bool:
    if isinstance(node.target, torch._ops.HigherOrderOperator):
        return True
    schema = getattr(node.target, "_schema", None)
    if schema is not None and schema.is_mutable:
        return _has_unsafe_mutation(node, schema)
    return node.is_impure() and node.target in _CONSTANT_SUBGRAPH_OPS


def _has_unsafe_graph_effects(exported_program: ExportedProgram) -> bool:
    """Return whether graph effects make constant folding unsafe."""
    return any(
        node.op == "call_function" and _has_unsafe_node_effect(node)
        for node in exported_program.graph.nodes
    )


class FoldConstantSubgraphsPass:
    """
    Fold selected constant subgraphs into direct constants before decomposition.

    This pass folds constant-only subgraphs such as:

    bias_0  bias_1  bias_2       # constant parameters
     \\    ||      //
         cat
          |
      split/chunk
          |
        getitem
          |
     addmm / linear

    Although the resulting bias slices are constant, the intermediate
    cat/split/getitem chain can prevent later lowering and delegation from
    recognizing them as constant inputs.

    Constant propagation evaluates eligible operations ahead of time and replaces
    each projection with a direct constant:

    constant bias_0 -> addmm / linear
    constant bias_1 -> addmm / linear
    constant bias_2 -> addmm / linear

    Folding is skipped when mutation, aliasing, impurity, or higher-order
    operations could make replacing a runtime value with persistent constant
    data unsafe.
    """

    def __call__(self, exported_program: ExportedProgram) -> ExportedProgram:
        # This runs before functionalization, so inspect schemas as well as the
        # graph signature for mutation.
        if any(
            "MUTATION" in str(spec.kind)
            for spec in exported_program.graph_signature.output_specs
        ):
            return exported_program
        if _has_unsafe_graph_effects(exported_program):
            return exported_program

        if not any(
            node.op == "call_function" and node.target in _CONSTANT_SUBGRAPH_OPS
            for node in exported_program.graph.nodes
        ):
            return exported_program

        if _has_unsafe_tuple_consumer(exported_program):
            return exported_program
        skipped = {
            node.target
            for node in exported_program.graph.nodes
            if node.op == "call_function" and node.target not in _CONSTANT_SUBGRAPH_OPS
        }
        folded_program = constant_prop_pass(
            exported_program, custom_skip_targets=skipped
        )
        # Constant propagation inserts an Edge clone for constant outputs, but
        # pre-decomposition graphs must remain entirely in the ATen dialect.
        replaced_clone = False
        for node in folded_program.graph.nodes:
            if node.target == exir_ops.edge.aten.clone.default:
                node.target = torch.ops.aten.clone.default
                replaced_clone = True
        if replaced_clone:
            folded_program.graph_module.recompile()
        return folded_program
