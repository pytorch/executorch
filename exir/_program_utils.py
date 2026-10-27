# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-unsafe

from collections.abc import Callable

import torch
from executorch.exir.dialects._ops import ops as exir_ops
from torch.export.exported_program import (
    ConstantArgument,
    ExportGraphSignature,
    InputSpec,
    OutputSpec,
)
from torch.export.graph_signature import OutputKind
from torch.fx.traceback import NodeSource, NodeSourceAction

_MUTATION_OUTPUT_CLONE_OP = "_mutation_output_clone_op"


def _get_mutation_output_clone_op(
    program: torch.export.ExportedProgram,
    clone_op: Callable[..., torch.Tensor] | None = None,
) -> Callable[..., torch.Tensor] | None:
    if clone_op is not None:
        return clone_op
    if program.dialect == "ATEN":
        return torch.ops.aten.clone.default
    kind = program.graph_module.meta.get(_MUTATION_OUTPUT_CLONE_OP)
    if kind == "aten":
        return torch.ops.aten.clone.default
    if kind == "edge":
        return exir_ops.edge.aten.clone.default
    if kind == "edge_dim_order":
        return exir_ops.edge.dim_order_ops._clone_dim_order.default
    return None


def _set_mutation_output_clone_op(
    gm: torch.fx.GraphModule, clone_op: Callable[..., torch.Tensor] | None
) -> None:
    """Record the clone dialect and layout as metadata preserved by deepcopy."""
    if clone_op is None:
        return
    if clone_op is torch.ops.aten.clone.default:
        kind = "aten"
    elif clone_op is exir_ops.edge.aten.clone.default:
        kind = "edge"
    elif clone_op is exir_ops.edge.dim_order_ops._clone_dim_order.default:
        kind = "edge_dim_order"
    else:
        raise ValueError(f"Unsupported mutation output clone operator: {clone_op}")
    gm.meta[_MUTATION_OUTPUT_CLONE_OP] = kind


def _disambiguate_mutation_outputs(
    signature: ExportGraphSignature,
    gm: torch.fx.GraphModule,
    *,
    clone_op: Callable[..., torch.Tensor] | None,
) -> bool:
    """Copy colliding mutation outputs in-place, preserving positional user outputs."""
    output_node = gm.graph.output_node()
    output_args = output_node.args[0]
    assert isinstance(output_args, (tuple, list))
    assert len(output_args) == len(
        signature.output_specs
    ), "Number of outputs changed after transformation"
    seen_mutations: set[tuple[OutputKind, torch.fx.Node]] = set()
    new_output_args = list(output_args)
    for i, (node, spec) in enumerate(zip(output_args, signature.output_specs)):
        if spec.kind not in (
            OutputKind.BUFFER_MUTATION,
            OutputKind.USER_INPUT_MUTATION,
        ):
            continue
        assert isinstance(node, torch.fx.Node)
        mutation = (spec.kind, node)
        if mutation in seen_mutations:
            if clone_op is None:
                raise RuntimeError(
                    "Mutation output repair requires an explicit clone operator "
                    "for programs without an Edge compile configuration"
                )
            # Mutation dictionaries are keyed by node name, while user outputs
            # are positional and may share a node. Copy only colliding writes.
            with gm.graph.inserting_before(output_node):
                clone = gm.graph.call_function(clone_op, args=(node,))
            clone.meta = {
                key: node.meta[key]
                for key in ("val", "stack_trace", "nn_module_stack", "debug_handle")
                if key in node.meta
            }
            clone.meta["from_node"] = [
                NodeSource(node, "_disambiguate_mutation_outputs", NodeSourceAction.CREATE)
            ]
            new_output_args[i] = clone
        seen_mutations.add(mutation)
    if new_output_args == list(output_args):
        return False
    output_node.args = (tuple(new_output_args),) + output_node.args[1:]
    gm.recompile()
    return True


def _get_updated_range_constraints(gm, old_range_constraints=None):
    def get_shape_env(gm):
        vals = [
            node.meta["val"]
            for node in gm.graph.nodes
            if node.meta.get("val", None) is not None
        ]
        from torch._guards import detect_fake_mode  # type: ignore[21]

        fake_mode = detect_fake_mode(vals)
        if fake_mode is not None:
            return fake_mode.shape_env
        for v in vals:
            if isinstance(v, torch.SymInt):
                return v.node.shape_env

    shape_env = get_shape_env(gm)
    if shape_env is None:
        return {}
    range_constraints = {
        shape_env.replacements.get(k, k): v for k, v in shape_env.var_to_range.items()
    }
    # Only when we have an unbacked symint, and it's used as constructor inputs,
    # runtime_var_to_range will make a difference compated to var_to_range.
    # e.g. [2, oo) -> [0, oo)
    for k, v in shape_env.var_to_range.items():
        if k not in shape_env.replacements:
            range_constraints[k] = v
    # As torch.export does, keep the declared range of a surviving symbol: the
    # ShapeEnv raises a backed symbol's declared min of 0 or 1 to 2.
    for k, v in (old_range_constraints or {}).items():
        if k in range_constraints:
            range_constraints[k] = v
    return range_constraints


def _get_updated_graph_signature(
    old_signature: ExportGraphSignature,
    new_gm: torch.fx.GraphModule,
) -> ExportGraphSignature:
    """Rebuild the positional signature without modifying the graph module."""
    new_input_specs = []
    i = 0
    for node in new_gm.graph.nodes:
        if node.op != "placeholder":
            continue

        assert i < len(
            old_signature.input_specs
        ), "Number of inputs changed after transformation"
        old_input_spec = old_signature.input_specs[i]
        arg = (
            old_input_spec.arg
            if isinstance(old_input_spec.arg, ConstantArgument)
            # pyre-fixme[20]: Argument `class_fqn` expected.
            else type(old_input_spec.arg)(node.name)
        )
        new_input_specs.append(
            InputSpec(
                old_input_spec.kind,
                arg,
                old_input_spec.target,
                persistent=old_input_spec.persistent,
            )
        )
        i += 1

    output_node = new_gm.graph.output_node()
    assert output_node.op == "output"

    output_args = output_node.args[0]
    assert isinstance(output_args, (tuple, list))
    assert len(output_args) == len(
        old_signature.output_specs
    ), "Number of outputs changed after transformation"
    new_output_specs = []
    for i, node in enumerate(output_node.args[0]):
        assert i < len(
            old_signature.output_specs
        ), "Number of outputs changed after transformation"
        old_output_spec = old_signature.output_specs[i]
        arg = (
            old_output_spec.arg
            if isinstance(old_output_spec.arg, ConstantArgument)
            # pyre-fixme[20]: Argument `class_fqn` expected.
            else type(old_output_spec.arg)(node.name)
        )
        new_output_specs.append(
            OutputSpec(old_output_spec.kind, arg, old_output_spec.target)
        )

    new_signature = ExportGraphSignature(
        input_specs=new_input_specs, output_specs=new_output_specs
    )
    return new_signature
