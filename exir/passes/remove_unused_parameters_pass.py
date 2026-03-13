# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import copy

import torch
from executorch.exir.graph_module import get_control_flow_submodules
from executorch.exir.passes.propagate_input_spec import (
    INPUT_SPEC_KEY,
    propagate_input_spec,
)
from torch.export.exported_program import (
    ExportedProgram,
    InputKind,
    InputSpec,
    OutputKind,
)
from torch.fx import GraphModule, Node


def remove_unused_parameters_pass(
    ep: ExportedProgram,
) -> ExportedProgram:
    """
    Remove unused parameters from the exported program.
    """

    propagate_input_spec(ep)

    # This pass operates in two phases:
    #  * First, find all top-level graph inputs that are used.
    #  * Second, go back through and strip out unused params.

    # Find unused params. Don't ever remove top-level user inputs, just params/buffers/constants.
    removable_kinds = {InputKind.PARAMETER, InputKind.BUFFER, InputKind.CONSTANT_TENSOR}

    # Key by arg.name (placeholder name) which is unique, unlike target which
    # may be shared across duplicated constant nodes.
    all_params: dict[str, InputSpec] = {
        spec.arg.name: spec
        for spec in ep.graph_signature.input_specs
        if spec.kind in removable_kinds
    }
    used_param_names = _find_used_param_names_recursive(ep.graph_module)

    mutated_buffer_targets = {
        spec.target
        for spec in ep.graph_signature.output_specs
        if spec.kind == OutputKind.BUFFER_MUTATION
    }
    for name, spec in all_params.items():
        if spec.target in mutated_buffer_targets:
            used_param_names.add(name)

    unused_params: dict[str, InputSpec] = {
        name: spec for name, spec in all_params.items() if name not in used_param_names
    }

    # Remove unused params from the graph, including recursive HOP submodules.
    # Note that target names may not be unique when constants are duplciated, so
    # use arg.name.
    unused_names = set(unused_params.keys())
    _remove_params_recursive(ep.graph_module, unused_names)

    # Update the EP signature.
    new_signature = copy.deepcopy(ep.graph_signature)
    for spec in unused_params.values():
        new_signature.input_specs.remove(spec)

    # Collect targets still in use after removal.
    remaining_targets = {
        spec.target for spec in new_signature.input_specs if spec.target is not None
    }

    # Delete state_dict/constants entries. Duplicated constants may use arg.name
    # as the state_dict key (from duplicate_constant_node), while originals use
    # the spec target. Clean up both.
    for name, spec in unused_params.items():
        if spec.target and spec.target not in remaining_targets:
            if spec.target in ep._state_dict:
                del ep._state_dict[spec.target]
            elif spec.target in ep.constants:
                del ep.constants[spec.target]
        # Also clean up entries keyed by arg.name (for duplicated constants)
        if name != spec.target:
            if name in ep._state_dict:
                del ep._state_dict[name]
            if name in ep.constants:
                del ep.constants[name]

    ep._graph_signature = new_signature
    ep.graph_module.recompile()

    return ep


def _find_used_param_names_recursive(
    module: GraphModule,
) -> set[str]:
    """Find arg.names of params that are used anywhere in the module or its submodules."""
    used_names: set[str] = set()

    for node in module.graph.nodes:
        if node.op == "placeholder":
            input_spec = node.meta.get(INPUT_SPEC_KEY, None)

            if input_spec is not None and _placeholder_node_has_direct_usages(node):
                used_names.add(input_spec.arg.name)

    for _, submodule, _ in get_control_flow_submodules(module):
        used_names.update(_find_used_param_names_recursive(submodule))

    return used_names


def _placeholder_node_has_direct_usages(
    node: Node,
) -> bool:
    """
    Returns true if the given placeholder node is directly used in the enclosing submodule.
    Usages where the node is passed as an operand to a submodule aren't counted.
    """

    # Check each user. If it's not control flow, or it's directly used by a control flow
    # op - return true. Direct usages include cond conditions, scan xs, etc. where they
    # are used in the HOP, not just passed through to the submodule.
    for user in node.users:
        if user.target == torch.ops.higher_order.cond:
            # cond(pred, true_fn, false_fn, operands)
            if user.args[0] == node:
                return True
        elif user.target == torch.ops.higher_order.map_impl:
            # map(f, mapped_args, operands)
            if node in user.args[1]:
                return True
        elif user.target == torch.ops.higher_order.scan:
            # scan(combine_fn, init, xs, additional_args)
            if user.args[1] == node or node in user.args[2]:
                return True
        elif user.target == torch.ops.higher_order.while_loop:
            # cond, body, carried_inputs, additional_inputs
            continue  # All while loop args are passed unmodified to submodules.
        else:
            # Not control flow, so it's a direct use.
            return True

    return False


def _remove_params_recursive(
    module: GraphModule,
    names_to_remove: set[str],
) -> None:
    # Find placeholder nodes corresponding to the params to remove.
    placeholders_to_remove = [
        node
        for node in module.graph.nodes
        if node.op == "placeholder"
        and node.meta.get(INPUT_SPEC_KEY) is not None
        and node.meta.get(INPUT_SPEC_KEY).arg.name in names_to_remove
    ]

    placeholders_to_remove_set = set(placeholders_to_remove)

    # Recurse into submodules first, then filter operands at this level.
    for _, submodule, node in get_control_flow_submodules(module):
        _remove_params_recursive(submodule, targets_to_remove)

    # Filter out removed placeholders from HOP operands
    for _, _submodule, node in get_control_flow_submodules(module):
        if node.target == torch.ops.higher_order.cond:
            # cond(pred, true_fn, false_fn, operands)
            new_operands = list(
                x for x in node.args[3] if x not in placeholders_to_remove_set
            )
            node.args = (node.args[0], node.args[1], node.args[2], new_operands)
        elif node.target == torch.ops.higher_order.map_impl:
            # map(f, mapped_args, operands)
            new_operands = list(
                x for x in node.args[2] if x not in placeholders_to_remove_set
            )
            node.args = (node.args[0], node.args[1], new_operands)
        elif node.target == torch.ops.higher_order.scan:
            # scan(combine_fn, init, xs, additional_args)
            new_operands = list(
                x for x in node.args[3] if x not in placeholders_to_remove_set
            )
            node.args = (node.args[0], node.args[1], node.args[2], new_operands)
        elif node.target == torch.ops.higher_order.while_loop:
            # while_loop(cond, body, carried_inputs, additional_inputs)
            new_operands = list(
                x for x in node.args[3] if x not in placeholders_to_remove_set
            )
            node.args = (node.args[0], node.args[1], node.args[2], new_operands)

    # Remove the placeholder nodes.
    for node in placeholders_to_remove:
        module.graph.erase_node(node)

    module.recompile()
