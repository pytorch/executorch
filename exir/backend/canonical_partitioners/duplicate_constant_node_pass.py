# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import copy
import logging
from typing import Dict, List, Optional, Set

import torch
from torch._export.utils import get_buffer, get_lifted_tensor_constant, get_param

from torch.export import ExportedProgram
from torch.export.exported_program import InputSpec, TensorArgument


def _get_attribute_or_constants(
    exported_program: ExportedProgram, node: torch.fx.Node
) -> Optional[torch.Tensor]:
    # get either attribute node or constant constant
    maybe_param = get_param(exported_program, node)
    maybe_buffer = get_buffer(exported_program, node)
    maybe_lifted_tensor = get_lifted_tensor_constant(exported_program, node)

    constant_or_attribute = None
    if maybe_param is not None:
        constant_or_attribute = maybe_param
    elif maybe_buffer is not None:
        constant_or_attribute = maybe_buffer
    elif maybe_lifted_tensor is not None:
        constant_or_attribute = maybe_lifted_tensor
    return constant_or_attribute


def _users_reading_a_copy(node: torch.fx.Node) -> List[List[torch.fx.Node]]:
    """
    The users of node that read a copy of it, in groups that share one. The
    users with the same delegation tag, or with none, form a group, so each
    program takes a constant once: for example, a program that lists the
    same lifted constant twice fails to unlift. The group with the graph
    output keeps the original, which the output specs name, and otherwise
    the first group does.
    """
    groups: Dict[Optional[str], List[torch.fx.Node]] = {}
    for user in node.users:
        groups.setdefault(user.meta.get("delegation_tag"), []).append(user)
    output = any(user.op == "output" for user in node.users)
    keeper = None if output else next(iter(groups), None)
    return [users for tag, users in groups.items() if tag != keeper]


# TODO: add other passes to duplicate call_function nodes
def duplicate_constant_node(
    exported_program: ExportedProgram, candidate_node: str
) -> Set[str]:
    """
    A pass to duplicate the attributes/constants node (the candidate_node) in the graph. Mostly used for duplicating light-weight data.
    If the data is too large, try tag it with "no_copy" to prevent high memory usage and make it as part of the output.
    A mutated buffer is never copied, because a copy would not see its updates: this raises instead.
    Users with the same delegation tag, or with none, share one placeholder, so tag the
    users before calling it: on a graph without tags it makes no copies.

    Args:
        exported_program: the exported program to be modified. If constants nodes are copied, they will be added as new
        placeholders that share the original target
        candidate_node: the name of the constant node to be duplicated

    Returns:
        The set of the names of the new constant nodes
    """
    to_be_copied = [
        node
        for node in exported_program.graph.nodes
        if node.name == candidate_node and node.op == "placeholder"
    ]
    if len(to_be_copied) == 0:
        logging.info("no constant node to be copied")
        return set()
    new_input_specs = []
    old_signature = exported_program.graph_signature
    copied_nodes = set()

    placeholder_idx = -1
    for node in exported_program.graph.nodes:
        if node.op != "placeholder":
            continue

        placeholder_idx += 1
        old_input_spec = old_signature.input_specs[placeholder_idx]
        old_input_spec_copy = copy.deepcopy(old_input_spec)
        if node == to_be_copied[0]:
            constant_or_attribute_node = node
            constant_or_attribute = _get_attribute_or_constants(exported_program, node)
            if constant_or_attribute is None:
                raise RuntimeError(
                    f"placeholder node for non-params, non-buffer, and non-tensor constants should not be tagged: {node} "
                )
            if (
                old_signature.inputs_to_buffers.get(node.name)
                in old_signature.buffers_to_mutate.values()
            ):
                raise RuntimeError(
                    f"mutated buffer ({node}) cannot be copied, because a copy "
                    "would not see its updates. Tag it only when every node "
                    "that reads or writes it has the same tag, as "
                    "tag_mutated_buffer does, or leave it untagged."
                )
            for ith, group in enumerate(_users_reading_a_copy(node)):
                copy_constant_or_attribute_fqn = node.name + f"_copy_{ith}"
                with exported_program.graph.inserting_before(
                    constant_or_attribute_node
                ):
                    copied_constant_or_attribute_node = (
                        exported_program.graph.placeholder(
                            copy_constant_or_attribute_fqn
                        )
                    )
                    copied_nodes.add(copy_constant_or_attribute_fqn)
                    logging.info(
                        f"Copying constant nodes {node.name} and creating {copy_constant_or_attribute_fqn}"
                    )
                    for k, v in node.meta.items():
                        copied_constant_or_attribute_node.meta[k] = v
                    copied_constant_or_attribute_node.meta["val"] = (
                        constant_or_attribute_node.meta["val"]
                    )
                    # This also reaches a constant that an op such as cat or
                    # index_put reads from a list.
                    for user in group:
                        user.replace_input_with(
                            constant_or_attribute_node,
                            copied_constant_or_attribute_node,
                        )
                    # The copy keeps the original target, so it reads the
                    # original tensor and needs no data of its own.
                    new_input_specs.append(
                        InputSpec(
                            kind=old_input_spec.kind,
                            arg=TensorArgument(name=copy_constant_or_attribute_fqn),
                            target=old_input_spec.target,
                            persistent=old_input_spec.persistent,
                        )
                    )
        # Ensure we add the original input spec to the last one, because all the copied nodes
        # are inserted before the candidate node.
        new_input_specs.append(old_input_spec_copy)

    exported_program.graph_signature.input_specs = new_input_specs
    exported_program.graph_module.recompile()
    exported_program._validate()
    return copied_nodes
