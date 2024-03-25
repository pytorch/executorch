# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import copy
import logging
from typing import Optional, Set

import torch
from torch._export.utils import get_buffer, get_lifted_tensor_constant, get_param

from torch.export import ExportedProgram
from torch.export.exported_program import InputSpec, TensorArgument
from torch.export.graph_signature import InputKind


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


# TODO: add other passes to duplicate call_function nodes
def duplicate_constant_node(
    exported_program: ExportedProgram, candidate_node: str
) -> Set[str]:
    """
    A pass to duplicate the attributes/constants node (the candidate_node) in the graph. Mostly used for duplicating light-weight data.
    If the data is too large, try tag it with "no_copy" to prevent high memory usage and make it as part of the output.

    Args:
        exported_program: the exported program to be modified. If constants nodes are copied, they will be added as new
        placeholder and the state_dict will be updated
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
    for idx, node in enumerate(exported_program.graph.nodes):
        if node.op != "placeholder":
            continue
        old_input_spec = old_signature.input_specs[idx]
        old_input_spec_copy = copy.deepcopy(old_input_spec)
        if node == to_be_copied[0]:
            constant_or_attribute_node = node
            constant_or_attribute = _get_attribute_or_constants(exported_program, node)
            if constant_or_attribute is None:
                raise RuntimeError(
                    f"placeholder node for non-params, non-buffer, and non-tensor constants should not be tagged: {node} "
                )
            users = list(node.users.keys())
            for ith in range(1, len(node.users)):
                copy_constant_or_attribute_fqn = node.name + f"_copy_{ith - 1}"
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
                    new_args = tuple(
                        [
                            (
                                arg
                                if arg != constant_or_attribute_node
                                else copied_constant_or_attribute_node
                            )
                            for arg in users[ith].args
                        ]
                    )
                    new_kwargs = dict(
                        {
                            (
                                key,
                                (
                                    value
                                    if value != constant_or_attribute_node
                                    else copied_constant_or_attribute_node
                                ),
                            )
                            for key, value in users[ith].kwargs
                        }
                    )
                    users[ith].args = new_args
                    users[ith].kwargs = new_kwargs
                    if old_input_spec.kind == InputKind.CONSTANT_TENSOR:
                        exported_program.constants[copy_constant_or_attribute_fqn] = (
                            copy.deepcopy(constant_or_attribute)
                        )
                    elif (
                        old_input_spec.kind == InputKind.BUFFER
                        and old_input_spec.persistent is False
                    ):
                        # non persistent buffer will be in the .constants
                        exported_program.constants[copy_constant_or_attribute_fqn] = (
                            copy.deepcopy(constant_or_attribute)
                        )
                    else:
                        exported_program.state_dict[copy_constant_or_attribute_fqn] = (
                            copy.deepcopy(constant_or_attribute)
                        )
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
