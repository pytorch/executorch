# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from typing import Dict, List, Optional, Tuple

import torch
from executorch.exir.operator.convert import is_inplace_variant

from torch.export.exported_program import (
    ExportedProgram,
    ExportGraphSignature,
    InputKind,
    OutputKind,
    OutputSpec,
)
from torch.export.graph_signature import TensorArgument
from torch.utils import _pytree as pytree
from torchgen.model import SchemaKind


def _insert_copy(
    gm: torch.fx.GraphModule,
    mutated_outputs: List[Optional[str]],
    input_name_to_node: Dict[str, torch.fx.Node],
):
    """
    Find the all the buffers and inputs that were mutated and insert copy_
    operators to reflect mutations.
    """
    output_node = gm.graph.output_node()
    assert output_node is not None
    outputs = pytree.tree_flatten(output_node.args)[0]
    assert len(outputs) == len(mutated_outputs)

    user_output_nodes = []
    buffer_output_nodes = []
    for return_node, mutated_node_name in zip(outputs, mutated_outputs):
        # User output, leave alone
        if mutated_node_name is None:
            user_output_nodes.append(return_node)
            continue

        # Mutable buffer grab the node
        if mutated_node_name in input_name_to_node:
            mutated_node = input_name_to_node[mutated_node_name]
        else:
            raise RuntimeError(
                f"Could not find {mutated_node_name} in either buffer or input nodes"
            )

        # insert copy
        with gm.graph.inserting_before(output_node):
            buffer_output = gm.graph.call_function(
                torch.ops.aten.copy_.default, (mutated_node, return_node)
            )
            # add output of copy to graph outputs
            buffer_output_nodes.append(buffer_output)

    with gm.graph.inserting_before(output_node):
        buffer_output_nodes.extend(user_output_nodes)
        # Remove old outputs
        new_output = gm.graph.output(tuple(buffer_output_nodes))
        output_node.replace_all_uses_with(new_output)
        gm.graph.erase_node(output_node)
    return buffer_output_nodes


def _is_inplace_node(node: torch.fx.Node) -> bool:
    """Check if a node is an inplace node."""
    return (
        node.op == "call_function"
        and hasattr(node.target, "_schema")
        and is_inplace_variant(
            node.target._schema.name, node.target._schema.overload_name  # pyre-ignore
        )
    )


def _inplace_lineage(
    output_arg: torch.fx.Node,
    gs: ExportGraphSignature,
    kind: SchemaKind,
) -> bool:
    """
    Walk the graph backwards to see if output_arg is ultimately the same as an input.
    """
    if kind != OutputKind.BUFFER_MUTATION and kind != OutputKind.USER_INPUT_MUTATION:
        return False

    while output_arg.op != "placeholder":
        if _is_inplace_node(output_arg):
            # From looking at native_functions.yaml, inplace ops always have self as the first arg
            output_arg = output_arg.args[0]  # pyre-ignore
        else:
            return False

    # If the output arg was a buffer then it needs to reach a buffer placeholder
    if kind == OutputKind.BUFFER_MUTATION:
        return output_arg.target in gs.inputs_to_buffers
    # If the output arg was a user input then it needs to reach a user input placeholder
    assert kind == OutputKind.USER_INPUT_MUTATION
    return output_arg.target in gs.user_inputs


def insert_write_back_for_buffers_pass(
    ep: ExportedProgram,
) -> Tuple[torch.fx.GraphModule, ExportGraphSignature]:
    gm: torch.fx.GraphModule = ep.graph_module
    lifted_inputs: List[Optional[str]] = []
    for in_spec in ep.graph_signature.input_specs:
        if in_spec.kind in (
            InputKind.BUFFER,
            InputKind.CONSTANT_TENSOR,
            InputKind.PARAMETER,
            InputKind.CUSTOM_OBJ,
        ):
            lifted_inputs.append(in_spec.target)
        elif in_spec.kind is InputKind.USER_INPUT and isinstance(
            in_spec.arg, TensorArgument
        ):
            lifted_inputs.append(in_spec.arg.name)
        else:
            lifted_inputs.append(None)

    input_name_to_node: Dict[str, torch.fx.Node] = {}

    placeholder_nodes = [node for node in gm.graph.nodes if node.op == "placeholder"]
    assert len(lifted_inputs) == len(placeholder_nodes)
    # Grab the all the non user inputs
    for input_node, lifted_node in zip(placeholder_nodes, lifted_inputs):
        if lifted_node is not None:
            input_name_to_node[lifted_node] = input_node

    output_node = gm.graph.output_node()

    # Grab the mutable buffer nodes in the outputs,
    mutated_outputs: List[Optional[str]] = []
    for i, out_spec in enumerate(ep.graph_signature.output_specs):
        # if the output arg is the input value then all operations on it are in-place
        # so there's no need to add a copy_ node
        if (
            out_spec.kind
            in (OutputKind.BUFFER_MUTATION, OutputKind.USER_INPUT_MUTATION)
            and
            # explicitly check if target exists (it should always be there)
            out_spec.target in input_name_to_node
            and
            # if the arg and target are not the same, we add a copy_ node.
            not _inplace_lineage(
                output_node.args[0][i],
                ep.graph_signature,
                ep.graph_signature.output_specs[i].kind,
            )
        ):
            mutated_outputs.append(out_spec.target)
        else:
            mutated_outputs.append(None)

    # insert the copy ops and update the outputs
    buffer_output_nodes = _insert_copy(gm, mutated_outputs, input_name_to_node)
    gm.graph.lint()
    gm.graph.eliminate_dead_code()
    gm.recompile()

    # patch the output signature to point to the new updated outputs
    new_output_specs: List[OutputSpec] = []
    i = 0
    for output_spec in ep.graph_signature.output_specs:
        if output_spec.kind in (
            OutputKind.BUFFER_MUTATION,
            OutputKind.USER_INPUT_MUTATION,
        ):
            output_spec.arg.name = buffer_output_nodes[i].name
            i += 1
        new_output_specs.append(output_spec)

    signature = ExportGraphSignature(
        input_specs=ep.graph_signature.input_specs,
        output_specs=new_output_specs,
    )

    return gm, signature
