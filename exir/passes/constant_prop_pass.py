# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-unsafe

import logging
from collections import OrderedDict
from typing import cast, Mapping, Optional

import torch
from executorch.exir.dialects._ops import ops as exir_ops
from executorch.exir.dialects.edge._ops import EdgeOpOverload
from executorch.exir.operator.util import _QUANT_PRIMITIVES
from executorch.exir.passes.replace_aten_with_edge_pass import aten_to_edge
from torch._export.utils import (
    get_buffer,
    get_lifted_tensor_constant,
    get_param,
    is_lifted_tensor_constant,
    is_param,
)
from torch._guards import detect_fake_mode
from torch.export import ExportedProgram
from torch.export.exported_program import InputKind, InputSpec, TensorArgument
from torch.utils import _pytree as pytree

# Avoid propagating constants for `exir.ops.edge.aten.full.default`.
# Propagating aten.full can significantly increase compiled model size.
_DEFAULT_SKIP_TARGETS_NO_QUANT = {exir_ops.edge.aten.full.default}
_DEFAULT_SKIP_TARGETS = set(_DEFAULT_SKIP_TARGETS_NO_QUANT)

# Do not const prop quantization primitives
_QUANT_PRIMITIVES_EDGE = [aten_to_edge(op) for op in _QUANT_PRIMITIVES]
_DEFAULT_SKIP_TARGETS.update(set(_QUANT_PRIMITIVES_EDGE))


_PRIMITIVE_TYPES = (
    float,
    int,
    bool,
    str,
    torch.Tensor,
    torch.device,
    torch.dtype,
    torch.layout,
)


def get_default_skip_targets_no_quant() -> set[EdgeOpOverload]:
    return _DEFAULT_SKIP_TARGETS_NO_QUANT


def is_const(
    arg,
    exported_program: ExportedProgram,
    const_node_to_tensor: Mapping[torch.fx.Node, torch.Tensor],
) -> bool:
    if isinstance(arg, (tuple, list)):
        return all(is_const(x, exported_program, const_node_to_tensor) for x in arg)
    elif isinstance(arg, dict):
        return all(
            is_const(x, exported_program, const_node_to_tensor) for x in arg.values()
        )
    elif isinstance(arg, _PRIMITIVE_TYPES):
        return True
    elif arg is None:
        return True
    elif not isinstance(arg, torch.fx.Node):
        return False
    elif arg in const_node_to_tensor:
        return True
    return False


def get_data(
    arg,
    exported_program: ExportedProgram,
    const_node_to_tensor: Mapping[torch.fx.Node, torch.Tensor],
):
    if isinstance(arg, (tuple, list)):
        return type(arg)(
            get_data(x, exported_program, const_node_to_tensor) for x in arg
        )
    elif isinstance(arg, _PRIMITIVE_TYPES):
        return arg
    elif arg in const_node_to_tensor:
        return const_node_to_tensor[arg]
    return None


def is_constant_buffer(program: "ExportedProgram", node: torch.fx.Node) -> bool:
    """Checks if the given node is a constant buffer."""

    if node.target not in program.graph_signature.inputs_to_buffers:
        return False
    fqn = program.graph_signature.inputs_to_buffers[node.target]
    # if the buffer is mutated then record that
    return fqn not in program.graph_signature.buffers_to_mutate.values()


def get_constant_placeholder_dict(
    exported_program: ExportedProgram,
) -> OrderedDict[torch.fx.Node, torch.Tensor]:
    """
    Returns a dictionary of placeholder node -> constant tensor.
    """
    const_node_to_tensor: OrderedDict[torch.fx.Node, torch.Tensor] = OrderedDict()
    for node in exported_program.graph.find_nodes(op="placeholder"):
        if is_param(exported_program, node):
            const_node_to_tensor[node] = cast(
                torch.Tensor, get_param(exported_program, node)
            )
        elif is_constant_buffer(exported_program, node):
            const_node_to_tensor[node] = cast(
                torch.Tensor, get_buffer(exported_program, node)
            )
        elif is_lifted_tensor_constant(exported_program, node):
            const_node_to_tensor[node] = cast(
                torch.Tensor, get_lifted_tensor_constant(exported_program, node)
            )
    return const_node_to_tensor


def get_propagated_const_tensor_dict(
    exported_program: ExportedProgram,
    custom_skip_targets: Optional[set[EdgeOpOverload]],
) -> OrderedDict[torch.fx.Node, torch.Tensor]:
    """
    Propagates constants and returns a dictionary of node->constant tensors.
    """
    # Initialize dict with all constant placeholders.
    const_node_to_tensor = get_constant_placeholder_dict(exported_program)

    if custom_skip_targets is not None:
        all_skip_targets = custom_skip_targets
    else:
        # Default set of targets to skip.
        all_skip_targets = _DEFAULT_SKIP_TARGETS

    for node in exported_program.graph.nodes:
        if node.op != "call_function" or node.target in all_skip_targets:
            continue

        if not is_const(
            node.args,
            exported_program,
            const_node_to_tensor,
        ) or not is_const(
            node.kwargs,
            exported_program,
            const_node_to_tensor,
        ):
            continue

        args_data, kwargs_data = pytree.tree_map(
            lambda x: get_data(x, exported_program, const_node_to_tensor),
            (node.args, node.kwargs),
        )
        # Disable grad for constant propagation, otherwise the generated tensor can't be copied
        # because of the grad_fn.
        with torch.no_grad():
            # Execute the `node.target` and create a new propagated constant tensor.
            prop_constant_tensor = node.target(*args_data, **kwargs_data)
        const_node_to_tensor[node] = prop_constant_tensor

    return const_node_to_tensor


def get_first_user_input(exported_program: ExportedProgram) -> torch.fx.Node:
    """Returns the first user input node in the graph."""
    first_user_input = None
    for node in exported_program.graph.nodes:
        if (
            node.op == "placeholder"
            and node.name in exported_program.graph_signature.user_inputs
        ):
            first_user_input = node
            break
    return first_user_input


def replace_with_constant_node(
    node: torch.fx.Node,
    prop_constant_tensor: torch.Tensor,
    first_user_input: torch.fx.Node,
    fake_mode,
    exported_program: ExportedProgram,
) -> tuple[torch.fx.Node, str]:
    # Add `prop_constant_tensor` to program.state_dict.
    prefix = "_prop_tensor_constant"
    prop_constant_tensor_fqn = f"{prefix}{len(exported_program.constants)}"
    # If prop_constant_tensor_fqn already exists in the state dict, we need
    # to create a new name. Find the largest suffix of "_prop_tensor_constant",
    # and increment it by 1 to form the new name.
    if prop_constant_tensor_fqn in exported_program.constants:
        suffix = 1 + max(
            (
                int(name[len(prefix) :])
                for name in exported_program.constants.keys()
                if name.startswith(prefix) and name[len(prefix) :].isdigit()
            ),
            default=-1,
        )
        prop_constant_tensor_fqn = f"{prefix}{suffix}"

    exported_program.constants[prop_constant_tensor_fqn] = prop_constant_tensor

    # Insert a new placeholder node for the propagated constant tensor.
    with exported_program.graph.inserting_before(first_user_input):
        const_placeholder_node = exported_program.graph.placeholder(
            prop_constant_tensor_fqn
        )

    # Update the meta data of the new placeholder (buffer) node.
    for k, v in node.meta.items():
        const_placeholder_node.meta[k] = v
    const_placeholder_node.meta["val"] = fake_mode.from_tensor(
        prop_constant_tensor, static_shapes=True
    )
    const_placeholder_node.meta["val"].constant = prop_constant_tensor

    # Replace the original node with the new constant node.
    node.replace_all_uses_with(const_placeholder_node)
    exported_program.graph.erase_node(node)

    return const_placeholder_node, prop_constant_tensor_fqn


def get_fake_mode(exported_program: ExportedProgram):
    fake_mode = detect_fake_mode(
        tuple(
            node.meta["val"]
            for node in exported_program.graph.nodes
            if node.op == "placeholder"
        )
    )
    assert fake_mode is not None
    return fake_mode


def erase_constant_node(
    exported_program: ExportedProgram,
    node: torch.fx.Node,
) -> None:
    # Remove corresponding tensor from param/constants dict.
    signature = exported_program.graph_signature
    if name := signature.inputs_to_parameters.get(node.name, None):
        exported_program.state_dict.pop(name, None)
    elif name := signature.inputs_to_lifted_tensor_constants.get(node.name, None):
        exported_program.constants.pop(name, None)
    elif name := signature.inputs_to_buffers.get(node.name, None):
        exported_program.constants.pop(name, None)
        exported_program.state_dict.pop(name, None)

    # Remove from graph.
    exported_program.graph.erase_node(node)


def create_constant_nodes_and_return_specs(
    const_node_to_tensor: Mapping[torch.fx.Node, torch.Tensor],
    exported_program: ExportedProgram,
) -> dict[str, InputSpec]:
    """
    Creates constant nodes for all entries in `const_node_to_tensor` and returns a node.name -> InputSpec dict.
    """
    name_to_spec_dict: dict[str, InputSpec] = {}

    fake_mode = get_fake_mode(exported_program)
    first_user_input = get_first_user_input(exported_program)

    # Iterate over nodes in reverse order.
    for node, prop_constant_tensor in reversed(const_node_to_tensor.items()):
        if all(x in const_node_to_tensor for x in node.users):
            # All users of this constant node are also constant, so we don't need to create a new constant node.
            erase_constant_node(exported_program, node)
            continue

        if node.op == "placeholder":
            continue

        const_placeholder_node, prop_constant_tensor_fqn = replace_with_constant_node(
            node, prop_constant_tensor, first_user_input, fake_mode, exported_program
        )

        # Create input spec for lifted constant.
        name_to_spec_dict[const_placeholder_node.name] = InputSpec(
            kind=InputKind.CONSTANT_TENSOR,
            arg=TensorArgument(name=const_placeholder_node.name),
            target=prop_constant_tensor_fqn,
            persistent=True,
        )
    return name_to_spec_dict


def _update_output_node_and_specs(exported_program: ExportedProgram) -> None:
    """
    Update the output node and output specs in the exported program.
    In case a constant node is used as output, we replace it with a clone of the constant node.
    """
    # Dict [node.name -> InputSpec]
    updated_constant_placeholders = get_constant_placeholder_dict(exported_program)
    output = exported_program.graph.find_nodes(op="output")[0]
    output_nodes = cast(list[torch.fx.Node], list(output.args[0]))
    output_specs = exported_program.graph_signature.output_specs
    assert len(output_nodes) == len(output_specs)

    for i in range(len(output_specs)):
        out_node = output_nodes[i]
        if out_node not in updated_constant_placeholders:
            continue

        with exported_program.graph.inserting_after(out_node):
            new_node = exported_program.graph.call_function(
                exir_ops.edge.aten.clone.default, (out_node,)
            )
        assert "val" in out_node.meta
        new_node.meta["val"] = out_node.meta["val"]
        output_nodes[i] = new_node

        # Update the constant-propagated output node.
        output_specs[i].arg = TensorArgument(name=output_nodes[i].name)

    output.args = (output_nodes,)


def constant_prop_pass(
    exported_program: ExportedProgram,
    custom_skip_targets: Optional[set[EdgeOpOverload]] = None,
) -> ExportedProgram:
    """
    This pass is for constant propagation for Exported Program with lifted parameters,
    as the parameters will not be shown up as `get_attr` but as `placeholder` to the graph.

    Args:
        exported_program: The ExportedProgram to perform constant propagation on.
        custom_skip_targets: Optional set of EdgeOpOverload targets to skip during constant propagation.

    Returns:
        The modified ExportedProgram with constant propagation applied.
    """
    if (
        len([node for node in exported_program.graph.nodes if node.op == "placeholder"])
        == 0
    ):
        return exported_program

    has_control_flow = [
        node
        for node in exported_program.graph.nodes
        if node.target == torch.ops.higher_order.cond
    ]
    if len(has_control_flow) > 0:
        logging.warning(
            "constant_prop_pass does not constant propagate in control flow modules"
        )

    const_node_to_tensor = get_propagated_const_tensor_dict(
        exported_program, custom_skip_targets
    )

    # Get old input specs.
    name_to_spec_dict = {
        s.arg.name: s for s in exported_program.graph_signature.input_specs
    }
    # Add the new constants to input specs dict.
    name_to_spec_dict.update(
        create_constant_nodes_and_return_specs(const_node_to_tensor, exported_program)
    )

    # Generate new input spec.
    new_input_specs = []
    for node in exported_program.graph.find_nodes(op="placeholder"):
        new_input_specs.append(name_to_spec_dict[node.name])
    exported_program.graph_signature.input_specs = new_input_specs

    _update_output_node_and_specs(exported_program)

    # Cleanup the graph.
    exported_program.graph.eliminate_dead_code()
    exported_program.graph_module.recompile()

    return exported_program
