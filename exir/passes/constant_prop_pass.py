# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-unsafe

import logging
import operator
from collections import OrderedDict
from typing import cast, Mapping, Optional

import torch
from executorch.exir import memory
from executorch.exir.dialects._ops import ops as exir_ops
from executorch.exir.dialects.edge._ops import EdgeOpOverload
from executorch.exir.operator.util import _QUANT_PRIMITIVES
from executorch.exir.passes.insert_write_back_for_buffers_pass import (
    _schema_paired_args,
)
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
from torch.multiprocessing.reductions import StorageWeakRef
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
    torch.memory_format,
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
    # Before run_decompositions, buffers_to_mutate does not list a buffer that
    # is written in place, so it looks like a constant here.
    written = _nodes_written_in_place(exported_program.graph_module)
    for node in exported_program.graph.find_nodes(op="placeholder"):
        if node in written:
            continue
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


# Higher order ops whose submodule takes, in order, the operands after it.
_WRAPPER_HOPS = (
    torch.ops.higher_order.wrap_with_set_grad_enabled,
    torch.ops.higher_order.wrap_with_autocast,
)


def _storages(node: torch.fx.Node) -> set[StorageWeakRef]:
    # Only strided tensors have a storage to share. Sparse ones raise here.
    return {
        StorageWeakRef(val.untyped_storage())
        for val in pytree.tree_leaves(node.meta.get("val"))
        if isinstance(val, torch.Tensor) and val.layout == torch.strided
    }


def _hop_written_operands(
    graph_module: torch.fx.GraphModule, node: torch.fx.Node
) -> list[torch.fx.Node]:
    """
    Returns the operands of a higher order op, such as a no_grad or autocast
    region, that an in-place op inside its submodules writes.
    """
    written = []
    for i, arg in enumerate(node.args):
        if not isinstance(arg, torch.fx.Node) or arg.op != "get_attr":
            continue
        submodule = getattr(graph_module, arg.target)
        if not isinstance(submodule, torch.fx.GraphModule):
            continue
        submodule_written = _nodes_written_in_place(submodule)
        placeholders = submodule.graph.find_nodes(op="placeholder")
        if node.target in _WRAPPER_HOPS:
            written += [
                operand
                for placeholder, operand in zip(placeholders, node.args[i + 1 :])
                if placeholder in submodule_written
                and isinstance(operand, torch.fx.Node)
            ]
        elif any(p in submodule_written for p in placeholders):
            # The operand layout is op specific, so any operand may be written.
            written += [
                leaf
                for leaf in pytree.tree_leaves(node.args)
                if isinstance(leaf, torch.fx.Node)
            ]
    return written


def _nodes_written_in_place(
    graph_module: torch.fx.GraphModule,
) -> set[torch.fx.Node]:
    """
    Returns the nodes whose value an in-place op writes, directly, through a
    view or inside a higher order op. Such a value has to stay a runtime
    tensor, not a constant.
    """
    written = set()
    for node in graph_module.graph.nodes:
        if node.op != "call_function":
            continue
        schema = getattr(node.target, "_schema", None)
        if schema is None:
            written.update(_hop_written_operands(graph_module, node))
            continue
        for arg, schema_arg in _schema_paired_args(node, schema):
            alias_info = schema_arg.alias_info
            if alias_info is not None and alias_info.is_write:
                written.update(
                    leaf
                    for leaf in pytree.tree_leaves(arg)
                    if isinstance(leaf, torch.fx.Node)
                )
    if not written:
        return written
    # Every view of a written tensor shares its storage, including views that
    # composite ops like atleast_2d or meshgrid return before decomposition.
    written_storages = set().union(*(_storages(node) for node in written))
    written.update(
        node
        for node in graph_module.graph.nodes
        if not _storages(node).isdisjoint(written_storages)
    )
    # A deserialized program does not share storage between a view and its
    # base, so a write through a view also marks the tensor the view reads.
    pending = list(written)
    while pending:
        node = pending.pop()
        if node.op != "call_function" or not node.args:
            continue
        base = node.args[0]
        is_view = node.target is operator.getitem or getattr(
            node.target, "is_view", False
        )
        if is_view and isinstance(base, torch.fx.Node) and base not in written:
            written.add(base)
            pending.append(base)
    return written


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

    written_in_place = _nodes_written_in_place(exported_program.graph_module)

    for node in exported_program.graph.nodes:
        if (
            node.op != "call_function"
            or node in written_in_place
            or node.target is memory.alloc
            or node.target in all_skip_targets
            # Ops with side effects (RNG draws, mutation) have to run at
            # runtime. `aten.rand` has no tensor inputs, so without this check
            # it would be folded into a single frozen draw.
            or node.is_impure()
        ):
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

            # ExecuTorch doesn't support zero strides, so we need to ensure the tensor is contiguous
            # if it has any zero strides from broadcasting/expansion operations
            if (
                isinstance(prop_constant_tensor, torch.Tensor)
                and 0 in prop_constant_tensor.stride()
            ):
                prop_constant_tensor = prop_constant_tensor.contiguous()

            # A folded constant must be a plain tensor that does not require
            # grad. no_grad is not enough when the op returns a view of a
            # parameter or the parameter itself.
            if isinstance(prop_constant_tensor, torch.Tensor):
                prop_constant_tensor = prop_constant_tensor.detach()
        const_node_to_tensor[node] = prop_constant_tensor

    # Only tensors can be lifted into placeholders. A non-tensor value read by
    # a node outside the fold stays a runtime op, together with the non-tensor
    # values it is computed from.
    for node in reversed(list(const_node_to_tensor)):
        if not isinstance(const_node_to_tensor[node], torch.Tensor) and any(
            user not in const_node_to_tensor for user in node.users
        ):
            del const_node_to_tensor[node]

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
    # A program re-entering this pass can already carry a
    # `_prop_tensor_constant*` in state_dict from an earlier run. Emission
    # resolves placeholders against state_dict before constants, so reusing
    # such a name lets the stale entry shadow the tensor written here. Both
    # the check and the suffix scan therefore span state_dict too.
    taken = exported_program.constants.keys() | exported_program.state_dict.keys()
    if prop_constant_tensor_fqn in taken:
        suffix = 1 + max(
            (
                int(name[len(prefix) :])
                for name in taken
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


# add _skip_dim_order to ensure the introduced correct clone node for different dim order schema
# TODO(gasoonjia): only relying on _clone_dim_order once we remove _skip_dim_order option in the EdgeCompileConfig
def _update_output_node_and_specs(
    exported_program: ExportedProgram, _skip_dim_order: bool
) -> None:
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

    clone_op = (
        exir_ops.edge.aten.clone.default
        if _skip_dim_order
        else exir_ops.edge.dim_order_ops._clone_dim_order.default
    )

    for i in range(len(output_specs)):
        out_node = output_nodes[i]
        if out_node not in updated_constant_placeholders:
            continue

        with exported_program.graph.inserting_after(out_node):
            new_node = exported_program.graph.call_function(clone_op, (out_node,))
        assert "val" in out_node.meta
        new_node.meta["val"] = out_node.meta["val"]
        output_nodes[i] = new_node

        # Update the constant-propagated output node.
        output_specs[i].arg = TensorArgument(name=output_nodes[i].name)

    output.args = (output_nodes,)


def constant_prop_pass(
    exported_program: ExportedProgram,
    custom_skip_targets: Optional[set[EdgeOpOverload]] = None,
    _skip_dim_order: bool = True,
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

    _update_output_node_and_specs(exported_program, _skip_dim_order=_skip_dim_order)

    # Cleanup the graph. A higher order op that writes its operands has no
    # users, yet it has to stay. Graph.eliminate_dead_code also cleans
    # submodules without this rule, which would drop a nested region, so only
    # the top-level graph is cleaned here.
    graph_module = exported_program.graph_module
    for node in reversed(graph_module.graph.nodes):
        if (
            node.op not in ("placeholder", "output")
            and not node.users
            and not node.is_impure()
            and not _hop_written_operands(graph_module, node)
        ):
            graph_module.graph.erase_node(node)
    graph_module.recompile()

    return exported_program
