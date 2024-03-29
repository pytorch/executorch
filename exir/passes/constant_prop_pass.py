# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import torch
from torch._export.utils import (
    get_buffer,
    get_param,
    is_buffer,
    is_lifted_tensor_constant,
    is_param,
)
from torch._guards import detect_fake_mode
from torch.export import ExportedProgram
from torch.export.exported_program import InputKind, InputSpec, TensorArgument


def is_const(arg, exported_program, const_data_list) -> bool:
    if isinstance(arg, (tuple, list)):
        return all(is_const(x, exported_program, const_data_list) for x in arg)
    elif isinstance(arg, dict):
        return all(is_const(x, exported_program, const_data_list) for x in arg.values())
    elif not isinstance(arg, torch.fx.Node) or arg.op != "placeholder":
        return False
    elif (
        is_param(exported_program, arg)
        or is_buffer(exported_program, arg)
        or is_lifted_tensor_constant(exported_program, arg)
        or arg.name in const_data_list
    ):
        return True
    return False


def get_data(exported_program, arg):
    if isinstance(arg, (tuple, list)):
        return [get_data(exported_program, x) for x in arg]
    elif is_param(exported_program, arg):
        return get_param(exported_program, arg)
    elif is_buffer(exported_program, arg):
        return get_buffer(exported_program, arg)
    elif arg.name in exported_program.constants:
        return exported_program.constants[arg.name]
    return None


def constant_prop_pass(exported_program: ExportedProgram) -> ExportedProgram:
    """
    This pass is for constant propagation for Exported Program with lifted parameters,
    as the parameters will not be shown up as `get_attr` but as `placeholder` to the graph.
    """
    if (
        len([node for node in exported_program.graph.nodes if node.op == "placeholder"])
        == 0
    ):
        return exported_program

    has_cond = [
        node
        for node in exported_program.graph.nodes
        if node.target == torch.ops.higher_order.cond
    ]
    if len(has_cond) > 0:
        raise RuntimeError("constant_prop_pass for control flow is not supported yet.")

    first_user_input = None
    for node in exported_program.graph.nodes:
        if (
            node.op == "placeholder"
            and node.name in exported_program.graph_signature.user_inputs
        ):
            first_user_input = node
            break

    buffers = exported_program.graph_signature.buffers
    prop_constant_data = []
    const_data_to_be_removed = set()

    fake_mode = detect_fake_mode(
        tuple(
            node.meta["val"]
            for node in exported_program.graph.nodes
            if node.op == "placeholder"
        )
    )
    assert fake_mode is not None

    for node in exported_program.graph.nodes:
        if node.op == "call_function":
            constant_data_name_list = [
                input_spec.target for input_spec in prop_constant_data
            ]
            if is_const(node.args, exported_program, constant_data_name_list):
                args_data = [get_data(exported_program, arg) for arg in node.args]
                kwargs_data = node.kwargs
                const_data_to_be_removed.update(node.args)
                prop_constant_tensor = node.target(*args_data, **kwargs_data)
                prop_constant_tensor_fqn = f"_prop_tensor_constant{len(buffers)}"

                with exported_program.graph.inserting_before(first_user_input):
                    const_placeholder_node = exported_program.graph.placeholder(
                        prop_constant_tensor_fqn
                    )
                    # Update the meta data of the new placeholder (buffer) node
                    for k, v in node.meta.items():
                        const_placeholder_node.meta[k] = v
                    const_placeholder_node.meta["val"] = fake_mode.from_tensor(
                        prop_constant_tensor, static_shapes=True
                    )
                    const_placeholder_node.meta["val"].constant = prop_constant_tensor

                    node.replace_all_uses_with(const_placeholder_node)
                    exported_program.graph.erase_node(node)
                    prop_constant_node_input_spec = InputSpec(
                        kind=InputKind.BUFFER,
                        arg=TensorArgument(name=const_placeholder_node.name),
                        target=prop_constant_tensor_fqn,
                        persistent=True,
                    )
                    prop_constant_data.append(prop_constant_node_input_spec)
                    buffers.append(prop_constant_tensor_fqn)
                    exported_program.state_dict[prop_constant_tensor_fqn] = (
                        prop_constant_tensor
                    )
                    exported_program.graph_signature.input_specs.append(
                        prop_constant_node_input_spec
                    )

    # Remove the propogated buffer from the state dict
    for node in exported_program.graph.nodes:
        if (
            node.op == "placeholder"
            and node in const_data_to_be_removed
            and len(node.users) == 0
        ):
            exported_program.state_dict.pop(node.name, None)
            exported_program.graph.erase_node(node)

    exported_program.graph_module.recompile()
    return exported_program
