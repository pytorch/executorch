# Copyright 2025 Arm Limited and/or its affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import torch
from executorch.backends.transforms.utils import (
    create_constant_placeholder,
    delete_constant_placeholder,
)
from executorch.exir import to_edge
from executorch.exir.dialects._ops import ops as exir_ops
from torch.export import export
from torch.export.graph_signature import InputKind


class EmptyNetwork(torch.nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x

    test_data: torch.Tensor = (torch.zeros(1),)


def _test_create_delete(kind: InputKind, persistent_buffer: bool = None):
    """
    Tests the utility functions create_constant_placeholder and delete_constant_placeholder
    """

    # Toy network with two nodes, input and output
    # The result should be 0 = 0
    module = EmptyNetwork()
    exported_program = export(module, args=module.test_data, strict=True)
    exported_program = to_edge(exported_program).exported_program()
    graph = exported_program.graph_module.graph
    assert len(graph.nodes) == 2
    assert exported_program.module()(torch.zeros(1)) == 0
    assert len(exported_program.graph_signature.input_specs) == 1
    assert len(exported_program.state_dict) == 0
    assert len(exported_program.constants) == 0

    const_name = "test_node"

    # Create one const node with value 1 and add it to the input
    input_node = list(graph.nodes)[0]
    with graph.inserting_before(input_node):
        const_node = create_constant_placeholder(
            exp_program=exported_program,
            graph=graph,
            kind=kind,
            name=const_name,
            data=torch.ones(1),
            persistent_buffer=persistent_buffer,
        )
    assert "val" in const_node.meta

    with graph.inserting_after(input_node):
        add_node = graph.create_node(
            "call_function",
            exir_ops.edge.aten.add.Tensor,
            args=(input_node, const_node),
            kwargs={},
        )

    output_node = graph.output_node()
    output_node.replace_input_with(input_node, add_node)

    # We should now have four nodes: test_node, input, add, output
    # The result should be 0 + 1 = 1
    assert exported_program.module()(torch.zeros(1)) == 1
    assert len(graph.nodes) == 4

    if kind == InputKind.PARAMETER:
        assert const_name in exported_program.graph_signature.inputs_to_parameters
        assert const_name in exported_program.state_dict
        assert len(exported_program.constants) == 0
    elif kind == InputKind.BUFFER and persistent_buffer:
        assert const_name in exported_program.graph_signature.inputs_to_buffers
        assert const_name in exported_program.state_dict
        assert len(exported_program.constants) == 0
    elif kind == InputKind.BUFFER and not persistent_buffer:
        assert const_name in exported_program.graph_signature.inputs_to_buffers
        assert len(exported_program.state_dict) == 0
        assert const_name in exported_program.constants
    elif kind == InputKind.CONSTANT_TENSOR:
        assert (
            const_name
            in exported_program.graph_signature.inputs_to_lifted_tensor_constants
        )
        assert len(exported_program.state_dict) == 0
        assert const_name in exported_program.constants
    else:
        raise RuntimeError("Wrong input kind")

    # Replacing the add op and using eliminate_dead_code() deletes the add op but not the input op
    output_node.replace_input_with(add_node, input_node)
    graph.eliminate_dead_code()
    assert len(graph.nodes) == 3

    # Delete the input op manually
    # The result should again be 0 = 0
    delete_constant_placeholder(exported_program, const_node)
    assert exported_program.module()(torch.zeros(1)) == 0
    assert len(graph.nodes) == 2
    assert len(exported_program.graph_signature.input_specs) == 1
    assert len(exported_program.state_dict) == 0
    assert len(exported_program.constants) == 0


def test_create_delete_parameter():
    _test_create_delete(InputKind.PARAMETER)


def test_create_delete_persistent_buffer():
    _test_create_delete(InputKind.BUFFER, True)


def test_create_delete_non_persistent_buffer():
    _test_create_delete(InputKind.BUFFER, False)


def test_create_delete_constant_tensor():
    _test_create_delete(InputKind.CONSTANT_TENSOR)
