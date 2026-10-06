# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Shared helpers for the native backend tests."""

import torch

from executorch.backends.native import get_default_compile_config
from executorch.backends.native.partitioner import NativePartitioner
from executorch.backends.native.passes import get_default_passes
from executorch.backends.native.serialization.schema import OpKind
from executorch.exir import to_edge, to_edge_transform_and_lower
from executorch.exir.program._program import lift_constant_tensor_pass
from torch.export.graph_signature import OutputKind, OutputSpec, TensorArgument


def lifted_constant_program(tensors):
    """An edge program returning its input and constants created by a transform."""
    ep = to_edge(
        torch.export.export(
            torch.nn.Identity(), (torch.zeros(1, device=tensors[0].device),)
        )
    ).exported_program()
    output = next(node for node in ep.graph.nodes if node.op == "output")
    constants = []
    with ep.graph.inserting_before(output):
        for index, tensor in enumerate(tensors):
            name = f"constant{index}"
            ep.graph_module.register_buffer(name, tensor)
            constants.append(ep.graph.get_attr(name))
    output.args = ((*output.args[0], *constants),)
    lift_constant_tensor_pass(ep)
    ep.graph_signature.output_specs = [
        OutputSpec(OutputKind.USER_OUTPUT, TensorArgument(name=node.name), None)
        for node in output.args[0]
    ]
    ep.validate()
    return ep


def _transformed(model, example_inputs, passes):
    edge = to_edge(
        torch.export.export(model, example_inputs),
        compile_config=get_default_compile_config(),
    )
    return edge.transform(passes).exported_program()


def _lower(model, example_inputs):
    ep = torch.export.export(model, example_inputs)
    return to_edge_transform_and_lower(
        ep,
        transform_passes=get_default_passes(),
        partitioner=[NativePartitioner()],
        compile_config=get_default_compile_config(),
    )


def _get_delegate_blob(edge):
    et = edge.to_executorch()
    delegates = et.executorch_program.backend_delegate_data
    assert len(delegates) == 1, f"Expected 1 delegate blob, got {len(delegates)}"
    return bytes(delegates[0].data)


def _call_function_targets(graph):
    return [n.target for n in graph.nodes if n.op_kind == OpKind.CALL_FUNCTION]
