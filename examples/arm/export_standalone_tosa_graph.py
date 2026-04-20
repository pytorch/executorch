# Copyright 2026 Arm Limited and/or its affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

""" This example demonstrates how to export a standalone TOSA graph with multiple outputs, where the outputs of the delegate
are reordered during partitioning. The example uses a simple convolutional network with three outputs of different shapes,
applies quantization, partitions for TOSA, and then checks the output shapes in the exported graph match the expected output
shapes from the original FX graph. """

import operator

import torch
from executorch.backends.arm.quantizer.arm_quantizer import (
    get_symmetric_quantization_config,
    TOSAQuantizer,
)
from executorch.backends.arm.tosa.partitioner import TosaCompileSpec, TOSAPartitioner
from executorch.backends.arm.tosa.specification import TosaSpecification
from executorch.exir import to_edge_transform_and_lower
from executorch.exir.passes.quantize_io_pass import QuantizeInputs, QuantizeOutputs
from torch import nn
from torch.fx import GraphModule
from torchao.quantization.pt2e.quantize_pt2e import convert_pt2e, prepare_pt2e


def _assert_one_delegate(graph_module: GraphModule):
    delegate_nodes = [
        n
        for n in graph_module.graph.nodes
        if n.target == torch.ops.higher_order.executorch_call_delegate
    ]
    assert (
        len(delegate_nodes) == 1
    ), f"Expected exactly one delegate node, found {len(delegate_nodes)}"


def _delegate_outputs_to_graph_outputs(graph_module: GraphModule) -> list[int]:
    """Returns the mapping from delegate outputs to graph outputs assuming there
    is only one delegate and the graph outputs are a tuple of getitem nodes from
    the delegate outputs. For instance:

    class GraphModule(torch.nn.Module):
        def forward(self, x: "i8[1, 1, 28, 28]"):
            lowered_module_0 = self.lowered_module_0
            executorch_call_delegate = torch.ops.higher_order.executorch_call_delegate(lowered_module_0, x)
            getitem: "i8[1, 3, 28, 28]" = executorch_call_delegate[0]
            getitem_1: "i8[1, 1, 28, 28]" = executorch_call_delegate[1]
            getitem_2: "i8[1, 2, 28, 28]" = executorch_call_delegate[2]
            return [getitem_1, getitem_2, getitem]
    The function would return [2, 0, 1] indicating that delegate output 0 maps to graph output 2,

    """
    graph_outputs_to_delegate = []
    _assert_one_delegate(graph_module)
    outputs = list(graph_module.graph.nodes)[-1].args[0]
    for out in outputs:
        if out.target == operator.getitem:
            graph_outputs_to_delegate.append(out.args[1])
    return [
        graph_outputs_to_delegate.index(i)
        for i in range(len(graph_outputs_to_delegate))
    ]


def get_delegate_output_shapes(graph_module: GraphModule) -> list[list[int]]:
    """Returns the shapes of the delegate outputs."""
    _assert_one_delegate(graph_module)
    delegate_node = [
        n
        for n in graph_module.graph.nodes
        if n.target == torch.ops.higher_order.executorch_call_delegate
    ][0]
    output_types = delegate_node.meta["val"]
    shapes = [list(tensor_type.shape) for tensor_type in output_types]
    return shapes


class Network(nn.Module):
    def __init__(self, batch_norm=False):
        super().__init__()
        self.conv2d_0 = nn.Sequential(
            nn.Conv2d(1, 8, 3, padding=1, bias=False),
            nn.BatchNorm2d(8) if batch_norm else nn.Identity(),
            nn.ReLU(),
        )
        self.conv2d_1 = nn.Sequential(
            nn.Conv2d(8, 8, 3, padding=1, bias=False),
            nn.BatchNorm2d(8) if batch_norm else nn.Identity(),
            nn.ReLU(),
        )
        self.conv2d_2 = nn.Sequential(
            nn.Conv2d(8, 8, 3, padding=1, bias=False),
            nn.BatchNorm2d(8) if batch_norm else nn.Identity(),
            nn.ReLU(),
        )
        self.out_0 = nn.Sequential(nn.Conv2d(8, 1, 3, padding=1, bias=False), nn.ReLU())
        self.out_1 = nn.Sequential(nn.Conv2d(8, 2, 3, padding=1, bias=False), nn.ReLU())
        self.out_2 = nn.Sequential(nn.Conv2d(8, 3, 3, padding=1, bias=False), nn.ReLU())

    def forward(self, x):
        x = self.conv2d_0(x)
        x = self.conv2d_1(x)
        x = self.conv2d_2(x)
        out1 = self.out_1(x)
        out0 = self.out_0(x)
        out2 = self.out_2(x)
        return out0, out1, out2


model = Network(batch_norm=True).eval()
spec = TosaSpecification.create_from_string("TOSA-1.0+INT")
compile_spec = TosaCompileSpec(
    tosa_spec=spec,
)
# Setup quantizer
quantizer = TOSAQuantizer(compile_spec)
quantizer.set_global(
    get_symmetric_quantization_config(is_qat=True, is_per_channel=False)
)
# Trace the model
example_inputs = torch.randn(1, 1, 28, 28)
fx_mod = torch.export.export(model, (example_inputs,)).module()

# Quantize the model
aten_gm = prepare_pt2e(fx_mod, quantizer)
aten_gm(example_inputs)
aten_gm = convert_pt2e(aten_gm)

# Export the quantized model to aten dialect
aten_ep = torch.export.export(aten_gm, args=(example_inputs,), strict=True)

# Lower to TOSA
part = TOSAPartitioner(compile_spec)
ep = to_edge_transform_and_lower(aten_ep, partitioner=[part])

# Remove all io-quant nodes. This simplifies the process of mapping delegate outputs to graph outputs.
ep = ep.transform(passes=[QuantizeInputs(ep, [0]), QuantizeOutputs(ep, [0, 1, 2])])
gm = ep.exported_program().graph_module

# Get output shapes from delegate node
delegate_output_shapes = get_delegate_output_shapes(gm)
# Find mapping between graph outputs and delegate outputs
output_mapping = _delegate_outputs_to_graph_outputs(gm)

# Get expected output shapes from fx_mod
output_shapes = [
    list(out.meta["val"].shape) for out in list(fx_mod.graph.nodes)[-1].args[0]
]

# Compare shapes
for i, shape in enumerate(delegate_output_shapes):
    assert (
        shape == output_shapes[output_mapping[i]]
    ), f"Delegate output shape {shape} does not match expected output shape {output_shapes[output_mapping[i]]}"
