# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from typing import List, Union

import torch
from executorch.exir.backend.backend_details import BackendDetails, PreprocessResult
from executorch.exir.backend.compile_spec_schema import CompileSpec
from executorch.exir.backend.utils import DelegateMappingBuilder
from executorch.exir.dialects._ops import ops as exir_ops
from torch import nn
from torch.export.exported_program import ExportedProgram


# A simple way to represent an op along with its delegate debug identifier.
class DummyOp:
    def __init__(
        self,
        op: str,
        delegate_debug_identifier: Union[int, str],
    ):
        self.op = op
        self.delegate_debug_identifier = delegate_debug_identifier
        self.__name__ = self.__repr__()

    def __repr__(self):
        return f"{self.op}"

    def serialize(self):
        return f"{self.op},{self.delegate_debug_identifier},"


"""
This demo implementation is mainly intended to show how the DelegateMappingBuilder should be used
in backends. There are two use cases represented here:
1. A list of decomposed ops are fused into a single backend op and how the delegate debug identifier
mapping is generated for that.
2. A single op is decomposed into two backend ops and we show how the delegate debug identifier mapping
is generated for that.

Here is what the graph looks like for the demo model ConvReLUAddModel implemented in this class:
input
 ↓
conv (debug_handle : 1)
 ↓
relu (debug_handle : 2)
 ↓
tan (debug_handle : 3)
 ↓
output

Here is what the graph that runs in the backend looks like:
input
 ↓
fused_conv_relu (delegate_debug_identifier : a)
 ↓
sin (delegate_debug_identifier : b)
 ↓
cos (delegate_debug_identifier : c)
 ↓
div (delegate_debug_identifier : d)
output

Here is what the delegate mapping looks like. The key is the delegate_debug_identifier and the value
is the debug handles.
{ a : (1,2), b : (3), c: (3), d: (3)}
(NOTE: Here a,b,c can be integers or strings, the decision is left to the user, but whatever is
used during the AOT process to generate the mapping should be the same int/string logged in the
runtime.)

NOTE: these two graphs are not necessarily functionally equivalent but rather representative
examples on how to generated delegate debug identifieres for various use cases such as fusion of ops
in the backend, decomposition of ops in the backend etc.
"""


class BackendWithDelegateMappingDemo(BackendDetails):
    @staticmethod
    def preprocess(
        edge_program: ExportedProgram,
        compile_specs: List[CompileSpec],
    ) -> PreprocessResult:
        processed_bytes = ""
        number_of_instruction = 0
        delegate_builder = DelegateMappingBuilder(generated_identifiers=True)

        for node in edge_program.graph.nodes:
            if node.op == "call_function":
                # Here we demonstrate case 1 where a couple of ops are fused into a single op.
                # In this case the pattern of conv + relu is detected and fused into a single
                # delegated op and the corresponding delegate debug identifier for that is generated
                # and stored in the serialized blob.
                if node.target == exir_ops.edge.aten.relu.default:
                    input_node = node.args[0]
                    if input_node.target == exir_ops.edge.aten.convolution.default:
                        delegate_debug_identifier = (
                            delegate_builder.insert_delegate_mapping_entry(
                                [input_node, node]
                            )
                        )
                        conv_relu_op = DummyOp(
                            "conv_relu_op",
                            delegate_debug_identifier,
                        )
                        number_of_instruction += 1
                        processed_bytes += conv_relu_op.serialize()

                # Here we demonstrate case 2 where a single op is decomposed into three backend ops.
                # In this case the tan op is detected and decomposed into sin, cos and div ops. The
                # corresponding delegate debug identifieres are generated for the three delegatged ops which
                # map to the original tan op. These delegate debug identifieres are then serialized into the
                # blob.
                elif node.target == exir_ops.edge.aten.tan.default:
                    delegate_debug_identifier = (
                        delegate_builder.insert_delegate_mapping_entry(node)
                    )
                    sin_decomp_from_addmm = DummyOp(
                        "sin_decomp_from_tan",
                        delegate_debug_identifier,
                    )
                    number_of_instruction += 1
                    processed_bytes += sin_decomp_from_addmm.serialize()

                    delegate_debug_identifier = (
                        delegate_builder.insert_delegate_mapping_entry(node)
                    )
                    cos_decomp_from_addmm = DummyOp(
                        "cos_decomp_from_tan",
                        delegate_debug_identifier,
                    )
                    number_of_instruction += 1
                    processed_bytes += cos_decomp_from_addmm.serialize()

                    delegate_debug_identifier = (
                        delegate_builder.insert_delegate_mapping_entry(node)
                    )
                    div_decomp_from_addmm = DummyOp(
                        "div_decomp_from_tan",
                        delegate_debug_identifier,
                    )
                    number_of_instruction += 1
                    processed_bytes += div_decomp_from_addmm.serialize()
            elif node.op in ["placeholder", "output", "get_attr"]:
                continue
            else:
                raise RuntimeError(
                    f"{node.op} is not supported in backend BackendWithCompilerDemo"
                )

        return PreprocessResult(
            processed_bytes=bytes(
                str(number_of_instruction) + "#" + processed_bytes, encoding="utf8"
            ),
            debug_handle_map=delegate_builder.get_delegate_mapping(),
        )

    @staticmethod
    # The sample model that will work with BackendWithDelegateMapping show above.
    def get_test_model_and_inputs():
        class SimpleConvNet(nn.Module):
            def __init__(self):
                super(SimpleConvNet, self).__init__()

                # First convolutional layer
                self.conv1 = nn.Conv2d(
                    in_channels=3, out_channels=16, kernel_size=3, stride=1, padding=1
                )
                self.relu1 = nn.ReLU()

                # Second convolutional layer
                self.conv2 = nn.Conv2d(
                    in_channels=16, out_channels=32, kernel_size=3, stride=1, padding=1
                )
                self.relu2 = nn.ReLU()

            def forward(self, x):
                # Forward pass through the first convolutional layer
                x = self.conv1(x)
                x = self.relu1(x)

                # Forward pass through the second convolutional layer
                x = self.conv2(x)
                x = self.relu2(x)

                return x

        class ConvReLUTanModel(nn.Module):
            def __init__(self):
                super(ConvReLUTanModel, self).__init__()

                # Define a convolutional layer
                self.conv_layer = SimpleConvNet()

            def forward(self, x):
                # Forward pass through convolutional layer
                conv_output = self.conv_layer(x)

                # Perform tan on conv_output
                tan_output = torch.tan(conv_output)

                return tan_output

        batch_size = 4
        channels = 3
        height = 64
        width = 64
        return (ConvReLUTanModel(), (torch.randn(batch_size, channels, height, width),))
