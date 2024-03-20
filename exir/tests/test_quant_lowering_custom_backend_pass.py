# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

import copy
import logging

from typing import Callable, Dict, final, List, Tuple

import executorch.exir as exir

import torch
import torch.nn as nn
import torch.nn.functional as F
from executorch.exir.backend.backend_details import BackendDetails, PreprocessResult
from executorch.exir.backend.canonical_partitioners.pattern_op_partitioner import (
    generate_pattern_op_partitions,
)
from executorch.exir.backend.compile_spec_schema import CompileSpec
from executorch.exir.backend.partitioner import (
    DelegationSpec,
    Partitioner,
    PartitionResult,
)
from executorch.exir.dialects._ops import ops as exir_ops

from executorch.exir.pass_base import ExportPass, map_args

from executorch.exir.passes.replace_aten_with_edge_pass import (
    aten_to_edge,
    should_lower_to_edge,
)
from torch import fx
from torch.export import ExportedProgram
from torch.fx import subgraph_rewriter
from torch.fx.passes.infra.pass_base import PassResult


# pyre-ignore
def _trace_and_lower_to_edge_ops(f: Callable) -> fx.GraphModule:
    gm = fx.symbolic_trace(f)
    for node in gm.graph.nodes:
        if node.op == "call_function" and should_lower_to_edge(node.target):
            node.target = aten_to_edge(node.target)
    gm.recompile()
    return gm


# Test Module that we are demonstrating.
# uses quantization flow, delegation
# and custom pass.
#
# The top half of the graph are delegated to the
# NPU. The operators in the NPU only supports
# quantized operator
#
# The bottom half of the graph are delegated to
# the DSP (via custom pass). It also only supports
# quantized operator.
#
#        x    constant_tensor
#        |         |
#   _____|_________|__________
#  |     |         |   (NPU) |
#  |   Conv2D      |         |
#  |     | \       |         |
#  |     |  \      |         |
#  |     |    \    |         |
#  |   Conv2D   \  |         |
#  |     |        Add        |
#  |     |         |         |
#  |     \         |         |
#  |      \       /          |
#  |       \     /           |
#  |          Add            |
#  |           |             |
#  |-------------------------
#              |
#   _________  | ____________
#  |          Conv2D         |
#  |           |       (DSP) |
#  |           |             |
#  |           |             |
#  |          Relu           |
#  |           |             |
#  |           |             |
#  |          MaxPool        |
#  |           |             |
#  |           |             |
#   --------------------------
#              |
#              |
class TestModel(nn.Module):
    __test__: bool = False

    def __init__(self, constant_tensor: torch.Tensor) -> None:
        super().__init__()
        self.constant_tensor = constant_tensor
        self.conv1 = torch.nn.Conv2d(
            in_channels=3, out_channels=16, kernel_size=3, padding=1
        )
        self.conv2 = torch.nn.Conv2d(
            in_channels=16, out_channels=16, kernel_size=3, padding=1
        )
        self.conv3 = torch.nn.Conv2d(
            in_channels=16, out_channels=16, kernel_size=3, padding=1
        )
        self.relu = torch.nn.ReLU()
        self.maxpool = torch.nn.MaxPool2d(kernel_size=3)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        a = self.conv1(x)
        b = self.conv2(a)
        c = a + self.constant_tensor
        z = self.conv3(b + c)
        return self.maxpool(self.relu(z))


class TestFunctionalLinearModel(nn.Module):
    __test__: bool = False

    def __init__(self) -> None:
        super().__init__()
        self.weight: torch.Tensor = torch.randn(5, 5)
        self.bias: torch.Tensor = torch.randn(
            5,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.linear(x, self.weight, self.bias)
        return x


class TestConvBatchNormModel(nn.Module):
    __test__: bool = False

    def __init__(self) -> None:
        super().__init__()
        self.conv = torch.nn.Conv2d(
            in_channels=3, out_channels=16, kernel_size=3, padding=1
        )
        self.bn = torch.nn.BatchNorm2d(16)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        a = self.conv(x)
        b = self.bn(a)
        return b


class ReplaceQuantizedOperatorsWithQualcommDSP(ExportPass):
    def get_qualcom_cdsp_replacements(
        self,
        # pyre-ignore
    ) -> List[Tuple[Callable, Callable, List[Callable]]]:
        def pattern_quantized_conv_relu(
            x: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor
        ) -> torch.Tensor:
            y = torch.ops.quantized_decomposed.dequantize_per_tensor.default(
                x, 1.0, 0, 0, 255, torch.uint8
            )
            z = torch.ops.aten.convolution.default(
                y, weight, bias, [1, 1], [1, 1], [1, 1], False, [0, 0], 1
            )
            t = torch.ops.aten.relu.default(z)
            return torch.ops.quantized_decomposed.quantize_per_tensor.default(
                t, 1.0, 0, 0, 255, torch.uint8
            )

        def replacement_quantized_conv_relu(
            x: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor
        ) -> torch.Tensor:
            # Replace with custom target specifc operator
            # Sigmoid is just for demonstration only.
            # It would by something like `qualcomm.ops.dsp.conv_relu`
            return torch.ops.aten.sigmoid(x)

        return [
            (
                _trace_and_lower_to_edge_ops(pattern_quantized_conv_relu),
                _trace_and_lower_to_edge_ops(replacement_quantized_conv_relu),
                [],
            )
            # More pattern-replacements could be added here
        ]

    def call(self, graph_module: fx.GraphModule) -> PassResult:
        new_graph_module = copy.deepcopy(graph_module)
        for pattern, replacement, match_filters in self.get_qualcom_cdsp_replacements():
            subgraph_rewriter.replace_pattern_with_filters(
                new_graph_module, pattern, replacement, match_filters
            )
        return PassResult(new_graph_module, True)


class PatternWrapper:
    # pyre-ignore Invalid type parameters [24]: Generic type `Callable` expects 2 type parameters.
    def __init__(self, pattern: Callable, inputs: Tuple) -> None:
        self.pattern = pattern
        self.inputs = inputs

    def get_graph_module(self) -> torch.fx.GraphModule:
        return (
            exir.capture(
                self.pattern,
                self.inputs,
                exir.CaptureConfig(),
            )
            .to_edge(
                exir.EdgeCompileConfig(
                    _check_ir_validity=False,
                )
            )
            .exported_program.graph_module
        )


class NonQuantizedConvolutionPatternExample(PatternWrapper):
    def __init__(self) -> None:
        def non_quantized_conv_pattern(
            x: torch.Tensor,
            weight: torch.Tensor,
            bias: torch.Tensor,
        ) -> torch.Tensor:
            return torch.ops.aten.convolution.default(
                x, weight, bias, [1, 1], [1, 1], [1, 1], False, [0, 0], 1
            )

        inputs: Tuple[torch.Tensor, torch.Tensor, torch.Tensor] = (
            torch.ones(1, 3, 3, 3, dtype=torch.float),
            torch.ones(1, 3, 3, 3),
            torch.ones(1),
        )
        super().__init__(
            non_quantized_conv_pattern,
            inputs,
        )


class QuantizedConvolutionPatternExample(PatternWrapper):
    def __init__(self) -> None:
        def quantized_conv_pattern(
            x: torch.Tensor,
            weight: torch.Tensor,
            bias: torch.Tensor,
        ) -> torch.Tensor:
            x = torch.ops.quantized_decomposed.dequantize_per_tensor(
                x, 1.0, 0, 0, 255, torch.uint8
            )
            x = torch.ops.aten.convolution.default(
                x, weight, bias, [1, 1], [1, 1], [1, 1], False, [0, 0], 1
            )
            x = torch.ops.quantized_decomposed.quantize_per_tensor(
                x, 1.0, 0, 0, 255, torch.uint8
            )
            return x

        inputs: Tuple[torch.Tensor, torch.Tensor, torch.Tensor] = (
            torch.ones(1, 3, 3, 3, dtype=torch.uint8),
            torch.ones(1, 3, 3, 3),
            torch.ones(1),
        )

        super().__init__(
            quantized_conv_pattern,
            inputs,
        )


class QuantizeAddNonQuantizePatternExample(PatternWrapper):
    def __init__(self) -> None:
        def quantized_add_pattern(
            x: torch.Tensor,
            y: torch.Tensor,
        ) -> torch.Tensor:
            x = torch.ops.quantized_decomposed.dequantize_per_tensor(
                x, 1.0, 0, 0, 255, torch.uint8
            )
            x = torch.ops.aten.add.Tensor(x, y)
            x = torch.ops.quantized_decomposed.quantize_per_tensor(
                x, 1.0, 0, 0, 255, torch.uint8
            )
            return x

        inputs: Tuple[torch.Tensor, torch.Tensor] = (
            torch.ones(1, 3, 3, 3, dtype=torch.uint8),
            torch.ones(1, 3, 3, 3, dtype=torch.float32),
        )
        super().__init__(
            quantized_add_pattern,
            inputs,
        )


class QuantizeAddQuantizePatternExample(PatternWrapper):
    def __init__(self) -> None:
        def quantized_add_pattern1(
            x: torch.Tensor,
            y: torch.Tensor,
        ) -> torch.Tensor:
            x = torch.ops.quantized_decomposed.dequantize_per_tensor(
                x, 1.0, 0, 0, 255, torch.uint8
            )
            y = torch.ops.quantized_decomposed.dequantize_per_tensor(
                y, 1.0, 0, 0, 255, torch.uint8
            )
            x = torch.ops.aten.add.Tensor(x, y)
            x = torch.ops.quantized_decomposed.quantize_per_tensor(
                x, 1.0, 0, 0, 255, torch.uint8
            )
            return x

        inputs: Tuple[torch.Tensor, torch.Tensor] = (
            torch.ones(1, 3, 3, 3, dtype=torch.uint8),
            torch.ones(1, 3, 3, 3, dtype=torch.uint8),
        )
        super().__init__(
            quantized_add_pattern1,
            inputs,
        )


class NonQuantizeSubPatternExample(PatternWrapper):
    def __init__(self) -> None:
        def non_quantized_sub_pattern(
            x: torch.Tensor,
            y: torch.Tensor,
        ) -> torch.Tensor:
            return torch.ops.aten.sub.Tensor(x, y)

        inputs: Tuple[torch.Tensor, torch.Tensor] = (
            torch.ones(1, 3, 3, 3, dtype=torch.uint8),
            torch.ones(1, 3, 3, 3, dtype=torch.uint8),
        )
        super().__init__(
            non_quantized_sub_pattern,
            inputs,
        )


class ReplaceQuantizedOperatorsWithQualcommNPU(ExportPass):
    # This pass will replace dq->op->q to quantize_op. Just as proof of concept, we replace the pattern
    # [dq->add->q] to [sub] and [dq->conv->q] to [conv]. In real case, the pattern should be [dq->add->q] => [qualcomm::add]
    # and [dq->conv->q] => [qualcomm::conv]
    def get_qualcom_npu_replacements(
        self,
    ) -> List[
        Tuple[torch.fx.GraphModule, torch.fx.GraphModule, List[torch.fx.GraphModule]]
    ]:
        return [
            (
                QuantizeAddQuantizePatternExample().get_graph_module(),
                NonQuantizeSubPatternExample().get_graph_module(),
                [],
            ),
            (
                QuantizeAddNonQuantizePatternExample().get_graph_module(),
                NonQuantizeSubPatternExample().get_graph_module(),
                [],
            ),
            (
                QuantizedConvolutionPatternExample().get_graph_module(),
                NonQuantizedConvolutionPatternExample().get_graph_module(),
                [],
            ),
        ]

    def set_replacement_node_metadata(
        self,
        match_and_replacements: List[torch.fx.subgraph_rewriter.ReplacedPatterns],
    ) -> None:
        """
        Sets the replacement node's metadata to be the metadata of the last node
        in the pattern graph
        """
        for m in match_and_replacements:
            last_node_in_pattern = m.nodes_map[m.anchor]

            # In this case we know that we only replaced the pattern with one
            # node/op
            assert len(m.replacements) == 1
            replaced_node = m.replacements[0]
            replaced_node.meta = last_node_in_pattern.meta

    def call(self, graph_module: fx.GraphModule) -> PassResult:
        new_graph_module = copy.deepcopy(graph_module)
        for pattern, replacement, match_filters in self.get_qualcom_npu_replacements():
            match_and_replacements = fx.subgraph_rewriter.replace_pattern_with_filters(
                new_graph_module, pattern, replacement, match_filters
            )
            self.set_replacement_node_metadata(match_and_replacements)

        for node in new_graph_module.graph.nodes:
            assert node.op != "call_function" or node.meta.get("val") is not None
        return PassResult(new_graph_module, True)


class DuplicateDequantNodePass(ExportPass):
    """
    Duplicates all of the dequantize nodes
    """

    # pyre-ignore
    dequant_map = {}  # Map of dequant results to its node's arguments

    # pyre-ignore
    def call_operator(self, op, args, kwargs, meta):
        if op == exir_ops.edge.quantized_decomposed.dequantize_per_tensor.default:
            res = super().call_operator(op, args, kwargs, meta)
            self.dequant_map[id(res)] = (op, args, kwargs, meta)
            return res

        # pyre-ignore
        def copy_dequant(value, arg_schema):
            if id(value) in self.dequant_map:
                dequant_node_args = self.dequant_map[id(value)]
                dequant_copy = self.call_operator(*dequant_node_args)
                return dequant_copy
            return value

        args, kwargs = map_args(op, copy_dequant, args, kwargs)
        return super().call_operator(op, args, kwargs, meta)


@final
class ConvAddBackendDemo(BackendDetails):
    @staticmethod
    def preprocess(
        edge_program: ExportedProgram,
        compile_specs: List[CompileSpec],
    ) -> PreprocessResult:
        sub_module = copy.deepcopy(edge_program.graph_module)
        modified_sub_module_with_delegate = ReplaceQuantizedOperatorsWithQualcommNPU()(
            sub_module
        )
        processed_bytes = ""
        if modified_sub_module_with_delegate:
            sub_module_with_delegate = modified_sub_module_with_delegate.graph_module
            for node in sub_module_with_delegate.graph.nodes:
                if node.op == "call_function":
                    processed_bytes += node.target.__name__ + ","
        return PreprocessResult(
            processed_bytes=bytes(processed_bytes, encoding="utf8"),
        )


@final
class QuantizedConvAddOpPartitioner(Partitioner):
    def __init__(self) -> None:
        self.patterns: List[torch.fx.Graph] = [
            QuantizedConvolutionPatternExample().get_graph_module().graph,
            QuantizeAddNonQuantizePatternExample().get_graph_module().graph,
            QuantizeAddQuantizePatternExample().get_graph_module().graph,
        ]

        self.delegation_spec = DelegationSpec(ConvAddBackendDemo.__name__, [])

    def partition(self, edge_exported_program: ExportedProgram) -> PartitionResult:
        partition_tags: Dict[str, DelegationSpec] = {}
        partition_list = generate_pattern_op_partitions(
            edge_exported_program.graph_module, patterns=self.patterns
        )
        logging.debug(partition_list)
        for partition in partition_list:
            for node in partition.nodes:
                delegation_tag = f"tag{partition.id}"
                node.meta["delegation_tag"] = delegation_tag
                partition_tags[delegation_tag] = self.delegation_spec
        return PartitionResult(
            tagged_exported_program=edge_exported_program,
            partition_tags=partition_tags,
        )
