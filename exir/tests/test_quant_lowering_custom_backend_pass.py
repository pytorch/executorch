# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

import copy
import logging
import unittest

from typing import Callable, Dict, final, List, Tuple

import executorch.exir as exir

import torch
import torch.nn as nn
import torch.nn.functional as F
from executorch.exir import CaptureConfig
from executorch.exir.backend.backend_api import to_backend
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

from torch.ao.quantization import (  # @manual
    default_dynamic_qconfig,
    get_default_qconfig_mapping,
    QConfig,
)

from torch.ao.quantization.backend_config.executorch import (
    get_executorch_backend_config,
)
from torch.ao.quantization.observer import (
    per_channel_weight_observer_range_neg_127_to_127,
    PlaceholderObserver,
)
from torch.ao.quantization.qconfig_mapping import (
    _get_symmetric_qnnpack_qconfig_mapping,
    QConfigMapping,
)
from torch.ao.quantization.quantize_fx import (
    _convert_to_reference_decomposed_fx,
    prepare_fx,
    prepare_qat_fx,
)
from torch.export import ExportedProgram
from torch.fx import subgraph_rewriter
from torch.fx.passes.infra.pass_base import PassResult
from torch.testing import FileCheck


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


class TestQuantLoweringCustomBackendPass(unittest.TestCase):
    def setUp(self) -> None:
        super().setUp()

    @torch.inference_mode()  # TODO Use  for capturing.
    def test(self) -> None:
        mod = TestModel(
            constant_tensor=torch.ones(
                1,
                16,
                256,
                256,
            ),
        )
        example_inputs = (torch.rand(1, 3, 256, 256),)
        mod.eval()
        # Original module:
        #     x -> conv -> conv_1 -> add_1 -> conv_2 -> relu -> max_pool -> out
        #               \         /
        #                -> add ->

        # Step 1: Source-to-source transformation
        # Converting fp32 operator to quantized operator (i.e. lower precision).
        #
        # As an intermediate step,
        # it converts fp32op(float input) to
        # `quantize->dequant->fp32op(float input)->quant->dequant` format
        #
        # For example, imagine the original graph was something like:
        # 3.14 -> multiplyByTwo() -> 6.28
        #
        # Then it will convert to:
        #
        # 3.14 -> 3 -> 3.0 -> multiplyByTwo -> 6.0 -> 6 -> 6.0
        #
        # Such additional indirections is necessary so that we can still run in CPU
        # using existing floating point operator but emulate the effect of quantization
        # which is useful for testing/debugging.
        qconfig_mapping = get_default_qconfig_mapping("qnnpack")
        prepared_mod = prepare_fx(
            mod,
            qconfig_mapping,
            example_inputs,
            backend_config=get_executorch_backend_config(),
        )
        converted_mod: torch.fx.GraphModule = _convert_to_reference_decomposed_fx(
            prepared_mod,
            backend_config=get_executorch_backend_config(),
        )

        # Step 2: EXIR capturing + duplicating dequant nodes
        captured_program = exir.capture(
            converted_mod, example_inputs, exir.CaptureConfig()
        ).to_edge(
            exir.EdgeCompileConfig(
                _check_ir_validity=False,
            )
        )

        # After quantization/tracing:
        #     x -> quant -> dequant -> conv -> quant_1 -> dequant_1 -> conv_1 -> quant_2 -> dequant_2 -> add_1 -> quant_4 -> dequant_4 -> conv_2 -> relu -> quant_5 -> dequant_5 -> max_pool -> out
        #                                                           \                                 /
        #                                                            -> add -> quant_3 -> dequant_3 ->
        # After duplication:
        #     x -> quant -> dequant_1 -> conv -> quant_1 -> dequant_3 -> conv_1 -> quant_2 -> dequant_7 -> add_1 -> quant_4 -> dequant_10 -> conv_2 -> relu -> quant_5 -> dequant_12 -> max_pool -> out
        #                                                \                                               /
        #                                                 -> dequant_5 -> add -> quant_3 -> dequant_8 ->

        # Step 3.1: Partitioning and delegation using to_backend()
        delegated_mod = to_backend(
            captured_program.transform(DuplicateDequantNodePass()).exported_program,
            QuantizedConvAddOpPartitioner(),
        )
        lowered_module_0 = delegated_mod.graph_module.lowered_module_0

        # The blob in the example backend is a list of ops, examining them to ensure they are replaced correctly.
        FileCheck().check(
            "aten.convolution.default,aten.sub.Tensor,aten.convolution.default,aten.sub.Tensor,"
        ).check_not(
            "executorch_exir_dialects_edge__ops_quantized_decomposed_quantize_per_tensor_tensor"
        ).check_not(
            "executorch_exir_dialects_edge__ops_quantized_decomposed_dequantize_per_tensor_tensor"
        ).run(
            lowered_module_0.processed_bytes
        )

        # After partitioning/to_backend:
        #                    -------------------------------------------------------------------------------------------------
        #     x -> quant -> | dequant_6 -> conv -> quant_1 -> dequant_7 -> conv_1 -> quant_2 -> dequant_9 -> add_1 -> quant_4 | -> dequant_11 -> conv_2 -> relu -> quant_5 -> dequant_12 -> max_pool -> out
        #                   |                              \                                               /                  |
        #                   |                               -> dequant_8 -> add -> quant_3 -> dequant_10 ->                   |
        #                    -------------------------------------------------------------------------------------------------

        # Check the toplevel graph
        FileCheck().check(
            "executorch_exir_dialects_edge__ops_quantized_decomposed_quantize_per_tensor_default"
        ).check("torch.ops.higher_order.executorch_call_delegate").check(
            "executorch_exir_dialects_edge__ops_quantized_decomposed_dequantize_per_tensor_default"
        ).check(
            "executorch_exir_dialects_edge__ops_quantized_decomposed_dequantize_per_tensor_default"
        ).check(
            "executorch_exir_dialects_edge__ops_aten_max_pool2d_with_indices_default"
        ).run(
            delegated_mod.graph_module.code
        )

        # Check lowered module
        FileCheck().check_count(
            "executorch_exir_dialects_edge__ops_quantized_decomposed_dequantize_per_tensor_default",
            5,
        ).run(lowered_module_0.original_module.graph_module.code)

        FileCheck().check(
            "executorch_exir_dialects_edge__ops_quantized_decomposed_dequantize_per_tensor_default"
        ).check("executorch_exir_dialects_edge__ops_aten_convolution_default").check(
            "executorch_exir_dialects_edge__ops_aten_relu_default"
        ).check_not(
            "executorch_exir_dialects_edge__ops_aten_sigmoid"
        ).run(
            delegated_mod.graph_module.code
        )

        # # Step 4:
        #   - Retracing to verify that it is still runnable before custom passes
        #   - Target-aware pass where it fuses quantized ConvRelu and MaxPool to a DSP
        # fused_mod = exir.capture(
        #     delegated_mod, example_inputs, exir.CaptureConfig()
        # ).to_edge(
        #     EdgeCompileConfig(
        #         passes=[ReplaceQuantizedOperatorsWithQualcommDSP()],
        #         _check_ir_validity=False,
        #     )
        # ).graph_module

        # FileCheck().check("torch.ops.aten.sigmoid").check_not(
        #     "torch.ops.aten.convolution.default"
        # ).run(fused_mod.code)

    def test_quantized_linear_dynamic(self) -> None:
        mod = TestFunctionalLinearModel()
        example_inputs = (torch.rand(1, 5),)
        mod.eval()
        # Original module:
        #     x -> linear -> out

        # Step 1: Source-to-source transformation
        # Converting fp32 operator to quantized operator (i.e. lower precision).
        #
        # As an intermediate step,
        # it converts fp32op(float input) to
        # `quantize->dequant->fp32op(float input)->quant->dequant` format
        #
        qconfig_mapping = QConfigMapping().set_object_type(
            F.linear, default_dynamic_qconfig
        )

        prepared_mod = prepare_fx(
            mod,
            qconfig_mapping,
            example_inputs,
            backend_config=get_executorch_backend_config(),
        )
        converted_mod: torch.fx.GraphModule = _convert_to_reference_decomposed_fx(
            prepared_mod,
            backend_config=get_executorch_backend_config(),
        )

        # Step 2: EXIR capturing
        capture_config = CaptureConfig(enable_aot=True, _unlift=True)
        captured_mod = (
            exir.capture(converted_mod, example_inputs, config=capture_config)
            .to_edge(
                exir.EdgeCompileConfig(
                    _check_ir_validity=False,
                )
            )
            .exported_program.graph_module
        )

        print("captured mod:", captured_mod)

        # After quantization/tracing:
        #                weight -> quant -> dequant* -> t* -\
        #     x -> choose_qparams* -> quant* -> dequant* -> addmm* -> out
        # note: nodes with `*` should be fused in delegation
        FileCheck().check(
            "executorch_exir_dialects_edge__ops_quantized_decomposed_choose_qparams_tensor"
        ).check(
            "executorch_exir_dialects_edge__ops_quantized_decomposed_quantize_per_tensor_tensor"
        ).check(
            "executorch_exir_dialects_edge__ops_aten_permute_copy_default"
        ).check(
            "executorch_exir_dialects_edge__ops_aten_addmm"
        ).run(
            captured_mod.code
        )

        # Step 3.1: Partitioning and delegation using to_backend()
        # delegated_mod = to_backend(captured_mod, QuantizedLinearDynamicOpPartitioner)
        # The blob in the example backend is a list of ops, examining them to ensure they are replaced correctly.
        # FileCheck().check(
        #     "convolution.default,sub.Tensor,convolution.default,sub.Tensor,"
        # ).check_not(
        #     "torch.ops.quantized_decomposed.quantize_per_tensor.default"
        # ).check_not(
        #     "torch.ops.quantized_decomposed.dequantize_per_tensor.default"
        # ).run(
        #     delegated_mod.lowered_module_0.processed_bytes
        # )

        # After partitioning/to_backend:
        #           -----------------------------------------------
        #     x -> | choose_qparams -> quant -> dequant -> linear  | -> out
        #           -----------------------------------------------

        # Check the toplevel graph
        # FileCheck().check(
        #     "torch.ops.quantized_decomposed.quantize_per_tensor.default"
        # ).check("lowered_module_0.forward").check(
        #     "torch.ops.quantized_decomposed.dequantize_per_tensor.default"
        # ).check(
        #     "torch.ops.aten.convolution.default"
        # ).check(
        #     "torch.ops.aten.relu.default"
        # ).check(
        #     "torch.ops.quantized_decomposed.quantize_per_tensor.default"
        # ).check(
        #     "torch.ops.quantized_decomposed.dequantize_per_tensor.default"
        # ).check(
        #     "torch.ops.aten.max_pool2d_with_indices.default"
        # ).run(
        #     delegated_mod.code
        # )

        # Check lowered module
        # FileCheck().check_count(
        #     "torch.ops.quantized_decomposed.dequantize_per_tensor.default", 5
        # ).run(delegated_mod.lowered_module_0.original_module.code)
        # FileCheck().check_count(
        #     "torch.ops.quantized_decomposed.quantize_per_tensor.default", 4
        # ).run(delegated_mod.lowered_module_0.original_module.code)
        # FileCheck().check_count("torch.ops.aten.add.Tensor", 2).run(
        #     delegated_mod.lowered_module_0.original_module.code
        # )
        # FileCheck().check_count("torch.ops.aten.convolution.default", 2).run(
        #     delegated_mod.lowered_module_0.original_module.code
        # )

        # # Step 4:
        #   - Retracing to verify that it is still runnable before custom passes
        #   - Target-aware pass where it fuses quantized ConvRelu and MaxPool to a DSP
        # FileCheck().check(
        #     "torch.ops.quantized_decomposed.dequantize_per_tensor.default"
        # ).check("torch.ops.aten.convolution.default").check(
        #     "torch.ops.aten.relu.default"
        # ).check_not(
        #     "torch.ops.aten.sigmoid"
        # ).run(
        #     delegated_mod.code
        # )
        # fused_mod = exir.capture(delegated_mod, example_inputs).to_edge(
        #     EdgeCompileConfig(passes=[ReplaceQuantizedOperatorsWithQualcommDSP()])
        # )

        # FileCheck().check("torch.ops.aten.sigmoid").check_not(
        #     "torch.ops.aten.convolution.default"
        # ).run(fused_mod.code)

    def test_quantized_linear_dynamic_symmetric_act_per_channel_weight(self) -> None:
        mod = TestFunctionalLinearModel()
        example_inputs = (torch.rand(1, 5),)
        mod.eval()
        # Original module:
        #     x -> linear -> out

        # Step 1: Source-to-source transformation
        # Converting fp32 operator to quantized operator (i.e. lower precision).
        #
        # As an intermediate step,
        # it converts fp32op(float input) to
        # `quantize->dequant->fp32op(float input)->quant->dequant` format
        #
        act_symmetric_quant_obs = PlaceholderObserver.with_args(
            dtype=torch.qint8,
            qscheme=torch.per_tensor_affine,
            quant_min=-128,
            quant_max=127,
            eps=2**-12,
            is_dynamic=True,
        )
        qconfig = QConfig(
            activation=act_symmetric_quant_obs,
            weight=per_channel_weight_observer_range_neg_127_to_127,
        )
        qconfig_mapping = QConfigMapping().set_object_type(F.linear, qconfig)

        prepared_mod = prepare_fx(
            mod,
            qconfig_mapping,
            example_inputs,
            backend_config=get_executorch_backend_config(),
        )
        converted_mod: torch.fx.GraphModule = _convert_to_reference_decomposed_fx(
            prepared_mod,
            backend_config=get_executorch_backend_config(),
        )
        print("converted:", converted_mod)

        # Step 2: EXIR capturing
        capture_config = CaptureConfig(enable_aot=True, _unlift=True)
        captured_mod = exir.capture(
            converted_mod, example_inputs, config=capture_config
        ).to_edge(
            exir.EdgeCompileConfig(
                _check_ir_validity=False,
            )
        )

        print("captured mod:", captured_mod)

        # After quantization/tracing:
        #                weight -> quant -> dequant* -> t* -\
        #     x -> choose_qparams* -> quant* -> dequant* -> addmm* -> out
        # note: nodes with `*` should be fused in delegation
        FileCheck().check(
            "executorch_exir_dialects_edge__ops_quantized_decomposed_choose_qparams_tensor"
        ).check(
            "executorch_exir_dialects_edge__ops_quantized_decomposed_quantize_per_tensor_tensor"
        ).check(
            "executorch_exir_dialects_edge__ops_aten_permute_copy_default"
        ).check(
            "executorch_exir_dialects_edge__ops_aten_addmm"
        ).run(
            captured_mod.exported_program.graph_module.code
        )

    def test_quantized_linear_dynamic_symmetric_act_per_tensor_weight(self) -> None:
        mod = TestFunctionalLinearModel()
        example_inputs = (torch.rand(1, 5),)
        mod.eval()
        # Original module:
        #     x -> linear -> out

        # Step 1: Source-to-source transformation
        # Converting fp32 operator to quantized operator (i.e. lower precision).
        #
        # As an intermediate step,
        # it converts fp32op(float input) to
        # `quantize->dequant->fp32op(float input)->quant->dequant` format
        #
        act_xnnpack_quant_obs = PlaceholderObserver.with_args(
            dtype=torch.qint8,
            qscheme=torch.per_tensor_affine,
            quant_min=-128,
            quant_max=127,
            eps=2**-12,
            is_dynamic=True,
        )
        qconfig = QConfig(
            activation=act_xnnpack_quant_obs,
            weight=torch.ao.quantization.observer.weight_observer_range_neg_127_to_127,
        )
        qconfig_mapping = QConfigMapping().set_object_type(F.linear, qconfig)

        prepared_mod = prepare_fx(
            mod,
            qconfig_mapping,
            example_inputs,
            backend_config=get_executorch_backend_config(),
        )
        converted_mod: torch.fx.GraphModule = _convert_to_reference_decomposed_fx(
            prepared_mod,
            backend_config=get_executorch_backend_config(),
        )
        print("converted:", converted_mod)

        # Step 2: EXIR capturing
        capture_config = CaptureConfig(enable_aot=True, _unlift=True)
        captured_mod = exir.capture(
            converted_mod, example_inputs, config=capture_config
        ).to_edge(
            exir.EdgeCompileConfig(
                _check_ir_validity=False,
            )
        )

        print("captured mod:", captured_mod)

        # After quantization/tracing:
        #                weight -> quant -> dequant* -> t* -\
        #     x -> choose_qparams* -> quant* -> dequant* -> addmm* -> out
        # note: nodes with `*` should be fused in delegation
        FileCheck().check(
            "executorch_exir_dialects_edge__ops_quantized_decomposed_choose_qparams_tensor"
        ).check(
            "executorch_exir_dialects_edge__ops_quantized_decomposed_quantize_per_tensor_tensor"
        ).check(
            "executorch_exir_dialects_edge__ops_aten_permute_copy_default"
        ).check(
            "executorch_exir_dialects_edge__ops_aten_addmm"
        ).run(
            captured_mod.exported_program.graph_module.code
        )

    def test_quantized_conv_bn_fusion(self) -> None:
        mod = TestConvBatchNormModel().eval()
        example_inputs = (torch.rand(1, 3, 5, 5),)
        # Original module:
        #       x -> conv -> bn -> out

        # Step 1: Source-to-source transformation
        # Converting fp32 operator to quantized operator (i.e. lower precision).
        #
        # As an intermediate step,
        # it converts fp32op(float input) to
        # `quantize->dequant->fp32op(float input)->quant->dequant` format
        #
        qconfig_mapping = _get_symmetric_qnnpack_qconfig_mapping()
        for prepare_fn in [prepare_fx, prepare_qat_fx]:
            prepared_mod = prepare_fn(
                mod,
                qconfig_mapping,
                example_inputs,
                backend_config=get_executorch_backend_config(),
            )
            converted_mod: torch.fx.GraphModule = _convert_to_reference_decomposed_fx(
                prepared_mod,
                backend_config=get_executorch_backend_config(),
            )
            print("converted:", converted_mod)
            captured_mod = exir.capture(converted_mod, example_inputs).to_edge(
                exir.EdgeCompileConfig(_check_ir_validity=False, _use_edge_ops=True)
            )

            print("captured mod:", captured_mod)

            # After quantization/tracing batchnorm should be fused away
            #        weight -> quant -> dequant* -> t* -\
            #       x -> quant -> dequant* -> conv -> quant -> out
            # note: nodes with `*` should be fused in delegation
            FileCheck().check_count(
                "executorch_exir_dialects_edge__ops_aten_native_batch_norm_default",
                0,
                exactly=True,
            ).run(captured_mod.exported_program.graph_module.code)

    def test_qat_linear(self) -> None:
        mod = TestFunctionalLinearModel().eval()
        example_inputs = (torch.ones(1, 5),)
        # Original module:
        #       x -> linear -> out

        # Step 1: Source-to-source transformation
        # Converting fp32 operator to quantized operator (i.e. lower precision).
        #
        # As an intermediate step,
        # it converts fp32op(float input) to
        # `quantize->dequant->fp32op(float input)->quant->dequant` format
        #

        qconfig_mapping = _get_symmetric_qnnpack_qconfig_mapping()
        for prepare_fn in [prepare_fx, prepare_qat_fx]:
            prepared_mod = prepare_fn(
                mod,
                qconfig_mapping,
                example_inputs,
                backend_config=get_executorch_backend_config(),
            )
            converted_mod: torch.fx.GraphModule = _convert_to_reference_decomposed_fx(
                prepared_mod,
                backend_config=get_executorch_backend_config(),
            )
            print("converted:", converted_mod)
            captured_mod = exir.capture(
                converted_mod,
                example_inputs,
            ).to_edge(
                exir.EdgeCompileConfig(_check_ir_validity=False, _use_edge_ops=True)
            )

            print("captured mod:", captured_mod)

            # After quantization/tracing batchnorm should be fused away
            #        weight -> quant -> dequant* -> t* -\
            #               x -> quant -> dequant* -> addmm -> quant -> dequant -> out
            # note: nodes with `*` should be fused in delegation
            FileCheck().check_count(
                "executorch_exir_dialects_edge__ops_quantized_decomposed_dequantize_per_tensor_default",
                3,
                exactly=True,
            ).run(captured_mod.exported_program.graph_module.code)

            FileCheck().check_count(
                "executorch_exir_dialects_edge__ops_quantized_decomposed_quantize_per_tensor_default",
                3,
                exactly=True,
            ).run(captured_mod.exported_program.graph_module.code)
