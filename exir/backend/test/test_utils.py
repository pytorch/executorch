# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import unittest

import torch
from executorch import exir
from executorch.exir import CaptureConfig
from executorch.exir.backend.backend_api import to_backend
from executorch.exir.backend.partitioner import Partitioner, PartitionResult
from executorch.exir.backend.test.op_partitioner_demo import AddMulPartitionerDemo
from executorch.exir.backend.utils import (
    get_non_lowered_nodes,
    is_identical_graph,
    remove_first_quant_and_last_dequant,
    replace_quantized_partition_with_op,
)

from executorch.exir.dialects._ops import bind_pattern_to_op, ops as exir_ops
from torch.ao.quantization import get_default_qconfig  # @manual
from torch.ao.quantization.backend_config.executorch import (
    get_executorch_backend_config,
)
from torch.ao.quantization.qconfig_mapping import (
    _get_symmetric_qnnpack_qconfig_mapping,
    QConfigMapping,
)
from torch.ao.quantization.quantize_fx import (
    _convert_to_reference_decomposed_fx,
    prepare_fx,
)
from torch.export import ExportedProgram
from torch.fx import symbolic_trace
from torch.fx.passes.utils.fuser_utils import legalize_graph
from torch.fx.passes.utils.matcher_utils import SubgraphMatcher
from torch.fx.passes.utils.source_matcher_utils import get_source_partitions
from torch.library import Library
from torch.testing import FileCheck

T_QuantPerTensor = exir_ops.edge.quantized_decomposed.quantize_per_tensor.default
T_DQuantPerTensor = exir_ops.edge.quantized_decomposed.dequantize_per_tensor.default


class TestPartitioners(unittest.TestCase):
    def test_identical_graph_with_unused_args(self):
        class MyModule(torch.nn.Module):
            def __init__(self):
                super().__init__()

            def forward(self, x, y):
                # y is not used arg
                return x

        m = MyModule()
        graph_module: torch.fx.GraphModule = symbolic_trace(m)
        is_matched = is_identical_graph(graph_module, graph_module)
        self.assertTrue(is_matched)

    def test_identical_graph_with_used_args(self):
        class MyModule(torch.nn.Module):
            def __init__(self):
                super().__init__()

            def forward(self, x, y):
                return x, y

        m = MyModule()
        graph_module: torch.fx.GraphModule = symbolic_trace(m)
        is_matched = is_identical_graph(graph_module, graph_module)
        self.assertTrue(is_matched)

    def test_identical_graph_for_linear(self):
        graph_module: torch.fx.GraphModule = symbolic_trace(torch.nn.Linear(10, 10))
        is_matched = is_identical_graph(graph_module, graph_module)
        self.assertTrue(is_matched)

    def test_identical_graph_for_composite_module(self):
        class MyModule(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.param = torch.nn.Parameter(torch.rand(3, 4))
                self.linear = torch.nn.Linear(4, 5)

            def forward(self, x):
                return self.linear(x + self.param).clamp(min=0.0, max=1.0)

        graph_module: torch.fx.GraphModule = symbolic_trace(MyModule())
        is_matched = is_identical_graph(graph_module, graph_module)
        self.assertTrue(is_matched)

    def test_not_identical_graph_for_args(self):
        class MyModule1(torch.nn.Module):
            def __init__(self):
                super().__init__()

            def forward(self, x, y):
                # y is not used arg
                return x + 1

        class MyModule2(torch.nn.Module):
            def __init__(self):
                super().__init__()

            def forward(self, x, y):
                return x + 1, y + 2

        graph_module_1: torch.fx.GraphModule = (
            exir.capture(
                MyModule1(),
                (torch.rand(3, 4), torch.rand(3, 4)),
                CaptureConfig(),
            )
            .to_edge()
            .exported_program.graph_module
        )
        graph_module_2: torch.fx.GraphModule = (
            exir.capture(
                MyModule2(),
                (torch.rand(3, 4), torch.rand(3, 4)),
                CaptureConfig(),
            )
            .to_edge()
            .exported_program.graph_module
        )
        is_matched = is_identical_graph(graph_module_1, graph_module_2)
        self.assertFalse(is_matched)

    def test_match_attrs(self):
        class LargeModel(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.weght = torch.nn.Parameter(torch.ones(3, 3))
                self.linear = torch.nn.Linear(3, 3)

            def forward(self, x):
                a = x + self.weght
                b = self.linear(x)
                return a, b

        inputs = (torch.ones(3, 3),)

        # Large model graph:
        # opcode         name               target              args                                         kwargs
        # -------------  -----------------  ------------------  -------------------------------------------  --------
        # placeholder    ph_0               ph_0                ()                                           {}
        # get_attr       _param_constant0   _param_constant0    ()                                           {}
        # call_function  add_tensor         aten.add.Tensor     (ph_0, _param_constant0)                     {}
        # get_attr       _param_constant1   _param_constant1    ()                                           {}
        # get_attr       _tensor_constant0  _tensor_constant0   ()                                           {}
        # call_function  addmm_default      aten.addmm.default  (_param_constant1, ph_0, _tensor_constant0)  {}
        # output         output             output              ([add_tensor, addmm_default],)               {}

        large_model = (
            exir.capture(
                LargeModel(),
                inputs,
                CaptureConfig(),
            )
            .to_edge(exir.EdgeCompileConfig(_check_ir_validity=False))
            .exported_program.graph_module
        )

        # Pattern graph:
        # opcode         name               target              args                                         kwargs
        # -------------  -----------------  ------------------  -------------------------------------------  --------
        # placeholder    ph_0               ph_0                ()                                           {}
        # get_attr       _param_constant0   _param_constant0    ()                                           {}
        # get_attr       _tensor_constant0  _tensor_constant0   ()                                           {}
        # call_function  addmm_default      aten.addmm.default  (_param_constant0, ph_0, _tensor_constant0)  {}
        # output         output             output              ([addmm_default],)                           {}

        pattern = (
            exir.capture(torch.nn.Linear(3, 3), inputs, CaptureConfig())
            .to_edge(exir.EdgeCompileConfig(_check_ir_validity=False))
            .exported_program.graph_module.graph
        )

        subgraph_matcher = SubgraphMatcher(pattern)
        match_result = subgraph_matcher.match(large_model.graph)

        # Should find exact one match
        self.assertEqual(len(match_result), 1)

    def test_remove_first_quant_and_last_dequant(self):
        qconfig_mapping = _get_symmetric_qnnpack_qconfig_mapping()
        linear = torch.nn.Linear(3, 4).eval()

        example_inputs = (torch.ones(1, 1, 3, dtype=torch.float),)
        prepared_linear = prepare_fx(
            linear,
            qconfig_mapping,
            example_inputs,
            backend_config=get_executorch_backend_config(),
        )

        converted_linear: torch.fx.GraphModule = _convert_to_reference_decomposed_fx(
            prepared_linear,
        )

        actual_static_quant_linear = (
            exir.capture(
                converted_linear,
                example_inputs,
                CaptureConfig(
                    enable_functionalization=False,
                ),
            )
            .to_edge(
                exir.EdgeCompileConfig(
                    _check_ir_validity=False,
                )
            )
            .exported_program.graph_module
        )

        # Original graph has exactly 3 dequantize ops and 3 quantize ops
        FileCheck().check_count(
            "executorch_exir_dialects_edge__ops_quantized_decomposed_dequantize_per_tensor_default",
            3,
            exactly=True,
        ).run(actual_static_quant_linear.code)
        FileCheck().check_count(
            "executorch_exir_dialects_edge__ops_quantized_decomposed_quantize_per_tensor_default",
            3,
            exactly=True,
        ).run(actual_static_quant_linear.code)

        # Remove first and last dequant in static quant
        remove_first_quant_and_last_dequant(actual_static_quant_linear)

        # Original graph has exactly 2 dequantize ops and 2 quantize ops
        FileCheck().check_count(
            "executorch_exir_dialects_edge__ops_quantized_decomposed_dequantize_per_tensor_default",
            2,
            exactly=True,
        ).run(actual_static_quant_linear.code)
        FileCheck().check_count(
            "executorch_exir_dialects_edge__ops_quantized_decomposed_quantize_per_tensor_default",
            2,
            exactly=True,
        ).run(actual_static_quant_linear.code)

    def test_invalid_partitioner_without_partitioner(self):
        """
        Tests replacing literals with placeholders in the case there are
        `getitem` calls which do not have a schema.
        """

        class InvalidPartitioner(Partitioner):
            """
            Partitions all add/mul nodes regardless of order
            """

            def __init__(self) -> None:
                # A valid partitioner should have partition_tags
                self.test = "a"

            def partition(
                self, edge_exported_program: ExportedProgram
            ) -> PartitionResult:
                return PartitionResult(
                    tagged_exported_program=edge_exported_program, partition_tags=None
                )

        exported_program = exir.capture(
            torch.nn.Linear(3, 3),
            (torch.randn(3, 3),),
            CaptureConfig(),
        ).to_edge(
            exir.EdgeCompileConfig(
                _check_ir_validity=False,
            )
        )

        error_msg = r"Partitioner <class 'executorch.exir.backend.test.test_utils.TestPartitioners.test_invalid_partitioner_without_partitioner.<locals>.InvalidPartitioner'> needs a `partition_tags` field containing a mapping of tags to delegate spec"
        with self.assertRaisesRegex(
            AssertionError,
            error_msg,
        ):
            _ = to_backend(exported_program.exported_program, InvalidPartitioner)

    test_lib = Library("test_lib", "DEF")

    @staticmethod
    @bind_pattern_to_op(
        test_lib, "test_q_linear(Tensor x, Tensor weight, Tensor bias) -> Tensor"
    )
    def q_linear(x, weight, bias):
        return x

    def test_replace_quantized_partition_with_op(self):
        class LinearModel(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.linear = torch.nn.Linear(3, 4)

            def forward(self, input):
                return self.linear(input)

        linear_model = LinearModel()
        example_inputs = (torch.ones(1, 1, 3, dtype=torch.float),)
        prepared_linear = prepare_fx(
            linear_model,
            QConfigMapping().set_object_type(
                torch.nn.Linear,
                get_default_qconfig("qnnpack"),
            ),
            example_inputs,
            backend_config=get_executorch_backend_config(),
        )

        converted_linear: torch.fx.GraphModule = _convert_to_reference_decomposed_fx(
            prepared_linear,
        )

        actual_static_quant_linear = (
            exir.capture(
                converted_linear,
                example_inputs,
                CaptureConfig(
                    enable_functionalization=False,
                ),
            )
            .to_edge(
                exir.EdgeCompileConfig(
                    _check_ir_validity=False,
                ),
            )
            .exported_program.graph_module
        )

        source_partitions_by_module = get_source_partitions(
            actual_static_quant_linear.graph,
            [torch.ao.nn.quantized.reference.modules.linear.Linear],
        )

        replace_quantized_partition_with_op(
            actual_static_quant_linear,
            list(source_partitions_by_module.values())[0][0],
            torch.ops.test_lib.test_q_linear,
        )

        legalize_graph(actual_static_quant_linear)

        FileCheck().check_count(
            "executorch_exir_dialects_edge__ops_quantized_decomposed_dequantize_per_tensor_default",
            1,
            exactly=True,
        ).run(actual_static_quant_linear.code)
        FileCheck().check_count(
            "executorch_exir_dialects_edge__ops_quantized_decomposed_quantize_per_tensor_default",
            1,
            exactly=True,
        ).run(actual_static_quant_linear.code)
        FileCheck().check_count("test_lib.test_q_linear", 1, exactly=True).run(
            actual_static_quant_linear.code
        )

    def test_get_non_lowered_nodes(self):
        class Model(torch.nn.Module):
            def __init__(self):
                super().__init__()

            def forward(self, a, x, b):
                y = torch.mm(a, x)
                z = y + b
                a = z - a
                y = torch.mm(a, x)
                z = y + b
                return z

        m = Model()
        inputs = (torch.randn(2, 2), torch.randn(2, 2), torch.randn(2, 2))
        edge = exir.capture(m, inputs, exir.CaptureConfig()).to_edge()
        edge.exported_program = to_backend(edge.exported_program, AddMulPartitionerDemo)
        edge.dump()
        number_of_cpu_nodes = get_non_lowered_nodes(edge.exported_program.graph)
        # Only sub is not not lowerable
        self.assertEqual(len(number_of_cpu_nodes), 1)
