# (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

# pyre-strict

import unittest
from dataclasses import dataclass
from typing import Optional

import executorch
import executorch.backends.cortex_m.ops.operators  # noqa

import torch
from executorch.backends.cortex_m.passes.replace_quant_nodes_pass import (
    ReplaceQuantNodesPass,
)
from executorch.exir.dialects._ops import ops as exir_ops
from torch.ao.quantization.observer import HistogramObserver
from torch.ao.quantization.quantize_pt2e import convert_pt2e, prepare_pt2e
from torch.ao.quantization.quantizer.quantizer import (
    QuantizationAnnotation,
    QuantizationSpec,
    Quantizer,
)
from torch.export import export, export_for_training
from torch.fx import GraphModule, Node


@dataclass(eq=True, frozen=True)
class QuantizationConfig:
    input_activation: Optional[QuantizationSpec]
    output_activation: Optional[QuantizationSpec]


class AddQuantizer(Quantizer):
    def __init__(self):
        super().__init__()

    @staticmethod
    def _get_qspec():
        return QuantizationSpec(
            dtype=torch.int8,
            quant_min=-128,
            quant_max=127,
            qscheme=torch.per_tensor_symmetric,
            is_dynamic=False,
            observer_or_fake_quant_ctr=HistogramObserver.with_args(eps=2**-12),
        )

    @staticmethod
    def _get_qconfig():
        return QuantizationConfig(
            input_activation=AddQuantizer._get_qspec(),
            output_activation=AddQuantizer._get_qspec(),
        )

    @staticmethod
    def _is_annotated(nodes: list[Node]):
        annotated = False
        for node in nodes:
            annotated = annotated or (
                "quantization_annotation" in node.meta
                and node.meta["quantization_annotation"]._annotated
            )
        return annotated

    def annotate(self, model: GraphModule) -> torch.fx.GraphModule:
        config = self._get_qconfig()
        annotated_partitions = []
        for node in model.graph.nodes:
            if node.op != "call_function" or node.target not in [
                torch.ops.aten.add.Tensor,
                torch.ops.aten.add_.Tensor,
            ]:
                continue

            if self._is_annotated([node]):
                continue

            input_qspec_map = {
                node.args[0]: config.input_activation,
                node.args[1]: config.input_activation,
            }
            output_qspec = config.output_activation

            node.meta["quantization_annotation"] = QuantizationAnnotation(
                input_qspec_map=input_qspec_map,
                output_qspec=output_qspec,
                _annotated=True,
            )
            annotated_partitions.append([node])
        return annotated_partitions

    def validate(self, model: GraphModule) -> None:
        pass


def check_count(
    graph_module: GraphModule, op: torch.fx.node.Target, expected_count: int
):
    """
    Check that the graph module contains exactly the expected number of nodes with the given op.
    """
    actual_count = 0
    for node in graph_module.graph.nodes:
        if node.op == "call_function" and node.target == op:
            actual_count += 1
    assert (
        actual_count == expected_count
    ), f"Expected {expected_count} {op} nodes, got {actual_count}"


class TestReplaceQuantOps(unittest.TestCase):
    """
    Test suite for the ReplaceQuantNodesPass which replaces quantized_decomposed quant/dequant ops
    with cortex_m specific implementations.
    """

    def test_replace_quant_ops(self):
        """
        Test that quantize_per_tensor and dequantize_per_tensor nodes are correctly replaced
        with their cortex_m equivalents while preserving the same functionality.
        """

        class M(torch.nn.Module):
            def __init__(self):
                super().__init__()

            def forward(self, x):
                return x + x

        m = M()
        example_inputs = (torch.randn(10, 11, 12),)

        # quantize
        captured_graph_module = export_for_training(
            m.eval(), example_inputs, strict=True
        ).module()
        quantizer = AddQuantizer()
        prepared_graph_module = prepare_pt2e(captured_graph_module, quantizer)
        converted_graph_module = convert_pt2e(prepared_graph_module)

        # export
        exported = export(converted_graph_module, example_inputs, strict=True)

        # to edge
        epm = executorch.exir.to_edge(
            exported,
            compile_config=executorch.exir.EdgeCompileConfig(_check_ir_validity=False),
        )
        graph_module = epm.exported_program().graph_module

        quant_count_before = 0
        dequant_count_before = 0
        for node in graph_module.graph.nodes:
            if node.op == "call_function":
                if (
                    node.target
                    == exir_ops.edge.quantized_decomposed.quantize_per_tensor.default
                ):
                    quant_count_before += 1
                elif (
                    node.target
                    == exir_ops.edge.quantized_decomposed.dequantize_per_tensor.default
                ):
                    dequant_count_before += 1
        edge_output = graph_module(*example_inputs)

        # to cortex_m
        epm = epm.transform(
            [
                ReplaceQuantNodesPass(),
            ]
        )
        graph_module = epm.exported_program().graph_module
        check_count(
            graph_module,
            exir_ops.edge.cortex_m.quantize_per_tensor.default,
            quant_count_before,
        )
        check_count(
            graph_module,
            exir_ops.edge.cortex_m.dequantize_per_tensor.default,
            dequant_count_before,
        )
        check_count(
            graph_module,
            exir_ops.edge.quantized_decomposed.quantize_per_tensor.default,
            0,
        )
        check_count(
            graph_module,
            exir_ops.edge.quantized_decomposed.dequantize_per_tensor.default,
            0,
        )
        cortex_m_output = graph_module(*example_inputs)

        # check output - numerical equivalence should be preserved
        torch.testing.assert_close(edge_output, cortex_m_output)

        # To executorch
        expm = epm.to_executorch()
        for op in expm.executorch_program.execution_plan[0].operators:
            if "quantize_per_tensor" in op.name:
                assert op.name in [
                    "cortex_m::quantize_per_tensor",
                    "cortex_m::dequantize_per_tensor",
                ], f"Unexpected op {op.name}"
