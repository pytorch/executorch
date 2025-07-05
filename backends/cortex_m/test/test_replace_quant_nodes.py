# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

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
from torch.export import export, export_for_training
from torch.fx import GraphModule
from torchao.quantization.pt2e.observer import HistogramObserver
from torchao.quantization.pt2e.quantize_pt2e import convert_pt2e, prepare_pt2e
from torchao.quantization.pt2e.quantizer import (
    QuantizationAnnotation,
    QuantizationSpec,
    Quantizer,
)


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
        qspec = AddQuantizer._get_qspec()
        return QuantizationConfig(
            input_activation=qspec,
            output_activation=qspec,
        )

    def annotate(self, model: GraphModule):
        config = self._get_qconfig()
        annotated_partitions = []

        for node in model.graph.nodes:
            if node.op != "call_function" or node.target not in [
                torch.ops.aten.add.Tensor,
                torch.ops.aten.add_.Tensor,
            ]:
                continue

            if (
                "quantization_annotation" in node.meta
                and node.meta["quantization_annotation"]._annotated
            ):
                continue

            input_qspec_map = {
                node.args[0]: config.input_activation,
                node.args[1]: config.input_activation,
            }

            node.meta["quantization_annotation"] = QuantizationAnnotation(
                input_qspec_map=input_qspec_map,
                output_qspec=config.output_activation,
                _annotated=True,
            )
            annotated_partitions.append([node])

        return annotated_partitions

    def validate(self, model: GraphModule) -> None:
        pass


def check_count(
    graph_module: GraphModule, op: torch.fx.node.Target, expected_count: int
):
    actual_count = sum(
        1
        for node in graph_module.graph.nodes
        if node.op == "call_function" and node.target == op
    )

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

        # Define a simple model that can be quantized
        class SimpleAddModel(torch.nn.Module):
            def forward(self, x):
                return x + x

        # Setup model and inputs
        model = SimpleAddModel()
        example_inputs = (torch.randn(10, 11, 12),)

        # Step 1: Export and quantize the model
        exported_model = export_for_training(
            model.eval(), example_inputs, strict=True
        ).module()
        prepared_model = prepare_pt2e(exported_model, AddQuantizer())
        quantized_model = convert_pt2e(prepared_model)

        # Step 2: Export to EXIR
        exported = export(quantized_model, example_inputs, strict=True)

        # Step 3: Convert to Edge
        edge_program = executorch.exir.to_edge(
            exported,
            compile_config=executorch.exir.EdgeCompileConfig(_check_ir_validity=False),
        )
        edge_graph = edge_program.exported_program().graph_module

        # Count quantization ops before replacement
        quant_count = 0
        dequant_count = 0
        for node in edge_graph.graph.nodes:
            if node.op == "call_function":
                if (
                    node.target
                    == exir_ops.edge.quantized_decomposed.quantize_per_tensor.default
                ):
                    quant_count += 1
                elif (
                    node.target
                    == exir_ops.edge.quantized_decomposed.dequantize_per_tensor.default
                ):
                    dequant_count += 1

        # Get output before transformation
        edge_output = edge_graph(*example_inputs)

        # Step 4: Apply ReplaceQuantNodesPass
        transformed_program = edge_program.transform([ReplaceQuantNodesPass()])
        transformed_graph = transformed_program.exported_program().graph_module

        # Step 5: Verify the transformation
        # Check that quantized_decomposed ops were replaced with cortex_m ops
        check_count(
            transformed_graph,
            exir_ops.edge.cortex_m.quantize_per_tensor.default,
            quant_count,
        )
        check_count(
            transformed_graph,
            exir_ops.edge.cortex_m.dequantize_per_tensor.default,
            dequant_count,
        )
        check_count(
            transformed_graph,
            exir_ops.edge.quantized_decomposed.quantize_per_tensor.default,
            0,
        )
        check_count(
            transformed_graph,
            exir_ops.edge.quantized_decomposed.dequantize_per_tensor.default,
            0,
        )

        # Step 6: Verify numerical equivalence
        transformed_output = transformed_graph(*example_inputs)
        torch.testing.assert_close(edge_output, transformed_output)

        # Step 7: Verify ExecuTorch program has the correct ops
        executorch_program = transformed_program.to_executorch()
        for op in executorch_program.executorch_program.execution_plan[0].operators:
            if "quantize_per_tensor" in op.name:
                assert op.name in [
                    "cortex_m::quantize_per_tensor",
                    "cortex_m::dequantize_per_tensor",
                ], f"Unexpected op {op.name}"
