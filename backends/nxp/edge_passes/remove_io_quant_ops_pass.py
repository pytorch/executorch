# Copyright 2025 NXP
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import torch

from executorch.exir import EdgeProgramManager
from executorch.exir.dialects._ops import ops as exir_ops
from executorch.exir.pass_base import ExportPass
from executorch.exir.passes.quantize_io_pass import QuantizeInputs, QuantizeOutputs
from torch.fx.passes.infra.pass_base import PassResult


class RemoveIOQuantOpsPass(ExportPass):

    def __init__(self, edge_program_manager: EdgeProgramManager):
        super().__init__()
        self._edge_program_manager = edge_program_manager

    def _get_quantizable_input_indices(self):
        exported_program = self._edge_program_manager.exported_program()

        graph = exported_program.graph_module.graph
        user_inputs = exported_program.graph_signature.user_inputs

        inputs_to_quantization = []

        for input_index, user_input in enumerate(user_inputs):
            placeholders = [
                n for n in graph.nodes if n.op == "placeholder" and n.name == user_input
            ]
            assert placeholders
            target_placeholder = placeholders[0]

            if len(target_placeholder.users) != 1:
                raise ValueError(f"Input {input_index} has more than one users")

            quantize = next(iter(target_placeholder.users))
            if (
                quantize.target
                != exir_ops.edge.quantized_decomposed.quantize_per_tensor.default
            ):
                continue

            inputs_to_quantization.append(input_index)

        return inputs_to_quantization

    def _get_quantizable_output_indices(self):
        exported_program = self._edge_program_manager.exported_program()

        graph = exported_program.graph_module.graph
        outputs = [n for n in graph.nodes if n.op == "output"]
        if len(outputs) != 1:
            raise NotImplementedError("Only 1 output node is supported.")

        outputs_to_quantization = []

        user_outputs = list(outputs[0].args[0])
        for output_index, user_output in enumerate(user_outputs):
            if (
                user_output.target
                != exir_ops.edge.quantized_decomposed.dequantize_per_tensor.default
            ):
                continue

            outputs_to_quantization.append(output_index)

        return outputs_to_quantization

    def call(self, graph_module: torch.fx.GraphModule):
        input_indices = self._get_quantizable_input_indices()
        output_indices = self._get_quantizable_output_indices()

        QuantizeInputs(self._edge_program_manager, input_indices).call(graph_module)
        QuantizeOutputs(self._edge_program_manager, output_indices).call(graph_module)

        return PassResult(graph_module, True)
