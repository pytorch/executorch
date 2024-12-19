# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import copy
import unittest

import torch
from executorch.exir import EdgeCompileConfig, to_edge_transform_and_lower
from executorch.exir.passes.quantize_io_pass import (
    get_config_method_name,
    QuantizeInputs,
    QuantizeOutputs,
)
from executorch.exir.tensor import get_scalar_type
from torch.ao.quantization.quantize_pt2e import convert_pt2e, prepare_pt2e

from torch.ao.quantization.quantizer.xnnpack_quantizer import (
    get_symmetric_quantization_config,
    XNNPACKQuantizer,
)
from torch.testing import FileCheck

op_str = {
    "q": "executorch_exir_dialects_edge__ops_quantized_decomposed_quantize_per_tensor_default",
    "dq": "executorch_exir_dialects_edge__ops_quantized_decomposed_dequantize_per_tensor_default",
}


class TestQuantIOPass(unittest.TestCase):
    class Add(torch.nn.Module):
        def forward(self, x, y):
            return x + y

    def _quantize(self, mod, example_inputs):
        quantizer = XNNPACKQuantizer()
        operator_config = get_symmetric_quantization_config()
        quantizer.set_global(operator_config)
        m = torch.export.export_for_training(
            mod, copy.deepcopy(example_inputs)
        ).module()
        m = prepare_pt2e(m, quantizer)
        _ = m(*example_inputs)
        m = convert_pt2e(m)
        exported_program = torch.export.export_for_training(m, example_inputs)
        return exported_program

    def _check_count(self, op, count, epm):
        code = epm.exported_program().graph_module.code
        FileCheck().check_count(op, count, exactly=True).run(code)

    def _get_edge_prog_manager(self, mod, example_inputs):
        exported_program = self._quantize(mod, example_inputs)
        edge_program_manager = to_edge_transform_and_lower(
            exported_program,
            transform_passes=[],
            partitioner=None,
            compile_config=EdgeCompileConfig(_check_ir_validity=False),
        )

        self._check_count(op_str["dq"], 3, edge_program_manager)
        self._check_count(op_str["q"], 3, edge_program_manager)
        return edge_program_manager

    def test_add_drop_q_inputs(self) -> None:
        example_inputs = (torch.randn(1, 5), torch.randn(1, 5))
        mod = self.Add().eval()
        edge_program_manager = self._get_edge_prog_manager(mod, example_inputs)
        reference_outputs = edge_program_manager.exported_program().module()(
            *example_inputs
        )

        edge_program_manager_qin = edge_program_manager.transform(
            [
                QuantizeInputs(
                    edge_program_manager=edge_program_manager,
                    quantized_inputs_idx=[0, 1],
                    method_name="forward",
                )
            ]
        )
        self._check_count(op_str["dq"], 3, edge_program_manager)
        self._check_count(op_str["q"], 1, edge_program_manager)

        quantized_example_inputs = []
        for i in range(len(example_inputs)):
            d = edge_program_manager_qin._config_methods
            scale = d[get_config_method_name("forward", "input", i, "scale")]
            zp = d[get_config_method_name("forward", "input", i, "zp")]
            quant_min = d[get_config_method_name("forward", "input", i, "quant_min")]
            quant_max = d[get_config_method_name("forward", "input", i, "quant_max")]
            dtype = get_scalar_type(
                d[get_config_method_name("forward", "input", i, "dtype")]
            )

            quantized_example_inputs.append(
                torch.ops.quantized_decomposed.quantize_per_tensor.default(
                    example_inputs[i], scale, zp, quant_min, quant_max, dtype
                ),
            )
        quantized_example_inputs = tuple(quantized_example_inputs)
        output = edge_program_manager_qin.exported_program().module()(
            *quantized_example_inputs
        )
        torch.testing.assert_close(
            reference_outputs[0],
            output[0],
        )

    def test_add_drop_dq_output(self) -> None:
        example_inputs = (torch.randn(1, 5), torch.randn(1, 5))
        mod = self.Add().eval()
        edge_program_manager = self._get_edge_prog_manager(mod, example_inputs)
        reference_outputs = edge_program_manager.exported_program().module()(
            *example_inputs
        )

        edge_program_manager_dqout = edge_program_manager.transform(
            [
                QuantizeOutputs(
                    edge_program_manager=edge_program_manager,
                    quantized_outputs_idx_list=[0],
                    method_name="forward",
                )
            ]
        )
        self._check_count(op_str["dq"], 2, edge_program_manager)
        self._check_count(op_str["q"], 3, edge_program_manager)

        quantized_outputs = edge_program_manager_dqout.exported_program().module()(
            *example_inputs
        )

        dequantized_outputs = []
        for i in range(len(quantized_outputs)):
            d = edge_program_manager_dqout._config_methods
            scale = d[get_config_method_name("forward", "output", i, "scale")]
            zp = d[get_config_method_name("forward", "output", i, "zp")]
            q_min = d[get_config_method_name("forward", "output", i, "quant_min")]
            q_max = d[get_config_method_name("forward", "output", i, "quant_max")]
            dtype = get_scalar_type(
                d[get_config_method_name("forward", "output", i, "dtype")]
            )
            dequantized_outputs.append(
                torch.ops.quantized_decomposed.dequantize_per_tensor.default(
                    quantized_outputs[i], scale, zp, q_min, q_max, dtype
                )
            )
        dequantized_outputs = tuple(dequantized_outputs)

        torch.testing.assert_close(
            reference_outputs[0],
            dequantized_outputs[0],
        )
