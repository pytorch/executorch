# Copyright 2025 Arm Limited and/or its affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import torch
from executorch.backends.arm.quantizer import (
    get_symmetric_quantization_config,
    TOSAQuantizer,
)
from executorch.backends.arm.test.tester.test_pipeline import TosaPipelineINT
from executorch.backends.arm.tosa import TosaSpecification
from executorch.backends.xnnpack.test.tester import Quantize


class AddSigmoidMul(torch.nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.sigmoid = torch.nn.Sigmoid()

    def forward(self, x, y):
        return self.sigmoid(x + y) * x


def get_selective_quantizer(modules):
    quantizer = TOSAQuantizer(TosaSpecification.create_from_string("TOSA-1.0+INT"))
    quantizer.set_global(get_symmetric_quantization_config())
    for module in modules:
        quantizer.set_module_type(module, None)

    return Quantize(quantizer, get_symmetric_quantization_config())


def test_qdq_squeezed_fp_op():
    """Test that a float operation surrounded by quantize-dequantize pairs
    is correctly handled by the partitioner and the TOSA backend.
    Pattern:
    q -> dq -> add -> q -> dq -> sigmoid -> q -> dq -> mul -> dq -> q
                        |_____Non-delegated____|
    """
    aten_op = "torch.ops.aten.add.Tensor"
    exir_op = "executorch_exir_dialects_edge__ops_aten_add_Tensor"
    module = AddSigmoidMul()
    x = torch.randn(2, 3, 4)
    y = torch.randn(2, 3, 4)
    pipeline = TosaPipelineINT(
        module=module, test_data=(x, y), aten_op=aten_op, exir_op=exir_op
    )
    pipeline.change_args("quantize", get_selective_quantizer([torch.nn.Sigmoid]))
    pipeline.change_args(
        "check_count.exir",
        {
            "torch.ops.higher_order.executorch_call_delegate": 2,
            "executorch_exir_dialects_edge__ops_aten_sigmoid_default": 1,
            "executorch_exir_dialects_edge__ops_quantized_decomposed_dequantize_per_tensor_default": 2,
            "executorch_exir_dialects_edge__ops_quantized_decomposed_quantize_per_tensor_default": 3,
        },
    )
    pipeline.run()


class MulAddSigmoidConv(torch.nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.sigmoid = torch.nn.Sigmoid()
        self.conv = torch.nn.Conv1d(3, 3, 1)

    def forward(self, x, y):
        return self.conv(self.sigmoid(x + y * x))


def test_quantized_to_float_transition():
    """Test that a model executing quantized ops followed by float ops
    is correctly handled by the partitioner and the TOSA backend.
    Pattern:
    q -> dq -> mul -> q -> dq -> add -> q -> dq -> sigmoid -> conv
                                           |____Non-delegated___|
    """
    aten_op = "torch.ops.aten.add.Tensor"
    exir_op = "executorch_exir_dialects_edge__ops_aten_add_Tensor"
    module = MulAddSigmoidConv()
    x = torch.randn(2, 3, 4)
    y = torch.randn(2, 3, 4)
    pipeline = TosaPipelineINT(
        module=module, test_data=(x, y), aten_op=aten_op, exir_op=exir_op
    )
    pipeline.change_args(
        "quantize", get_selective_quantizer([torch.nn.Sigmoid, torch.nn.Conv1d])
    )
    pipeline.change_args(
        "check_count.exir",
        {
            "torch.ops.higher_order.executorch_call_delegate": 1,
            "executorch_exir_dialects_edge__ops_aten_sigmoid_default": 1,
            "executorch_exir_dialects_edge__ops_aten_convolution_default": 1,
            "executorch_exir_dialects_edge__ops_quantized_decomposed_dequantize_per_tensor_default": 1,
            "executorch_exir_dialects_edge__ops_quantized_decomposed_quantize_per_tensor_default": 2,
        },
    )
    pipeline.run()
