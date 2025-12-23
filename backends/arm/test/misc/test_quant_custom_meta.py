# Copyright 2025 Arm Limited and/or its affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import pytest
import torch
from executorch.backends.arm.test.tester.test_pipeline import TosaPipelineINT


class AddSigmoidMul(torch.nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.sigmoid = torch.nn.Sigmoid()

    def forward(self, x, y):
        return self.sigmoid(x + y) * x


@pytest.mark.parametrize("fp_extension", [True, False])
def test_qdq_squeezed_fp_op_tosa_INT_FP(fp_extension: bool):
    """Test that a float operation surrounded by quantize-dequantize pairs
    is correctly handled by the partitioner and the TOSA backend.
    Pattern:
    q -> dq -> add -> q -> dq -> sigmoid -> q -> dq -> mul -> dq -> q
                        |_____unquantized_____|
    """
    aten_op = "torch.ops.aten.add.Tensor"
    exir_op = "executorch_exir_dialects_edge__ops_aten_add_Tensor"
    module = AddSigmoidMul()
    x = torch.randn(2, 3, 4)
    y = torch.randn(2, 3, 4)
    pipeline = TosaPipelineINT(
        module=module,
        test_data=(x, y),
        aten_op=aten_op,
        exir_op=exir_op,
        tosa_extensions=["FP"] if fp_extension else None,
    )
    pipeline.quantizer.set_module_type(torch.nn.Sigmoid, None)  # type: ignore

    if not fp_extension:
        # In case we don't have the FP extension, the unquantized part of the
        # graph should not be delegated to the Arm backend. Modify the op count
        # checks to reflect this behavior.
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


@pytest.mark.parametrize("fp_extension", [True, False])
def test_quantized_to_float_transition_tosa_INT_FP(fp_extension: bool):
    """Test that a model executing quantized ops followed by float ops
    is correctly handled by the partitioner and the TOSA backend.
    Pattern:
    q -> dq -> mul -> q -> dq -> add -> q -> dq -> sigmoid -> conv
                                           |___unquantized___|
    """
    aten_op = "torch.ops.aten.add.Tensor"
    exir_op = "executorch_exir_dialects_edge__ops_aten_add_Tensor"
    module = MulAddSigmoidConv()
    x = torch.randn(2, 3, 4)
    y = torch.randn(2, 3, 4)
    pipeline = TosaPipelineINT(
        module=module,
        test_data=(x, y),
        aten_op=aten_op,
        exir_op=exir_op,
        tosa_extensions=["FP"] if fp_extension else None,
    )
    if not fp_extension:
        # In case we don't have the FP extension, the unquantized part of the
        # graph should not be delegated to the Arm backend. Modify the op count
        # checks to reflect this behavior.
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
    pipeline.quantizer.set_module_type(torch.nn.Sigmoid, None)  # type: ignore
    pipeline.quantizer.set_module_type(torch.nn.Conv1d, None)  # type: ignore

    pipeline.run()
