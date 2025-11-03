# Copyright 2025 Arm Limited and/or its affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from typing import Tuple

import torch
from executorch.backends.arm._passes.decompose_softmax_pass import DecomposeSoftmaxPass

from executorch.backends.arm.test.tester.test_pipeline import PassPipeline

input_t = Tuple[torch.Tensor]  # Input x


class Softmax(torch.nn.Module):
    """
    Basic torch.nn.softmax layer model
    """

    def __init__(self):
        super(Softmax, self).__init__()
        self.softmax = torch.nn.Softmax(dim=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.softmax(x)
        return x

    def get_inputs(self) -> input_t:
        return (torch.rand(2, 3),)


class SoftmaxLog(torch.nn.Module):
    """
    Basic torch.nn.log_softmax layer model
    """

    def __init__(self):
        super(SoftmaxLog, self).__init__()
        self.softmax = torch.nn.LogSoftmax(dim=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.softmax(x)
        return x

    def get_inputs(self) -> input_t:
        return (torch.rand(2, 3),)


def test_softmax_basic_tosa_FP():
    module = Softmax()
    pipeline = PassPipeline[input_t](
        module,
        module.get_inputs(),
        quantize=False,
        ops_before_pass={
            "executorch_exir_dialects_edge__ops_aten__softmax_default": 1,
        },
        ops_not_before_pass=[
            "executorch_exir_dialects_edge__ops_aten_mul_Tensor",
            "executorch_exir_dialects_edge__ops_aten_reciprocal_default",
            "executorch_exir_dialects_edge__ops_aten_sum_dim_IntList",
            "executorch_exir_dialects_edge__ops_aten_exp_default",
        ],
        ops_after_pass={
            "executorch_exir_dialects_edge__ops_aten_mul_Tensor": 1,
            "executorch_exir_dialects_edge__ops_aten_exp_default": 1,
            "executorch_exir_dialects_edge__ops_aten_reciprocal_default": 1,
            "executorch_exir_dialects_edge__ops_aten_sum_dim_IntList": 1,
        },
        ops_not_after_pass=["executorch_exir_dialects_edge__ops_aten__softmax_default"],
        pass_list=[DecomposeSoftmaxPass],
    )
    pipeline.run()


def test_softmax_log_tosa_FP():
    module = SoftmaxLog()
    pipeline = PassPipeline[input_t](
        module,
        module.get_inputs(),
        quantize=False,
        ops_before_pass={
            "executorch_exir_dialects_edge__ops_aten__log_softmax_default": 1,
        },
        ops_not_before_pass=[
            "executorch_exir_dialects_edge__ops_aten_mul_Tensor",
            "executorch_exir_dialects_edge__ops_aten_reciprocal_default",
            "executorch_exir_dialects_edge__ops_aten_sum_dim_IntList",
            "executorch_exir_dialects_edge__ops_aten_exp_default",
        ],
        ops_after_pass={
            "executorch_exir_dialects_edge__ops_aten_mul_Tensor": 1,
            "executorch_exir_dialects_edge__ops_aten_exp_default": 1,
            "executorch_exir_dialects_edge__ops_aten_reciprocal_default": 1,
            "executorch_exir_dialects_edge__ops_aten_sum_dim_IntList": 1,
        },
        ops_not_after_pass=[
            "executorch_exir_dialects_edge__ops_aten__log_softmax_default"
        ],
        pass_list=[DecomposeSoftmaxPass],
    )
    pipeline.run()
