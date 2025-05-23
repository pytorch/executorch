# Copyright 2025 Arm Limited and/or its affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from typing import Tuple

import torch

from executorch.backends.arm.test import common
from executorch.backends.arm.test.tester.test_pipeline import (
    EthosU55PipelineBI,
    EthosU85PipelineBI,
    TosaPipelineBI,
    TosaPipelineMI,
)

input_t = Tuple[torch.Tensor]

aten_op_q_decomposed_q = "torch.ops.quantized_decomposed.quantize_per_tensor.default"
exir_op_q_decomposed = "executorch_exir_dialects_edge__ops_quantized_decomposed_quantize_per_tensor_default"


class VectorNormModel(torch.nn.Module):
    def __init__(
        self,
        ord=None,
        dim=1,
        keepdim=False,
    ):
        """
        A simple module that applies torch.linalg.vector_norm to its input.
        Ord is 2 by default.
        """
        super().__init__()
        self.ord = ord
        self.dim = dim
        self.keepdim = keepdim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.ord is None and self.dim is None:
            return torch.linalg.vector_norm(x, keepdim=self.keepdim)
        elif self.ord is None:
            return torch.linalg.vector_norm(x, dim=self.dim, keepdim=self.keepdim)
        elif self.dim is None:
            return torch.linalg.vector_norm(x, ord=self.ord, keepdim=self.keepdim)
        else:
            return torch.linalg.vector_norm(
                x, ord=self.ord, dim=self.dim, keepdim=self.keepdim
            )


test_modules = {
    "default": (VectorNormModel(dim=1), (torch.rand(10, 4),)),
    "ord1": (VectorNormModel(ord=1, dim=1), (torch.rand(10, 4),)),
    "ord2": (VectorNormModel(ord=2, dim=1), (torch.rand(10, 20),)),
    # Norm computed along a specific dimension of a 3D tensor
    "dim_3d": (VectorNormModel(dim=2), (torch.rand(4, 5, 6),)),
}


@common.parametrize("test_module", test_modules)
def test_vector_norm_tosa_MI(test_module):
    model, input_tensor = test_module

    # We decompose LinalgVectorNorm before quantize stage to have annotations
    # with q/dq nodes. In case of MI, this operator will be decomposed
    # by global decompositions.
    aten_op = "torch.ops.aten.linalg_vector_norm.default"
    # Should not found this op
    exir_op = "executorch_exir_dialects_edge__ops_aten_linalg_vector_norm_default"

    pipeline = TosaPipelineMI[input_t](model, input_tensor, aten_op, exir_op)

    pipeline.change_args("run_method_and_compare_outputs", qtol=1, atol=1e-4, rtol=1e-4)
    pipeline.run()


@common.parametrize("test_module", test_modules)
def test_vector_norm_tosa_BI(test_module):
    model, input_tensor = test_module

    # Should not found this op
    exir_op = "executorch_exir_dialects_edge__ops_aten_linalg_vector_norm_default"

    pipeline = TosaPipelineBI[input_t](
        model,
        input_tensor,
        aten_op_q_decomposed_q,
        exir_op,
        symmetric_io_quantization=True,
    )
    pipeline.change_args("run_method_and_compare_outputs", qtol=1, atol=1, rtol=1)
    pipeline.run()


@common.parametrize("test_module", test_modules)
@common.XfailIfNoCorstone300
def test_vector_norm_u55_BI_fvp(test_module):
    model, input_tensor = test_module

    pipeline = EthosU55PipelineBI[input_t](
        model,
        input_tensor,
        aten_op_q_decomposed_q,
        exir_op_q_decomposed,
        run_on_fvp=True,
        symmetric_io_quantization=True,
    )
    pipeline.change_args("run_method_and_compare_outputs", qtol=1, atol=1, rtol=1)
    pipeline.pop_stage("check_not.exir")
    pipeline.run()


@common.parametrize("test_module", test_modules)
@common.XfailIfNoCorstone300
def test_vector_norm_u85_BI_fvp(test_module):
    model, input_tensor = test_module

    # The should be decomposed and annotated in DecomposeLinalgVectorNorm pass.
    pipeline = EthosU85PipelineBI[input_t](
        model,
        input_tensor,
        aten_op_q_decomposed_q,
        exir_op_q_decomposed,
        run_on_fvp=True,
        symmetric_io_quantization=True,
    )
    pipeline.change_args("run_method_and_compare_outputs", qtol=1, atol=1, rtol=1)
    pipeline.pop_stage("check_not.exir")
    pipeline.run()
