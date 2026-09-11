# Copyright 2026 Arm Limited and/or its affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from typing import Tuple

import torch
from executorch.backends.arm.test import common
from executorch.backends.arm.test.tester.test_pipeline import (
    OpNotSupportedPipeline,
    TosaPipelineFP,
    VgfPipeline,
)

aten_op = "torch.ops.aten.prod.dim_int"
exir_op = "executorch_exir_dialects_edge__ops_aten_prod_dim_int"
input_t1 = Tuple[torch.Tensor]


class Prod(torch.nn.Module):
    def __init__(self, dim: int, keepdim: bool):
        super().__init__()
        self.dim = dim
        self.keepdim = keepdim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.prod(x, dim=self.dim, keepdim=self.keepdim)


test_parameters = {
    "rank1_dim0": lambda: (torch.rand(4), 0, True),
    "rank1_dim0_squeeze": lambda: (torch.rand(4), 0, False),
    "rank2_dim1": lambda: (torch.rand(2, 3), 1, True),
    "rank2_dim1_squeeze": lambda: (torch.rand(2, 3), 1, False),
    "rank3_dim_neg1": lambda: (torch.rand(2, 3, 4), -1, True),
    "rank3_dim_neg1_squeeze": lambda: (torch.rand(2, 3, 4), -1, False),
    "rank2_dim1_fp16": lambda: (torch.rand(2, 3, dtype=torch.float16), 1, True),
    "rank2_dim1_squeeze_fp16": lambda: (
        torch.rand(2, 3, dtype=torch.float16),
        1,
        False,
    ),
    "rank2_dim1_bf16": lambda: (torch.rand(2, 3, dtype=torch.bfloat16), 1, True),
    "rank2_dim1_squeeze_bf16": lambda: (
        torch.rand(2, 3, dtype=torch.bfloat16),
        1,
        False,
    ),
}


unsupported_integer_test_parameters = {
    "rank1_dim0_int8": lambda: (
        torch.randint(1, 4, (4,), dtype=torch.int8),
        0,
        True,
    ),
    "rank2_dim1_int32": lambda: (
        torch.randint(1, 4, (2, 3), dtype=torch.int32),
        1,
        True,
    ),
    "rank2_dim1_squeeze_int32": lambda: (
        torch.randint(1, 4, (2, 3), dtype=torch.int32),
        1,
        False,
    ),
}


@common.parametrize("test_data", test_parameters)
def test_prod_tosa_FP(test_data: input_t1) -> None:
    x, dim, keepdim = test_data()
    pipeline = TosaPipelineFP[input_t1](
        Prod(dim, keepdim),
        (x,),
        aten_op,
        exir_op,
        tosa_extensions=["bf16"],
    )
    pipeline.run()


@common.parametrize("test_data", unsupported_integer_test_parameters)
def test_prod_tosa_FP_integer_input_not_delegated(test_data: input_t1) -> None:
    x, dim, keepdim = test_data()
    pipeline = OpNotSupportedPipeline[input_t1](
        Prod(dim, keepdim),
        (x,),
        {exir_op: 1},
    )
    pipeline.run()


@common.parametrize("test_data", test_parameters)
@common.SkipIfNoModelConverter
def test_prod_vgf_no_quant(test_data: input_t1) -> None:
    x, dim, keepdim = test_data()
    pipeline = VgfPipeline[input_t1](
        Prod(dim, keepdim),
        (x,),
        aten_op,
        exir_op,
        run_on_vulkan_runtime=True,
        quantize=False,
    )
    pipeline.run()


@common.parametrize("test_data", test_parameters)
@common.SkipIfNoModelConverter
def test_prod_vgf_quant(test_data: input_t1) -> None:
    x, dim, keepdim = test_data()
    pipeline = VgfPipeline[input_t1](
        Prod(dim, keepdim),
        (x,),
        aten_op,
        exir_op,
        run_on_vulkan_runtime=True,
        quantize=True,
    )
    pipeline.run()
