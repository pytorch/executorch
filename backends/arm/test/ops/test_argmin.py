# Copyright 2026 Arm Limited and/or its affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import torch
from executorch.backends.arm.test import common
from executorch.backends.arm.test.tester.test_pipeline import (
    OpNotSupportedPipeline,
    TosaPipelineFP,
    VgfPipeline,
)

aten_op = "torch.ops.aten.argmin.default"
exir_op = "executorch_exir_dialects_edge__ops_aten_argmin_default"
to_copy_exir_op = (
    "executorch_exir_dialects_edge__ops_dim_order_ops__to_dim_order_copy_default"
)
input_t = tuple[torch.Tensor]


class Argmin(torch.nn.Module):
    def __init__(self, dim=-1, keepdim=False, narrow=True):
        super().__init__()
        self.dim = dim
        self.keepdim = keepdim
        self.narrow = narrow

    def forward(self, x):
        indices = x.argmin(self.dim, keepdim=self.keepdim)
        return indices.to(torch.int32) if self.narrow else indices


@common.parametrize(
    "test_data",
    {
        "rank_2_neg1": lambda: (torch.randn(64, 16), -1),
        "rank_2_dim0": lambda: (torch.randn(4, 5), 0),
        "rank_4_dim2": lambda: (torch.randn(1, 3, 4, 5), 2),
        "fp16": lambda: (torch.randn(4, 5, dtype=torch.float16), -1),
        "bf16": lambda: (torch.randn(4, 5, dtype=torch.bfloat16), -1),
        "ties_and_special_values": lambda: (
            torch.tensor(
                [
                    [2.0, -3.0, -3.0, 1.0],
                    [0.0, -0.0, 1.0, 2.0],
                    [float("inf"), -1.0, float("-inf"), float("-inf")],
                    [1.0, float("nan"), -2.0, float("nan")],
                ]
            ),
            -1,
        ),
    },
)
def test_argmin_tosa_FP(test_data):
    x, dim = test_data()
    pipeline = TosaPipelineFP[input_t](
        Argmin(dim),
        (x,),
        aten_op,
        [exir_op, to_copy_exir_op],
        tosa_extensions=["bf16"],
        atol=0,
        rtol=0,
    )
    pipeline.count_tosa_ops({"NEGATE": 1, "ARGMAX": 1})
    pipeline.run()


@common.SkipIfNoModelConverter
def test_argmin_vgf_no_quant():
    pipeline = VgfPipeline[input_t](
        Argmin(-1),
        (torch.randn(64, 16),),
        aten_op,
        [exir_op, to_copy_exir_op],
        quantize=False,
        tosa_extensions=[],
        run_on_vulkan_runtime=False,
        n_expected_delegates=1,
    )
    pipeline.run()


@common.parametrize(
    "test_data",
    {
        "int64_output": lambda: (Argmin(narrow=False), torch.rand(2, 5)),
        "keepdim": lambda: (Argmin(keepdim=True), torch.rand(2, 5)),
        "all_dims": lambda: (Argmin(dim=None), torch.rand(2, 5)),
        "integer_input": lambda: (
            Argmin(),
            torch.tensor([[-128, -127, 127]], dtype=torch.int8),
        ),
    },
)
def test_argmin_tosa_FP_not_delegated(test_data):
    model, x = test_data()
    pipeline = OpNotSupportedPipeline[input_t](model, (x,), {exir_op: 1})
    pipeline.run()
