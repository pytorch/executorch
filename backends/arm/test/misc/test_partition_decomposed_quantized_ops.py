# Copyright 2025 Arm Limited and/or its affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# Test that tosa_supported_operators reject operators that are not
# quantized properly. This is typically a consequence of a torch op
# such a Softplus that is decompsed into many other ops without
# surrounding q/dq nodes.

from typing import Tuple

import torch
from executorch.backends.arm.test import common

from executorch.backends.arm.test.tester.test_pipeline import (
    TosaPipelineFP,
    TosaPipelineINT,
)

input_t1 = Tuple[torch.Tensor]
softplus_aten_op: list[str] = [
    "torch.ops.aten.add.Tensor",
    "torch.ops.aten.softplus.default",
]
softplus_exir_op: list[str] = [
    "executorch_exir_dialects_edge__ops_aten_add_Tensor",
    "executorch_exir_dialects_edge__ops_aten_mul_Tensor",
    "executorch_exir_dialects_edge__ops_aten_exp_default",
    "executorch_exir_dialects_edge__ops_aten_div_Tensor",
]

linear_residual_aten_op: list[str] = [
    "torch.ops.aten.linear.default",
    "torch.ops.aten.gelu.default",
    "torch.ops.aten.dropout.default",
    "torch.ops.aten.add.Tensor",
]
linear_residual_exir_op: list[str] = [
    "executorch_exir_dialects_edge__ops_aten_gelu_default",
    "executorch_exir_dialects_edge__ops_dim_order_ops__clone_dim_order_default",
    "executorch_exir_dialects_edge__ops_aten_linear_default",
    "executorch_exir_dialects_edge__ops_aten_add_Tensor",
]


test_data: dict[input_t1] = {
    "3d_rand": (torch.rand(1, 5, 5),),
}


class SoftplusModule(torch.nn.Module):
    """Module containing an addition followed by a Softplus. Softplus is currently not supported by TosaBackend."""

    def __init__(self):
        super().__init__()
        self.softplus = torch.nn.Softplus()

    def forward(self, x: torch.Tensor):
        return self.softplus(x + x)


class LinearResidualModule(torch.nn.Module):
    """Module containing a residual and a linear layer followed by GELU and a Dropout.
    GELU is currently not supported by TosaBackend nor TosaQuantizer.
    """

    def __init__(
        self,
    ):
        super().__init__()
        self.linear = torch.nn.Linear(in_features=5, out_features=3)
        self.gelu = torch.nn.GELU()
        self.dropout = torch.nn.Dropout(0.5)

    def forward(self, x: torch.Tensor):
        x1 = self.linear(x)
        x2 = self.gelu(x1)
        x3 = self.dropout(x2)
        return x1 + x3


# Softplus is decomposed which messes up the quantization. This test tests that CheckProperQuantization does not
# partition nodes where quantization is not as expected.
@common.parametrize("test_data", test_data)
def test_softplus_tosa_FP(test_data: input_t1):
    pipeline = TosaPipelineFP[input_t1](
        SoftplusModule(),
        test_data=test_data,
        aten_op=softplus_aten_op,
        exir_op=softplus_exir_op,
    )
    # remove check_count.exir as there will be more than one delegate
    pipeline.pop_stage("check_count.exir")
    pipeline.run()


@common.parametrize("test_data", test_data)
def test_softplus_tosa_INT(test_data: input_t1):
    pipeline = TosaPipelineINT[input_t1](
        SoftplusModule(),
        test_data=test_data,
        aten_op=softplus_aten_op,
        exir_op=softplus_exir_op,
    )
    pipeline.pop_stage("check_not.exir")
    # check that all ops in softplus_exir_op except add are rejected
    pipeline.add_stage_after(
        "to_edge_transform_and_lower",
        pipeline.tester.check,
        softplus_exir_op[1:],
        suffix="exir_post_partition",
    )
    pipeline.run()


# Since GELU will not be quantized by TosaQuantizer, the Dropout's input will not be quantized either.
# If so, the Dropout should not be partitioned by TosaPartitioner for TOSA INT profile. This test tests that the
# partitioner indeed does not partition the Dropout (clone) for TOSA INT.
@common.parametrize(
    "test_data",
    test_data,
    {"3d_rand": "MLETORCH-909: Partition test to not rely on unsupported ops"},
    strict=False,
)
def test_linear_residaul_tosa_FP(test_data: input_t1):
    pipeline = TosaPipelineFP[input_t1](
        LinearResidualModule(),
        test_data=test_data,
        aten_op=linear_residual_aten_op,
        exir_op=linear_residual_exir_op,
        use_to_edge_transform_and_lower=True,
    )
    # remove check_count.exir as there will be more than one delegate
    pipeline.pop_stage("check_count.exir")
    pipeline.pop_stage("check_not.exir")
    # check that all ops in linear_residual_exir_op except GELU are partitioned
    pipeline.add_stage_after(
        "to_edge_transform_and_lower",
        pipeline.tester.check_not,
        linear_residual_exir_op[1:],
        suffix="exir_post_partition",
    )
    pipeline.add_stage_after(
        "to_edge_transform_and_lower",
        pipeline.tester.check,
        linear_residual_exir_op[:1],
        suffix="exir_post_partition",
    )
    pipeline.run()


@common.parametrize(
    "test_data",
    test_data,
    {"3d_rand": "MLETORCH-855: Issue with Quantization folding."},
    strict=False,
)
def test_linear_residual_tosa_INT(test_data: input_t1):
    pipeline = TosaPipelineINT[input_t1](
        LinearResidualModule(),
        test_data=test_data,
        aten_op=linear_residual_aten_op,
        exir_op=linear_residual_exir_op,
        use_to_edge_transform_and_lower=True,
    )
    # remove check_count.exir as there will be more than one delegate
    pipeline.pop_stage("check_count.exir")
    pipeline.pop_stage("check_not.exir")
    # check that all ops in linear_residual_exir_op except GELU and Dropout are partitioned
    pipeline.add_stage_after(
        "to_edge_transform_and_lower",
        pipeline.tester.check_not,
        linear_residual_exir_op[2:],
        suffix="exir_post_partition",
    )
    pipeline.add_stage_after(
        "to_edge_transform_and_lower",
        pipeline.tester.check,
        linear_residual_exir_op[:2],
        suffix="exir_post_partition",
    )
    pipeline.run()
