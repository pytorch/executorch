# Copyright 2025 Arm Limited and/or its affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import torch
from executorch.backends.arm.test import common
from executorch.backends.arm.test.tester.test_pipeline import (
    EthosU55PipelineBI,
    EthosU85PipelineBI,
    TosaPipelineBI,
    TosaPipelineMI,
)


class GroupNorm(torch.nn.Module):

    def __init__(
        self,
        num_groups: int,
        num_channels: int,
        eps: float = 1e-5,
        affine: bool = True,
    ):
        super().__init__()
        self.group_norm = torch.nn.GroupNorm(
            num_groups,
            num_channels,
            eps=eps,
            affine=affine,
        )

    def forward(
        self,
        x: torch.Tensor,
    ):
        return self.group_norm(x)


input_t = tuple[torch.Tensor]
test_data_suite = {
    "rand_4_6_groups_1": ((torch.rand(4, 6),), GroupNorm(1, 6)),
    "rand_4_6_groups_2": ((torch.rand(4, 6),), GroupNorm(2, 6)),
    "rand_4_6_groups_6": ((torch.rand(4, 6),), GroupNorm(6, 6)),
    "rand_4_6_8_groups_2_eps_no_affine": (
        (torch.rand(4, 6, 8),),
        GroupNorm(2, 6, eps=1e-3, affine=False),
    ),
    "randn_1_12_8_6_groups_6_eps": (
        (torch.randn(1, 12, 8, 6),),
        GroupNorm(6, 12, eps=1e-2),
    ),
    "randn_1_12_8_6_groups_12": ((torch.randn(1, 12, 8, 6),), GroupNorm(12, 12)),
    "rand_6_8_10_12_groups_1": ((torch.rand(6, 8, 10, 12),), GroupNorm(1, 8)),
    "rand_6_8_10_12_groups_4_no_affine": (
        (torch.rand(6, 8, 10, 12),),
        GroupNorm(4, 8, affine=False),
    ),
    "rand_6_8_10_12_groups_8": ((torch.rand(6, 8, 10, 12),), GroupNorm(8, 8)),
}


@common.parametrize("test_data", test_data_suite)
def test_native_group_norm_tosa_MI(test_data):
    aten_op = "torch.ops.aten.group_norm.default"
    exir_op = "executorch_exir_dialects_edge__ops_aten_native_group_norm_default"
    pipeline = TosaPipelineMI[input_t](
        test_data[1],
        test_data[0],
        aten_op=aten_op,
        exir_op=exir_op,
    )
    pipeline.run()


@common.parametrize(
    "test_data",
    test_data_suite,
    xfails={
        "randn_1_12_8_6_groups_12": "MLETORCH-925: Fix numerical issue for aten.native_group_norm",
        "rand_6_8_10_12_groups_1": "MLETORCH-925: Fix numerical issue for aten.native_group_norm",
        "rand_6_8_10_12_groups_4_no_affine": "MLETORCH-925: Fix numerical issue for aten.native_group_norm",
        "rand_6_8_10_12_groups_8": "MLETORCH-925: Fix numerical issue for aten.native_group_norm",
    },
    strict=False,
)
def test_native_group_norm_tosa_BI(test_data):
    aten_op = "torch.ops.aten.sub.Tensor"  # 'sub' op arbitrarily chosen to confirm groupnorm was decomposed
    exir_op = "executorch_exir_dialects_edge__ops_aten_native_group_norm_default"
    pipeline = TosaPipelineBI[input_t](
        test_data[1],
        test_data[0],
        aten_op=aten_op,
        exir_op=exir_op,
        atol=0.1,  # TODO: "MLETORCH-925: Fix numerical issue for aten.native_group_norm"
    )
    pipeline.run()


@common.parametrize(
    "test_data",
    test_data_suite,
    xfails={
        "randn_1_12_8_6_groups_12": "MLETORCH-925: Fix numerical issue for aten.native_group_norm",
        "rand_6_8_10_12_groups_1": "MLETORCH-925: Fix numerical issue for aten.native_group_norm",
        "rand_6_8_10_12_groups_4_no_affine": "MLETORCH-925: Fix numerical issue for aten.native_group_norm",
        "rand_6_8_10_12_groups_8": "MLETORCH-925: Fix numerical issue for aten.native_group_norm",
    },
    strict=False,
)
@common.XfailIfNoCorstone300
def test_native_group_norm_u55_BI(test_data):
    pipeline = EthosU55PipelineBI[input_t](
        test_data[1],
        test_data[0],
        "torch.ops.aten.sub.Tensor",  # 'sub' op arbitrarily chosen to confirm groupnorm was decomposed
        run_on_fvp=True,
        atol=0.1,  # TODO: "MLETORCH-925: Fix numerical issue for aten.native_group_norm"
    )
    pipeline.change_args("run_method_and_compare_outputs", atol=1, qtol=1)
    pipeline.run()


@common.parametrize(
    "test_data",
    test_data_suite,
    xfails={
        "randn_1_12_8_6_groups_12": "MLETORCH-925: Fix numerical issue for aten.native_group_norm",
        "rand_6_8_10_12_groups_1": "MLETORCH-925: Fix numerical issue for aten.native_group_norm",
        "rand_6_8_10_12_groups_4_no_affine": "MLETORCH-925: Fix numerical issue for aten.native_group_norm",
        "rand_6_8_10_12_groups_8": "MLETORCH-925: Fix numerical issue for aten.native_group_norm",
    },
    strict=False,
)
@common.XfailIfNoCorstone320
def test_native_group_norm_u85_BI(test_data):
    pipeline = EthosU85PipelineBI[input_t](
        test_data[1],
        test_data[0],
        "torch.ops.aten.sub.Tensor",  # 'sub' op arbitrarily chosen to confirm groupnorm was decomposed
        run_on_fvp=True,
        atol=0.1,  # TODO: "MLETORCH-925: Fix numerical issue for aten.native_group_norm"
    )
    pipeline.change_args("run_method_and_compare_outputs", atol=1, qtol=1)
    pipeline.run()
