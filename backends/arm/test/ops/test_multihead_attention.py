# Copyright 2025 Arm Limited and/or its affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import pytest
import torch
from executorch.backends.arm.test import common
from executorch.backends.arm.test.tester.test_pipeline import (
    EthosU55PipelineBI,
    EthosU85PipelineBI,
    TosaPipelineBI,
    TosaPipelineMI,
)


class MultiheadAttention(torch.nn.MultiheadAttention):
    def forward(self, *args, **kwargs):
        return super().forward(*args, **kwargs)


input_t1 = tuple[torch.Tensor, torch.nn.Module]
test_suite = {
    # test_name, (x,), embed_dim, num_heads, batch_first
    "rand_2d": lambda: (
        (torch.rand(6, 3),),
        MultiheadAttention(embed_dim=3, num_heads=3, batch_first=True),
    ),
    "randn_2d": lambda: (
        (torch.randn(2, 4),),
        MultiheadAttention(embed_dim=4, num_heads=2, batch_first=True),
    ),
    "randn_3d": lambda: (
        (torch.randn(3, 2, 4),),
        MultiheadAttention(embed_dim=4, num_heads=2, batch_first=False),
    ),
}


@common.parametrize(
    "test_data",
    test_suite,
)
def test_multihead_attention_tosa_MI(test_data: input_t1):
    test_data, module = test_data()
    pipeline = TosaPipelineMI(module, (*test_data, *test_data, *test_data), [], [])
    pipeline.run()


@common.parametrize(
    "test_data",
    test_suite,
)
def test_multihead_attention_tosa_BI(test_data):
    test_data, module = test_data()
    pipeline = TosaPipelineBI(module, (*test_data, *test_data, *test_data), [], [])
    pipeline.run()


@common.parametrize(
    "test_data",
    test_suite,
)
@pytest.mark.xfail(reason="MLETORCH-1102: Numerical issues on FVP")
@common.XfailIfNoCorstone300
def test_multihead_attention_u55_BI(test_data: input_t1):
    test_data, module = test_data()
    pipeline = EthosU55PipelineBI(
        module,
        (*test_data, *test_data, *test_data),
        [],
        [],
        use_to_edge_transform_and_lower=True,
        run_on_fvp=True,
    )
    pipeline.pop_stage("check_count.exir")
    pipeline.run()


@common.parametrize(
    "test_data",
    test_suite,
)
@pytest.mark.xfail(reason="MLETORCH-1102: Numerical issues on FVP")
@common.XfailIfNoCorstone320
def test_multihead_attention_u85_BI(test_data: input_t1):
    test_data, module = test_data()
    pipeline = EthosU85PipelineBI(
        module,
        (*test_data, *test_data, *test_data),
        [],
        [],
        use_to_edge_transform_and_lower=True,
        run_on_fvp=True,
    )
    pipeline.run()
