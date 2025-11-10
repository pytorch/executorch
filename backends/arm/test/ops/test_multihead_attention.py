# Copyright 2025 Arm Limited and/or its affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import torch
from executorch.backends.arm.test import common
from executorch.backends.arm.test.tester.test_pipeline import (
    EthosU55PipelineINT,
    EthosU85PipelineINT,
    TosaPipelineFP,
    TosaPipelineINT,
    VgfPipeline,
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
def test_multihead_attention_tosa_FP(test_data: input_t1):
    test_data, module = test_data()
    pipeline = TosaPipelineFP(module, (*test_data, *test_data, *test_data), [], [])
    pipeline.run()


@common.parametrize(
    "test_data",
    test_suite,
)
def test_multihead_attention_tosa_INT(test_data):
    test_data, module = test_data()
    pipeline = TosaPipelineINT(
        module,
        (*test_data, *test_data, *test_data),
        [],
        [],
        # TODO: Per-channel quantization is broken (MLETORCH-1144)
        per_channel_quantization=False,
    )
    pipeline.run()


@common.parametrize(
    "test_data",
    test_suite,
)
@common.XfailIfNoCorstone300
def test_multihead_attention_u55_INT(test_data: input_t1):
    test_data, module = test_data()
    pipeline = EthosU55PipelineINT(
        module,
        (*test_data, *test_data, *test_data),
        [],
        [],
        use_to_edge_transform_and_lower=True,
        # TODO: Per-channel quantization is broken (MLETORCH-1144)
        per_channel_quantization=False,
    )
    pipeline.pop_stage("check_count.exir")
    pipeline.run()


@common.parametrize(
    "test_data",
    test_suite,
)
@common.XfailIfNoCorstone320
def test_multihead_attention_u85_INT(test_data: input_t1):
    test_data, module = test_data()
    pipeline = EthosU85PipelineINT(
        module,
        (*test_data, *test_data, *test_data),
        [],
        [],
        use_to_edge_transform_and_lower=True,
        # TODO: Per-channel quantization is broken (MLETORCH-1144)
        per_channel_quantization=False,
    )
    pipeline.run()


@common.parametrize(
    "test_data",
    test_suite,
)
@common.SkipIfNoModelConverter
def test_multihead_attention_vgf_FP(test_data: input_t1):
    test_data_vals, module = test_data()
    pipeline = VgfPipeline[input_t1](
        module,
        (*test_data_vals, *test_data_vals, *test_data_vals),
        [],
        [],
        tosa_version="TOSA-1.0+FP",
    )
    pipeline.run()


@common.parametrize(
    "test_data",
    test_suite,
)
@common.SkipIfNoModelConverter
def test_multihead_attention_vgf_INT(test_data: input_t1):
    test_data_vals, module = test_data()
    pipeline = VgfPipeline[input_t1](
        module,
        (*test_data_vals, *test_data_vals, *test_data_vals),
        [],
        [],
        tosa_version="TOSA-1.0+INT",
        # TODO: Per-channel quantization is broken (MLETORCH-1144)
        per_channel_quantization=False,
    )
    pipeline.run()
