# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
# Copyright 2024-2025 Arm Limited and/or its affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.


from typing import Tuple

import pytest

import torch
from executorch.backends.arm.test import common
from executorch.backends.arm.test.tester.test_pipeline import (
    EthosU55PipelineBI,
    TosaPipelineBI,
    TosaPipelineMI,
)

input_t1 = Tuple[torch.Tensor]  # Input x


test_data_suite = {
    # (test_name, test_data, [num_features, affine, track_running_stats, weight, bias, running_mean, running_var,] )
    "zeros_affineT_runStatsT_default_weight_bias_mean_var": lambda: (
        torch.zeros(1, 32, 112, 112),
        [
            32,
            True,
            True,
        ],
    ),
    "zeros_affineF_runStatsT_default_weight_bias_mean_var": lambda: (
        torch.zeros(1, 32, 112, 112),
        [
            32,
            False,
            True,
        ],
    ),
    "zeros_affineT_runStatsT_rand_weight_bias_mean_var": lambda: (
        torch.zeros(1, 32, 112, 112),
        [
            32,
            True,
            True,
            torch.rand(32),
            torch.rand(32),
            torch.rand(32),
            torch.rand(32),
        ],
    ),
    "zeros_affineF_runStatsT_rand_weight_bias_mean_var": lambda: (
        torch.zeros(1, 32, 112, 112),
        [
            32,
            False,
            True,
            torch.rand(32),
            torch.rand(32),
            torch.rand(32),
            torch.rand(32),
        ],
    ),
    "ones_affineT_runStatsT_default_weight_bias_mean_var": lambda: (
        torch.ones(1, 32, 112, 112),
        [
            32,
            True,
            True,
        ],
    ),
    "ones_affineF_runStatsT_default_weight_bias_mean_var": lambda: (
        torch.ones(1, 32, 112, 112),
        [
            32,
            False,
            True,
        ],
    ),
    "ones_affineT_runStatsT_rand_weight_bias_mean_var": lambda: (
        torch.ones(1, 32, 112, 112),
        [
            32,
            True,
            True,
            torch.rand(32),
            torch.rand(32),
            torch.rand(32),
            torch.rand(32),
        ],
    ),
    "ones_affineF_runStatsT_rand_weight_bias_mean_var": lambda: (
        torch.ones(1, 32, 112, 112),
        [
            32,
            False,
            True,
            torch.rand(32),
            torch.rand(32),
            torch.rand(32),
            torch.rand(32),
        ],
    ),
    "rand_affineT_runStatsT_default_weight_bias_mean_var": lambda: (
        torch.rand(1, 32, 112, 112),
        [
            32,
            True,
            True,
        ],
    ),
    "rand_affineF_runStatsT_default_weight_bias_mean_var": lambda: (
        torch.rand(1, 32, 112, 112),
        [
            32,
            False,
            True,
        ],
    ),
    "rand_affineT_runStatsT_rand_weight_bias_mean_var": lambda: (
        torch.rand(1, 32, 112, 112),
        [
            32,
            True,
            True,
            torch.rand(32),
            torch.rand(32),
            torch.rand(32),
            torch.rand(32),
        ],
    ),
    "rand_affineF_runStatsT_rand_weight_bias_mean_var": lambda: (
        torch.rand(1, 32, 112, 112),
        [
            32,
            False,
            True,
            torch.rand(32),
            torch.rand(32),
            torch.rand(32),
            torch.rand(32),
        ],
    ),
    "randn_affineT_runStatsT_default_weight_bias_mean_var": lambda: (
        torch.randn(1, 32, 112, 112),
        [
            32,
            True,
            True,
        ],
    ),
    "randn_affineF_runStatsT_default_weight_bias_mean_var": lambda: (
        torch.randn(1, 32, 112, 112),
        [
            32,
            False,
            True,
        ],
    ),
    "randn_affineT_runStatsT_rand_weight_bias_mean_var": lambda: (
        torch.randn(1, 32, 112, 112),
        [
            32,
            True,
            True,
            torch.rand(32),
            torch.rand(32),
            torch.rand(32),
            torch.rand(32),
        ],
    ),
    "randn_affineF_runStatsT_rand_weight_bias_mean_var": lambda: (
        torch.randn(1, 32, 112, 112),
        [
            32,
            False,
            True,
            torch.rand(32),
            torch.rand(32),
            torch.rand(32),
            torch.rand(32),
        ],
    ),
    # Test some different sizes
    "size_3_4_5_6_affineT_runStatsT_rand_weight_bias_mean_var": lambda: (
        torch.rand(3, 4, 5, 6),
        [4, True, True, torch.rand(4), torch.rand(4), torch.rand(4), torch.rand(4)],
    ),
    "size_3_4_5_6_affineF_runStatsT_rand_weight_bias_mean_var": lambda: (
        torch.rand(3, 4, 5, 6),
        [4, True, True, torch.rand(4), torch.rand(4), torch.rand(4), torch.rand(4)],
    ),
    "size_1_3_254_254_affineT_runStatsT_rand_weight_bias_mean_var": lambda: (
        torch.rand(1, 3, 254, 254),
        [3, True, True, torch.rand(3), torch.rand(3), torch.rand(3), torch.rand(3)],
    ),
    "size_1_3_254_254_affineF_runStatsT_rand_weight_bias_mean_var": lambda: (
        torch.rand(1, 3, 254, 254),
        [3, True, True, torch.rand(3), torch.rand(3), torch.rand(3), torch.rand(3)],
    ),
    # Test combination of weight and bias
    "check_weight_bias_affineT_runStatsT_none_none": lambda: (
        torch.rand(1, 32, 112, 112),
        [32, True, True, None, None],
    ),
    "check_weight_bias_affineF_runStatsT_none_none": lambda: (
        torch.rand(1, 32, 112, 112),
        [32, False, True, None, None],
    ),
    "check_weight_bias_affineT_runStatsT_weight_none": lambda: (
        torch.rand(1, 32, 112, 112),
        [32, True, True, torch.rand(32)],
    ),
    "check_weight_bias_affineF_runStatsT_weight_none": lambda: (
        torch.rand(1, 32, 112, 112),
        [32, False, True, torch.rand(32)],
    ),
    "check_weight_bias_affineT_runStatsT_none_bias": lambda: (
        torch.rand(1, 32, 112, 112),
        [32, True, True, None, torch.rand(32)],
    ),
    "check_weight_bias_affineF_runStatsT_none_bias": lambda: (
        torch.rand(1, 32, 112, 112),
        [32, False, True, None, torch.rand(32)],
    ),
    "check_weight_bias_affineT_runStatsT_weight_bias": lambda: (
        torch.rand(1, 32, 112, 112),
        [32, True, True, torch.rand(32), torch.rand(32)],
    ),
    "check_weight_bias_affineF_runStatsT_weight_bias": lambda: (
        torch.rand(1, 32, 112, 112),
        [32, False, True, torch.rand(32), torch.rand(32)],
    ),
    # Test combination of running_mean and running_var
    "check_mean_var_affineT_runStatsT_none_none": lambda: (
        torch.randn(1, 32, 112, 112),
        [32, True, True, torch.rand(32), torch.rand(32), None, None],
    ),
    "check_mean_var_affineF_runStatsT_none_none": lambda: (
        torch.randn(1, 32, 112, 112),
        [32, False, True, torch.rand(32), torch.rand(32), None, None],
    ),
    "check_mean_var_affineT_runStatsT_mean_none": lambda: (
        torch.randn(1, 32, 112, 112),
        [32, True, True, torch.rand(32), torch.rand(32), torch.rand(32), None],
    ),
    "check_mean_var_affineF_runStatsT_mean_none": lambda: (
        torch.randn(1, 32, 112, 112),
        [32, False, True, torch.rand(32), torch.rand(32), torch.rand(32), None],
    ),
    "check_mean_var_affineT_runStatsT_none_var": lambda: (
        torch.randn(1, 32, 112, 112),
        [32, True, True, torch.rand(32), torch.rand(32), None, torch.rand(32)],
    ),
    "check_mean_var_affineF_runStatsT_none_var": lambda: (
        torch.randn(1, 32, 112, 112),
        [32, False, True, torch.rand(32), torch.rand(32), None, torch.rand(32)],
    ),
    "check_mean_var_affineT_runStatsT_mean_var": lambda: (
        torch.randn(1, 32, 112, 112),
        [
            32,
            True,
            True,
            torch.rand(32),
            torch.rand(32),
            torch.rand(32),
            torch.rand(32),
        ],
    ),
    "check_mean_var_affineF_runStatsT_mean_var": lambda: (
        torch.randn(1, 32, 112, 112),
        [
            32,
            False,
            True,
            torch.rand(32),
            torch.rand(32),
            torch.rand(32),
            torch.rand(32),
        ],
    ),
}

test_no_stats_data_suite = {
    # (test_name, test_data, [num_features, affine, track_running_stats, weight, bias, running_mean, running_var, ] )
    "zeros_affineT_runStatsF_default_weight_bias": lambda: (
        torch.zeros(1, 32, 112, 112),
        [
            32,
            True,
            False,
        ],
    ),
    "zeros_affineF_runStatsF_default_weight_bias": lambda: (
        torch.zeros(1, 32, 112, 112),
        [
            32,
            False,
            False,
        ],
    ),
    "zeros_affineT_runStatsF_rand_weight_bias": lambda: (
        torch.zeros(1, 32, 112, 112),
        [32, True, False, torch.rand(32), torch.rand(32)],
    ),
    "zeros_affineF_runStatsF_rand_weight_bias": lambda: (
        torch.zeros(1, 32, 112, 112),
        [32, False, False, torch.rand(32), torch.rand(32)],
    ),
    "ones_affineT_runStatsF_default_weight_bias": lambda: (
        torch.ones(1, 32, 112, 112),
        [
            32,
            True,
            False,
        ],
    ),
    "ones_affineF_runStatsF_default_weight_bias": lambda: (
        torch.ones(1, 32, 112, 112),
        [
            32,
            False,
            False,
        ],
    ),
    "ones_affineT_runStatsF_rand_weight_bias": lambda: (
        torch.ones(1, 32, 112, 112),
        [32, True, False, torch.rand(32), torch.rand(32)],
    ),
    "ones_affineF_runStatsF": lambda: (
        torch.ones(1, 32, 112, 112),
        [32, False, False, torch.rand(32), torch.rand(32)],
    ),
    "rand_affineT_runStatsF_default_weight_bias": lambda: (
        torch.rand(1, 32, 112, 112),
        [
            32,
            True,
            False,
        ],
    ),
    "rand_affineF_runStatsF_default_weight_bias": lambda: (
        torch.rand(1, 32, 112, 112),
        [
            32,
            False,
            False,
        ],
    ),
    "rand_affineT_runStatsF_rand_weight_bias": lambda: (
        torch.rand(1, 32, 112, 112),
        [32, True, False, torch.rand(32), torch.rand(32)],
    ),
    "rand_affineF_runStatsF_rand_weight_bias": lambda: (
        torch.rand(1, 32, 112, 112),
        [32, False, False, torch.rand(32), torch.rand(32)],
    ),
    "randn_affineT_runStatsF_default_weight_bias": lambda: (
        torch.randn(1, 32, 112, 112),
        [
            32,
            True,
            False,
        ],
    ),
    "randn_affineF_runStatsF_default_weight_bias": lambda: (
        torch.randn(1, 32, 112, 112),
        [
            32,
            False,
            False,
        ],
    ),
    "randn_affineT_runStatsF_rand_weight_bias": lambda: (
        torch.randn(1, 32, 112, 112),
        [32, True, False, torch.rand(32), torch.rand(32)],
    ),
    "randn_affineF_runStatsF_rand_weight_bias": lambda: (
        torch.randn(1, 32, 112, 112),
        [32, False, False, torch.rand(32), torch.rand(32)],
    ),
    # Test some different sizes
    "size_3_4_5_6_affineT_runStatsF_rand_weight_bias_mean_var": lambda: (
        torch.rand(3, 4, 5, 6),
        [4, True, False, torch.rand(4), torch.rand(4)],
    ),
    "size_3_4_5_6_affineF_runStatsF_rand_weight_bias_mean_var": lambda: (
        torch.rand(3, 4, 5, 6),
        [4, True, False, torch.rand(4), torch.rand(4)],
    ),
    "size_1_3_254_254_affineT_runStatsF_rand_weight_bias_mean_var": lambda: (
        torch.rand(1, 3, 254, 254),
        [3, True, False, torch.rand(3), torch.rand(3)],
    ),
    "size_1_3_254_254_affineF_runStatsF_rand_weight_bias_mean_var": lambda: (
        torch.rand(1, 3, 254, 254),
        [3, True, False, torch.rand(3), torch.rand(3)],
    ),
    # Test combination of weight and bias
    "check_weight_bias_affineT_runStatsF_none_none": lambda: (
        torch.rand(1, 32, 112, 112),
        [32, True, False, None, None],
    ),
    "check_weight_bias_affineF_runStatsF_none_none": lambda: (
        torch.rand(1, 32, 112, 112),
        [32, False, False, None, None],
    ),
    "check_weight_bias_affineT_runStatsF_weight_none": lambda: (
        torch.rand(1, 32, 112, 112),
        [32, True, False, torch.rand(32)],
    ),
    "check_weight_bias_affineF_runStatsF_weight_none": lambda: (
        torch.rand(1, 32, 112, 112),
        [32, False, False, torch.rand(32)],
    ),
    "check_weight_bias_affineT_runStatsF_none_bias": lambda: (
        torch.rand(1, 32, 112, 112),
        [32, True, False, None, torch.rand(32)],
    ),
    "check_weight_bias_affineF_runStatsF_none_bias": lambda: (
        torch.rand(1, 32, 112, 112),
        [32, False, False, None, torch.rand(32)],
    ),
    "check_weight_bias_affineT_runStatsF_weight_bias": lambda: (
        torch.rand(1, 32, 112, 112),
        [32, True, False, torch.rand(32), torch.rand(32)],
    ),
    "check_weight_bias_affineF_runStatsF_weight_bias": lambda: (
        torch.rand(1, 32, 112, 112),
        [32, False, False, torch.rand(32), torch.rand(32)],
    ),
}


class BatchNorm2d(torch.nn.Module):
    def __init__(
        self,
        num_features: int = 32,
        affine: bool = False,
        track_running_stats: bool = True,
        weights: torch.tensor = None,
        bias: torch.tensor = None,
        running_mean: torch.tensor = None,
        running_var: torch.tensor = None,
    ):
        super().__init__()
        self.batch_norm_2d = torch.nn.BatchNorm2d(
            num_features, affine=affine, track_running_stats=track_running_stats
        )
        if weights is not None:
            self.batch_norm_2d.weight = torch.nn.Parameter(weights)
        if bias is not None:
            self.batch_norm_2d.bias = torch.nn.Parameter(bias)
        if running_mean is not None:
            self.batch_norm_2d.running_mean = running_mean
        if running_var is not None:
            self.batch_norm_2d.running_var = running_var

    def forward(self, x):
        return self.batch_norm_2d(x)


@common.parametrize("test_data", test_data_suite)
def test_native_batch_norm_legit_tosa_MI_no_training(test_data: Tuple):
    test_data, model_params = test_data()
    pipeline = TosaPipelineMI[input_t1](
        BatchNorm2d(*model_params),
        (test_data,),
        aten_op=[],
        exir_op="executorch_exir_dialects_edge__ops_aten__native_batch_norm_legit_no_training_default",
    )
    pipeline.run()


@common.parametrize("test_data", test_no_stats_data_suite)
# Expected to fail since not inplemented
@pytest.mark.skip  # Not implemented, skip until it is.
def test_native_batch_norm_legit_tosa_MI(test_data: Tuple):
    test_data, model_params = test_data()
    pipeline = TosaPipelineMI[input_t1](
        BatchNorm2d(*model_params),
        (test_data,),
        aten_op=[],
        exir_op="executorch_exir_dialects_edge__ops_aten__native_batch_norm_legit_no_training_default",
    )
    pipeline.pop_stage("check_count.exir")
    pipeline.run()


# Expected to fail since TOSAQuantizer cannot quantize a BatchNorm layer
# TODO(MLETORCH-100)
@common.parametrize("test_data", test_data_suite)
@pytest.mark.skip  # Not implemented, skip until it is.
def test_native_batch_norm_legit_tosa_BI_no_training(test_data: Tuple):
    test_data, model_params = test_data()
    pipeline = TosaPipelineBI[input_t1](
        BatchNorm2d(*model_params),
        (test_data,),
        aten_op="torch.ops.aten._native_batch_norm_legit_no_training.default",
        exir_op="executorch_exir_dialects_edge__ops_aten__native_batch_norm_legit_no_training_default",
    )
    pipeline.run()


# Expected to fail since EthosUQuantizer (TOSAQuantizer (BI)) cannot quantize a BatchNorm layer
# TODO(MLETORCH-100)
@common.parametrize("test_data", test_data_suite)
@pytest.mark.skip  # Not implemented, skip until it is.
def test_native_batch_norm_legit_u55_BI_no_training(test_data: Tuple):
    test_data, model_params = test_data()
    pipeline = EthosU55PipelineBI[input_t1](
        BatchNorm2d(*model_params),
        test_data,
        aten_ops="torch.ops.aten._native_batch_norm_legit_no_training.default",
        exir_ops="executorch_exir_dialects_edge__ops_aten__native_batch_norm_legit_no_training_default",
        run_on_fvp=True,
    )
    pipeline.run()
