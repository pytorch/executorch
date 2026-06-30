# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
# Copyright 2024-2026 Arm Limited and/or its affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from typing import Tuple

import pytest

import torch
from executorch.backends.arm.test import common
from executorch.backends.arm.test.tester.test_pipeline import (
    EthosU55PipelineINT,
    EthosU85PipelineINT,
    TosaPipelineFP,
    TosaPipelineINT,
    VgfPipeline,
)

aten_op = "torch.ops.aten.softmax.default"  # Used for checking that we do not have softmax in the graph after decompose
exir_op = "executorch_exir_dialects_edge__ops_aten__softmax_tensor"
input_t1 = Tuple[torch.Tensor]  # Input x


class Softmax(torch.nn.Module):
    def __init__(self, dim: int = -1):
        super().__init__()
        self.softmax = torch.nn.Softmax(dim=dim)

    def forward(self, x):
        return self.softmax(x)

    test_data = {
        "ones": lambda: ((torch.ones(10, 10),), 1),
        "ones_neg_dim": lambda: ((torch.ones(1, 3, 4),), -1),
        "bigger_numbers": lambda: ((1000 * torch.rand(1, 16, 64, 64),), -1),
        "randn_neg_dim": lambda: ((torch.randn(1, 5, 8, 7),), -3),
        "zeros": lambda: ((torch.zeros(1, 8, 5, 2),), 0),
        "zeros_neg_dim": lambda: ((torch.zeros(1, 7, 8, 9),), -4),
        "rand": lambda: ((torch.rand(1, 2, 5, 8),), 2),
        "rand_neg_dim": lambda: ((torch.rand(1, 10, 8, 10),), -2),
        "randn_mult_batches": lambda: ((torch.randn(2, 10, 10, 10),), 3),
    }


@common.parametrize("test_data", Softmax.test_data)
def test_softmax_tosa_FP(test_data):
    data, dim = test_data()
    pipeline = TosaPipelineFP[input_t1](Softmax(dim), data, [])
    pipeline.add_stage_after(
        "to_edge_transform_and_lower", pipeline.tester.check_not, [exir_op]
    )
    pipeline.run()


@common.parametrize("test_data", Softmax.test_data)
def test_softmax_tosa_INT(test_data):
    data, dim = test_data()
    pipeline = TosaPipelineINT[input_t1](Softmax(dim), data, [])
    pipeline.add_stage_after("quantize", pipeline.tester.check_not, [aten_op])
    pipeline.change_args("run_method_and_compare_outputs", qtol=1)
    pipeline.run()


@common.parametrize("test_data", Softmax.test_data)
@common.XfailIfNoCorstone300
def test_softmax_u55_INT(test_data):
    data, dim = test_data()
    pipeline = EthosU55PipelineINT[input_t1](
        Softmax(dim),
        data,
        [],
    )
    pipeline.add_stage_after("quantize", pipeline.tester.check_not, [aten_op])
    pipeline.change_args("run_method_and_compare_outputs", qtol=1)
    pipeline.run()


@common.parametrize("test_data", Softmax.test_data)
@common.XfailIfNoCorstone320
def test_softmax_u85_INT(test_data):
    data, dim = test_data()
    pipeline = EthosU85PipelineINT[input_t1](
        Softmax(dim),
        data,
        [],
    )
    pipeline.add_stage_after("quantize", pipeline.tester.check_not, [aten_op])
    pipeline.change_args("run_method_and_compare_outputs", qtol=1)
    pipeline.run()


@common.parametrize("test_data", Softmax.test_data)
@common.SkipIfNoModelConverter
def test_softmax_vgf_no_quant(test_data):
    data, dim = test_data()
    pipeline = VgfPipeline[input_t1](
        Softmax(dim),
        data,
        [],
        quantize=False,
    )
    pipeline.add_stage_after(
        "to_edge_transform_and_lower", pipeline.tester.check_not, [exir_op]
    )
    pipeline.run()


@common.parametrize("test_data", Softmax.test_data)
@common.SkipIfNoModelConverter
def test_softmax_vgf_quant(test_data):
    data, dim = test_data()
    pipeline = VgfPipeline[input_t1](
        Softmax(dim),
        data,
        [],
        quantize=True,
    )
    pipeline.add_stage_after("quantize", pipeline.tester.check_not, [aten_op])
    # TODO: MLETORCH-1136 Change args of run_method_and_compare_outputs of the vgf tests
    # pipeline.change_args("run_method_and_compare_outputs", qtol=1)
    pipeline.run()


# ---------------------------------------------------------------------------
# a16w8 (int16 IO + int8 weights) softmax FVP coverage.
#
# Sweeps a multi-head-attention-shaped softmax over a wide range of
# pre-softmax input magnitudes to surface int16 numerics issues in the
# lowered graph (e.g. the Ethos-U85 ReduceSum int16 silent-zero issue in the
# softmax decomposition, fixed by the follow-up Vela patch in this stack).
# ---------------------------------------------------------------------------


class MultiHeadAttentionSoftmax(torch.nn.Module):
    """Generic multi-head-attention softmax: reshape -> softmax(dim=-1) -> flatten.

    H heads, M query tokens, W K/V window. Output shape: (N, T, H*M*W).
    """

    H = 4
    M = 1
    W = 16
    IN_FEATURES = H * M * W  # 64

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        n, t, _ = x.shape
        x = x.reshape(n, t, self.H, self.M, self.W)
        x = torch.softmax(x, dim=-1)
        x = x.reshape(n, t, self.IN_FEATURES)
        return x


# (input_low, input_high) per case. Keys are the parametrize ids.
# Range coverage spans realistic post-1/sqrt(d) attention logits (typically
# in [-10, +10]) plus a couple of wider buffer cases. atol below is sized
# at ~1.5x the observed FVP max-abs softmax error across the sweep at
# qtol=0, measured against the quantized reference.
mha_softmax_sweep = {
    "range_neg0p01_to_0p01": (-0.01, 0.01),
    "range_neg0p1_to_0p1": (-0.1, 0.1),
    "range_neg1_to_1": (-1.0, 1.0),
    "range_neg3_to_3": (-3.0, 3.0),
    "range_neg5_to_5": (-5.0, 5.0),
    "range_neg10_to_10": (-10.0, 10.0),
    "range_neg30_to_30": (-30.0, 30.0),
}

_MHA_ATOL = 0.003


def _make_mha_softmax_inputs(
    input_low: float, input_high: float, num_test: int = 8, seed: int = 42
) -> Tuple[torch.Tensor]:
    # Local Generator so this helper does not mutate the global RNG state
    # and the test suite stays order-independent.
    gen = torch.Generator().manual_seed(seed)
    span = input_high - input_low
    return (
        torch.rand(
            num_test,
            1,
            MultiHeadAttentionSoftmax.IN_FEATURES,
            generator=gen,
        )
        * span
        + input_low,
    )


@common.parametrize("sweep_case", mha_softmax_sweep)
@common.XfailIfNoCorstone300
def test_mha_softmax_a16w8_u55_INT(sweep_case: Tuple[float, float]) -> None:
    input_low, input_high = sweep_case
    pipeline = EthosU55PipelineINT[input_t1](
        MultiHeadAttentionSoftmax(),
        _make_mha_softmax_inputs(input_low, input_high),
        [],
        exir_ops=[],
        a16w8_quantization=True,
        symmetric_io_quantization=True,
        epsilon=2**-16,
        atol=_MHA_ATOL,
    )
    pipeline.add_stage_after("quantize", pipeline.tester.check_not, [aten_op])
    pipeline.run()


# All cases hit the Ethos-U85 int16 ReduceSum silent-zero issue inside the
# softmax decomposition. strict=False so the test target stays green both
# on stock Vela 5.0 (cases XFAIL) and once the upstream Vela fix lands
# (cases XPASS).
# Upstream report:
#   https://gitlab.arm.com/artificial-intelligence/ethos-u/ethos-u-vela/-/issues/23
@common.parametrize("sweep_case", mha_softmax_sweep)
@common.XfailIfNoCorstone320
@pytest.mark.xfail(
    reason=(
        "Ethos-U85 int16 ReduceSum silent-zero in softmax decomposition: "
        "https://gitlab.arm.com/artificial-intelligence/ethos-u/ethos-u-vela/-/issues/23"
    ),
    strict=False,
)
def test_mha_softmax_a16w8_u85_INT(sweep_case: Tuple[float, float]) -> None:
    input_low, input_high = sweep_case
    pipeline = EthosU85PipelineINT[input_t1](
        MultiHeadAttentionSoftmax(),
        _make_mha_softmax_inputs(input_low, input_high),
        [],
        exir_ops=[],
        a16w8_quantization=True,
        symmetric_io_quantization=True,
        epsilon=2**-16,
        atol=_MHA_ATOL,
    )
    pipeline.add_stage_after("quantize", pipeline.tester.check_not, [aten_op])
    pipeline.run()
