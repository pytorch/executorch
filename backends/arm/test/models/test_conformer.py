# Copyright 2025 Arm Limited and/or its affiliates.
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

from torchaudio.models import Conformer  # type: ignore[import-untyped]

input_t = Tuple[torch.Tensor, torch.IntTensor]  # Input x, y


def get_test_inputs(dim, lengths, num_examples):
    return (torch.rand(num_examples, int(lengths.max()), dim), lengths)


class TestConformer:
    """Tests Torchaudio Conformer"""

    # Adjust nbr below as we increase op support. Note: most of the delegates
    # calls are directly consecutive to each other in the .pte. The reason
    # for that is some assert ops are removed by passes in the
    # .to_executorch step, i.e. after Arm partitioner.
    aten_ops = ["torch.ops.aten._assert_scalar.default"]

    # TODO(MLETORCH-635): reduce tolerance
    atol = 0.4
    rtol = 0.4

    dim = 16
    num_examples = 10
    lengths = torch.randint(1, 100, (num_examples,), dtype=torch.int32)
    model_example_inputs = get_test_inputs(dim, lengths, num_examples)
    conformer = Conformer(
        input_dim=dim,
        num_heads=4,
        ffn_dim=64,
        num_layers=2,
        depthwise_conv_kernel_size=31,
    )
    conformer = conformer.eval()


def test_conformer_tosa_FP():
    pipeline = TosaPipelineFP[input_t](
        TestConformer.conformer,
        TestConformer.model_example_inputs,
        aten_op=TestConformer.aten_ops,
        exir_op=[],
        use_to_edge_transform_and_lower=True,
    )
    pipeline.run()


def test_conformer_tosa_INT():
    pipeline = TosaPipelineINT[input_t](
        TestConformer.conformer,
        TestConformer.model_example_inputs,
        aten_op=[],
        exir_op=[],
        use_to_edge_transform_and_lower=True,
        atol=TestConformer.atol,
        rtol=TestConformer.rtol,
    )
    pipeline.run()


@common.XfailIfNoCorstone300
@pytest.mark.xfail(
    reason="TODO(MLETORCH-635): Expected failure under FVP option, but test passed."
)
def test_conformer_u55_INT():
    pipeline = EthosU55PipelineINT[input_t](
        TestConformer.conformer,
        TestConformer.model_example_inputs,
        aten_ops=[],
        exir_ops=[],
        use_to_edge_transform_and_lower=True,
        atol=TestConformer.atol,
        rtol=TestConformer.rtol,
    )
    pipeline.pop_stage("check_count.exir")
    pipeline.run()


@common.XfailIfNoCorstone320
def test_conformer_u85_INT():
    pipeline = EthosU85PipelineINT[input_t](
        TestConformer.conformer,
        TestConformer.model_example_inputs,
        aten_ops=[],
        exir_ops=[],
        use_to_edge_transform_and_lower=True,
        atol=TestConformer.atol,
        rtol=TestConformer.rtol,
    )
    pipeline.run()


@common.SkipIfNoModelConverter
def test_conformer_vgf_quant():
    pipeline = VgfPipeline[input_t](
        TestConformer.conformer,
        TestConformer.model_example_inputs,
        aten_op=[],
        exir_op=[],
        use_to_edge_transform_and_lower=True,
        atol=TestConformer.atol,
        rtol=TestConformer.rtol,
        quantize=True,
    )
    pipeline.run()


@common.SkipIfNoModelConverter
def test_conformer_vgf_no_quant():
    pipeline = VgfPipeline[input_t](
        TestConformer.conformer,
        TestConformer.model_example_inputs,
        aten_op=TestConformer.aten_ops,
        exir_op=[],
        use_to_edge_transform_and_lower=True,
        quantize=False,
    )
    pipeline.run()
