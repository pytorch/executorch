# Copyright 2025 Arm Limited and/or its affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from typing import Tuple

import pytest

import torch

from executorch.backends.arm.test import common
from executorch.backends.arm.test.tester.test_pipeline import (
    EthosU55PipelineBI,
    EthosU85PipelineBI,
    TosaPipelineBI,
    TosaPipelineMI,
)

from torchaudio.models import Conformer

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


def test_conformer_tosa_MI():
    pipeline = TosaPipelineMI[input_t](
        TestConformer.conformer,
        TestConformer.model_example_inputs,
        aten_op=TestConformer.aten_ops,
        exir_op=[],
        use_to_edge_transform_and_lower=True,
    )
    pipeline.run()


def test_conformer_tosa_BI():
    pipeline = TosaPipelineBI[input_t](
        TestConformer.conformer,
        TestConformer.model_example_inputs,
        aten_op=TestConformer.aten_ops,
        exir_op=[],
        use_to_edge_transform_and_lower=True,
    )
    pipeline.pop_stage("check_count.exir")
    pipeline.change_args(
        "run_method_and_compare_outputs",
        get_test_inputs(
            TestConformer.dim, TestConformer.lengths, TestConformer.num_examples
        ),
        rtol=1.0,
        atol=3.0,
    )
    pipeline.run()


@common.XfailIfNoCorstone300
@pytest.mark.xfail(
    reason="TODO(MLETORCH-635): Expected failure under FVP option, but test passed."
)
def test_conformer_u55_BI():
    pipeline = EthosU55PipelineBI[input_t](
        TestConformer.conformer,
        TestConformer.model_example_inputs,
        aten_ops=TestConformer.aten_ops,
        exir_ops=[],
        use_to_edge_transform_and_lower=True,
        run_on_fvp=True,
    )
    pipeline.change_args(
        "run_method_and_compare_outputs",
        get_test_inputs(
            TestConformer.dim, TestConformer.lengths, TestConformer.num_examples
        ),
        rtol=1.0,
        atol=5.0,
    )
    pipeline.run()


@common.XfailIfNoCorstone320
@pytest.mark.xfail(reason="All IO needs to have the same data type (MLETORCH-635)")
def test_conformer_u85_BI():
    pipeline = EthosU85PipelineBI[input_t](
        TestConformer.conformer,
        TestConformer.model_example_inputs,
        aten_ops=TestConformer.aten_ops,
        exir_ops=[],
        use_to_edge_transform_and_lower=True,
        run_on_fvp=True,
    )
    pipeline.change_args(
        "run_method_and_compare_outputs",
        get_test_inputs(
            TestConformer.dim, TestConformer.lengths, TestConformer.num_examples
        ),
        rtol=1.0,
        atol=5.0,
    )
    pipeline.run()
