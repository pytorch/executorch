# Copyright 2026 Arm Limited and/or its affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.
"""Full-delegation regression for Conv1d -> activation -> Conv1d on
Ethos-U85.
"""

import pytest
import torch

from executorch.backends.arm.test import common
from executorch.backends.arm.test.tester.test_pipeline import EthosU85PipelineINT
from executorch.backends.test.harness.stages import StageType


@pytest.mark.parametrize(
    "activation",
    [torch.nn.SiLU, torch.nn.ELU, torch.nn.Mish],
    ids=["SiLU", "ELU", "Mish"],
)
@common.XfailIfNoCorstone320
def test_u85_conv1d_activation_conv1d_single_delegate(activation):
    """Require one INT8 delegate and no computational CPU fallbacks."""
    model = torch.nn.Sequential(
        torch.nn.Conv1d(16, 16, 3, padding=1),
        activation(),
        torch.nn.Conv1d(16, 16, 3, padding=1),
    ).eval()
    inputs = (torch.randn(1, 16, 32),)

    pipeline = EthosU85PipelineINT(
        model,
        inputs,
        aten_ops=[],
        exir_ops=[],
    )

    # The standard pipeline checks that there is exactly one delegate.
    # Also check for unsupported operations left on the CPU (the original
    # issue had two delegates with Mish's exp/log1p/tanh/where in between).
    def assert_no_cpu_fallback():
        edge = pipeline.tester.get_artifact(StageType.TO_EDGE_TRANSFORM_AND_LOWER)
        op_names = [
            str(node.target)
            for node in edge.exported_program().graph.nodes
            if node.op == "call_function"
        ]
        skipped = (
            "executorch_call_delegate",
            "getitem",
            "quantize_per",
            "dequantize_per",
        )
        cpu_ops = [
            name for name in op_names if not any(token in name for token in skipped)
        ]
        assert not cpu_ops, (activation.__name__, cpu_ops)

    pipeline.add_stage_after(
        "to_edge_transform_and_lower",
        assert_no_cpu_fallback,
    )
    pipeline.run()
