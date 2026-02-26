# Copyright 2026 Arm Limited and/or its affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from typing import Tuple

import pytest
import torch
from executorch.backends.arm.test import common
from executorch.backends.arm.test.tester.test_pipeline import TosaPipelineINT
from executorch.backends.test.harness.stages.stage import StageType

input_t1 = Tuple[torch.Tensor, int]

exir_op = "executorch_exir_dialects_edge__ops_aten_full_default"


class FullLike(torch.nn.Module):
    """Since full_like is replaced with full, we only need to test on reference model, not FVP."""

    test_parameters = {
        "full_like_int_val": lambda: (torch.randn(2, 2, 2, 2) * 50, 3),
        "full_like_float_val": lambda: (torch.randn(2, 4, 5, 2) * 50, 3.2),
    }

    def forward(self, input_tensor: torch.Tensor, value):
        # Our backend can't handle tensors without users, which input_tensor doesn't have
        # when the full_like is converted to a full. Therefore involve it in the output.
        return input_tensor + torch.full_like(
            input_tensor, value, dtype=torch.float32, memory_format=torch.channels_last
        )


@common.parametrize("test_data", FullLike.test_parameters)
def test_preserves_kwargs_tosa_INT(test_data):
    pipeline = TosaPipelineINT[input_t1](
        FullLike(),
        test_data(),
        aten_op=[],
        exir_op=exir_op,
    )
    pipeline.run()

    # Test that kwarg memory_format survived quantization.
    graph_module = pipeline.tester.get_artifact(StageType.EXPORT).graph_module
    nodes = graph_module.graph.nodes
    for n in nodes:
        if n.target == torch.ops.aten.full_like.default:
            assert n.meta["val"].dim_order() == (0, 2, 3, 1)
            break
    else:
        pytest.fail("Did not find torch.ops.aten.full_like")
