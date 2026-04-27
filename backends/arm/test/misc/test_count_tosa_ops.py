# Copyright 2026 Arm Limited and/or its affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import pytest
import torch

from executorch.backends.arm.test.tester.test_pipeline import TosaPipelineFP


class AddModule(torch.nn.Module):
    def forward(self, x, y):
        return x + y


def test_count_tosa_ops_add():
    model = AddModule()
    test_data = (torch.randn(1, 8, 8, 8), torch.randn(1, 8, 8, 8))
    pipeline = TosaPipelineFP[type(test_data)](
        model,
        test_data,
        ["torch.ops.aten.add.Tensor"],
        run_on_tosa_ref_model=False,
    )
    pipeline.count_tosa_ops({"ADD": 1, "SUB": 0})
    pipeline.run()


def test_count_tosa_ops_2_adds():
    model = AddModule()
    test_data = (torch.randn(1, 8, 8, 8), torch.randn(1, 8, 8, 8))
    pipeline = TosaPipelineFP[type(test_data)](
        model,
        test_data,
        ["torch.ops.aten.add.Tensor"],
        run_on_tosa_ref_model=False,
    )
    pipeline.count_tosa_ops({"ADD": 2})
    with pytest.raises(AssertionError, match="Expected 2 occurrences of TOSA op ADD"):
        pipeline.run()
