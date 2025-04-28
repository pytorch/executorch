# Copyright 2025 Arm Limited and/or its affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from typing import Tuple

import torch
from executorch.backends.arm._passes.remove_clone_pass import RemoveClonePass

from executorch.backends.arm.test.tester.test_pipeline import PassPipeline

input_t = Tuple[torch.Tensor]  # Input x


class Clone(torch.nn.Module):
    """
    Basic remove layer model to test RemoveClonePass
    """

    def __init__(self):
        super(Clone, self).__init__()

    def forward(self, x):
        return torch.clone(x)

    def get_inputs(self) -> input_t:
        return (torch.rand(3, 1),)


def test_remove_clone_tosa_BI():
    module = Clone()
    pipeline = PassPipeline[input_t](
        module,
        module.get_inputs(),
        tosa_version="TOSA-0.80+BI",
        ops_before_pass={
            "executorch_exir_dialects_edge__ops_aten_clone_default": 1,
        },
        ops_not_after_pass=["executorch_exir_dialects_edge__ops_aten_clone_default"],
        pass_list=[RemoveClonePass],
    )
    pipeline.run()
