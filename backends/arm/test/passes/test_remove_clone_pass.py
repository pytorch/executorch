# Copyright 2025 Arm Limited and/or its affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from typing import Tuple

import torch
from executorch.backends.arm._passes.remove_noop_pass import RemoveNoopPass

from executorch.backends.arm.test.tester.test_pipeline import PassPipeline

input_t = Tuple[torch.Tensor]  # Input x


class Clone(torch.nn.Module):
    """
    Basic remove layer model to test RemoveNoopePass
    """

    def __init__(self):
        super(Clone, self).__init__()

    def forward(self, x):
        return torch.clone(x)

    def get_inputs(self) -> input_t:
        return (torch.rand(3, 1),)


def test_remove_noop_tosa_INT_clone():
    module = Clone()
    pipeline = PassPipeline[input_t](
        module,
        module.get_inputs(),
        quantize=True,
        ops_before_pass={
            "executorch_exir_dialects_edge__ops_dim_order_ops__clone_dim_order_default": 1,
        },
        ops_not_after_pass=[
            "executorch_exir_dialects_edge__ops_dim_order_ops__clone_dim_order_default"
        ],
        pass_list=[RemoveNoopPass],
    )
    pipeline.run()
