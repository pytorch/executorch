# Copyright (c) Meta Platforms, Inc. and affiliates.
# Copyright 2026 Arm Limited and/or its affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from typing import Tuple

import torch
from executorch.backends.arm._passes.fuse_quantized_activation_pass import (
    FuseQuantizedActivationPass,
)
from executorch.backends.arm.test.tester.test_pipeline import PassPipeline

input_t = Tuple[torch.Tensor]


class ConvRelu(torch.nn.Module):
    """Conv2d followed by ReLU — existing fuseable behavior."""

    def __init__(self) -> None:
        super().__init__()
        self.conv = torch.nn.Conv2d(3, 3, 3, padding=1)
        self.relu = torch.nn.ReLU()

    def get_inputs(self) -> input_t:
        return (torch.randn(1, 3, 8, 8),)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.relu(self.conv(x))


def test_fuse_relu_after_conv_quantized() -> None:
    """Existing behavior: ReLU after conv is fused in quantized graph."""
    module = ConvRelu()
    pipeline = PassPipeline[input_t](
        module,
        module.get_inputs(),
        quantize=True,
        ops_before_pass={
            "executorch_exir_dialects_edge__ops_aten_relu_default": 1,
        },
        ops_not_after_pass=[
            "executorch_exir_dialects_edge__ops_aten_relu_default",
        ],
        pass_list=[FuseQuantizedActivationPass],
    )
    pipeline.run()
