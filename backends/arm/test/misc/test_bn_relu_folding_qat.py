# Copyright 2025 Arm Limited and/or its affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from typing import Tuple

import torch
import torch.nn.functional as F
from executorch.backends.arm.quantizer.arm_quantizer import (
    get_symmetric_quantization_config,
    TOSAQuantizer,
)
from executorch.backends.arm.test import common, conftest
from executorch.backends.arm.test.tester.test_pipeline import TosaPipelineBI

from executorch.backends.xnnpack.test.tester.tester import Quantize
from torch import nn


input_t1 = Tuple[torch.Tensor]  # Input x


class ConvModule(torch.nn.Module):
    input_shape = (1, 28, 28)
    batch_size = 64
    test_data: input_t1 = (torch.randn(batch_size, *input_shape),)

    def __init__(self, batch_norm: bool = True) -> None:
        super().__init__()
        self.conv = torch.nn.Conv2d(1, 16, 3, stride=2)
        self.bn = nn.BatchNorm2d(num_features=16) if batch_norm else nn.Identity()

    def forward(self, x: torch.Tensor):
        x = self.conv(x)
        x = self.bn(x)
        x = F.relu(x)

        return x


models = {
    "conv_bn_relu": ConvModule(batch_norm=True),
    "conv_relu": ConvModule(batch_norm=False),
}


@common.parametrize("model", models)
def test_qat_tosa_BI(model: torch.nn.Module):
    pipeline = TosaPipelineBI[input_t1](model, model.test_data, [], [], qtol=1)
    tosa_version = conftest.get_option("tosa_version")
    tosa_profiles = {
        "0.80": common.TosaSpecification.create_from_string("TOSA-0.80+BI"),
        "1.0": common.TosaSpecification.create_from_string("TOSA-1.0+INT"),
    }
    tosa_spec = tosa_profiles[tosa_version]
    quantizer = TOSAQuantizer(tosa_spec)
    pipeline.change_args(
        "quantize",
        Quantize(
            quantizer=quantizer,
            quantization_config=get_symmetric_quantization_config(is_qat=True),
            is_qat=True,
        ),
    )
    pipeline.run()
