# Copyright 2026 Arm Limited and/or its affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import torch

from executorch.backends.arm.quantizer import (
    get_uint8_io_quantization_config,
    TOSAQuantizer,
)
from executorch.backends.arm.test import common
from executorch.backends.arm.test.tester.test_pipeline import QuantizationPipeline


class SimpleMLP(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = torch.nn.Linear(4, 8)
        self.relu = torch.nn.ReLU()
        self.fc2 = torch.nn.Linear(8, 4)

    def forward(self, x):
        return self.fc2(self.relu(self.fc1(x)))


def test_uint8_io_quantization_config_tosa_INT_applies_to_io():
    model = SimpleMLP().eval()
    test_data = (torch.rand(1, 4),)
    compile_spec = common.get_tosa_compile_spec("TOSA-1.0+INT")
    quantizer = TOSAQuantizer(compile_spec)
    quantizer.set_io(get_uint8_io_quantization_config())

    io_config = get_uint8_io_quantization_config()
    pipeline = QuantizationPipeline(
        model,
        test_data,
        quantizer=quantizer,
        input_qspecs={io_config.input_activation: 1},
        output_qspecs={io_config.output_activation: 1},
    )
    pipeline.run()
