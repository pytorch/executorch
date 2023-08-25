# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# @file models.py
# Simple models for demonstration purposes.

from dataclasses import dataclass

from typing import Any, Tuple

import torch
from executorch.exir.backend.compile_spec_schema import CompileSpec


class MulModule(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()

    def forward(self, input, other):
        return input * other

    @staticmethod
    def get_example_inputs():
        return (torch.randn(3, 2), torch.randn(3, 2))


class LinearModule(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(3, 3)

    def forward(self, arg):
        return self.linear(arg)

    @staticmethod
    def get_example_inputs():
        return (torch.randn(3, 3),)


class AddModule(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x, y):
        z = x + y
        z = z + x
        z = z + x
        z = z + z
        return z

    @staticmethod
    def get_example_inputs():
        return (torch.ones(1), torch.ones(1))


class AddMulModule(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, a, x, b):
        y = torch.mm(a, x)
        z = torch.add(y, b)
        return z

    @staticmethod
    def get_example_inputs():
        return (torch.ones(2, 2), 2 * torch.ones(2, 2), 3 * torch.ones(2, 2))

    def get_compile_spec(self):
        max_value = self.get_example_inputs()[0].shape[0]
        return [CompileSpec("max_value", bytes([max_value]))]


MODEL_NAME_TO_MODEL = {
    "mul": lambda: (MulModule(), MulModule.get_example_inputs()),
    "linear": lambda: (LinearModule(), LinearModule.get_example_inputs()),
    "add": lambda: (AddModule(), AddModule.get_example_inputs()),
    "add_mul": lambda: (AddMulModule(), AddMulModule.get_example_inputs()),
    "mv2": ("mobilenet_v2", "MV2Model"),
    "mv3": ("mobilenet_v3", "MV3Model"),
    "vit": ("torchvision_vit", "TorchVisionViTModel"),
    "w2l": ("wav2letter", "Wav2LetterModel"),
    "ic3": ("inception_v3", "InceptionV3Model"),
    "ic4": ("inception_v4", "InceptionV4Model"),
    "resnet18": ("resnet", "ResNet18Model"),
    "resnet50": ("resnet", "ResNet50Model"),
}


@dataclass
class OptimizationOptions(object):
    quantization: bool
    xnnpack_delegation: bool


MODEL_NAME_TO_OPTIONS = {
    "linear": OptimizationOptions(True, True),
    "add": OptimizationOptions(True, True),
    "add_mul": OptimizationOptions(True, True),
    "mv2": OptimizationOptions(True, True),
}
