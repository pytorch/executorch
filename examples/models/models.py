# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# @file models.py
# Simple models for demonstration purposes.

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


def gen_mobilenet_v3_model_inputs() -> Tuple[torch.nn.Module, Any]:
    # Unfortunately lack of consistent interface on example models in this file
    # and how we obtain oss models result in changes like this.
    # we should probably fix this if all the MVP model's export example
    # wiil be added here.
    # For now, to unblock, not planning to land those changes in the current diff
    from ..models.mobilenet_v3 import MV3Model

    return MV3Model.get_model(), MV3Model.get_example_inputs()


def gen_mobilenet_v2_model_inputs() -> Tuple[torch.nn.Module, Any]:
    from ..models.mobilenet_v2 import MV2Model

    return MV2Model.get_model(), MV2Model.get_example_inputs()


def gen_torchvision_vit_model_and_inputs() -> Tuple[torch.nn.Module, Any]:
    from ..models.torchvision_vit import TorchVisionViTModel

    return TorchVisionViTModel.get_model(), TorchVisionViTModel.get_example_inputs()


def gen_wav2letter_model_and_inputs() -> Tuple[torch.nn.Module, Any]:
    from ..models.wav2letter import Wav2LetterModel

    model = Wav2LetterModel()
    return model.get_model(), model.get_example_inputs()

def gen_llama2_model_and_inputs() -> Tuple[torch.nn.Module, Any]:
    from ..models.llama2 import LLAMA2Model

    model = LLAMA2Model()
    return model.get_model(), model.get_example_inputs()

MODEL_NAME_TO_MODEL = {
    "mul": lambda: (MulModule(), MulModule.get_example_inputs()),
    "linear": lambda: (LinearModule(), LinearModule.get_example_inputs()),
    "add": lambda: (AddModule(), AddModule.get_example_inputs()),
    "add_mul": lambda: (AddMulModule(), AddMulModule.get_example_inputs()),
    "mv2": gen_mobilenet_v2_model_inputs,
    "mv3": gen_mobilenet_v3_model_inputs,
    "vit": gen_torchvision_vit_model_and_inputs,
    "w2l": gen_wav2letter_model_and_inputs,
    "llama2": gen_llama2_model_and_inputs,
}
