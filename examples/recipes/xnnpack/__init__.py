# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.


from dataclasses import dataclass


@dataclass
class OptimizationOptions(object):
    xnnpack_quantization: bool
    xnnpack_delegation: bool


MODEL_NAME_TO_OPTIONS = {
    "linear": OptimizationOptions(True, True),
    "add": OptimizationOptions(True, True),
    "add_mul": OptimizationOptions(True, True),
    "dl3": OptimizationOptions(True, True),
    "ic3": OptimizationOptions(True, False),
    "ic4": OptimizationOptions(True, False),
    "mv2": OptimizationOptions(True, True),
    "mv3": OptimizationOptions(True, True),
    "resnet18": OptimizationOptions(True, True),
    "resnet50": OptimizationOptions(True, True),
    "vit": OptimizationOptions(False, True),
    "w2l": OptimizationOptions(False, True),
    "edsr": OptimizationOptions(True, False),
    "mobilebert": OptimizationOptions(True, False),
}

__all__ = [MODEL_NAME_TO_OPTIONS]
