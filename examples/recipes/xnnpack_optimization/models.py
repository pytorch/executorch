# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from dataclasses import dataclass


@dataclass
class OptimizationOptions(object):
    quantization: bool
    xnnpack_delegation: bool


MODEL_NAME_TO_OPTIONS = {
    "linear": OptimizationOptions(True, True),
    "add": OptimizationOptions(True, True),
    "add_mul": OptimizationOptions(True, True),
    "mv2": OptimizationOptions(True, True),
    "mv3": OptimizationOptions(False, True),
    "ic3": OptimizationOptions(True, False),
    "ic4": OptimizationOptions(
        True, False
    ),  # TODO[T163161310]: takes a long time to export to exec prog and save inception_v4 quantized model
    "w2l": OptimizationOptions(False, True),
    "dl3": OptimizationOptions(True, False),
    "vit": OptimizationOptions(True, False),
}
