# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from dataclasses import dataclass


@dataclass
class OptimizationOptions(object):
    quantization: bool = False
    xnnpack_delegation: bool = False


MODEL_NAME_TO_OPTIONS = {
    "linear": OptimizationOptions(True, True),
    "add": OptimizationOptions(True, True),
    "add_mul": OptimizationOptions(True, True),
    "mv2": OptimizationOptions(True, True),
    "mv3": OptimizationOptions(False, True),
}
