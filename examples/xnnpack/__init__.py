# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from dataclasses import dataclass


@dataclass
class XNNPackOptions(object):
    quantization: bool
    delegation: bool


MODEL_NAME_TO_OPTIONS = {
    "linear": XNNPackOptions(True, True),
    "add": XNNPackOptions(True, True),
    "add_mul": XNNPackOptions(True, True),
    "dl3": XNNPackOptions(True, True),
    "ic3": XNNPackOptions(True, False),
    "ic4": XNNPackOptions(True, False),
    "mv2": XNNPackOptions(True, True),
    "mv3": XNNPackOptions(True, True),
    "resnet18": XNNPackOptions(True, True),
    "resnet50": XNNPackOptions(True, True),
    "vit": XNNPackOptions(False, True),
    "w2l": XNNPackOptions(False, True),
    "edsr": XNNPackOptions(True, False),
    "mobilebert": XNNPackOptions(True, False),
}


__all__ = [MODEL_NAME_TO_OPTIONS]
