# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-unsafe

from dataclasses import dataclass


@dataclass
class XNNPACKOptions(object):
    quantization: bool
    delegation: bool


MODEL_NAME_TO_OPTIONS = {
    "linear": XNNPACKOptions(True, True),
    "add": XNNPACKOptions(True, True),
    "add_mul": XNNPACKOptions(True, True),
    "dl3": XNNPACKOptions(True, True),
    "ic3": XNNPACKOptions(True, True),
    "ic4": XNNPACKOptions(True, True),
    "mv2": XNNPACKOptions(True, True),
    "mv3": XNNPACKOptions(True, True),
    "resnet18": XNNPACKOptions(True, True),
    "resnet50": XNNPACKOptions(True, True),
    "vit": XNNPACKOptions(True, True),
    "w2l": XNNPACKOptions(True, True),
    "edsr": XNNPACKOptions(True, True),
    "mobilebert": XNNPACKOptions(True, True),
    "llama2": XNNPACKOptions(False, True),
    "emformer_join": XNNPACKOptions(True, True),
    "emformer_predict": XNNPACKOptions(True, True),
    "emformer_transcribe": XNNPACKOptions(True, True),
}


__all__ = [MODEL_NAME_TO_OPTIONS]
