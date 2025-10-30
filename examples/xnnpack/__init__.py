# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-unsafe

from dataclasses import dataclass
from enum import Enum


class QuantType(Enum):
    NONE = 1
    # Used for Operations that don't have weights
    STATIC_PER_TENSOR = 2
    # Used best for CNN/RNN Models with Conv layers
    STATIC_PER_CHANNEL = 3
    # Used for Linear Layers and Transformer Based Models
    DYNAMIC_PER_CHANNEL = 4


@dataclass
class XNNPACKOptions(object):
    quantization: QuantType
    delegation: bool


MODEL_NAME_TO_OPTIONS = {
    "linear": XNNPACKOptions(QuantType.STATIC_PER_CHANNEL, True),
    "add": XNNPACKOptions(QuantType.STATIC_PER_TENSOR, True),
    "add_mul": XNNPACKOptions(QuantType.STATIC_PER_TENSOR, True),
    "dl3": XNNPACKOptions(QuantType.STATIC_PER_CHANNEL, True),
    "ic3": XNNPACKOptions(QuantType.STATIC_PER_CHANNEL, True),
    "ic4": XNNPACKOptions(QuantType.STATIC_PER_CHANNEL, True),
    "mv2": XNNPACKOptions(QuantType.STATIC_PER_CHANNEL, True),
    "mv3": XNNPACKOptions(QuantType.STATIC_PER_CHANNEL, True),
    "resnet18": XNNPACKOptions(QuantType.STATIC_PER_CHANNEL, True),
    "resnet50": XNNPACKOptions(QuantType.STATIC_PER_CHANNEL, True),
    "vit": XNNPACKOptions(QuantType.DYNAMIC_PER_CHANNEL, True),
    "w2l": XNNPACKOptions(QuantType.DYNAMIC_PER_CHANNEL, True),
    "edsr": XNNPACKOptions(QuantType.STATIC_PER_CHANNEL, True),
    "mobilebert": XNNPACKOptions(QuantType.DYNAMIC_PER_CHANNEL, True),
    "llama2": XNNPACKOptions(QuantType.DYNAMIC_PER_CHANNEL, True),
    "emformer_join": XNNPACKOptions(QuantType.DYNAMIC_PER_CHANNEL, True),
    "emformer_predict": XNNPACKOptions(QuantType.DYNAMIC_PER_CHANNEL, True),
    "emformer_transcribe": XNNPACKOptions(QuantType.STATIC_PER_CHANNEL, True),
}


__all__ = [MODEL_NAME_TO_OPTIONS]
