# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from .quantization_configs import (  # noqa
    CMSIS_SOFTMAX_SCALE,
    CMSIS_SOFTMAX_ZERO_POINT,
    INT8_ACTIVATION_PER_CHANNEL_QSPEC,
    INT8_ACTIVATION_PER_TENSOR_QSPEC,
    INT8_PER_CHANNEL_CONFIG,
    INT8_PER_TENSOR_CONFIG,
    INT8_WEIGHT_PER_CHANNEL_QSPEC,
    INT8_WEIGHT_PER_TENSOR_QSPEC,
    SOFTMAX_OUTPUT_FIXED_QSPEC,
    SOFTMAX_PER_TENSOR_CONFIG,
)
from .quantizer import CortexMQuantizer, SharedQspecQuantizer  # noqa
