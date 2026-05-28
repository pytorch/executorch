# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# pyre-unsafe
# LICENSE file in the root directory of this source tree.

from torch import nn

from .convert_weights import convert_hf_to_custom  # noqa: F401
from .gemma4_attention import (  # noqa: F401
    apply_rotary_emb,
    apply_rotary_emb_single,
    Gemma4KVCache,
    rotate_half,
)
from .gemma4_config import Gemma4Config  # noqa: F401
from .gemma4_decoder_layer import Gemma4MLP  # noqa: F401
from .gemma4_model import create_gemma4_model, Gemma4Model  # noqa: F401

RMSNorm = nn.RMSNorm


class RMSNormNoWeight(nn.RMSNorm):
    def __init__(self, normalized_shape, eps=1e-5, **kwargs):
        super().__init__(
            normalized_shape,
            eps=eps,
            elementwise_affine=False,
            **kwargs,
        )
