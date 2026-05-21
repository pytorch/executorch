# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# pyre-unsafe
# LICENSE file in the root directory of this source tree.

"""Gemma 4 RMSNorm — thin wrapper over ``torch.nn.RMSNorm``."""

from functools import partial

from torch import nn


class RMSNorm(nn.RMSNorm):
    """Gemma 4 RMSNorm: ``y = (x / rms(x)) * weight`` (or no weight).

    Pass ``with_scale=False`` for the v-norm (and the unused router norm),
    which omit the learnable weight entirely.
    """

    def __init__(self, dim: int, eps: float = 1e-6, with_scale: bool = True):
        super().__init__(dim, eps=eps, elementwise_affine=with_scale)


# V-norm in attention uses RMSNorm without learnable weight.
RMSNormNoWeight = partial(RMSNorm, with_scale=False)
