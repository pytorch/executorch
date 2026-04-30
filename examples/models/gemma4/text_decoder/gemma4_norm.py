# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# pyre-unsafe
# LICENSE file in the root directory of this source tree.

from functools import partial

from transformers.models.gemma4.modeling_gemma4 import Gemma4RMSNorm as RMSNorm

# V-norm in attention uses RMSNorm without learnable weight.
RMSNormNoWeight = partial(RMSNorm, with_scale=False)
