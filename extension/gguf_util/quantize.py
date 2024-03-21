# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from executorch.extension.gguf_util.load_gguf import GGUFModelArgs
from torch import nn

# TODO (mnachin): Move this file to torchao


def quantize(
    pt_model: nn.Module,
    gguf_model_args: GGUFModelArgs,
) -> nn.Module:
    # TODO (mnachin): Implement quantization
    return pt_model
