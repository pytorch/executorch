# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import torch
from executorch.examples.models.llama2.llama_transformer import RMSNorm


def replace_rms_norm_with_native_rms_norm(module: torch.nn.Module):
    for name, child in module.named_children():
        if isinstance(child, RMSNorm):
            rms_norm = torch.nn.RMSNorm(child.dim, eps=child.eps)
            rms_norm.weight = child.weight
            setattr(
                module,
                name,
                rms_norm,
            )
        else:
            replace_rms_norm_with_native_rms_norm(child)
    return module
