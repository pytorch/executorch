# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import torch
from executorch.examples.models.llama.norm import Gemma3RMSNorm, RMSNorm
from torch import nn


def replace_rms_norm_with_native_rms_norm(module: torch.nn.Module):
    """Replace custom norm implementations with torch.nn.RMSNorm.

    Handles both standard RMSNorm and Gemma3RMSNorm with appropriate
    weight scaling conversions.
    """
    for name, child in module.named_children():
        if isinstance(child, RMSNorm):
            # Standard RMSNorm: direct replacement
            rms_norm = torch.nn.RMSNorm(child.dim, eps=child.eps)
            rms_norm.weight = child.weight
            setattr(
                module,
                name,
                rms_norm,
            )
        elif isinstance(child, Gemma3RMSNorm):
            # Gemma3RMSNorm: convert weight scaling from (1.0 + w) to w
            rms_norm = torch.nn.RMSNorm(child.dim, eps=child.eps)
            rms_norm.weight = nn.Parameter(1.0 + child.weight)
            setattr(
                module,
                name,
                rms_norm,
            )
        else:
            replace_rms_norm_with_native_rms_norm(child)
    return module
