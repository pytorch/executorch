# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import executorch.backends.vulkan.custom_ops_lib  # noqa
import torch

from executorch.examples.models.llama.rope import RotaryEmbedding


class VkRotaryEmbedding(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(
        self,
        xq: torch.Tensor,
        xk: torch.Tensor,
        freqs_cos: torch.Tensor,
        freqs_sin: torch.Tensor,
    ):
        xq_out, xk_out = torch.ops.et_vk.apply_rotary_emb(xq, xk, freqs_cos, freqs_sin)
        return xq_out, xk_out


def replace_with_vulkan_rotary_emb(module: torch.nn.Module):
    for name, child in module.named_children():
        if isinstance(child, RotaryEmbedding):
            new_module = VkRotaryEmbedding()
            setattr(module, name, new_module)
        else:
            replace_with_vulkan_rotary_emb(child)

    return module
