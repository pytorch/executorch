# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-unsafe

import torch

import torch.nn.functional as F

from executorch.examples.models.llama2.llama_transformer import FeedForward
from torch import nn


class FeedForwardCustom(nn.Module):
    def __init__(self, w1, w2, w3):
        super().__init__()
        self.w1 = w1
        self.w2 = w2
        self.w3 = w3

    def forward(self, x):
        return self.w2(torch.op.llama.fht(F.silu(self.w1(x)) * self.w3(x)))


def _replace_feed_forward_with_custom(module: torch.nn.Module):
    for name, child in module.named_children():
        if isinstance(child, FeedForward):
            setattr(module, name, FeedForwardCustom(child.w1, child.w2, child.w3))
        else:
            _replace_feed_forward_with_custom(child)


def replace_feed_forward_with_custom(module: torch.nn.Module) -> torch.nn.Module:
    from executorch.extension.llm.custom_ops import fht  # noqa

    _replace_feed_forward_with_custom(module)
    return module
