# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-unsafe

# Helper functions for tranforming the model to be able to run SpinQuant.
# See https://github.com/facebookresearch/SpinQuant for more details about SpinQuant.


import torch

import torch.nn.functional as F

from executorch.examples.models.llama.feed_forward import FeedForward
from torch import nn


def _inject_fast_hadamard_transform_cuda_for_spin_quant(module: torch.nn.Module):
    """
    SpinQuant needs two Hadmard matrixes: R3 and R4. Here we are only injecting R4 in the feed forward layer.
    R3 needs to be injected as well when KV cache quantization is enabled.
    """
    try:
        from fast_hadamard_transform import hadamard_transform
    except ImportError:
        raise ImportError(
            "Please install fast-hadamard-transform: pip install fast-hadamard-transform"
        )

    class FeedForwardCudaCustom(nn.Module):
        def __init__(self, w1, w2, w3):
            super().__init__()
            self.w1 = w1
            self.w2 = w2
            self.w3 = w3

        def forward(self, x):
            w = F.silu(self.w1(x)) * self.w3(x)
            n = w.shape[-1]
            return self.w2(hadamard_transform(w.contiguous()) / torch.tensor(n).sqrt())

    for name, child in module.named_children():
        if isinstance(child, FeedForward):
            setattr(module, name, FeedForwardCudaCustom(child.w1, child.w2, child.w3))
        else:
            _inject_fast_hadamard_transform_cuda_for_spin_quant(child)


def inject_fast_hadamard_transform_cuda_for_spin_quant(
    module: torch.nn.Module,
) -> torch.nn.Module:
    _inject_fast_hadamard_transform_cuda_for_spin_quant(module)
    return module


def _inject_fast_hadamard_transform_native_for_spin_quant(module: torch.nn.Module):
    """
    SpinQuant needs two Hadmard matrixes: R3 and R4. Here we are only injecting R4 in the feed forward layer.
    R3 needs to be injected as well when KV cache quantization is enabled.
    """

    class FeedForwardNativeCustom(nn.Module):
        def __init__(self, w1, w2, w3):
            super().__init__()
            self.w1 = w1
            self.w2 = w2
            self.w3 = w3

        def forward(self, x):
            return self.w2(
                torch.ops.llama.fast_hadamard_transform(F.silu(self.w1(x)) * self.w3(x))
            )

    for name, child in module.named_children():
        if isinstance(child, FeedForward):
            setattr(module, name, FeedForwardNativeCustom(child.w1, child.w2, child.w3))
        else:
            _inject_fast_hadamard_transform_native_for_spin_quant(child)


def inject_fast_hadamard_transform_native_for_spin_quant(
    module: torch.nn.Module,
) -> torch.nn.Module:
    _inject_fast_hadamard_transform_native_for_spin_quant(module)
    return module
