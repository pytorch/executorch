# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import torch

from ..llama_transformer import Transformer


def materialze_broadcast_of_rope_freq_cis(
    module: torch.nn.Module,
):
    assert isinstance(module, Transformer)
    assert module.rope.freqs_cos.dim() == 2
    dim0 = module.rope.freqs_cos.size(0)
    dim1 = module.rope.freqs_cos.size(1)
    module_attention = module.layers[0].attention
    assert (
        module_attention.n_local_kv_heads == module_attention.n_local_heads
    ), f"For rope freqs to be materialized for broadcast, q, k, v num heads must match. For q got {module_attention.n_kv_heads} for k got {module_attention.n_local_heads} and v got {module_attention.n_local_kv_heads}"
    num_heads = module_attention.n_local_heads
    module.rope.freqs_cos = module.rope.freqs_cos.view(dim0, 1, dim1)
    module.rope.freqs_cos = module.rope.freqs_cos.expand(
        dim0, num_heads, dim1
    ).contiguous()
    assert module.rope.freqs_sin.dim() == 2
    assert dim0 == module.rope.freqs_sin.size(
        0
    ), f"sin and cos freq table sizes must match. Mismatch found at dim 0: {dim0} vs {module.rope.freqs_sin.size(0)}"
    assert dim1 == module.rope.freqs_sin.size(
        1
    ), f"sin and cos freq table sizes must match. Mismatch found at dim 1: {dim1} vs {module.rope.freqs_sin.size(1)}"
    module.rope.freqs_sin = module.rope.freqs_sin.view(dim0, 1, dim1)
    module.rope.freqs_sin = module.rope.freqs_sin.expand(
        dim0, num_heads, dim1
    ).contiguous()
    return module
