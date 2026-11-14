# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""RMSNorm cases: LLM hidden and head norms, with and without a weight, rows
whose magnitude stresses eps and the sum of squares, and dynamic shapes."""

import torch
from executorch.backends.native.test.samples.cases import Case, chain, draws, Inputs
from torch.export import Dim


def _norm(dim: int, eps: float = 1e-5, affine: bool = True) -> torch.nn.RMSNorm:
    norm = torch.nn.RMSNorm(dim, eps=eps, elementwise_affine=affine)
    if affine:
        torch.nn.init.uniform_(norm.weight, 0.5, 1.5)
    return norm


def _scaled_rows(inputs: list[Inputs], scales: list[float]) -> list[Inputs]:
    for (x,) in inputs:
        x *= torch.tensor(scales)[:, None]
    return inputs


def cases() -> list[Case]:
    torch.manual_seed(0)
    # One channel far above the rest, as in LLM activations.
    outlier = draws(2, (16, 2048))
    for (x,) in outlier:
        x[:, 3] *= 1000
    tokens, batch = Dim("tokens", min=1, max=64), Dim("batch", min=1, max=4)
    return [
        # A hidden size that is not a power of two.
        Case("decode", _norm(6656), draws(8, (1, 6656))),
        Case("prefill", _norm(2048), draws(2, (64, 2048))),
        Case("no_weight", _norm(6656, affine=False), draws(4, (4, 6656))),
        # Query and key norms over the head dim of [batch, tokens, heads, 128].
        Case("head_norm", _norm(128), draws(4, (1, 16, 8, 128))),
        Case(
            "head_norm_no_weight",
            _norm(128, affine=False),
            draws(4, (1, 16, 8, 128)),
        ),
        # Rows down to a mean square below eps, and a zero row.
        Case(
            "tiny_rows",
            _norm(6656, eps=1e-8),
            _scaled_rows(
                draws(2, (8, 6656)), [1, 1e-2, 1e-3, 3e-4, 1e-4, 3e-5, 1e-5, 0]
            ),
        ),
        # Squares that overflow float16.
        Case(
            "large_rows",
            _norm(2048),
            _scaled_rows(draws(2, (4, 2048)), [1, 1e2, 1e3, 1e4]),
        ),
        Case("outlier_channel", _norm(2048), outlier),
        Case("leading_batch", _norm(512), draws(4, (2, 7, 512))),
        Case("odd_dim", _norm(1001), draws(4, (4, 1001))),
        chain("dynamic_tokens", _norm(1024), (tokens, 1024)),
        chain("dynamic_batch_tokens", _norm(512), (batch, tokens, 512)),
        chain(
            "dynamic_head_norm_no_weight",
            _norm(128, affine=False),
            (1, tokens, 8, 128),
        ),
    ]
