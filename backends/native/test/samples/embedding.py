# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Embedding cases: the backend test suite's, LLM vocabularies and token
shapes, index edge cases, and dynamic shapes."""

from functools import partial

import torch
from executorch.backends.native.test.samples.cases import (
    Case,
    chain,
    draws,
    Parallel,
    Sampler,
    suite_cases,
)
from executorch.backends.native.test.samples.schemes import GGUF_SUPER_SCALES
from executorch.backends.test.suite.operators import test_embedding
from torch.export import Dim


def _index_sampler(vocab: int) -> Sampler:
    """A sampler of uniform int64 indices into a `vocab`-row table."""
    return partial(torch.randint, vocab)


def cases() -> list[Case]:
    suite = suite_cases(test_embedding, samples=4)
    torch.manual_seed(0)
    tokens, batch = Dim("tokens", min=1, max=64), Dim("batch", min=1, max=4)
    return [
        *suite,
        Case(
            "decode",
            torch.nn.Embedding(4096, 256),
            draws(8, (1,), sampler=_index_sampler(4096)),
        ),
        Case(
            "prefill",
            torch.nn.Embedding(4096, 256),
            draws(2, (64,), sampler=_index_sampler(4096)),
        ),
        Case(
            "wide",
            torch.nn.Embedding(512, 2048),
            draws(4, (4,), sampler=_index_sampler(512)),
        ),
        Case(
            "leading_batch",
            torch.nn.Embedding(1000, 256),
            draws(4, (2, 7), sampler=_index_sampler(1000)),
        ),
        Case("scalar_index", torch.nn.Embedding(64, 256), [(torch.tensor(5),)]),
        Case(
            "edge_rows",
            torch.nn.Embedding(512, 256),
            [(torch.tensor([0, 511, 0, 511, 7, 7]),)],
        ),
        Case(
            "int32_indices",
            torch.nn.Embedding(512, 256),
            [(x.int(),) for (x,) in draws(2, (16,), sampler=_index_sampler(512))],
        ),
        # A zero row 0, and padding_idx in the graph, in the float and torchao
        # packages.
        Case(
            "padding_idx",
            torch.nn.Embedding(512, 256, padding_idx=0),
            [(torch.tensor([0, 3, 0, 9]),)],
        ),
        # One output per family of GGUF super-block scales.
        Case(
            "scale_families",
            Parallel(*(torch.nn.Embedding(64, 256) for _ in GGUF_SUPER_SCALES["q4_k"])),
            draws(2, (8,), sampler=_index_sampler(64)),
        ),
        chain(
            "dynamic_tokens",
            torch.nn.Embedding(4096, 256),
            (tokens,),
            sampler=_index_sampler(4096),
        ),
        chain(
            "dynamic_batch_tokens",
            torch.nn.Embedding(1000, 256),
            (batch, tokens),
            sampler=_index_sampler(1000),
        ),
    ]
