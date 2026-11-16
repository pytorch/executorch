# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Linear cases: the backend test suite's, FACTO's, shapes and inputs that
stress accumulation and quantization, and dynamic shapes."""

import torch
from executorch.backends.native.test.samples.cases import (
    Case,
    chain,
    draws,
    facto_cases,
    suite_cases,
)
from executorch.backends.native.test.samples.schemes import GGUF_SUPER_SCALES
from executorch.backends.test.suite.operators import test_linear
from executorch.extension.llm.export.gguf import QK_K
from facto.inputgen.argument.type import ArgType
from facto.inputgen.specs.model import ConstraintProducer as cp, InPosArg, OutArg, Spec
from torch.export import Dim


def _unit(*constraints) -> list:
    """Float32 values in [-1, 1] under `constraints`."""
    return [
        cp.Dtype.Eq(lambda deps: torch.float32),
        *constraints,
        cp.Value.Ge(lambda deps, dtype, struct: -1),
        cp.Value.Le(lambda deps, dtype, struct: 1),
    ]


# in_features is a multiple of every torchao scheme's group size and, in some
# tuples, of GGUF's block; out_features is large enough to fit an error's
# abs_bias and rel_bias from one sample.
_FACTO_SPEC = Spec(
    op="linear.default",
    inspec=[
        InPosArg(
            ArgType.Tensor,
            name="input",
            deps=[1],
            constraints=_unit(
                cp.Rank.Ge(lambda deps: 1),
                cp.Rank.Le(lambda deps: 3),
                cp.Size.In(
                    lambda deps, r, d: [deps[0].size(1)] if d == r - 1 else range(1, 9)
                ),
            ),
        ),
        InPosArg(
            ArgType.Tensor,
            name="weight",
            constraints=_unit(
                cp.Rank.Eq(lambda deps: 2),
                cp.Size.In(lambda deps, r, d: [128, 256, 384] if d else [16, 48, 96]),
            ),
        ),
        InPosArg(
            ArgType.TensorOpt,
            name="bias",
            deps=[1],
            constraints=_unit(
                cp.Rank.Eq(lambda deps: 1),
                cp.Size.Eq(lambda deps, r, d: deps[0].size(0)),
            ),
        ),
    ],
    outspec=[OutArg(ArgType.Tensor)],
)


def _facto_case(name: str, x, weight, bias) -> Case:
    model = torch.nn.Linear(weight.size(1), weight.size(0), bias=bias is not None)
    model.weight.data = weight
    if bias is not None:
        model.bias.data = bias
    return Case(name, model, [(x,)])


class _Parallel(torch.nn.Module):
    """nn.Linears on one input, each giving its own output."""

    def __init__(self, *linears: torch.nn.Linear) -> None:
        super().__init__()
        self.linears = torch.nn.ModuleList(linears)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, ...]:
        return tuple(m(x) for m in self.linears)


class _Pair(torch.nn.Module):
    """Two modules, each on its own input."""

    def __init__(self, first: torch.nn.Module, second: torch.nn.Module) -> None:
        super().__init__()
        self.first, self.second = first, second

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> tuple[torch.Tensor, ...]:
        return self.first(x), self.second(y)


_XNNPACK_SHAPES = ((3, 4, 4), (4, 37, 17), (4, 17, 37))


def _batch_dims(count: int, batch: int) -> tuple[Dim, ...]:
    return tuple(Dim(f"batch{i}", min=1, max=batch) for i in range(count))


def _structural(
    suffix: str, k: int, hidden: int, n: int, batch: Dim, batch2: Dim
) -> list[Case]:
    """Parallel, sequential and both, each linear `k` to `hidden` but the last of
    a sequence, `hidden` to `n`."""

    def sequential() -> torch.nn.Module:
        return torch.nn.Sequential(
            torch.nn.Linear(k, hidden), torch.nn.Linear(hidden, n)
        )

    return [
        chain(
            f"dynamic_parallel{suffix}",
            _Pair(torch.nn.Linear(k, hidden), torch.nn.Linear(k, hidden)),
            (batch, k),
            (batch, k),
        ),
        chain(f"dynamic_sequential{suffix}", sequential(), (batch, k)),
        chain(
            f"dynamic_parallel_and_sequential{suffix}",
            _Pair(torch.nn.Linear(k, hidden), sequential()),
            (batch, k),
            (batch2, k),
        ),
    ]


def _dynamic_cases() -> list[Case]:
    """The XNNPACK delegate's dynamic linear tests, with batch dims down to 1,
    and LLM token shapes every quantization scheme fits."""
    cases = []
    for batch, k, n in _XNNPACK_SHAPES:
        for batch_dims in (1, 2):
            for bias in (True, False):
                name = f"dynamic_{batch_dims}batch_{k}_{n}" + (
                    "" if bias else "_no_bias"
                )
                shape = (*_batch_dims(batch_dims, batch), k)
                cases.append(chain(name, torch.nn.Linear(k, n, bias=bias), shape))
    batch, batch2 = Dim("batch", min=1, max=100), Dim("batch2", min=1, max=100)
    cases += [
        chain("dynamic_two_batch", torch.nn.Linear(14, 15), (batch, batch, 14)),
        *_structural("", 4, 5, 3, batch, batch2),
    ]
    tokens = Dim("tokens", min=1, max=64)
    cases.append(chain("dynamic_tokens", torch.nn.Linear(256, 64), (tokens, 256)))
    for batch, k, n in _XNNPACK_SHAPES:
        for batch_dims in (1, 2):
            for bias in (True, False):
                name = f"dynamic_{batch_dims}batch_{k}_{n}_relu" + (
                    "" if bias else "_no_bias"
                )
                relu = torch.nn.Sequential(
                    torch.nn.Linear(k, n, bias=bias), torch.nn.ReLU()
                )
                cases.append(chain(name, relu, (*_batch_dims(batch_dims, batch), k)))
    rows = Dim("rows", min=1, max=16)
    return [
        *cases,
        # A static batch of one, as an LLM's prefill and decode.
        chain("dynamic_tokens_3d", torch.nn.Linear(256, 64), (1, tokens, 256)),
        chain("dynamic_two_batch_256", torch.nn.Linear(256, 64), (rows, rows, 256)),
        *_structural("_256", 256, 256, 64, tokens, Dim("tokens2", min=1, max=64)),
    ]


def cases() -> list[Case]:
    suite = suite_cases(test_linear, samples=4)
    torch.manual_seed(0)
    # One channel far above the rest, as in LLM activations.
    outlier = draws(4, (16, 256))
    for (x,) in outlier:
        x[:, 3] *= 32
    zero_row = draws(2, (8, 256))
    for (x,) in zero_row:
        x[2] = 0
    return [
        *suite,
        Case("decode", torch.nn.Linear(2048, 512), draws(8, (1, 2048))),
        Case("prefill", torch.nn.Linear(1024, 256), draws(2, (64, 1024))),
        Case("large_k", torch.nn.Linear(8192, 32), draws(4, (4, 8192))),
        Case(
            "leading_batch_no_bias",
            torch.nn.Linear(256, 96, bias=False),
            draws(4, (2, 7, 256)),
        ),
        Case("outlier_channel", torch.nn.Linear(256, 128), outlier),
        Case("zero_row", torch.nn.Linear(256, 64), zero_row),
        # One output per family of GGUF super-block scales.
        Case(
            "scale_families",
            _Parallel(*(torch.nn.Linear(256, 32) for _ in GGUF_SUPER_SCALES["q4_k"])),
            draws(2, (4, 256)),
        ),
        # FACTO varies one argument at a time, so only every fifth tuple is kept,
        # plus those whose in_features fit GGUF blocks. Its values lie on a 1/8
        # grid, which makes float32 math exact: these cases cover shapes more
        # than rounding.
        *facto_cases(
            _FACTO_SPEC,
            _facto_case,
            lambda i, args: i % 5 == 0 or args[1].size(1) % QK_K == 0,
        ),
        *_dynamic_cases(),
    ]
