# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Cases for core elementwise, selection and shape ops. Outputs are floating
point; comparisons and indices are cast to the method dtype."""

from collections.abc import Callable

import torch
from executorch.backends.native.test.samples.cases import Case, chain, draws, Inputs
from torch.export import Dim


class _Fn(torch.nn.Module):
    def __init__(self, fn: Callable[..., object]) -> None:
        super().__init__()
        self.fn = fn

    def forward(self, *args):
        return self.fn(*args)


class _OnLinear(torch.nn.Module):
    """`fn` of a linear's output, so weight-only schemes quantize its weight."""

    def __init__(self, fn: Callable[..., object], out_features: int = 64) -> None:
        super().__init__()
        # The same weights whatever ran before in this process.
        with torch.random.fork_rng():
            torch.manual_seed(0)
            self.linear = torch.nn.Linear(256, out_features)
        self.fn = fn

    def forward(self, x):
        return self.fn(self.linear(x))


def _quantized(name: str, fn: Callable[..., object], out_features: int = 64) -> Case:
    """A case over linear outputs; only cases named quantized_* hold a linear."""
    return Case(f"quantized_{name}", _OnLinear(fn, out_features), draws(2, (8, 256)))


def _tokens() -> Dim:
    return Dim("tokens", min=1, max=64)


def _pair(width: int) -> tuple[tuple[Dim, int], tuple[Dim, int]]:
    """Two inputs of shape (tokens, width) that share one dynamic dim."""
    tokens = _tokens()
    return (tokens, width), (tokens, width)


def _scaled(inputs: list[Inputs], scale: float) -> list[Inputs]:
    return [tuple(t * scale for t in x) for x in inputs]


def _ties(rows: int, cols: int) -> list[Inputs]:
    """Small integers, so equal elements are common."""
    generator = torch.Generator().manual_seed(0)
    return [
        tuple(
            torch.randint(-3, 4, (rows, cols), generator=generator).float()
            for _ in range(2)
        )
    ]


def split_cases() -> list[Case]:
    return [
        # A fused projection split into its parts.
        Case(
            "last_dim",
            _Fn(lambda x: torch.split(x, [128, 64, 64], dim=-1)),
            draws(2, (4, 256)),
        ),
        Case(
            "middle_dim",
            _Fn(lambda x: torch.split(x, [3, 1, 4], dim=1)),
            draws(2, (2, 8, 32)),
        ),
        Case("outer_dim", _Fn(lambda x: torch.split(x, 2, dim=0)), draws(2, (6, 64))),
        chain(
            "dynamic_tokens",
            _Fn(lambda x: torch.split(x, [128, 64, 64], dim=-1)),
            (_tokens(), 256),
        ),
        _quantized("fused_projection", lambda y: torch.split(y, [32, 16, 16], dim=-1)),
    ]


def _permuted_rows(rows: int, cols: int, count: int) -> list[Inputs]:
    """Rows of distinct integers, exact in every dtype for cols <= 256."""
    generator = torch.Generator().manual_seed(0)
    return [
        (
            torch.stack(
                [torch.randperm(cols, generator=generator) for _ in range(rows)]
            ).float()
            - cols // 2,
        )
        for _ in range(count)
    ]


def _topk(k: int, largest: bool = True) -> _Fn:
    def fn(x: torch.Tensor):
        values, indices = torch.topk(x, k, largest=largest)
        return values, indices.to(x.dtype)

    return _Fn(fn)


def _nan_rows() -> list[Inputs]:
    nan, inf = float("nan"), float("inf")
    return [
        (
            torch.tensor(
                [
                    [1.0, nan, -inf, 3.0, inf, 0.5],
                    [inf, -2.0, nan, -inf, 4.0, 2.0],
                    [nan, 0.0, -1.0, 5.0, -inf, inf],
                ]
            ),
        )
    ]


def topk_cases() -> list[Case]:
    return [
        # MoE routing: 8 of 128 experts per token.
        Case("router", _topk(8), _permuted_rows(16, 128, 2)),
        Case("smallest", _topk(4, largest=False), _permuted_rows(4, 256, 2)),
        Case("all", _topk(32), _permuted_rows(4, 32, 2)),
        # Values only: half-type draws tie, and tied indices may come out in
        # another order.
        chain(
            "dynamic_tokens", _Fn(lambda x: torch.topk(x, 8).values), (_tokens(), 128)
        ),
        _quantized("router", lambda s: torch.topk(s, 8).values),
        # NaN ranks above every value, then +inf; -inf ranks last.
        Case("nan_and_inf_largest", _topk(4), _nan_rows()),
        Case("nan_and_inf_smallest", _topk(4, largest=False), _nan_rows()),
    ]
