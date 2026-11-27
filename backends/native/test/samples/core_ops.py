# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Cases for core elementwise, selection and shape ops. Outputs are floating
point; comparisons and indices are cast to the method dtype."""

from collections.abc import Callable

import torch
import torch.nn.functional as F
from executorch.backends.native.test.samples.cases import Case, chain, draws, Inputs
from torch.export import Dim


class _Fn(torch.nn.Module):
    def __init__(self, fn: Callable[..., object]) -> None:
        super().__init__()
        self.fn = fn

    def forward(self, *args):
        return self.fn(*args)


_NAN = float("nan")


def _specials() -> list[Inputs]:
    """Ordinary values beside NaN and both infinities."""
    nan, inf = float("nan"), float("inf")
    row = torch.tensor([-2.0, nan, 0.5, inf, 3.0, -inf, -0.25, 1.5])
    return [(row.repeat(4, 8),)]


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


def _reduce(fn: Callable[..., torch.Tensor]) -> Callable[[], list[Case]]:
    def cases() -> list[Case]:
        tokens = Dim("tokens", min=1, max=64)
        return [
            Case("last", _Fn(lambda x: fn(x, (-1,))), draws(4, (4, 8, 32))),
            Case(
                "last_keepdim",
                _Fn(lambda x: fn(x, (-1,), keepdim=True)),
                draws(2, (4, 8, 32)),
            ),
            Case("leading", _Fn(lambda x: fn(x, (0,))), draws(2, (16, 8, 32))),
            Case(
                "inner_keepdim",
                _Fn(lambda x: fn(x, (1,), keepdim=True)),
                draws(2, (2, 16, 4, 8)),
            ),
            Case("outer_two", _Fn(lambda x: fn(x, (0, 2))), draws(2, (8, 4, 16, 2))),
            chain("dynamic_tokens", _Fn(lambda x: fn(x, (-1,))), (tokens, 64)),
            _quantized("last", lambda y: fn(y, (-1,))),
        ]

    return cases


def sum_cases() -> list[Case]:
    return _reduce(lambda x, dims, keepdim=False: x.sum(dims, keepdim=keepdim))()


def mean_cases() -> list[Case]:
    return _reduce(lambda x, dims, keepdim=False: x.mean(dims, keepdim=keepdim))()


def amax_cases() -> list[Case]:
    # A row with a NaN has NaN as its maximum. GPU argmax kernels find no
    # index in a row of -inf or -FLT_MAX alone, mixed or not.
    nan, inf = float("nan"), float("inf")
    low = torch.finfo(torch.float32).min
    rows = torch.tensor(
        [
            [1.0, nan, -2.0],
            [3.0, 0.0, 1.0],
            [nan, nan, nan],
            [-inf, -inf, -inf],
            [low, low, low],
            [-inf, low, -inf],
            [-inf, -inf, low],
        ]
    )
    return _reduce(lambda x, dims, keepdim=False: x.amax(dims, keepdim=keepdim))() + [
        Case("special_rows", _Fn(lambda x: x.amax(-1)), [(rows,)]),
        Case("special_columns", _Fn(lambda x: x.amax(0)), [(rows.t().contiguous(),)]),
    ]


def _softmax(fn: Callable[..., torch.Tensor]) -> Callable[[], list[Case]]:
    def cases() -> list[Case]:
        tokens = Dim("tokens", min=1, max=64)
        return [
            Case("last", _Fn(lambda x: fn(x, -1)), draws(4, (4, 8, 32))),
            # Logits a few units apart, as attention scores are.
            Case("wide", _Fn(lambda x: fn(x, -1)), _scaled(draws(2, (4, 64)), 8.0)),
            # Probabilities that would underflow in a log of a softmax.
            Case("peaked", _Fn(lambda x: fn(x, -1)), _scaled(draws(2, (4, 16)), 60.0)),
            Case("leading", _Fn(lambda x: fn(x, 0)), draws(2, (16, 8, 4))),
            Case("inner", _Fn(lambda x: fn(x, 1)), draws(2, (2, 16, 4, 8))),
            chain("dynamic_tokens", _Fn(lambda x: fn(x, -1)), (tokens, 64)),
            _quantized("logits", lambda y: fn(y, -1)),
        ]

    return cases


def softmax_cases() -> list[Case]:
    return _softmax(torch.softmax)()


def _special_pairs() -> list[Inputs]:
    """NaN, both infinities, both zeros and ordinary values, each against every
    other."""
    nan, inf = float("nan"), float("inf")
    values = torch.tensor([nan, inf, -inf, 0.0, -0.0, 1.0, -1.0, 2.5])
    return [(values.repeat(8, 1), values[:, None].repeat(1, 8))]


def _equality(op: Callable[..., torch.Tensor]) -> Callable[[], list[Case]]:
    def cases() -> list[Case]:
        inf = float("inf")
        return [
            Case("ties", _Fn(lambda x, y: op(x, y).to(x.dtype)), _ties(8, 64)),
            Case(
                "special_pairs",
                _Fn(lambda x, y: op(x, y).to(x.dtype)),
                _special_pairs(),
            ),
            Case(
                "broadcast",
                _Fn(lambda x, y: op(x < 0, y < 0).to(x.dtype)),
                draws(2, (2, 8, 64), (8, 1)),
            ),
            Case(
                "scalar",
                _Fn(lambda x: op((x < 0).to(x.dtype), 1.0).to(x.dtype)),
                draws(4, (4, 256)),
            ),
            Case("nan_scalar", _Fn(lambda x: op(x, _NAN).to(x.dtype)), _specials()),
            Case("inf_scalar", _Fn(lambda x: op(x, inf).to(x.dtype)), _specials()),
            # Positions against a step, as a state reset at position 0 does.
            Case(
                "integer_positions",
                _Fn(lambda x: x * op(torch.arange(x.shape[-1]), 5).to(x.dtype)),
                draws(2, (4, 64)),
            ),
            Case(
                "integer_tensors",
                _Fn(lambda x, i: x * op(i, torch.arange(4)).to(x.dtype)),
                [(torch.ones(2, 4), torch.tensor([[0, 2, 2, 3], [3, 1, -2, 0]]))],
            ),
            chain(
                "dynamic_tokens",
                _Fn(lambda x, y: op(x < 0, y < 0).to(x.dtype)),
                *_pair(64),
            ),
            # A mask times y, so a mask flipped by rounding near y = 0 costs
            # little.
            _quantized(
                "masked", lambda y: y * op((y < 0).to(y.dtype), 1.0).to(y.dtype)
            ),
        ]

    return cases


def ne_cases() -> list[Case]:
    return _equality(torch.ne)()


def repeat_interleave_cases() -> list[Case]:
    # repeat_interleave decomposes to unsqueeze, expand and reshape through a
    # rank-5 tensor with a leading 1.
    return [
        Case(
            "heads",
            _Fn(lambda x: x.repeat_interleave(2, dim=2) * 2),
            draws(2, (1, 4, 2, 16)),
        ),
        Case(
            "rank5_permute",
            _Fn(lambda x: x.permute(0, 2, 1, 3, 4) * 2),
            draws(2, (1, 2, 3, 4, 5)),
        ),
        chain(
            "dynamic_tokens",
            _Fn(lambda x: x.repeat_interleave(2, dim=2) * 2),
            (1, _tokens(), 2, 16),
        ),
    ]


def softplus_cases() -> list[Case]:
    return [
        Case("unit", _Fn(F.softplus), draws(4, (4, 256))),
        # Past the threshold of 20, and where exp overflows.
        Case("wide", _Fn(F.softplus), _scaled(draws(2, (8, 256)), 40.0)),
        Case(
            "beta",
            _Fn(lambda x: F.softplus(x, beta=2.0, threshold=10.0)),
            _scaled(draws(2, (4, 256)), 8.0),
        ),
        Case(
            "beta_half_threshold_30",
            _Fn(lambda x: F.softplus(x, beta=0.5, threshold=30.0)),
            _scaled(draws(2, (4, 256)), 40.0),
        ),
        Case(
            "threshold_10",
            _Fn(lambda x: F.softplus(x, threshold=10.0)),
            _scaled(draws(2, (4, 256)), 16.0),
        ),
        Case(
            "negative_beta",
            _Fn(lambda x: F.softplus(x, beta=-1.0)),
            _scaled(draws(2, (4, 256)), 8.0),
        ),
        chain("dynamic_tokens", _Fn(F.softplus), (_tokens(), 64)),
        _quantized("gate", F.softplus),
    ]


def rms_norm_cases() -> list[Case]:
    # Qwen3.8 27B's hidden size, one token's row, with and without a weight.
    weighted = torch.nn.RMSNorm(5120, eps=1e-6)
    with torch.no_grad(), torch.random.fork_rng():
        torch.manual_seed(0)
        weighted.weight.normal_(1.0, 0.1)
    return [
        Case(
            "hidden_5120",
            _Fn(lambda x: F.rms_norm(x, (5120,), eps=1e-6)),
            draws(2, (1, 5120)),
        ),
        Case("hidden_5120_weight", weighted, draws(2, (1, 5120))),
    ]
