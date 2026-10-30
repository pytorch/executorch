# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Linear cases: the backend test suite's, FACTO's, and shapes and inputs that
stress accumulation and quantization."""

import torch
from executorch.backends.native.test.samples.cases import (
    Case,
    draws,
    facto_cases,
    suite_cases,
)
from executorch.backends.test.suite.operators import test_linear
from facto.inputgen.argument.type import ArgType
from facto.inputgen.specs.model import ConstraintProducer as cp, InPosArg, OutArg, Spec


def _unit(*constraints) -> list:
    """Float32 values in [-1, 1] under `constraints`."""
    return [
        cp.Dtype.Eq(lambda deps: torch.float32),
        *constraints,
        cp.Value.Ge(lambda deps, dtype, struct: -1),
        cp.Value.Le(lambda deps, dtype, struct: 1),
    ]


# in_features is a multiple of every scheme's group size, and out_features is
# large enough to fit an error's abs_bias and rel_bias from one sample.
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
        # FACTO varies one argument at a time, so only every fifth tuple is kept.
        # Its values lie on a 1/8 grid, which makes float32 math exact: these
        # cases cover shapes more than rounding.
        *facto_cases(_FACTO_SPEC, _facto_case, lambda i, _: i % 5 == 0),
    ]
