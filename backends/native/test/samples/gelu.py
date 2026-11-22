# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""GELU cases: the backend test suite's, exact and tanh approximations over
ordinary and wide inputs, and dynamic tokens."""

import torch
from executorch.backends.native.test.samples.cases import (
    Case,
    chain,
    draws,
    suite_cases,
)
from executorch.backends.test.suite.operators import test_gelu
from torch.export import Dim


def cases() -> list[Case]:
    tokens = Dim("tokens", min=1, max=64)
    exact, tanh = torch.nn.GELU(), torch.nn.GELU(approximate="tanh")
    wide = [(8 * x,) for (x,) in draws(2, (8, 256))]
    return [
        *suite_cases(test_gelu, samples=4),
        Case("none", exact, draws(4, (4, 256))),
        Case("tanh", tanh, draws(4, (4, 256))),
        Case("none_wide", exact, wide),
        Case("tanh_wide", tanh, wide),
        chain("dynamic_tokens_none", exact, (tokens, 64)),
        chain("dynamic_tokens_tanh", tanh, (tokens, 64)),
    ]
