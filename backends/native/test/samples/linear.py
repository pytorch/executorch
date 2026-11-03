# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Linear cases: the backend test suite's, and shapes and inputs that stress
accumulation and quantization."""

import torch
from executorch.backends.native.test.samples.cases import Case, draws, suite_cases
from executorch.backends.test.suite.operators import test_linear


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
    return suite + [
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
    ]
