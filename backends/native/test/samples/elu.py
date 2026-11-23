# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""ELU cases: the backend test suite's, convolutional decoder activations,
non-finite and near-zero inputs, and dynamic shapes."""

import torch
from executorch.backends.native.test.samples.cases import (
    Case,
    chain,
    draws,
    suite_cases,
)
from executorch.backends.test.suite.operators import test_elu
from torch.export import Dim


def cases() -> list[Case]:
    suite = suite_cases(test_elu, samples=4)
    elu = torch.nn.ELU()
    frames = Dim("frames", min=1, max=512)
    special = torch.tensor(
        [float("nan"), float("inf"), float("-inf"), 0.0, -0.0, -1e-7, 1e-7, -20.0]
    )
    return [
        *suite,
        Case("decoder", elu, draws(2, (1, 64, 1920))),
        Case("wide_range", elu, [(8 * x,) for (x,) in draws(4, (4, 256))]),
        Case("special", elu, [(special,)]),
        chain("dynamic_frames", elu, (1, 64, frames)),
    ]
