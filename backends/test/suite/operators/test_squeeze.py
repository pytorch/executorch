# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-unsafe

import torch

from executorch.backends.test.suite.operators import parameterize_by_dtype


class SqueezeModel(torch.nn.Module):
    def forward(self, x):
        return torch.squeeze(x)


class SqueezeDimModel(torch.nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, x):
        return torch.squeeze(x, dim=self.dim)


@parameterize_by_dtype
def test_squeeze_dtype(test_runner, dtype) -> None:
    test_runner.lower_and_run_model(
        SqueezeModel(),
        (torch.rand(1, 3, 1, 5).to(dtype),),
    )


def test_squeeze_specific_dimension(test_runner) -> None:
    test_runner.lower_and_run_model(
        SqueezeDimModel(dim=0),
        (torch.randn(1, 3, 5),),
    )

    test_runner.lower_and_run_model(
        SqueezeDimModel(dim=2),
        (torch.randn(3, 4, 1, 5),),
    )

    test_runner.lower_and_run_model(
        SqueezeDimModel(dim=-1),
        (torch.randn(3, 4, 5, 1),),
    )


def test_squeeze_no_effect(test_runner) -> None:
    test_runner.lower_and_run_model(
        SqueezeDimModel(dim=1),
        (torch.randn(3, 4, 5),),
    )

    test_runner.lower_and_run_model(
        SqueezeModel(),
        (torch.randn(3, 4, 5),),
    )


def test_squeeze_multiple_dims(test_runner) -> None:
    test_runner.lower_and_run_model(
        SqueezeModel(),
        (torch.randn(1, 3, 1, 5, 1),),
    )

    test_runner.lower_and_run_model(
        SqueezeDimModel(dim=(0, 1)),
        (torch.randn(1, 1, 1),),
    )
