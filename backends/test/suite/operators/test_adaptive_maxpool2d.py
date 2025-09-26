# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-unsafe


import torch

from executorch.backends.test.suite.operators import parameterize_by_dtype


class Model(torch.nn.Module):
    def __init__(
        self,
        output_size=(5, 5),
        return_indices=False,
    ):
        super().__init__()
        self.adaptive_maxpool = torch.nn.AdaptiveMaxPool2d(
            output_size=output_size,
            return_indices=return_indices,
        )

    def forward(self, x):
        return self.adaptive_maxpool(x)


@parameterize_by_dtype
def test_adaptive_maxpool2d_dtype(test_runner, dtype) -> None:
    # Input shape: (batch_size, channels, height, width)
    test_runner.lower_and_run_model(
        Model().to(dtype),
        ((torch.rand(1, 8, 20, 20) * 10).to(dtype),),
    )


def test_adaptive_maxpool2d_output_size(test_runner) -> None:
    # Test with different output sizes
    test_runner.lower_and_run_model(
        Model(output_size=1),
        (torch.randn(1, 8, 20, 20),),
    )
    test_runner.lower_and_run_model(
        Model(output_size=(1, 1)),
        (torch.randn(1, 8, 20, 20),),
    )
    test_runner.lower_and_run_model(
        Model(output_size=(10, 10)),
        (torch.randn(1, 8, 20, 20),),
    )
    test_runner.lower_and_run_model(
        Model(output_size=(5, 10)),
        (torch.randn(1, 8, 20, 20),),
    )


def test_adaptive_maxpool2d_return_indices(test_runner) -> None:
    # Test with return_indices=True
    class ModelWithIndices(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.adaptive_maxpool = torch.nn.AdaptiveMaxPool2d(
                output_size=(5, 5),
                return_indices=True,
            )

        def forward(self, x):
            return self.adaptive_maxpool(x)

    input_tensor = torch.randn(1, 8, 20, 20)

    test_runner.lower_and_run_model(
        ModelWithIndices(),
        (input_tensor,),
    )


def test_adaptive_maxpool2d_batch_sizes(test_runner) -> None:
    # Test with batch inputs
    test_runner.lower_and_run_model(
        Model(),
        (torch.randn(2, 8, 20, 20),),
    )
    test_runner.lower_and_run_model(
        Model(),
        (torch.randn(8, 8, 20, 20),),
    )
    test_runner.lower_and_run_model(
        Model(),
        (torch.randn(16, 8, 20, 20),),
    )


def test_adaptive_maxpool2d_input_sizes(test_runner) -> None:
    # Test with different input sizes
    test_runner.lower_and_run_model(
        Model(),
        (torch.randn(1, 4, 20, 20),),
    )
    test_runner.lower_and_run_model(
        Model(),
        (torch.randn(1, 16, 20, 20),),
    )
    test_runner.lower_and_run_model(
        Model(),
        (torch.randn(1, 8, 10, 10),),
    )
    test_runner.lower_and_run_model(
        Model(),
        (torch.randn(1, 8, 30, 30),),
    )
    test_runner.lower_and_run_model(
        Model(),
        (torch.randn(1, 8, 15, 25),),
    )
