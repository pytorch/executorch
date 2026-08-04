# Copyright 2026 Arm Limited and/or its affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import executorch.backends.arm.tosa.dialect  # noqa: F401

import torch
from executorch.backends.arm.tosa.specification import (
    TosaLoweringContext,
    TosaSpecification,
)
from executorch.exir.dialects._ops import ops as exir_ops
from torch._subclasses.fake_tensor import FakeTensorMode


def test_gather_tosa_FP_fake() -> None:
    values = torch.randn((1, 4, 3), dtype=torch.float32)
    indices = torch.tensor([[0, 2]], dtype=torch.int32)

    with TosaLoweringContext(
        TosaSpecification.create_from_string("TOSA-1.0+FP")
    ), FakeTensorMode() as mode:
        output = exir_ops.backend.tosa.GATHER.default(
            mode.from_tensor(values),
            mode.from_tensor(indices),
        )

    assert output.dtype == values.dtype
    assert tuple(output.shape) == (1, 2, 3)


def test_gather_tosa_FP_real() -> None:
    values = torch.tensor(
        [[[1.0, 10.0], [2.0, 20.0], [3.0, 30.0], [4.0, 40.0]]],
        dtype=torch.float32,
    )
    indices = torch.tensor([[3, 1]], dtype=torch.int32)

    with TosaLoweringContext(TosaSpecification.create_from_string("TOSA-1.0+FP")):
        output = exir_ops.backend.tosa.GATHER.default(values, indices)

    expected = values[:, indices[0], :]
    torch.testing.assert_close(output, expected)
