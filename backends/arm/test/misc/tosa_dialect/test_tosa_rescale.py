# Copyright 2026 Arm Limited and/or its affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import executorch.backends.arm.tosa.dialect  # noqa: F401
import pytest
import torch

from executorch.backends.arm.tosa.specification import (
    TosaLoweringContext,
    TosaSpecification,
)
from executorch.exir.dialects._ops import ops as exir_ops
from torch._subclasses.fake_tensor import FakeTensor, FakeTensorMode


@pytest.mark.parametrize(
    "kwargs",
    [
        {},
        {"input_unsigned": False, "output_unsigned": False},
    ],
)
def test_rescale_real_impl_with_and_without_kwargs(kwargs):
    input_tensor = torch.tensor(
        [[1, -2, 3], [4, 0, -5]],
        dtype=torch.int32,
    )

    with TosaLoweringContext(
        TosaSpecification.create_from_string("TOSA-1.0+INT")
    ), FakeTensorMode() as mode:
        fake_output = exir_ops.backend.tosa.RESCALE.default(
            mode.from_tensor(input_tensor),
            torch.int32,
            [1.0],
            0,
            0,
            **kwargs,
        )

    assert isinstance(fake_output, FakeTensor)
    assert fake_output.dtype == torch.int32
    assert tuple(fake_output.shape) == tuple(input_tensor.shape)

    with TosaLoweringContext(TosaSpecification.create_from_string("TOSA-1.0+INT")):
        output = exir_ops.backend.tosa.RESCALE.default(
            input_tensor,
            torch.int32,
            [1.0],
            0,
            0,
            **kwargs,
        )

    assert not isinstance(output, FakeTensor)
    assert output.dtype == torch.int32
    assert tuple(output.shape) == tuple(input_tensor.shape)
    assert torch.equal(output, input_tensor)
