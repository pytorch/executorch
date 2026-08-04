# Copyright 2026 Arm Limited and/or its affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import executorch.backends.arm.tosa.dialect  # noqa: F401
import pytest
import torch
from executorch.backends.arm.tosa.dialect.lib import TosaValueError
from executorch.backends.arm.tosa.specification import (
    TosaLoweringContext,
    TosaSpecification,
)
from executorch.exir.dialects._ops import ops as exir_ops
from torch._subclasses.fake_tensor import FakeTensorMode


def test_cast_tosa_int() -> None:
    tosa_spec = TosaSpecification.create_from_string("TOSA-1.0+INT")

    with TosaLoweringContext(tosa_spec), FakeTensorMode() as mode:
        input_tensor = mode.from_tensor(torch.ones((2, 3), dtype=torch.int8))
        output = exir_ops.backend.tosa.CAST.default(input_tensor, torch.int32)

    assert output.dtype == torch.int32
    assert tuple(output.shape) == (2, 3)


def test_cast_rejects_unsupported_profile_dtype_pair() -> None:
    tosa_spec = TosaSpecification.create_from_string("TOSA-1.0+INT")

    with TosaLoweringContext(tosa_spec), FakeTensorMode() as mode:
        input_tensor = mode.from_tensor(torch.ones((2, 3), dtype=torch.int8))
        with pytest.raises(TosaValueError, match="Unsupported CAST"):
            exir_ops.backend.tosa.CAST.default(input_tensor, torch.float32)
