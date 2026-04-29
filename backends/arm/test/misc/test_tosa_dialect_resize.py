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


def test_bilinear_resize_rejects_exact_one_sixteenth_downscale():
    with TosaLoweringContext(
        TosaSpecification.create_from_string("TOSA-1.0+INT")
    ), FakeTensorMode() as mode:
        with pytest.raises(
            TosaValueError,
            match="Bilinear RESIZE downscale must be strictly greater than 1/16",
        ):
            exir_ops.backend.tosa.RESIZE.default(
                mode.from_tensor(
                    torch.randint(0, 10, (1, 3, 256, 448), dtype=torch.int8)
                ),
                [2, 32, 2, 32],
                [15, 15],
                [-15, -15],
                resize_mode="bilinear",
            )
