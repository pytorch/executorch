# Copyright 2026 Arm Limited and/or its affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.


from executorch.backends.arm.tosa.dialect.ops_registration import register_fake_tosa_op
from executorch.backends.arm.tosa.specification import TosaSpecification


@register_fake_tosa_op(
    "CONST_SHAPE(int[] shape) -> int[]",  # schema
    TosaSpecification.all_versions_and_profiles(),
)
def CONST_SHAPE(shape: list[int]) -> list[int]:
    """CONST_SHAPE operator creates a constant shape tensor."""

    return shape
