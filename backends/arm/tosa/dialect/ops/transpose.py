# Copyright 2025 Arm Limited and/or its affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import torch
from executorch.backends.arm.tosa.dialect.lib import TosaValueError
from executorch.backends.arm.tosa.dialect.ops_registration import register_fake_tosa_op

from executorch.backends.arm.tosa.specification import TosaSpecification


@register_fake_tosa_op(
    "TRANSPOSE(Tensor input, int[] perms) -> Tensor",  # schema
    (
        TosaSpecification.create_from_string("TOSA-1.0+FP"),
        TosaSpecification.create_from_string("TOSA-1.0+INT"),
    ),  # target TOSA specifications
)
def TRANSPOSE(a, perms):
    # The TOSA TRANSPOSE only do the transpose in the TOSA serialized world,
    # so just return the same shape and type.

    # For certain operators we need the data in a specific data format. Changing tosa_dim_order
    # is not sufficient as we also need transpose the data.
    # By utilizing an edge IR passthrough operator we can keep the edge program in
    # channels-first/contiguous and get the desired behavior in the TOSA lowering.

    if len(perms) not in (4, 5, 6):
        raise TosaValueError(
            f"Only 4D, 5D and 6D tensors are supported, got {len(perms)}: {perms}",
            op="TRANSPOSE",
        )

    return torch.empty_like(a, dtype=a.dtype)
