# Copyright 2026 Arm Limited and/or its affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from typing import List

import torch

from executorch.backends.arm.tosa.dialect.lib import TosaValueError
from executorch.backends.arm.tosa.dialect.ops_registration import register_fake_tosa_op

from executorch.backends.arm.tosa.specification import (
    get_context_spec,
    TosaSpecification,
)


@register_fake_tosa_op(
    "PAD(Tensor input1, SymInt[] padding, *, Scalar value) -> Tensor",  # schema
    (
        TosaSpecification.create_from_string("TOSA-1.0+INT"),
        TosaSpecification.create_from_string("TOSA-1.0+FP"),
    ),  # target TOSA specifications
)
def PAD(a: torch.Tensor, padding: List[int | torch.SymInt], *, value):
    tosa_spec = get_context_spec()

    supported_dtypes = {torch.bool}
    if tosa_spec.support_integer():
        supported_dtypes.update({torch.int8, torch.int16, torch.int32})
    if tosa_spec.support_float():
        supported_dtypes.update({torch.float16, torch.float32})
    if tosa_spec.support_extension("bf16"):
        supported_dtypes.add(torch.bfloat16)
    if a.dtype not in supported_dtypes:
        raise TosaValueError(
            f"Input tensor dtype {a.dtype} is not supported by the target TOSA specification."
            f" Supported dtypes are: {supported_dtypes}",
            op="PAD",
        )

    if len(padding) != 2 * len(a.shape):
        raise TosaValueError(
            f"Padding length {len(padding)} is not compatible with input rank {len(a.shape)}",
            op="PAD",
        )

    # new shape:
    new_shape: List[int | torch.SymInt] = []
    for i, d in enumerate(a.shape):
        pad_before = padding[i * 2]
        pad_after = padding[i * 2 + 1]
        new_shape.append(pad_before + d + pad_after)

    # return a new tensor with the new shape
    return torch.empty(size=new_shape, dtype=a.dtype)
