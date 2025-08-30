# Copyright 2025 Arm Limited and/or its affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import torch
from executorch.backends.arm.tosa.dialect.lib import TosaValueError
from executorch.backends.arm.tosa.dialect.ops_registration import register_fake_tosa_op

from executorch.backends.arm.tosa_specification import (
    get_context_spec,
    TosaSpecification,
)


@register_fake_tosa_op(
    "TABLE(Tensor input1, Tensor table) -> Tensor",  # schema
    (
        TosaSpecification.create_from_string("TOSA-1.0+INT"),
    ),  # target TOSA specifications
)
def TABLE(a, table):
    tosa_spec = get_context_spec()
    # verifiy input types according to the spec
    if not tosa_spec.support_integer():
        raise TosaValueError(
            f"TOSA spec {tosa_spec} doesn't support integers", op="TABLE"
        )

    if a.dtype == torch.int8:
        if table.shape != torch.Size((256,)):
            raise TosaValueError(
                f"Table of wrong size ({table.shape}!={torch.Size((256,))}", op="TABLE"
            )
        if table.dtype != torch.int8:
            raise TosaValueError(f"Table dtype {table.dtype} is not int8", op="TABLE")
        return_dtype = torch.int8
    elif a.dtype == torch.int16:
        if not tosa_spec.support_extension("int16"):
            raise TosaValueError(
                f"Context TOSA spec {tosa_spec} doesn't support int16", op="TABLE"
            )
        if table.shape != torch.Size((513,)):
            raise TosaValueError(
                f"Table of wrong size ({table.shape}!={torch.Size((513,))})", op="TABLE"
            )
        if table.dtype != torch.int16:
            raise TosaValueError(f"Table dtype {table.dtype} is not int32", op="TABLE")
        return_dtype = torch.int32
    else:
        raise TosaValueError(f"Unsupported dtype {a.dtype} for {tosa_spec}", op="TABLE")

    return torch.empty_like(a, dtype=return_dtype)
