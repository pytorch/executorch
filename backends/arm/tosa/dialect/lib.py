# Copyright 2025 Arm Limited and/or its affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from typing import Callable

from executorch.exir.dialects._ops import _BACKEND_OP_LIB, ops as exir_ops
from torch.library import Library, register_fake
from torchgen.model import FunctionSchema

# create a torch library for the TOSA dialect
# This defines a library to include Backend Dialect Operators in Executorch
tosa_lib = Library("tosa", "DEF")


def register_tosa_dialect_op(op_schema, func) -> Callable:
    if tosa_lib.ns not in _BACKEND_OP_LIB:
        _BACKEND_OP_LIB.append(tosa_lib.ns)

    if "::" in op_schema:
        raise ValueError("The schema should not contain a namespace.")

    # Parse the op_schema into a FunctionSchema
    func_schema = FunctionSchema.parse(op_schema)
    overload_name = func_schema.name.overload_name
    if overload_name:
        raise ValueError(
            "The TOSA dialect does not support overload names in the op schema."
        )

    opname = func_schema.name.name.base
    tosa_lib.define(op_schema)

    overload_name = "default"
    op_qualified_name = f"{tosa_lib.ns}::{opname}"

    register_fake(op_qualified_name, func, lib=tosa_lib)

    op = getattr(getattr(getattr(exir_ops.backend, tosa_lib.ns), opname), overload_name)

    # For now, since the TOSA operators are only used for lowering and serialization in the backend
    # the op doesn't need to be callable. This can be changed in the future if needed to support
    # execution of TOSA ops directly.
    def not_callable():
        raise RuntimeError("TOSA dialect op is not callable")

    op.__equvalent_callable__ = not_callable

    return op


class TosaValueError(ValueError):
    def __init__(self, message="A TOSA value error occurred", *args, **kwargs):
        super().__init__(message, *args, **kwargs)
        self.op = kwargs.get("op", None)

    def __str__(self):
        base_message = super().__str__()
        if self.op is not None:
            return f"{base_message} (TOSA op: {self.op})"
        return base_message
