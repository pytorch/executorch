# Copyright 2025-2026 Arm Limited and/or its affiliates.
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


def register_tosa_dialect_op(
    op_schema,
    fake_func,
    real_func: Callable | None = None,
    real_dispatch_key: str = "CompositeExplicitAutograd",
) -> Callable:
    """Register a TOSA dialect operator with the backend op library.

    Args:
        op_schema (str): Operator schema without namespace or overload name.
        fake_func (Callable): Fake implementation used for registration.
        real_func (Optional[Callable]): Optional eager implementation.
        real_dispatch_key (str): Dispatch key used for the eager implementation.

    Returns:
        Callable: Backend dialect operator handle exposed via ``exir_ops``.

    """
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

    register_fake(op_qualified_name, fake_func, lib=tosa_lib)
    if real_func is not None:
        tosa_lib.impl(opname, real_func, real_dispatch_key)
    op = getattr(getattr(getattr(exir_ops.backend, tosa_lib.ns), opname), overload_name)

    return op


class TosaValueError(ValueError):
    """Error type that annotates failures with the originating TOSA op."""

    def __init__(self, message="A TOSA value error occurred", *args, op=None):
        """Initialise the error with optional operator metadata.

        Args:
            message (str): Human-readable error message.
            *args: Additional arguments forwarded to ``ValueError``.
            op: Optional operator identifier included in the string output.

        """
        super().__init__(message, *args)
        self.op = op

    def __str__(self):
        """Return the base message, appending the operator when provided."""
        base_message = super().__str__()
        if self.op is not None:
            return f"{base_message} (TOSA op: {self.op})"
        return base_message
