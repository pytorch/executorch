# Copyright 2025 Arm Limited and/or its affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.


from typing import Callable, Iterable, List, ParamSpec, TypeVar

from executorch.backends.arm.tosa.dialect.lib import register_tosa_dialect_op

from executorch.backends.arm.tosa.specification import (
    get_context_spec,
    TosaSpecification,
)

P = ParamSpec("P")
R = TypeVar("R")

# The list of registered ops are not yet used, except for registration
_tosa_registered_ops: dict[TosaSpecification, list[Callable]] = {
    TosaSpecification.create_from_string("TOSA-1.0+FP"): [],
    TosaSpecification.create_from_string("TOSA-1.0+INT"): [],
}

# Mapping to ensure we only register a given function once.
_registered_tosa_ops_by_func: dict[Callable, Callable] = {}


def register_fake_tosa_op(
    op_schema: str, tosa_specs: Iterable[TosaSpecification]
) -> Callable[[Callable[P, R]], Callable[P, R]]:
    """
    Decorator for registering a TOSA operation.

    Parameters:
      op_schema : A string that defines the operation schema.
      tosa_specs : Iterable of TOSA specification strings,
                    e.g. ("TOSA-1.0+INT", "TOSA-1.0+FP").

    The decorated function is registered with the given op_schema by calling
    register_tosa_dialect_op(op_schema, func) only once per function. The resulting
    callable is then inserted into _tosa_registered_ops for each spec.
    """

    def decorator(func: Callable[P, R]) -> Callable[P, R]:
        # Only call register_tosa_dialect_op if the function hasn't been registered yet.
        if func not in _registered_tosa_ops_by_func:
            op_callable = register_tosa_dialect_op(op_schema, func)
            _registered_tosa_ops_by_func[func] = op_callable
        else:
            op_callable = _registered_tosa_ops_by_func[func]

        # For each TOSA spec, ensure the operation is added only once.
        for spec in tosa_specs:
            if spec not in _tosa_registered_ops:
                raise ValueError(f"TOSA spec {spec} not listed for registrations")
            if op_callable not in _tosa_registered_ops[spec]:
                _tosa_registered_ops[spec].append(op_callable)

        # return the original function
        return func

    return decorator


def get_registered_tosa_ops() -> List[Callable]:
    tosa_spec = get_context_spec()
    return _tosa_registered_ops[tosa_spec]
