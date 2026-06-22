# Copyright 2026 Arm Limited and/or its affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.
"""Fake-op support for the generic TOSA ``CUSTOM`` dialect op.

The serialized TOSA ``CUSTOM`` op is intentionally generic: it carries a
stable operator identity (for example ``myns.my_op``) plus an
opaque payload in ``implementation_attrs``. That is enough for serialization,
but not enough for FakeTensor propagation unless we also teach the compiler how
to model the output tensors of the specific wrapped op.

This module provides a lightweight registration mechanism for those compiler
side fake implementations:

1. A lowering pass rewrites an op to ``exir_ops.backend.tosa.CUSTOM.default``.
2. The wrapped custom op registers a thin adapter with
   ``@register_fake_tosa("namespace::op")``.
3. The generic ``CUSTOM`` fake implementation looks up that adapter by the
   ``operator_name`` argument and invokes it with the full custom-op calling
   convention ``(inputs, operator_name, domain_name, implementation_attrs)``.

The adapter should stay thin: it should only translate from the generic TOSA
CUSTOM signature back to the wrapped op's fake semantics. The real semantic
logic should continue to live in the original fake implementation where
possible.

"""

import inspect
from collections.abc import Callable

import torch
from executorch.backends.arm.tosa.dialect.ops_registration import register_fake_tosa_op

from executorch.backends.arm.tosa.specification import (
    get_context_spec,
    TosaSpecification,
)

_TOSA_CUSTOM_FAKE_IMPLS: dict[str, Callable] = {}


def _normalize_tosa_custom_operator_name(operator_name: str) -> str:
    """Normalize operator names so ``ns::op`` and ``ns.op`` map identically."""
    return operator_name.replace("::", ".")


def validate_tosa_custom_fake_impl(fake_impl: object) -> Callable:
    """Validate the signature expected by ``register_fake_tosa``.

    Registered fake implementations must accept the generic TOSA CUSTOM fake
    calling convention:

    ``(inputs, operator_name, domain_name, implementation_attrs)``

    and return ``list[Tensor]``.

    """
    if not callable(fake_impl):
        raise TypeError(
            "Expected tosa.CUSTOM fake impl to be callable, " f"got {type(fake_impl)}"
        )

    params = tuple(inspect.signature(fake_impl).parameters.values())
    positional_kinds = {
        inspect.Parameter.POSITIONAL_ONLY,
        inspect.Parameter.POSITIONAL_OR_KEYWORD,
    }
    if len(params) != 4 or any(param.kind not in positional_kinds for param in params):
        raise TypeError(
            "tosa.CUSTOM fake impl must have signature "
            "(inputs, operator_name, domain_name, implementation_attrs)"
        )
    return fake_impl


def register_fake_tosa(operator_name: str) -> Callable[[Callable], Callable]:
    """Register a fake implementation for a specific wrapped TOSA custom op.

    Args:
        operator_name: Stable custom operator identifier. Both ``ns::op`` and
            ``ns.op`` spellings are accepted.

    Returns:
        A decorator that registers a callable with signature
        ``(inputs, operator_name, domain_name, implementation_attrs)`` and
        returning ``list[Tensor]``.

    Example:
        ``@register_fake_tosa("my_namespace::my_op")``

    """
    normalized_name = _normalize_tosa_custom_operator_name(operator_name)

    def decorator(fake_impl: Callable) -> Callable:
        validated = validate_tosa_custom_fake_impl(fake_impl)
        _TOSA_CUSTOM_FAKE_IMPLS[normalized_name] = validated
        return fake_impl

    return decorator


def has_fake_tosa_impl(operator_name: str) -> bool:
    """Return whether a wrapped custom op has a registered fake impl."""
    normalized_name = _normalize_tosa_custom_operator_name(operator_name)
    return normalized_name in _TOSA_CUSTOM_FAKE_IMPLS


def run_registered_fake_tosa_impl(
    inputs: list[torch.Tensor],
    operator_name: str,
    domain_name: str,
    implementation_attrs: list[int],
) -> list[torch.Tensor]:
    """Invoke the registered fake implementation for a wrapped custom op."""
    normalized_name = _normalize_tosa_custom_operator_name(operator_name)
    fake_impl = _TOSA_CUSTOM_FAKE_IMPLS.get(normalized_name)
    if fake_impl is None:
        raise RuntimeError(
            f"tosa.CUSTOM requires a registered fake impl for {normalized_name}"
        )
    outputs = fake_impl(inputs, operator_name, domain_name, implementation_attrs)
    if not isinstance(outputs, list):
        raise TypeError(
            "tosa.CUSTOM fake impl must return list[Tensor], " f"got {type(outputs)}"
        )
    if not outputs:
        raise RuntimeError("tosa.CUSTOM fake impl must return at least one output")
    if not all(isinstance(output, torch.Tensor) for output in outputs):
        raise TypeError("tosa.CUSTOM fake impl must return list[Tensor]")
    return outputs


@register_fake_tosa_op(
    "CUSTOM(Tensor[] inputs, str operator_name, str domain_name, int[] implementation_attrs) -> Tensor[]",
    TosaSpecification.all_versions_and_profiles(),
)
def CUSTOM(
    inputs: list[torch.Tensor],
    operator_name: str,
    domain_name: str,
    implementation_attrs: list[int],
) -> list[torch.Tensor]:
    """Fake implementation for TOSA CUSTOM op.

    The CUSTOM op is backend-defined. The fake implementation dispatches to a
    registered compiler-side fake implementation for the specific custom op.

    """
    _ = get_context_spec()  # ensure a spec context exists
    if not inputs:
        raise RuntimeError("tosa.CUSTOM requires at least one input tensor")
    return run_registered_fake_tosa_impl(
        inputs,
        operator_name,
        domain_name,
        implementation_attrs,
    )
