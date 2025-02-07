# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

from typing import Iterable, List, Optional, Set, Union

import sympy

import torch
from torch.utils._sympy.value_ranges import bound_sympy, ValueRanges


def eval_expr(symint: Union[int, torch.SymInt]) -> Optional[int]:
    """
    Evaluate a symint to int. Returns None if symint's symoblic expr
    can not be evaluated to valid integer according to the hints.
    """
    if isinstance(symint, int):
        return symint
    node = symint.node
    shape_env = node.shape_env
    expr = node.expr
    try:
        output = shape_env.size_hint(expr)
    except torch.fx.experimental.symbolic_shapes.GuardOnDataDependentSymNode:
        return None
    return int(output)


def eval_upper_bound(maybe_symint: Union[int, torch.SymInt]) -> int:
    """
    Evaluate a symint to its uppper bound value. Returns None if symint's symoblic expr's
    upper bound can not be evaluated to valid integer according to the constraints in shape_env.
    """
    if isinstance(maybe_symint, int):
        return maybe_symint
    node = maybe_symint.node
    shape_env = node.shape_env
    expr = node.expr
    var_range: ValueRanges = bound_sympy(  # pyre-ignore[24]
        expr, shape_env.var_to_range
    )
    upper_bound = var_range.upper
    # This import is needed temporarily until we update the pinned torch version.

    try:
        from torch.utils._sympy.numbers import int_oo  # @manual
    except ImportError:
        int_oo = None

    if isinstance(upper_bound, sympy.Integer):
        concrete_upper = int(var_range.upper)
        assert isinstance(
            concrete_upper, int
        ), f"Expect upper bound to be a concrete int but got {concrete_upper}"
        return concrete_upper
    elif int_oo is not None and upper_bound is int_oo:
        return int_oo
    else:
        raise RuntimeError(
            f"Expect upper bound to be sympy.Integer or int_oo. but got {upper_bound}"
        )


def eval_shape(shape: Iterable[Union[int, torch.SymInt]]):  # pyre-ignore[3]
    """
    Shape maybe immutable so we return a new shape. Return None for
    dimensions that are unbacked e.g. first dimension of nonzero's output.
    """
    new_shape = []
    for _, s in enumerate(shape):
        new_shape.append(eval_expr(s))
    return new_shape


def eval_shape_upper_bound(shape: Iterable[Union[int, torch.SymInt]]) -> List[int]:
    new_shape = []
    for _, s in enumerate(shape):
        new_shape.append(eval_upper_bound(s))
    return new_shape


def collect_free_symbols(
    shape: Iterable[Union[int, torch.SymInt]]
) -> Set[sympy.Symbol]:
    symset = set()
    for sz in shape:
        if not isinstance(sz, torch.SymInt):
            continue
        symset.update(sz.node.expr.free_symbols)
    return symset
