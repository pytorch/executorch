# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

import functools
import itertools
from typing import Dict, Iterable, List, Optional, Set, Tuple, Union

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
        if hasattr(shape_env, "guarding_hint_or_throw"):
            output = shape_env.guarding_hint_or_throw(expr)
        else:
            # size_hint is deprecated, delete this code path.
            output = shape_env.size_hint(expr)
    except torch.fx.experimental.symbolic_shapes.GuardOnDataDependentSymNode:
        return None
    return int(output)


# The most symbol value combinations evaluated for one size.
_MAX_ENUMERATED_POINTS = 1024


# A size and the ranges and ceiling it is enumerated over.
_EnumerationKey = Tuple[sympy.Expr, Tuple[Tuple[sympy.Symbol, int, int], ...], int]


class EnumerationBudget:
    """
    Limits how many symbol value combinations eval_upper_bound evaluates to tighten
    the finite bounds of the sizes it is given for. A size is tried only when all its
    combinations fit in what is left, and is then charged the combinations it
    evaluated. Each distinct size is charged once, whether or not its result is
    already cached, so the bounds do not depend on what earlier programs evaluated. A
    budget of 0 tries no such size. Sizes with no interval bound are not charged.
    """

    def __init__(self, points: int) -> None:
        self.remaining = points
        self._charged: Set[_EnumerationKey] = set()

    def admits(self, key: _EnumerationKey, points: int) -> bool:
        return key in self._charged or points <= self.remaining

    def charge(self, key: _EnumerationKey, evaluated: int) -> None:
        if key not in self._charged:
            self.remaining -= evaluated
            self._charged.add(key)


@functools.lru_cache(maxsize=4096)
def _max_over_points(
    expr: sympy.Expr,
    ranges: Tuple[Tuple[sympy.Symbol, int, int], ...],
    ceiling: Optional[int],
) -> Tuple[Optional[int], int]:
    """
    Return the largest value of expr over ranges, or None when it cannot tell, and
    how many combinations were evaluated.
    """
    # Programs reuse the same few size expressions across many tensors, so the
    # evaluation is cached.
    symbols = [symbol for symbol, _, _ in ranges]
    largest = None
    evaluated = 0
    # Sizes mostly grow with their symbols, so trying large values first usually
    # reaches the ceiling, past which the caller's min() cannot change, early.
    for values in itertools.product(
        *(range(upper, lower - 1, -1) for _, lower, upper in ranges)
    ):
        evaluated += 1
        try:
            result = expr.xreplace(dict(zip(symbols, map(sympy.Integer, values))))
        except ZeroDivisionError:
            # A conditional size also evaluates the branch it does not take, so a
            # division by zero there says nothing about the program.
            if expr.has(sympy.Piecewise):
                return None, evaluated
            # Otherwise the size divides by zero here, so the program cannot run
            # with these values and needs no memory for them.
            continue
        except Exception:
            return None, evaluated
        if not isinstance(result, sympy.Integer):
            return None, evaluated
        largest = int(result) if largest is None else max(largest, int(result))
        if ceiling is not None and largest >= ceiling:
            break
    return largest, evaluated


def _enumeration_ranges(
    expr: sympy.Expr,
    var_to_range: Dict[sympy.Symbol, ValueRanges],  # pyre-ignore[24]
    below_minimum: int = 2,
) -> Optional[Tuple[Tuple[Tuple[sympy.Symbol, int, int], ...], int]]:
    """
    Return the integer range to try for each symbol of expr, starting up to
    below_minimum values under its ShapeEnv minimum, and the number of
    combinations. Returns None when expr has no function such as // or %, since
    interval arithmetic is usually close for those and enumerating would only cost
    time, when a symbol has no finite range, or when there are more than
    _MAX_ENUMERATED_POINTS combinations.
    """
    if not expr.atoms(sympy.Function):
        return None
    ranges = []
    points = 1
    for symbol in sorted(expr.free_symbols, key=str):
        value_range = var_to_range.get(symbol)
        if value_range is None or not (
            isinstance(value_range.lower, sympy.Integer)
            and isinstance(value_range.upper, sympy.Integer)
        ):
            return None
        # The ShapeEnv raises a declared minimum of 0 or 1 to 2, and raises the root
        # of a derived dim such as dim - 1 until that dim is at least 2, so the
        # program can run up to 2 below the recorded minimum, so by default those
        # values are tried too. Like the interval bound, this is a best effort there,
        # since the ShapeEnv simplified expr for its own range.
        lower, upper = int(value_range.lower), int(value_range.upper)
        lower = min(lower, max(lower - below_minimum, 0))
        points *= upper - lower + 1
        if points > _MAX_ENUMERATED_POINTS:
            return None
        ranges.append((symbol, lower, upper))
    return tuple(ranges), points


def _enumerated_upper_bound(
    expr: sympy.Expr,
    var_to_range: Dict[sympy.Symbol, ValueRanges],  # pyre-ignore[24]
    ceiling: int,
    budget: EnumerationBudget,
) -> Optional[int]:
    """
    Evaluate expr at every combination of its symbols' integer values and return the
    largest result, or a result of at least ceiling once one is found. Returns None
    when _enumeration_ranges does, when there are more combinations than budget has
    left, or when no combination can be evaluated. A division by zero skips that
    combination, unless expr is conditional, and any other failure returns None.
    """
    enumeration = _enumeration_ranges(expr, var_to_range)
    if enumeration is None:
        return None
    ranges, points = enumeration
    key = (expr, ranges, ceiling)
    if not budget.admits(key, points):
        return None
    largest, evaluated = _max_over_points(*key)
    budget.charge(key, evaluated)
    return largest


def eval_upper_bound(
    maybe_symint: Union[int, torch.SymInt],
    budget: Optional[EnumerationBudget] = None,
) -> int:
    """
    Evaluate a symint to its uppper bound value. Returns None if symint's symoblic expr's
    upper bound can not be evaluated to valid integer according to the constraints in shape_env.
    With a budget, a finite bound is tightened by enumeration as far as the budget
    allows, and a size with no interval bound is enumerated outside the budget.
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
        # Interval arithmetic can overshoot by far: for x // (x // 8) with x in [8, 71]
        # it gives 71, while the largest value is 15. Keeping the smaller of the two
        # means the bound can only get tighter.
        enumerated_upper = (
            None
            if budget is None
            else _enumerated_upper_bound(
                expr, shape_env.var_to_range, concrete_upper, budget
            )
        )
        if enumerated_upper is not None:
            concrete_upper = min(concrete_upper, enumerated_upper)
        assert isinstance(
            concrete_upper, int
        ), f"Expect upper bound to be a concrete int but got {concrete_upper}"
        return concrete_upper
    elif int_oo is not None and upper_bound is int_oo:
        # Interval arithmetic gives no bound when a size divides by a range that
        # contains 0, even when every symbol is bounded, and the trace hint below
        # is then often too small. So unlike tightening a finite bound, this is
        # not left to what the budget has left, and when the values below the
        # minimum push the size over the cap or fail, it is tried with fewer of them.
        if budget is not None:
            for below_minimum in (2, 1, 0):
                enumeration = _enumeration_ranges(
                    expr, shape_env.var_to_range, below_minimum
                )
                if enumeration is not None:
                    largest, _ = _max_over_points(expr, enumeration[0], None)
                    if largest is not None:
                        return largest
        hint = eval_expr(maybe_symint)
        if hint is not None:
            return hint
        raise RuntimeError(
            f"Cannot evaluate a finite upper bound for symbolic expression {expr} "
            "(int_oo) and no trace hint is available."
        )
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
