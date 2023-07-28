# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from typing import Optional, Set, Union

import sympy

import torch


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


def eval_shape(shape):
    """
    Shape maybe immutable so we return a new shape. Return None for
    dimensions that are unbacked e.g. first dimension of nonzero's output.
    """
    new_shape = []
    for _, s in enumerate(shape):
        new_shape.append(eval_expr(s))
    return new_shape


def collect_free_symbols(shape) -> Set[sympy.Symbol]:
    symset = set()
    for sz in shape:
        if not isinstance(sz, torch.SymInt):
            continue
        symset.update(sz.node.expr.free_symbols)
    return symset
