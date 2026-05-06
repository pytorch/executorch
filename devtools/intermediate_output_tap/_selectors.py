# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

"""
Predicates for selecting which FX nodes to tap.

A `NodeSelector` is just `Callable[[fx.Node], bool]`. The provided builders
let you compose them by op type, by `nn_module_stack` path, by arbitrary meta
tag, and via boolean combinators.

Examples:
    selector = select_any(
        select_by_op_type("aten.linear.default", "aten.matmul.default"),
        select_by_module_path("layers.*.attention"),
    )
    selector = select_all(selector, select_not(select_by_op_type("aten.view.default")))
"""

from __future__ import annotations

import fnmatch
from collections.abc import Callable
from typing import Any

import torch.fx as fx


NodeSelector = Callable[[fx.Node], bool]


def select_all_call_function(
    exclude: tuple[str, ...] = ("getitem",),
) -> NodeSelector:
    """Match every `call_function` node whose target name is not in `exclude`."""
    excluded = set(exclude)

    def predicate(n: fx.Node) -> bool:
        if n.op != "call_function":
            return False
        target_name = getattr(n.target, "__name__", str(n.target))
        # `getitem` shows up as the builtin name; also normalise common aten suffixes.
        return target_name not in excluded and str(n.target) not in excluded

    return predicate


def select_by_op_type(*op_targets: str) -> NodeSelector:
    """
    Match nodes whose `str(node.target)` ends with any of `op_targets`.

    The "ends with" rule lets the user write either the short name
    ("aten.linear.default") or a fully-qualified name and have it match.
    """
    if not op_targets:
        raise ValueError("select_by_op_type requires at least one op target")
    suffixes = tuple(op_targets)

    def predicate(n: fx.Node) -> bool:
        if n.op != "call_function":
            return False
        target_str = str(n.target)
        return any(target_str.endswith(s) or target_str == s for s in suffixes)

    return predicate


def select_by_module_path(pattern: str) -> NodeSelector:
    """
    Match nodes whose `nn_module_stack` (the chain of nn.Module attribute names
    walked to reach this op during tracing) contains a path matching `pattern`.
    `pattern` is a shell-glob (fnmatch) — e.g. "layers.*", "layers.0.attention",
    "*.attention.*".
    """

    def predicate(n: fx.Node) -> bool:
        stack = n.meta.get("nn_module_stack")
        if not stack:
            return False
        # nn_module_stack is an OrderedDict: id -> (qualified_path, module_type)
        for entry in stack.values():
            path = entry[0] if isinstance(entry, tuple) else entry
            if fnmatch.fnmatchcase(path, pattern):
                return True
        return False

    return predicate


# Sentinel: matches when the meta key exists at all, regardless of value.
_ANY_VALUE: object = object()


def select_by_meta_tag(key: str, value: Any = _ANY_VALUE) -> NodeSelector:
    """
    Match nodes that carry `node.meta[key]`. If `value` is provided, also
    requires `node.meta[key] == value`.
    """

    def predicate(n: fx.Node) -> bool:
        if key not in n.meta:
            return False
        if value is _ANY_VALUE:
            return True
        return n.meta[key] == value

    return predicate


def select_any(*selectors: NodeSelector) -> NodeSelector:
    """Match if ANY of `selectors` matches."""
    if not selectors:
        return lambda _n: False
    sels = tuple(selectors)
    return lambda n: any(s(n) for s in sels)


def select_all(*selectors: NodeSelector) -> NodeSelector:
    """Match if ALL of `selectors` match."""
    if not selectors:
        return lambda _n: True
    sels = tuple(selectors)
    return lambda n: all(s(n) for s in sels)


def select_not(selector: NodeSelector) -> NodeSelector:
    """Match if `selector` does NOT match."""
    return lambda n: not selector(n)
