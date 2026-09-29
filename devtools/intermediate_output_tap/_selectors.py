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


def _instance_path(entry: object) -> str | None:
    """Extract the qualified-path string from an `nn_module_stack` entry."""
    if isinstance(entry, tuple):
        return entry[0] if entry else None
    return str(entry) if entry is not None else None


def _bare_class_name(mod_type: object) -> str:
    """Extract the bare class name from an `nn_module_stack` type entry."""
    cls_name = getattr(mod_type, "__name__", None)
    if cls_name is None:
        cls_name = str(mod_type).rsplit(".", 1)[-1].rstrip("'>")
    return cls_name


def _select_by_module_instance(
    instance_predicate: Callable[[object], bool],
    output_only: bool,
) -> NodeSelector:
    """
    Build a selector that matches nodes inside `nn_module_stack` instances
    accepted by `instance_predicate` (called with each entry value).

    If `output_only` is True, match only the *terminal* node of each matching
    module instance — i.e., a node N is matched iff (a) N is inside a matching
    instance, and (b) no user of N shares any matching instance with N.
    """

    def matching_instance_ids(n: fx.Node) -> list:
        stack = n.meta.get("nn_module_stack")
        if not stack:
            return []
        return [mod_id for mod_id, entry in stack.items() if instance_predicate(entry)]

    if not output_only:

        def predicate(n: fx.Node) -> bool:
            return bool(matching_instance_ids(n))

        return predicate

    def predicate_terminal(n: fx.Node) -> bool:
        my_ids = matching_instance_ids(n)
        if not my_ids:
            return False
        my_id_set = set(my_ids)
        for user in n.users:
            user_stack = user.meta.get("nn_module_stack") or {}
            if any(uid in my_id_set for uid in user_stack.keys()):
                return False
        return True

    return predicate_terminal


def select_by_module_path(
    *patterns: str,
    output_only: bool = True,
) -> NodeSelector:
    """
    Match nodes whose `nn_module_stack` (the chain of nn.Module attribute names
    walked to reach this op during tracing) contains a path matching ANY of
    `patterns`. Each pattern is a shell-glob (fnmatch) — e.g. "layers.*",
    "layers.0.attention", "*.attention.*".

    Args:
        patterns: One or more fnmatch path patterns to match.
        output_only: If True (default), match only the *terminal* node of
            each matching module instance — i.e., a node N is matched iff
            (a) N is inside a module whose path matches, and (b) no user of
            N is inside the same module instance. This taps only the
            value(s) flowing out of the module, not every intermediate op.
            Set to False to match every intermediate op inside matching
            modules.

    Example:
        # Tap only the value flowing out of each attention block.
        select_by_module_path("*.attention")

        # Tap every intermediate op inside layers 5 through 9.
        select_by_module_path(
            *[f"*.layers.{i}.*" for i in range(5, 10)],
            output_only=False,
        )
    """
    if not patterns:
        raise ValueError("select_by_module_path requires at least one pattern")
    pats = tuple(patterns)

    def instance_predicate(entry: object) -> bool:
        path = _instance_path(entry)
        if path is None:
            return False
        return any(fnmatch.fnmatchcase(path, p) for p in pats)

    return _select_by_module_instance(instance_predicate, output_only)


def select_by_module_class(
    *class_names: str,
    output_only: bool = True,
) -> NodeSelector:
    """
    Match nodes whose `nn_module_stack` contains a module of one of
    `class_names`.

    `class_names` are matched against the bare class name (e.g. "Attention",
    "RMSNorm"), not the fully-qualified type. This lets the selector survive
    moves between modules, and avoids the fnmatch escaping needed for paths.

    The `nn_module_stack` entry's second element is typically either a class
    object or a string like `"my.pkg.module.MyClass"`; both forms are handled.

    Args:
        class_names: One or more bare class names to match.
        output_only: If True (default), match only the *terminal* node of
            each matching module instance — i.e., a node N is matched iff
            (a) N is inside an instance of a target class, and (b) no user
            of N is inside the same module instance. This taps only the
            value(s) flowing out of the module, not every intermediate op.
            Set to False to match every intermediate op inside matching
            modules.

    Example:
        # Tap only the value flowing out of each RMSNorm.
        select_by_module_class("RMSNorm")

        # Tap every intermediate op inside any RMSNorm.
        select_by_module_class("RMSNorm", output_only=False)
    """
    if not class_names:
        raise ValueError("select_by_module_class requires at least one class name")
    names = set(class_names)

    def instance_predicate(entry: object) -> bool:
        if not isinstance(entry, tuple) or len(entry) < 2:
            return False
        return _bare_class_name(entry[1]) in names

    return _select_by_module_instance(instance_predicate, output_only)


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
