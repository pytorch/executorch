# Copyright (c) Qualcomm Innovation Center, Inc.
# All rights reserved
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""observe_pass — decorator to auto-collect pass input/output via Observatory."""

from __future__ import annotations

import functools
import logging
from typing import Any, Callable, Optional, Type, Union

from torch.fx.passes.infra.pass_base import PassResult


def observe_pass(
    target: Union[Callable, Type, None] = None,
    *,
    name: Optional[str] = None,
    collect_input: bool = True,
    collect_output: bool = True,
) -> Any:
    """Wrap a pass to auto-collect its graph via Observatory.

    Works as a class decorator, instance wrapper, or function wrapper.

    Args:
        target: A PassBase subclass (class decorator), a pass instance, or a
            callable pass. When ``None``, returns a parameterized decorator.
        name: Override the record name (default: derived from class/function name).
        collect_input: Collect the input graph before the pass runs (default True).
        collect_output: Collect the output graph after the pass runs (default True).
    """
    collect_both = collect_input and collect_output

    def _base_name(obj: Any) -> str:
        if name:
            return name
        if isinstance(obj, type):
            return obj.__name__
        cls_name = type(obj).__name__
        if cls_name == "function":
            return getattr(obj, "__name__", "pass")
        return cls_name

    def _collect_artifact(record_name: str, artifact: Any) -> None:
        from .observatory import Observatory

        try:
            Observatory.collect(record_name, artifact)
        except Exception as exc:
            logging.debug("[observe_pass] collection failed for %s: %s", record_name, exc)

    def _output_graph(gm: Any, result: Any) -> Any:
        if isinstance(result, tuple) and hasattr(result, "graph_module"):
            return result.graph_module
        if result is None:
            return gm
        return result

    def _wrap_callable(fn: Callable) -> Callable:
        @functools.wraps(fn)
        def wrapper(gm: Any, *args: Any, **kwargs: Any) -> Any:
            base = _base_name(fn)
            if collect_input:
                _collect_artifact(f"{base}/input" if collect_both else base, gm)
            result = fn(gm, *args, **kwargs)
            if collect_output:
                _collect_artifact(
                    f"{base}/output" if collect_both else base,
                    _output_graph(gm, result),
                )
            return result

        return wrapper

    def _wrap_class(cls: Type) -> Type:
        original_call = cls.__call__

        @functools.wraps(original_call)
        def patched_call(self: Any, gm: Any, *args: Any, **kwargs: Any) -> Any:
            base = _base_name(self)
            if collect_input:
                _collect_artifact(f"{base}/input" if collect_both else base, gm)
            result = original_call(self, gm, *args, **kwargs)
            if collect_output:
                _collect_artifact(
                    f"{base}/output" if collect_both else base,
                    _output_graph(gm, result),
                )
            return result

        cls.__call__ = patched_call
        return cls

    # Dispatch based on how observe_pass was called.
    if target is None:
        # Parameterized: @observe_pass(name="X", collect_output=False)
        def decorator(t: Any) -> Any:
            if isinstance(t, type):
                return _wrap_class(t)
            return _wrap_callable(t)

        return decorator

    if isinstance(target, type):
        return _wrap_class(target)

    return _wrap_callable(target)
