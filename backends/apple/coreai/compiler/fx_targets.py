# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Introspection of fx ``call_function`` targets across edge and ATen dialects.

Shared because the partitioner sees edge-dialect targets while the converter
sees plain ATen ones, and the two spell an op's name differently. Reading
``target.__name__`` directly is the bug this module exists to prevent.
"""

from typing import Any, Optional

from executorch.exir.dialects.edge._ops import EdgeOpOverload


def underlying_target(target: Any) -> Any:
    """Unwrap an ``EdgeOpOverload`` to its plain ATen overload.

    ExecuTorch edge ops wrap the ATen overload in an ``EdgeOpOverload`` whose
    ``__name__`` is prefixed (``"aten.view.default"``); the overload at
    ``target._op`` has the bare name (``"view.default"``) that Core AI keys on.
    Plain ``OpOverload``s also expose a ``_op`` attribute, but its ``__name__``
    is empty, so only genuine edge ops may be unwrapped.
    """
    if isinstance(target, EdgeOpOverload):
        return target._op
    return target


def target_name(target: Any) -> Optional[str]:
    """Unprefixed op name, e.g. ``"view.default"``, or None if it has none."""
    return getattr(underlying_target(target), "__name__", None)


def target_namespace(target: Any) -> Optional[str]:
    """Namespace of the target, e.g. ``"aten"``, or None if it has none."""
    return getattr(underlying_target(target), "namespace", None)
