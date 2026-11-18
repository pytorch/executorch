# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Which conv nodes become ``triton::convolution``.

``plan(node)`` is the replacement pass's preflight for convolutions
(see ``triton/routing.py``): it binds the node's arguments, checks the
templates' ``supports()``, and classifies the node without touching the graph.
The replacement produces exactly the output strides the node recorded, which
must be a dense layout. ``OPTIONAL`` means a template runs it, its
``worth_routing`` (the measured range where it beats the regular lowering)
accepts it, and the regular lowering could run it too; ``REQUIRED`` means a
template runs it and the regular lowering cannot.

A failure while planning follows the same contract: it raises
``ReplacementError`` when the regular lowering cannot compile the node, and
otherwise leaves the node unchanged.
"""

from __future__ import annotations

import dataclasses
import logging
from typing import Any, Optional

import torch
from executorch.backends.cuda.triton.kernels.conv.conv import TEMPLATES
from executorch.backends.cuda.triton.kernels.conv.core.common import (
    bind_conv,
    ConvCall,
    ConvParams,
    dense_strides,
    out_shape_and_stride,
)
from executorch.backends.cuda.triton.routing import (
    ReplacementError,
    ReplacementPlan,
    Route,
    unsupported,
)
from torch._subclasses.fake_tensor import FakeTensor
from torch.fx import Node

logger: logging.Logger = logging.getLogger(__name__)


def _unsupported(reason: str) -> ReplacementPlan:
    return unsupported("conv", reason)


def regular_lowering_gap(target: Any, call: ConvCall) -> Optional[str]:
    """Why the backend's regular lowering cannot compile this conv, or None.

    Inductor's Triton templates need dilation 1 and a forward convolution of
    rank 2 or 3 (rank 1 only for depthwise). The backend decomposes
    ``conv1d.default``, ``conv_transpose1d.default`` and
    ``conv_transpose2d.input`` into 2-D forward convolutions before Inductor
    sees them; ``conv1d.padding`` stays a 1-D convolution.
    """
    p, w = call.params, call.weight
    if any(d != 1 for d in p.dilation):
        return f"dilation {list(p.dilation)} has no Inductor template"
    decomposed = (target._schema.name, target._schema.overload_name) in (
        ("aten::conv1d", ""),
        ("aten::conv_transpose1d", ""),
        ("aten::conv_transpose2d", "input"),
    )
    if p.transposed and not decomposed:
        return f"{p.rank}-D transposed convolution has no Inductor template"
    depthwise = p.groups > 1 and w.shape[1] == 1 and w.shape[0] == p.groups
    if p.rank == 1 and not decomposed and not depthwise:
        return "1-D convolution has no Inductor template"
    return None


def _is_static(t: Optional[torch.Tensor]) -> bool:
    return t is None or (
        isinstance(t, FakeTensor)
        and all(type(s) is int for s in (*t.shape, *t.stride(), t.storage_offset()))
    )


def _check(node: Node, call: ConvCall, gap: Optional[str]) -> ReplacementPlan:
    """An UNSUPPORTED plan, or an OPTIONAL one labelled with the template."""
    expected = node.meta.get("val")
    if not all(_is_static(t) for t in (call.input, call.weight, call.bias, expected)):
        return _unsupported("symbolic shapes or missing fake tensors")
    if call.input.device.type != "cuda":
        return _unsupported(f"input on {call.input.device}")
    if not dense_strides(expected.shape, expected.stride()):
        return _unsupported(
            f"the node's output strides {list(expected.stride())} are not a "
            "dense layout"
        )
    size, _ = out_shape_and_stride(call)
    if size != tuple(expected.shape):
        return _unsupported(
            f"output size {list(size)} differs from the node's {list(expected.shape)}"
        )
    call = dataclasses.replace(call, output_strides=tuple(expected.stride()))
    reasons = []
    for template in TEMPLATES:
        reason = template.supports(call)
        # A call the regular lowering cannot compile is routed whenever a
        # template runs it; otherwise only where the template is worth it.
        if reason is None and gap is None:
            reason = template.worth_routing(call)
        if reason is None:
            return ReplacementPlan(Route.OPTIONAL, "", "conv", template.name)
        reasons.append(reason)
    return _unsupported(f"no template takes it: {'; '.join(reasons)}")


def plan(node: Node) -> ReplacementPlan:
    """Classifies a conv node; never changes the graph."""
    if torch.version.hip is not None:
        return _unsupported("ROCm")

    def value(arg: Any) -> Any:
        return arg.meta.get("val") if isinstance(arg, Node) else arg

    try:
        input, weight, bias, params = bind_conv(
            node.target, node.args, node.kwargs, value
        )
        call = ConvCall(value(input), value(weight), value(bias), params)
    except Exception as e:  # noqa: BLE001 - nothing is classified yet
        return _unsupported(f"cannot bind arguments: {e!r}")
    gap = regular_lowering_gap(node.target, call)
    try:
        result = _check(node, call, gap)
    except Exception as e:
        if gap is not None:
            raise ReplacementError(
                f"{call.signature()} ({gap}): preflight failed: {e!r}"
            ) from e
        logger.warning("Conv preflight failed for %s: %r", node.name, e)
        return _unsupported(f"preflight failed: {e!r}")
    if result.route is Route.UNSUPPORTED:
        if gap is not None:
            result = _unsupported(f"{result.reason}; regular lowering: {gap}")
        return result
    route = Route.OPTIONAL if gap is None else Route.REQUIRED
    # The output follows the strides the node recorded.
    call = dataclasses.replace(call, output_strides=tuple(node.meta["val"].stride()))
    return ReplacementPlan(
        route,
        gap or "",
        "conv",
        f"{result.label} template",
        torch.ops.triton.convolution.default,
        (input, weight, bias, *_canonical_args(params), list(call.output_strides)),
        call.signature(),
    )


def _canonical_args(p: ConvParams) -> tuple[Any, ...]:
    return (
        list(p.stride),
        list(p.padding_before),
        list(p.padding_after),
        list(p.dilation),
        p.transposed,
        list(p.output_padding),
        p.groups,
    )
