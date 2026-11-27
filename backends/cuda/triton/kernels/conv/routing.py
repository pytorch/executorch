# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Which conv nodes become ``triton::convolution``.

``plan(node)`` is the replacement pass's preflight for convolutions
(see ``triton/routing.py``): it binds the node's arguments and lists the two
implementations without touching the graph. ``triton::convolution`` can run
the call if a template's ``supports()`` accepts it; it produces exactly the
output strides the node recorded, which must be a dense layout. The regular
lowering can run it unless ``regular_lowering_gap`` knows it cannot. The
static choice is the replacement where the regular lowering cannot run the
call or a template's ``worth_routing`` (the range where it beats the regular
lowering) accepts it.

A failure while planning raises ``ReplacementError`` when the regular
lowering cannot compile the node, and otherwise leaves the node unchanged.
"""

from __future__ import annotations

import dataclasses
import logging
from typing import Any, Optional

import torch
from executorch.backends.cuda.triton.kernels.conv.conv import TEMPLATES
from executorch.backends.cuda.triton.kernels.conv.core.common import (
    bind_conv,
    CONV_OVERLOADS,
    ConvCall,
    ConvParams,
    dense_strides,
    out_shape_and_stride,
)
from executorch.backends.cuda.triton.routing import (
    Candidate,
    is_static,
    REGULAR_LOWERING,
    regular_only,
    ReplacementError,
    ReplacementPlan,
)
from executorch.exir.dialects._ops import ops as exir_ops
from torch.fx import Node

logger: logging.Logger = logging.getLogger(__name__)

# The edge ops of the conv overloads ``bind_conv`` binds.
CONV_TARGETS = tuple(
    getattr(
        getattr(exir_ops.edge.aten, name.removeprefix("aten::")), overload or "default"
    )
    for name, overload in CONV_OVERLOADS
)


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


@dataclasses.dataclass(frozen=True)
class _Support:
    # The first template that runs the call, and the first that is also
    # worth routing it to (its static range accepts it).
    template: Optional[str] = None
    worth: Optional[str] = None
    # Why no template runs it, or why none is worth it.
    reason: str = ""


def _check(node: Node, call: ConvCall) -> _Support:
    expected = node.meta.get("val")
    if not all(is_static(t) for t in (call.input, call.weight, call.bias, expected)):
        return _Support(reason="symbolic shapes or missing fake tensors")
    if call.input.device.type != "cuda":
        return _Support(reason=f"input on {call.input.device}")
    if not dense_strides(expected.shape, expected.stride()):
        return _Support(
            reason=f"the node's output strides {list(expected.stride())} are not a "
            "dense layout"
        )
    size, _ = out_shape_and_stride(call)
    if size != tuple(expected.shape):
        return _Support(
            reason=f"output size {list(size)} differs from the node's "
            f"{list(expected.shape)}"
        )
    call = dataclasses.replace(call, output_strides=tuple(expected.stride()))
    reasons, not_worth = [], []
    first = None
    for template in TEMPLATES:
        reason = template.supports(call)
        if reason is not None:
            reasons.append(reason)
            continue
        first = first or template.name
        worth = template.worth_routing(call)
        if worth is None:
            return _Support(first, template.name)
        not_worth.append(worth)
    if first is None:
        return _Support(reason=f"no template takes it: {'; '.join(reasons)}")
    return _Support(first, reason="; ".join(reasons + not_worth))


def plan(node: Node) -> ReplacementPlan:
    """The implementations of a conv node; never changes the graph."""
    if torch.version.hip is not None:
        return regular_only("conv", "ROCm")

    def value(arg: Any) -> Any:
        return arg.meta.get("val") if isinstance(arg, Node) else arg

    try:
        input, weight, bias, params = bind_conv(
            node.target, node.args, node.kwargs, value
        )
        call = ConvCall(value(input), value(weight), value(bias), params)
    except Exception as e:  # noqa: BLE001 - nothing is classified yet
        return regular_only("conv", f"cannot bind arguments: {e!r}")
    gap = regular_lowering_gap(node.target, call)
    try:
        support = _check(node, call)
    except Exception as e:
        if gap is not None:
            raise ReplacementError(
                f"{call.signature()} ({gap}): preflight failed: {e!r}"
            ) from e
        logger.warning("Conv preflight failed for %s: %r", node.name, e)
        return regular_only("conv", f"preflight failed: {e!r}")
    regular = Candidate(REGULAR_LOWERING, gap)
    if support.template is None:
        return ReplacementPlan(
            "conv",
            (regular, Candidate("conv", support.reason)),
            signature=call.signature(),
        )
    # The output follows the strides the node recorded.
    call = dataclasses.replace(call, output_strides=tuple(node.meta["val"].stride()))
    label = support.worth if gap is None and support.worth else support.template
    replacement = Candidate(
        "conv",
        None,
        f"{label} template",
        torch.ops.triton.convolution.default,
        (input, weight, bias, *_canonical_args(params), list(call.output_strides)),
    )
    worth = gap is not None or support.worth is not None
    return ReplacementPlan(
        "conv",
        (regular, replacement),
        "conv" if worth else REGULAR_LOWERING,
        support.reason,
        call.signature(),
        measure=True,
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
