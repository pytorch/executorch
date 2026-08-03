# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

"""Re-inplace contiguous ``slice_copy`` nodes as lightweight slices (#10917).

This is the slice analog of :class:`ReplaceViewCopyWithViewPass`.  A
``slice_copy`` addresses a sub-region of its input's storage; when that
sub-region is *contiguous* it can, in principle, be re-inplaced as a zero-copy
view into the base buffer instead of emitting a full-copy ``slice_copy`` kernel.

Scope note (see #10917):
    ``ReplaceViewCopyWithViewPass`` can reuse ``memory.view`` because a view
    aliases the *entire* base buffer -- same ``nbytes`` and offset ``0`` (the
    ``_ViewSpec`` guards ``nbytes == base.nbytes``).  A slice aliases only a
    *sub-region* at a non-zero byte offset with fewer bytes than the base, and
    ExecuTorch has no offset-based aliasing mechanism in memory planning today.
    Fully eliminating the copy therefore requires (a) memory-planning support
    for offset sub-buffer aliasing and (b) a lightweight runtime op
    (``et_slice``) mirroring ``et_view``.  That runtime design is under
    discussion with the maintainer.

    This pass implements the piece that is well-defined regardless of that
    design decision: correctly identifying which ``slice_copy`` nodes are
    *eligible* (contiguous) for re-inplacing.  The rewrite is gated behind the
    offset-aliasing support and is a no-op until it lands.
"""

import logging

import torch
from executorch.exir.dialects._ops import ops
from torch.fx.passes.infra.pass_base import PassBase, PassResult

logger: logging.Logger = logging.getLogger(__name__)


def _is_slice_copy(node: torch.fx.Node) -> bool:
    return node.op == "call_function" and node.target in (
        torch.ops.aten.slice_copy.Tensor,
        ops.edge.aten.slice_copy.Tensor,
    )


def is_contiguous_slice_copy(node: torch.fx.Node) -> bool:
    """Return True if ``node`` is a ``slice_copy`` whose result is a contiguous
    sub-region of a contiguous input, and is therefore eligible to be
    re-inplaced as a zero-copy slice.

    A slice ``self[start:end:step]`` along ``dim`` is a contiguous sub-buffer of
    a contiguous input only when it is taken along the outermost (first) storage
    dimension with unit step.  Slicing an inner dimension, or using ``step > 1``,
    produces a strided (non-contiguous) result that cannot alias the base buffer
    without a copy.

    Signature: ``slice_copy.Tensor(self, dim=0, start=None, end=None, step=1)``.
    """
    if not _is_slice_copy(node):
        return False

    args = node.args
    self_arg = args[0]

    # dim defaults to 0; normalize negatives against the input rank.
    dim = args[1] if len(args) > 1 else 0
    step = args[4] if len(args) > 4 else 1

    if step != 1:
        return False

    rank = None
    self_val = self_arg.meta.get("val") if isinstance(self_arg, torch.fx.Node) else None
    if self_val is not None and hasattr(self_val, "dim"):
        rank = self_val.dim()

    if isinstance(dim, int) and dim < 0:
        if rank is None:
            # Cannot resolve a negative dim to the outermost dim without rank.
            return False
        dim = dim + rank

    # Only an outermost-dim, unit-step slice of a contiguous input is a
    # contiguous sub-buffer that could alias the base storage.
    return dim == 0


class ReplaceSliceCopyWithSlicePass(PassBase):
    """Re-inplace eligible (contiguous) ``slice_copy`` nodes as lightweight
    slices.

    Until offset-based sub-buffer aliasing lands in memory planning (see the
    module docstring and #10917), this pass only *identifies* eligible nodes and
    performs no graph mutation, so it is safe to run in the pipeline.
    """

    def __init__(self) -> None:
        super().__init__()

    def call(self, graph_module: torch.fx.GraphModule) -> PassResult:
        n_eligible = 0
        for module in graph_module.modules():
            if not isinstance(module, torch.fx.GraphModule):
                continue
            for node in module.graph.nodes:
                # A slice feeding the graph output can have its pointer modified
                # at runtime, mirroring the view_copy pass's output guard.
                if is_contiguous_slice_copy(node) and all(
                    u.op != "output" for u in node.users
                ):
                    n_eligible += 1

        logger.debug(
            "ReplaceSliceCopyWithSlicePass: %d contiguous slice_copy node(s) "
            "eligible for re-inplacing (rewrite pending offset-aliasing support, "
            "#10917).",
            n_eligible,
        )
        # No mutation yet -> report unchanged.
        return PassResult(graph_module, False)
