# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
reinplace pass for the native backend.

reinplace rewrites functional ops into their in-place forms (e.g. relu -> relu_),
which are not in the core-ATen opset, so callers must disable IR validity
checking via ``EdgeCompileConfig(_check_ir_validity=False)`` (the same config the
MLX backend requires).
"""

from typing import Tuple

import torch

from executorch.exir.dialects._ops import ops as _edge_ops
from executorch.exir.pass_base import ExportedProgramPassBase, ExportedProgramPassResult

_edge = _edge_ops.edge.aten

# Single source of truth: base names of the ops the backend can execute in-place.
# reinplace rewrites the functional edge op to its mutating variant (e.g.
# relu -> relu_), and the partitioner must claim that variant to keep it in the
# delegate (see backend_inplace_aten_variants).
_INPLACE_OP_NAMES: Tuple[str, ...] = (
    "relu",
    "gelu",
    "sigmoid",
    "index_put",
    "index_copy",
)

# Edge ops the backend can execute in-place, consumed as reinplace's op set.
BACKEND_INPLACE_OPS: frozenset = frozenset(
    getattr(_edge, name).default for name in _INPLACE_OP_NAMES
)


def backend_inplace_aten_variants() -> set:
    """Mutating aten variants that reinplace introduces for BACKEND_INPLACE_OPS.

    reinplace rewrites e.g. relu -> aten.relu_.default; those in-place ops are
    not core-tagged, so the partitioner must claim them explicitly to keep them
    inside the delegate.
    """
    ops = set()
    for name in _INPLACE_OP_NAMES:
        packet = getattr(torch.ops.aten, f"{name}_", None)
        overload = getattr(packet, "default", None) if packet is not None else None
        if overload is not None:
            ops.add(overload)
    return ops


class NativeReinplacePass(ExportedProgramPassBase):
    """Rewrite backend-supported elementwise ops into their in-place edge forms.

    Runs as an EP-aware pass because ``reinplace_pass`` needs ``graph_signature``
    to protect mutable inputs/buffers. A fresh op set is passed each call so
    ``reinplace_pass`` cannot mutate shared state, and passing an explicit
    ``ops_to_inplace`` fully replaces reinplace's default set.
    """

    def call(self, exported_program) -> ExportedProgramPassResult:
        from executorch.exir.passes.reinplace import reinplace_pass

        exported_program = reinplace_pass(
            exported_program, ops_to_inplace=set(BACKEND_INPLACE_OPS)
        )
        return ExportedProgramPassResult(exported_program, True)
