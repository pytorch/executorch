# Copyright 2026 Arm Limited and/or its affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from typing import Set, Type

import torch
from executorch.backends.arm._passes.arm_pass import ArmPass
from executorch.backends.arm.tosa.mapping import TosaSpecialDtype
from executorch.exir import to_edge
from executorch.exir.dialects._ops import ops as exir_ops
from executorch.exir.pass_base import ExportPass


class _EmitShapePass(ArmPass):
    @property
    def _passes_required_after(self) -> Set[Type[ExportPass]]:
        return set()

    def call_operator(self, op, args, kwargs, meta, updated: bool | None = False):
        # Inject a CONST_SHAPE once, then proceed normally.
        print(f"op: {op}")
        if op == exir_ops.edge.aten.add.Tensor:
            print("Injecting CONST_SHAPE")
            shape = super().call_shape_operator(
                exir_ops.backend.tosa.CONST_SHAPE.default,
                ([1, 3],),
                {},
                meta,
                True,
            )
            return shape
        else:
            return super().call_operator(op, args, kwargs, meta, updated)


def test_const_shape_injects_meta_no_target():
    class M(torch.nn.Module):
        def forward(self, x):
            return x + 1

    exported = torch.export.export(M(), (torch.randn(1),))
    edge = to_edge(exported).transform([_EmitShapePass()])

    gm = edge.exported_program().graph_module

    const_shape_nodes = [
        n
        for n in gm.graph.nodes
        if n.target == exir_ops.backend.tosa.CONST_SHAPE.default
    ]

    assert const_shape_nodes
    for n in const_shape_nodes:
        assert n.meta[TosaSpecialDtype.meta_key()] == TosaSpecialDtype.SHAPE
