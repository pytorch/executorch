# Copyright 2026 Arm Limited and/or its affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import json
import operator
from typing import Set, Type

import torch
from executorch.backends.arm._passes import ArmPass
from executorch.backends.arm._passes.arm_pass_utils import create_node
from executorch.backends.arm.tosa.dialect.ops.custom import register_fake_tosa
from executorch.exir.dialects._ops import ops as exir_ops
from executorch.exir.pass_base import ExportPass, PassResult


@register_fake_tosa("grid_sampler_2d")
def _grid_sampler_2d_custom_fake_impl(
    inputs, operator_name, domain_name, implementation_attrs
) -> list[torch.Tensor]:
    _ = (operator_name, domain_name, implementation_attrs)
    input_tensor, grid = inputs
    output_shape = (
        input_tensor.shape[0],
        input_tensor.shape[1],
        grid.shape[1],
        grid.shape[2],
    )
    return [
        torch.empty(
            output_shape,
            dtype=input_tensor.dtype,
            device=input_tensor.device,
        )
    ]


class RewriteGridSamplerToTosaCustomPass(ArmPass):
    """Rewrite ``aten.grid_sampler_2d`` nodes to ``tosa.CUSTOM``."""

    targeted_ops = (exir_ops.edge.aten.grid_sampler_2d.default,)
    _passes_required_after: Set[Type[ExportPass]] = set()

    @staticmethod
    def _encode_payload(
        interpolation_mode: int, padding_mode: int, align_corners: bool
    ) -> list[int]:
        payload = {
            "version": 1,
            "type": "arm_custom_shader",
            "op": "grid_sampler_2d",
            "interpolation_mode": int(interpolation_mode),
            "padding_mode": int(padding_mode),
            "align_corners": bool(align_corners),
            "shader": {"encoding": "placeholder", "entry_point": "main"},
        }
        return list(json.dumps(payload, sort_keys=True).encode("utf-8"))

    def call(self, graph_module):
        modified = False
        for node in graph_module.graph.nodes:
            if (
                node.op != "call_function"
                or node.target != exir_ops.edge.aten.grid_sampler_2d.default
            ):
                continue

            modified = True
            input_tensor, grid, interpolation_mode, padding_mode, align_corners = (
                node.args
            )

            implementation_attrs = self._encode_payload(
                interpolation_mode=interpolation_mode,
                padding_mode=padding_mode,
                align_corners=align_corners,
            )

            with graph_module.graph.inserting_before(node):
                custom_node = create_node(
                    graph_module.graph,
                    op_target=exir_ops.backend.tosa.CUSTOM.default,
                    args=([input_tensor, grid],),
                    kwargs={
                        "operator_name": "grid_sampler_2d",
                        "domain_name": "arm.custom_shader",
                        "implementation_attrs": implementation_attrs,
                    },
                    from_node=node,
                    inherit_qparams=True,
                )

            with graph_module.graph.inserting_after(custom_node):
                getitem_node = graph_module.graph.create_node(
                    "call_function",
                    operator.getitem,
                    args=(custom_node, 0),
                    kwargs={},
                )
                # The getitem is a temporary FX node removed during TOSA
                # serialization. Keep the original tensor metadata until then.
                getitem_node.meta = dict(node.meta)
                node.replace_all_uses_with(getitem_node)
                graph_module.graph.erase_node(node)

        if modified:
            graph_module.graph.lint()
            graph_module.recompile()
            graph_module = super().call(graph_module).graph_module

        return PassResult(graph_module, modified)
