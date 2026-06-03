# Copyright 2026 Arm Limited and/or its affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import operator
from typing import Set, Type

import torch
from executorch.backends.arm._passes import ArmPass
from executorch.backends.arm._passes.arm_pass_utils import create_node
from executorch.backends.arm.constants import NHWC_INVERSE_ORDER, NHWC_ORDER
from executorch.backends.arm.tosa.dialect.ops.custom import register_fake_tosa
from executorch.backends.arm.vgf.shaders.grid_sampler import (
    build_grid_sampler_2d_payload,
    CUSTOM_SHADER_DOMAIN_NAME,
    encode_payload,
    grid_sampler_2d_operator_name,
)
from executorch.exir.dialects._ops import ops as exir_ops
from executorch.exir.pass_base import ExportPass, PassResult
from torch.fx.passes.shape_prop import _extract_tensor_metadata


def _grid_sampler_2d_custom_fake_impl(
    inputs, operator_name, domain_name, implementation_attrs
) -> list[torch.Tensor]:
    _ = (operator_name, domain_name, implementation_attrs)
    input_tensor, grid = inputs
    return [
        torch.empty(
            (
                input_tensor.shape[0],
                grid.shape[1],
                grid.shape[2],
                input_tensor.shape[-1],
            ),
            dtype=input_tensor.dtype,
            device=input_tensor.device,
        )
    ]


def _register_grid_sampler_2d_custom_fake_impl(
    interpolation_mode: int,
    padding_mode: int,
    align_corners: bool,
) -> None:
    operator_name = grid_sampler_2d_operator_name(
        interpolation_mode=interpolation_mode,
        padding_mode=padding_mode,
        align_corners=align_corners,
    )

    def _grid_sampler_2d_custom_fake_impl_variant(
        inputs, operator_name, domain_name, implementation_attrs
    ) -> list[torch.Tensor]:
        return _grid_sampler_2d_custom_fake_impl(
            inputs,
            operator_name,
            domain_name,
            implementation_attrs,
        )

    register_fake_tosa(operator_name)(_grid_sampler_2d_custom_fake_impl_variant)


for interpolation_mode in (0, 1, 2):
    for padding_mode in (0, 1, 2):
        for align_corners in (False, True):
            _register_grid_sampler_2d_custom_fake_impl(
                interpolation_mode=interpolation_mode,
                padding_mode=padding_mode,
                align_corners=align_corners,
            )


def _set_fake_tensor_meta(node: torch.fx.Node, value) -> None:
    node.meta["val"] = value
    if isinstance(value, list):
        if value:
            node.meta["tensor_meta"] = _extract_tensor_metadata(value[0])
    else:
        node.meta["tensor_meta"] = _extract_tensor_metadata(value)


class RewriteGridSamplerToTosaCustomPass(ArmPass):
    """Rewrite ``aten.grid_sampler_2d`` nodes to ``tosa.CUSTOM``."""

    targeted_ops = (exir_ops.edge.aten.grid_sampler_2d.default,)
    _passes_required_after: Set[Type[ExportPass]] = set()

    @staticmethod
    def _encode_payload(
        interpolation_mode: int, padding_mode: int, align_corners: bool
    ) -> list[int]:
        payload = build_grid_sampler_2d_payload(
            interpolation_mode=interpolation_mode,
            padding_mode=padding_mode,
            align_corners=align_corners,
        )
        return encode_payload(payload)

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
            operator_name = grid_sampler_2d_operator_name(
                interpolation_mode=interpolation_mode,
                padding_mode=padding_mode,
                align_corners=align_corners,
            )

            with graph_module.graph.inserting_before(node):
                nhwc_input = create_node(
                    graph_module.graph,
                    op_target=exir_ops.edge.aten.permute_copy.default,
                    args=(input_tensor, list(NHWC_ORDER)),
                    from_node=input_tensor,
                )
                _set_fake_tensor_meta(
                    nhwc_input,
                    exir_ops.edge.aten.permute_copy.default(
                        input_tensor.meta["val"], list(NHWC_ORDER)
                    ),
                )

                custom_node = create_node(
                    graph_module.graph,
                    op_target=exir_ops.backend.tosa.CUSTOM.default,
                    args=([nhwc_input, grid],),
                    kwargs={
                        "operator_name": operator_name,
                        "domain_name": CUSTOM_SHADER_DOMAIN_NAME,
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
                custom_output = _grid_sampler_2d_custom_fake_impl(
                    [nhwc_input.meta["val"], grid.meta["val"]],
                    operator_name,
                    CUSTOM_SHADER_DOMAIN_NAME,
                    implementation_attrs,
                )[0]
                _set_fake_tensor_meta(custom_node, [custom_output])
                getitem_node.meta = dict(node.meta)
                _set_fake_tensor_meta(getitem_node, custom_output)

            with graph_module.graph.inserting_after(getitem_node):
                output = create_node(
                    graph_module.graph,
                    op_target=exir_ops.edge.aten.permute_copy.default,
                    args=(getitem_node, list(NHWC_INVERSE_ORDER)),
                    from_node=node,
                )
                output.meta = dict(node.meta)
                _set_fake_tensor_meta(
                    output,
                    exir_ops.edge.aten.permute_copy.default(
                        custom_output, list(NHWC_INVERSE_ORDER)
                    ),
                )
                node.replace_all_uses_with(output)
                graph_module.graph.erase_node(node)

        if modified:
            graph_module.graph.lint()
            graph_module.recompile()
            graph_module = super().call(graph_module).graph_module

        return PassResult(graph_module, modified)
