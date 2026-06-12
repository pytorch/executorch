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


def _is_static_nchw_with_channels(node: torch.fx.Node, channels: int) -> bool:
    value = node.meta.get("val")
    return (
        isinstance(value, torch.Tensor)
        and len(value.shape) == 4
        and int(value.shape[1]) == channels
    )


def _can_pad_c3_for_sampler(
    input_tensor: torch.fx.Node,
    interpolation_mode: int,
    align_corners: bool,
) -> bool:
    value = input_tensor.meta.get("val")
    return (
        isinstance(value, torch.Tensor)
        and len(value.shape) == 4
        and int(value.shape[0]) == 1
        and int(value.shape[1]) == 3
        and value.dtype is torch.float32
        and int(interpolation_mode) in (0, 1)
        and not bool(align_corners)
    )


class RewriteGridSamplerToTosaCustomPass(ArmPass):
    """Rewrite ``aten.grid_sampler_2d`` nodes to ``tosa.CUSTOM``."""

    targeted_ops = (exir_ops.edge.aten.grid_sampler_2d.default,)
    _passes_required_after: Set[Type[ExportPass]] = set()

    @staticmethod
    def _encode_payload(
        interpolation_mode: int,
        padding_mode: int,
        align_corners: bool,
        input_tensor: torch.fx.Node,
    ) -> list[int]:
        input_val = input_tensor.meta.get("val")
        if input_val is None:
            raise RuntimeError("grid_sampler_2d input is missing tensor metadata")
        payload = build_grid_sampler_2d_payload(
            interpolation_mode=interpolation_mode,
            padding_mode=padding_mode,
            align_corners=align_corners,
            input_shape=tuple(input_val.shape),
            input_dtype=input_val.dtype,
        )
        return encode_payload(payload)

    @staticmethod
    def _pad_c3_input_to_c4(
        graph_module: torch.fx.GraphModule,
        input_tensor: torch.fx.Node,
    ) -> torch.fx.Node:
        input_val = input_tensor.meta["val"]
        first_channel = create_node(
            graph_module.graph,
            op_target=exir_ops.edge.aten.slice_copy.Tensor,
            args=(input_tensor, 1, 0, 1, 1),
            from_node=input_tensor,
        )
        first_channel_val = exir_ops.edge.aten.slice_copy.Tensor(input_val, 1, 0, 1, 1)
        _set_fake_tensor_meta(first_channel, first_channel_val)

        zero_channel = create_node(
            graph_module.graph,
            op_target=exir_ops.edge.aten.sub.Tensor,
            args=(first_channel, first_channel),
            kwargs={"alpha": 1},
            from_node=input_tensor,
        )
        _set_fake_tensor_meta(
            zero_channel,
            exir_ops.edge.aten.sub.Tensor(first_channel_val, first_channel_val),
        )

        padded_input = create_node(
            graph_module.graph,
            op_target=exir_ops.edge.aten.cat.default,
            args=([input_tensor, zero_channel], 1),
            from_node=input_tensor,
        )
        _set_fake_tensor_meta(
            padded_input,
            exir_ops.edge.aten.cat.default([input_val, zero_channel.meta["val"]], 1),
        )
        return padded_input

    @staticmethod
    def _slice_c4_output_to_c3(
        graph_module: torch.fx.GraphModule,
        output: torch.fx.Node,
        original_node: torch.fx.Node,
    ) -> torch.fx.Node:
        output_val = output.meta["val"]
        sliced_output = create_node(
            graph_module.graph,
            op_target=exir_ops.edge.aten.slice_copy.Tensor,
            args=(output, 1, 0, 3, 1),
            from_node=original_node,
        )
        sliced_output.meta = dict(original_node.meta)
        _set_fake_tensor_meta(
            sliced_output,
            exir_ops.edge.aten.slice_copy.Tensor(output_val, 1, 0, 3, 1),
        )
        return sliced_output

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
            pad_c3_for_sampler = _can_pad_c3_for_sampler(
                input_tensor,
                interpolation_mode,
                align_corners,
            )

            operator_name = grid_sampler_2d_operator_name(
                interpolation_mode=interpolation_mode,
                padding_mode=padding_mode,
                align_corners=align_corners,
            )

            with graph_module.graph.inserting_before(node):
                custom_input = (
                    self._pad_c3_input_to_c4(graph_module, input_tensor)
                    if pad_c3_for_sampler
                    else input_tensor
                )
                implementation_attrs = self._encode_payload(
                    interpolation_mode=interpolation_mode,
                    padding_mode=padding_mode,
                    align_corners=align_corners,
                    input_tensor=custom_input,
                )
                nhwc_input = create_node(
                    graph_module.graph,
                    op_target=exir_ops.edge.aten.permute_copy.default,
                    args=(custom_input, list(NHWC_ORDER)),
                    from_node=custom_input,
                )
                _set_fake_tensor_meta(
                    nhwc_input,
                    exir_ops.edge.aten.permute_copy.default(
                        custom_input.meta["val"], list(NHWC_ORDER)
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
                if pad_c3_for_sampler:
                    with graph_module.graph.inserting_after(output):
                        replacement = self._slice_c4_output_to_c3(
                            graph_module, output, node
                        )
                else:
                    replacement = output
                node.replace_all_uses_with(replacement)
                graph_module.graph.erase_node(node)

        if modified:
            graph_module.graph.lint()
            graph_module.recompile()
            graph_module = super().call(graph_module).graph_module

        return PassResult(graph_module, modified)
