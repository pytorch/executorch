# Copyright 2026 Arm Limited and/or its affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import math
import operator
from typing import Set, Type

import torch
from executorch.backends.arm._passes import ArmPass
from executorch.backends.arm._passes.arm_pass_utils import create_node
from executorch.backends.arm._passes.fold_qdq_with_annotated_qparams_pass import (
    get_input_qparams,
    get_output_qparams,
)
from executorch.backends.arm._passes.quant_args import QuantArgs
from executorch.backends.arm.constants import NHWC_INVERSE_ORDER, NHWC_ORDER
from executorch.backends.arm.tosa.dialect.ops.custom import register_fake_tosa
from executorch.backends.arm.vgf.shaders.grid_sampler import (
    build_grid_sampler_2d_payload,
    CUSTOM_SHADER_DOMAIN_NAME,
    encode_payload,
    grid_sampler_2d_operator_name,
)
from executorch.backends.transforms.utils import create_constant_placeholder
from executorch.exir import ExportedProgram
from executorch.exir.dialects._ops import ops as exir_ops
from executorch.exir.pass_base import ExportPass, PassResult
from torch.export.graph_signature import InputKind
from torch.fx.passes.shape_prop import _extract_tensor_metadata


def _grid_sampler_2d_custom_fake_impl(
    inputs, operator_name, domain_name, implementation_attrs
) -> list[torch.Tensor]:
    _ = (operator_name, domain_name, implementation_attrs)
    input_tensor, grid, *_ = inputs
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
    align_corners: bool,  # noqa: ARG001
) -> bool:
    value = input_tensor.meta.get("val")
    return (
        isinstance(value, torch.Tensor)
        and len(value.shape) == 4
        and int(value.shape[0]) == 1
        and int(value.shape[1]) == 3
        and value.dtype in (torch.float32, torch.int8)
        and int(interpolation_mode) in (0, 1)
    )


def _uses_grid_sampler_int8_snorm_qparams(qparams: QuantArgs) -> bool:
    return (
        not qparams.per_channel
        and math.isclose(
            qparams.get_scale_per_tensor(), 1.0 / 127.0, rel_tol=1e-6, abs_tol=1e-9
        )
        and qparams.get_zp_per_tensor() == 0
        and qparams.qmin == -127
        and qparams.qmax == 127
        and qparams.dtype == torch.int8
    )


def _uses_grid_sampler_int8_snorm_metadata(node: torch.fx.Node) -> bool:
    try:
        input_qparams = get_input_qparams(node)
        output_qparams = get_output_qparams(node)
    except ValueError:
        return False

    image_qparams = input_qparams.get(0)
    if image_qparams is None:
        return False
    if not output_qparams:
        return False

    return _uses_grid_sampler_int8_snorm_qparams(
        image_qparams
    ) and _uses_grid_sampler_int8_snorm_qparams(next(iter(output_qparams.values())))


def _supports_quantized_grid_custom(qparams: QuantArgs) -> bool:
    return not qparams.per_channel and qparams.dtype == torch.int8


def _permute_to_nhwc(
    graph: torch.fx.Graph,
    tensor: torch.fx.Node,
    from_node: torch.fx.Node,
) -> torch.fx.Node:
    nhwc_tensor = create_node(
        graph,
        op_target=exir_ops.edge.aten.permute_copy.default,
        args=(tensor, list(NHWC_ORDER)),
        from_node=from_node,
    )
    _set_fake_tensor_meta(
        nhwc_tensor,
        exir_ops.edge.aten.permute_copy.default(tensor.meta["val"], list(NHWC_ORDER)),
    )
    return nhwc_tensor


class RewriteGridSamplerToTosaCustomPass(ArmPass):
    """Rewrite ``aten.grid_sampler_2d`` nodes to ``tosa.CUSTOM``."""

    targeted_ops = (exir_ops.edge.aten.grid_sampler_2d.default,)
    _passes_required_after: Set[Type[ExportPass]] = set()

    def __init__(
        self,
        exported_program: ExportedProgram | None = None,
        *args,
        **kwargs,
    ) -> None:
        super().__init__(*args, **kwargs)
        self.exported_program = exported_program

    @staticmethod
    def _encode_payload(
        interpolation_mode: int,
        padding_mode: int,
        align_corners: bool,
        input_tensor: torch.fx.Node,
        output_tensor: torch.fx.Node,
        output_dtype: torch.dtype | None = None,
        grid_dtype: torch.dtype | None = None,
        extra_tensor_input_vkformats: list[str] | None = None,
    ) -> list[int]:
        input_val = input_tensor.meta.get("val")
        if input_val is None:
            raise RuntimeError("grid_sampler_2d input is missing tensor metadata")
        output_val = output_tensor.meta.get("val")
        if output_val is None:
            raise RuntimeError("grid_sampler_2d output is missing tensor metadata")
        payload = build_grid_sampler_2d_payload(
            interpolation_mode=interpolation_mode,
            padding_mode=padding_mode,
            align_corners=align_corners,
            input_shape=tuple(input_val.shape),
            output_shape=tuple(output_val.shape),
            input_dtype=input_val.dtype,
            output_dtype=output_dtype,
            grid_dtype=grid_dtype,
            extra_tensor_input_vkformats=extra_tensor_input_vkformats,
        )
        return encode_payload(payload)

    def _get_first_user_input_placeholder(self, graph: torch.fx.Graph) -> torch.fx.Node:
        if self.exported_program is None:
            raise RuntimeError(
                "RewriteGridSamplerToTosaCustomPass requires ExportedProgram context "
                "to create constant placeholders"
            )
        user_input_names = {
            spec.arg.name
            for spec in self.exported_program.graph_signature.input_specs
            if spec.kind == InputKind.USER_INPUT
        }
        for graph_node in graph.nodes:
            if graph_node.op != "placeholder":
                continue
            if (
                graph_node.name in user_input_names
                or graph_node.target in user_input_names
            ):
                return graph_node
        raise RuntimeError("Failed to find a user input placeholder in the graph")

    def _create_grid_qparam_placeholders(
        self,
        graph: torch.fx.Graph,
        node: torch.fx.Node,
        grid_qparams: QuantArgs,
    ) -> tuple[torch.fx.Node, torch.fx.Node]:
        if self.exported_program is None:
            raise RuntimeError(
                "RewriteGridSamplerToTosaCustomPass requires ExportedProgram context "
                "to create qparam placeholders"
            )

        first_user_input = self._get_first_user_input_placeholder(graph)
        base_name = node.name.replace(".", "_")
        scale_name = f"{base_name}_grid_scale"
        zp_name = f"{base_name}_grid_zero_point"

        with graph.inserting_before(first_user_input):
            scale_node = create_constant_placeholder(
                self.exported_program,
                graph,
                scale_name,
                InputKind.CONSTANT_TENSOR,
                torch.tensor(
                    [grid_qparams.get_scale_per_tensor()], dtype=torch.float32
                ),
            )
            zp_node = create_constant_placeholder(
                self.exported_program,
                graph,
                zp_name,
                InputKind.CONSTANT_TENSOR,
                torch.tensor([grid_qparams.get_zp_per_tensor()], dtype=torch.int32),
            )

        return scale_node, zp_node

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

        padded_input = create_node(
            graph_module.graph,
            op_target=exir_ops.edge.aten.cat.default,
            args=([input_tensor, first_channel], 1),
            from_node=input_tensor,
        )
        _set_fake_tensor_meta(
            padded_input,
            exir_ops.edge.aten.cat.default([input_val, first_channel_val], 1),
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
            use_quantized_image_payload = _uses_grid_sampler_int8_snorm_metadata(node)
            output_dtype = torch.int8 if use_quantized_image_payload else None
            grid_qparams = None
            grid_qparam_constants: tuple[torch.fx.Node, torch.fx.Node] | None = None
            quantized_grid = not grid.meta["val"].dtype.is_floating_point
            pad_c3_for_sampler = _can_pad_c3_for_sampler(
                input_tensor,
                interpolation_mode,
                align_corners,
            )
            if quantized_grid:
                grid_qparams = get_input_qparams(node).get(1)
                if grid_qparams is None:
                    raise RuntimeError(
                        "Quantized grid_sampler rewrite is missing grid input qparams"
                    )
                if not _supports_quantized_grid_custom(grid_qparams):
                    raise RuntimeError(
                        "grid_sampler rewrite only supports per-tensor int8 grids; "
                        "unsupported qparams should have been dequantized earlier"
                    )
                grid_qparam_constants = self._create_grid_qparam_placeholders(
                    graph_module.graph, node, grid_qparams
                )
            if use_quantized_image_payload and grid_qparam_constants is None:
                raise RuntimeError(
                    "grid_sampler int8 sampler rewrite requires a quantized int8 "
                    "grid with explicit scale/zero-point shader inputs"
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
                    output_tensor=node,
                    output_dtype=output_dtype,
                    grid_dtype=(
                        grid.meta["val"].dtype if grid_qparams is not None else None
                    ),
                    extra_tensor_input_vkformats=(
                        ["VK_FORMAT_R32_SFLOAT", "VK_FORMAT_R32_SINT"]
                        if grid_qparam_constants is not None
                        else None
                    ),
                )
                nhwc_input = _permute_to_nhwc(
                    graph_module.graph,
                    custom_input,
                    custom_input,
                )
                custom_inputs = [nhwc_input, grid]
                if grid_qparam_constants is not None:
                    custom_inputs.extend(grid_qparam_constants)
                custom_node = create_node(
                    graph_module.graph,
                    op_target=exir_ops.backend.tosa.CUSTOM.default,
                    args=(custom_inputs,),
                    kwargs={
                        "operator_name": operator_name,
                        "domain_name": CUSTOM_SHADER_DOMAIN_NAME,
                        "implementation_attrs": implementation_attrs,
                    },
                    from_node=node,
                    inherit_qparams=True,
                )
                if grid_qparams is not None and "input_qparams" in custom_node.meta:
                    custom_node.meta["input_qparams"] = {
                        idx: qargs
                        for idx, qargs in custom_node.meta["input_qparams"].items()
                        if idx != 1
                    }
            with graph_module.graph.inserting_after(custom_node):
                getitem_node = graph_module.graph.create_node(
                    "call_function",
                    operator.getitem,
                    args=(custom_node, 0),
                    kwargs={},
                )
                custom_output = _grid_sampler_2d_custom_fake_impl(
                    [input_node.meta["val"] for input_node in custom_inputs],
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
            graph_module.graph.eliminate_dead_code()
            graph_module.graph.lint()
            graph_module.recompile()
            graph_module = super().call(graph_module).graph_module

        return PassResult(graph_module, modified)
