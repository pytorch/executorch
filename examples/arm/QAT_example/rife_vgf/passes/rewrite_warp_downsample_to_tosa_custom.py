# Copyright 2026 Arm Limited and/or its affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import math
import operator
from typing import Set, Type

import torch
from examples.arm.QAT_example.rife_vgf.shaders import (
    build_warp_downsample_payload,
    warp_downsample_operator_name,
)
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
    CUSTOM_SHADER_DOMAIN_NAME,
    encode_payload,
)
from executorch.exir.dialects._ops import ops as exir_ops
from executorch.exir.pass_base import ExportPass, PassResult
from torch.fx.passes.shape_prop import _extract_tensor_metadata


def _target_name(target: object) -> str | None:
    schema = getattr(target, "_schema", None)
    schema_name = getattr(schema, "name", None)
    overload_name = getattr(schema, "overload_name", None)
    if isinstance(schema_name, str) and schema_name.startswith("rife::"):
        name = schema_name.replace("::", ".", 1)
        if overload_name in (None, "", "default"):
            return f"{name}.default"
        if isinstance(overload_name, str):
            return f"{name}.{overload_name}"

    target_name = getattr(target, "__name__", None)
    if isinstance(target_name, str):
        if target_name.startswith("rife."):
            return target_name
        namespace = getattr(target, "namespace", None)
        if isinstance(namespace, str):
            return f"{namespace}.{target_name}"
    return None


def _target_scale(target: object) -> int | None:
    target_name = _target_name(target)
    for scale in (2, 4, 8):
        if target_name == f"rife.warp_downsample{scale}.default":
            return scale
    return None


def _warp_downsample_custom_fake_impl(
    inputs, operator_name, domain_name, implementation_attrs
) -> list[torch.Tensor]:
    assert domain_name == CUSTOM_SHADER_DOMAIN_NAME
    _ = implementation_attrs
    input_tensor, flow = inputs
    _ = flow
    scale = None
    for candidate in (2, 4, 8):
        if operator_name == warp_downsample_operator_name(candidate):
            scale = candidate
            break
    assert scale is not None
    return [
        torch.empty(
            (
                input_tensor.shape[0],
                input_tensor.shape[1] // scale,
                input_tensor.shape[2] // scale,
                input_tensor.shape[-1],
            ),
            dtype=input_tensor.dtype,
            device=input_tensor.device,
        )
    ]


for _scale in (2, 4, 8):
    register_fake_tosa(warp_downsample_operator_name(_scale))(
        _warp_downsample_custom_fake_impl
    )


def _set_fake_tensor_meta(node: torch.fx.Node, value) -> None:
    node.meta["val"] = value
    if isinstance(value, list):
        if value:
            node.meta["tensor_meta"] = _extract_tensor_metadata(value[0])
    else:
        node.meta["tensor_meta"] = _extract_tensor_metadata(value)


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


def _node_precedes(node: torch.fx.Node, reference: torch.fx.Node) -> bool:
    if node.graph is not reference.graph:
        return False
    for candidate in node.graph.nodes:
        if candidate is node:
            return True
        if candidate is reference:
            return False
    return False


def _is_supported_input(input_tensor: torch.fx.Node) -> bool:
    value = input_tensor.meta.get("val")
    return (
        isinstance(value, torch.Tensor)
        and len(value.shape) == 4
        and int(value.shape[0]) == 1
        and int(value.shape[1]) in (3, 4)
        and value.dtype == torch.int8
    )


def _pad_c3_to_c4(
    graph: torch.fx.Graph,
    input_tensor: torch.fx.Node,
    from_node: torch.fx.Node,
) -> torch.fx.Node:
    input_val = input_tensor.meta["val"]
    for user in input_tensor.users:
        if (
            user.op == "call_function"
            and user.target == exir_ops.edge.aten.constant_pad_nd.default
            and len(user.args) >= 3
            and user.args[0] is input_tensor
            and isinstance(user.args[1], (list, tuple))
            and list(user.args[1]) == [0, 0, 0, 0, 0, 1]
            and user.args[2] == 0
            and _node_precedes(user, from_node)
        ):
            user_val = user.meta.get("val")
            if (
                isinstance(user_val, torch.Tensor)
                and len(user_val.shape) == 4
                and tuple(user_val.shape[:1]) == tuple(input_val.shape[:1])
                and int(user_val.shape[1]) == 4
                and tuple(user_val.shape[2:]) == tuple(input_val.shape[2:])
            ):
                return user

    padded = create_node(
        graph,
        op_target=exir_ops.edge.aten.constant_pad_nd.default,
        args=(input_tensor, [0, 0, 0, 0, 0, 1], 0),
        from_node=from_node,
        inherit_qparams=True,
    )
    _set_fake_tensor_meta(
        padded,
        exir_ops.edge.aten.constant_pad_nd.default(
            input_tensor.meta["val"], [0, 0, 0, 0, 0, 1], 0
        ),
    )
    return padded


def _slice_c4_to_c3(
    graph: torch.fx.Graph,
    output_tensor: torch.fx.Node,
    from_node: torch.fx.Node,
) -> torch.fx.Node:
    sliced = create_node(
        graph,
        op_target=exir_ops.edge.aten.slice_copy.Tensor,
        args=(output_tensor, 1, 0, 3),
        from_node=from_node,
        inherit_qparams=True,
    )
    _set_fake_tensor_meta(
        sliced,
        exir_ops.edge.aten.slice_copy.Tensor(output_tensor.meta["val"], 1, 0, 3),
    )
    return sliced


def _flow_arg_and_channel_offset(
    flow: torch.fx.Node,
) -> tuple[torch.fx.Node, int]:
    if (
        flow.op != "call_function"
        or flow.target != exir_ops.edge.aten.slice_copy.Tensor
        or len(flow.args) < 4
    ):
        return flow, 0

    flow_arg, dim, start, end, *step = flow.args
    if (
        not isinstance(flow_arg, torch.fx.Node)
        or dim != 1
        or not isinstance(start, int)
        or not isinstance(end, int)
        or (start, end) not in ((0, 2), (2, 4))
        or step not in ([], [1])
        or flow.kwargs.get("step", 1) != 1
    ):
        return flow, 0

    flow_arg_value = flow_arg.meta.get("val")
    flow_value = flow.meta.get("val")
    if (
        isinstance(flow_arg_value, torch.Tensor)
        and isinstance(flow_value, torch.Tensor)
        and len(flow_arg_value.shape) == 4
        and len(flow_value.shape) == 4
        and tuple(flow_arg_value.shape[:2]) == (1, 4)
        and tuple(flow_value.shape[:2]) == (1, 2)
    ):
        return flow_arg, int(start)
    return flow, 0


def _restore_normalized_boundary_input(input_tensor: torch.fx.Node) -> torch.fx.Node:
    """Undo delegate IO layout normalization for sampler-backed C4 inputs.

    NormalizeDelegateIOLayoutPass exposes channels-last tensors at delegate
    boundaries by changing the placeholder shape to NHWC and inserting an
    inverse permute back to NCHW. For warp_downsample, the custom shader rewrite
    then inserts an NCHW->NHWC permute, and later cleanup can cancel the pair,
    leaving a public COMBINED_IMAGE_SAMPLER input. Keep the public boundary as a
    tensor by using the original placeholder as NCHW and letting the rewrite add
    the shader-local NHWC permute.

    """
    if (
        input_tensor.op != "call_function"
        or input_tensor.target != exir_ops.edge.aten.permute_copy.default
        or len(input_tensor.args) < 2
        or len(input_tensor.users) != 1
    ):
        return input_tensor
    permutation = input_tensor.args[1]
    if not isinstance(permutation, (list, tuple)) or list(permutation) != list(
        NHWC_INVERSE_ORDER
    ):
        return input_tensor

    source = input_tensor.args[0]
    if (
        not isinstance(source, torch.fx.Node)
        or source.op != "placeholder"
        or len(source.users) != 1
    ):
        return input_tensor

    input_value = input_tensor.meta.get("val")
    source_value = source.meta.get("val")
    if (
        not isinstance(input_value, torch.Tensor)
        or not isinstance(source_value, torch.Tensor)
        or tuple(source_value.shape)
        != tuple(input_value.shape[axis] for axis in NHWC_ORDER)
    ):
        return input_tensor

    source.meta["val"] = input_value
    source.meta["tensor_meta"] = _extract_tensor_metadata(input_value)
    for key in ("input_qparams", "output_qparams"):
        if key in input_tensor.meta:
            source.meta[key] = input_tensor.meta[key]
    return source


def _is_supported_flow(flow: torch.fx.Node, flow_channel_offset: int) -> bool:
    value = flow.meta.get("val")
    return (
        isinstance(value, torch.Tensor)
        and len(value.shape) == 4
        and int(value.shape[0]) == 1
        and int(value.shape[1]) >= flow_channel_offset + 2
        and int(value.shape[1]) in (2, 4)
        and value.dtype == torch.int8
    )


def _has_supported_shape_contract(
    input_tensor: torch.fx.Node,
    flow: torch.fx.Node,
    output: torch.fx.Node,
    scale: int,
    flow_channel_offset: int,
) -> bool:
    input_value = input_tensor.meta.get("val")
    flow_value = flow.meta.get("val")
    output_value = output.meta.get("val")
    if not (
        isinstance(input_value, torch.Tensor)
        and isinstance(flow_value, torch.Tensor)
        and isinstance(output_value, torch.Tensor)
        and len(input_value.shape) == 4
        and len(flow_value.shape) == 4
        and len(output_value.shape) == 4
    ):
        return False

    input_n, input_c, input_h, input_w = (int(dim) for dim in input_value.shape)
    flow_n, flow_c, flow_h, flow_w = (int(dim) for dim in flow_value.shape)
    output_n, output_c, output_h, output_w = (int(dim) for dim in output_value.shape)
    return (
        input_n == flow_n == output_n == 1
        and input_c == output_c
        and flow_c >= flow_channel_offset + 2
        and flow_h == input_h
        and flow_w == input_w
        and input_h % scale == 0
        and input_w % scale == 0
        and output_h == input_h // scale
        and output_w == input_w // scale
    )


def _uses_int8_snorm_qparams(qparams: QuantArgs) -> bool:
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


def _uses_warp_downsample_int8_snorm_metadata(node: torch.fx.Node) -> bool:
    try:
        input_qparams = get_input_qparams(node)
        output_qparams = get_output_qparams(node)
    except ValueError:
        return False
    image_qparams = input_qparams.get(0)
    if image_qparams is None or not output_qparams:
        return False
    return _uses_int8_snorm_qparams(image_qparams) and _uses_int8_snorm_qparams(
        next(iter(output_qparams.values()))
    )


class RewriteWarpDownsampleToTosaCustomPass(ArmPass):
    """Rewrite ``rife.warp_downsample{2,4,8}`` nodes to ``tosa.CUSTOM``."""

    _passes_required_after: Set[Type[ExportPass]] = set()

    @staticmethod
    def _encode_payload(
        scale: int,
        input_tensor: torch.fx.Node,
        flow_tensor: torch.fx.Node,
        output_tensor: torch.fx.Node,
        output_shape: tuple[int, ...] | None = None,
        output_dtype: torch.dtype | None = None,
        flow_qparams: QuantArgs | None = None,
        flow_channel_offset: int = 0,
    ) -> list[int]:
        input_val = input_tensor.meta.get("val")
        flow_val = flow_tensor.meta.get("val")
        output_val = output_tensor.meta.get("val")
        if input_val is None or flow_val is None or output_val is None:
            raise RuntimeError("warp_downsample node is missing tensor metadata")
        if flow_qparams is None:
            raise RuntimeError("int8 warp_downsample flow is missing input qparams")
        payload = build_warp_downsample_payload(
            scale=scale,
            input_shape=tuple(input_val.shape),
            output_shape=(
                output_shape if output_shape is not None else tuple(output_val.shape)
            ),
            input_dtype=input_val.dtype,
            output_dtype=output_dtype if output_dtype is not None else output_val.dtype,
            flow_dtype=flow_val.dtype,
            flow_scale=(
                flow_qparams.get_scale_per_tensor()
                if flow_qparams is not None
                else None
            ),
            flow_zero_point=(
                flow_qparams.get_zp_per_tensor() if flow_qparams is not None else None
            ),
            flow_channel_offset=flow_channel_offset,
        )
        return encode_payload(payload)

    def call(self, graph_module):  # noqa: C901
        modified = False
        for node in list(graph_module.graph.nodes):
            if node.op != "call_function":
                continue
            scale = _target_scale(node.target)
            if scale is None:
                continue

            if len(node.args) != 2:
                raise RuntimeError("warp_downsample VGF rewrite requires two inputs")
            input_tensor, flow = node.args
            flow, flow_channel_offset = _flow_arg_and_channel_offset(flow)
            try:
                input_qparams = get_input_qparams(node)
            except ValueError:
                input_qparams = {}
            flow_qparams = input_qparams.get(1)
            use_quantized_image_payload = _uses_warp_downsample_int8_snorm_metadata(
                node
            )
            if not use_quantized_image_payload:
                raise RuntimeError(
                    "warp_downsample int8 VGF rewrite requires SNORM qparams "
                    "scale=1/127, zp=0, qmin=-127, qmax=127 on input/output"
                )
            if flow_qparams is None:
                raise RuntimeError("int8 warp_downsample flow is missing input qparams")
            output_dtype = torch.int8
            if not _is_supported_input(input_tensor):
                raise RuntimeError(
                    "warp_downsample VGF rewrite requires int8 NCHW "
                    "input [1, 3 or 4, H, W]"
                )
            if not _is_supported_flow(flow, flow_channel_offset):
                raise RuntimeError(
                    "warp_downsample VGF rewrite requires int8 NCHW flow "
                    "[1, 2 or 4, H, W] with enough channels for flow_channel_offset"
                )
            if not _has_supported_shape_contract(
                input_tensor, flow, node, scale, flow_channel_offset
            ):
                raise RuntimeError(
                    "warp_downsample VGF rewrite requires input [1, C, H, W], "
                    "flow [1, 2 or 4, H, W], and output "
                    "[1, C, H / scale, W / scale]"
                )

            operator_name = warp_downsample_operator_name(scale)
            output_channel_count = int(node.meta["val"].shape[1])
            with graph_module.graph.inserting_before(node):
                input_tensor = _restore_normalized_boundary_input(input_tensor)
                custom_input = input_tensor
                if int(custom_input.meta["val"].shape[1]) == 3:
                    custom_input = _pad_c3_to_c4(graph_module.graph, custom_input, node)
                custom_output_shape = tuple(node.meta["val"].shape)
                if output_channel_count == 3:
                    custom_output_shape = (
                        int(node.meta["val"].shape[0]),
                        4,
                        int(node.meta["val"].shape[2]),
                        int(node.meta["val"].shape[3]),
                    )
                implementation_attrs = self._encode_payload(
                    scale,
                    custom_input,
                    flow,
                    node,
                    output_shape=custom_output_shape,
                    output_dtype=output_dtype,
                    flow_qparams=flow_qparams,
                    flow_channel_offset=flow_channel_offset,
                )
                nhwc_input = _permute_to_nhwc(
                    graph_module.graph,
                    custom_input,
                    custom_input,
                )
                custom_node = create_node(
                    graph_module.graph,
                    op_target=exir_ops.backend.tosa.CUSTOM.default,
                    args=([nhwc_input, flow],),
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
                custom_output = _warp_downsample_custom_fake_impl(
                    [nhwc_input.meta["val"], flow.meta["val"]],
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
                nchw_custom_output = exir_ops.edge.aten.permute_copy.default(
                    custom_output, list(NHWC_INVERSE_ORDER)
                )
                _set_fake_tensor_meta(output, nchw_custom_output)

            if output_channel_count == 3:
                with graph_module.graph.inserting_after(output):
                    output = _slice_c4_to_c3(graph_module.graph, output, node)
                    output.meta = dict(node.meta)
                    _set_fake_tensor_meta(output, node.meta["val"])

            node.replace_all_uses_with(output)
            graph_module.graph.erase_node(node)
            modified = True

        if modified:
            graph_module.graph.eliminate_dead_code()
            graph_module.graph.lint()
            graph_module.recompile()
            graph_module = super().call(graph_module).graph_module

        return PassResult(graph_module, modified)
