# Copyright 2026 Arm Limited and/or its affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.
"""Fuse canonical quantized flow-offset grid sampling for VGF.

The pass matches Edge-dialect graphs equivalent to::

    base_grid = cat(expand(linspace_x), expand(linspace_y))
    grid = permute(base_grid + flow[:, i:i + 2], [0, 2, 3, 1])
    output = grid_sample(image, grid, bilinear, border, align_corners=True)

The base grid may pass through an identity dequantize/quantize pair. Images
must have shape ``[1, 3|4, H, W]``, flow must have shape ``[1, 4, H, W]``,
and ``i`` must be zero or two. Image and output quantization ranges must be
``[-127, 127]`` because the custom shader stores them in SNORM images. The
flow tensor may use the full int8 range because the shader reads it directly.

"""

import logging
import math
import operator
from typing import TypeGuard

import torch
from executorch.backends.arm._passes.fold_qdq_with_annotated_qparams_pass import (
    get_input_qparams,
    get_output_qparams,
)
from executorch.backends.arm._passes.quant_args import QuantArgs
from executorch.backends.arm.constants import NHWC_INVERSE_ORDER, NHWC_ORDER
from executorch.backends.arm.tosa.dialect.ops.custom import register_fake_tosa
from executorch.backends.arm.vgf._passes.rewrite_grid_sampler_to_tosa_custom import (
    _set_fake_tensor_meta,
    RewriteGridSamplerToTosaCustomPass,
)
from executorch.backends.arm.vgf.shaders.grid_sampler import (
    _FlowOffsetShaderCompilationError,
    build_flow_offset_grid_sampler_payload,
    CUSTOM_SHADER_DOMAIN_NAME,
    encode_payload,
    flow_offset_grid_sampler_operator_name,
)
from executorch.exir.dialects._ops import ops as exir_ops
from executorch.exir.pass_base import PassResult
from torch.fx import GraphModule, Node

logger = logging.getLogger(__name__)


def _flow_offset_grid_sampler_fake_impl(
    inputs, operator_name, domain_name, implementation_attrs
) -> list[torch.Tensor]:
    _ = (operator_name, domain_name, implementation_attrs)
    return [torch.empty_like(inputs[0])]


register_fake_tosa(flow_offset_grid_sampler_operator_name())(
    _flow_offset_grid_sampler_fake_impl
)


def _target_is(node: object, target: str) -> TypeGuard[Node]:
    return isinstance(node, Node) and target in str(node.target).replace("::", ".")


def _literal_list(value: object) -> list[int] | None:
    if not isinstance(value, (list, tuple)):
        return None
    if not all(isinstance(item, int) for item in value):
        return None
    return [item for item in value if isinstance(item, int)]


def _is_identity_requantization(
    source_qparams: tuple[object, ...], destination_qparams: tuple[object, ...]
) -> bool:
    """Return whether requantization preserves every source integer value.

    Requantization is affine, so checking that both source range endpoints
    remain within half a quantization unit proves that rounding preserves all
    intermediate values. The destination range must also contain the source
    range to prevent clipping.

    Args:
        source_qparams (tuple[object, ...]): Source per-tensor quantization
            parameters.
        destination_qparams (tuple[object, ...]): Destination per-tensor
            quantization parameters.

    Returns:
        bool: Whether requantization is an identity for the source range.

    """
    if len(source_qparams) != 5 or len(destination_qparams) != 5:
        return False
    source_scale, source_zp, source_qmin, source_qmax, source_dtype = source_qparams
    (
        destination_scale,
        destination_zp,
        destination_qmin,
        destination_qmax,
        destination_dtype,
    ) = destination_qparams
    if (
        not isinstance(source_scale, (int, float))
        or not isinstance(destination_scale, (int, float))
        or not isinstance(source_zp, int)
        or not isinstance(destination_zp, int)
        or not isinstance(source_qmin, int)
        or not isinstance(source_qmax, int)
        or not isinstance(destination_qmin, int)
        or not isinstance(destination_qmax, int)
        or source_dtype != destination_dtype
        or not math.isfinite(source_scale)
        or not math.isfinite(destination_scale)
        or source_scale <= 0
        or destination_scale <= 0
        or source_qmin > source_qmax
        or destination_qmin > source_qmin
        or destination_qmax < source_qmax
    ):
        return False

    # QAT scales may drift slightly while the requantization remains an identity.
    for value in (source_qmin, source_qmax):
        destination_value = (
            value - source_zp
        ) * source_scale / destination_scale + destination_zp
        if abs(destination_value - value) >= 0.5:
            return False
    return True


def _unwrap_quantized_base_grid(node: Node) -> Node | None:
    """Unwrap an identity dequantize-quantize pair around a base grid.

    Args:
        node (Node): Candidate quantize node.

    Returns:
        Node | None: Unwrapped base-grid node, or ``None`` when the wrapper
        changes integer values or has an unsupported form.

    """
    if not _target_is(node, "quantized_decomposed.quantize_per_tensor.default"):
        return None
    dequantize = node.args[0]
    if not _target_is(dequantize, "quantized_decomposed.dequantize_per_tensor.default"):
        return None
    if (
        len(node.args) != 6
        or len(dequantize.args) != 6
        or not _is_identity_requantization(
            tuple(dequantize.args[1:]), tuple(node.args[1:])
        )
        or dequantize.kwargs.get("out_dtype") not in (None, torch.float32)
    ):
        return None
    base_grid = dequantize.args[0]
    return base_grid if isinstance(base_grid, Node) else None


def _matches_linspace_axis(
    node: object,
    *,
    steps: int,
    view_shape: list[int],
    expand_shape: list[int],
) -> bool:
    """Match one expanded ``linspace(-1, 1)`` canonical-grid axis.

    Args:
        node (object): Candidate expand node.
        steps (int): Expected number of linspace steps.
        view_shape (list[int]): Expected shape before expansion.
        expand_shape (list[int]): Expected expanded shape.

    Returns:
        bool: Whether the node matches the expected axis construction.

    """
    if not _target_is(node, "aten.expand_copy.default"):
        return False
    view = node.args[0]
    actual_expand_shape = _literal_list(node.args[1])
    if not _target_is(view, "aten.view_copy.default"):
        return False
    linspace = view.args[0]
    actual_view_shape = _literal_list(view.args[1])
    if not _target_is(linspace, "aten.linspace.default"):
        return False
    if linspace.kwargs.get("dtype") not in (None, torch.float32):
        return False
    if tuple(linspace.args[:3]) != (-1.0, 1.0, steps):
        return False
    return actual_view_shape == view_shape and actual_expand_shape == expand_shape


def _matches_canonical_base_grid(node: Node, *, height: int, width: int) -> bool:
    """Match a quantized canonical XY grid built from two linspace axes.

    Args:
        node (Node): Candidate quantized base-grid node.
        height (int): Expected grid height.
        width (int): Expected grid width.

    Returns:
        bool: Whether the node constructs the expected canonical base grid.

    """
    base_grid = _unwrap_quantized_base_grid(node)
    if not _target_is(base_grid, "aten.cat.default"):
        return False
    tensors = base_grid.args[0]
    cat_dim = base_grid.args[1]
    return (
        isinstance(tensors, (list, tuple))
        and len(tensors) == 2
        and isinstance(cat_dim, int)
        and cat_dim == 1
        and _matches_linspace_axis(
            tensors[0],
            steps=width,
            view_shape=[1, 1, 1, width],
            expand_shape=[1, -1, height, -1],
        )
        and _matches_linspace_axis(
            tensors[1],
            steps=height,
            view_shape=[1, 1, height, 1],
            expand_shape=[1, -1, -1, width],
        )
    )


def _match_flow_offset_add(
    add: Node,
    *,
    spatial_shape: tuple[int, int],
) -> tuple[Node, int] | None:
    """Match ``base_grid + flow[:, i:i + 2]`` in either operand order.

    Args:
        add (Node): Candidate addition node.
        spatial_shape (tuple[int, int]): Expected flow height and width.

    Returns:
        tuple[Node, int] | None: Flow node and first selected channel, or
        ``None`` when the addition is unsupported.

    """
    for base_grid, flow_slice in (
        (add.args[0], add.args[1]),
        (add.args[1], add.args[0]),
    ):
        if not _target_is(flow_slice, "aten.slice_copy.Tensor"):
            continue
        flow, dim, start, end, *step = flow_slice.args
        if (
            not isinstance(flow, Node)
            or dim != 1
            or step not in ([], [1])
            or flow_slice.kwargs.get("step", 1) != 1
        ):
            continue
        if not isinstance(start, int) or not isinstance(end, int):
            continue
        if (start, end) not in ((0, 2), (2, 4)):
            continue
        flow_value = flow.meta.get("val")
        if (
            not isinstance(flow_value, torch.Tensor)
            or len(flow_value.shape) != 4
            or tuple(flow_value.shape[:2]) != (1, 4)
        ):
            continue
        height, width = map(int, flow_value.shape[2:])
        if height <= 1 or width <= 1:
            continue
        if spatial_shape != (height, width):
            continue
        if isinstance(base_grid, Node) and _matches_canonical_base_grid(
            base_grid, height=height, width=width
        ):
            return flow, int(start)
    return None


def _match_flow_offset_grid(node: Node) -> tuple[Node, int] | None:
    """Match a grid sampler that adds flow to a canonical base grid.

    The supported sampler uses bilinear interpolation, border padding, and
    ``align_corners=True``. Its grid is the NHWC permutation of a canonical
    base grid plus flow channels zero-to-one or two-to-three.

    Args:
        node (Node): Candidate ``aten.grid_sampler_2d`` node.

    Returns:
        tuple[Node, int] | None: Flow node and first selected channel, or
        ``None`` when the complete pattern is unsupported.

    """
    if node.target != exir_ops.edge.aten.grid_sampler_2d.default:
        return None
    input_tensor, grid, interpolation, padding, align_corners = node.args
    if (interpolation, padding, align_corners) != (0, 1, True):
        return None
    if not isinstance(input_tensor, Node):
        return None
    input_value = input_tensor.meta.get("val")
    output_value = node.meta.get("val")
    if (
        not isinstance(input_value, torch.Tensor)
        or len(input_value.shape) != 4
        or int(input_value.shape[0]) != 1
        or int(input_value.shape[1]) not in (3, 4)
        or not isinstance(output_value, torch.Tensor)
        or tuple(output_value.shape) != tuple(input_value.shape)
    ):
        return None
    if not _target_is(grid, "aten.permute_copy.default"):
        return None
    if _literal_list(grid.args[1]) != [0, 2, 3, 1]:
        return None
    add = grid.args[0]
    if not _target_is(add, "aten.add.Tensor"):
        return None
    if len(add.args) > 2 and add.args[2] != 1:
        return None
    if add.kwargs.get("alpha", 1) != 1:
        return None

    return _match_flow_offset_add(
        add,
        spatial_shape=(
            int(input_value.shape[2]),
            int(input_value.shape[3]),
        ),
    )


def _is_per_tensor_int8(qparams: QuantArgs | None) -> TypeGuard[QuantArgs]:
    return (
        qparams is not None and not qparams.per_channel and qparams.dtype == torch.int8
    )


def _get_flow_offset_qparams(
    node: Node, flow: Node
) -> tuple[QuantArgs, QuantArgs, QuantArgs] | None:
    """Get qparams supported by the fused shader.

    Image and output qparams must use the SNORM-safe ``[-127, 127]`` range.
    Flow may use the full int8 range because it is read as a tensor rather than
    sampled as an image.

    Args:
        node (Node): Matched grid-sampler node.
        flow (Node): Matched four-channel flow tensor.

    Returns:
        tuple[QuantArgs, QuantArgs, QuantArgs] | None: Image, flow, and output
        qparams, or ``None`` when any qparams are unsupported.

    """
    try:
        image_qparams = get_input_qparams(node).get(0)
        flow_qparams = next(iter(get_output_qparams(flow).values()), None)
        output_qparams = next(iter(get_output_qparams(node).values()), None)
    except ValueError:
        return None
    if not _is_per_tensor_int8(image_qparams):
        return None
    if (image_qparams.qmin, image_qparams.qmax) != (-127, 127):
        return None
    if not _is_per_tensor_int8(flow_qparams):
        return None
    if not _is_per_tensor_int8(output_qparams):
        return None
    if (output_qparams.qmin, output_qparams.qmax) != (-127, 127):
        return None
    return image_qparams, flow_qparams, output_qparams


class FuseGridSamplerFlowOffsetPass(RewriteGridSamplerToTosaCustomPass):
    """Fuse ``grid_sample(canonical_grid + flow)`` into one custom shader."""

    targeted_ops = ()  # type: ignore[assignment]
    _passes_required_after = set()

    def call(self, graph_module: GraphModule) -> PassResult:
        modified = False
        for node in list(graph_module.graph.nodes):
            if node.op != "call_function":
                continue
            match = _match_flow_offset_grid(node)
            if match is None:
                continue
            flow, flow_channel_offset = match
            qparams = _get_flow_offset_qparams(node, flow)
            if qparams is None:
                continue
            rewritten = self._rewrite_flow_offset(
                graph_module,
                node,
                flow,
                flow_channel_offset,
                *qparams,
            )
            modified = rewritten or modified

        if modified:
            graph_module.graph.eliminate_dead_code()
            graph_module.graph.lint()
            graph_module.recompile()
        return PassResult(graph_module, modified)

    def _rewrite_flow_offset(
        self,
        graph_module: GraphModule,
        node: Node,
        flow: Node,
        flow_channel_offset: int,
        image_qparams: QuantArgs,
        flow_qparams: QuantArgs,
        output_qparams: QuantArgs,
    ) -> bool:
        input_tensor = node.args[0]
        if not isinstance(input_tensor, Node):
            raise RuntimeError("grid sampler image input must be a tensor node")
        input_value = input_tensor.meta["val"]
        output_value = node.meta["val"]
        flow_value = flow.meta["val"]
        if tuple(input_value.shape[2:]) != tuple(output_value.shape[2:]) or tuple(
            flow_value.shape[2:]
        ) != tuple(output_value.shape[2:]):
            raise RuntimeError("flow-offset fusion requires equal input/output H/W")

        custom_shape = list(input_value.shape)
        custom_shape[1] = 4
        try:
            payload = build_flow_offset_grid_sampler_payload(
                input_shape=tuple(custom_shape),
                output_shape=tuple(custom_shape),
                flow_shape=tuple(flow_value.shape),
                input_scale=float(image_qparams.get_scale_per_tensor()),
                input_zero_point=int(image_qparams.get_zp_per_tensor()),
                output_scale=float(output_qparams.get_scale_per_tensor()),
                output_zero_point=int(output_qparams.get_zp_per_tensor()),
                flow_scale=float(flow_qparams.get_scale_per_tensor()),
                flow_zero_point=int(flow_qparams.get_zp_per_tensor()),
                flow_channel_offset=flow_channel_offset,
            )
        except _FlowOffsetShaderCompilationError as error:
            logger.warning("Skipping flow-offset grid sampler fusion: %s", error)
            return False

        graph = graph_module.graph
        with graph.inserting_before(node):
            custom_input = (
                self._pad_c3_input_to_c4(
                    graph_module, input_tensor, input_qparams=image_qparams
                )
                if int(input_value.shape[1]) == 3
                else input_tensor
            )
            nhwc_input = graph.call_function(
                exir_ops.edge.aten.permute_copy.default,
                args=(custom_input, list(NHWC_ORDER)),
            )
            nhwc_input.meta = dict(custom_input.meta)
            _set_fake_tensor_meta(
                nhwc_input,
                exir_ops.edge.aten.permute_copy.default(
                    custom_input.meta["val"], list(NHWC_ORDER)
                ),
            )
            custom_node = graph.call_function(
                exir_ops.backend.tosa.CUSTOM.default,
                args=([nhwc_input, flow],),
                kwargs={
                    "operator_name": flow_offset_grid_sampler_operator_name(),
                    "domain_name": CUSTOM_SHADER_DOMAIN_NAME,
                    "implementation_attrs": encode_payload(payload),
                },
            )
            custom_node.meta = dict(node.meta)
            if "input_qparams" in custom_node.meta:
                custom_node.meta["input_qparams"] = {
                    0: custom_node.meta["input_qparams"][0]
                }

        with graph.inserting_after(custom_node):
            getitem = graph.call_function(operator.getitem, args=(custom_node, 0))
            custom_output = torch.empty_like(nhwc_input.meta["val"])
            _set_fake_tensor_meta(custom_node, [custom_output])
            getitem.meta = dict(node.meta)
            _set_fake_tensor_meta(getitem, custom_output)
        with graph.inserting_after(getitem):
            output = graph.call_function(
                exir_ops.edge.aten.permute_copy.default,
                args=(getitem, list(NHWC_INVERSE_ORDER)),
            )
            output.meta = dict(node.meta)
            _set_fake_tensor_meta(
                output,
                exir_ops.edge.aten.permute_copy.default(
                    custom_output, list(NHWC_INVERSE_ORDER)
                ),
            )
        if int(input_value.shape[1]) == 3:
            with graph.inserting_after(output):
                replacement = self._slice_c4_output_to_c3(graph_module, output, node)
        else:
            replacement = output
        node.replace_all_uses_with(replacement)
        graph.erase_node(node)
        return True
