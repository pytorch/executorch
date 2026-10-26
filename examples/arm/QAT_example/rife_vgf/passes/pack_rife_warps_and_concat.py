# Copyright 2026 Arm Limited and/or its affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import json
import logging
import math
import operator

import torch
from examples.arm.QAT_example.rife_vgf.shaders.pack_warps import (
    build_packed_warp_payload,
    OPERATOR_NAME,
)
from executorch.backends.arm._passes import ArmPass
from executorch.backends.arm._passes.arm_pass_utils import (
    get_param_tensor,
    is_param_node,
)
from executorch.backends.arm._passes.fold_qdq_with_annotated_qparams_pass import (
    get_input_qparams,
)
from executorch.backends.arm.tosa.dialect.ops.custom import register_fake_tosa
from executorch.backends.arm.tosa.mapping import TosaSpecialDtype
from executorch.backends.arm.vgf.shaders import grid_sampler
from executorch.backends.arm.vgf.shaders.grid_sampler import (
    CUSTOM_SHADER_DOMAIN_NAME,
    flow_offset_grid_sampler_operator_name,
)
from executorch.backends.transforms.utils import create_constant_placeholder
from executorch.exir.dialects._ops import ops as exir_ops
from executorch.exir.pass_base import ExportPass, PassResult
from torch.export import ExportedProgram
from torch.export.graph_signature import InputKind
from torch.fx import GraphModule, Node
from torch.fx.passes.shape_prop import _extract_tensor_metadata

logger = logging.getLogger(__name__)
TOSA = exir_ops.backend.tosa


def _is_op(node, target, arity):
    return (
        isinstance(node, Node)
        and node.op == "call_function"
        and node.target == target
        and len(node.args) == arity
    )


def _shape_values(value):
    if isinstance(value, Node):
        if not _is_op(value, TOSA.CONST_SHAPE.default, 1):
            raise ValueError("Expected a constant shape.")
        value = value.args[0]
    if not isinstance(value, (list, tuple)) or not all(type(v) is int for v in value):
        raise ValueError("Expected a static integer shape.")
    return list(value)


def _tensor_shape(node):
    value = node.meta.get("val")
    if (
        not isinstance(value, torch.Tensor)
        or node.meta.get(TosaSpecialDtype.meta_key()) is not None
        or not all(type(dim) is int for dim in value.shape)
    ):
        raise ValueError("Expected a static native tensor.")
    return tuple(value.shape)


def _constant_tensor(program, node):
    views = []
    while _is_op(node, TOSA.RESHAPE.default, 2) or _is_op(
        node, TOSA.TRANSPOSE.default, 2
    ):
        views.append((node.target, _shape_values(node.args[1])))
        node = node.args[0]
    if not isinstance(node, Node) or not is_param_node(program, node):
        raise ValueError("Expected an immutable tensor constant.")
    buffer = program.graph_signature.inputs_to_buffers.get(node.name)
    if (
        buffer is not None
        and buffer in program.graph_signature.buffers_to_mutate.values()
    ):
        raise ValueError("Cannot rewrite mutable buffers.")
    if node.meta.get(TosaSpecialDtype.meta_key()) is not None:
        raise ValueError("Expected a native tensor constant.")
    value = get_param_tensor(program, node)
    if value is None:
        raise ValueError("Expected a tensor constant.")
    value = value.detach().cpu()
    for target, shape in reversed(views):
        value = (
            value.reshape(shape)
            if target == TOSA.RESHAPE.default
            else value.permute(shape)
        )
    return value


PARAMETERS = (
    "flow_scale",
    "flow_zero_point",
    "input_scale",
    "input_zero_point",
    "output_scale",
    "output_zero_point",
)


def _validate_sampling_qparams(payload):
    for key in ("flow_scale", "input_scale", "output_scale"):
        value = payload.get(key)
        if type(value) not in (int, float) or not math.isfinite(value) or value <= 0:
            raise ValueError("Expected finite positive sampling scales.")
    for key in ("flow_zero_point", "input_zero_point", "output_zero_point"):
        value = payload.get(key)
        if type(value) is not int or not -128 <= value <= 127:
            raise ValueError("Expected signed INT8 sampling zero points.")
    if payload.get("flow_channel_offset") not in (0, 2):
        raise ValueError("Expected a valid two-channel flow selection.")


def _payload(custom):
    if (
        not _is_op(custom, TOSA.CUSTOM.default, 1)
        or custom.kwargs.get("operator_name")
        != flow_offset_grid_sampler_operator_name()
        or custom.kwargs.get("domain_name") != CUSTOM_SHADER_DOMAIN_NAME
    ):
        raise ValueError("Expected the original INT8 align-corners sampler.")
    try:
        payload = json.loads(bytes(custom.kwargs["implementation_attrs"]))
    except (KeyError, TypeError, ValueError):
        raise ValueError("Expected a valid sampler payload.") from None
    if not isinstance(payload, dict):
        raise ValueError("Expected a sampler payload object.")
    _validate_sampling_qparams(payload)
    if (
        payload.get("input_0_vkformat") != "VK_FORMAT_R8G8B8A8_SNORM"
        or payload.get("output_0_vkformat") != "VK_FORMAT_R8G8B8A8_SNORM"
    ):
        raise ValueError("Expected original SNORM image descriptors.")
    expected = {
        "entry_point": "main",
        "shader_language": "SPIR-V",
        "input_0_type": "Image",
        "input_0_binding": 0,
        "input_0_descriptorset": 0,
        "input_0_vkdescriptortype": "VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER",
        "input_0_sampler": grid_sampler._sampler_config(0, 1),
        "input_1_type": "Tensor",
        "input_1_vkformat": "VK_FORMAT_R8_SINT",
        "input_1_binding": 1,
        "input_1_descriptorset": 0,
        "input_1_vkdescriptortype": "VK_DESCRIPTOR_TYPE_TENSOR_ARM",
        "output_0_type": "Image",
        "output_0_binding": 2,
        "output_0_descriptorset": 0,
        "output_0_vkdescriptortype": "VK_DESCRIPTOR_TYPE_STORAGE_IMAGE",
    }
    values = custom.meta["val"]
    if (
        not isinstance(values, (list, tuple))
        or len(values) != 1
        or len(custom.args[0]) != 2
    ):
        raise ValueError("Expected the original sampler IO arity.")
    shape = tuple(values[0].shape)
    if (
        len(shape) != 4
        or shape[0] != 1
        or shape[-1] != 4
        or not all(type(dim) is int and dim > 1 for dim in shape[1:3])
        or _tensor_shape(custom.args[0][0]) != shape
        or _tensor_shape(custom.args[0][1]) != (1, 4, shape[1], shape[2])
        or any(n.meta["val"].dtype != torch.int8 for n in custom.args[0])
        or values[0].dtype != torch.int8
    ):
        raise ValueError("Expected signed INT8 full-size image and flow tensors.")
    expected["workgroup_sizes"] = [(shape[2] + 7) // 8, (shape[1] + 7) // 8, 1]
    if any(payload.get(key) != value for key, value in expected.items()):
        raise ValueError("Sampler descriptors or dispatch differ from the original.")
    return payload


def _sampler_from_output(node):
    if not _is_op(node, operator.getitem, 2) or node.args[1] != 0:
        raise ValueError("Expected one original custom output.")
    custom = node.args[0]
    _payload(custom)
    if len(custom.users) != 1:
        raise ValueError("Cannot pack a shared custom output.")
    return custom


def _symmetric_weight(conv):
    weight = get_input_qparams(conv).get(1)
    if weight is None or torch.any(torch.as_tensor(weight.zp) != 0):
        raise ValueError("Channel padding requires symmetric convolution weights.")


@register_fake_tosa(OPERATOR_NAME)
def _packed_fake(inputs, operator_name, domain_name, implementation_attrs):
    return [inputs[0].new_empty((*inputs[0].shape[:-1], 32))]


def _match_pixel_shuffle(slices, height, width):
    mask, features, flow = slices
    nodes = set(slices)
    if not _is_op(mask, TOSA.RESHAPE.default, 2):
        raise ValueError("Expected the original mask reshape.")
    mask = mask.args[0]
    nodes.add(mask)
    for node, start, size in (
        (mask, [0, 4, 0, 0], [1, 1, height // 2, width // 2]),
        (features, [0, 0, 0, 5], [1, height // 2, width // 2, 8]),
        (flow, [0, 0, 0, 0], [1, height // 2, width // 2, 4]),
    ):
        if (
            not _is_op(node, TOSA.SLICE.default, 3)
            or _shape_values(node.args[1]) != start
            or _shape_values(node.args[2]) != size
        ):
            raise ValueError("Expected the original flow/mask/feature slices.")
    nhwc = flow.args[0]
    if features.args[0] is not nhwc or not _is_op(nhwc, TOSA.TRANSPOSE.default, 2):
        raise ValueError("Expected one shared PixelShuffle output.")
    nchw = nhwc.args[0]
    if mask.args[0] is not nchw or not _is_op(nchw, TOSA.RESHAPE.default, 2):
        raise ValueError("Expected the original NCHW PixelShuffle view.")
    transpose = nchw.args[0]
    if not _is_op(transpose, TOSA.TRANSPOSE.default, 2):
        raise ValueError("Expected the PixelShuffle phase transpose.")
    reshape = transpose.args[0]
    if not _is_op(reshape, TOSA.RESHAPE.default, 2):
        raise ValueError("Expected the PixelShuffle phase reshape.")
    phase = reshape.args[0]
    if (
        _tensor_shape(phase) != (1, height // 4, width // 4, 52)
        or phase.meta["val"].dtype != torch.int8
        or _shape_values(reshape.args[1]) != [1, height // 4, width // 4, 13, 2, 2]
        or _shape_values(transpose.args[1]) != [0, 3, 1, 4, 2, 5]
        or _shape_values(nchw.args[1]) != [1, 13, height // 2, width // 2]
        or _shape_values(nhwc.args[1]) != [0, 2, 3, 1]
    ):
        raise ValueError("Expected the original C52 PixelShuffle layout.")
    nodes.update((nhwc, nchw, transpose, reshape))
    return phase, nodes


def _match_block0_tail(concat, customs):
    _, height, width, _ = _tensor_shape(concat)
    if height % 4 or width % 4:
        raise ValueError("Expected dimensions divisible by four.")
    nodes = set()
    slices = []
    for rescale, channels in zip(concat.args[0][5:], (1, 8, 4)):
        if (
            not _is_op(rescale, TOSA.RESCALE.default, 5)
            or rescale.args[1:] != (torch.int8, [0.0625], 0, 0)
            or any(rescale.kwargs.values())
        ):
            raise ValueError("Expected the original signed 1/16 state rescale.")
        resize = rescale.args[0]
        if (
            not _is_op(resize, TOSA.RESIZE.default, 4)
            or [_shape_values(v) for v in resize.args[1:]]
            != [[4, 2, 4, 2], [-1, -1], [1, 1]]
            or resize.kwargs != {"resize_mode": "bilinear"}
            or resize.meta["val"].dtype != torch.int32
        ):
            raise ValueError("Expected the original integer 2x state resize.")
        source = resize.args[0]
        if _tensor_shape(source) != (1, height // 2, width // 2, channels):
            raise ValueError("Expected half-size state tensors.")
        nodes.update((rescale, resize, source))
        slices.append(source)
    phase, shuffle_nodes = _match_pixel_shuffle(slices, height, width)
    nodes.update(shuffle_nodes)
    flow_transpose = customs[0].args[0][1]
    residuals = [n for n in flow_transpose.users if n not in customs]
    if len(residuals) != 1 or not _is_op(residuals[0], TOSA.RESCALE.default, 5):
        raise ValueError("Expected one previous-flow rescale.")
    residual = residuals[0]
    if (
        residual.args[1] != torch.int32
        or len(residual.args[2]) != 1
        or residual.args[4] != 0
        or any(residual.kwargs.values())
    ):
        raise ValueError("Expected a signed per-tensor INT32 previous-flow rescale.")
    nodes.add(flow_transpose)
    allowed = nodes | set(customs) | {concat, residual}
    if any(user not in allowed for node in nodes for user in node.users):
        raise ValueError("The block0 tail has additional consumers.")
    positions = {node: i for i, node in enumerate(concat.graph.nodes)}
    if any(positions[user] <= positions[concat] for user in residual.users):
        raise ValueError("Previous flow is needed before the packed shader.")
    return phase, residual


def _phase_major_parameters(program, phase):
    channels = _tensor_shape(phase)[-1]
    if channels not in (20, 52):
        raise ValueError("Expected a C20 or C52 PixelShuffle input.")
    if not _is_op(phase, TOSA.RESCALE.default, 5) or len(phase.users) != 1:
        raise ValueError("Expected a private PixelShuffle output rescale.")
    producer = phase.args[0]
    if (
        not _is_op(producer, TOSA.TRANSPOSE_CONV2D.default, 5)
        or set(producer.users) != {phase}
        or producer.meta["val"].dtype != torch.int32
        or phase.args[1] != torch.int8
        or len(phase.args[2]) not in (1, channels)
    ):
        raise ValueError("Expected a private PixelShuffle transpose convolution.")
    _symmetric_weight(producer)
    weight = _constant_tensor(program, producer.args[1])
    bias = _constant_tensor(program, producer.args[2])
    if (
        weight.ndim != 4
        or weight.shape[0] != channels
        or weight.dtype != torch.int8
        or bias.shape != (channels,)
        or bias.dtype != torch.int32
    ):
        raise ValueError("Expected native PixelShuffle convolution weights and bias.")
    # Make each PixelShuffle phase contiguous at the producer.
    order = torch.arange(channels).reshape(channels // 4, 4).t().reshape(-1)
    scales = phase.args[2]
    if len(scales) == channels:
        scales = [scales[i] for i in order.tolist()]
    return producer, weight[order], bias[order], scales


def _reorder_pixel_shuffle_output(program, phase, parameters):
    producer, weight, bias, scales = parameters
    graph = phase.graph
    constants = []
    for suffix, value in (("weight", weight), ("bias", bias)):
        name = f"{producer.name}_phase_major_{suffix}"
        while any(n.name == name for n in graph.nodes):
            name += "_new"
        with graph.inserting_before(next(iter(graph.nodes))):
            constants.append(
                create_constant_placeholder(
                    program, graph, name, InputKind.CONSTANT_TENSOR, value
                )
            )
    producer.args = (producer.args[0], *constants, *producer.args[3:])
    phase.args = (*phase.args[:2], scales, *phase.args[3:])


class PackRifeWarpsAndConcatPass(ArmPass):
    """Pack four private RIFE warps and their C28 concat into one C32 shader.

    Fuse a compatible block0 PixelShuffle and state preparation into the shader.
    Reorder block0 output channels during export so each PixelShuffle read
    accesses 13 contiguous channels. Keep sampling quantization boundaries.
    Four zero weight columns make the following convolution ignore padding.
    Unsupported graphs are left unchanged.

    """

    _passes_required_after: set[type[ExportPass]] = set()

    def __init__(self, exported_program: ExportedProgram | None = None):
        super().__init__()
        self.exported_program = exported_program

    def _match(self, graph_module):
        candidates = [
            n
            for n in graph_module.graph.nodes
            if _is_op(n, TOSA.CONCAT.default, 1)
            and n.kwargs.get("axis") == 3
            and [_tensor_shape(x)[-1] for x in n.args[0]] == [3, 3, 4, 4, 1, 1, 8, 4]
        ]
        if len(candidates) != 1:
            raise ValueError("Expected one eight-input C28 RIFE concat.")
        concat = candidates[0]
        inputs = list(concat.args[0])
        shape = _tensor_shape(concat)
        if (
            len(shape) != 4
            or shape[0] != 1
            or shape[-1] != 28
            or any(
                _tensor_shape(n) != (*shape[:-1], channels)
                or n.meta["val"].dtype != torch.int8
                for n, channels in zip(inputs, (3, 3, 4, 4, 1, 1, 8, 4))
            )
            or concat.meta["val"].dtype != torch.int8
        ):
            raise ValueError("Expected matching signed INT8 concat inputs.")
        crops = inputs[:2]
        if not all(
            _is_op(n, TOSA.SLICE.default, 3)
            and _shape_values(n.args[1]) == [0, 0, 0, 0]
            and _shape_values(n.args[2]) == list(n.meta["val"].shape)
            for n in crops
        ):
            raise ValueError("Expected the original two RGB output crops.")
        outputs = [n.args[0] for n in crops] + inputs[2:4]
        customs = [_sampler_from_output(n) for n in outputs]
        if (
            len(set(customs)) != 4
            or any(
                set(n.users) != {consumer}
                for n, consumer in zip(outputs, crops + [concat, concat])
            )
            or any(set(n.users) != {concat} for n in crops)
        ):
            raise ValueError("Packed warp outputs must only feed the concat.")
        payloads = [_payload(n) for n in customs]
        parameters = {k: payloads[0][k] for k in PARAMETERS}
        if any({k: p[k] for k in PARAMETERS} != parameters for p in payloads) or [
            p["flow_channel_offset"] for p in payloads
        ] != [0, 2, 0, 2]:
            raise ValueError("Warp packing requires identical sampling qparams.")
        flow = customs[0].args[0][1]
        if any(n.args[0][1] is not flow for n in customs) or (
            not _is_op(flow, TOSA.TRANSPOSE.default, 2)
            or flow.args[0] is not inputs[-1]
            or _shape_values(flow.args[1]) != [0, 3, 1, 2]
        ):
            raise ValueError("Expected the shared transposed concat flow.")
        images = [n.args[0][0] for n in customs]
        if any(_tensor_shape(n) != (*shape[:-1], 4) for n in images + outputs):
            raise ValueError("Expected matching full-size RGBA images.")
        timestep = _constant_tensor(self.exported_program, inputs[4])
        if timestep.dtype != torch.int8 or not torch.all(
            timestep == timestep.flatten()[0]
        ):
            raise ValueError("Expected a constant signed INT8 timestep.")
        users = list(concat.users)
        if len(users) != 1 or not _is_op(users[0], TOSA.CONV2D.default, 6):
            raise ValueError("Expected one convolution after the concat.")
        conv = users[0]
        _symmetric_weight(conv)
        weight = _constant_tensor(self.exported_program, conv.args[1])
        if weight.dtype != torch.int8 or weight.ndim != 4 or weight.shape[-1] != 28:
            raise ValueError("Expected signed INT8 C28 convolution weights.")
        return (
            concat,
            images,
            payloads,
            int(timestep.flatten()[0]),
            conv,
            weight,
            customs,
        )

    def call(self, graph_module: GraphModule) -> PassResult:
        """Replace compatible warps after TOSA lowering and rescale insertion.

        Args:
            graph_module (GraphModule): Lowered graph with explicit qparams.

        Returns:
            PassResult: Rewritten graph, or the unchanged unsupported graph.

        """
        exported_program = self.exported_program
        if exported_program is None:
            raise RuntimeError(
                "PackRifeWarpsAndConcatPass requires an ExportedProgram."
            )
        try:
            concat, images, payloads, timestep, conv, weight, customs = self._match(
                graph_module
            )
        except ValueError as error:
            logger.debug("Skipping packed RIFE warps: %s", error)
            return PassResult(graph_module, False)

        phase_parameters = None
        try:
            tail = _match_block0_tail(concat, customs)
            phase_parameters = _phase_major_parameters(exported_program, tail[0])
        except ValueError as error:
            logger.debug("Keeping the original PixelShuffle layout: %s", error)
            tail = None
        if tail is None:
            payload = build_packed_warp_payload(payloads, timestep)
            shader_inputs = [*images, *concat.args[0][5:]]
        else:
            payload = build_packed_warp_payload(
                payloads, timestep, fuse_block0_tail=True
            )
            shader_inputs = [*images, tail[0]]
            _reorder_pixel_shuffle_output(exported_program, tail[0], phase_parameters)
        graph = graph_module.graph
        extended = torch.zeros((*weight.shape[:-1], 32), dtype=torch.int8)
        extended[..., :28] = weight
        name = f"{conv.name}_c32_weight"
        while any(n.name == name for n in graph.nodes):
            name += "_new"
        with graph.inserting_before(next(iter(graph.nodes))):
            new_weight = create_constant_placeholder(
                exported_program, graph, name, InputKind.CONSTANT_TENSOR, extended
            )
        conv.args = (conv.args[0], new_weight, *conv.args[2:])
        with graph.inserting_before(concat):
            custom = graph.call_function(
                TOSA.CUSTOM.default,
                (shader_inputs,),
                {
                    "operator_name": OPERATOR_NAME,
                    "domain_name": CUSTOM_SHADER_DOMAIN_NAME,
                    "implementation_attrs": list(json.dumps(payload).encode()),
                },
            )
        value = concat.meta["val"].new_empty((*_tensor_shape(concat)[:-1], 32))
        custom.meta = {
            **concat.meta,
            "val": [value],
            "tensor_meta": _extract_tensor_metadata(value),
        }
        # Keep the concat node's users and identity, but replace its computation
        # with the shader output. Dead-code removal drops the private old warps.
        concat.target, concat.args, concat.kwargs = operator.getitem, (custom, 0), {}
        concat.meta = {
            **concat.meta,
            "val": value,
            "tensor_meta": _extract_tensor_metadata(value),
        }
        if tail is not None:
            residual = tail[1]
            flow_value = residual.args[0].meta["val"]
            shape = [*value.shape[:-1], 4]
            with graph.inserting_after(concat):
                starts = graph.call_function(TOSA.CONST_SHAPE.default, ([0, 0, 0, 24],))
            with graph.inserting_after(starts):
                sizes = graph.call_function(TOSA.CONST_SHAPE.default, (shape,))
            for node in (starts, sizes):
                node.meta = {
                    "val": node.args[0],
                    TosaSpecialDtype.meta_key(): TosaSpecialDtype.SHAPE,
                }
            with graph.inserting_after(sizes):
                flow_slice = graph.call_function(
                    TOSA.SLICE.default, (concat, starts, sizes)
                )
            sliced_value = value.new_empty(shape)
            flow_slice.meta = {
                "val": sliced_value,
                "tensor_meta": _extract_tensor_metadata(sliced_value),
            }
            with graph.inserting_after(flow_slice):
                flow = graph.call_function(
                    TOSA.TRANSPOSE.default, (flow_slice, [0, 3, 1, 2])
                )
            flow.meta = {
                **residual.args[0].meta,
                "val": flow_value,
                "tensor_meta": _extract_tensor_metadata(flow_value),
            }
            residual.args = (flow, *residual.args[1:])
            flow.append(residual)
        graph.eliminate_dead_code()
        graph.lint()
        graph_module.recompile()
        return PassResult(graph_module, True)
