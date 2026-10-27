# Copyright 2026 Arm Limited and/or its affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import json
import logging
import operator

import torch
from examples.arm.QAT_example.rife_vgf.shaders.final_warps import (
    BLEND_OPERATOR_NAME,
    build_final_warps_payload,
    OPERATOR_NAME,
)
from examples.arm.QAT_example.rife_vgf.shaders.pack_warps import (
    OPERATOR_NAME as PACK_OPERATOR_NAME,
)
from executorch.backends.arm._passes import ArmPass
from executorch.backends.arm.operators.op_tosa_rescale import (
    _compute_multiplier_and_shift,
)
from executorch.backends.arm.tosa.dialect.ops.custom import register_fake_tosa
from executorch.backends.arm.vgf.shaders.grid_sampler import (
    CUSTOM_SHADER_DOMAIN_NAME,
    flow_offset_grid_sampler_operator_name,
)
from executorch.exir.pass_base import ExportPass, PassResult
from torch.export import ExportedProgram
from torch.fx import GraphModule

from .pack_rife_warps_and_concat import (
    _constant_tensor,
    _is_op,
    _payload,
    _phase_major_parameters,
    _reorder_pixel_shuffle_output,
    _shape_values,
    _tensor_shape,
    TOSA,
)
from .rife_blend import match_rife_blend

logger = logging.getLogger(__name__)


@register_fake_tosa(OPERATOR_NAME)
@register_fake_tosa(BLEND_OPERATOR_NAME)
def _final_warps_fake(inputs, operator_name, domain_name, implementation_attrs):
    _, height, width, _ = inputs[0].shape
    if operator_name == BLEND_OPERATOR_NAME:
        return [inputs[0].new_empty((1, 3, height, width))]
    return [
        inputs[0].new_empty((1, height, width, 4)),
        inputs[1].new_empty((1, height, width, 4)),
        inputs[0].new_empty((1, 1, height, width)),
    ]


def _match_previous_flow(rescale, height, width):
    transpose = rescale.args[0]
    if not _is_op(transpose, TOSA.TRANSPOSE.default, 2) or _shape_values(
        transpose.args[1]
    ) != [0, 3, 1, 2]:
        raise ValueError("Expected the previous C32 flow transpose.")
    crop = transpose.args[0]
    if (
        not _is_op(crop, TOSA.SLICE.default, 3)
        or _shape_values(crop.args[1]) != [0, 0, 0, 24]
        or _shape_values(crop.args[2]) != [1, height, width, 4]
    ):
        raise ValueError("Expected previous flow in C32 channels 24-27.")
    packed = crop.args[0]
    if (
        not _is_op(packed, operator.getitem, 2)
        or packed.args[1] != 0
        or not _is_op(packed.args[0], TOSA.CUSTOM.default, 1)
        or packed.args[0].kwargs.get("operator_name") != PACK_OPERATOR_NAME
        or packed.args[0].kwargs.get("domain_name") != CUSTOM_SHADER_DOMAIN_NAME
        or _tensor_shape(packed) != (1, height, width, 32)
        or packed.meta["val"].dtype != torch.int8
    ):
        raise ValueError("Expected the packed block0 shader's C32 output.")
    return packed, {rescale, transpose, crop}


def _match_pixel_shuffle(delta, height, width):
    crop = delta.args[0]
    if (
        not _is_op(crop, TOSA.SLICE.default, 3)
        or _shape_values(crop.args[1]) != [0, 0, 0, 0]
        or _shape_values(crop.args[2]) != [1, 4, height, width]
    ):
        raise ValueError("Expected block1's four-channel flow update.")
    nchw = crop.args[0]
    if not _is_op(nchw, TOSA.RESHAPE.default, 2) or _shape_values(nchw.args[1]) != [
        1,
        5,
        height,
        width,
    ]:
        raise ValueError("Expected block1's C5 PixelShuffle output.")
    transpose = nchw.args[0]
    if not _is_op(transpose, TOSA.TRANSPOSE.default, 2) or _shape_values(
        transpose.args[1]
    ) != [0, 3, 1, 4, 2, 5]:
        raise ValueError("Expected the PixelShuffle phase transpose.")
    reshape = transpose.args[0]
    if (
        not _is_op(reshape, TOSA.RESHAPE.default, 2)
        or _shape_values(reshape.args[1]) != [1, height // 2, width // 2, 5, 2, 2]
        or _tensor_shape(reshape.args[0]) != (1, height // 2, width // 2, 20)
        or reshape.args[0].meta["val"].dtype != torch.int8
    ):
        raise ValueError("Expected the original C20 phase layout.")
    masks = [n for n in nchw.users if n is not crop]
    if len(masks) != 1:
        raise ValueError("Expected one private mask slice.")
    mask = masks[0]
    if (
        not _is_op(mask, TOSA.SLICE.default, 3)
        or _shape_values(mask.args[1]) != [0, 4, 0, 0]
        or _shape_values(mask.args[2]) != [1, 1, height, width]
        or len(mask.users) != 1
    ):
        raise ValueError("Expected block1's one-channel mask.")
    table = next(iter(mask.users))
    if not _is_op(table, TOSA.TABLE.default, 2):
        raise ValueError("Expected the quantized sigmoid lookup.")
    return reshape.args[0], table, {delta, crop, nchw, transpose, reshape, mask}


def _validate_rescales(previous, delta, total):
    bounds = []
    for node, dtype in (
        (previous, torch.int32),
        (delta, torch.int32),
        (total, torch.int8),
    ):
        if (
            not _is_op(node, TOSA.RESCALE.default, 5)
            or node.args[1] != dtype
            or len(node.args[2]) != 1
            or any(node.kwargs.values())
            or not all(-128 <= node.args[i] <= 127 for i in (3, 4))
        ):
            raise ValueError("Expected signed per-tensor flow rescales.")
        multipliers, shifts = _compute_multiplier_and_shift(node.args[2])
        if dtype == torch.int32:
            if node.args[4] != 0:
                raise ValueError("Expected zero INT32 output zero point.")
            bounds.append(
                [
                    ((value - node.args[3]) * multipliers[0] + (1 << (shifts[0] - 1)))
                    >> shifts[0]
                    for value in (-128, 127)
                ]
            )
    if total.args[3] != 0 or any(
        not -(1 << 31) <= value < (1 << 31)
        for value in [*bounds[0], *bounds[1], *(sum(v) for v in zip(*bounds))]
    ):
        raise ValueError("Flow addition could overflow INT32.")


class FuseRifeFinalWarpsPass(ArmPass):
    """Fuse the IFNetTwoBlock tail, including a compatible final image blend."""

    _passes_required_after: set[type[ExportPass]] = set()

    def __init__(self, exported_program: ExportedProgram | None = None):
        super().__init__()
        self.exported_program = exported_program

    def _match(self, graph_module):
        customs = [
            n
            for n in graph_module.graph.nodes
            if _is_op(n, TOSA.CUSTOM.default, 1)
            and n.kwargs.get("operator_name")
            == flow_offset_grid_sampler_operator_name()
        ]
        if len(customs) != 2:
            raise ValueError("Expected exactly two remaining image warps.")
        payloads = [_payload(n) for n in customs]
        if (
            [p["flow_channel_offset"] for p in payloads] != [0, 2]
            or any(
                payloads[0][k] != payloads[1][k]
                for k in ("flow_scale", "flow_zero_point", "flow_normalization")
            )
            or customs[0].args[0][1] is not customs[1].args[0][1]
        ):
            raise ValueError("Expected two compatible warps sharing final flow.")
        flow = customs[0].args[0][1]
        if not _is_op(flow, TOSA.RESCALE.default, 5):
            raise ValueError("Expected the final flow rescale.")
        add = flow.args[0]
        if not _is_op(add, TOSA.ADD.default, 2):
            raise ValueError("Expected the flow residual addition.")
        previous, delta = add.args
        _validate_rescales(previous, delta, flow)
        _, _, height, width = _tensor_shape(flow)
        if height % 2 or width % 2:
            raise ValueError("Expected even spatial dimensions.")
        packed, nodes = _match_previous_flow(previous, height, width)
        phase, table, shuffle_nodes = _match_pixel_shuffle(delta, height, width)
        positions = {node: i for i, node in enumerate(graph_module.graph.nodes)}
        if any(
            positions[node] >= positions[customs[0]]
            for node in [*(custom.args[0][0] for custom in customs), packed, phase]
        ):
            raise ValueError("A shader input is unavailable at the first warp.")
        nodes.update(shuffle_nodes | {add, flow})
        if any(
            user not in nodes | set(customs) | {table}
            for n in nodes
            for user in n.users
        ):
            raise ValueError("The final tail has additional consumers.")
        sigmoid = _constant_tensor(self.exported_program, table.args[1])
        if sigmoid.dtype != torch.int8 or sigmoid.shape != (256,):
            raise ValueError("Expected an INT8 sigmoid table.")
        outputs = []
        for custom in customs:
            users = list(custom.users)
            if (
                len(users) != 1
                or not _is_op(users[0], operator.getitem, 2)
                or users[0].args[1] != 0
            ):
                raise ValueError("Expected one output per original warp.")
            outputs.append(users[0])
        parameters = _phase_major_parameters(self.exported_program, phase)
        return (
            customs,
            payloads,
            (previous, delta, flow),
            packed,
            phase,
            table,
            sigmoid,
            outputs,
            parameters,
        )

    def call(self, graph_module: GraphModule) -> PassResult:
        """Replace a private block1 tail and both image warps with one shader.

        Args:
            graph_module (GraphModule): TOSA graph with explicit quantization.

        Returns:
            PassResult: Rewritten graph, or the unchanged unsupported graph.

        """
        if self.exported_program is None:
            raise RuntimeError("FuseRifeFinalWarpsPass requires an ExportedProgram.")
        try:
            (
                customs,
                payloads,
                rescales,
                packed,
                phase,
                table,
                sigmoid,
                outputs,
                parameters,
            ) = self._match(graph_module)
        except ValueError as error:
            logger.debug("Keeping the original RIFE final tail: %s", error)
            return PassResult(graph_module, False)
        try:
            blend = match_rife_blend(self.exported_program, outputs, table)
        except ValueError as error:
            logger.debug("Keeping the RIFE image blend in TOSA: %s", error)
            blend = None
        payload = build_final_warps_payload(payloads, rescales, sigmoid, blend=blend)
        _reorder_pixel_shuffle_output(self.exported_program, phase, parameters)
        graph = graph_module.graph
        with graph.inserting_before(customs[0]):
            custom = graph.call_function(
                TOSA.CUSTOM.default,
                ([customs[0].args[0][0], customs[1].args[0][0], packed, phase],),
                {
                    "operator_name": (
                        OPERATOR_NAME if blend is None else BLEND_OPERATOR_NAME
                    ),
                    "domain_name": CUSTOM_SHADER_DOMAIN_NAME,
                    "implementation_attrs": list(json.dumps(payload).encode()),
                },
            )
        results = [*outputs, table] if blend is None else [blend.output]
        custom.meta = {"val": [n.meta["val"] for n in results]}
        for index, node in enumerate(results):
            node.target, node.args, node.kwargs = operator.getitem, (custom, index), {}
        graph.eliminate_dead_code()
        graph.lint()
        graph_module.recompile()
        return PassResult(graph_module, True)
