# Copyright 2026 Arm Limited and/or its affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import logging
import operator

import torch
from executorch.backends.arm._passes import ArmPass
from executorch.backends.arm.tosa.mapping import TosaSpecialDtype
from executorch.backends.arm.vgf.shaders.grid_sampler import (
    CUSTOM_SHADER_DOMAIN_NAME,
    flow_offset_grid_sampler_operator_name,
)
from executorch.backends.transforms.utils import create_constant_placeholder
from executorch.exir.pass_base import ExportPass, PassResult
from torch.export import ExportedProgram
from torch.export.graph_signature import InputKind
from torch.fx import GraphModule
from torch.fx.passes.shape_prop import _extract_tensor_metadata

from ..shaders.final_warps import BLEND_OPERATOR_NAME
from .pack_rife_warps_and_concat import (
    _constant_tensor,
    _is_op,
    _shape_values,
    _symmetric_weight,
    _tensor_shape,
    OPERATOR_NAME,
    TOSA,
)

logger = logging.getLogger(__name__)


def _ignores_alpha(custom, image):
    if (
        not _is_op(custom, TOSA.CUSTOM.default, 1)
        or custom.kwargs.get("domain_name") != CUSTOM_SHADER_DOMAIN_NAME
    ):
        return False
    name = custom.kwargs.get("operator_name")
    if name in (OPERATOR_NAME, BLEND_OPERATOR_NAME):
        return image in custom.args[0][:2] and image not in custom.args[0][2:]
    if (
        name != flow_offset_grid_sampler_operator_name()
        or custom.args[0][0] is not image
    ):
        return False
    for output in custom.users:
        if not _is_op(output, operator.getitem, 2) or output.args[1] != 0:
            return False
        for user in output.users:
            channel_axis = 3
            crops = [user]
            if _is_op(user, TOSA.TRANSPOSE.default, 2):
                if _shape_values(user.args[1]) != [0, 3, 1, 2]:
                    return False
                channel_axis = 1
                crops = list(user.users)
            for crop in crops:
                if (
                    not _is_op(crop, TOSA.SLICE.default, 3)
                    or _shape_values(crop.args[1])[channel_axis] != 0
                    or _shape_values(crop.args[2])[channel_axis] != 3
                ):
                    return False
    return True


def _reusable_padding(image):
    for candidate in image.users:
        if (
            _is_op(candidate, TOSA.PAD.default, 2)
            and _shape_values(candidate.args[1]) == [0, 0, 0, 0, 0, 0, 0, 1]
            and all(_ignores_alpha(user, candidate) for user in candidate.users)
        ):
            return candidate
    return None


class PadRifeBlock0ConcatPass(ArmPass):
    """Carry block0 timestep and padding through four C4 concat branches."""

    _passes_required_after: set[type[ExportPass]] = set()

    def __init__(self, exported_program: ExportedProgram | None = None):
        super().__init__()
        self.exported_program = exported_program

    def _match(self, concat):
        if concat.kwargs.get("axis") != 3 or [
            _tensor_shape(node)[-1] for node in concat.args[0]
        ] != [3, 3, 4, 4, 1]:
            raise ValueError("Expected the block0 RGB, features and timestep concat.")
        shape = _tensor_shape(concat)
        if (
            len(shape) != 4
            or shape[-1] != 15
            or concat.meta["val"].dtype != torch.int8
            or any(
                _tensor_shape(node)[:-1] != shape[:-1]
                or node.meta["val"].dtype != torch.int8
                for node in concat.args[0]
            )
        ):
            raise ValueError("Expected matching signed INT8 NHWC concat inputs.")
        users = list(concat.users)
        if (
            len(users) != 1
            or not _is_op(users[0], TOSA.CONV2D.default, 6)
            or users[0].args[0] is not concat
        ):
            raise ValueError("Expected one convolution consuming the block0 concat.")
        conv = users[0]
        _symmetric_weight(conv)
        weight = _constant_tensor(self.exported_program, conv.args[1])
        if weight.dtype != torch.int8 or weight.ndim != 4 or weight.shape[-1] != 15:
            raise ValueError("Expected signed INT8 C15 convolution weights.")
        timestep = _constant_tensor(self.exported_program, concat.args[0][4])
        if timestep.dtype != torch.int8 or not torch.all(
            timestep == timestep.flatten()[0]
        ):
            raise ValueError("Expected a uniform signed INT8 timestep constant.")
        for rescale in concat.args[0][:2]:
            if (
                not _is_op(rescale, TOSA.RESCALE.default, 5)
                or rescale.args[1:] != (torch.int8, [0.25], 0, 0)
                or any(rescale.kwargs.values())
                or set(rescale.users) != {concat}
            ):
                raise ValueError("Expected a private signed 1/4 image rescale.")
            resize = rescale.args[0]
            if (
                not _is_op(resize, TOSA.RESIZE.default, 4)
                or [_shape_values(arg) for arg in resize.args[1:]]
                != [[2, 4, 2, 4], [1, 1], [-1, -1]]
                or resize.kwargs != {"resize_mode": "bilinear"}
                or resize.meta["val"].dtype != torch.int32
                or set(resize.users) != {rescale}
                or _tensor_shape(resize.args[0])
                != (shape[0], shape[1] * 2, shape[2] * 2, 3)
                or resize.args[0].meta["val"].dtype != torch.int8
            ):
                raise ValueError("Expected the original half-size RGB resize.")
        return conv, weight, int(timestep.flatten()[0])

    def call(self, graph_module: GraphModule) -> PassResult:
        exported_program = self.exported_program
        if exported_program is None:
            raise RuntimeError("PadRifeBlock0ConcatPass requires an ExportedProgram.")
        graph = graph_module.graph
        modified = False
        for concat in list(graph.nodes):
            if not _is_op(concat, TOSA.CONCAT.default, 1):
                continue
            try:
                conv, weight, timestep = self._match(concat)
            except ValueError as error:
                logger.debug("Skipping block0 concat padding: %s", error)
                continue
            shape = _tensor_shape(concat)
            extended = weight.new_zeros((*weight.shape[:-1], 16))
            extended[..., :3] = weight[..., :3]
            extended[..., 3] = weight[..., 14]
            extended[..., 4:7] = weight[..., 3:6]
            extended[..., 8:16] = weight[..., 6:14]
            name = f"{concat.name}_c16_weight"
            while any(node.name == name for node in graph.nodes):
                name += "_new"
            with graph.inserting_before(next(iter(graph.nodes))):
                new_weight = create_constant_placeholder(
                    exported_program, graph, name, InputKind.CONSTANT_TENSOR, extended
                )
            for rescale, code in zip(concat.args[0][:2], (timestep, 0)):
                resize = rescale.args[0]
                image = _reusable_padding(resize.args[0])
                if image is None:
                    with graph.inserting_before(resize):
                        padding = graph.call_function(
                            TOSA.CONST_SHAPE.default, ([0, 0, 0, 0, 0, 0, 0, 1],)
                        )
                        padding.meta = {
                            "val": [0, 0, 0, 0, 0, 0, 0, 1],
                            TosaSpecialDtype.meta_key(): TosaSpecialDtype.SHAPE,
                        }
                        image = graph.call_function(
                            TOSA.PAD.default, (resize.args[0], padding), {"value": code}
                        )
                        image.meta = dict(resize.args[0].meta)
                else:
                    # The recognized shaders only use RGB; their alpha padding is unused.
                    image.kwargs = {"value": code}
                    padding = image.args[1]
                    if isinstance(padding, torch.fx.Node) and list(graph.nodes).index(
                        padding
                    ) > list(graph.nodes).index(resize):
                        resize.prepend(padding)
                    resize.prepend(image)
                resize.args = (image, *resize.args[1:])
                # Constant bilinear input becomes 4 * code, then RESCALE divides by 4.
                for node in (image, resize, rescale):
                    value = node.meta["val"].new_empty((*_tensor_shape(node)[:-1], 4))
                    node.meta = {
                        **node.meta,
                        "val": value,
                        "tensor_meta": _extract_tensor_metadata(value),
                    }
            concat.args = (list(concat.args[0][:4]),)
            value = concat.meta["val"].new_empty((*shape[:-1], 16))
            concat.meta = {
                **concat.meta,
                "val": value,
                "tensor_meta": _extract_tensor_metadata(value),
            }
            conv.args = (concat, new_weight, *conv.args[2:])
            modified = True
        if modified:
            graph.eliminate_dead_code()
            graph.lint()
            graph_module.recompile()
        return PassResult(graph_module, modified)
