# Copyright 2026 Arm Limited and/or its affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from types import SimpleNamespace

import pytest
import torch
import torch.nn.functional as F
from examples.arm.QAT_example.rife_vgf.passes import pad_rife_block0_concat as pad
from executorch.backends.arm._passes.arm_pass_manager import (
    _ExportedProgramGraphPassAdapter,
)
from executorch.backends.transforms.utils import create_constant_placeholder
from torch.export.graph_signature import InputKind

TOSA = pad.TOSA


class _Inputs(torch.nn.Module):
    def forward(self, *inputs):
        return inputs[0]


def _program():
    tensors = [torch.zeros(1, 8, 12, 3, dtype=torch.int8)] * 2
    tensors += [torch.zeros(1, 4, 6, 4, dtype=torch.int8)] * 2
    program = torch.export.export(_Inputs(), tuple(tensors))
    graph = program.graph
    graph.erase_node(graph.output_node())
    inputs = list(graph.nodes)
    constants = []
    with graph.inserting_before(inputs[0]):
        for name, tensor in (
            (
                "weight",
                torch.arange(45)
                .remainder(11)
                .sub(5)
                .to(torch.int8)
                .reshape(3, 1, 1, 15),
            ),
            ("bias", torch.tensor([-17, 9, 23], dtype=torch.int32)),
            ("timestep", torch.full((1, 4, 6, 1), 40, dtype=torch.int8)),
        ):
            constants.append(
                create_constant_placeholder(
                    program, graph, name, InputKind.CONSTANT_TENSOR, tensor
                )
            )
    branches = []
    for image in inputs[:2]:
        resize = graph.call_function(
            TOSA.RESIZE.default,
            (image, [2, 4, 2, 4], [1, 1], [-1, -1]),
            {"resize_mode": "bilinear"},
        )
        resize.meta["val"] = image.meta["val"].new_empty(
            (1, 4, 6, 3), dtype=torch.int32
        )
        rescale = graph.call_function(
            TOSA.RESCALE.default, (resize, torch.int8, [0.25], 0, 0)
        )
        rescale.meta["val"] = image.meta["val"].new_empty((1, 4, 6, 3))
        branches.append(rescale)
    concat = graph.call_function(
        TOSA.CONCAT.default, ([*branches, *inputs[2:], constants[2]],), {"axis": 3}
    )
    concat.meta["val"] = inputs[0].meta["val"].new_empty((1, 4, 6, 15))
    conv = graph.call_function(
        TOSA.CONV2D.default,
        (concat, *constants[:2], [1, 1], [0, 0, 0, 0], [1, 1]),
    )
    conv.meta["val"] = inputs[0].meta["val"].new_empty((1, 4, 6, 3), dtype=torch.int32)
    conv.meta["input_qparams"] = {0: SimpleNamespace(zp=-11), 1: SimpleNamespace(zp=0)}
    graph.output((conv,))
    return program, inputs, concat, conv


@pytest.mark.parametrize("adapted", [False, True])
@pytest.mark.parametrize("zero_point", [-127, 0, 17])
def test_pads_concat_and_preserves_convolution(adapted, zero_point):
    program, _, concat, conv = _program()
    old_weight = conv.args[1]
    weight = pad._constant_tensor(program, old_weight).clone()
    bias = pad._constant_tensor(program, conv.args[2]).double()
    output_value = conv.meta["val"]
    conv.meta["input_qparams"][0].zp = zero_point
    result = (
        _ExportedProgramGraphPassAdapter(pad.PadRifeBlock0ConcatPass()).call(program)
        if adapted
        else pad.PadRifeBlock0ConcatPass(program)(program.graph_module)
    )
    assert result.modified
    assert tuple(concat.meta["val"].shape) == (1, 4, 6, 16)
    assert tuple(concat.meta["tensor_meta"].shape) == (1, 4, 6, 16)
    assert len(concat.args[0]) == 4
    assert [pad._tensor_shape(n)[-1] for n in concat.args[0]] == [4, 4, 4, 4]
    for branch, code in zip(concat.args[0][:2], (40, 0)):
        resize = branch.args[0]
        image = resize.args[0]
        assert image.target == TOSA.PAD.default
        assert image.kwargs == {"value": code}
        assert pad._shape_values(image.args[1]) == [0, 0, 0, 0, 0, 0, 0, 1]
        assert tuple(image.meta["val"].shape) == (1, 8, 12, 4)
        assert tuple(resize.meta["val"].shape) == (1, 4, 6, 4)
    extended = pad._constant_tensor(program, conv.args[1])
    assert torch.equal(extended[..., :3], weight[..., :3])
    assert torch.equal(extended[..., 3], weight[..., 14])
    assert torch.equal(extended[..., 4:7], weight[..., 3:6])
    assert torch.count_nonzero(extended[..., 7]) == 0
    assert torch.equal(extended[..., 8:], weight[..., 6:14])
    assert torch.equal(pad._constant_tensor(program, old_weight), weight)
    assert conv.meta["val"].shape == output_value.shape
    assert conv.meta["val"].dtype == output_value.dtype

    generator = torch.Generator().manual_seed(2026)
    images = torch.randint(-128, 128, (2, 1, 8, 12, 3), generator=generator).double()
    features = torch.randint(-128, 128, (1, 4, 6, 8), generator=generator).double()

    def resize_image(image):
        sums = image.reshape(1, 4, 2, 6, 2, -1).sum((2, 4))
        return torch.floor((sums + 2) / 4)

    rgb = [resize_image(image) for image in images]
    values = torch.cat((*rgb, features, torch.full((1, 4, 6, 1), 40)), dim=-1)
    before = F.conv2d(
        values.permute(0, 3, 1, 2) - zero_point,
        weight.permute(0, 3, 1, 2).double(),
        bias,
    )
    padded = [
        resize_image(torch.cat((image, torch.full((1, 8, 12, 1), code)), dim=-1))
        for image, code in zip(images, (40, 0))
    ]
    values = torch.cat((*padded, features), dim=-1)
    after = F.conv2d(
        values.permute(0, 3, 1, 2) - zero_point,
        extended.permute(0, 3, 1, 2).double(),
        bias,
    )
    assert torch.equal(before, after)
    assert not pad.PadRifeBlock0ConcatPass(program)(program.graph_module).modified
    program.graph.lint()


@pytest.mark.parametrize(
    "unsupported",
    [
        "axis",
        "dtype",
        "channels",
        "weight_zp",
        "weight_shape",
        "shared",
        "timestep",
        "rescale",
        "resize",
        "shared_resize",
    ],
)
def test_unsupported_concat_is_unchanged(unsupported):
    program, inputs, concat, conv = _program()
    if unsupported == "axis":
        concat.kwargs = {"axis": 1}
    elif unsupported == "dtype":
        inputs[0].meta["val"] = inputs[0].meta["val"].to(torch.uint8)
    elif unsupported == "channels":
        inputs[0].meta["val"] = inputs[0].meta["val"].new_empty((1, 4, 6, 4))
    elif unsupported == "weight_zp":
        conv.meta["input_qparams"][1].zp = 1
    elif unsupported == "weight_shape":
        program.constants["weight"] = torch.zeros(3, 1, 1, 16, dtype=torch.int8)
    elif unsupported == "timestep":
        program.constants["timestep"][0, 0, 0, 0] = 39
    elif unsupported == "rescale":
        node = concat.args[0][0]
        node.args = (node.args[0], torch.int8, [0.5], 0, 0)
    elif unsupported == "resize":
        node = concat.args[0][0].args[0]
        node.kwargs = {"resize_mode": "nearest"}
    elif unsupported == "shared_resize":
        program.graph.output_node().args = ((conv, concat.args[0][0].args[0]),)
    else:
        program.graph.output_node().args = ((conv, concat),)
    before = str(program.graph)
    constants = dict(program.constants)
    assert not pad.PadRifeBlock0ConcatPass(program)(program.graph_module).modified
    assert str(program.graph) == before
    assert program.constants.keys() == constants.keys()
    assert all(torch.equal(program.constants[k], v) for k, v in constants.items())


def test_requires_exported_program():
    program, _, _, _ = _program()
    with pytest.raises(RuntimeError, match="requires an ExportedProgram"):
        pad.PadRifeBlock0ConcatPass()(program.graph_module)


@pytest.mark.parametrize(
    "consumer",
    ["packed", "warp_rgb", "warp_rgb_nchw", "warp_rgba", "warp_rgba_nchw", "unknown"],
)
def test_reuses_padding_only_when_alpha_is_unused(consumer):
    import operator

    program, inputs, concat, conv = _program()
    graph = program.graph
    with graph.inserting_before(graph.output_node()):
        image = graph.call_function(
            TOSA.PAD.default, (inputs[0], [0, 0, 0, 0, 0, 0, 0, 1]), {"value": -11}
        )
        image.meta["val"] = inputs[0].meta["val"].new_empty((1, 8, 12, 4))
        name = (
            pad.OPERATOR_NAME
            if consumer == "packed"
            else pad.flow_offset_grid_sampler_operator_name()
        )
        custom = graph.call_function(
            TOSA.CUSTOM.default,
            ([image],),
            {
                "domain_name": pad.CUSTOM_SHADER_DOMAIN_NAME,
                "operator_name": "unknown" if consumer == "unknown" else name,
            },
        )
        output = graph.call_function(operator.getitem, (custom, 0))
        channels = 4 if consumer.startswith("warp_rgba") else 3
        shape = [1, 8, 12, channels]
        if consumer.endswith("_nchw"):
            output = graph.call_function(TOSA.TRANSPOSE.default, (output, [0, 3, 1, 2]))
            shape = [1, channels, 8, 12]
        crop = graph.call_function(TOSA.SLICE.default, (output, [0, 0, 0, 0], shape))
    graph.output_node().args = ((conv, crop),)
    assert pad.PadRifeBlock0ConcatPass(program)(program.graph_module).modified
    padded_input = concat.args[0][0].args[0].args[0]
    if consumer in ("packed", "warp_rgb", "warp_rgb_nchw"):
        assert padded_input is image
        assert image.kwargs["value"] == 40
        assert list(graph.nodes).index(image) < list(graph.nodes).index(concat)
    else:
        assert padded_input is not image
        assert image.kwargs["value"] == -11
    graph.lint()


def test_preserves_order_when_pads_share_a_shape_constant():
    program, inputs, concat, conv = _program()
    graph = program.graph
    with graph.inserting_before(graph.output_node()):
        shape = graph.call_function(
            TOSA.CONST_SHAPE.default, ([0, 0, 0, 0, 0, 0, 0, 1],)
        )
        images = []
        for source in inputs[:2]:
            image = graph.call_function(
                TOSA.PAD.default, (source, shape), {"value": -11}
            )
            image.meta["val"] = source.meta["val"].new_empty((1, 8, 12, 4))
            images.append(image)
        custom = graph.call_function(
            TOSA.CUSTOM.default,
            (images,),
            {
                "domain_name": pad.CUSTOM_SHADER_DOMAIN_NAME,
                "operator_name": pad.OPERATOR_NAME,
            },
        )
    graph.output_node().args = ((conv, custom),)
    assert pad.PadRifeBlock0ConcatPass(program)(program.graph_module).modified
    assert [branch.args[0].args[0] for branch in concat.args[0][:2]] == images
    assert sum(n.target == TOSA.PAD.default for n in graph.nodes) == 2
    graph.lint()
