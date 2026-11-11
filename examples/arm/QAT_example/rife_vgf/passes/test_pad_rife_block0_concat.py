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


def _program(factor=2):
    tensors = [torch.zeros(1, 4 * factor, 6 * factor, 3, dtype=torch.int8)] * 2
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
            (
                image,
                [2, 2 * factor, 2, 2 * factor],
                [factor - 1, factor - 1],
                [1 - factor, 1 - factor],
            ),
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
@pytest.mark.parametrize("factor", [2, 4])
def test_pads_concat_and_preserves_convolution(adapted, zero_point, factor):
    program, _, concat, conv = _program(factor)
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
        assert image.args[2].args == ([code], [1], torch.int8)
        assert pad._shape_values(image.args[1]) == [0, 0, 0, 0, 0, 0, 0, 1]
        assert tuple(image.meta["val"].shape) == (1, 4 * factor, 6 * factor, 4)
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
    images = torch.randint(
        -128, 128, (2, 1, 4 * factor, 6 * factor, 3), generator=generator
    ).double()
    features = torch.randint(-128, 128, (1, 4, 6, 8), generator=generator).double()

    def resize_image(image):
        center = factor // 2 - 1
        sums = sum(
            image[:, center + y :: factor, center + x :: factor, :]
            for y in (0, 1)
            for x in (0, 1)
        )
        return torch.floor((sums + 2) / 4)

    rgb = [resize_image(image) for image in images]
    for image, resized in zip(images, rgb):
        reference = F.interpolate(
            image.permute(0, 3, 1, 2),
            scale_factor=1.0 / factor,
            mode="bilinear",
            align_corners=False,
        ).permute(0, 2, 3, 1)
        assert torch.equal(resized, torch.floor(reference + 0.5))
    values = torch.cat((*rgb, features, torch.full((1, 4, 6, 1), 40)), dim=-1)
    before = F.conv2d(
        values.permute(0, 3, 1, 2) - zero_point,
        weight.permute(0, 3, 1, 2).double(),
        bias,
    )
    padded = [
        resize_image(
            torch.cat((image, torch.full((1, 4 * factor, 6 * factor, 1), code)), dim=-1)
        )
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
        "resize_offset",
        "resize_border",
        "resize_shape",
        "shared_resize",
    ],
)
@pytest.mark.parametrize("factor", [2, 4])
def test_unsupported_concat_is_unchanged(unsupported, factor):
    program, inputs, concat, conv = _program(factor)
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
    elif unsupported in ("resize_offset", "resize_border"):
        node = concat.args[0][0].args[0]
        args = list(node.args)
        args[2 if unsupported == "resize_offset" else 3] = [0, 0]
        node.args = tuple(args)
    elif unsupported == "resize_shape":
        image = inputs[0]
        image.meta["val"] = image.meta["val"].new_empty(
            (1, 4 * factor, 6 * factor + 1, 3)
        )
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


@pytest.mark.parametrize("factor", [2, 4])
@pytest.mark.parametrize("before_resize", [False, True])
@pytest.mark.parametrize("code", [-11, 0, 40])
def test_reuses_padding_only_when_value_matches(factor, before_resize, code):
    program, inputs, concat, conv = _program(factor)
    graph = program.graph
    resize = concat.args[0][0].args[0]
    with graph.inserting_before(resize if before_resize else graph.output_node()):
        pad_value = graph.call_function(TOSA.CONST.default, ([code], [1], torch.int8))
        pad_value.meta["val"] = inputs[0].meta["val"].new_empty((1,))
        image = graph.call_function(
            TOSA.PAD.default, (inputs[0], [0, 0, 0, 0, 0, 0, 0, 1], pad_value)
        )
        image.meta["val"] = (
            inputs[0].meta["val"].new_empty((1, 4 * factor, 6 * factor, 4))
        )
        consumer = graph.call_function(TOSA.ADD.default, (image, image))
        consumer.meta = dict(image.meta)
    graph.output_node().args = ((conv, consumer),)
    original_args = image.args
    assert pad.PadRifeBlock0ConcatPass(program)(program.graph_module).modified
    padded_input = concat.args[0][0].args[0].args[0]
    assert (padded_input is image) == (code == 40)
    assert image.args == original_args
    assert pad_value.args == ([code], [1], torch.int8)
    assert consumer.args == (image, image)
    graph.lint()


@pytest.mark.parametrize("final_blend", [False, True])
def test_preserves_shared_padding_values(final_blend):
    program, inputs, concat, conv = _program()
    graph = program.graph
    with graph.inserting_before(graph.output_node()):
        shape = graph.call_function(
            TOSA.CONST_SHAPE.default, ([0, 0, 0, 0, 0, 0, 0, 1],)
        )
        pad_value = graph.call_function(TOSA.CONST.default, ([-11], [1], torch.int8))
        images = []
        for source in inputs[:2]:
            image = graph.call_function(TOSA.PAD.default, (source, shape, pad_value))
            image.meta["val"] = source.meta["val"].new_empty((1, 8, 12, 4))
            images.append(image)
        custom = graph.call_function(
            TOSA.CUSTOM.default,
            (images,),
            {
                "domain_name": "test",
                "operator_name": "test",
            },
        )
        outputs = [conv, custom]
        if final_blend:
            outputs.append(
                graph.call_function(
                    TOSA.CUSTOM.default,
                    (images,),
                    {
                        "domain_name": "test",
                        "operator_name": "test",
                    },
                )
            )
    graph.output_node().args = (tuple(outputs),)
    assert pad.PadRifeBlock0ConcatPass(program)(program.graph_module).modified
    assert all(
        branch.args[0].args[0] is not image
        for branch, image in zip(concat.args[0][:2], images)
    )
    assert all(image.args[2] is pad_value for image in images)
    assert pad_value.args == ([-11], [1], torch.int8)
    assert sum(n.target == TOSA.PAD.default for n in graph.nodes) == 4
    graph.lint()
