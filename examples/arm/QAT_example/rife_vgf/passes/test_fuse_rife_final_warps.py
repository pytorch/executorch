# Copyright 2026 Arm Limited and/or its affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import json
import operator
from types import SimpleNamespace
from unittest.mock import patch

import pytest
import torch
import torch.nn.functional as F
from examples.arm.QAT_example.rife_vgf.passes import fuse_rife_final_warps as fuse
from examples.arm.QAT_example.rife_vgf.shaders import final_warps, pack_warps
from executorch.backends.arm._passes.arm_pass_manager import (
    _ExportedProgramGraphPassAdapter,
)
from executorch.backends.arm.vgf.shaders import grid_sampler
from executorch.backends.transforms.utils import create_constant_placeholder
from torch.export.graph_signature import (
    InputKind,
    OutputKind,
    OutputSpec,
    TensorArgument,
)

TOSA = fuse.TOSA


class _Inputs(torch.nn.Module):
    def forward(self, *inputs):
        return inputs[0]


def _program(height=4, width=6):
    program = torch.export.export(
        _Inputs(),
        (
            torch.zeros(1, height, width, 4, dtype=torch.int8),
            torch.zeros(1, height, width, 4, dtype=torch.int8),
            torch.zeros(1, height, width, 32, dtype=torch.int8),
            torch.zeros(1, height // 2, width // 2, 2, dtype=torch.int8),
        ),
    )
    graph = program.graph
    graph.erase_node(graph.output_node())
    inputs = list(graph.nodes)

    def node(target, args, shape, dtype=torch.int8, **kwargs):
        n = graph.call_function(target, args, kwargs)
        n.meta["val"] = inputs[0].meta["val"].new_empty(shape, dtype=dtype)
        return n

    def constant(name, value):
        with graph.inserting_before(next(iter(graph.nodes))):
            return create_constant_placeholder(
                program, graph, name, InputKind.CONSTANT_TENSOR, value
            )

    packed_custom = graph.call_function(
        TOSA.CUSTOM.default,
        ([inputs[2]],),
        {
            "operator_name": pack_warps.OPERATOR_NAME,
            "domain_name": grid_sampler.CUSTOM_SHADER_DOMAIN_NAME,
        },
    )
    packed_custom.meta["val"] = [inputs[2].meta["val"]]
    packed = node(operator.getitem, (packed_custom, 0), (1, height, width, 32))
    crop = node(
        TOSA.SLICE.default,
        (packed, [0, 0, 0, 24], [1, height, width, 4]),
        (1, height, width, 4),
    )
    transpose = node(
        TOSA.TRANSPOSE.default, (crop, [0, 3, 1, 2]), (1, 4, height, width)
    )
    previous = node(
        TOSA.RESCALE.default,
        (transpose, torch.int32, [65536.0], -14, 0),
        (1, 4, height, width),
        torch.int32,
    )
    weight = constant("weight", torch.arange(40, dtype=torch.int8).reshape(20, 1, 1, 2))
    bias = constant("bias", torch.arange(20, dtype=torch.int32))
    producer = node(
        TOSA.TRANSPOSE_CONV2D.default,
        (inputs[3], weight, bias, [0, 0, 0, 0], [1, 1]),
        (1, height // 2, width // 2, 20),
        torch.int32,
    )
    producer.meta["input_qparams"] = {1: SimpleNamespace(zp=0)}
    phase = node(
        TOSA.RESCALE.default,
        (producer, torch.int8, [0.001 * (i + 1) for i in range(20)], 0, 7),
        (1, height // 2, width // 2, 20),
    )
    reshape = node(
        TOSA.RESHAPE.default,
        (phase, [1, height // 2, width // 2, 5, 2, 2]),
        (1, height // 2, width // 2, 5, 2, 2),
    )
    shuffle = node(
        TOSA.TRANSPOSE.default,
        (reshape, [0, 3, 1, 4, 2, 5]),
        (1, 5, height // 2, 2, width // 2, 2),
    )
    nchw = node(
        TOSA.RESHAPE.default, (shuffle, [1, 5, height, width]), (1, 5, height, width)
    )
    delta_crop = node(
        TOSA.SLICE.default,
        (nchw, [0, 0, 0, 0], [1, 4, height, width]),
        (1, 4, height, width),
    )
    delta = node(
        TOSA.RESCALE.default,
        (delta_crop, torch.int32, [524288.0], 7, 0),
        (1, 4, height, width),
        torch.int32,
    )
    mask = node(
        TOSA.SLICE.default,
        (nchw, [0, 4, 0, 0], [1, 1, height, width]),
        (1, 1, height, width),
    )
    add = node(TOSA.ADD.default, (previous, delta), (1, 4, height, width), torch.int32)
    flow = node(
        TOSA.RESCALE.default, (add, torch.int8, [0.00001], 0, -1), (1, 4, height, width)
    )
    outputs = []
    for image, offset in zip(inputs[:2], (0, 2)):
        with patch.object(
            grid_sampler,
            "_compile_flow_offset_grid_sampler_shader",
            return_value="unused",
        ):
            payload = grid_sampler.build_flow_offset_grid_sampler_payload(
                input_shape=(1, 4, height, width),
                output_shape=(1, 4, height, width),
                flow_shape=(1, 4, height, width),
                flow_scale=0.01,
                flow_zero_point=-1,
                input_scale=0.003,
                input_zero_point=-92,
                output_scale=0.01 + offset * 0.001,
                output_zero_point=-14,
                flow_channel_offset=offset,
            )
        custom = graph.call_function(
            TOSA.CUSTOM.default,
            ([image, flow],),
            {
                "operator_name": grid_sampler.flow_offset_grid_sampler_operator_name(),
                "domain_name": grid_sampler.CUSTOM_SHADER_DOMAIN_NAME,
                "implementation_attrs": list(json.dumps(payload).encode()),
            },
        )
        custom.meta["val"] = [image.meta["val"]]
        outputs.append(node(operator.getitem, (custom, 0), (1, height, width, 4)))
    lut = constant("sigmoid", torch.arange(-128, 128, dtype=torch.int8))
    table = node(TOSA.TABLE.default, (mask, lut), (1, 1, height, width))
    # This remaining consumer represents the final graph's mask preparation.
    blend_input = node(
        TOSA.RESCALE.default,
        (table, torch.int32, [1.0], -127, 0),
        (1, 1, height, width),
        torch.int32,
    )
    graph.output((*outputs, blend_input))
    program.graph_signature.output_specs[:] = [
        OutputSpec(OutputKind.USER_OUTPUT, TensorArgument(name=n.name), None)
        for n in [*outputs, blend_input]
    ]
    return program, packed, phase, previous, delta, flow, table, blend_input


@pytest.mark.parametrize("adapted", [False, True])
@pytest.mark.parametrize("shape", [(4, 6), (8, 12)])
def test_fuses_tail_and_preserves_blend_inputs(adapted, shape):
    program, packed, phase, previous, delta, flow, table, blend = _program(*shape)
    producer = phase.args[0]
    old_weight, old_bias = [
        fuse._constant_tensor(program, n).clone() for n in producer.args[1:3]
    ]
    old_scales = list(phase.args[2])
    with patch.object(fuse, "build_final_warps_payload", return_value={}) as build:
        result = (
            _ExportedProgramGraphPassAdapter(fuse.FuseRifeFinalWarpsPass()).call(
                program
            )
            if adapted
            else fuse.FuseRifeFinalWarpsPass(program)(program.graph_module)
        )
    assert result.modified
    custom = table.args[0]
    assert custom.kwargs["operator_name"] == final_warps.OPERATOR_NAME
    assert list(custom.args[0][2:]) == [packed, phase]
    assert [tuple(v.shape) for v in custom.meta["val"]] == [
        (1, *shape, 4),
        (1, *shape, 4),
        (1, 1, *shape),
    ]
    assert blend.args[0] is table
    assert table.target == operator.getitem and table.args[1] == 2
    assert build.call_args.args[1] == (previous, delta, flow)
    assert not any(
        n.target
        in (
            TOSA.SLICE.default,
            TOSA.TRANSPOSE.default,
            TOSA.ADD.default,
            TOSA.TABLE.default,
        )
        for n in program.graph.nodes
    )
    order = torch.arange(20).reshape(5, 4).t().reshape(-1)
    assert torch.equal(
        fuse._constant_tensor(program, producer.args[1]), old_weight[order]
    )
    assert torch.equal(
        fuse._constant_tensor(program, producer.args[2]), old_bias[order]
    )
    assert phase.args[2] == [old_scales[i] for i in order]
    source = torch.arange(2 * 3 * 20).reshape(1, 2, 3, 20)
    expected = F.pixel_shuffle(source.permute(0, 3, 1, 2), 2).permute(0, 2, 3, 1)
    reordered = source[..., order]
    for y in range(4):
        for x in range(6):
            offset = (2 * (y % 2) + x % 2) * 5
            assert torch.equal(
                reordered[0, y // 2, x // 2, offset : offset + 5], expected[0, y, x]
            )
    program.graph.lint()
    assert not fuse.FuseRifeFinalWarpsPass(program)(program.graph_module).modified


@pytest.mark.parametrize(
    "unsupported",
    ["shared_flow", "shared_phase", "wrong_channels", "unsigned", "weight_zp"],
)
def test_leaves_unsupported_graph_unchanged(unsupported):
    program, packed, phase, previous, delta, flow, table, blend = _program()
    if unsupported.startswith("shared"):
        output = program.graph.output_node()
        output.args = (
            (*output.args[0], flow if unsupported == "shared_flow" else phase),
        )
    elif unsupported == "wrong_channels":
        previous.args[0].args[0].args = (packed, [0, 0, 0, 20], [1, 4, 6, 4])
    elif unsupported == "unsigned":
        delta.kwargs = {"input_unsigned": True}
    else:
        phase.args[0].meta["input_qparams"][1].zp = 1
    original = str(program.graph)
    with patch.object(fuse, "build_final_warps_payload") as build:
        assert not fuse.FuseRifeFinalWarpsPass(program)(program.graph_module).modified
    build.assert_not_called()
    assert str(program.graph) == original


def test_payload_preserves_sampling_and_single_round_integer_rescales():
    program = _program()[0]
    _, payloads, rescales, _, _, _, sigmoid, _, _ = fuse.FuseRifeFinalWarpsPass(
        program
    )._match(program.graph_module)
    with patch.object(
        final_warps, "_compile_shader_source", return_value="compiled"
    ) as compile_shader:
        payload = final_warps.build_final_warps_payload(payloads, rescales, sigmoid)
    source = compile_shader.call_args.args[0]
    assert "@" not in source
    assert "phase * 5u" in source
    assert "* 8u + 6u" in source
    assert "imulExtended" in source
    assert "rescaleSum(rescalePrevious(code) + rescaleDelta" in source
    assert "SIGMOID_TABLE[int(state[4]) + 128]" in source
    for index in range(2):
        assert payload[f"output_{index}_type"] == "Image"
        assert (
            payload[f"output_{index}_vkformat"] == payloads[index]["output_0_vkformat"]
        )
    assert payload["input_2_type"] == "Buffer"
    assert payload["output_2_type"] == "Tensor"
    assert not any(key.startswith("output_3_") for key in payload)


def test_fake_outputs_match_shader_contract():
    image = torch.empty(1, 8, 12, 4, dtype=torch.int8)
    outputs = fuse._final_warps_fake([image, image], None, None, None)
    assert [tuple(v.shape) for v in outputs] == [
        (1, 8, 12, 4),
        (1, 8, 12, 4),
        (1, 1, 8, 12),
    ]
    assert all(v.dtype == torch.int8 for v in outputs)
