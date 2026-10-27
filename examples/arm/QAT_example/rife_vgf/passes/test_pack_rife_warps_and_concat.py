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
from examples.arm.QAT_example.rife_vgf.passes import pack_rife_warps_and_concat as pack
from examples.arm.QAT_example.rife_vgf.shaders import pack_warps
from executorch.backends.arm._passes.arm_pass_manager import (
    _ExportedProgramGraphPassAdapter,
)
from executorch.backends.arm.tosa.specification import (
    TosaLoweringContext,
    TosaSpecification,
)
from executorch.backends.arm.vgf.shaders import grid_sampler
from executorch.backends.transforms.utils import create_constant_placeholder
from torch.export.graph_signature import (
    InputKind,
    OutputKind,
    OutputSpec,
    TensorArgument,
)

TOSA = pack.TOSA


class _Inputs(torch.nn.Module):
    def forward(self, *inputs):
        return inputs[0]


def _program(code=45, width=6, normalization=(1.0, 1.0)):
    tensors = [
        torch.zeros(1, 4, width, c, dtype=torch.int8) for c in (4, 4, 4, 4, 1, 8, 4)
    ]
    program = torch.export.export(_Inputs(), tuple(tensors))
    graph = program.graph
    graph.erase_node(graph.output_node())
    inputs = list(graph.nodes)

    def node(target, args, shape, **kwargs):
        result = graph.call_function(target, args, kwargs)
        result.meta["val"] = inputs[0].meta["val"].new_empty(shape)
        return result

    def constant(name, tensor):
        with graph.inserting_before(next(iter(graph.nodes))):
            return create_constant_placeholder(
                program, graph, name, InputKind.CONSTANT_TENSOR, tensor
            )

    flow = node(TOSA.TRANSPOSE.default, (inputs[-1], [0, 3, 1, 2]), (1, 4, 4, width))
    outputs = []
    for image, offset in zip(inputs[:4], (0, 2, 0, 2)):
        with patch.object(
            grid_sampler,
            "_compile_flow_offset_grid_sampler_shader",
            return_value="unused",
        ):
            payload = grid_sampler.build_flow_offset_grid_sampler_payload(
                input_shape=(1, 4, 4, width),
                output_shape=(1, 4, 4, width),
                flow_shape=(1, 4, 4, width),
                flow_scale=0.01,
                flow_zero_point=-14,
                input_scale=0.003,
                input_zero_point=-92,
                output_scale=0.01,
                output_zero_point=-14,
                flow_channel_offset=offset,
                flow_normalization=normalization,
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
        output = node(operator.getitem, (custom, 0), (1, 4, width, 4))
        if len(outputs) < 2:
            output = node(
                TOSA.SLICE.default,
                (output, [0, 0, 0, 0], [1, 4, width, 3]),
                (1, 4, width, 3),
            )
        outputs.append(output)
    timestep = constant(
        "timestep", torch.full((1, 4, width, 1), code, dtype=torch.int8)
    )
    concat = node(
        TOSA.CONCAT.default,
        ([*outputs, timestep, *inputs[4:]],),
        (1, 4, width, 28),
        axis=3,
    )
    weight = constant(
        "weight",
        torch.arange(84).remainder(11).sub(5).to(torch.int8).reshape(3, 1, 1, 28),
    )
    bias = constant("bias", torch.tensor([-17, 9, 23], dtype=torch.int32))
    conv = node(
        TOSA.CONV2D.default,
        (concat, weight, bias, [1, 1], [0, 0, 0, 0], [1, 1]),
        (1, 4, width, 3),
    )
    conv.meta["input_qparams"] = {0: SimpleNamespace(zp=-11), 1: SimpleNamespace(zp=0)}
    graph.output((conv,))
    return program, inputs, concat, conv, weight


@pytest.mark.parametrize("code", [-128, -1, 0, 45, 127])
def test_packs_private_warps_and_preserves_convolution(code):
    program, inputs, concat, conv, old_weight = _program(code)
    original = pack._constant_tensor(program, old_weight).clone()
    with patch.object(pack, "build_packed_warp_payload", return_value={}) as build:
        assert (
            _ExportedProgramGraphPassAdapter(pack.PackRifeWarpsAndConcatPass())
            .call(program)
            .modified
        )
    assert build.call_args.args[1] == code
    assert concat.target == operator.getitem
    custom = concat.args[0]
    assert custom.args[0] == [*inputs[:4], *inputs[4:]]
    assert custom.kwargs["operator_name"] == pack_warps.OPERATOR_NAME
    assert tuple(concat.meta["val"].shape) == (1, 4, 6, 32)
    assert sum(n.target == TOSA.CUSTOM.default for n in program.graph.nodes) == 1
    assert not any(
        n.target in (TOSA.SLICE.default, TOSA.CONCAT.default)
        for n in program.graph.nodes
    )
    weight = pack._constant_tensor(program, conv.args[1])
    assert torch.equal(weight[..., :28], original)
    assert torch.count_nonzero(weight[..., 28:]) == 0
    assert torch.equal(pack._constant_tensor(program, old_weight), original)

    # Arbitrary values in the dummy channels must not change INT32 accumulators,
    # including when the activation zero point is nonzero.
    x = torch.arange(24 * 28).remainder(256).sub(128).reshape(1, 4, 6, 28).double()
    extended = torch.cat((x, torch.full((1, 4, 6, 4), 73.0)), dim=-1)
    bias = pack._constant_tensor(program, conv.args[2]).double()
    before = F.conv2d(
        x.permute(0, 3, 1, 2) + 11, original.permute(0, 3, 1, 2).double(), bias
    )
    after = F.conv2d(
        extended.permute(0, 3, 1, 2) + 11, weight.permute(0, 3, 1, 2).double(), bias
    )
    assert torch.equal(before, after)
    assert not pack.PackRifeWarpsAndConcatPass(program)(program.graph_module).modified
    program.graph.lint()


@pytest.mark.parametrize(
    "unsupported",
    [
        "domain",
        "operator",
        "qparams",
        "normalization",
        "sampler",
        "offset",
        "shared",
        "weight_zp",
        "timestep",
    ],
)
def test_unsupported_pattern_is_unchanged(unsupported):
    program, _, concat, conv, _ = _program()
    custom = next(n for n in program.graph.nodes if n.target == TOSA.CUSTOM.default)
    if unsupported in ("domain", "operator"):
        key = "domain_name" if unsupported == "domain" else "operator_name"
        custom.kwargs = {**custom.kwargs, key: "unknown"}
    elif unsupported in ("qparams", "sampler", "offset", "normalization"):
        payload = json.loads(bytes(custom.kwargs["implementation_attrs"]))
        if unsupported == "qparams":
            payload["output_scale"] *= 2
        elif unsupported == "sampler":
            payload["input_0_sampler"] = {}
        elif unsupported == "normalization":
            payload["flow_normalization"] = [0.25, 0.125]
        else:
            payload["flow_channel_offset"] = 2
        custom.kwargs = {
            **custom.kwargs,
            "implementation_attrs": list(json.dumps(payload).encode()),
        }
    elif unsupported == "shared":
        program.graph.output_node().args = ((conv, concat.args[0][0]),)
    elif unsupported == "weight_zp":
        conv.meta["input_qparams"][1].zp = 1
    else:
        program.constants["timestep"][0, 0, 0, 0] = 46
    before = str(program.graph)
    constants = set(program.constants)
    with patch.object(pack, "build_packed_warp_payload") as build:
        assert not pack.PackRifeWarpsAndConcatPass(program)(
            program.graph_module
        ).modified
        build.assert_not_called()
    assert str(program.graph) == before
    assert set(program.constants) == constants


@pytest.mark.parametrize("normalization", [(1.0, 1.0), (0.25, 0.125)])
def test_payload_binding_contract_and_source_specialization(normalization):
    program, _, _, _, _ = _program(normalization=normalization)
    payloads = [
        pack._payload(n) for n in program.graph.nodes if n.target == TOSA.CUSTOM.default
    ]
    with patch.object(
        pack_warps, "_compile_shader_source", return_value="compiled"
    ) as compile_shader:
        payload = pack_warps.build_packed_warp_payload(payloads, -128)
    source = compile_shader.call_args.args[0]
    assert "@" not in source
    assert "values[14] = int8_t(-128);" in source
    factors = ", ".join(format(value, ".9g") for value in normalization)
    assert f"const vec2 FLOW_NORMALIZATION = vec2({factors});" in source
    for i in range(7):
        assert payload[f"input_{i}_binding"] == i
        assert payload[f"input_{i}_type"] == ("Image" if i < 4 else "Tensor")
    assert payload["output_0_binding"] == 7
    assert payload["output_0_type"] == "Buffer"
    assert payload["workgroup_sizes"] == [1, 1, 1]


def _tail_program():
    program, inputs, concat, conv, _ = _program(width=8)
    graph = program.graph
    customs = [n for n in graph.nodes if n.target == TOSA.CUSTOM.default]
    flow_transpose = customs[0].args[0][1]
    constants = []
    with graph.inserting_before(next(iter(graph.nodes))):
        for name, value in (
            ("phase_input", torch.arange(4, dtype=torch.int8).reshape(1, 1, 2, 2)),
            ("phase_weight", torch.arange(104, dtype=torch.int8).reshape(52, 1, 1, 2)),
            ("phase_bias", torch.arange(52, dtype=torch.int32)),
        ):
            constants.append(
                create_constant_placeholder(
                    program, graph, name, InputKind.CONSTANT_TENSOR, value
                )
            )

    def node(target, args, shape, dtype=torch.int8, **kwargs):
        result = graph.call_function(target, args, kwargs)
        result.meta["val"] = inputs[0].meta["val"].new_empty(shape, dtype=dtype)
        return result

    with graph.inserting_before(flow_transpose):
        producer = node(
            TOSA.TRANSPOSE_CONV2D.default,
            (*constants, [0, 0, 0, 0], [1, 1]),
            (1, 1, 2, 52),
            torch.int32,
        )
        producer.meta["input_qparams"] = {1: SimpleNamespace(zp=0)}
        phase = node(
            TOSA.RESCALE.default,
            (producer, torch.int8, [0.001 * (i + 1) for i in range(52)], 0, -14),
            (1, 1, 2, 52),
        )
        reshape = node(
            TOSA.RESHAPE.default, (phase, [1, 1, 2, 13, 2, 2]), (1, 1, 2, 13, 2, 2)
        )
        transpose = node(
            TOSA.TRANSPOSE.default, (reshape, [0, 3, 1, 4, 2, 5]), (1, 13, 1, 2, 2, 2)
        )
        nchw = node(TOSA.RESHAPE.default, (transpose, [1, 13, 2, 4]), (1, 13, 2, 4))
        nhwc = node(TOSA.TRANSPOSE.default, (nchw, [0, 2, 3, 1]), (1, 2, 4, 13))
        mask = node(
            TOSA.SLICE.default, (nchw, [0, 4, 0, 0], [1, 1, 2, 4]), (1, 1, 2, 4)
        )
        mask = node(TOSA.RESHAPE.default, (mask, [1, 2, 4, 1]), (1, 2, 4, 1))
        features = node(
            TOSA.SLICE.default, (nhwc, [0, 0, 0, 5], [1, 2, 4, 8]), (1, 2, 4, 8)
        )
        flow = node(
            TOSA.SLICE.default, (nhwc, [0, 0, 0, 0], [1, 2, 4, 4]), (1, 2, 4, 4)
        )
        states = []
        for source, channels in zip((mask, features, flow), (1, 8, 4)):
            resize = node(
                TOSA.RESIZE.default,
                (source, [4, 2, 4, 2], [-1, -1], [1, 1]),
                (1, 4, 8, channels),
                torch.int32,
                resize_mode="bilinear",
            )
            states.append(
                node(
                    TOSA.RESCALE.default,
                    (resize, torch.int8, [0.0625], 0, 0),
                    (1, 4, 8, channels),
                )
            )
    flow_transpose.args = (states[2], [0, 3, 1, 2])
    concat.args = ([*concat.args[0][:5], *states],)
    with graph.inserting_before(customs[0]):
        residual = node(
            TOSA.RESCALE.default,
            (flow_transpose, torch.int32, [91861.15014648438], -14, 0),
            (1, 4, 4, 8),
            torch.int32,
        )
    graph.output_node().args = ((conv, residual),)
    program.graph_signature.output_specs.append(
        OutputSpec(OutputKind.USER_OUTPUT, TensorArgument(name=residual.name), None)
    )
    return program, phase, concat, customs, residual


@pytest.mark.parametrize("adapted", [False, True])
def test_fuses_block0_tail_with_phase_major_input(adapted):
    program, phase, concat, customs, residual = _tail_program()
    conv = program.graph.output_node().args[0][0]
    expected = pack._match_block0_tail(concat, customs)
    assert expected == (phase, residual)
    rescale_parameters = residual.args[1:]
    rescale_kwargs = dict(residual.kwargs)
    with patch.object(pack, "build_packed_warp_payload", return_value={}) as build:
        result = (
            _ExportedProgramGraphPassAdapter(pack.PackRifeWarpsAndConcatPass()).call(
                program
            )
            if adapted
            else pack.PackRifeWarpsAndConcatPass(program)(program.graph_module)
        )
        assert result.modified
    assert build.call_args.kwargs == {"fuse_block0_tail": True}
    custom = concat.args[0]
    assert len(custom.args[0]) == 5
    assert custom.args[0][-1] is phase
    assert [v.dtype for v in custom.meta["val"]] == [torch.int8]
    assert [tuple(v.shape) for v in custom.meta["val"]] == [(1, 4, 8, 32)]
    assert residual.target == TOSA.RESCALE.default
    assert residual.args[1:] == rescale_parameters
    assert residual.kwargs == rescale_kwargs
    assert residual.meta["val"].dtype == torch.int32
    flow = residual.args[0]
    assert flow.target == TOSA.TRANSPOSE.default
    flow_slice = flow.args[0]
    assert flow.args[1] == [0, 3, 1, 2]
    assert flow_slice.target == TOSA.SLICE.default
    assert flow_slice.args[0] is concat
    assert set(custom.users) == {concat}
    assert set(concat.users) == {conv, flow_slice}
    assert conv.args[0] is concat
    assert pack._shape_values(flow_slice.args[1]) == [0, 0, 0, 24]
    assert pack._shape_values(flow_slice.args[2]) == [1, 4, 8, 4]
    for node in flow_slice.args[1:]:
        assert (
            node.meta[pack.TosaSpecialDtype.meta_key()] == pack.TosaSpecialDtype.SHAPE
        )
    assert tuple(flow_slice.meta["val"].shape) == (1, 4, 8, 4)
    assert tuple(flow.meta["val"].shape) == (1, 4, 4, 8)
    assert flow.meta["val"].dtype == torch.int8
    assert flow.meta["tensor_meta"].dtype == torch.int8
    assert program.graph.output_node().args[0][1] is residual
    assert not any(
        n.target in (TOSA.RESIZE.default, TOSA.RESHAPE.default)
        for n in program.graph.nodes
    )
    assert sum(n.target == TOSA.RESCALE.default for n in program.graph.nodes) == 2
    assert sum(n.target == TOSA.SLICE.default for n in program.graph.nodes) == 1
    assert sum(n.target == TOSA.TRANSPOSE.default for n in program.graph.nodes) == 1
    positions = {node: i for i, node in enumerate(program.graph.nodes)}
    assert (
        positions[custom]
        < positions[concat]
        < positions[flow_slice]
        < positions[flow]
        < positions[residual]
    )
    program.graph.lint()


def test_fused_payload_reads_thirteen_channels_and_writes_only_c32():
    program, _, _, customs, _ = _tail_program()
    payloads = [pack._payload(n) for n in customs]
    with patch.object(
        pack_warps, "_compile_shader_source", return_value="compiled"
    ) as compile_shader:
        payload = pack_warps.build_packed_warp_payload(
            payloads, 45, fuse_block0_tail=True
        )
    source = compile_shader.call_args.args[0]
    assert "@" not in source
    assert "#define FUSE_BLOCK0_TAIL 1" in source
    assert "out int8_t result[13]" in source
    assert "phase * 13u" in source
    assert "[52]" not in source
    assert "tensorSizeARM(tailPhase, 2)" in source
    assert payload["input_4_binding"] == 4
    assert "input_5_binding" not in payload
    assert payload["output_0_binding"] == 5
    assert not any(key.startswith("output_1_") for key in payload)
    assert "previousFlow" not in source
    assert "tensorWriteARM" not in source
    assert "rescalePreviousFlow" not in source
    assert "imulExtended" not in source
    fake_outputs = pack._packed_fake(
        [n.meta["val"] for n in customs[0].args[0][:1]] * 4
        + [torch.empty(1, 1, 2, 52, dtype=torch.int8)],
        pack_warps.OPERATOR_NAME,
        grid_sampler.CUSTOM_SHADER_DOMAIN_NAME,
        [],
    )
    assert [tuple(v.shape) for v in fake_outputs] == [(1, 4, 8, 32)]
    assert [v.dtype for v in fake_outputs] == [torch.int8]


@pytest.mark.parametrize("per_channel", [False, True])
def test_phase_major_convolution_preserves_pixel_shuffle_values(per_channel):
    program, phase, concat, _, _ = _tail_program()
    producer = phase.args[0]
    original_nodes = producer.args[1:3]
    source, old_weight, old_bias = (
        pack._constant_tensor(program, n) for n in producer.args[:3]
    )
    if not per_channel:
        phase.args = (*phase.args[:2], [0.125], *phase.args[3:])
    scales = phase.args[2]
    before = (
        F.conv_transpose2d(
            source.permute(0, 3, 1, 2).double(),
            old_weight.permute(3, 0, 1, 2).double(),
            old_bias.double(),
        )
        .permute(0, 2, 3, 1)
        .to(torch.int32)
    )
    with patch.object(pack, "build_packed_warp_payload", return_value={}):
        assert pack.PackRifeWarpsAndConcatPass(program)(program.graph_module).modified
    new_weight, new_bias = (
        pack._constant_tensor(program, n) for n in producer.args[1:3]
    )
    after = (
        F.conv_transpose2d(
            source.permute(0, 3, 1, 2).double(),
            new_weight.permute(3, 0, 1, 2).double(),
            new_bias.double(),
        )
        .permute(0, 2, 3, 1)
        .to(torch.int32)
    )
    with TosaLoweringContext(TosaSpecification.create_from_string("TOSA-1.0+INT")):
        original = TOSA.RESCALE.default(before, torch.int8, scales, 0, -14)
        reordered = TOSA.RESCALE.default(after, *phase.args[1:])
    shuffled = F.pixel_shuffle(original.permute(0, 3, 1, 2), 2).permute(0, 2, 3, 1)
    actual = torch.empty_like(shuffled)
    for y in range(actual.shape[1]):
        for x in range(actual.shape[2]):
            offset = ((y % 2) * 2 + x % 2) * 13
            actual[0, y, x] = reordered[0, y // 2, x // 2, offset : offset + 13]
    assert torch.equal(actual, shuffled)
    assert torch.equal(pack._constant_tensor(program, original_nodes[0]), old_weight)
    assert torch.equal(pack._constant_tensor(program, original_nodes[1]), old_bias)


@pytest.mark.parametrize("shared", ["phase", "convolution"])
def test_shared_block0_output_is_not_reordered(shared):
    program, phase, concat, _, residual = _tail_program()
    extra = phase if shared == "phase" else phase.args[0]
    conv = program.graph.output_node().args[0][0]
    program.graph.output_node().args = ((conv, residual, extra),)
    original_args = phase.args[0].args
    with patch.object(pack, "build_packed_warp_payload", return_value={}) as build:
        assert pack.PackRifeWarpsAndConcatPass(program)(program.graph_module).modified
    assert not build.call_args.kwargs
    assert phase.args[0].args == original_args
    assert len(concat.args[0].args[0]) == 7


@pytest.mark.parametrize("unsupported", ["phase", "resize", "rounding", "shared"])
def test_unsupported_tail_keeps_original_state_preparation(unsupported):
    program, phase, concat, customs, residual = _tail_program()
    if unsupported == "phase":
        transpose = next(
            n for n in program.graph.nodes if n.target == TOSA.TRANSPOSE.default
        )
        transpose.args = (transpose.args[0], [0, 3, 1, 5, 2, 4])
    elif unsupported == "resize":
        resize = next(n for n in program.graph.nodes if n.target == TOSA.RESIZE.default)
        resize.kwargs = {"resize_mode": "nearest"}
    elif unsupported == "rounding":
        rescale = concat.args[0][5]
        rescale.args = (*rescale.args[:2], [0.125], *rescale.args[3:])
    else:
        program.graph.output_node().args = ((residual, concat.args[0][5]),)
    with pytest.raises(ValueError):
        pack._match_block0_tail(concat, customs)


def test_original_pixel_shuffle_and_integer_resize_arithmetic():
    phase = (
        torch.arange(1 * 3 * 4 * 52)
        .remainder(256)
        .sub(128)
        .to(torch.int8)
        .reshape(1, 3, 4, 52)
    )
    shuffled = F.pixel_shuffle(phase.permute(0, 3, 1, 2), 2).permute(0, 2, 3, 1)
    packed = phase.reshape(1, 3, 4, 13, 4).transpose(-1, -2).reshape_as(phase)
    interpolated = F.interpolate(
        shuffled.permute(0, 3, 1, 2).float(),
        scale_factor=2,
        mode="bilinear",
        align_corners=False,
    ).permute(0, 2, 3, 1)
    with TosaLoweringContext(TosaSpecification.create_from_string("TOSA-1.0+INT")):
        reference = TOSA.RESCALE.default(
            (interpolated * 16).to(torch.int32),
            torch.int8,
            [0.0625],
            0,
            0,
        )
    actual = torch.empty_like(reference)
    for y in range(actual.shape[1]):
        for x in range(actual.shape[2]):
            ly, lx = (2 * y - 1) // 4, (2 * x - 1) // 4
            dy, dx = 2 * y - 1 - 4 * ly, 2 * x - 1 - 4 * lx
            acc = torch.zeros(13, dtype=torch.int32)
            for cy, cx, weight in (
                (ly, lx, (4 - dy) * (4 - dx)),
                (ly, lx + 1, (4 - dy) * dx),
                (ly + 1, lx, dy * (4 - dx)),
                (ly + 1, lx + 1, dy * dx),
            ):
                cy, cx = (
                    min(max(cy, 0), shuffled.shape[1] - 1),
                    min(max(cx, 0), shuffled.shape[2] - 1),
                )
                offset = ((cy % 2) * 2 + cx % 2) * 13
                acc += (
                    packed[0, cy // 2, cx // 2, offset : offset + 13].to(torch.int32)
                    * weight
                )
            actual[0, y, x] = ((acc + 8) >> 4).clamp(-128, 127).to(torch.int8)
    assert torch.equal(actual, reference)
