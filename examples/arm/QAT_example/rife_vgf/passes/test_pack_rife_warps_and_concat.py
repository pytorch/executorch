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
from executorch.backends.arm.vgf.shaders import grid_sampler
from executorch.backends.transforms.utils import create_constant_placeholder
from torch.export.graph_signature import InputKind

TOSA = pack.TOSA


class _Inputs(torch.nn.Module):
    def forward(self, *inputs):
        return inputs[0]


def _program(code=45):
    tensors = [torch.zeros(1, 4, 6, c, dtype=torch.int8) for c in (4, 4, 4, 4, 1, 8, 4)]
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

    flow = node(TOSA.TRANSPOSE.default, (inputs[-1], [0, 3, 1, 2]), (1, 4, 4, 6))
    outputs = []
    for image, offset in zip(inputs[:4], (0, 2, 0, 2)):
        with patch.object(
            grid_sampler,
            "_compile_flow_offset_grid_sampler_shader",
            return_value="unused",
        ):
            payload = grid_sampler.build_flow_offset_grid_sampler_payload(
                input_shape=(1, 4, 4, 6),
                output_shape=(1, 4, 4, 6),
                flow_shape=(1, 4, 4, 6),
                flow_scale=0.01,
                flow_zero_point=-14,
                input_scale=0.003,
                input_zero_point=-92,
                output_scale=0.01,
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
        output = node(operator.getitem, (custom, 0), (1, 4, 6, 4))
        if len(outputs) < 2:
            output = node(
                TOSA.SLICE.default, (output, [0, 0, 0, 0], [1, 4, 6, 3]), (1, 4, 6, 3)
            )
        outputs.append(output)
    timestep = constant("timestep", torch.full((1, 4, 6, 1), code, dtype=torch.int8))
    concat = node(
        TOSA.CONCAT.default, ([*outputs, timestep, *inputs[4:]],), (1, 4, 6, 28), axis=3
    )
    weight = constant(
        "weight",
        torch.arange(84).remainder(11).sub(5).to(torch.int8).reshape(3, 1, 1, 28),
    )
    bias = constant("bias", torch.tensor([-17, 9, 23], dtype=torch.int32))
    conv = node(
        TOSA.CONV2D.default,
        (concat, weight, bias, [1, 1], [0, 0, 0, 0], [1, 1]),
        (1, 4, 6, 3),
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
    elif unsupported in ("qparams", "sampler", "offset"):
        payload = json.loads(bytes(custom.kwargs["implementation_attrs"]))
        if unsupported == "qparams":
            payload["output_scale"] *= 2
        elif unsupported == "sampler":
            payload["input_0_sampler"] = {}
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


def test_payload_binding_contract_and_source_specialization():
    program, _, _, _, _ = _program()
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
    for i in range(7):
        assert payload[f"input_{i}_binding"] == i
        assert payload[f"input_{i}_type"] == ("Image" if i < 4 else "Tensor")
    assert payload["output_0_binding"] == 7
    assert payload["output_0_type"] == "Buffer"
    assert payload["workgroup_sizes"] == [1, 1, 1]
