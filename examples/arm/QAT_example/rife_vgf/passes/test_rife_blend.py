# Copyright 2026 Arm Limited and/or its affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import operator
from unittest.mock import patch

import pytest
import torch
from examples.arm.QAT_example.rife_vgf.passes import fuse_rife_final_warps as fuse
from examples.arm.QAT_example.rife_vgf.passes.rife_blend import (
    _apply_rescale,
    match_rife_blend,
)
from examples.arm.QAT_example.rife_vgf.passes.test_fuse_rife_final_warps import _program
from examples.arm.QAT_example.rife_vgf.shaders import final_warps
from executorch.backends.arm._passes.arm_pass_manager import (
    _ExportedProgramGraphPassAdapter,
)
from executorch.backends.arm.tosa.specification import (
    TosaLoweringContext,
    TosaSpecification,
)
from executorch.backends.transforms.utils import create_constant_placeholder
from torch.export.graph_signature import (
    InputKind,
    OutputKind,
    OutputSpec,
    TensorArgument,
)

TOSA = fuse.TOSA


def _blend_program(height=4, width=6):
    program, packed, phase, previous, delta, flow, sigmoid, old_blend = _program(
        height, width
    )
    graph = program.graph
    warps = list(graph.output_node().args[0][:2])
    graph.erase_node(graph.output_node())
    graph.erase_node(old_blend)
    image_shape = (1, 3, height, width)
    mask_shape = (1, 1, height, width)

    def node(target, args, shape, dtype=torch.int32):
        n = graph.call_function(target, args)
        n.meta["val"] = packed.meta["val"].new_empty(shape, dtype=dtype)
        return n

    def rescale(source, dtype, scale, zp, out_zp=0):
        return node(
            TOSA.RESCALE.default,
            (source, dtype, [scale], zp, out_zp),
            tuple(source.meta["val"].shape),
            dtype,
        )

    with graph.inserting_before(next(iter(graph.nodes))):
        one = create_constant_placeholder(
            program,
            graph,
            "one",
            InputKind.CONSTANT_TENSOR,
            torch.tensor([[[[127]]]], dtype=torch.int8),
        )
    scalar = rescale(one, torch.int32, 524288.0, -127)
    up_mask = rescale(sigmoid, torch.int32, 523948.54785029794, -127)
    subtract = node(TOSA.SUB.default, (scalar, up_mask), mask_shape)
    complement = rescale(subtract, torch.int8, 1.9079762253925467e-6, 0, -127)
    sides = []
    centered_masks = []
    for warp, mask, multiply_scale, add_scale in zip(
        warps,
        (sigmoid, complement),
        (0.003937005347410795, 0.0039439108250957705),
        (524288.0, 523729.7373878841),
    ):
        transpose = node(
            TOSA.TRANSPOSE.default,
            (warp, [0, 3, 1, 2]),
            (1, 4, height, width),
            torch.int8,
        )
        crop = node(
            TOSA.SLICE.default,
            (transpose, [0, 0, 0, 0], list(image_shape)),
            image_shape,
            torch.int8,
        )
        image = rescale(crop, torch.int32, 1.0, -14)
        centered_mask = rescale(mask, torch.int32, 1.0, -127)
        centered_masks.append(centered_mask)
        multiply = node(TOSA.MUL.default, (image, centered_mask), image_shape)
        weighted = rescale(multiply, torch.int8, multiply_scale, 0, -127)
        sides.append(rescale(weighted, torch.int32, add_scale, -127))
    add = node(TOSA.ADD.default, tuple(sides), image_shape)
    result = rescale(add, torch.int8, 1.9043485036435921e-6, 0, -127)
    graph.output((result,))
    program.graph_signature.output_specs[:] = [
        OutputSpec(OutputKind.USER_OUTPUT, TensorArgument(name=result.name), None)
    ]
    return program, warps, sigmoid, complement, centered_masks, result


@pytest.mark.parametrize("adapted", [False, True])
def test_fuses_blend_and_writes_only_final_rgb(adapted):
    program, warps, sigmoid, _, _, result = _blend_program()
    expected = match_rife_blend(program, warps, sigmoid)
    with patch.object(fuse, "build_final_warps_payload", return_value={}) as build:
        outcome = (
            _ExportedProgramGraphPassAdapter(fuse.FuseRifeFinalWarpsPass()).call(
                program
            )
            if adapted
            else fuse.FuseRifeFinalWarpsPass(program)(program.graph_module)
        )
    assert outcome.modified
    assert result.target == operator.getitem and result.args[1] == 0
    custom = result.args[0]
    assert custom.kwargs["operator_name"] == final_warps.BLEND_OPERATOR_NAME
    assert len(custom.meta["val"]) == 1
    assert tuple(custom.meta["val"][0].shape) == (1, 3, 4, 6)
    assert build.call_args.kwargs["blend"].rescales == expected.rescales
    assert not any(
        n.target
        in (
            TOSA.MUL.default,
            TOSA.ADD.default,
            TOSA.SUB.default,
            TOSA.TABLE.default,
            TOSA.SLICE.default,
            TOSA.TRANSPOSE.default,
        )
        for n in program.graph.nodes
    )
    program.graph.lint()


@pytest.mark.parametrize(
    "unsupported", ["shared_mask", "shifted_mul", "scaled_mask", "unsigned", "overflow"]
)
def test_keeps_unsupported_blend_in_tosa(unsupported):
    program, warps, sigmoid, _, masks, result = _blend_program()
    if unsupported == "shared_mask":
        with program.graph.inserting_before(program.graph.output_node()):
            program.graph.call_function(
                TOSA.RESCALE.default, (sigmoid, torch.int32, [1.0], -127, 0)
            )
    elif unsupported == "shifted_mul":
        next(n for n in program.graph.nodes if n.target == TOSA.MUL.default).kwargs = {
            "shift": 1
        }
    elif unsupported == "scaled_mask":
        masks[0].args = (*masks[0].args[:2], [2.0], *masks[0].args[3:])
    elif unsupported == "unsigned":
        result.kwargs = {"output_unsigned": True}
    else:
        left, right = result.args[0].args
        left.args = (*left.args[:2], [8388608.0], *left.args[3:])
        right.args = (*right.args[:2], [8388608.0], *right.args[3:])
    with pytest.raises(ValueError):
        match_rife_blend(program, warps, sigmoid)
    with patch.object(fuse, "build_final_warps_payload", return_value={}) as build:
        assert fuse.FuseRifeFinalWarpsPass(program)(program.graph_module).modified
    assert build.call_args.kwargs["blend"] is None
    assert result.target == TOSA.RESCALE.default


def test_integer_blend_matches_tosa_for_all_image_and_mask_codes():
    program, warps, sigmoid, complement_node, _, _ = _blend_program()
    blend = match_rife_blend(program, warps, sigmoid)
    constant_rescale, mask_rescale = complement_node.args[0].args
    lookup = fuse._constant_tensor(program, sigmoid.args[1])
    constant = fuse._constant_tensor(program, constant_rescale.args[0])
    mul0, mul1, add0, add1, output = blend.rescales
    codes = torch.arange(-128, 128, dtype=torch.int32)
    lookup = lookup.reshape(256, 1)
    images = [
        codes.reshape(1, 256).expand(256, 256),
        codes.flip(0).reshape(1, 256).expand(256, 256),
    ]
    with TosaLoweringContext(TosaSpecification.create_from_string("TOSA-1.0+INT")):

        def apply(x, n):
            return TOSA.RESCALE.default(x, *n.args[1:], **n.kwargs)

        complement = apply(
            apply(constant.reshape(()), constant_rescale) - apply(lookup, mask_rescale),
            complement_node,
        )
        expected_left = apply((images[0].int() + 14) * (lookup.int() + 127), mul0)
        expected_right = apply((images[1].int() + 14) * (complement.int() + 127), mul1)
        expected = apply(
            apply(expected_left, add0) + apply(expected_right, add1), output
        )
    assert torch.equal(complement.reshape(-1), blend.complement)

    left = _apply_rescale((images[0] + 14) * (lookup.int() + 127), mul0)
    right = _apply_rescale(
        (images[1] + 14) * (blend.complement.int().reshape(256, 1) + 127), mul1
    )
    actual = _apply_rescale(
        _apply_rescale(left, add0) + _apply_rescale(right, add1), output
    )
    assert torch.equal(actual, expected.to(torch.int64))


def test_blend_payload_and_fake_output_contract():
    program, warps, sigmoid, _, _, _ = _blend_program()
    _, payloads, rescales, _, _, _, table, _, _ = fuse.FuseRifeFinalWarpsPass(
        program
    )._match(program.graph_module)
    blend = match_rife_blend(program, warps, sigmoid)
    with patch.object(
        final_warps, "_compile_shader_source", return_value="compiled"
    ) as compile_shader:
        payload = final_warps.build_final_warps_payload(
            payloads, rescales, table, blend=blend
        )
    source = compile_shader.call_args.args[0]
    assert "@" not in source
    assert "#define FUSE_BLEND 1" in source
    assert "rescaleOutput(rescaleAdd0(left) + rescaleAdd1(right))" in source
    assert "tensorWriteARM(output0" in source
    assert source.count("imulExtended") == 8
    assert "int(roundEven(float(value -" not in source
    assert payload["output_0_type"] == "Tensor"
    assert payload["output_0_vkformat"] == "VK_FORMAT_R8_SINT"
    assert not any(key.startswith(("output_1_", "output_2_")) for key in payload)
    image = torch.empty(1, 8, 12, 4, dtype=torch.int8)
    values = fuse._final_warps_fake(
        [image, image], final_warps.BLEND_OPERATOR_NAME, None, None
    )
    assert len(values) == 1 and tuple(values[0].shape) == (1, 3, 8, 12)
