# Copyright 2026 Arm Limited and/or its affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import base64
from importlib.resources import files

import pytest
import torch
from executorch.backends.arm.vgf.shaders.grid_sampler import (
    build_grid_sampler_2d_payload,
    decode_payload,
    encode_payload,
    GRID_SAMPLER_2D_SAMPLER_ALIGN_CORNERS_SHADER_BINARY,
    GRID_SAMPLER_2D_SAMPLER_ALIGN_CORNERS_SHADER_SOURCE,
    GRID_SAMPLER_2D_SAMPLER_INT8_ALIGN_CORNERS_SHADER_BINARY,
    GRID_SAMPLER_2D_SAMPLER_INT8_ALIGN_CORNERS_SHADER_SOURCE,
    GRID_SAMPLER_2D_SAMPLER_INT8_SHADER_BINARY,
    GRID_SAMPLER_2D_SAMPLER_INT8_SHADER_SOURCE,
    GRID_SAMPLER_2D_SAMPLER_INT8_VK_FORMAT,
    GRID_SAMPLER_2D_SAMPLER_SHADER_BINARY,
    GRID_SAMPLER_2D_SAMPLER_SHADER_SOURCE,
    GRID_SAMPLER_2D_SAMPLER_VK_FORMAT,
    GRID_SAMPLER_2D_SHADER_BINARY,
    GRID_SAMPLER_2D_SHADER_ENTRY_POINT,
    GRID_SAMPLER_2D_SHADER_LANGUAGE,
    GRID_SAMPLER_2D_SHADER_SOURCE,
    GRID_SAMPLER_2D_VK_FORMAT,
    GRID_SAMPLER_2D_WORKGROUP_SIZES,
)


def _shader_code_from_resource(shader_file: str) -> str:
    return "".join(
        files("executorch.backends.arm.vgf.shaders")
        .joinpath(shader_file)
        .read_text(encoding="utf-8")
        .split()
    )


def test_grid_sampler_2d_custom_shader_payload_no_target_round_trip():
    payload = build_grid_sampler_2d_payload(
        interpolation_mode=0,
        padding_mode=2,
        align_corners=True,
    )
    decoded = decode_payload(encode_payload(payload))

    assert decoded["entry_point"] == GRID_SAMPLER_2D_SHADER_ENTRY_POINT
    assert decoded["workgroup_sizes"] == GRID_SAMPLER_2D_WORKGROUP_SIZES
    assert decoded["shader_language"] == GRID_SAMPLER_2D_SHADER_LANGUAGE
    assert base64.b64decode(decoded["shader_code"])[:4] == b"\x03\x02\x23\x07"
    assert decoded["input_0_type"] == "Tensor"
    assert decoded["input_0_vkformat"] == GRID_SAMPLER_2D_VK_FORMAT
    assert decoded["input_0_vkdescriptortype"] == "VK_DESCRIPTOR_TYPE_STORAGE_BUFFER"
    assert decoded["input_0_binding"] == 0
    assert decoded["input_1_type"] == "Tensor"
    assert decoded["input_1_vkformat"] == GRID_SAMPLER_2D_VK_FORMAT
    assert decoded["input_1_vkdescriptortype"] == "VK_DESCRIPTOR_TYPE_STORAGE_BUFFER"
    assert decoded["input_1_binding"] == 1
    assert decoded["output_0_type"] == "Tensor"
    assert decoded["output_0_vkformat"] == GRID_SAMPLER_2D_VK_FORMAT
    assert decoded["output_0_vkdescriptortype"] == "VK_DESCRIPTOR_TYPE_STORAGE_BUFFER"
    assert decoded["output_0_binding"] == 2


def test_grid_sampler_2d_custom_shader_payload_no_target_uses_sampler_for_c4():
    payload = build_grid_sampler_2d_payload(
        interpolation_mode=0,
        padding_mode=0,
        align_corners=False,
        input_shape=(1, 4, 8, 8),
        input_dtype=torch.float32,
    )

    assert payload["shader_language"] == GRID_SAMPLER_2D_SHADER_LANGUAGE
    assert base64.b64decode(payload["shader_code"])[:4] == b"\x03\x02\x23\x07"
    assert payload["input_0_type"] == "Image"
    assert payload["input_0_vkformat"] == GRID_SAMPLER_2D_SAMPLER_VK_FORMAT
    assert (
        payload["input_0_vkdescriptortype"]
        == "VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER"
    )
    assert payload["input_1_type"] == "Tensor"
    assert payload["input_1_vkformat"] == GRID_SAMPLER_2D_VK_FORMAT
    assert payload["input_1_vkdescriptortype"] == "VK_DESCRIPTOR_TYPE_TENSOR_ARM"
    assert payload["output_0_type"] == "Image"
    assert payload["output_0_vkformat"] == GRID_SAMPLER_2D_SAMPLER_VK_FORMAT
    assert payload["output_0_vkdescriptortype"] == "VK_DESCRIPTOR_TYPE_STORAGE_IMAGE"
    assert payload["input_0_sampler"] == {
        "address_mode_u": "VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_BORDER",
        "address_mode_v": "VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_BORDER",
        "border_color": "VK_BORDER_COLOR_FLOAT_TRANSPARENT_BLACK",
        "mag_filter": "VK_FILTER_LINEAR",
        "min_filter": "VK_FILTER_LINEAR",
    }


def test_grid_sampler_2d_custom_shader_payload_no_target_uses_int8_sampler_for_c4():
    payload = build_grid_sampler_2d_payload(
        interpolation_mode=0,
        padding_mode=0,
        align_corners=False,
        input_shape=(1, 4, 8, 8),
        input_dtype=torch.int8,
        output_dtype=torch.int8,
    )

    assert payload["shader_language"] == GRID_SAMPLER_2D_SHADER_LANGUAGE
    assert base64.b64decode(payload["shader_code"])[:4] == b"\x03\x02\x23\x07"
    assert payload["input_0_type"] == "Image"
    assert payload["input_0_vkformat"] == GRID_SAMPLER_2D_SAMPLER_INT8_VK_FORMAT
    assert (
        payload["input_0_vkdescriptortype"]
        == "VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER"
    )
    assert payload["input_1_type"] == "Tensor"
    assert payload["input_1_vkformat"] == GRID_SAMPLER_2D_VK_FORMAT
    assert payload["input_1_vkdescriptortype"] == "VK_DESCRIPTOR_TYPE_TENSOR_ARM"
    assert payload["output_0_type"] == "Image"
    assert payload["output_0_vkformat"] == GRID_SAMPLER_2D_SAMPLER_INT8_VK_FORMAT
    assert payload["output_0_vkdescriptortype"] == "VK_DESCRIPTOR_TYPE_STORAGE_IMAGE"


def test_grid_sampler_2d_custom_shader_payload_no_target_keeps_c3_on_buffer():
    payload = build_grid_sampler_2d_payload(
        interpolation_mode=0,
        padding_mode=0,
        align_corners=False,
        input_shape=(1, 3, 8, 8),
        input_dtype=torch.float32,
    )

    assert payload["shader_language"] == GRID_SAMPLER_2D_SHADER_LANGUAGE
    assert payload["input_0_type"] == "Tensor"
    assert payload["input_0_vkdescriptortype"] == "VK_DESCRIPTOR_TYPE_STORAGE_BUFFER"
    assert payload["output_0_type"] == "Tensor"
    assert payload["output_0_vkdescriptortype"] == "VK_DESCRIPTOR_TYPE_STORAGE_BUFFER"
    assert "input_0_sampler" not in payload


def test_grid_sampler_2d_custom_shader_payload_no_target_align_corners_sampler():
    payload = build_grid_sampler_2d_payload(
        interpolation_mode=0,
        padding_mode=0,
        align_corners=True,
        input_shape=(1, 4, 8, 8),
        input_dtype=torch.float32,
    )

    assert payload["input_0_type"] == "Image"
    assert payload["input_0_vkformat"] == GRID_SAMPLER_2D_SAMPLER_VK_FORMAT
    assert (
        payload["input_0_vkdescriptortype"]
        == "VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER"
    )
    assert payload["output_0_type"] == "Image"
    assert payload["output_0_vkformat"] == GRID_SAMPLER_2D_SAMPLER_VK_FORMAT
    assert payload["output_0_vkdescriptortype"] == "VK_DESCRIPTOR_TYPE_STORAGE_IMAGE"
    assert payload["shader_code"] == _shader_code_from_resource(
        GRID_SAMPLER_2D_SAMPLER_ALIGN_CORNERS_SHADER_BINARY
    )


def test_grid_sampler_2d_custom_shader_payload_no_target_int8_align_corners_sampler():
    payload = build_grid_sampler_2d_payload(
        interpolation_mode=0,
        padding_mode=0,
        align_corners=True,
        input_shape=(1, 4, 8, 8),
        input_dtype=torch.int8,
        output_dtype=torch.int8,
    )

    assert payload["input_0_type"] == "Image"
    assert payload["input_0_vkformat"] == GRID_SAMPLER_2D_SAMPLER_INT8_VK_FORMAT
    assert (
        payload["input_0_vkdescriptortype"]
        == "VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER"
    )
    assert payload["output_0_type"] == "Image"
    assert payload["output_0_vkformat"] == GRID_SAMPLER_2D_SAMPLER_INT8_VK_FORMAT
    assert payload["output_0_vkdescriptortype"] == "VK_DESCRIPTOR_TYPE_STORAGE_IMAGE"
    assert payload["shader_code"] == _shader_code_from_resource(
        GRID_SAMPLER_2D_SAMPLER_INT8_ALIGN_CORNERS_SHADER_BINARY
    )


def test_grid_sampler_2d_custom_shader_payload_no_target_bicubic_buffer():
    payload = build_grid_sampler_2d_payload(
        interpolation_mode=2,
        padding_mode=0,
        align_corners=False,
        input_shape=(1, 4, 8, 8),
        input_dtype=torch.float32,
    )

    assert payload["input_0_type"] == "Tensor"
    assert payload["input_0_vkdescriptortype"] == "VK_DESCRIPTOR_TYPE_STORAGE_BUFFER"
    assert payload["output_0_type"] == "Tensor"
    assert payload["output_0_vkdescriptortype"] == "VK_DESCRIPTOR_TYPE_STORAGE_BUFFER"
    assert "input_0_sampler" not in payload


def test_grid_sampler_2d_custom_shader_payload_no_target_uses_spirv():
    payload = build_grid_sampler_2d_payload(
        interpolation_mode=0,
        padding_mode=0,
        align_corners=False,
    )

    shader_binary = base64.b64decode(payload["shader_code"])

    assert payload["shader_language"] == "SPIR-V"
    assert shader_binary[:4] == b"\x03\x02\x23\x07"


def test_grid_sampler_2d_custom_shader_payload_no_target_has_shader_resources():
    assert GRID_SAMPLER_2D_SHADER_SOURCE == "grid_sampler.glsl"
    assert GRID_SAMPLER_2D_SHADER_BINARY == "grid_sampler.spirv.b64"
    assert GRID_SAMPLER_2D_SAMPLER_SHADER_SOURCE == "grid_sampler_sampler.glsl"
    assert GRID_SAMPLER_2D_SAMPLER_SHADER_BINARY == "grid_sampler_sampler.spirv.b64"
    assert (
        GRID_SAMPLER_2D_SAMPLER_ALIGN_CORNERS_SHADER_SOURCE
        == "grid_sampler_sampler_align_corners.glsl"
    )
    assert (
        GRID_SAMPLER_2D_SAMPLER_ALIGN_CORNERS_SHADER_BINARY
        == "grid_sampler_sampler_align_corners.spirv.b64"
    )
    assert (
        GRID_SAMPLER_2D_SAMPLER_INT8_SHADER_SOURCE == "grid_sampler_sampler_int8.glsl"
    )
    assert (
        GRID_SAMPLER_2D_SAMPLER_INT8_SHADER_BINARY
        == "grid_sampler_sampler_int8.spirv.b64"
    )
    assert (
        GRID_SAMPLER_2D_SAMPLER_INT8_ALIGN_CORNERS_SHADER_SOURCE
        == "grid_sampler_sampler_int8_align_corners.glsl"
    )
    assert (
        GRID_SAMPLER_2D_SAMPLER_INT8_ALIGN_CORNERS_SHADER_BINARY
        == "grid_sampler_sampler_int8_align_corners.spirv.b64"
    )


def test_grid_sampler_2d_custom_shader_payload_no_target_rejects_bad_modes():
    with pytest.raises(ValueError, match="Unsupported interpolation_mode"):
        build_grid_sampler_2d_payload(
            interpolation_mode=99,
            padding_mode=0,
            align_corners=False,
        )

    with pytest.raises(ValueError, match="Unsupported padding_mode"):
        build_grid_sampler_2d_payload(
            interpolation_mode=0,
            padding_mode=99,
            align_corners=False,
        )
