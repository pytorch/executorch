# Copyright 2026 Arm Limited and/or its affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import json
from importlib.resources import files
from typing import Any

CUSTOM_SHADER_DOMAIN_NAME = "com.arm.VulkanCustomShader"
GRID_SAMPLER_2D_OPERATOR_NAME = "torch.nn.functional.grid_sample"
GRID_SAMPLER_2D_WORKGROUP_SIZES = [8, 8, 1]
GRID_SAMPLER_2D_SHADER_ENTRY_POINT = "main"
GRID_SAMPLER_2D_SHADER_LANGUAGE = "SPIR-V"
GRID_SAMPLER_2D_VK_FORMAT = "VK_FORMAT_R32_SFLOAT"
GRID_SAMPLER_2D_SHADER_SOURCE = "grid_sampler.glsl"
GRID_SAMPLER_2D_SHADER_BINARY = "grid_sampler.spirv.b64"
GRID_SAMPLER_2D_SAMPLER_SHADER_SOURCE = "grid_sampler_sampler.glsl"
GRID_SAMPLER_2D_SAMPLER_SHADER_BINARY = "grid_sampler_sampler.spirv.b64"
GRID_SAMPLER_2D_SAMPLER_ALIGN_CORNERS_SHADER_SOURCE = (
    "grid_sampler_sampler_align_corners.glsl"
)
GRID_SAMPLER_2D_SAMPLER_ALIGN_CORNERS_SHADER_BINARY = (
    "grid_sampler_sampler_align_corners.spirv.b64"
)
GRID_SAMPLER_2D_SAMPLER_INT8_SHADER_SOURCE = "grid_sampler_sampler_int8.glsl"
GRID_SAMPLER_2D_SAMPLER_INT8_SHADER_BINARY = "grid_sampler_sampler_int8.spirv.b64"
GRID_SAMPLER_2D_SAMPLER_INT8_ALIGN_CORNERS_SHADER_SOURCE = (
    "grid_sampler_sampler_int8_align_corners.glsl"
)
GRID_SAMPLER_2D_SAMPLER_INT8_ALIGN_CORNERS_SHADER_BINARY = (
    "grid_sampler_sampler_int8_align_corners.spirv.b64"
)
GRID_SAMPLER_2D_SAMPLER_VK_FORMAT = "VK_FORMAT_R32G32B32A32_SFLOAT"
GRID_SAMPLER_2D_SAMPLER_INT8_VK_FORMAT = "VK_FORMAT_R8G8B8A8_SNORM"

_INTERPOLATION_MODE_NAMES = {
    0: "bilinear",
    1: "nearest",
    2: "bicubic",
}
_PADDING_MODE_NAMES = {
    0: "zeros",
    1: "border",
    2: "reflection",
}


def _mode_name(
    mode: int,
    names: dict[int, str],
    mode_kind: str,
) -> str:
    if mode not in names:
        raise ValueError(
            f"Unsupported {mode_kind} {mode} for {GRID_SAMPLER_2D_OPERATOR_NAME}"
        )
    return names[mode]


def grid_sampler_2d_operator_name(
    interpolation_mode: int,
    padding_mode: int,
    align_corners: bool,
) -> str:
    interpolation = _mode_name(
        int(interpolation_mode),
        _INTERPOLATION_MODE_NAMES,
        "interpolation_mode",
    )
    padding = _mode_name(
        int(padding_mode),
        _PADDING_MODE_NAMES,
        "padding_mode",
    )
    return (
        f"{GRID_SAMPLER_2D_OPERATOR_NAME}"
        f".mode.{interpolation}"
        f".padding.{padding}"
        f".align_corners.{align_corners}"
    )


def build_grid_sampler_2d_payload(
    interpolation_mode: int,
    padding_mode: int,
    align_corners: bool,
    input_shape: tuple[int, ...] | None = None,
    input_dtype: Any | None = None,
    output_dtype: Any | None = None,
) -> dict[str, Any]:
    _mode_name(
        int(interpolation_mode),
        _INTERPOLATION_MODE_NAMES,
        "interpolation_mode",
    )
    _mode_name(
        int(padding_mode),
        _PADDING_MODE_NAMES,
        "padding_mode",
    )
    if output_dtype is None:
        output_dtype = input_dtype

    sampler_vk_format = _sampler_vk_format(input_dtype, output_dtype)
    use_sampler = (
        input_shape is not None
        and len(input_shape) == 4
        and int(input_shape[0]) == 1
        and int(input_shape[1]) == 4
        and sampler_vk_format is not None
        and int(interpolation_mode) in (0, 1)
    )
    shader_file = (
        _sampler_shader_file(sampler_vk_format, align_corners=align_corners)
        if use_sampler
        else GRID_SAMPLER_2D_SHADER_BINARY
    )
    shader_code = "".join(
        files(__package__).joinpath(shader_file).read_text(encoding="utf-8").split()
    )

    payload = {
        "entry_point": GRID_SAMPLER_2D_SHADER_ENTRY_POINT,
        "workgroup_sizes": GRID_SAMPLER_2D_WORKGROUP_SIZES,
        "shader_language": GRID_SAMPLER_2D_SHADER_LANGUAGE,
        "shader_code": shader_code,
        "input_0_binding": 0,
        "input_0_descriptorset": 0,
        "input_1_type": "Tensor",
        "input_1_vkformat": GRID_SAMPLER_2D_VK_FORMAT,
        "input_1_binding": 1,
        "input_1_descriptorset": 0,
        "output_0_binding": 2,
        "output_0_descriptorset": 0,
    }
    if use_sampler:
        payload.update(
            {
                "input_0_type": "Image",
                "input_0_vkformat": sampler_vk_format,
                "input_0_vkdescriptortype": (
                    "VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER"
                ),
                "input_0_sampler": _sampler_config(
                    interpolation_mode=interpolation_mode,
                    padding_mode=padding_mode,
                ),
                "input_1_vkdescriptortype": "VK_DESCRIPTOR_TYPE_TENSOR_ARM",
                "output_0_type": "Image",
                "output_0_vkformat": sampler_vk_format,
                "output_0_vkdescriptortype": "VK_DESCRIPTOR_TYPE_STORAGE_IMAGE",
            }
        )
    else:
        payload.update(
            {
                "input_0_type": "Tensor",
                "input_0_vkformat": GRID_SAMPLER_2D_VK_FORMAT,
                "input_0_vkdescriptortype": "VK_DESCRIPTOR_TYPE_STORAGE_BUFFER",
                "input_1_vkdescriptortype": "VK_DESCRIPTOR_TYPE_STORAGE_BUFFER",
                "output_0_type": "Tensor",
                "output_0_vkformat": GRID_SAMPLER_2D_VK_FORMAT,
                "output_0_vkdescriptortype": "VK_DESCRIPTOR_TYPE_STORAGE_BUFFER",
            }
        )
    return payload


def _sampler_vk_format(input_dtype: Any | None, output_dtype: Any | None) -> str | None:
    if str(input_dtype) != str(output_dtype):
        return None
    if str(input_dtype) == "torch.float32":
        return GRID_SAMPLER_2D_SAMPLER_VK_FORMAT
    if str(input_dtype) == "torch.int8":
        return GRID_SAMPLER_2D_SAMPLER_INT8_VK_FORMAT
    return None


def _sampler_shader_file(
    sampler_vk_format: str | None,
    align_corners: bool,
) -> str:
    if sampler_vk_format == GRID_SAMPLER_2D_SAMPLER_INT8_VK_FORMAT:
        if align_corners:
            return GRID_SAMPLER_2D_SAMPLER_INT8_ALIGN_CORNERS_SHADER_BINARY
        return GRID_SAMPLER_2D_SAMPLER_INT8_SHADER_BINARY
    if align_corners:
        return GRID_SAMPLER_2D_SAMPLER_ALIGN_CORNERS_SHADER_BINARY
    return GRID_SAMPLER_2D_SAMPLER_SHADER_BINARY


def _sampler_config(interpolation_mode: int, padding_mode: int) -> dict[str, str]:
    interpolation = _mode_name(
        int(interpolation_mode),
        _INTERPOLATION_MODE_NAMES,
        "interpolation_mode",
    )
    padding = _mode_name(
        int(padding_mode),
        _PADDING_MODE_NAMES,
        "padding_mode",
    )

    filter_mode = (
        "VK_FILTER_NEAREST" if interpolation == "nearest" else "VK_FILTER_LINEAR"
    )
    if padding == "zeros":
        address_mode = "VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_BORDER"
    elif padding == "border":
        address_mode = "VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE"
    else:
        address_mode = "VK_SAMPLER_ADDRESS_MODE_MIRRORED_REPEAT"

    return {
        "min_filter": filter_mode,
        "mag_filter": filter_mode,
        "address_mode_u": address_mode,
        "address_mode_v": address_mode,
        "border_color": "VK_BORDER_COLOR_FLOAT_TRANSPARENT_BLACK",
    }


def encode_payload(payload: dict[str, Any]) -> list[int]:
    return list(json.dumps(payload, sort_keys=True).encode("utf-8"))


def decode_payload(implementation_attrs: list[int]) -> dict[str, Any]:
    return json.loads(bytes(implementation_attrs).decode("utf-8"))
