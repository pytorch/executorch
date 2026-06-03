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


def build_grid_sampler_2d_payload(
    interpolation_mode: int,
    padding_mode: int,
    align_corners: bool,
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
    shader_code = "".join(
        files(__package__)
        .joinpath(GRID_SAMPLER_2D_SHADER_BINARY)
        .read_text(encoding="utf-8")
        .split()
    )

    return {
        "entry_point": GRID_SAMPLER_2D_SHADER_ENTRY_POINT,
        "workgroup_sizes": GRID_SAMPLER_2D_WORKGROUP_SIZES,
        "shader_language": GRID_SAMPLER_2D_SHADER_LANGUAGE,
        "shader_code": shader_code,
        "input_0_type": "Tensor",
        "input_0_vkformat": GRID_SAMPLER_2D_VK_FORMAT,
        "input_0_vkdescriptortype": "VK_DESCRIPTOR_TYPE_STORAGE_BUFFER",
        "input_0_binding": 0,
        "input_0_descriptorset": 0,
        "input_1_type": "Tensor",
        "input_1_vkformat": GRID_SAMPLER_2D_VK_FORMAT,
        "input_1_vkdescriptortype": "VK_DESCRIPTOR_TYPE_STORAGE_BUFFER",
        "input_1_binding": 1,
        "input_1_descriptorset": 0,
        "output_0_type": "Tensor",
        "output_0_vkformat": GRID_SAMPLER_2D_VK_FORMAT,
        "output_0_vkdescriptortype": "VK_DESCRIPTOR_TYPE_STORAGE_BUFFER",
        "output_0_binding": 2,
        "output_0_descriptorset": 0,
    }


def encode_payload(payload: dict[str, Any]) -> list[int]:
    return list(json.dumps(payload, sort_keys=True).encode("utf-8"))


def decode_payload(implementation_attrs: list[int]) -> dict[str, Any]:
    return json.loads(bytes(implementation_attrs).decode("utf-8"))
