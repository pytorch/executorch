# Copyright 2026 Arm Limited and/or its affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import base64
import json
import shutil
import sys
from pathlib import Path

import pytest
import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[4]))

from backends.arm.test._custom_vgf_test_utils import (
    EncodeSamplerGridSampleToTosaCustomPass,
    register_test_shader_library_ops,
    rewrite_aten_grid_sample_to_test_shader,
)
from executorch.backends.arm.tosa.specification import (
    TosaLoweringContext,
    TosaSpecification,
)
from executorch.backends.arm.vgf.shaders.grid_sampler import (
    build_grid_sampler_2d_payload,
    decode_payload,
    encode_payload,
    GRID_SAMPLER_2D_SHADER_BINARY,
    GRID_SAMPLER_2D_SHADER_ENTRY_POINT,
    GRID_SAMPLER_2D_SHADER_LANGUAGE,
    GRID_SAMPLER_2D_SHADER_SOURCE,
    GRID_SAMPLER_2D_VK_FORMAT,
    GRID_SAMPLER_2D_WORKGROUP_SIZES,
)
from torch.export import export


class _GridSampleModule(torch.nn.Module):
    def __init__(
        self,
        mode: str = "bilinear",
        padding_mode: str = "zeros",
        align_corners: bool = False,
    ) -> None:
        super().__init__()
        self.mode = mode
        self.padding_mode = padding_mode
        self.align_corners = align_corners

    def forward(self, x: torch.Tensor, grid: torch.Tensor) -> torch.Tensor:
        return F.grid_sample(
            x,
            grid,
            mode=self.mode,
            padding_mode=self.padding_mode,
            align_corners=self.align_corners,
        )


def _decode_sampler_payload(
    mode: str | None = None,
    padding_mode: str | None = None,
    align_corners: bool = False,
) -> dict[str, object]:
    if shutil.which("glslc") is None:
        pytest.skip("glslc not found")
    register_test_shader_library_ops()
    module = _GridSampleModule("bilinear", "zeros", align_corners)
    example_inputs = (
        torch.randn(1, 4, 8, 8).contiguous(memory_format=torch.channels_last),
        torch.randn(1, 4, 4, 2),
    )
    exported = export(module, example_inputs)
    graph_module = exported.graph_module
    rewrite_aten_grid_sample_to_test_shader(graph_module)

    for node in graph_module.graph.nodes:
        if "arm_test_vulkan_custom_shader.grid_sample" not in str(node.target):
            continue
        updated_kwargs = dict(node.kwargs)
        if mode is not None:
            updated_kwargs["mode"] = mode
        if padding_mode is not None:
            updated_kwargs["padding_mode"] = padding_mode
        node.kwargs = updated_kwargs

    with TosaLoweringContext(TosaSpecification.create_from_string("TOSA-1.0+FP")):
        EncodeSamplerGridSampleToTosaCustomPass().call(graph_module)

    custom_node = next(
        node
        for node in graph_module.graph.nodes
        if "tosa.CUSTOM.default" in str(node.target)
    )
    return json.loads(bytes(custom_node.kwargs["implementation_attrs"]).decode("utf-8"))


# Covers basic payload encoding and decoding for shader metadata.
# Checks bindings, workgroup sizes, language, and formats are preserved.
def test_buffer_shader_payload_vgf_encodes_bindings_and_formats():
    payload = decode_payload(
        encode_payload(
            build_grid_sampler_2d_payload(
                interpolation_mode=0,
                padding_mode=0,
                align_corners=False,
            )
        )
    )

    assert payload["entry_point"] == GRID_SAMPLER_2D_SHADER_ENTRY_POINT
    assert payload["workgroup_sizes"] == GRID_SAMPLER_2D_WORKGROUP_SIZES
    assert payload["shader_language"] == GRID_SAMPLER_2D_SHADER_LANGUAGE
    assert payload["input_0_binding"] == 0
    assert payload["input_1_binding"] == 1
    assert payload["output_0_binding"] == 2
    assert payload["input_0_vkformat"] == GRID_SAMPLER_2D_VK_FORMAT
    assert payload["input_1_vkformat"] == GRID_SAMPLER_2D_VK_FORMAT
    assert payload["output_0_vkformat"] == GRID_SAMPLER_2D_VK_FORMAT


# Covers sampler-specific payload fields for sampled image inputs.
# Checks filter, address mode, and border color are encoded in the payload.
def test_sampler_shader_payload_vgf_encodes_sampler_fields():
    payload = _decode_sampler_payload()

    assert (
        payload["input_0_vkdescriptortype"]
        == "VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER"
    )
    assert payload["input_1_vkdescriptortype"] == "VK_DESCRIPTOR_TYPE_TENSOR_ARM"
    assert payload["input_1_vkformat"] == "VK_FORMAT_R32_SFLOAT"
    assert payload["output_0_vkdescriptortype"] == "VK_DESCRIPTOR_TYPE_STORAGE_IMAGE"
    assert payload["input_0_sampler"] == {
        "address_mode_u": "VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_BORDER",
        "address_mode_v": "VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_BORDER",
        "border_color": "VK_BORDER_COLOR_FLOAT_TRANSPARENT_BLACK",
        "mag_filter": "VK_FILTER_LINEAR",
        "min_filter": "VK_FILTER_LINEAR",
    }


# Covers the local shader asset contract used by the tests.
# Checks the expected GLSL/SPIR-V asset names and that the SPIR-V bytes look valid.
def test_shader_payload_vgf_uses_expected_glsl_and_spirv_asset():
    buffer_payload = build_grid_sampler_2d_payload(
        interpolation_mode=0,
        padding_mode=0,
        align_corners=False,
    )

    assert GRID_SAMPLER_2D_SHADER_SOURCE == "grid_sampler.glsl"
    assert GRID_SAMPLER_2D_SHADER_BINARY == "grid_sampler.spirv.b64"
    assert buffer_payload["shader_language"] == "SPIR-V"
    assert base64.b64decode(buffer_payload["shader_code"])[:4] == b"\x03\x02\x23\x07"


# Covers validation of unsupported shader option values.
# Checks invalid mode and padding_mode values raise instead of encoding silently.
def test_shader_payload_vgf_rejects_invalid_mode_values():
    with pytest.raises(RuntimeError, match="Unsupported grid_sample mode"):
        _decode_sampler_payload(mode="garbage")

    with pytest.raises(RuntimeError, match="Unsupported grid_sample padding_mode"):
        _decode_sampler_payload(padding_mode="garbage")


# Covers storage-image outputs, which should not carry sampler state.
# Checks output payloads omit sampler metadata for storage images.
def test_storage_image_payload_vgf_does_not_require_sampler_fields():
    payload = _decode_sampler_payload()

    assert payload["output_0_vkdescriptortype"] == "VK_DESCRIPTOR_TYPE_STORAGE_IMAGE"
    assert "output_0_sampler" not in payload
