# Copyright 2026 Arm Limited and/or its affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import base64
import shutil
import subprocess  # nosec B404
import tempfile
from pathlib import Path
from typing import Any

from executorch.backends.arm.vgf.shaders.grid_sampler import (
    GRID_SAMPLER_2D_QUANTIZED_GRID_VK_FORMAT,
    GRID_SAMPLER_2D_SAMPLER_INT8_VK_FORMAT,
    GRID_SAMPLER_2D_SHADER_ENTRY_POINT,
    GRID_SAMPLER_2D_SHADER_LANGUAGE,
)

WARP_DOWNSAMPLE_OPERATOR_PREFIX = "rife.warp_downsample"
WARP_DOWNSAMPLE_WORKGROUP_SIZES = [8, 8, 1]
SUPPORTED_WARP_DOWNSAMPLE_SCALES = (2, 4, 8)


def warp_downsample_operator_name(scale: int) -> str:
    if scale not in SUPPORTED_WARP_DOWNSAMPLE_SCALES:
        raise ValueError(f"Unsupported warp_downsample scale {scale}")
    return f"{WARP_DOWNSAMPLE_OPERATOR_PREFIX}{scale}"


def _dispatch_shape_for_output_shape(output_shape: tuple[int, ...]) -> list[int]:
    if len(output_shape) != 4:
        raise ValueError(
            "warp_downsample output_shape must be rank 4 NCHW, "
            f"got shape {output_shape}"
        )
    output_batch = int(output_shape[0])
    output_height = int(output_shape[2])
    output_width = int(output_shape[3])
    group_x, group_y, group_z = WARP_DOWNSAMPLE_WORKGROUP_SIZES
    return [
        (output_width + group_x - 1) // group_x,
        (output_height + group_y - 1) // group_y,
        (output_batch + group_z - 1) // group_z,
    ]


def _sampler_config() -> dict[str, str]:
    return {
        "min_filter": "VK_FILTER_LINEAR",
        "mag_filter": "VK_FILTER_LINEAR",
        "address_mode_u": "VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE",
        "address_mode_v": "VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE",
        "border_color": "VK_BORDER_COLOR_FLOAT_TRANSPARENT_BLACK",
    }


def _format_float(value: float) -> str:
    return format(float(value), ".9g")


def _flow_shader_source(
    *,
    scale: int,
    flow_dtype: Any | None,
    flow_scale: float | None,
    flow_zero_point: int | None,
    flow_channel_offset: int,
) -> str:
    offset0 = scale // 2 - 1
    offset1 = scale // 2
    if str(flow_dtype) != "torch.int8":
        raise ValueError("warp_downsample flow payload supports only int8")
    if flow_scale is None or flow_zero_point is None:
        raise ValueError("int8 flow requires flow_scale and flow_zero_point")
    read_flow_value = f"""  int8_t value[1];
  tensorReadARM(flow, coords, value);
  return (float(value[0]) - float({int(flow_zero_point)})) * {_format_float(flow_scale)};"""
    return f"""// Copyright 2026 Arm Limited and/or its affiliates.
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#version 450
#extension GL_ARM_tensors : require
#extension GL_EXT_shader_explicit_arithmetic_types_int8 : require
layout(set = 0, binding = 0) uniform sampler2D inputImage;
layout(set = 0, binding = 1) uniform tensorARM<int8_t, 4> flow;
layout(set = 0, binding = 2, rgba8_snorm) uniform writeonly image2D outImage;

layout(local_size_x = 8, local_size_y = 8, local_size_z = 1) in;

const int kScale = {scale};
const int kOffset0 = {offset0};
const int kOffset1 = {offset1};
const uint kFlowChannelOffset = {int(flow_channel_offset)}u;

float readFlowChannel(ivec2 p, uint channel) {{
  uint coords[4] = uint[](0u, channel, uint(p.y), uint(p.x));
{read_flow_value}
}}

vec2 readFlowXY(ivec2 p) {{
  return vec2(
      readFlowChannel(p, kFlowChannelOffset),
      readFlowChannel(p, kFlowChannelOffset + 1u));
}}

vec2 alignCornersUv(vec2 gridXY) {{
  vec2 inputSize = vec2(textureSize(inputImage, 0));
  vec2 texel = (gridXY + vec2(1.0)) * vec2(0.5) * (inputSize - vec2(1.0));
  return (texel + vec2(0.5)) / inputSize;
}}

vec2 baseGridXY(ivec2 p, ivec2 fullSize) {{
  return vec2(
      (2.0 * float(p.x)) / float(fullSize.x - 1) - 1.0,
      (2.0 * float(p.y)) / float(fullSize.y - 1) - 1.0);
}}

vec4 sampleWarped(ivec2 p, ivec2 fullSize) {{
  vec2 flowXY = readFlowXY(p);
  vec2 gridXY = baseGridXY(p, fullSize) + flowXY;
  return texture(inputImage, alignCornersUv(gridXY));
}}

void main() {{
  ivec2 outSize = imageSize(outImage);
  ivec2 gid = ivec2(gl_GlobalInvocationID.xy);
  if (gid.x >= outSize.x || gid.y >= outSize.y) {{
    return;
  }}

  ivec2 fullSize = textureSize(inputImage, 0);
  ivec2 base = gid * kScale;
  ivec2 p00 = base + ivec2(kOffset0, kOffset0);
  ivec2 p01 = base + ivec2(kOffset1, kOffset0);
  ivec2 p10 = base + ivec2(kOffset0, kOffset1);
  ivec2 p11 = base + ivec2(kOffset1, kOffset1);

  vec4 value = 0.25 * (
      sampleWarped(p00, fullSize) +
      sampleWarped(p01, fullSize) +
      sampleWarped(p10, fullSize) +
      sampleWarped(p11, fullSize));
  imageStore(outImage, gid, value);
}}
"""


def _compile_shader_source(source: str) -> str:
    glslc = shutil.which("glslc")
    if glslc is None:
        raise RuntimeError("glslc is required to compile RIFE VGF custom shaders")
    with tempfile.TemporaryDirectory() as tmpdir:
        source_path = Path(tmpdir) / "warp_downsample.glsl"
        spirv_path = Path(tmpdir) / "warp_downsample.spv"
        source_path.write_text(source, encoding="utf-8")
        subprocess.run(  # nosec B603
            [glslc, "-fshader-stage=compute", str(source_path), "-o", str(spirv_path)],
            check=True,
        )
        return base64.b64encode(spirv_path.read_bytes()).decode("ascii")


def _shader_code(
    *,
    scale: int,
    flow_dtype: Any | None,
    flow_scale: float | None,
    flow_zero_point: int | None,
    flow_channel_offset: int,
) -> str:
    return _compile_shader_source(
        _flow_shader_source(
            scale=scale,
            flow_dtype=flow_dtype,
            flow_scale=flow_scale,
            flow_zero_point=flow_zero_point,
            flow_channel_offset=flow_channel_offset,
        )
    )


def build_warp_downsample_payload(
    scale: int,
    input_shape: tuple[int, ...],
    output_shape: tuple[int, ...],
    input_dtype: Any,
    output_dtype: Any | None = None,
    flow_dtype: Any | None = None,
    flow_scale: float | None = None,
    flow_zero_point: int | None = None,
    flow_channel_offset: int = 0,
) -> dict[str, Any]:
    if scale not in SUPPORTED_WARP_DOWNSAMPLE_SCALES:
        raise ValueError(f"Unsupported warp_downsample scale {scale}")
    if output_dtype is None:
        output_dtype = input_dtype
    if str(input_dtype) != "torch.int8" or str(output_dtype) != "torch.int8":
        raise ValueError(
            "warp_downsample supports only matching int8 RGBA image payloads"
        )
    if len(input_shape) != 4 or int(input_shape[0]) != 1 or int(input_shape[1]) != 4:
        raise ValueError(
            "warp_downsample currently requires NCHW input shape [1, 4, H, W]"
        )
    if str(flow_dtype) != "torch.int8":
        raise ValueError("warp_downsample supports only int8 flow payloads")
    shader_code = _shader_code(
        scale=scale,
        flow_dtype=flow_dtype,
        flow_scale=flow_scale,
        flow_zero_point=flow_zero_point,
        flow_channel_offset=flow_channel_offset,
    )
    return {
        "entry_point": GRID_SAMPLER_2D_SHADER_ENTRY_POINT,
        # Current runtime consumes this field as dispatch counts, not local
        # shader workgroup size. The shader uses an 8x8 output-space work
        # volume per workgroup.
        "workgroup_sizes": _dispatch_shape_for_output_shape(output_shape),
        "shader_language": GRID_SAMPLER_2D_SHADER_LANGUAGE,
        "shader_code": shader_code,
        "input_0_binding": 0,
        "input_0_descriptorset": 0,
        "input_0_type": "Image",
        "input_0_vkformat": GRID_SAMPLER_2D_SAMPLER_INT8_VK_FORMAT,
        "input_0_vkdescriptortype": "VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER",
        "input_0_sampler": _sampler_config(),
        "input_1_binding": 1,
        "input_1_descriptorset": 0,
        "input_1_type": "Tensor",
        "input_1_vkformat": GRID_SAMPLER_2D_QUANTIZED_GRID_VK_FORMAT,
        "input_1_vkdescriptortype": "VK_DESCRIPTOR_TYPE_TENSOR_ARM",
        "output_0_binding": 2,
        "output_0_descriptorset": 0,
        "output_0_type": "Image",
        "output_0_vkformat": GRID_SAMPLER_2D_SAMPLER_INT8_VK_FORMAT,
        "output_0_vkdescriptortype": "VK_DESCRIPTOR_TYPE_STORAGE_IMAGE",
    }
