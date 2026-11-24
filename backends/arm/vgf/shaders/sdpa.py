# Copyright 2026 Arm Limited and/or its affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

import base64
import functools
import math
import shutil
import subprocess  # nosec B404 - fixed shader compiler invocation
import tempfile
from pathlib import Path
from typing import Any


CUSTOM_SDPA_SHADER_ENTRY_POINT = "main"
CUSTOM_SDPA_SHADER_LANGUAGE = "SPIR-V"
CUSTOM_SDPA_SHADER_LOCAL_SIZE_X = 64
_FLOAT32_VK_FORMAT = "VK_FORMAT_R32_SFLOAT"


class _CustomSDPAShaderCompilationError(RuntimeError):
    pass


def _format_float(value: float) -> str:
    return format(float(value), ".9g")


def _effective_scale(head_dim: int, scale: float | None) -> float:
    if scale is not None:
        return float(scale)
    return 1.0 / math.sqrt(float(head_dim))


def _dispatch_shape_for_output_shape(output_shape: tuple[int, ...]) -> list[int]:
    if len(output_shape) != 4:
        raise ValueError(f"SDPA output shape must be rank 4 BSHD, got {output_shape}")
    total_scalars = (
        int(output_shape[0])
        * int(output_shape[1])
        * int(output_shape[2])
        * int(output_shape[3])
    )
    return [
        (total_scalars + CUSTOM_SDPA_SHADER_LOCAL_SIZE_X - 1)
        // CUSTOM_SDPA_SHADER_LOCAL_SIZE_X,
        1,
        1,
    ]


def _decode_shape_params_bshd(
    query_shape: tuple[int, ...],
    key_shape: tuple[int, ...],
) -> tuple[int, int, int, int, int]:
    if int(query_shape[1]) != 1:
        raise ValueError("decode SDPA shader payload requires query sequence length 1")
    if int(query_shape[0]) != int(key_shape[0]):
        raise ValueError("decode SDPA shader payload requires matching batch size")
    if int(query_shape[3]) != int(key_shape[3]):
        raise ValueError("decode SDPA shader payload requires matching head dimension")
    if int(query_shape[2]) % int(key_shape[2]) != 0:
        raise ValueError(
            "decode SDPA shader payload requires query heads divisible by KV heads"
        )
    return (
        int(query_shape[0]),
        int(query_shape[2]),
        int(key_shape[2]),
        int(key_shape[1]),
        int(query_shape[3]),
    )


def _decode_shape_params_bhsd(
    query_shape: tuple[int, ...],
    key_shape: tuple[int, ...],
) -> tuple[int, int, int, int, int]:
    if int(query_shape[2]) != 1:
        raise ValueError("decode SDPA shader payload requires query sequence length 1")
    if int(query_shape[0]) != int(key_shape[0]):
        raise ValueError("decode SDPA shader payload requires matching batch size")
    if int(query_shape[3]) != int(key_shape[3]):
        raise ValueError("decode SDPA shader payload requires matching head dimension")
    if int(query_shape[1]) % int(key_shape[1]) != 0:
        raise ValueError(
            "decode SDPA shader payload requires query heads divisible by KV heads"
        )
    return (
        int(query_shape[0]),
        int(query_shape[1]),
        int(key_shape[1]),
        int(key_shape[2]),
        int(query_shape[3]),
    )


def _decode_shape_params(
    query_shape: tuple[int, ...],
    key_shape: tuple[int, ...],
    value_shape: tuple[int, ...],
    *,
    layout: str,
) -> tuple[int, int, int, int, int]:
    if len(query_shape) != 4 or len(key_shape) != 4 or len(value_shape) != 4:
        raise ValueError("decode SDPA shader payload expects rank-4 BSHD tensors")
    if tuple(key_shape) != tuple(value_shape):
        raise ValueError("decode SDPA shader payload requires key/value shape match")

    if layout == "BSHD":
        return _decode_shape_params_bshd(query_shape, key_shape)

    if layout == "BHSD":
        return _decode_shape_params_bhsd(query_shape, key_shape)

    raise ValueError(f"Unsupported SDPA shader layout: {layout}")


def _build_shader_source(
    *,
    batch_size: int,
    query_heads: int,
    key_value_heads: int,
    sequence_length: int,
    active_sequence_length: int,
    head_dim: int,
    scale: float,
    layout: str,
) -> str:
    if layout == "BSHD":
        key_value_offset = """uint key_value_offset(uint batch_idx, uint sequence_idx, uint head_idx, uint dim_idx) {
  return ((((batch_idx * SEQUENCE_LENGTH) + sequence_idx) * KEY_VALUE_HEADS) + head_idx) * HEAD_DIM + dim_idx;
}"""
    elif layout == "BHSD":
        key_value_offset = """uint key_value_offset(uint batch_idx, uint sequence_idx, uint head_idx, uint dim_idx) {
  return ((((batch_idx * KEY_VALUE_HEADS) + head_idx) * SEQUENCE_LENGTH) + sequence_idx) * HEAD_DIM + dim_idx;
}"""
    else:
        raise ValueError(f"Unsupported SDPA shader layout: {layout}")
    return f"""#version 450
layout(local_size_x = {CUSTOM_SDPA_SHADER_LOCAL_SIZE_X}, local_size_y = 1, local_size_z = 1) in;
layout(set = 0, binding = 0) readonly buffer Query {{ float query[]; }};
layout(set = 0, binding = 1) readonly buffer Key {{ float key[]; }};
layout(set = 0, binding = 2) readonly buffer Value {{ float value[]; }};
layout(set = 0, binding = 3) writeonly buffer OutputBuffer {{ float output_buffer[]; }};

const uint BATCH_SIZE = {int(batch_size)}u;
const uint QUERY_HEADS = {int(query_heads)}u;
const uint KEY_VALUE_HEADS = {int(key_value_heads)}u;
const uint SEQUENCE_LENGTH = {int(sequence_length)}u;
const uint ACTIVE_SEQUENCE_LENGTH = {int(active_sequence_length)}u;
const uint HEAD_DIM = {int(head_dim)}u;
const uint NUM_REPS = {int(query_heads // key_value_heads)}u;
const float ATTN_SCALE = {_format_float(scale)};

uint query_offset(uint batch_idx, uint head_idx, uint dim_idx) {{
  return (((batch_idx * QUERY_HEADS) + head_idx) * HEAD_DIM) + dim_idx;
}}

{key_value_offset}

void main() {{
  uint output_idx = gl_GlobalInvocationID.x;
  uint total_outputs = BATCH_SIZE * QUERY_HEADS * HEAD_DIM;
  if (output_idx >= total_outputs) {{
    return;
  }}

  uint dim_idx = output_idx % HEAD_DIM;
  uint output_head_idx = (output_idx / HEAD_DIM) % QUERY_HEADS;
  uint batch_idx = output_idx / (QUERY_HEADS * HEAD_DIM);
  uint key_value_head_idx = output_head_idx / NUM_REPS;

  float max_logit = -3.402823466e+38;
  for (uint sequence_idx = 0; sequence_idx < ACTIVE_SEQUENCE_LENGTH; ++sequence_idx) {{
    float dot_product = 0.0;
    for (uint inner_dim_idx = 0; inner_dim_idx < HEAD_DIM; ++inner_dim_idx) {{
      dot_product +=
          query[query_offset(batch_idx, output_head_idx, inner_dim_idx)] *
          key[key_value_offset(batch_idx, sequence_idx, key_value_head_idx, inner_dim_idx)];
    }}
    max_logit = max(max_logit, dot_product * ATTN_SCALE);
  }}

  float normalizer = 0.0;
  float accumulator = 0.0;
  for (uint sequence_idx = 0; sequence_idx < ACTIVE_SEQUENCE_LENGTH; ++sequence_idx) {{
    float dot_product = 0.0;
    for (uint inner_dim_idx = 0; inner_dim_idx < HEAD_DIM; ++inner_dim_idx) {{
      dot_product +=
          query[query_offset(batch_idx, output_head_idx, inner_dim_idx)] *
          key[key_value_offset(batch_idx, sequence_idx, key_value_head_idx, inner_dim_idx)];
    }}
    float weight = exp((dot_product * ATTN_SCALE) - max_logit);
    normalizer += weight;
    accumulator +=
        weight * value[key_value_offset(batch_idx, sequence_idx, key_value_head_idx, dim_idx)];
  }}

  output_buffer[output_idx] = accumulator / normalizer;
}}
"""


@functools.lru_cache(maxsize=32)
def _compile_shader_source(source: str) -> str:
    glslc = shutil.which("glslc")
    if glslc is None:
        raise _CustomSDPAShaderCompilationError(
            "glslc is required for fused decode SDPA custom shaders"
        )
    try:
        with tempfile.TemporaryDirectory() as tmpdir:
            source_path = Path(tmpdir) / "sdpa_decode.glsl"
            spirv_path = Path(tmpdir) / "sdpa_decode.spv"
            source_path.write_text(source, encoding="utf-8")
            subprocess.run(  # nosec B603 - glslc path is resolved from PATH.
                [
                    glslc,
                    "-fshader-stage=compute",
                    str(source_path),
                    "-o",
                    str(spirv_path),
                ],
                check=True,
            )
            return base64.b64encode(spirv_path.read_bytes()).decode("ascii")
    except (OSError, subprocess.CalledProcessError) as error:
        raise _CustomSDPAShaderCompilationError(
            "failed to compile fused decode SDPA custom shader"
        ) from error


def build_custom_sdpa_shader_payload(
    query_shape: tuple[int, ...],
    key_shape: tuple[int, ...],
    value_shape: tuple[int, ...],
    start_pos: int,
    dropout_p: float,
    is_causal: bool,
    scale: float | None,
    *,
    layout: str = "BSHD",
) -> dict[str, Any]:
    """Build Vulkan shader metadata for fixed-position decode fused SDPA."""
    batch_size, query_heads, key_value_heads, sequence_length, head_dim = (
        _decode_shape_params(
            query_shape,
            key_shape,
            value_shape,
            layout=layout,
        )
    )
    if float(dropout_p) != 0.0:
        raise ValueError("decode SDPA shader payload only supports dropout_p == 0.0")
    if not is_causal:
        raise ValueError("decode SDPA shader payload only supports causal attention")

    active_sequence_length = min(sequence_length, int(start_pos) + 1)
    effective_scale = _effective_scale(head_dim, scale)
    shader_source = _build_shader_source(
        batch_size=batch_size,
        query_heads=query_heads,
        key_value_heads=key_value_heads,
        sequence_length=sequence_length,
        active_sequence_length=active_sequence_length,
        head_dim=head_dim,
        scale=effective_scale,
        layout=layout,
    )

    return {
        "entry_point": CUSTOM_SDPA_SHADER_ENTRY_POINT,
        "workgroup_sizes": _dispatch_shape_for_output_shape(query_shape),
        "is_vkshader": True,
        "shader_code": _compile_shader_source(shader_source),
        "shader_language": CUSTOM_SDPA_SHADER_LANGUAGE,
        "push_constants": "",
        "layout": layout,
        "query_shape": list(query_shape),
        "key_shape": list(key_shape),
        "value_shape": list(value_shape),
        "start_pos": int(start_pos),
        "active_sequence_length": active_sequence_length,
        "dropout_p": float(dropout_p),
        "is_causal": bool(is_causal),
        "scale": None if scale is None else float(scale),
        "effective_scale": effective_scale,
        "input_0_binding": 0,
        "input_1_binding": 1,
        "input_2_binding": 2,
        "output_0_binding": 3,
        "input_0_type": "Buffer",
        "input_1_type": "Buffer",
        "input_2_type": "Buffer",
        "output_0_type": "Buffer",
        "input_0_vkdescriptortype": "VK_DESCRIPTOR_TYPE_STORAGE_BUFFER",
        "input_1_vkdescriptortype": "VK_DESCRIPTOR_TYPE_STORAGE_BUFFER",
        "input_2_vkdescriptortype": "VK_DESCRIPTOR_TYPE_STORAGE_BUFFER",
        "output_0_vkdescriptortype": "VK_DESCRIPTOR_TYPE_STORAGE_BUFFER",
        "input_0_descriptorset": 0,
        "input_1_descriptorset": 0,
        "input_2_descriptorset": 0,
        "output_0_descriptorset": 0,
        "input_0_vkformat": _FLOAT32_VK_FORMAT,
        "input_1_vkformat": _FLOAT32_VK_FORMAT,
        "input_2_vkformat": _FLOAT32_VK_FORMAT,
        "output_0_vkformat": _FLOAT32_VK_FORMAT,
    }
