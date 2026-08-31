# Copyright 2026 Arm Limited and/or its affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import base64
import json
import shutil
import subprocess  # nosec B404 - fixed shader compiler invocation
import tempfile
from importlib.resources import files
from pathlib import Path
from typing import Any, Sequence

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
GRID_SAMPLER_2D_QUANTIZED_GRID_VK_FORMAT = "VK_FORMAT_R8_SINT"
FLOW_OFFSET_GRID_SAMPLER_OPERATOR_NAME = "torch.nn.functional.grid_sample.flow_offset"
FLOW_OFFSET_GRID_SAMPLER_SHADER_SOURCE = (
    "flow_offset_grid_sampler_int8_align_corners.glsl"
)


class _FlowOffsetShaderCompilationError(RuntimeError):
    pass


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
    """Build the custom operator name for a 2D grid sampler variant.

    Args:
        interpolation_mode (int): PyTorch grid_sample interpolation mode.
        padding_mode (int): PyTorch grid_sample padding mode.
        align_corners (bool): Whether grid_sample aligns tensor corners.

    Returns:
        str: Fully qualified custom operator name.

    """
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
    output_shape: tuple[int, ...] | None = None,
    input_dtype: Any | None = None,
    output_dtype: Any | None = None,
    grid_dtype: Any | None = None,
    extra_tensor_input_vkformats: Sequence[str] | None = None,
) -> dict[str, Any]:
    """Build Vulkan custom shader metadata for a 2D grid sampler variant.

    Args:
        interpolation_mode (int): PyTorch grid_sample interpolation mode.
        padding_mode (int): PyTorch grid_sample padding mode.
        align_corners (bool): Whether grid_sample aligns tensor corners.
        input_shape (tuple[int, ...] | None): Input tensor shape, used to
            select sampler-backed shader metadata when supported.
        output_shape (tuple[int, ...] | None): Output tensor shape, required
            to derive dispatch counts for the shader launch.
        input_dtype (Any | None): Input tensor dtype, used to select sampler
            Vulkan formats when supported.
        output_dtype (Any | None): Output tensor dtype. Defaults to
            input_dtype when omitted.
        grid_dtype (Any | None): Grid tensor dtype, used to select the
            quantized-grid sampler path when supported.
        extra_tensor_input_vkformats (Sequence[str] | None): Vulkan formats
            for any additional tensor inputs appended after the grid input.

    Returns:
        dict[str, Any]: Custom shader metadata payload.

    """
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
    if output_shape is None:
        raise ValueError("grid_sampler payload requires output_shape for dispatch")
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
    use_quantized_grid = str(grid_dtype) == "torch.int8"
    if use_quantized_grid and not (
        use_sampler
        and str(input_dtype) == "torch.int8"
        and str(output_dtype) == "torch.int8"
    ):
        raise ValueError(
            "Quantized grid-sample payload is only supported for the int8 sampler path"
        )
    if sampler_vk_format == GRID_SAMPLER_2D_SAMPLER_INT8_VK_FORMAT and (
        not use_quantized_grid or len(extra_tensor_input_vkformats or ()) != 2
    ):
        raise ValueError(
            "Int8 sampler grid-sample payload requires an int8 grid and "
            "explicit scale/zero-point tensor inputs"
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
        # Current runtime consumes this field as dispatch counts, not local
        # shader workgroup size. The current grid-sample shaders use a 2D
        # output-space work model with an 8x8 work volume per workgroup.
        "workgroup_sizes": _dispatch_shape_for_output_shape(output_shape),
        "shader_language": GRID_SAMPLER_2D_SHADER_LANGUAGE,
        "shader_code": shader_code,
        "input_0_binding": 0,
        "input_0_descriptorset": 0,
        "input_1_type": "Tensor",
        "input_1_vkformat": (
            GRID_SAMPLER_2D_QUANTIZED_GRID_VK_FORMAT
            if use_quantized_grid
            else GRID_SAMPLER_2D_VK_FORMAT
        ),
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
                "input_1_vkdescriptortype": "VK_DESCRIPTOR_TYPE_TENSOR_ARM",
                "output_0_type": "Tensor",
                "output_0_vkformat": GRID_SAMPLER_2D_VK_FORMAT,
                "output_0_vkdescriptortype": "VK_DESCRIPTOR_TYPE_STORAGE_BUFFER",
            }
        )
    extra_tensor_input_vkformats = extra_tensor_input_vkformats or ()
    for extra_idx, vk_format in enumerate(extra_tensor_input_vkformats):
        input_idx = 2 + extra_idx
        payload.update(
            {
                f"input_{input_idx}_type": "Tensor",
                f"input_{input_idx}_vkformat": vk_format,
                f"input_{input_idx}_binding": 3 + extra_idx,
                f"input_{input_idx}_descriptorset": 0,
                f"input_{input_idx}_vkdescriptortype": "VK_DESCRIPTOR_TYPE_STORAGE_BUFFER",
            }
        )
    return payload


def _dispatch_shape_for_output_shape(output_shape: tuple[int, ...]) -> list[int]:
    if len(output_shape) != 4:
        raise ValueError(
            f"grid_sampler output_shape must be rank 4 NCHW, got shape {output_shape}"
        )
    output_batch = int(output_shape[0])
    output_height = int(output_shape[2])
    output_width = int(output_shape[3])
    group_x, group_y, group_z = GRID_SAMPLER_2D_WORKGROUP_SIZES
    return [
        (output_width + group_x - 1) // group_x,
        (output_height + group_y - 1) // group_y,
        (output_batch + group_z - 1) // group_z,
    ]


def flow_offset_grid_sampler_operator_name() -> str:
    """Return the custom operator name for fused flow-offset sampling.

    Returns:
        str: Fully qualified custom operator name.

    """
    return FLOW_OFFSET_GRID_SAMPLER_OPERATOR_NAME


def _format_float(value: float) -> str:
    return format(float(value), ".9g")


def _compile_flow_offset_grid_sampler_shader(
    *,
    input_scale: float,
    input_zero_point: int,
    output_scale: float,
    output_zero_point: int,
    flow_scale: float,
    flow_zero_point: int,
    flow_channel_offset: int,
) -> str:
    source = (
        files(__package__)
        .joinpath(FLOW_OFFSET_GRID_SAMPLER_SHADER_SOURCE)
        .read_text(encoding="utf-8")
        .replace("@INPUT_SCALE@", _format_float(input_scale))
        .replace("@INPUT_ZERO_POINT@", str(int(input_zero_point)))
        .replace("@OUTPUT_SCALE@", _format_float(output_scale))
        .replace("@OUTPUT_ZERO_POINT@", str(int(output_zero_point)))
        .replace("@FLOW_SCALE@", _format_float(flow_scale))
        .replace("@FLOW_ZERO_POINT@", str(int(flow_zero_point)))
        .replace("@FLOW_X_CHANNEL@", f"{int(flow_channel_offset)}u")
        .replace("@FLOW_Y_CHANNEL@", f"{int(flow_channel_offset) + 1}u")
    )
    glslc = shutil.which("glslc")
    if glslc is None:
        raise _FlowOffsetShaderCompilationError(
            "glslc is required for fused flow-offset grid sampling"
        )
    try:
        with tempfile.TemporaryDirectory() as tmpdir:
            source_path = Path(tmpdir) / "flow_offset_grid_sampler.glsl"
            spirv_path = Path(tmpdir) / "flow_offset_grid_sampler.spv"
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
        raise _FlowOffsetShaderCompilationError(
            "failed to compile fused flow-offset grid sampler shader"
        ) from error


def build_flow_offset_grid_sampler_payload(
    *,
    input_shape: tuple[int, ...],
    output_shape: tuple[int, ...],
    flow_shape: tuple[int, ...],
    input_scale: float,
    input_zero_point: int,
    output_scale: float,
    output_zero_point: int,
    flow_scale: float,
    flow_zero_point: int,
    flow_channel_offset: int,
) -> dict[str, Any]:
    """Build custom shader metadata for fused flow-offset grid sampling.

    Args:
        input_shape (tuple[int, ...]): Static NCHW image input shape.
        output_shape (tuple[int, ...]): Static NCHW output shape.
        flow_shape (tuple[int, ...]): Static NCHW four-channel flow shape.
        input_scale (float): Image input quantization scale.
        input_zero_point (int): Image input quantization zero point.
        output_scale (float): Output quantization scale.
        output_zero_point (int): Output quantization zero point.
        flow_scale (float): Flow input quantization scale.
        flow_zero_point (int): Flow input quantization zero point.
        flow_channel_offset (int): First flow channel, either zero or two.

    Returns:
        dict[str, Any]: Vulkan custom shader metadata payload.

    """
    if len(input_shape) != 4 or tuple(input_shape[:2]) != (1, 4):
        raise ValueError(f"expected static NCHW [1,4,H,W] input, got {input_shape}")
    if len(output_shape) != 4 or tuple(output_shape[:2]) != (1, 4):
        raise ValueError(f"expected static NCHW [1,4,H,W] output, got {output_shape}")
    if len(flow_shape) != 4 or tuple(flow_shape[:2]) != (1, 4):
        raise ValueError(f"expected static NCHW [1,4,H,W] flow, got {flow_shape}")
    if tuple(input_shape[2:]) != tuple(output_shape[2:]):
        raise ValueError("input and output spatial shapes must match")
    if tuple(flow_shape[2:]) != tuple(output_shape[2:]):
        raise ValueError("flow and output spatial shapes must match")
    if any(int(dim) <= 1 for dim in output_shape[2:]):
        raise ValueError("flow-offset grid sampling requires H and W greater than 1")
    if flow_channel_offset not in (0, 2):
        raise ValueError("flow_channel_offset must be 0 or 2")

    sampler_vk_format = GRID_SAMPLER_2D_SAMPLER_INT8_VK_FORMAT
    return {
        "entry_point": GRID_SAMPLER_2D_SHADER_ENTRY_POINT,
        "workgroup_sizes": _dispatch_shape_for_output_shape(output_shape),
        "shader_language": GRID_SAMPLER_2D_SHADER_LANGUAGE,
        "shader_code": _compile_flow_offset_grid_sampler_shader(
            input_scale=input_scale,
            input_zero_point=input_zero_point,
            output_scale=output_scale,
            output_zero_point=output_zero_point,
            flow_scale=flow_scale,
            flow_zero_point=flow_zero_point,
            flow_channel_offset=flow_channel_offset,
        ),
        "operator_name": FLOW_OFFSET_GRID_SAMPLER_OPERATOR_NAME,
        "input_scale": input_scale,
        "input_zero_point": input_zero_point,
        "output_scale": output_scale,
        "output_zero_point": output_zero_point,
        "flow_scale": flow_scale,
        "flow_zero_point": flow_zero_point,
        "flow_channel_offset": flow_channel_offset,
        "input_0_binding": 0,
        "input_0_descriptorset": 0,
        "input_0_type": "Image",
        "input_0_vkformat": sampler_vk_format,
        "input_0_vkdescriptortype": "VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER",
        "input_0_sampler": _sampler_config(interpolation_mode=0, padding_mode=1),
        "input_1_binding": 1,
        "input_1_descriptorset": 0,
        "input_1_type": "Tensor",
        "input_1_vkformat": GRID_SAMPLER_2D_QUANTIZED_GRID_VK_FORMAT,
        "input_1_vkdescriptortype": "VK_DESCRIPTOR_TYPE_TENSOR_ARM",
        "output_0_binding": 2,
        "output_0_descriptorset": 0,
        "output_0_type": "Image",
        "output_0_vkformat": sampler_vk_format,
        "output_0_vkdescriptortype": "VK_DESCRIPTOR_TYPE_STORAGE_IMAGE",
    }


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
    """Encode a custom shader payload as implementation attributes.

    Args:
        payload (dict[str, Any]): Custom shader metadata payload.

    Returns:
        list[int]: UTF-8 JSON bytes represented as integer attributes.

    """
    return list(json.dumps(payload, sort_keys=True).encode("utf-8"))


def decode_payload(implementation_attrs: list[int]) -> dict[str, Any]:
    """Decode implementation attributes into a custom shader payload.

    Args:
        implementation_attrs (list[int]): UTF-8 JSON bytes represented as
            integer attributes.

    Returns:
        dict[str, Any]: Custom shader metadata payload.

    """
    return json.loads(bytes(implementation_attrs).decode("utf-8"))
