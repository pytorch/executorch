# Copyright 2026 Arm Limited and/or its affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from pathlib import Path

from executorch.backends.arm.operators.op_tosa_rescale import (
    _compute_multiplier_and_shift,
)

from . import _compile_shader_source

OPERATOR_NAME = "rife.final_warps"
BLEND_OPERATOR_NAME = "rife.final_blend"


def _rescale_source(name, node):
    multipliers, shifts = _compute_multiplier_and_shift(node.args[2])
    shift = shifts[0]
    rounding = 1 << (shift - 1)
    result = (
        f"int((uint(high) << {32-shift}u) | (rounded >> {shift}u))"
        if shift < 32
        else "high" if shift == 32 else f"(high >> {shift-32})"
    )
    return f"""int {name}(int value) {{
  int high, low;
  imulExtended(value - ({node.args[3]}), {multipliers[0]}, high, low);
  uint rounded = uint(low) + {rounding & 0xffffffff}u;
  high += {rounding >> 32} + int(rounded < uint(low));
  return {result} + ({node.args[4]});
}}"""


def build_final_warps_payload(payloads, rescales, sigmoid_table, *, blend=None):
    """Compile block1 PixelShuffle, flow update, mask lookup and both warps.

    Args:
        payloads (list[dict]): Validated original warp descriptors and qparams.
        rescales (tuple[Node, Node, Node]): Previous flow, delta and sum rescales.
        sigmoid_table (Tensor): Original signed INT8 sigmoid lookup table.
        blend (RifeBlend | None): Optional quantized final image blend.

    Returns:
        dict: Shader payload producing warped images and mask, or final RGB.

    """
    source = Path(__file__).with_name("final_warps.glsl").read_text()
    normalization = payloads[0].get("flow_normalization", [1.0, 1.0])
    for axis, value in zip(("X", "Y"), normalization):
        source = source.replace(f"@FLOW_NORMALIZATION_{axis}@", format(value, ".9g"))
    functions = [
        _rescale_source(name, node)
        for name, node in zip(
            ("rescalePrevious", "rescaleDelta", "rescaleSum"), rescales
        )
    ]
    zero_points = ""
    if blend is not None:
        functions.extend(
            _rescale_source(name, node)
            for name, node in zip(
                (
                    "rescaleMul0",
                    "rescaleMul1",
                    "rescaleAdd0",
                    "rescaleAdd1",
                    "rescaleOutput",
                ),
                blend.rescales,
            )
        )
        zero_points = "\n".join(
            f"const int {prefix}{i}_ZERO_POINT = {value};"
            for prefix, values in (
                ("IMAGE", blend.image_zero_points),
                ("MASK", blend.mask_zero_points),
            )
            for i, value in enumerate(values)
        )
    source = source.replace("@FUSE_BLEND@", str(int(blend is not None)))
    source = source.replace("@BLEND_ZERO_POINTS@", zero_points)
    source = source.replace(
        "@COMPLEMENT_TABLE@",
        ", ".join(map(str, blend.complement.tolist())) if blend is not None else "",
    )
    source = source.replace(
        "@RESCALES@",
        "\n".join(functions),
    )
    source = source.replace(
        "@SIGMOID_TABLE@", ", ".join(map(str, sigmoid_table.tolist()))
    )
    for key in ("flow_scale", "flow_zero_point"):
        source = source.replace(
            "@" + key.upper() + "@", format(payloads[0][key], ".9g")
        )
    for index, original in enumerate(payloads):
        for prefix in ("input", "output"):
            for suffix in ("scale", "zero_point"):
                source = source.replace(
                    f"@{prefix.upper()}{index}_{suffix.upper()}@",
                    format(original[f"{prefix}_{suffix}"], ".9g"),
                )
    payload = {
        "entry_point": "main",
        "shader_language": "SPIR-V",
        "shader_code": _compile_shader_source(source),
        "workgroup_sizes": payloads[0]["workgroup_sizes"],
    }
    for index, original in enumerate(payloads):
        for key, value in original.items():
            if key.startswith("input_0_") or (
                blend is None and key.startswith("output_0_")
            ):
                prefix = "input" if key.startswith("input") else "output"
                payload[key.replace(f"{prefix}_0_", f"{prefix}_{index}_")] = (
                    index + (4 if prefix == "output" else 0)
                    if key.endswith("_binding")
                    else value
                )
    bindings = [
        ("input_2", 2, "Buffer"),
        ("input_3", 3, "Tensor"),
        ("output_2", 6, "Tensor") if blend is None else ("output_0", 4, "Tensor"),
    ]
    for prefix, binding, kind in bindings:
        payload.update(
            {
                f"{prefix}_binding": binding,
                f"{prefix}_descriptorset": 0,
                f"{prefix}_type": kind,
                f"{prefix}_vkformat": "VK_FORMAT_R8_SINT",
                f"{prefix}_vkdescriptortype": (
                    "VK_DESCRIPTOR_TYPE_TENSOR_ARM"
                    if kind == "Tensor"
                    else "VK_DESCRIPTOR_TYPE_STORAGE_BUFFER"
                ),
            }
        )
    return payload
