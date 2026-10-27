# Copyright 2026 Arm Limited and/or its affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from pathlib import Path

from . import _compile_shader_source

OPERATOR_NAME = "rife.warp_pack_concat"


def build_packed_warp_payload(payloads, timestep, *, fuse_block0_tail=False):
    """Compile four compatible samplers with a direct C32 concat output.

    Args:
        payloads (list[dict]): Validated original sampler descriptors and qparams.
        timestep (int): Constant signed INT8 timestep code.
        fuse_block0_tail (bool): Read the block0 phase tensor and pack its
            resized state into C32.

    Returns:
        dict: Vulkan shader payload with separate state or fused block0 input.

    """
    source = Path(__file__).with_name("pack_warps.glsl").read_text()
    normalization = payloads[0].get("flow_normalization", [1.0, 1.0])
    for axis, value in zip(("X", "Y"), normalization):
        source = source.replace(f"@FLOW_NORMALIZATION_{axis}@", format(value, ".9g"))
    for key in (
        "flow_scale",
        "flow_zero_point",
        "input_scale",
        "input_zero_point",
        "output_scale",
        "output_zero_point",
    ):
        source = source.replace(
            "@" + key.upper() + "@", format(payloads[0][key], ".9g")
        )
    source = source.replace("@TIMESTEP@", str(timestep))
    source = source.replace("@FUSE_BLOCK0_TAIL@", str(int(fuse_block0_tail)))
    payload = {
        "entry_point": "main",
        "shader_language": "SPIR-V",
        "shader_code": _compile_shader_source(source),
        "workgroup_sizes": payloads[0]["workgroup_sizes"],
    }
    for index, original in enumerate(payloads):
        for key, value in original.items():
            if key.startswith("input_0_"):
                payload[key.replace("input_0_", f"input_{index}_")] = (
                    index if key.endswith("_binding") else value
                )
    bindings = (
        [
            ("input_4", 4, "Tensor", "R8_SINT"),
            ("output_0", 5, "Buffer", "R8_SINT"),
        ]
        if fuse_block0_tail
        else [(f"input_{i}", i, "Tensor", "R8_SINT") for i in range(4, 7)]
        + [("output_0", 7, "Buffer", "R8_SINT")]
    )
    for prefix, binding, kind, vk_format in bindings:
        payload.update(
            {
                f"{prefix}_binding": binding,
                f"{prefix}_descriptorset": 0,
                f"{prefix}_type": kind,
                f"{prefix}_vkformat": f"VK_FORMAT_{vk_format}",
                f"{prefix}_vkdescriptortype": (
                    "VK_DESCRIPTOR_TYPE_TENSOR_ARM"
                    if kind == "Tensor"
                    else "VK_DESCRIPTOR_TYPE_STORAGE_BUFFER"
                ),
            }
        )
    return payload
