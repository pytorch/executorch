# Copyright 2026 Arm Limited and/or its affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from pathlib import Path

from . import _compile_shader_source

OPERATOR_NAME = "rife.warp_pack_concat"


def build_packed_warp_payload(payloads, timestep):
    """Compile four compatible samplers with a direct C32 concat output.

    Args:
        payloads (list[dict]): Validated original sampler descriptors and qparams.
        timestep (int): Constant signed INT8 timestep code.

    Returns:
        dict: Vulkan shader payload with four images and three state inputs.

    """
    source = Path(__file__).with_name("pack_warps.glsl").read_text()
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
    for index in range(4, 8):
        prefix = f"{'input' if index < 7 else 'output'}_{index if index < 7 else 0}"
        payload.update(
            {
                f"{prefix}_binding": index,
                f"{prefix}_descriptorset": 0,
                f"{prefix}_type": "Tensor" if index < 7 else "Buffer",
                f"{prefix}_vkformat": "VK_FORMAT_R8_SINT",
                f"{prefix}_vkdescriptortype": (
                    "VK_DESCRIPTOR_TYPE_TENSOR_ARM"
                    if index < 7
                    else "VK_DESCRIPTOR_TYPE_STORAGE_BUFFER"
                ),
            }
        )
    return payload
