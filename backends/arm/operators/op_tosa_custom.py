# Copyright 2026 Arm Limited and/or its affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import json
from typing import Any, List

import torch
import tosa_serializer as ts

from executorch.backends.arm.operators.node_visitor import (
    NodeVisitor,
    register_node_visitor,
)
from executorch.backends.arm.tosa.mapping import TosaArg

_VULKAN_CUSTOM_SHADER_DOMAIN = "com.arm.VulkanCustomShader"


def _vk_format_component_count(vk_format: str) -> int:
    component_count = {
        "VK_FORMAT_R8_BOOL_ARM": 1,
        "VK_FORMAT_R8_UINT": 1,
        "VK_FORMAT_R8_SINT": 1,
        "VK_FORMAT_R16_UINT": 1,
        "VK_FORMAT_R16_SINT": 1,
        "VK_FORMAT_R16_SFLOAT": 1,
        "VK_FORMAT_R32_UINT": 1,
        "VK_FORMAT_R32_SINT": 1,
        "VK_FORMAT_R32_SFLOAT": 1,
        "VK_FORMAT_R64_SINT": 1,
        "VK_FORMAT_R8G8_UINT": 2,
        "VK_FORMAT_R8G8_SINT": 2,
        "VK_FORMAT_R16G16_UINT": 2,
        "VK_FORMAT_R16G16_SINT": 2,
        "VK_FORMAT_R16G16_SFLOAT": 2,
        "VK_FORMAT_R32G32_UINT": 2,
        "VK_FORMAT_R32G32_SINT": 2,
        "VK_FORMAT_R32G32_SFLOAT": 2,
        "VK_FORMAT_R8G8B8A8_UINT": 4,
        "VK_FORMAT_R8G8B8A8_SINT": 4,
        "VK_FORMAT_R8G8B8A8_SNORM": 4,
        "VK_FORMAT_R16G16B16A16_UINT": 4,
        "VK_FORMAT_R16G16B16A16_SINT": 4,
        "VK_FORMAT_R16G16B16A16_SFLOAT": 4,
        "VK_FORMAT_R32G32B32A32_UINT": 4,
        "VK_FORMAT_R32G32B32A32_SINT": 4,
        "VK_FORMAT_R32G32B32A32_SFLOAT": 4,
    }.get(vk_format)
    if component_count is None:
        raise ValueError(f"Unsupported image VkFormat '{vk_format}'")
    return component_count


def _validate_image_tensor_arg(arg: TosaArg, arg_name: str, vk_format: str) -> None:
    if arg.shape is None:
        raise ValueError(f"{arg_name} must have a statically known shape")
    if len(arg.shape) not in (3, 4):
        raise ValueError(
            f"{arg_name} image tensors must be rank 3 or 4, got shape {arg.shape}"
        )
    if len(arg.shape) == 4 and arg.shape[0] != 1:
        raise ValueError(
            f"{arg_name} image tensors must have batch size 1, got shape {arg.shape}"
        )
    channels = int(arg.shape[-1])
    format_component_count = _vk_format_component_count(vk_format)
    if channels != format_component_count:
        raise ValueError(
            f"{arg_name} channel dimension {channels} does not match image format "
            f"{vk_format} component count {format_component_count}"
        )


def _validate_vulkan_custom_shader_payload(
    domain_name: str,
    implementation_attrs: list[int],
    inputs: list[TosaArg],
    output: TosaArg,
) -> None:
    if domain_name != _VULKAN_CUSTOM_SHADER_DOMAIN:
        return

    if not implementation_attrs:
        raise ValueError(
            "Vulkan custom shader tosa.CUSTOM requires non-empty JSON "
            "implementation_attrs"
        )

    payload = json.loads(bytes(implementation_attrs).decode("utf-8"))

    for input_idx, input_arg in enumerate(inputs):
        if payload.get(f"input_{input_idx}_type") != "Image":
            continue
        vk_format = payload.get(f"input_{input_idx}_vkformat")
        if not isinstance(vk_format, str):
            raise ValueError(f"Missing input_{input_idx}_vkformat for image input")
        _validate_image_tensor_arg(input_arg, f"input_{input_idx}", vk_format)

    if payload.get("output_0_type") == "Image":
        vk_format = payload.get("output_0_vkformat")
        if not isinstance(vk_format, str):
            raise ValueError("Missing output_0_vkformat for image output")
        _validate_image_tensor_arg(output, "output_0", vk_format)


@register_node_visitor
class CustomVisitor(NodeVisitor):
    """Lower the TOSA CUSTOM op from the TOSA backend dialect."""

    target = "tosa.CUSTOM.default"

    def define_node(
        self,
        node: torch.fx.Node,
        tosa_graph: Any,
        inputs: List[TosaArg],
        output: TosaArg,
    ) -> None:
        allowed_kwargs = {"operator_name", "domain_name", "implementation_attrs"}
        unexpected = set(node.kwargs.keys()) - allowed_kwargs
        if unexpected:
            raise ValueError(
                f"tosa.CUSTOM received unexpected kwargs: {sorted(unexpected)}"
            )

        operator_name = node.kwargs.get("operator_name")
        domain_name = node.kwargs.get("domain_name")
        implementation_attrs = node.kwargs.get("implementation_attrs")

        if operator_name is None or domain_name is None:
            raise ValueError(
                "tosa.CUSTOM requires operator_name and domain_name in kwargs"
            )
        if not isinstance(operator_name, str) or not isinstance(domain_name, str):
            raise TypeError(
                "tosa.CUSTOM requires operator_name and domain_name to be strings"
            )

        if implementation_attrs is None:
            impl_list = []
        elif isinstance(implementation_attrs, list):
            # NOTE: PyTorch schemas do not support a bytes type; we pass
            # implementation_attrs as int[] representing raw bytes.
            impl_list = [int(x) for x in implementation_attrs]
        else:
            raise TypeError(
                "implementation_attrs must be None or list[int]; "
                f"got {type(implementation_attrs)}"
            )

        expanded = [TosaArg(item, self.tosa_spec) for item in inputs[0].special]
        _validate_vulkan_custom_shader_payload(
            domain_name=domain_name,
            implementation_attrs=impl_list,
            inputs=expanded,
            output=output,
        )

        attr = ts.TosaSerializerAttribute()
        attr.CustomAttribute(
            operator_name=operator_name,
            domain_name=domain_name,
            implementation_attrs=impl_list,
        )

        input_names = [arg.name for arg in expanded]
        output_names = (
            output.multiple_output_names
            if getattr(output, "multiple_output_names", None)
            else [output.name]
        )
        if len(output_names) != 1:
            # TODO: Support multi-output CUSTOM ops with per-output meta/shape.
            raise ValueError(
                f"tosa.CUSTOM currently requires a single output, got {len(output_names)}"
            )
        self._serialize_operator(
            node,
            tosa_graph,
            ts.Op.CUSTOM,
            input_names,
            output_names,
            attr,
        )
