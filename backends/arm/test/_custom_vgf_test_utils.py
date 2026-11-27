# Copyright 2026 Arm Limited and/or its affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

import base64
import json
import operator
import subprocess  # nosec B404 - required to invoke trusted local shader tool
from collections.abc import Callable
from pathlib import Path
from typing import Optional

import torch
import torch.nn.functional as F
from executorch.backends.arm._passes.arm_pass import ArmPass
from executorch.backends.arm.constants import NHWC_INVERSE_ORDER, NHWC_ORDER
from executorch.backends.arm.tosa.dialect.ops.custom import (
    has_fake_tosa_impl,
    register_fake_tosa,
)
from executorch.exir.dialects._ops import ops as exir_ops
from torch.fx.passes.infra.pass_base import PassResult
from torch.fx.passes.shape_prop import _extract_tensor_metadata
from torch.library import impl, register_fake

TEST_SHADER_NAMESPACE = "arm_test_vulkan_custom_shader"
TEST_SHADER_DOMAIN = "com.arm.VulkanCustomShader"
TEST_GRID_SAMPLE_OPERATOR = "torch.nn.functional.grid_sample"
TEST_GRID_READ_TENSOR_OPERATOR = "arm.test.grid_read_tensor_debug"
TEST_ADD_OPERATOR = "torch.add"

THREES_NAMESPACE = "arm_test_shader_ops"
THREES_DOMAIN = "com.arm.VulkanCustomShader"
THREES_OPERATOR = "arm.test.threes"
THREES_IMAGE_PACKED_OPERATOR = "arm.test.threes_image_packed"
IDENTITY_OPERATOR = "arm.test.identity"
IDENTITY_IMAGE_PACKED_OPERATOR = "arm.test.identity_image_packed"

_TEST_SHADER_LIB: Optional[torch.library.Library] = None
_TEST_THREES_LIB: Optional[torch.library.Library] = None
_TEST_SHADER_REGISTERED = False
_TEST_THREES_REGISTERED = False
_GRID_SAMPLE_TOSA_FAKE_IMPLS: dict[
    bool,
    Callable[[list[torch.Tensor], str, str, list[int]], list[torch.Tensor]],
] = {}
_ADD_TOSA_FAKE_IMPL: (
    Callable[[list[torch.Tensor], str, str, list[int]], list[torch.Tensor]] | None
) = None
_THREES_TOSA_FAKE_IMPLS: dict[
    str,
    Callable[[list[torch.Tensor], str, str, list[int]], list[torch.Tensor]],
] = {}

_ASSET_DIR = Path(__file__).resolve().parent / "assets"


def _set_fake_tensor_meta(node: torch.fx.Node, value) -> None:
    node.meta["val"] = value
    if isinstance(value, list):
        if value:
            node.meta["tensor_meta"] = _extract_tensor_metadata(value[0])
    else:
        node.meta["tensor_meta"] = _extract_tensor_metadata(value)


def _decode_payload_attrs(implementation_attrs: list[int]) -> dict[str, object]:
    return json.loads(bytes(implementation_attrs).decode("utf-8"))


def _grid_sample_tosa_fake(
    inputs: list[torch.Tensor],
    implementation_attrs: list[int],
) -> torch.Tensor:
    input_tensor, grid = inputs
    payload = _decode_payload_attrs(implementation_attrs)
    if payload.get("input_0_type") == "Image":
        return torch.empty(
            (
                input_tensor.shape[0],
                grid.shape[1],
                grid.shape[2],
                input_tensor.shape[-1],
            ),
            dtype=input_tensor.dtype,
            device=input_tensor.device,
        )
    return torch.empty(
        (
            input_tensor.shape[0],
            input_tensor.shape[1],
            grid.shape[1],
            grid.shape[2],
        ),
        dtype=input_tensor.dtype,
        device=input_tensor.device,
    )


def _grid_read_tensor_tosa_fake(inputs: list[torch.Tensor]) -> torch.Tensor:
    _, grid = inputs
    return torch.empty(
        (grid.shape[0], grid.shape[3], grid.shape[1], grid.shape[2]),
        dtype=grid.dtype,
        device=grid.device,
    )


def _compile_glsl_to_spirv(shader_name: str) -> bytes:
    result = (
        subprocess.run(  # nosec B603, B607 - trusted local tool with fixed arguments
            [
                "glslc",
                "-fshader-stage=compute",
                "-o",
                "-",
                str(_ASSET_DIR / shader_name),
            ],
            check=True,
            stdout=subprocess.PIPE,
        )
    )
    return result.stdout


def register_test_shader_library_ops() -> None:  # noqa: C901
    global _TEST_SHADER_LIB, _TEST_SHADER_REGISTERED, _GRID_SAMPLE_TOSA_FAKE_IMPLS, _ADD_TOSA_FAKE_IMPL
    if _TEST_SHADER_REGISTERED:
        return

    _TEST_SHADER_LIB = torch.library.Library(TEST_SHADER_NAMESPACE, "DEF")
    lib = _TEST_SHADER_LIB
    lib.define(
        "grid_sample(Tensor input, Tensor grid, str? mode=None, "
        "str? padding_mode=None, bool? align_corners=None) -> Tensor"
    )
    lib.define(
        "grid_sample_buffer_debug(Tensor input, Tensor grid, str? mode=None, "
        "str? padding_mode=None, bool? align_corners=None) -> Tensor"
    )
    lib.define(
        "grid_sample_buffer_nchw_debug(Tensor input, Tensor grid, str? mode=None, "
        "str? padding_mode=None, bool? align_corners=None) -> Tensor"
    )
    lib.define(
        "grid_read_tensor_debug(Tensor input, Tensor grid, str? mode=None, "
        "str? padding_mode=None, bool? align_corners=None) -> Tensor"
    )
    lib.define("add(Tensor a, Tensor b) -> Tensor")

    @impl(lib, "grid_sample", dispatch_key="CompositeExplicitAutograd")
    def _grid_sample_impl(
        input: torch.Tensor,
        grid: torch.Tensor,
        mode: Optional[str] = None,
        padding_mode: Optional[str] = None,
        align_corners: Optional[bool] = None,
    ) -> torch.Tensor:
        return F.grid_sample(
            input,
            grid,
            mode=mode or "bilinear",
            padding_mode=padding_mode or "zeros",
            align_corners=align_corners,
        )

    @register_fake(f"{TEST_SHADER_NAMESPACE}::grid_sample")
    def _grid_sample_fake(
        input: torch.Tensor,
        grid: torch.Tensor,
        mode: Optional[str] = None,
        padding_mode: Optional[str] = None,
        align_corners: Optional[bool] = None,
    ) -> torch.Tensor:
        _ = (mode, padding_mode, align_corners)
        return torch.empty(
            (
                input.shape[0],
                input.shape[1],
                grid.shape[1],
                grid.shape[2],
            ),
            dtype=input.dtype,
            device=input.device,
        )

    @impl(lib, "grid_sample_buffer_debug", dispatch_key="CompositeExplicitAutograd")
    def _grid_sample_buffer_debug_impl(
        input: torch.Tensor,
        grid: torch.Tensor,
        mode: Optional[str] = None,
        padding_mode: Optional[str] = None,
        align_corners: Optional[bool] = None,
    ) -> torch.Tensor:
        return F.grid_sample(
            input,
            grid,
            mode=mode or "bilinear",
            padding_mode=padding_mode or "zeros",
            align_corners=align_corners,
        )

    @register_fake(f"{TEST_SHADER_NAMESPACE}::grid_sample_buffer_debug")
    def _grid_sample_buffer_debug_fake(
        input: torch.Tensor,
        grid: torch.Tensor,
        mode: Optional[str] = None,
        padding_mode: Optional[str] = None,
        align_corners: Optional[bool] = None,
    ) -> torch.Tensor:
        return _grid_sample_fake(
            input,
            grid,
            mode=mode,
            padding_mode=padding_mode,
            align_corners=align_corners,
        )

    @impl(
        lib, "grid_sample_buffer_nchw_debug", dispatch_key="CompositeExplicitAutograd"
    )
    def _grid_sample_buffer_nchw_debug_impl(
        input: torch.Tensor,
        grid: torch.Tensor,
        mode: Optional[str] = None,
        padding_mode: Optional[str] = None,
        align_corners: Optional[bool] = None,
    ) -> torch.Tensor:
        return F.grid_sample(
            input,
            grid,
            mode=mode or "bilinear",
            padding_mode=padding_mode or "zeros",
            align_corners=align_corners,
        ).contiguous()

    @register_fake(f"{TEST_SHADER_NAMESPACE}::grid_sample_buffer_nchw_debug")
    def _grid_sample_buffer_nchw_debug_fake(
        input: torch.Tensor,
        grid: torch.Tensor,
        mode: Optional[str] = None,
        padding_mode: Optional[str] = None,
        align_corners: Optional[bool] = None,
    ) -> torch.Tensor:
        _ = (mode, padding_mode, align_corners)
        return torch.empty(
            (input.shape[0], input.shape[1], grid.shape[1], grid.shape[2]),
            dtype=input.dtype,
            device=input.device,
        )

    @impl(lib, "grid_read_tensor_debug", dispatch_key="CompositeExplicitAutograd")
    def _grid_read_tensor_debug_impl(
        input: torch.Tensor,
        grid: torch.Tensor,
        mode: Optional[str] = None,
        padding_mode: Optional[str] = None,
        align_corners: Optional[bool] = None,
    ) -> torch.Tensor:
        _ = (input, mode, padding_mode, align_corners)
        return grid.permute(0, 3, 1, 2).contiguous()

    @register_fake(f"{TEST_SHADER_NAMESPACE}::grid_read_tensor_debug")
    def _grid_read_tensor_debug_fake(
        input: torch.Tensor,
        grid: torch.Tensor,
        mode: Optional[str] = None,
        padding_mode: Optional[str] = None,
        align_corners: Optional[bool] = None,
    ) -> torch.Tensor:
        _ = (input, mode, padding_mode, align_corners)
        return torch.empty(
            (grid.shape[0], grid.shape[3], grid.shape[1], grid.shape[2]),
            dtype=grid.dtype,
            device=grid.device,
        )

    @register_fake_tosa(f"{TEST_GRID_SAMPLE_OPERATOR}.align_corners.True")
    def _grid_sample_tosa_fake_true(
        inputs: list[torch.Tensor],
        operator_name: str,
        domain_name: str,
        implementation_attrs: list[int],
    ) -> list[torch.Tensor]:
        _ = implementation_attrs
        assert operator_name == f"{TEST_GRID_SAMPLE_OPERATOR}.align_corners.True"
        assert domain_name == TEST_SHADER_DOMAIN
        return [_grid_sample_tosa_fake(inputs, implementation_attrs)]

    @register_fake_tosa(f"{TEST_GRID_SAMPLE_OPERATOR}.align_corners.False")
    def _grid_sample_tosa_fake_false(
        inputs: list[torch.Tensor],
        operator_name: str,
        domain_name: str,
        implementation_attrs: list[int],
    ) -> list[torch.Tensor]:
        _ = implementation_attrs
        assert operator_name == f"{TEST_GRID_SAMPLE_OPERATOR}.align_corners.False"
        assert domain_name == TEST_SHADER_DOMAIN
        return [_grid_sample_tosa_fake(inputs, implementation_attrs)]

    _GRID_SAMPLE_TOSA_FAKE_IMPLS = {
        True: _grid_sample_tosa_fake_true,
        False: _grid_sample_tosa_fake_false,
    }

    @register_fake_tosa(TEST_GRID_READ_TENSOR_OPERATOR)
    def _grid_read_tensor_tosa_fake_impl(
        inputs: list[torch.Tensor],
        operator_name: str,
        domain_name: str,
        implementation_attrs: list[int],
    ) -> list[torch.Tensor]:
        _ = implementation_attrs
        assert operator_name == TEST_GRID_READ_TENSOR_OPERATOR
        assert domain_name == TEST_SHADER_DOMAIN
        return [_grid_read_tensor_tosa_fake(inputs)]

    @impl(lib, "add", dispatch_key="CompositeExplicitAutograd")
    def _add_impl(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        return a + b

    @register_fake(f"{TEST_SHADER_NAMESPACE}::add")
    def _add_fake(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        return torch.empty_like(a)

    @register_fake_tosa(TEST_ADD_OPERATOR)
    def _add_tosa_fake_impl(
        inputs: list[torch.Tensor],
        operator_name: str,
        domain_name: str,
        implementation_attrs: list[int],
    ) -> list[torch.Tensor]:
        _ = implementation_attrs
        assert operator_name == TEST_ADD_OPERATOR
        assert domain_name == TEST_SHADER_DOMAIN
        return [_add_fake(inputs[0], inputs[1])]

    _ADD_TOSA_FAKE_IMPL = _add_tosa_fake_impl
    _TEST_SHADER_REGISTERED = True


def register_test_threes_library_ops() -> None:  # noqa: C901
    global _TEST_THREES_LIB, _TEST_THREES_REGISTERED, _THREES_TOSA_FAKE_IMPLS
    if _TEST_THREES_REGISTERED:
        return

    _TEST_THREES_LIB = torch.library.Library(THREES_NAMESPACE, "DEF")
    lib = _TEST_THREES_LIB
    lib.define("threes(Tensor x) -> Tensor")
    lib.define("threes_image_packed(Tensor x) -> Tensor")
    lib.define("identity(Tensor x) -> Tensor")
    lib.define("identity_image_packed(Tensor x) -> Tensor")

    @impl(lib, "threes", dispatch_key="CompositeExplicitAutograd")
    def _threes_impl(x: torch.Tensor) -> torch.Tensor:
        return x * 3.0 + 33.0

    @register_fake(f"{THREES_NAMESPACE}::threes")
    def _threes_fake(x: torch.Tensor) -> torch.Tensor:
        return torch.empty_like(x)

    @impl(lib, "threes_image_packed", dispatch_key="CompositeExplicitAutograd")
    def _threes_image_packed_impl(x: torch.Tensor) -> torch.Tensor:
        return x * 3.0 + 33.0

    @register_fake(f"{THREES_NAMESPACE}::threes_image_packed")
    def _threes_image_packed_fake(x: torch.Tensor) -> torch.Tensor:
        return torch.empty_like(x)

    @impl(lib, "identity", dispatch_key="CompositeExplicitAutograd")
    def _identity_impl(x: torch.Tensor) -> torch.Tensor:
        return x

    @register_fake(f"{THREES_NAMESPACE}::identity")
    def _identity_fake(x: torch.Tensor) -> torch.Tensor:
        return torch.empty_like(x)

    @impl(lib, "identity_image_packed", dispatch_key="CompositeExplicitAutograd")
    def _identity_image_packed_impl(x: torch.Tensor) -> torch.Tensor:
        return x

    @register_fake(f"{THREES_NAMESPACE}::identity_image_packed")
    def _identity_image_packed_fake(x: torch.Tensor) -> torch.Tensor:
        return torch.empty_like(x)

    @register_fake_tosa(THREES_OPERATOR)
    def _threes_tosa_fake_impl(
        inputs: list[torch.Tensor],
        operator_name: str,
        domain_name: str,
        implementation_attrs: list[int],
    ) -> list[torch.Tensor]:
        _ = implementation_attrs
        assert operator_name == THREES_OPERATOR
        assert domain_name == THREES_DOMAIN
        return [_threes_fake(inputs[0])]

    @register_fake_tosa(THREES_IMAGE_PACKED_OPERATOR)
    def _threes_image_packed_tosa_fake_impl(
        inputs: list[torch.Tensor],
        operator_name: str,
        domain_name: str,
        implementation_attrs: list[int],
    ) -> list[torch.Tensor]:
        _ = implementation_attrs
        assert operator_name == THREES_IMAGE_PACKED_OPERATOR
        assert domain_name == THREES_DOMAIN
        return [_threes_image_packed_fake(inputs[0])]

    @register_fake_tosa(IDENTITY_OPERATOR)
    def _identity_tosa_fake_impl(
        inputs: list[torch.Tensor],
        operator_name: str,
        domain_name: str,
        implementation_attrs: list[int],
    ) -> list[torch.Tensor]:
        _ = implementation_attrs
        assert operator_name == IDENTITY_OPERATOR
        assert domain_name == THREES_DOMAIN
        return [_identity_fake(inputs[0])]

    @register_fake_tosa(IDENTITY_IMAGE_PACKED_OPERATOR)
    def _identity_image_packed_tosa_fake_impl(
        inputs: list[torch.Tensor],
        operator_name: str,
        domain_name: str,
        implementation_attrs: list[int],
    ) -> list[torch.Tensor]:
        _ = implementation_attrs
        assert operator_name == IDENTITY_IMAGE_PACKED_OPERATOR
        assert domain_name == THREES_DOMAIN
        return [_identity_image_packed_fake(inputs[0])]

    _THREES_TOSA_FAKE_IMPLS = {
        THREES_OPERATOR: _threes_tosa_fake_impl,
        THREES_IMAGE_PACKED_OPERATOR: _threes_image_packed_tosa_fake_impl,
        IDENTITY_OPERATOR: _identity_tosa_fake_impl,
        IDENTITY_IMAGE_PACKED_OPERATOR: _identity_image_packed_tosa_fake_impl,
    }
    _TEST_THREES_REGISTERED = True


def register_test_shader_partition_ops(partitioner) -> None:
    partitioner.register_custom_partition_op(
        torch.ops.arm_test_vulkan_custom_shader.grid_sample.default
    )
    partitioner.register_custom_partition_op(
        torch.ops.arm_test_vulkan_custom_shader.grid_sample_buffer_debug.default
    )
    partitioner.register_custom_partition_op(
        torch.ops.arm_test_vulkan_custom_shader.grid_sample_buffer_nchw_debug.default
    )
    partitioner.register_custom_partition_op(
        torch.ops.arm_test_vulkan_custom_shader.grid_read_tensor_debug.default
    )
    partitioner.register_custom_partition_op(
        torch.ops.arm_test_vulkan_custom_shader.add.default
    )


def register_test_threes_partition_ops(partitioner) -> None:
    partitioner.register_custom_partition_op(
        torch.ops.arm_test_shader_ops.threes.default
    )
    partitioner.register_custom_partition_op(
        torch.ops.arm_test_shader_ops.threes_image_packed.default
    )
    partitioner.register_custom_partition_op(
        torch.ops.arm_test_shader_ops.identity.default
    )
    partitioner.register_custom_partition_op(
        torch.ops.arm_test_shader_ops.identity_image_packed.default
    )


def rewrite_aten_grid_sample_to_test_shader(graph_module: torch.fx.GraphModule) -> bool:
    graph = graph_module.graph
    modified = False
    for node in list(graph.nodes):
        if node.op != "call_function" or "grid_sampler" not in str(node.target):
            continue
        input_tensor = node.args[0]
        grid = node.args[1]
        with graph.inserting_before(node):
            new_node = graph.call_function(
                torch.ops.arm_test_vulkan_custom_shader.grid_sample.default,
                args=(input_tensor, grid),
                kwargs={
                    "mode": node.kwargs.get("mode"),
                    "padding_mode": node.kwargs.get("padding_mode"),
                    "align_corners": node.kwargs.get("align_corners"),
                },
            )
            new_node.meta = dict(node.meta)
            input_val = input_tensor.meta["val"]
            grid_val = grid.meta["val"]
            _set_fake_tensor_meta(
                new_node,
                torch.empty(
                    (
                        input_val.shape[0],
                        input_val.shape[1],
                        grid_val.shape[1],
                        grid_val.shape[2],
                    ),
                    dtype=input_val.dtype,
                    device=input_val.device,
                ),
            )
        node.replace_all_uses_with(new_node)
        graph.erase_node(node)
        modified = True
    if modified:
        graph_module.recompile()
    return modified


def rewrite_aten_add_to_test_shader(graph_module: torch.fx.GraphModule) -> bool:
    graph = graph_module.graph
    modified = False
    for node in list(graph.nodes):
        if node.op != "call_function" or node.target != torch.ops.aten.add.Tensor:
            continue
        with graph.inserting_before(node):
            new_node = graph.call_function(
                torch.ops.arm_test_vulkan_custom_shader.add.default,
                args=node.args[:2],
                kwargs={},
            )
            new_node.meta = dict(node.meta)
        node.replace_all_uses_with(new_node)
        graph.erase_node(node)
        modified = True
    if modified:
        graph_module.recompile()
    return modified


class EncodeSamplerGridSampleToTosaCustomPass(ArmPass):
    _passes_required_after = set()

    @staticmethod
    def _infer_vkformat(input_node: torch.fx.Node, expect_nchw: bool) -> str:
        val = input_node.meta["val"]
        shape = tuple(val.shape)
        channels = int(shape[1] if expect_nchw else shape[-1])
        if val.dtype != torch.float32:
            raise RuntimeError(f"Unsupported dtype for vkformat: {val.dtype}")
        if channels == 1:
            return "VK_FORMAT_R32_SFLOAT"
        if channels == 2:
            return "VK_FORMAT_R32G32_SFLOAT"
        if channels == 4:
            return "VK_FORMAT_R32G32B32A32_SFLOAT"
        if channels == 3:
            raise ValueError(
                "Image-backed grid_sample requires 1, 2, or 4 channels; got 3"
            )
        raise RuntimeError(f"Unsupported channel count for grid_sample: {channels}")

    @staticmethod
    def _make_nhwc_fake(
        input_val: torch.Tensor,
        grid_val: torch.Tensor,
    ) -> torch.Tensor:
        return torch.empty(
            (
                input_val.shape[0],
                grid_val.shape[1],
                grid_val.shape[2],
                input_val.shape[1],
            ),
            dtype=input_val.dtype,
            device=input_val.device,
        )

    def call(self, graph_module):  # noqa: C901
        graph = graph_module.graph
        modified = False
        for node in list(graph.nodes):
            if node.op != "call_function":
                continue
            target_name = str(node.target)
            if (
                "arm_test_vulkan_custom_shader.grid_sample" not in target_name
                and "arm_test_vulkan_custom_shader.grid_read_tensor_debug"
                not in target_name
            ):
                continue

            input_tensor, grid = node.args[:2]
            mode = node.kwargs.get("mode") or "bilinear"
            padding_mode = node.kwargs.get("padding_mode") or "zeros"
            align_corners = node.kwargs.get("align_corners")

            sampler = {}
            if mode == "bilinear":
                sampler["mag_filter"] = "VK_FILTER_LINEAR"
                sampler["min_filter"] = "VK_FILTER_LINEAR"
            elif mode == "nearest":
                sampler["mag_filter"] = "VK_FILTER_NEAREST"
                sampler["min_filter"] = "VK_FILTER_NEAREST"
            elif mode == "bicubic":
                sampler["mag_filter"] = "VK_FILTER_LINEAR"
                sampler["min_filter"] = "VK_FILTER_LINEAR"
            else:
                raise RuntimeError(f"Unsupported grid_sample mode: {mode}")

            if padding_mode == "zeros":
                sampler["address_mode_u"] = "VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_BORDER"
                sampler["address_mode_v"] = "VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_BORDER"
                sampler["border_color"] = "VK_BORDER_COLOR_FLOAT_TRANSPARENT_BLACK"
            elif padding_mode == "border":
                sampler["address_mode_u"] = "VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE"
                sampler["address_mode_v"] = "VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE"
            elif padding_mode == "reflection":
                sampler["address_mode_u"] = "VK_SAMPLER_ADDRESS_MODE_MIRRORED_REPEAT"
                sampler["address_mode_v"] = "VK_SAMPLER_ADDRESS_MODE_MIRRORED_REPEAT"
            else:
                raise RuntimeError(
                    f"Unsupported grid_sample padding_mode: {padding_mode}"
                )

            shader_name = "test_grid_sample_sampler.glsl"
            input_type = "Image"
            input_descriptor_type = "VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER"
            input_vkformat = self._infer_vkformat(input_tensor, expect_nchw=True)
            output_type = "Image"
            output_descriptor_type = "VK_DESCRIPTOR_TYPE_STORAGE_IMAGE"
            output_vkformat = self._infer_vkformat(input_tensor, expect_nchw=True)
            include_sampler = True
            if "grid_sample_buffer_nchw_debug" in target_name:
                shader_name = "test_grid_sample_buffer_nchw_debug.glsl"
                input_type = "Buffer"
                input_descriptor_type = "VK_DESCRIPTOR_TYPE_STORAGE_BUFFER"
                input_vkformat = "VK_FORMAT_R32_SFLOAT"
                output_type = "Buffer"
                output_descriptor_type = "VK_DESCRIPTOR_TYPE_STORAGE_BUFFER"
                output_vkformat = "VK_FORMAT_R32_SFLOAT"
                include_sampler = False
            elif "grid_read_tensor_debug" in target_name:
                shader_name = "test_grid_read_tensor_debug.glsl"
                input_type = "Buffer"
                input_descriptor_type = "VK_DESCRIPTOR_TYPE_STORAGE_BUFFER"
                input_vkformat = "VK_FORMAT_R32_SFLOAT"
                output_type = "Buffer"
                output_descriptor_type = "VK_DESCRIPTOR_TYPE_STORAGE_BUFFER"
                output_vkformat = "VK_FORMAT_R32_SFLOAT"
                include_sampler = False
            elif "grid_sample_buffer_debug" in target_name:
                shader_name = "test_grid_sample_sampler_buffer_debug.glsl"
                output_type = "Buffer"
                output_descriptor_type = "VK_DESCRIPTOR_TYPE_STORAGE_BUFFER"
                output_vkformat = "VK_FORMAT_R32_SFLOAT"
            payload = {
                "entry_point": "main",
                "workgroup_sizes": [8, 8, 1],
                "is_vkshader": True,
                "shader_code": base64.b64encode(
                    _compile_glsl_to_spirv(shader_name)
                ).decode("ascii"),
                "shader_language": "SPIR-V",
                "push_constants": "",
                "input_0_binding": 0,
                "input_1_binding": 1,
                "output_0_binding": 2,
                "input_0_vkdescriptortype": input_descriptor_type,
                "input_1_vkdescriptortype": "VK_DESCRIPTOR_TYPE_TENSOR_ARM",
                "output_0_vkdescriptortype": output_descriptor_type,
                "input_0_descriptorset": 0,
                "input_1_descriptorset": 0,
                "output_0_descriptorset": 0,
                "input_0_type": input_type,
                "input_1_type": "Tensor",
                "output_0_type": output_type,
                "input_0_vkformat": input_vkformat,
                "input_1_vkformat": "VK_FORMAT_R32_SFLOAT",
                "output_0_vkformat": output_vkformat,
            }
            if include_sampler:
                payload["input_0_sampler"] = sampler
            implementation_attrs = list(json.dumps(payload).encode("utf-8"))
            operator_name = (
                TEST_GRID_READ_TENSOR_OPERATOR
                if "grid_read_tensor_debug" in target_name
                else f"{TEST_GRID_SAMPLE_OPERATOR}.align_corners.{align_corners is True}"
            )

            if not has_fake_tosa_impl(operator_name):
                raise RuntimeError(
                    f"tosa.CUSTOM fake impl is not registered for {operator_name}"
                )

            with graph.inserting_before(node):
                use_nhwc_shader_contract = (
                    "grid_sample_buffer_nchw_debug" not in target_name
                    and "grid_read_tensor_debug" not in target_name
                )
                custom_input = input_tensor
                if use_nhwc_shader_contract:
                    custom_input = graph.call_function(
                        exir_ops.edge.aten.permute_copy.default,
                        args=(input_tensor, list(NHWC_ORDER)),
                        kwargs={},
                    )
                    custom_input.meta = dict(input_tensor.meta)
                    _set_fake_tensor_meta(
                        custom_input,
                        exir_ops.edge.aten.permute_copy.default(
                            input_tensor.meta["val"], list(NHWC_ORDER)
                        ),
                    )

                tosa_custom = graph.call_function(
                    exir_ops.backend.tosa.CUSTOM.default,
                    args=([custom_input, grid],),
                    kwargs={
                        "operator_name": operator_name,
                        "domain_name": TEST_SHADER_DOMAIN,
                        "implementation_attrs": implementation_attrs,
                    },
                )
                if (
                    "grid_sample_buffer_nchw_debug" in target_name
                    or "grid_read_tensor_debug" in target_name
                ):
                    grid_val = grid.meta["val"]
                    if "grid_read_tensor_debug" in target_name:
                        fake_outputs = [
                            torch.empty(
                                (
                                    grid_val.shape[0],
                                    grid_val.shape[3],
                                    grid_val.shape[1],
                                    grid_val.shape[2],
                                ),
                                dtype=grid_val.dtype,
                                device=grid_val.device,
                            )
                        ]
                    else:
                        input_val = input_tensor.meta["val"]
                        fake_outputs = [
                            torch.empty(
                                (
                                    input_val.shape[0],
                                    input_val.shape[1],
                                    grid_val.shape[1],
                                    grid_val.shape[2],
                                ),
                                dtype=input_val.dtype,
                                device=input_val.device,
                            )
                        ]
                else:
                    fake_outputs = [
                        self._make_nhwc_fake(input_tensor.meta["val"], grid.meta["val"])
                    ]
                tosa_custom.meta = dict(node.meta)
                _set_fake_tensor_meta(tosa_custom, fake_outputs)
                custom_output = graph.call_function(
                    operator.getitem, args=(tosa_custom, 0), kwargs={}
                )
                custom_output.meta = dict(node.meta)
                _set_fake_tensor_meta(custom_output, fake_outputs[0])

                if use_nhwc_shader_contract:
                    output = graph.call_function(
                        exir_ops.edge.aten.permute_copy.default,
                        args=(custom_output, list(NHWC_INVERSE_ORDER)),
                        kwargs={},
                    )
                    output.meta = dict(node.meta)
                    _set_fake_tensor_meta(
                        output,
                        exir_ops.edge.aten.permute_copy.default(
                            custom_output.meta["val"], list(NHWC_INVERSE_ORDER)
                        ),
                    )
                else:
                    output = custom_output

            node.replace_all_uses_with(output)
            graph.erase_node(node)
            modified = True

        if modified:
            graph_module.recompile()
        return PassResult(graph_module, modified)


class EncodeTestAddToTosaCustomPass(ArmPass):
    _passes_required_after = set()

    def call(self, graph_module):
        graph = graph_module.graph
        modified = False
        for node in list(graph.nodes):
            if node.op != "call_function":
                continue
            if "arm_test_vulkan_custom_shader.add" not in str(node.target):
                continue

            a, b = node.args[:2]
            payload = {
                "entry_point": "main",
                "workgroup_sizes": [64, 1, 1],
                "is_vkshader": True,
                "shader_code": base64.b64encode(
                    _compile_glsl_to_spirv("test_add_buffer.glsl")
                ).decode("ascii"),
                "shader_language": "SPIR-V",
                "push_constants": "",
                "input_0_binding": 0,
                "input_1_binding": 1,
                "output_0_binding": 2,
                "input_0_type": "Buffer",
                "input_1_type": "Buffer",
                "output_0_type": "Buffer",
                "input_0_vkdescriptortype": "VK_DESCRIPTOR_TYPE_STORAGE_BUFFER",
                "input_1_vkdescriptortype": "VK_DESCRIPTOR_TYPE_STORAGE_BUFFER",
                "output_0_vkdescriptortype": "VK_DESCRIPTOR_TYPE_STORAGE_BUFFER",
                "input_0_descriptorset": 0,
                "input_1_descriptorset": 0,
                "output_0_descriptorset": 0,
                "input_0_vkformat": "VK_FORMAT_R32_SFLOAT",
                "input_1_vkformat": "VK_FORMAT_R32_SFLOAT",
                "output_0_vkformat": "VK_FORMAT_R32_SFLOAT",
            }
            implementation_attrs = list(json.dumps(payload).encode("utf-8"))
            with graph.inserting_before(node):
                tosa_custom = graph.call_function(
                    exir_ops.backend.tosa.CUSTOM.default,
                    args=([a, b],),
                    kwargs={
                        "operator_name": TEST_ADD_OPERATOR,
                        "domain_name": TEST_SHADER_DOMAIN,
                        "implementation_attrs": implementation_attrs,
                    },
                )
                add_tosa_fake_impl = _ADD_TOSA_FAKE_IMPL
                assert add_tosa_fake_impl is not None
                fake_outputs = add_tosa_fake_impl(
                    [a.meta["val"], b.meta["val"]],
                    TEST_ADD_OPERATOR,
                    TEST_SHADER_DOMAIN,
                    implementation_attrs,
                )
                tosa_custom.meta = dict(node.meta)
                _set_fake_tensor_meta(tosa_custom, fake_outputs)
                output = graph.call_function(
                    operator.getitem, args=(tosa_custom, 0), kwargs={}
                )
                output.meta = dict(node.meta)
                _set_fake_tensor_meta(output, fake_outputs[0])

            node.replace_all_uses_with(output)
            graph.erase_node(node)
            modified = True

        if modified:
            graph_module.recompile()
        return PassResult(graph_module, modified)


class EncodeThreesToTosaCustomPass(ArmPass):
    _passes_required_after = set()

    @staticmethod
    def _make_nhwc_fake(input_val: torch.Tensor) -> torch.Tensor:
        return torch.empty(
            (
                input_val.shape[0],
                input_val.shape[2],
                input_val.shape[3],
                input_val.shape[1],
            ),
            dtype=input_val.dtype,
            device=input_val.device,
        )

    def call(self, graph_module):
        graph = graph_module.graph
        modified = False
        for node in list(graph.nodes):
            if node.op != "call_function":
                continue
            target_name = str(node.target)
            if (
                "arm_test_shader_ops.threes" not in target_name
                and "arm_test_shader_ops.identity" not in target_name
            ):
                continue

            (x,) = node.args[:1]
            operator_name = THREES_OPERATOR
            shader_name = "test_threes_buffer.glsl"
            use_nhwc_shader_contract = False
            if "threes_image_packed" in target_name:
                operator_name = THREES_IMAGE_PACKED_OPERATOR
                use_nhwc_shader_contract = True
            elif "identity_image_packed" in target_name:
                operator_name = IDENTITY_IMAGE_PACKED_OPERATOR
                shader_name = "test_identity_buffer.glsl"
                use_nhwc_shader_contract = True
            elif "identity" in target_name:
                operator_name = IDENTITY_OPERATOR
                shader_name = "test_identity_buffer.glsl"
            payload = {
                "entry_point": "main",
                "workgroup_sizes": [64, 1, 1],
                "is_vkshader": True,
                "shader_code": base64.b64encode(
                    _compile_glsl_to_spirv(shader_name)
                ).decode("ascii"),
                "shader_language": "SPIR-V",
                "push_constants": "",
                "input_0_binding": 0,
                "output_0_binding": 1,
                "input_0_type": "Buffer",
                "output_0_type": "Buffer",
                "input_0_vkdescriptortype": "VK_DESCRIPTOR_TYPE_STORAGE_BUFFER",
                "output_0_vkdescriptortype": "VK_DESCRIPTOR_TYPE_STORAGE_BUFFER",
                "input_0_descriptorset": 0,
                "output_0_descriptorset": 0,
                "input_0_vkformat": "VK_FORMAT_R32_SFLOAT",
                "output_0_vkformat": "VK_FORMAT_R32_SFLOAT",
            }
            implementation_attrs = list(json.dumps(payload).encode("utf-8"))

            with graph.inserting_before(node):
                custom_input = x
                if use_nhwc_shader_contract:
                    custom_input = graph.call_function(
                        exir_ops.edge.aten.permute_copy.default,
                        args=(x, list(NHWC_ORDER)),
                        kwargs={},
                    )
                    custom_input.meta = dict(x.meta)
                    _set_fake_tensor_meta(
                        custom_input,
                        exir_ops.edge.aten.permute_copy.default(
                            x.meta["val"], list(NHWC_ORDER)
                        ),
                    )

                tosa_custom = graph.call_function(
                    exir_ops.backend.tosa.CUSTOM.default,
                    args=([custom_input],),
                    kwargs={
                        "operator_name": operator_name,
                        "domain_name": THREES_DOMAIN,
                        "implementation_attrs": implementation_attrs,
                    },
                )
                if use_nhwc_shader_contract:
                    fake_outputs = [self._make_nhwc_fake(x.meta["val"])]
                else:
                    fake_outputs = _THREES_TOSA_FAKE_IMPLS[operator_name](
                        [x.meta["val"]],
                        operator_name,
                        THREES_DOMAIN,
                        implementation_attrs,
                    )
                tosa_custom.meta = dict(node.meta)
                _set_fake_tensor_meta(tosa_custom, fake_outputs)
                custom_output = graph.call_function(
                    operator.getitem, args=(tosa_custom, 0), kwargs={}
                )
                custom_output.meta = dict(node.meta)
                _set_fake_tensor_meta(custom_output, fake_outputs[0])

                if use_nhwc_shader_contract:
                    output = graph.call_function(
                        exir_ops.edge.aten.permute_copy.default,
                        args=(custom_output, list(NHWC_INVERSE_ORDER)),
                        kwargs={},
                    )
                    output.meta = dict(node.meta)
                    _set_fake_tensor_meta(
                        output,
                        exir_ops.edge.aten.permute_copy.default(
                            custom_output.meta["val"], list(NHWC_INVERSE_ORDER)
                        ),
                    )
                else:
                    output = custom_output

            node.replace_all_uses_with(output)
            graph.erase_node(node)
            modified = True

        if modified:
            graph_module.recompile()
        return PassResult(graph_module, modified)
