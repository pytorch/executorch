# Copyright 2026 Arm Limited and/or its affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import sys
from pathlib import Path

import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[4]))

from backends.arm.test.runtime._vgf_runtime_test_utils import (
    alias_groups,
    lower_sampler_and_threes_vgf,
    lower_sampler_vgf,
    lower_threes_vgf,
    make_sampler_probe_inputs,
    segment_types,
    xfail_if_legacy_model_converter_release,
)
from executorch.backends.arm.test import common

pytestmark = xfail_if_legacy_model_converter_release()


def _has_alias_pair(vgf_json: dict, lhs: str, rhs: str) -> bool:
    for group in alias_groups(vgf_json).values():
        descriptor_types = {resource["vk_descriptor_type"] for resource in group}
        if {lhs, rhs}.issubset(descriptor_types):
            return True
    return False


def _has_alias_relations(vgf_json: dict, lhs: str, bridge: str, rhs: str) -> bool:
    return _has_alias_pair(vgf_json, lhs, bridge) and _has_alias_pair(
        vgf_json, bridge, rhs
    )


class _ComputeComputeThrees(torch.nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = torch.ops.arm_test_shader_ops.threes.default(x)
        return torch.ops.arm_test_shader_ops.threes.default(y)


class _GraphComputeComputeThrees(torch.nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x * 2.0
        y = torch.ops.arm_test_shader_ops.threes.default(x)
        return torch.ops.arm_test_shader_ops.threes.default(y)


class _ComputeGraphGraphThrees(torch.nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = torch.ops.arm_test_shader_ops.threes.default(x)
        y = y * 0.5
        return y * 2.0


class _GraphThenThrees(torch.nn.Module):
    def forward(self, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        return torch.ops.arm_test_shader_ops.threes.default(a + b)


class _GraphThenSampler(torch.nn.Module):
    def forward(self, x: torch.Tensor, grid: torch.Tensor) -> torch.Tensor:
        x = x * 2.0 + 1.0
        return F.grid_sample(
            x,
            grid,
            mode="bilinear",
            padding_mode="zeros",
            align_corners=False,
        )


class _SamplerThenGraph(torch.nn.Module):
    def forward(self, x: torch.Tensor, grid: torch.Tensor) -> torch.Tensor:
        y = F.grid_sample(
            x,
            grid,
            mode="bilinear",
            padding_mode="zeros",
            align_corners=False,
        )
        return y * 0.5 + 3.0


class _SamplerThenSampler(torch.nn.Module):
    def forward(
        self, x: torch.Tensor, grid0: torch.Tensor, grid1: torch.Tensor
    ) -> torch.Tensor:
        y = F.grid_sample(
            x,
            grid0,
            mode="bilinear",
            padding_mode="zeros",
            align_corners=False,
        )
        return F.grid_sample(
            y,
            grid1,
            mode="bilinear",
            padding_mode="zeros",
            align_corners=False,
        )


class _SamplerThenThrees(torch.nn.Module):
    def forward(self, x: torch.Tensor, grid: torch.Tensor) -> torch.Tensor:
        y = F.grid_sample(
            x,
            grid,
            mode="bilinear",
            padding_mode="zeros",
            align_corners=False,
        )
        return torch.ops.arm_test_shader_ops.threes.default(y)


class _IdentityBufferOnly(torch.nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.ops.arm_test_shader_ops.identity.default(x)


class _IdentitySampler(torch.nn.Module):
    def forward(self, x: torch.Tensor, grid: torch.Tensor) -> torch.Tensor:
        return F.grid_sample(
            x,
            grid,
            mode="bilinear",
            padding_mode="zeros",
            align_corners=False,
        )


class _IdentitySamplerBufferDebug(torch.nn.Module):
    def forward(self, x: torch.Tensor, grid: torch.Tensor) -> torch.Tensor:
        return torch.ops.arm_test_vulkan_custom_shader.grid_sample_buffer_debug.default(
            x,
            grid,
            mode="bilinear",
            padding_mode="zeros",
            align_corners=False,
        )


class _IdentitySamplerBufferNchwDebug(torch.nn.Module):
    def forward(self, x: torch.Tensor, grid: torch.Tensor) -> torch.Tensor:
        return torch.ops.arm_test_vulkan_custom_shader.grid_sample_buffer_nchw_debug.default(
            x,
            grid,
            mode="bilinear",
            padding_mode="zeros",
            align_corners=False,
        )


class _GridReadTensorDebug(torch.nn.Module):
    def forward(self, x: torch.Tensor, grid: torch.Tensor) -> torch.Tensor:
        return torch.ops.arm_test_vulkan_custom_shader.grid_read_tensor_debug.default(
            x,
            grid,
            mode="bilinear",
            padding_mode="zeros",
            align_corners=False,
        )


class _IdentityPackedThenSamplerBufferDebug(torch.nn.Module):
    def forward(self, x: torch.Tensor, grid: torch.Tensor) -> torch.Tensor:
        y = torch.ops.arm_test_shader_ops.identity_image_packed.default(x)
        return torch.ops.arm_test_vulkan_custom_shader.grid_sample_buffer_debug.default(
            y,
            grid,
            mode="bilinear",
            padding_mode="zeros",
            align_corners=False,
        )


class _IdentityBufferThenSamplerBufferNchwDebug(torch.nn.Module):
    def forward(self, x: torch.Tensor, grid: torch.Tensor) -> torch.Tensor:
        y = torch.ops.arm_test_shader_ops.identity.default(x)
        return torch.ops.arm_test_vulkan_custom_shader.grid_sample_buffer_nchw_debug.default(
            y,
            grid,
            mode="bilinear",
            padding_mode="zeros",
            align_corners=False,
        )


class _IdentityPackedThenSampler(torch.nn.Module):
    def forward(self, x: torch.Tensor, grid: torch.Tensor) -> torch.Tensor:
        y = torch.ops.arm_test_shader_ops.identity_image_packed.default(x)
        return F.grid_sample(
            y,
            grid,
            mode="bilinear",
            padding_mode="zeros",
            align_corners=False,
        )


class _ThreesThenSampler(torch.nn.Module):
    def forward(self, x: torch.Tensor, grid: torch.Tensor) -> torch.Tensor:
        y = torch.ops.arm_test_shader_ops.threes.default(x)
        return F.grid_sample(
            y,
            grid,
            mode="bilinear",
            padding_mode="zeros",
            align_corners=False,
        )


# Covers a pure compute-to-compute path using two unary buffer-backed custom shader stages.
# Checks the lowered VGF contains two compute segments and runtime output matches eager execution within runtime tolerance.
@common.SkipIfNoModelConverter
def test_compute_compute_sequence_executes(tmp_path):
    x = torch.randn(256)
    expected, actual, vgf_json = lower_threes_vgf(
        _ComputeComputeThrees(), (x,), tmp_path
    )

    assert torch.allclose(expected, actual, atol=1e-4, rtol=0.0)
    assert segment_types(vgf_json) == ["COMPUTE", "COMPUTE"]


# Covers a graph-to-compute-to-compute flow with a graph op before two unary custom shader stages.
# Checks the lowered VGF contains graph then compute then compute and runtime output matches eager execution within runtime tolerance.
@common.SkipIfNoModelConverter
def test_graph_compute_compute_sequence_executes(tmp_path):
    x = torch.randn(256)
    expected, actual, vgf_json = lower_threes_vgf(
        _GraphComputeComputeThrees(), (x,), tmp_path
    )

    assert torch.allclose(expected, actual, atol=1e-4, rtol=0.0)
    assert segment_types(vgf_json) == ["GRAPH", "COMPUTE", "COMPUTE"]


# Covers a unary compute flow followed by two graph ops in the source graph.
# Checks runtime output matches eager execution and that VGF emits graph segments around the compute stage for constants and tail graph work.
@common.SkipIfNoModelConverter
def test_compute_graph_graph_sequence_executes(tmp_path):
    x = torch.randn(256)
    expected, actual, vgf_json = lower_threes_vgf(
        _ComputeGraphGraphThrees(), (x,), tmp_path
    )

    assert torch.allclose(expected, actual, atol=1e-4, rtol=0.0)
    assert segment_types(vgf_json) == ["GRAPH", "COMPUTE", "GRAPH"]


# Covers the tensor/storage-buffer alias handoff used by graph-to-buffer custom shader execution.
# Checks a single alias group contains both tensor and storage-buffer descriptors.
@common.SkipIfNoModelConverter
def test_tensor_storage_buffer_alias_pair(tmp_path):
    a = torch.randn(256)
    b = torch.randn(256)
    _, _, vgf_json = lower_threes_vgf(_GraphThenThrees(), (a, b), tmp_path)

    assert _has_alias_pair(
        vgf_json,
        "VK_DESCRIPTOR_TYPE_TENSOR_ARM",
        "VK_DESCRIPTOR_TYPE_STORAGE_BUFFER",
    )


# Covers the tensor/combined-image-sampler alias handoff used by graph-to-sampler execution.
# Checks a single alias group contains both tensor and combined-image-sampler descriptors.
@common.SkipIfNoModelConverter
def test_tensor_combined_image_sampler_alias_pair(tmp_path):
    x, grid = make_sampler_probe_inputs()
    _, _, vgf_json = lower_sampler_vgf(_GraphThenSampler(), (x, grid), tmp_path)

    assert _has_alias_pair(
        vgf_json,
        "VK_DESCRIPTOR_TYPE_TENSOR_ARM",
        "VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER",
    )


# Covers the tensor/storage-image alias handoff used by shader-to-graph execution.
# Checks a single alias group contains both tensor and storage-image descriptors.
@common.SkipIfNoModelConverter
def test_tensor_storage_image_alias_pair(tmp_path):
    x, grid = make_sampler_probe_inputs()
    _, _, vgf_json = lower_sampler_vgf(_SamplerThenGraph(), (x, grid), tmp_path)

    assert _has_alias_pair(
        vgf_json,
        "VK_DESCRIPTOR_TYPE_TENSOR_ARM",
        "VK_DESCRIPTOR_TYPE_STORAGE_IMAGE",
    )


# Covers the storage-image/combined-image-sampler alias handoff across consecutive sampler stages.
# Checks a single alias group contains both storage-image and combined-image-sampler descriptors.
@common.SkipIfNoModelConverter
def test_storage_image_combined_image_sampler_alias_pair(tmp_path):
    x, grid0 = make_sampler_probe_inputs()
    grid1 = grid0.clone()
    _, actual, vgf_json = lower_sampler_vgf(
        _SamplerThenSampler(), (x, grid0, grid1), tmp_path
    )

    assert actual is not None
    assert _has_alias_pair(
        vgf_json,
        "VK_DESCRIPTOR_TYPE_STORAGE_IMAGE",
        "VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER",
    )


# Covers a runtime smoke test for tensor-backed alias connectivity between storage-image and storage-buffer stages.
# Checks the VGF contains image<->tensor and tensor<->buffer alias relations on this path.
# This intentionally checks only part of the connectivity story; exact bridge/resource topology belongs to VGF generator testing.
@common.SkipIfNoModelConverter
def test_storage_image_storage_buffer_alias_relations(tmp_path):
    x, grid = make_sampler_probe_inputs()
    expected, actual, vgf_json = lower_sampler_and_threes_vgf(
        _SamplerThenThrees(), (x, grid), tmp_path
    )

    assert torch.allclose(expected, actual, atol=1e-3, rtol=1e-2)
    assert _has_alias_relations(
        vgf_json,
        "VK_DESCRIPTOR_TYPE_STORAGE_IMAGE",
        "VK_DESCRIPTOR_TYPE_TENSOR_ARM",
        "VK_DESCRIPTOR_TYPE_STORAGE_BUFFER",
    )


# Covers a runtime smoke test for tensor-backed alias connectivity between storage-buffer and combined-image-sampler stages.
# Checks the VGF contains buffer<->tensor and tensor<->combined-image-sampler alias relations on this path.
# This intentionally checks only part of the connectivity story; exact bridge/resource topology belongs to VGF generator testing.
@common.SkipIfNoModelConverter
def test_storage_buffer_combined_image_sampler_alias_relations(tmp_path):
    x, grid = make_sampler_probe_inputs()
    expected, actual, vgf_json = lower_sampler_and_threes_vgf(
        _ThreesThenSampler(), (x, grid), tmp_path
    )

    assert torch.allclose(expected, actual, atol=1e-3, rtol=1e-2)
    assert _has_alias_relations(
        vgf_json,
        "VK_DESCRIPTOR_TYPE_STORAGE_BUFFER",
        "VK_DESCRIPTOR_TYPE_TENSOR_ARM",
        "VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER",
    )


# Temporary step-by-step debug for the storage-buffer -> combined-image-sampler path.
# Checks identity-buffer, sampler-only, and identity-buffer-then-sampler stages separately and reports which stage first diverges.
@common.SkipIfNoModelConverter
def test_storage_buffer_combined_image_sampler_alias_pair_debug_steps(tmp_path):
    x, grid = make_sampler_probe_inputs()
    top_left_x = (2.0 * 0.0 + 1.0) / x.shape[-1] - 1.0
    top_left_y = (2.0 * 0.0 + 1.0) / x.shape[-2] - 1.0
    grid[..., 0] = top_left_x
    grid[..., 1] = top_left_y

    identity_dir = tmp_path / "identity_buffer_only"
    identity_dir.mkdir()
    expected_identity, actual_identity, _ = lower_threes_vgf(
        _IdentityBufferOnly(), (x,), identity_dir
    )

    sampler_dir = tmp_path / "sampler_only"
    sampler_dir.mkdir()
    expected_sampler, actual_sampler, _ = lower_sampler_vgf(
        _IdentitySamplerBufferDebug(), (x, grid), sampler_dir
    )

    sampler_buffer_nchw_dir = tmp_path / "sampler_buffer_nchw_only"
    sampler_buffer_nchw_dir.mkdir()
    expected_sampler_buffer_nchw, actual_sampler_buffer_nchw, _ = lower_sampler_vgf(
        _IdentitySamplerBufferNchwDebug(), (x, grid), sampler_buffer_nchw_dir
    )

    grid_read_dir = tmp_path / "grid_read_tensor_only"
    grid_read_dir.mkdir()
    expected_grid_read, actual_grid_read, _ = lower_sampler_vgf(
        _GridReadTensorDebug(), (x, grid), grid_read_dir
    )

    pipeline_dir = tmp_path / "identity_buffer_then_sampler"
    pipeline_dir.mkdir()
    expected_pipeline, actual_pipeline, _ = lower_sampler_and_threes_vgf(
        _IdentityPackedThenSamplerBufferDebug(), (x, grid), pipeline_dir
    )

    pipeline_buffer_nchw_dir = tmp_path / "identity_buffer_then_sampler_buffer_nchw"
    pipeline_buffer_nchw_dir.mkdir()
    expected_pipeline_buffer_nchw, actual_pipeline_buffer_nchw, _ = (
        lower_sampler_and_threes_vgf(
            _IdentityBufferThenSamplerBufferNchwDebug(),
            (x, grid),
            pipeline_buffer_nchw_dir,
        )
    )

    failures = []
    if not torch.allclose(expected_identity, actual_identity, atol=1e-6, rtol=0.0):
        failures.append(
            "identity_buffer_only "
            f"max_abs_diff={(expected_identity - actual_identity).abs().max().item():.6f}"
        )
    if not torch.allclose(expected_sampler, actual_sampler, atol=1e-3, rtol=1e-2):
        torch.set_printoptions(threshold=100000, linewidth=240, sci_mode=False)
        print("sampler_only expected:")
        print(expected_sampler)
        print("sampler_only actual:")
        print(actual_sampler)
        failures.append(
            "sampler_only "
            f"max_abs_diff={(expected_sampler - actual_sampler).abs().max().item():.6f}"
        )
    if not torch.allclose(
        expected_sampler_buffer_nchw,
        actual_sampler_buffer_nchw,
        atol=1e-3,
        rtol=1e-2,
    ):
        torch.set_printoptions(threshold=100000, linewidth=240, sci_mode=False)
        print("sampler_buffer_nchw_only expected:")
        print(expected_sampler_buffer_nchw)
        print("sampler_buffer_nchw_only actual:")
        print(actual_sampler_buffer_nchw)
        failures.append(
            "sampler_buffer_nchw_only "
            f"max_abs_diff={(expected_sampler_buffer_nchw - actual_sampler_buffer_nchw).abs().max().item():.6f}"
        )
    if not torch.allclose(expected_grid_read, actual_grid_read, atol=1e-6, rtol=0.0):
        torch.set_printoptions(threshold=100000, linewidth=240, sci_mode=False)
        print("grid_read_tensor_only expected:")
        print(expected_grid_read)
        print("grid_read_tensor_only actual:")
        print(actual_grid_read)
        failures.append(
            "grid_read_tensor_only "
            f"max_abs_diff={(expected_grid_read - actual_grid_read).abs().max().item():.6f}"
        )
    if not torch.allclose(expected_pipeline, actual_pipeline, atol=1e-3, rtol=1e-2):
        failures.append(
            "identity_buffer_then_sampler "
            f"max_abs_diff={(expected_pipeline - actual_pipeline).abs().max().item():.6f}"
        )
    if not torch.allclose(
        expected_pipeline_buffer_nchw,
        actual_pipeline_buffer_nchw,
        atol=1e-3,
        rtol=1e-2,
    ):
        failures.append(
            "identity_buffer_then_sampler_buffer_nchw "
            f"max_abs_diff={(expected_pipeline_buffer_nchw - actual_pipeline_buffer_nchw).abs().max().item():.6f}"
        )

    assert not failures, "; ".join(failures)
