# Copyright 2026 Arm Limited and/or its affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import sys
from pathlib import Path

import pytest
import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[4]))

from backends.arm.test.runtime._vgf_runtime_test_utils import (
    alias_groups,
    lower_add_vgf,
    lower_in_tree_vgf,
    make_identity_grid,
    make_input_tensor,
    xfail_if_legacy_model_converter_release,
)
from executorch.backends.arm.test import common

pytestmark = xfail_if_legacy_model_converter_release()


class _IdentityGridSample(torch.nn.Module):
    def forward(self, x: torch.Tensor, grid: torch.Tensor) -> torch.Tensor:
        return F.grid_sample(
            x,
            grid,
            mode="bilinear",
            padding_mode="zeros",
            align_corners=False,
        )


class _GraphToShader(torch.nn.Module):
    def forward(self, x: torch.Tensor, grid: torch.Tensor) -> torch.Tensor:
        return F.grid_sample(
            x * 2.0 + 1.0,
            grid,
            mode="bilinear",
            padding_mode="zeros",
            align_corners=False,
        )


class _ShaderToGraph(torch.nn.Module):
    def forward(self, x: torch.Tensor, grid: torch.Tensor) -> torch.Tensor:
        y = F.grid_sample(
            x, grid, mode="bilinear", padding_mode="zeros", align_corners=False
        )
        return y * 0.5 + 3.0


class _EndToEnd(torch.nn.Module):
    def forward(self, x: torch.Tensor, grid: torch.Tensor) -> torch.Tensor:
        y = F.grid_sample(
            x * 2.0 + 1.0,
            grid,
            mode="bilinear",
            padding_mode="zeros",
            align_corners=False,
        )
        return y * 0.5 + 3.0


class _BinaryAddShader(torch.nn.Module):
    def forward(self, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        return a + b


class _DuplicatedInputAddShader(torch.nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + x


# Covers the simplest runtime path through the in-tree grid-sample flow.
# Checks runtime execution matches eager output for an identity-style sample.
@common.SkipIfNoModelConverter
def test_tensor_input_buffer_output_identity_shader(tmp_path):
    x = make_input_tensor(4, 4)
    grid = make_identity_grid(4, 4)
    expected, actual, _ = lower_in_tree_vgf(_IdentityGridSample(), (x, grid), tmp_path)

    assert torch.allclose(expected, actual, atol=1e-6, rtol=0.0)


# Covers graph work feeding the shader path.
# Checks a graph-produced tensor is consumed correctly by the runtime shader segment.
@common.SkipIfNoModelConverter
def test_graph_tensor_to_shader_buffer_handoff(tmp_path):
    x = make_input_tensor(4, 4)
    grid = make_identity_grid(4, 4)
    expected, actual, _ = lower_in_tree_vgf(_GraphToShader(), (x, grid), tmp_path)

    assert torch.allclose(expected, actual, atol=1e-6, rtol=0.0)


# Covers graph work after the shader path.
# Checks shader output is consumed correctly by following graph ops at runtime.
@common.SkipIfNoModelConverter
def test_shader_buffer_to_graph_tensor_handoff(tmp_path):
    x = make_input_tensor(4, 4)
    grid = make_identity_grid(4, 4)
    expected, actual, _ = lower_in_tree_vgf(_ShaderToGraph(), (x, grid), tmp_path)

    assert torch.allclose(expected, actual, atol=1e-6, rtol=0.0)


# Covers artifact-level tensor/buffer aliasing in the generated VGF.
# Checks at least one alias group spans tensor and storage-buffer descriptors.
@common.SkipIfNoModelConverter
def test_tensor_buffer_alias_group_reuses_backing_memory(tmp_path):
    x = make_input_tensor(4, 4)
    grid = make_identity_grid(4, 4)
    _, _, vgf_json = lower_in_tree_vgf(_GraphToShader(), (x, grid), tmp_path)
    groups = alias_groups(vgf_json)

    assert groups
    assert any(
        {resource["vk_descriptor_type"] for resource in group}
        >= {
            "VK_DESCRIPTOR_TYPE_TENSOR_ARM",
            "VK_DESCRIPTOR_TYPE_STORAGE_BUFFER",
        }
        for group in groups.values()
    )


# Covers the end-to-end tensor/buffer runtime flow with graph ops on both sides.
# Checks numerics across the full lowered pipeline match eager execution.
@common.SkipIfNoModelConverter
def test_tensor_buffer_runtime_executes_end_to_end(tmp_path):
    x = make_input_tensor(4, 4)
    grid = make_identity_grid(4, 4)
    expected, actual, _ = lower_in_tree_vgf(_EndToEnd(), (x, grid), tmp_path)

    assert torch.allclose(expected, actual, atol=1e-6, rtol=0.0)


# Covers the standalone two-input storage-buffer shader path.
# Checks runtime execution matches eager output for a minimal binary add case.
@common.SkipIfNoModelConverter
def test_two_input_add_buffer_shader_executes(tmp_path):
    a = torch.randn(256)
    b = torch.randn(256)
    expected, actual, _ = lower_add_vgf(_BinaryAddShader(), (a, b), tmp_path)

    assert torch.allclose(expected, actual, atol=1e-6, rtol=0.0)


# Covers the two-input storage-buffer shader path when both inputs are the same tensor.
# Checks runtime execution matches eager output for the duplicated-input add case.
@pytest.mark.xfail(
    reason="model-converter drops duplicated custom shader inputs", strict=True
)
@common.SkipIfNoModelConverter
def test_two_input_add_buffer_shader_with_duplicated_input_executes(tmp_path):
    x = torch.randn(256)
    expected, actual, _ = lower_add_vgf(_DuplicatedInputAddShader(), (x,), tmp_path)

    assert torch.allclose(expected, actual, atol=1e-6, rtol=0.0)
