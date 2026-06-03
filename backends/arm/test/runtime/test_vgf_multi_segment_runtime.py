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
    lower_in_tree_vgf,
    lower_sampler_vgf,
    make_identity_grid,
    make_input_tensor,
    make_sampler_probe_inputs,
    xfail_if_legacy_model_converter_release,
)
from executorch.backends.arm.test import common

pytestmark = xfail_if_legacy_model_converter_release()


class _GraphThenShader(torch.nn.Module):
    def forward(self, x: torch.Tensor, grid: torch.Tensor) -> torch.Tensor:
        return F.grid_sample(
            x * 2.0 + 1.0,
            grid,
            mode="bilinear",
            padding_mode="zeros",
            align_corners=False,
        )


class _ShaderThenGraph(torch.nn.Module):
    def forward(self, x: torch.Tensor, grid: torch.Tensor) -> torch.Tensor:
        y = F.grid_sample(
            x, grid, mode="bilinear", padding_mode="zeros", align_corners=False
        )
        return y * 0.5 + 3.0


class _GraphShaderGraph(torch.nn.Module):
    def forward(self, x: torch.Tensor, grid: torch.Tensor) -> torch.Tensor:
        x = x * 2.0 + 1.0
        y = F.grid_sample(
            x, grid, mode="bilinear", padding_mode="zeros", align_corners=False
        )
        return y * 0.5 + 3.0


class _ShaderGraphShader(torch.nn.Module):
    def forward(
        self, x: torch.Tensor, grid0: torch.Tensor, grid1: torch.Tensor
    ) -> torch.Tensor:
        y = F.grid_sample(
            x, grid0, mode="bilinear", padding_mode="zeros", align_corners=False
        )
        y = y * 0.5 + 3.0
        return F.grid_sample(
            y, grid1, mode="bilinear", padding_mode="zeros", align_corners=False
        )


class _GraphShaderGraphShader(torch.nn.Module):
    def forward(
        self, x: torch.Tensor, grid0: torch.Tensor, grid1: torch.Tensor
    ) -> torch.Tensor:
        x = x * 2.0 + 1.0
        y = F.grid_sample(
            x, grid0, mode="bilinear", padding_mode="zeros", align_corners=False
        )
        y = y * 0.5 + 3.0
        return F.grid_sample(
            y, grid1, mode="bilinear", padding_mode="zeros", align_corners=False
        )


# Covers a simple graph-to-shader two-segment pipeline.
# Checks numerics match eager execution across the segment boundary.
@common.SkipIfNoModelConverter
def test_graph_then_shader_segment_executes(tmp_path):
    x = make_input_tensor(4, 4)
    grid = make_identity_grid(4, 4)
    expected, actual, _ = lower_in_tree_vgf(_GraphThenShader(), (x, grid), tmp_path)

    assert torch.allclose(expected, actual, atol=1e-6, rtol=0.0)


# Covers a simple shader-to-graph two-segment pipeline.
# Checks numerics match eager execution across the segment boundary.
@common.SkipIfNoModelConverter
def test_shader_then_graph_segment_executes(tmp_path):
    x = make_input_tensor(4, 4)
    grid = make_identity_grid(4, 4)
    expected, actual, _ = lower_in_tree_vgf(_ShaderThenGraph(), (x, grid), tmp_path)

    assert torch.allclose(expected, actual, atol=1e-6, rtol=0.0)


# Covers a graph-shader-graph three-segment pipeline.
# Checks runtime execution remains correct through both handoff directions.
@common.SkipIfNoModelConverter
def test_graph_shader_graph_executes(tmp_path):
    x = make_input_tensor(4, 4)
    grid = make_identity_grid(4, 4)
    expected, actual, _ = lower_in_tree_vgf(_GraphShaderGraph(), (x, grid), tmp_path)

    assert torch.allclose(expected, actual, atol=1e-6, rtol=0.0)


# Covers a shader-graph-shader three-segment pipeline.
# Checks repeated segment transitions preserve correctness through runtime execution.
@common.SkipIfNoModelConverter
def test_shader_graph_shader_executes(tmp_path):
    x = make_input_tensor(4, 4)
    grid0 = make_identity_grid(4, 4)
    grid1 = make_identity_grid(4, 4)
    expected, actual, _ = lower_in_tree_vgf(
        _ShaderGraphShader(), (x, grid0, grid1), tmp_path
    )

    assert torch.allclose(expected, actual, atol=1e-6, rtol=0.0)


# Covers a longer mixed graph/shader pipeline with four logical stages.
# Checks numerics remain correct through multiple segment transitions.
@common.SkipIfNoModelConverter
def test_graph_shader_graph_shader_executes(tmp_path):
    x = make_input_tensor(4, 4)
    grid0 = make_identity_grid(4, 4)
    grid1 = make_identity_grid(4, 4)
    expected, actual, _ = lower_in_tree_vgf(
        _GraphShaderGraphShader(), (x, grid0, grid1), tmp_path
    )

    assert torch.allclose(expected, actual, atol=1e-6, rtol=0.0)


# Covers the multi-segment sampler/image runtime path specifically.
# Checks repeated sampled stages match eager execution within the expected tolerance.
@common.SkipIfNoModelConverter
def test_multi_segment_sampler_path_executes(tmp_path):
    x, grid0 = make_sampler_probe_inputs()
    grid1 = grid0.clone()
    expected, actual, _ = lower_sampler_vgf(
        _ShaderGraphShader(), (x, grid0, grid1), tmp_path
    )

    assert torch.allclose(expected, actual, atol=1e-3, rtol=1e-2)
