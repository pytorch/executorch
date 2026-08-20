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
    lower_sampler_vgf,
    lower_threes_vgf,
    make_sampler_probe_inputs,
    xfail_if_legacy_model_converter_release,
)
from executorch.backends.arm.test import common

pytestmark = xfail_if_legacy_model_converter_release()


class _ThreesModule(torch.nn.Module):
    def forward(self, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        x = a + b
        return torch.ops.arm_test_shader_ops.threes.default(x)


class _SamplerGraphConsumer(torch.nn.Module):
    def forward(self, x: torch.Tensor, grid: torch.Tensor) -> torch.Tensor:
        y = F.grid_sample(
            x,
            grid,
            mode="bilinear",
            padding_mode="zeros",
            align_corners=False,
        )
        return y * 0.5 + 3.0


class _GraphSamplerGraphConsumer(torch.nn.Module):
    def forward(self, x: torch.Tensor, grid: torch.Tensor) -> torch.Tensor:
        x = x * 2.0 + 1.0
        y = F.grid_sample(
            x,
            grid,
            mode="bilinear",
            padding_mode="zeros",
            align_corners=False,
        )
        return y * 0.5 + 3.0


# Covers runtime execution for the standalone threes buffer shader path.
# Checks numerics match eager execution and that tensor/buffer aliasing appears in the VGF.
@common.SkipIfNoModelConverter
def test_tensor_buffer_alias_group_executes_correctly(tmp_path):
    a = torch.randn(256)
    b = torch.randn(256)
    expected, actual, vgf_json = lower_threes_vgf(_ThreesModule(), (a, b), tmp_path)
    groups = alias_groups(vgf_json)

    assert torch.allclose(expected, actual, atol=1e-5, rtol=0.0)
    assert any(
        {resource["vk_descriptor_type"] for resource in group}
        >= {
            "VK_DESCRIPTOR_TYPE_TENSOR_ARM",
            "VK_DESCRIPTOR_TYPE_STORAGE_BUFFER",
        }
        for group in groups.values()
    )


# Covers runtime execution for storage-image to tensor aliasing.
# Checks numerics match eager execution and that tensor/storage-image aliasing is present.
@common.SkipIfNoModelConverter
def test_tensor_image_alias_group_executes_correctly(tmp_path):
    x, grid = make_sampler_probe_inputs()
    expected, actual, vgf_json = lower_sampler_vgf(
        _SamplerGraphConsumer(), (x, grid), tmp_path
    )
    groups = alias_groups(vgf_json)

    assert torch.allclose(expected, actual, atol=1e-3, rtol=1e-2)
    assert any(
        {resource["vk_descriptor_type"] for resource in group}
        >= {
            "VK_DESCRIPTOR_TYPE_TENSOR_ARM",
            "VK_DESCRIPTOR_TYPE_STORAGE_IMAGE",
        }
        for group in groups.values()
    )


# Covers graph-to-sampler aliasing on the sampled-image path.
# Checks the VGF contains an alias group spanning tensor and combined-image-sampler resources.
@common.SkipIfNoModelConverter
def test_image_sampler_alias_group_executes_correctly(tmp_path):
    x, grid = make_sampler_probe_inputs()
    _, _, vgf_json = lower_sampler_vgf(
        _GraphSamplerGraphConsumer(), (x, grid), tmp_path
    )
    groups = alias_groups(vgf_json)

    assert any(
        {resource["vk_descriptor_type"] for resource in group}
        >= {
            "VK_DESCRIPTOR_TYPE_TENSOR_ARM",
            "VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER",
        }
        for group in groups.values()
    )


# Covers shader-to-graph aliasing on the sampled-image path.
# Checks the VGF contains an alias group spanning storage-image and tensor resources.
@common.SkipIfNoModelConverter
def test_graph_consumes_tensor_alias_of_image_output(tmp_path):
    x, grid = make_sampler_probe_inputs()
    _, _, vgf_json = lower_sampler_vgf(_SamplerGraphConsumer(), (x, grid), tmp_path)
    groups = alias_groups(vgf_json)

    assert any(
        {resource["vk_descriptor_type"] for resource in group}
        >= {
            "VK_DESCRIPTOR_TYPE_STORAGE_IMAGE",
            "VK_DESCRIPTOR_TYPE_TENSOR_ARM",
        }
        for group in groups.values()
    )
