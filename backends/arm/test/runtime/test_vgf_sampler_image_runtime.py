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
    lower_sampler_vgf,
    make_identity_grid,
    make_input_tensor,
    make_sampler_probe_inputs,
    xfail_if_legacy_model_converter_release,
)
from executorch.backends.arm.test import common

pytestmark = xfail_if_legacy_model_converter_release()


class _IdentitySampler(torch.nn.Module):
    def forward(self, x: torch.Tensor, grid: torch.Tensor) -> torch.Tensor:
        return F.grid_sample(
            x,
            grid,
            mode="bilinear",
            padding_mode="zeros",
            align_corners=False,
        )


class _GraphConsumerSampler(torch.nn.Module):
    def forward(self, x: torch.Tensor, grid: torch.Tensor) -> torch.Tensor:
        y = F.grid_sample(
            x,
            grid,
            mode="bilinear",
            padding_mode="zeros",
            align_corners=False,
        )
        return y * 0.5 + 3.0


# Covers the basic sampler/image runtime path.
# Checks sampled-image input can be read and returned correctly at runtime.
@common.SkipIfNoModelConverter
def test_sampled_image_to_tensor_identity_read(tmp_path):
    x = make_input_tensor(4, 4).contiguous(memory_format=torch.channels_last)
    grid = make_identity_grid(4, 4)
    expected, actual, _ = lower_sampler_vgf(_IdentitySampler(), (x, grid), tmp_path)

    assert torch.allclose(expected, actual, atol=1e-6, rtol=0.0)


# Covers exact texel-center sampling behavior.
# Checks exact sample points match eager output on the clean probe rows.
@common.SkipIfNoModelConverter
def test_sampled_image_exact_texel_center_reads(tmp_path):
    x, grid = make_sampler_probe_inputs()
    expected, actual, _ = lower_sampler_vgf(_IdentitySampler(), (x, grid), tmp_path)

    assert torch.equal(expected[0, 0, 0], actual[0, 0, 0])
    assert torch.equal(expected[0, 0, 1], actual[0, 0, 1])


# Covers linear interpolation behavior on the sampler path.
# Checks runtime output matches eager output within the expected tolerance.
@common.SkipIfNoModelConverter
def test_sampled_image_linear_interpolation_probe(tmp_path):
    x, grid = make_sampler_probe_inputs()
    expected, actual, _ = lower_sampler_vgf(_IdentitySampler(), (x, grid), tmp_path)

    assert torch.allclose(expected, actual, atol=1e-3, rtol=1e-2)


# Covers storage-image output feeding later graph/tensor consumption.
# Checks the runtime numerics match and the generated VGF contains a storage-image resource.
@common.SkipIfNoModelConverter
def test_storage_image_output_can_round_trip_to_graph_tensor(tmp_path):
    x, grid = make_sampler_probe_inputs()
    expected, actual, vgf_json = lower_sampler_vgf(
        _GraphConsumerSampler(), (x, grid), tmp_path
    )

    assert torch.allclose(expected, actual, atol=1e-3, rtol=1e-2)
    assert any(
        resource["vk_descriptor_type"] == "VK_DESCRIPTOR_TYPE_STORAGE_IMAGE"
        for resource in vgf_json["resources"]
    )


# Covers sampler metadata requirements for combined-image-sampler resources.
# Checks every combined-image-sampler MRT entry carries sampler_config in the VGF dump.
@common.SkipIfNoModelConverter
def test_combined_image_sampler_requires_sampler_config(tmp_path):
    x, grid = make_sampler_probe_inputs()
    _, _, vgf_json = lower_sampler_vgf(_GraphConsumerSampler(), (x, grid), tmp_path)
    combined_image_samplers = [
        resource
        for resource in vgf_json["resources"]
        if resource["vk_descriptor_type"] == "VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER"
    ]

    assert combined_image_samplers
    assert all("sampler_config" in resource for resource in combined_image_samplers)
