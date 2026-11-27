# Copyright 2026 Arm Limited and/or its affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import torch
import torch.nn.functional as F
from executorch.backends.arm.quantizer import (
    get_vgf_snorm_quantization_config,
    VgfQuantizer,
)
from executorch.backends.arm.quantizer.quantization_config import (
    _is_canonical_flow_offset_grid_sampler,
    VGFQuantizationConfig,
)
from executorch.backends.arm.vgf import VgfCompileSpec
from torch.export import export
from torchao.quantization.pt2e.quantize_pt2e import convert_pt2e, prepare_pt2e
from torchao.quantization.pt2e.quantizer import QuantizationSpec


class FlowOffsetSamplerChain(torch.nn.Module):
    def forward(self, image: torch.Tensor, flow: torch.Tensor) -> torch.Tensor:
        image = torch.relu(image)
        height, width = image.shape[-2:]
        horizontal = torch.linspace(
            -1.0, 1.0, width, device=image.device, dtype=image.dtype
        ).view(1, 1, 1, width)
        vertical = torch.linspace(
            -1.0, 1.0, height, device=image.device, dtype=image.dtype
        ).view(1, 1, height, 1)
        base_grid = torch.cat(
            (
                horizontal.expand(1, -1, height, -1),
                vertical.expand(1, -1, -1, width),
            ),
            dim=1,
        )
        grid = (base_grid + flow[:, :2]).permute(0, 2, 3, 1)
        sampled = F.grid_sample(
            image,
            grid,
            mode="bilinear",
            padding_mode="border",
            align_corners=True,
        )
        return torch.relu(sampled)


def test_vgf_snorm_quantization_avoids_requantization_boundaries():
    inputs = (
        torch.randn(1, 4, 8, 12),
        torch.randn(1, 4, 8, 12) * 0.01,
    )
    exported = export(FlowOffsetSamplerChain(), inputs, strict=True).module()
    grid_sampler = next(
        node
        for node in exported.graph.nodes
        if node.target == torch.ops.aten.grid_sampler.default
    )
    assert _is_canonical_flow_offset_grid_sampler(grid_sampler)

    config = get_vgf_snorm_quantization_config(is_per_channel=False)
    assert isinstance(config, VGFQuantizationConfig)
    for qspec in (config.input_activation, config.output_activation):
        assert isinstance(qspec, QuantizationSpec)
        assert (qspec.quant_min, qspec.quant_max) == (-127, 127)

    quantizer = VgfQuantizer(
        VgfCompileSpec()._set_preserve_io_quantization(True),
        use_composable_quantizer=True,
    )
    quantizer.set_global(config)
    quantizer.set_io(config)
    prepared = prepare_pt2e(exported, quantizer)
    prepared(*inputs)
    converted = convert_pt2e(prepared)

    quantize = torch.ops.quantized_decomposed.quantize_per_tensor.default
    dequantize = torch.ops.quantized_decomposed.dequantize_per_tensor.default
    requantize_nodes = [
        node
        for node in converted.graph.nodes
        if node.target == quantize
        and isinstance(node.args[0], torch.fx.Node)
        and node.args[0].target == dequantize
    ]
    assert not requantize_nodes
