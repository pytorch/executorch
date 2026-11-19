# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Quantize a small model for the portable kernels with the fused_quant recipe.

Run with ``python -m executorch.frontends.fused_quant.quick_start``.
"""

import argparse

import torch
from executorch.export import export
from executorch.frontends.fused_quant.quantizer.quantizer import (
    FusedQuantQuantizer,
    get_core_aten_quantizers,
)
from executorch.frontends.fused_quant.recipe import get_fused_quant_portable_recipe
from torchao.quantization.pt2e import (
    HistogramObserver,
    MinMaxObserver,
    PerChannelMinMaxObserver,
)
from torchao.quantization.pt2e.quantizer import QuantizationConfig, QuantizationSpec


class ConvNet(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.conv = torch.nn.Conv2d(3, 8, kernel_size=3, padding=1)
        self.linear = torch.nn.Linear(8 * 8 * 8, 10)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear(torch.relu(self.conv(x)).flatten(1))


ACTIVATION = QuantizationSpec(
    dtype=torch.int8,
    quant_min=-128,
    quant_max=127,
    qscheme=torch.per_tensor_affine,
    observer_or_fake_quant_ctr=HistogramObserver.with_args(eps=2**-12),
)


def weight_spec(per_channel: bool) -> QuantizationSpec:
    return QuantizationSpec(
        dtype=torch.int8,
        quant_min=-127,
        quant_max=127,
        qscheme=(
            torch.per_channel_symmetric if per_channel else torch.per_tensor_symmetric
        ),
        ch_axis=0 if per_channel else None,
        observer_or_fake_quant_ctr=(
            PerChannelMinMaxObserver.with_args(eps=2**-12)
            if per_channel
            else MinMaxObserver
        ),
    )


def main(output_path: str) -> float:
    """Saves the program to ``output_path`` and returns its largest absolute
    difference from the float model on one input."""
    torch.manual_seed(0)
    model = ConvNet().eval()
    # The example inputs are also the calibration set.
    calibration_inputs = [(torch.randn(1, 3, 8, 8),) for _ in range(8)]

    # QuantizationConfig(input_activation, output_activation, weight, bias)
    config = QuantizationConfig(ACTIVATION, ACTIVATION, weight_spec(False), None)
    weighted_config = QuantizationConfig(
        ACTIVATION, ACTIVATION, weight_spec(True), None
    )
    quantizer = FusedQuantQuantizer(get_core_aten_quantizers(config, weighted_config))

    session = export(
        model=model,
        example_inputs=calibration_inputs,
        export_recipe=get_fused_quant_portable_recipe(quantizer),
    )
    session.save_to_pte(output_path)

    (example,) = calibration_inputs[0]
    (actual,) = session.run_method("forward", (example,))
    with torch.no_grad():
        expected = model(example)
    return (actual - expected).abs().max().item()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", default="fused_quant_portable.pte")
    args = parser.parse_args()
    error = main(args.output)
    print(f"Saved {args.output}; largest difference from the float model: {error:.4f}")
