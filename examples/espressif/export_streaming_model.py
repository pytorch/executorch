# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.
#
# SPDX-License-Identifier: BSD-3-Clause

"""
Export a streaming keyword-spotting model for the ESP-NN delegate, with the
streaming state owned by the delegate and the program's one input and one
output left to the application's buffers.

Each call takes one frame of 40 features and returns one int8 score; the
convolutions see the last frames, kept in two buffers. Replace the random
weights and calibration frames with a trained model and real features.

    python examples/espressif/export_streaming_model.py --output kws.pte
"""

from __future__ import annotations

import argparse
from dataclasses import replace

import torch
from executorch.backends.espressif import EspNNQuantizer, to_espnn_edge
from executorch.exir import ExecutorchBackendConfig
from executorch.exir.passes import MemoryPlanningPass
from executorch.exir.passes.quantize_io_pass import quantize_input, quantize_output
from torchao.quantization.pt2e.observer import MinMaxObserver
from torchao.quantization.pt2e.quantize_pt2e import convert_pt2e, prepare_pt2e

FEATURES = 40


class StreamingKws(torch.nn.Module):
    """Two dilated temporal convolutions over a frame history, then a score."""

    def __init__(self):
        super().__init__()
        self.register_buffer("history1", torch.zeros(1, FEATURES, 2, 1))
        self.register_buffer("history2", torch.zeros(1, 32, 4, 1))
        self.conv1 = torch.nn.Conv2d(FEATURES, 32, (3, 1))
        self.conv2 = torch.nn.Conv2d(32, 32, (3, 1), dilation=(2, 1))
        self.score = torch.nn.Linear(32, 1)

    def forward(self, frame):
        window1 = torch.cat([self.history1, frame], dim=2)
        self.history1.copy_(window1[:, :, 1:])
        hidden = torch.relu(self.conv1(window1))
        window2 = torch.cat([self.history2, hidden], dim=2)
        self.history2.copy_(window2[:, :, 1:])
        hidden = torch.relu(self.conv2(window2))
        return torch.sigmoid(self.score(hidden.flatten(1)))


def minmax_quantizer() -> EspNNQuantizer:
    """EspNNQuantizer with min/max activation ranges instead of histograms."""
    quantizer = EspNNQuantizer()
    config = quantizer.global_config
    minmax = MinMaxObserver.with_args(eps=2**-12)
    return quantizer.set_global(
        replace(
            config,
            input_activation=replace(
                config.input_activation, observer_or_fake_quant_ctr=minmax
            ),
            output_activation=replace(
                config.output_activation, observer_or_fake_quant_ctr=minmax
            ),
        )
    )


def lower(calibration_frames: int = 200, seed: int = 0):
    """Quantizes and lowers the model; returns the EdgeProgramManager."""
    torch.manual_seed(seed)
    model = StreamingKws().eval()
    example = (torch.zeros(1, FEATURES, 1, 1),)
    prepared = prepare_pt2e(
        torch.export.export(model, example, strict=True).module(), minmax_quantizer()
    )
    # Calibrate on consecutive frames, so the histories see realistic values.
    for _ in range(calibration_frames):
        prepared(torch.randn(example[0].shape))
    quantized = convert_pt2e(prepared)
    # Calibration updated the histories; export from the initial state.
    quantized.history1.zero_()
    quantized.history2.zero_()

    program = torch.export.export(quantized, example, strict=True)
    # The application passes int8 features and reads an int8 score.
    quantize_input(program, 0)
    quantize_output(program, 0)
    edge, report = to_espnn_edge(
        program, allow_cpu_fallback=False, delegate_mutable_buffers=True
    )
    print(
        f"{report['partitions']} partition(s), {report['delegated_ops']} delegated "
        f"operators, delegate-owned buffers: {report['delegated_buffers']}"
    )
    return edge


def executorch_config() -> ExecutorchBackendConfig:
    """Inputs and outputs live in the application's buffers."""
    return ExecutorchBackendConfig(
        memory_planning_pass=MemoryPlanningPass(
            alloc_graph_input=False, alloc_graph_output=False
        )
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", default="streaming_kws_espnn.pte")
    args = parser.parse_args()
    with torch.no_grad():
        program = lower().to_executorch(executorch_config()).buffer
    with open(args.output, "wb") as f:
        f.write(program)
    print(f"wrote {args.output} ({len(program)} bytes)")


if __name__ == "__main__":
    main()
