# Copyright 2026 Mwaniki Njagi.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.
#
# SPDX-License-Identifier: BSD-3-Clause

"""
Minimal example: quantize a torchvision model to int8 with the ESP-NN
quantizer and lower it to an ExecuTorch program for Espressif chips.

    python examples/espressif/quantize_and_lower.py --output mobilenet_v2_espnn.pte
"""

from __future__ import annotations

import argparse

import torch
import torchvision
from executorch.backends.espressif import EspNNQuantizer, to_espnn_edge
from torchao.quantization.pt2e.quantize_pt2e import convert_pt2e, prepare_pt2e


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default="mobilenet_v2_espnn.pte")
    args = parser.parse_args()

    model = torchvision.models.mobilenet_v2(weights="DEFAULT").eval()
    example_inputs = (torch.randn(1, 3, 224, 224),)

    exported = torch.export.export(model, example_inputs).module()
    prepared = prepare_pt2e(exported, EspNNQuantizer())
    with torch.no_grad():
        for _ in range(16):
            prepared(torch.randn(1, 3, 224, 224))
    quantized = convert_pt2e(prepared)

    program = torch.export.export(quantized, example_inputs)
    edge, report = to_espnn_edge(program)
    print(
        f"{report['partitions']} EspNN partition(s), {report['delegated_ops']} delegated operators, "
        f"{len(report['cpu_ops'])} operators left for the CPU"
    )

    with open(args.output, "wb") as file:
        file.write(edge.to_executorch().buffer)
    print(f"wrote {args.output}")


if __name__ == "__main__":
    main()
