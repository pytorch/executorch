# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Generate Native wheel fixtures with the exporter from this checkout."""

import argparse
from pathlib import Path

import numpy as np
import torch
from executorch.exir.native import to_native


class Add(torch.nn.Module):
    def forward(self, value: torch.Tensor) -> torch.Tensor:
        return value + value


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(0)
    conv_bn = torch.nn.Sequential(
        torch.nn.Conv2d(3, 4, kernel_size=3, padding=1),
        torch.nn.BatchNorm2d(4),
        torch.nn.ReLU(),
    )
    with torch.no_grad():
        conv_bn[1].running_mean.copy_(torch.linspace(-1, 1, 4))
        conv_bn[1].running_var.copy_(torch.linspace(0.5, 1.5, 4))
        conv_bn[1].weight.copy_(torch.linspace(0.8, 1.2, 4))
        conv_bn[1].bias.copy_(torch.linspace(-0.2, 0.2, 4))
    models = {
        "add": (Add(), torch.arange(1, 7, dtype=torch.float32).reshape(1, 6)),
        "conv_bn": (
            conv_bn,
            torch.randn(1, 3, 8, 8),
        ),
    }
    for name, (model, value) in models.items():
        model.eval()
        with torch.no_grad():
            expected = model(value)
        to_native(torch.export.export(model, (value,))).save(
            str(args.output / f"{name}.ptn")
        )
        np.savez(
            args.output / f"{name}.npz",
            input=value.numpy(),
            expected=expected.numpy(),
        )


if __name__ == "__main__":
    main()
