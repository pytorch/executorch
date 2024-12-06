# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-unsafe

import argparse

import os

import torch
from executorch.exir import to_edge

from executorch.extension.training.examples.XOR.model import Net, TrainingNet
from torch.export import export
from torch.export.experimental import _export_forward_backward


def _export_model():
    net = TrainingNet(Net())
    x = torch.randn(1, 2)

    # Captures the forward graph. The graph will look similar to the model definition now.
    # Will move to export_for_training soon which is the api planned to be supported in the long term.
    ep = export(net, (x, torch.ones(1, dtype=torch.int64)))
    # Captures the backward graph. The exported_program now contains the joint forward and backward graph.
    ep = _export_forward_backward(ep)
    # Lower the graph to edge dialect.
    ep = to_edge(ep)
    # Lower the graph to executorch.
    ep = ep.to_executorch()


def main() -> None:
    torch.manual_seed(0)
    parser = argparse.ArgumentParser(
        prog="export_model",
        description="Exports an nn.Module model to ExecuTorch .pte files",
    )
    parser.add_argument(
        "--outdir",
        type=str,
        required=True,
        help="Path to the directory to write xor.pte files to",
    )
    args = parser.parse_args()

    ep = _export_model()

    # Write out the .pte file.
    os.makedirs(args.outdir, exist_ok=True)
    outfile = os.path.join(args.outdir, "xor.pte")
    with open(outfile, "wb") as fp:
        fp.write(
            ep.buffer,
        )


if __name__ == "__main__":
    main()
