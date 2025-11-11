# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-unsafe

import argparse

import os

import torch
from executorch.exir import ExecutorchBackendConfig, to_edge

from executorch.extension.training.examples.XOR.model import Net, TrainingNet
from torch.export import export
from torch.export.experimental import _export_forward_backward


def _export_model(external_mutable_weights: bool = False):
    net = TrainingNet(Net())
    x = torch.randn(1, 2)

    # Captures the forward graph. The graph will look similar to the model definition now.
    ep = export(net, (x, torch.ones(1, dtype=torch.int64)), strict=True)
    # Captures the backward graph. The exported_program now contains the joint forward and backward graph.
    ep = _export_forward_backward(ep)
    # Lower the graph to edge dialect.
    ep = to_edge(ep)
    # Lower the graph to executorch.
    ep = ep.to_executorch(
        config=ExecutorchBackendConfig(
            external_mutable_weights=external_mutable_weights
        )
    )
    return ep


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
        help="Path to the directory to write xor.pte and xor.ptd files to",
    )
    parser.add_argument(
        "--external",
        action="store_true",
        help="Export the model with external weights",
    )
    args = parser.parse_args()

    ep = _export_model(args.external)

    # Write out the .pte file.
    os.makedirs(args.outdir, exist_ok=True)
    outfile = os.path.join(args.outdir, "xor.pte")
    with open(outfile, "wb") as fp:
        ep.write_to_file(fp)

    if args.external:
        # current infra doesnt easily allow renaming this file, so just hackily do it here.
        ep._tensor_data["xor"] = ep._tensor_data.pop("_default_external_constant")
        ep.write_tensor_data_to_file(args.outdir)


if __name__ == "__main__":
    main()
