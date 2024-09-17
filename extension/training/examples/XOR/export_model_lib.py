# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-unsafe

import os

import torch
from executorch.exir import to_edge
from executorch.extension.training.examples.XOR.model import TrainingNet
from torch.export._trace import _export
from torch.export.experimental import _export_forward_backward

from .model import Net


def export_model(outdir):
    net = TrainingNet(Net())
    x = torch.randn(1, 2)

    # Captures the forward graph. The graph will look similar to the model definition now.
    # Will move to export_for_training soon which is the api planned to be supported in the long term.
    ep = _export(net, (x, torch.ones(1, dtype=torch.int64)), pre_dispatch=True)
    # Captures the backward graph. The exported_program now contains the joint forward and backward graph.
    ep = _export_forward_backward(ep)
    # Lower the graph to edge dialect.
    ep = to_edge(ep)
    # Lower the graph to executorch.
    ep = ep.to_executorch()

    # Write out the .pte file.
    os.makedirs(outdir, exist_ok=True)
    outfile = os.path.join(outdir, "xor.pte")
    with open(outfile, "wb") as fp:
        fp.write(
            ep.buffer,
        )
