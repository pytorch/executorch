# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from typing import Dict, List

import executorch.exir as exir
import torch


def get_bilinear_2d_graphs():
    class bilinear2d(torch.nn.Module):
        def __init__(self, align_corners):
            super().__init__()
            self.align_corners = align_corners

        def forward(self, x):
            return torch.nn.functional.interpolate(
                x,
                scale_factor=2,
                mode="bilinear",
                align_corners=self.align_corners,
                antialias=False,
            )

    sample_inputs = (torch.randn(1, 3, 4, 4),)
    _bilinear2d_graphs = {}
    capture_configs = [
        exir.CaptureConfig(enable_aot=True, _unlift=False),
        exir.CaptureConfig(enable_aot=True, _unlift=True),
    ]
    for align_corners in [True, False]:
        for config in capture_configs:
            edge = exir.capture(
                bilinear2d(align_corners), sample_inputs, config
            ).to_edge()
            _bilinear2d_graphs[edge.exported_program.graph_module] = align_corners
    return _bilinear2d_graphs


bilinear2d_graphs: Dict[torch.fx.GraphModule, bool] = get_bilinear_2d_graphs()

Graphs: List[torch.fx.GraphModule] = list(bilinear2d_graphs.keys())
