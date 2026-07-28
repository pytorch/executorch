# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""`aten.div.Tensor_mode` module for the WebGPU op-test framework.

`FloorDivideModule` is imported by `cases.py`. Same-shape elementwise
`div(a, b, rounding_mode="floor")`. The kernel computes `floor(a/b)` mirroring
the Vulkan `floor_divide` glsl (`floor(X/Y)`); this differs from torch's own
fmod-corrected `div_floor` at rare fp boundaries, so the suite goldens against a
`floor(a/b)` `golden_fn` (Vulkan-faithful), not this module's eager output.
"""

import torch


class FloorDivideModule(torch.nn.Module):
    def forward(self, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        return torch.div(a, b, rounding_mode="floor")
