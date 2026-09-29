# Copyright 2026 Arm Limited and/or its affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import torch
from torch._higher_order_ops.map import map as torch_map


class PrimOpsModel(torch.nn.Module):
    def forward(self, xs, y):
        def map_fn(x, y):
            return x + y

        return torch_map(map_fn, xs, y)


ModelUnderTest = PrimOpsModel()
ModelInputs = (torch.ones(2, 4), torch.ones(4))
