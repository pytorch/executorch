# Copyright (c) Qualcomm Innovation Center, Inc.
# All rights reserved
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.
import _operator

import torch
from executorch.exir.dialects._ops import ops as exir_ops


supported_modules = [
    torch.nn.BatchNorm2d,
    torch.nn.Conv2d,
    torch.nn.Hardtanh,
    torch.nn.Linear,
    torch.nn.ReLU,
    torch.nn.Embedding,
    "forward",
]

not_supported_operator = [
    exir_ops.edge.aten.arange.start_step,
    exir_ops.edge.aten.index.Tensor,
    exir_ops.edge.aten.full.default,
]

allow_list_operator = [
    _operator.getitem,
]
