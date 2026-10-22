# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import executorch.backends.transforms.channels_last_ops  # noqa: F401

import torch

from executorch.exir.dialects._ops import ops as exir_ops
from torch.fx.node import Target

ATEN_PERMUTE_COPY = exir_ops.edge.aten.permute_copy.default
LAYOUT_PERMUTE_COPY = exir_ops.edge.channels_last.permute_copy.default
PERMUTE_COPY_TARGETS: frozenset[Target] = frozenset(
    (ATEN_PERMUTE_COPY, LAYOUT_PERMUTE_COPY)
)


def is_permute_copy(node: torch.fx.Node) -> bool:
    return node.op == "call_function" and node.target in PERMUTE_COPY_TARGETS
