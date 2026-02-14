# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

from typing import List, Optional, Set

import torch

from executorch.exir.tensor import dim_order_from_stride

# Format-preserving ops: output layout must match primary input. Include out-variants
# because when SpecPropPass runs, OutVarPass has already converted e.g. clone.default
# to clone.out.
FORMAT_PRESERVING_OPS: Set[object] = {
    torch.ops.aten.clone.out,
    torch.ops.aten.clone.default,
    torch.ops.aten.clone.memory_format,
    torch.ops.aten.copy_.default,
    torch.ops.aten.contiguous.default,
    torch.ops.aten.relu.default,
    torch.ops.aten.silu.default,
    torch.ops.aten.gelu.default,
    torch.ops.aten.add.Tensor,
    torch.ops.aten.mul.Tensor,
    torch.ops.aten.div.Tensor,
}


def dim_order_from_fake_tensor(t: torch.Tensor) -> Optional[List[int]]:
    """
    Derive ExecuTorch dim_order from a tensor's strides (e.g. contiguous -> [0,1,2,3],
    channels_last -> [0,2,3,1]). Returns None if layout cannot be expressed (e.g. 0 in strides).
    """
    try:
        st = t.stride()
        result = dim_order_from_stride(st)
        return list(result)
    except ValueError:
        return None


def should_propagate_dim_order(op: object) -> bool:
    """True if the op is format-preserving and we should propagate primary input dim_order to out."""
    return op in FORMAT_PRESERVING_OPS
