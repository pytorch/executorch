# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import torch
from executorch.exir.dialects._ops import ops as exir_ops
from executorch.exir.dim_order_utils import get_memory_format

from torch.library import impl, Library

lib = Library("dim_order_ops", "DEF")
lib.define(
    "_to_dim_order_copy(Tensor self, *, ScalarType? dtype=None, Layout? layout=None, Device? device=None, bool? pin_memory=None, bool non_blocking=False, int[]? dim_order=None) -> Tensor"
)

# Out variant drops TensorOptions
lib.define(
    "_to_dim_order_copy.out(Tensor self, *, bool non_blocking=False, int[]? dim_order=None, Tensor(a!) out) -> Tensor(a!)"
)


def _op_impl(target, *args, **kwargs):
    kwargs["memory_format"] = get_memory_format(kwargs.get("dim_order", None))
    _ = kwargs.pop("dim_order", None)
    res = target(*args, **kwargs)
    # assert list(res.dim_order()) == dim_order
    return res


@impl(lib, "_to_dim_order_copy", "CompositeImplicitAutograd")
def _to_dim_order_copy_impl(*args, **kwargs):
    return _op_impl(torch.ops.aten._to_copy, *args, **kwargs)


@impl(lib, "_to_dim_order_copy.out", "CompositeImplicitAutograd")
def _to_dim_order_copy_out_impl(*args, **kwargs):
    return _op_impl(torch.ops.aten._to_copy.out, *args, **kwargs)


"""
Defines a map of aten or edge ops to the corresponding dim_order ops for quick lookup
"""
DimOrderOpsMap = {
    "aten._to_copy.default": exir_ops.edge.dim_order_ops._to_dim_order_copy.default,
}

"""
Defines a map of aten or edge ops to the corresponding memory format ops for quick lookup
"""
MemoryFormatOpsMap = {
    "dim_order_ops._to_dim_order_copy.default": exir_ops.edge.aten._to_copy.default,
}

# If we are replacing an aten op with a dim_order op, we must have a 1:1 mapping through these dicts.
assert len(DimOrderOpsMap) == len(MemoryFormatOpsMap)

# TODO stricter check for 1:1 mapping
