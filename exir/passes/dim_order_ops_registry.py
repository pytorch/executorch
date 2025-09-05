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

lib.define(
    "_empty_dim_order(int[] size, *, ScalarType? dtype=None, Layout? layout=None, Device? device=None, bool? pin_memory=None, int[]? dim_order=None) -> Tensor"
)

# Out variant of aten::_to_copy and aten::empty drops TensorOptions, so do their dim order variants
lib.define(
    "_to_dim_order_copy.out(Tensor self, *, bool non_blocking=False, int[]? dim_order=None, Tensor(a!) out) -> Tensor(a!)"
)

lib.define(
    "_empty_dim_order.out(int[] size, *, int[]? dim_order=None, Tensor(a!) out) -> Tensor(a!)"
)

lib.define(
    "_clone_dim_order(Tensor self, *, bool non_blocking=False, int[]? dim_order=None) -> Tensor"
)

lib.define(
    "_clone_dim_order.out(Tensor self, *, bool non_blocking=False, int[]? dim_order=None, Tensor(a!) out) -> Tensor(a!)"
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


@impl(lib, "_empty_dim_order", "CompositeImplicitAutograd")
def _empty_dim_order_impl(*args, **kwargs):
    return _op_impl(torch.ops.aten.empty.memory_format, *args, **kwargs)


@impl(lib, "_empty_dim_order.out", "CompositeImplicitAutograd")
def _empty_dim_order_out_impl(*args, **kwargs):
    return _op_impl(torch.ops.aten.empty.out, *args, **kwargs)


@impl(lib, "_clone_dim_order", "CompositeImplicitAutograd")
def _clone_dim_order_impl(*args, **kwargs):
    return _op_impl(torch.ops.aten.clone.default, *args, **kwargs)


@impl(lib, "_clone_dim_order.out", "CompositeImplicitAutograd")
def _clone_dim_order_out_impl(*args, **kwargs):
    return _op_impl(torch.ops.aten.clone.out, *args, **kwargs)


"""
Defines a map of edge ops to the corresponding dim_order ops for quick lookup
"""
DimOrderOpsMap = {
    exir_ops.edge.aten._to_copy.default: exir_ops.edge.dim_order_ops._to_dim_order_copy.default,
    exir_ops.edge.aten.empty.memory_format: exir_ops.edge.dim_order_ops._empty_dim_order.default,
    exir_ops.edge.aten.clone.default: exir_ops.edge.dim_order_ops._clone_dim_order.default,
}

"""
Defines a map of edge ops to the corresponding memory format ops for quick lookup, which is the revert of DimOrderOpsMap
"""
MemoryFormatOpsMap = {v: k for k, v in DimOrderOpsMap.items()}
