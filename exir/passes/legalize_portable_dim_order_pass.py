# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from typing import Optional

import executorch.exir.passes.dim_order_ops_registry  # noqa: F401

import torch
from executorch.exir.dialects._ops import ops as exir_ops
from executorch.exir.dim_order_utils import (
    get_dim_order,
    is_channel_last_dim_order,
    is_contiguous_dim_order,
)
from executorch.exir.pass_base import ExportPass, NodeMetadata, ProxyValue
from torch._prims_common import is_contiguous_for_memory_format_or_false
from torch.fx.experimental.symbolic_shapes import statically_known_true


# aten.copy_ writes mutated buffers and inputs back, and reinplace_pass can
# turn a copy into copy_.
_IN_PLACE_COPY_OPS = (torch.ops.aten.copy_.default, exir_ops.edge.aten.copy_.default)


def _meta_with_own_memory(value: ProxyValue) -> NodeMetadata:
    # reinplace_pass may let a node reuse the memory of its input. A copy of the
    # node's output into another layout needs memory of its own.
    meta = NodeMetadata(value.node.meta)
    meta.data.pop("_share_alloc_with_arg_idx", None)
    meta.data.pop("_shared_alloc_offset", None)
    return meta


def _same_sizes(a, b) -> bool:
    # Tracing can assume that a symbolic size is not 1, and so prove it equal to
    # a static size, although it can be 1 at runtime. Compare the sizes as
    # written instead.
    return len(a) == len(b) and all(
        (isinstance(x, int) and isinstance(y, int) and x == y)
        or (
            isinstance(x, torch.SymInt)
            and isinstance(y, torch.SymInt)
            and x.node.expr == y.node.expr
        )
        for x, y in zip(a, b)
    )


def _copy_is_correct(op, destination, source, memory_format, has_dim_order) -> bool:
    if op in _IN_PLACE_COPY_OPS and not has_dim_order(source):
        return False
    # A single value is the same in any memory order.
    if statically_known_true(source.numel() == 1):
        return True
    # The kernels treat missing leading dims as size 1.
    source = source[(None,) * (destination.dim() - source.dim())]
    if _same_sizes(source.shape, destination.shape):
        return is_contiguous_for_memory_format_or_false(
            source, memory_format=memory_format
        )
    # A symbolic size can be 1 at runtime even when tracing assumes it is not.
    broadcasts = any(
        isinstance(s, int) and isinstance(d, int) and s != d
        for s, d in zip(source.shape, destination.shape)
    )
    return broadcasts and is_contiguous_for_memory_format_or_false(
        destination, memory_format=torch.contiguous_format
    )


class LegalizePortableDimOrderPass(ExportPass):
    """Insert dim-order copies before portable ops that need another layout.

    Runs during ``edge -> executorch`` lowering on leftover edge ops whose
    portable kernels require default dim order on their primary input. The
    portable ``copy`` kernels read the source in the memory order of the tensor
    they copy into, except when they broadcast: then they write that tensor as
    if it were contiguous. The in-place ``copy_`` also needs both tensors to
    have the same dim order. When a copy would go wrong, the source is first
    copied into the dim order of the destination, and into its shape when the
    destination is channels last.
    """

    _copy_op = exir_ops.edge.dim_order_ops._to_dim_order_copy.default
    _target_ops = {
        exir_ops.edge.aten._adaptive_avg_pool2d.default,
        exir_ops.edge.aten.avg_pool2d.default,
        exir_ops.edge.aten.expand_copy.default,
        exir_ops.edge.aten.mean.dim,
        exir_ops.edge.aten.sum.dim_IntList,
    }

    def call_operator(self, op, args, kwargs, meta):
        if op == exir_ops.edge.aten.copy.default or op in _IN_PLACE_COPY_OPS:
            return self._legalize_copy(op, args, kwargs, meta)
        if op not in self._target_ops:
            return super().call_operator(op, args, kwargs, meta)

        input_arg = args[0]
        if isinstance(input_arg, ProxyValue) and input_arg.is_tensor():
            input_tensor: Optional[torch.Tensor] = input_arg.to_tensor()
            input_meta = _meta_with_own_memory(input_arg)
        elif isinstance(input_arg, torch.Tensor):
            input_tensor = input_arg
            input_meta = meta.copy()
        else:
            input_tensor = None
            input_meta = meta.copy()

        if input_tensor is None or tuple(
            int(d) for d in input_tensor.dim_order()
        ) == tuple(range(input_tensor.dim())):
            return super().call_operator(op, args, kwargs, meta)

        contiguous_dim_order = get_dim_order(
            torch.contiguous_format, input_tensor.dim()
        )
        assert contiguous_dim_order is not None

        legalized_input = super().call_operator(
            self._copy_op,
            (input_arg,),
            {"dim_order": contiguous_dim_order},
            input_meta,
        )
        return super().call_operator(
            op,
            (legalized_input, *args[1:]),
            kwargs,
            meta,
        )

    def _legalize_copy(self, op, args, kwargs, meta):
        destination, source = args[0], args[1]
        if not (isinstance(destination, ProxyValue) and isinstance(source, ProxyValue)):
            return super().call_operator(op, args, kwargs, meta)
        destination_tensor = destination.to_tensor()
        # exir can convert a tensor only to these two dim orders.
        if is_channel_last_dim_order(destination_tensor):
            memory_format = torch.channels_last
            has_dim_order = is_channel_last_dim_order
        elif is_contiguous_dim_order(destination_tensor):
            memory_format = torch.contiguous_format
            has_dim_order = is_contiguous_dim_order
        else:
            return super().call_operator(op, args, kwargs, meta)
        if _copy_is_correct(
            op, destination_tensor, source.to_tensor(), memory_format, has_dim_order
        ):
            return super().call_operator(op, args, kwargs, meta)

        if memory_format == torch.channels_last and not _same_sizes(
            source.to_tensor().shape, destination_tensor.shape
        ):
            # Through self, so that the pass legalizes the input of expand_copy too.
            source = self.call_operator(
                exir_ops.edge.aten.expand_copy.default,
                (source, self.call_size_operator_all(destination, meta)),
                {},
                _meta_with_own_memory(source),
            )
        source_tensor = source.to_tensor()
        if not has_dim_order(source_tensor):
            source = super().call_operator(
                self._copy_op,
                (source,),
                {"dim_order": get_dim_order(memory_format, source_tensor.dim())},
                _meta_with_own_memory(source),
            )
        return super().call_operator(op, (destination, source, *args[2:]), kwargs, meta)
