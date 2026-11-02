# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import copy
import logging
from typing import List, Optional

import torch
from executorch.exir.dialects.edge._ops import EdgeOpOverload
from executorch.exir.dim_order_utils import get_dim_order, get_memory_format
from executorch.exir.pass_base import ExportPass, ProxyValue
from executorch.exir.passes.dim_order_ops_registry import (
    DimOrderOpsMap,
    MemoryFormatOpsMap,
)
from executorch.exir.tensor import dim_order_from_stride

logger = logging.getLogger(__file__)
logger.setLevel(logging.INFO)


class ReplaceEmptyLikePass(ExportPass):
    """
    Replaces aten.empty_like with aten.empty.memory_format.

    It runs after view ops have become copies and no-op views are removed, so
    a preserve_format buffer gets the layout its input has in the Edge
    program. The eager strides can differ: a permuted view is a contiguous
    copy here.
    The replacement takes only the input's sizes, so with static shapes it
    does not depend on the input and partitioners can merge around it.
    Any layout other than contiguous or 4-D channels last becomes
    aten.empty_permuted, as PyTorch's decomposition does.
    """

    def call_operator(self, op, args, kwargs, meta):
        if op != torch.ops.aten.empty_like.default:
            return super().call_operator(op, args, kwargs, meta)

        source = args[0]
        tensor = source.to_tensor()
        sizes = self.call_size_operator_all(source, meta)
        new_kwargs = dict(kwargs)
        memory_format = new_kwargs.pop("memory_format", None)
        new_kwargs["dtype"] = kwargs.get("dtype") or tensor.dtype
        new_kwargs["layout"] = kwargs.get("layout") or tensor.layout
        new_kwargs["device"] = kwargs.get("device") or tensor.device

        if memory_format in (None, torch.preserve_format):
            dim_order = [
                int(d)
                for d in dim_order_from_stride(tensor.stride(), tuple(tensor.shape))
            ]
            if dim_order not in (list(range(tensor.dim())), [0, 2, 3, 1]):
                return super().call_operator(
                    torch.ops.aten.empty_permuted.default,
                    (sizes, dim_order),
                    new_kwargs,
                    meta,
                )
            memory_format = get_memory_format(dim_order)
        new_kwargs["memory_format"] = memory_format

        return super().call_operator(
            torch.ops.aten.empty.memory_format, (sizes,), new_kwargs, meta
        )


class MemoryFormatOpsPass(ExportPass):
    """
    This pass replaces ops which takes torch.memory_format as an argument with
    'equivalent' op which takes dim_order. This is towards the larger ExecuTorch
    goal to move away from torch.memory_format. There is a 1:1 mapping between
    the aten op and the new edge dialect dim_order op.
    """

    def call_operator(self, op, args, kwargs, meta):
        if not (isinstance(op, EdgeOpOverload) and op in DimOrderOpsMap):
            return super().call_operator(
                op,
                args,
                kwargs,
                meta,
            )

        # new kwargs with dim_order, and no memory_format for the new op
        nkwargs = dict(copy.deepcopy(kwargs))  # orig kwargs are immutable

        # Get the target memory format for the EdgeOp, defaulting to
        # preserve_format (clone() with no memory_format kwarg preserves
        # the input's layout instead of forcing contiguous).
        mem_format = nkwargs.pop("memory_format", torch.preserve_format)

        # Get input tensor and ndim
        input_tensor: Optional[torch.Tensor] = None
        if isinstance(args[0], ProxyValue) and args[0].is_tensor():
            input_tensor = args[0].to_tensor()
            ndim = input_tensor.dim()
        elif isinstance(args[0], torch.Tensor):
            input_tensor = args[0]
            ndim = input_tensor.dim()
        elif isinstance(args[0], torch.fx.immutable_collections.immutable_list):
            ndim = len(args[0])
        else:
            assert (
                0
            ), f"Expecting a Tensor, a ProxyValue, or a Sequence, but got {type(args[0])}"

        # Derive dim_order based on memory format
        dim_order: List[int]
        if mem_format in (None, torch.preserve_format):
            # preserve_format: inherit dim_order from input tensor
            if input_tensor is not None:
                dim_order = [int(d) for d in input_tensor.dim_order()]
            else:
                # Fallback to contiguous if no single input tensor is available
                # (e.g. list inputs like torch.stack).
                dim_order = list(range(ndim))
        else:
            # Explicit memory format (contiguous_format, channels_last, etc.)
            dim_order = get_dim_order(mem_format, ndim)

        nkwargs["dim_order"] = dim_order
        logger.debug(
            f"{op.__name__} = rank: {ndim}, memory_format: {mem_format}."
            f" {DimOrderOpsMap[op].__name__} = dim_order: {nkwargs['dim_order']}"
        )

        t = DimOrderOpsMap[op]

        return super().call_operator(
            t,
            args,
            nkwargs,
            meta,
        )


class DimOrderOpsRevertPass(ExportPass):
    """
    This pass is to revert the dim_order ops back to the memory format ops.
    """

    def call_operator(self, op, args, kwargs, meta):
        if not (isinstance(op, EdgeOpOverload) and op in MemoryFormatOpsMap):
            return super().call_operator(
                op,
                args,
                kwargs,
                meta,
            )

        # new kwargs with dim_order, and no memory_format for the new op
        nkwargs = dict(copy.deepcopy(kwargs))  # orig kwargs are immutable

        # can always get the shape, assuming rank is specialized
        if isinstance(args[0], ProxyValue) and args[0].is_tensor():
            ndim = args[0].to_tensor().dim()
        elif isinstance(args[0], torch.Tensor):
            ndim = args[0].dim()
        elif isinstance(args[0], torch.fx.immutable_collections.immutable_list):
            ndim = len(args[0])
        else:
            assert 0, f"Expecting a Tensor or a ProxyValue but got {type(args[0])}"

        # get the "to" memory format for the EdgeOp
        default_dim_order = list(range(ndim))
        dim_order = nkwargs.pop("dim_order", default_dim_order)

        nkwargs["memory_format"] = get_memory_format(dim_order)

        logger.debug(
            f" {op.__name__} = dim_order: {dim_order}."
            f" {MemoryFormatOpsMap[op].__name__} = rank: {ndim}, memory_format: {nkwargs['memory_format']}."
        )

        t = MemoryFormatOpsMap[op]

        return super().call_operator(
            t,
            args,
            nkwargs,
            meta,
        )
