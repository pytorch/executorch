# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import copy
import logging

import torch
from executorch.exir.dialects.edge._ops import EdgeOpOverload
from executorch.exir.dim_order_utils import get_dim_order
from executorch.exir.pass_base import ExportPass
from executorch.exir.passes.dim_order_ops_registry import DimOrderOpsMap

from torch._export.pass_infra.proxy_value import ProxyValue

logger = logging.getLogger(__file__)
logger.setLevel(logging.INFO)


class MemoryFormatOpsPass(ExportPass):
    """
    This pass replaces ops which takes torch.memory_format as an argument with
    'equivalent' op which takes dim_order. This is towards the larger ExecuTorch
    goal to move away from torch.memory_format. There is a 1:1 mapping between
    the aten op and the new edge dialect dim_order op.
    """

    def call_operator(self, op, args, kwargs, meta):
        if not (isinstance(op, EdgeOpOverload) and op.__name__ in DimOrderOpsMap):
            return super().call_operator(
                op,
                args,
                kwargs,
                meta,
            )
        # new kwargs with dim_order, and no memory_format for the new op
        nkwargs = dict(copy.deepcopy(kwargs))  # orig kwargs are immutable

        # get the "to" memory format for the EdgeOp
        mem_format = nkwargs.pop("memory_format", torch.contiguous_format)

        # can always get the shape, assuming rank is specialized
        if isinstance(args[0], ProxyValue) and args[0].is_tensor():
            ndim = args[0].to_tensor().dim()
        elif isinstance(args[0], torch.Tensor):
            ndim = args[0].dim()
        else:
            assert 0, f"Expecting a Tensor or a ProxyValue buy got {type(args[0])}"

        nkwargs["dim_order"] = get_dim_order(mem_format, ndim)
        logger.debug(
            f"_to_copy = rank: {ndim}, memory_format: {mem_format}."
            f" _to_dim_order_copy = dim_order: {nkwargs['dim_order']}"
        )

        t = DimOrderOpsMap[op.__name__]

        return super().call_operator(
            t,
            args,
            nkwargs,
            meta,
        )
