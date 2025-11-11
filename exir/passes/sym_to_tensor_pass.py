# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

from typing import Union

import torch
from executorch.exir.pass_base import ExportPass, map_args, NodeMetadata, ProxyValue
from torch import SymBool, SymFloat, SymInt
from torch.utils._pytree import PyTree


class SymToTensorPass(ExportPass):
    """
    The dispatcher implicitly converts SymInt/SymFloats to tensors, but
    sometimes this doesn't comply with the operator's schema which ExecuTorch
    heavily relies on. So this pass inserts a
    torch.ops.aten.scalar_tensor.default operator before these SymInts are used
    so that it matches the schema of the operator.
    """

    # pyre-ignore
    def call_operator(self, op, args, kwargs, meta: NodeMetadata):
        # pyre-ignore
        def is_sym(value, arg) -> bool:
            if isinstance(value, ProxyValue) and not value.is_tensor():
                if isinstance(arg.type, torch.TensorType) and type(value.data) in {
                    SymInt,
                    SymFloat,
                    SymBool,
                }:
                    return True
            return False

        def corresponding_dtype(
            symbol: Union[SymInt, SymFloat, SymBool]
        ) -> torch.dtype:
            if isinstance(symbol, SymInt):
                return torch.int32
            elif isinstance(symbol, SymFloat):
                return torch.float32
            elif isinstance(symbol, SymBool):
                return torch.bool
            else:
                raise AssertionError(f"Unsupported data type: {type(symbol)}")

        def try_coerce(value: PyTree, arg: torch.Argument) -> PyTree:
            if is_sym(value, arg):
                return self.call_operator(
                    torch.ops.aten.scalar_tensor.default,
                    (value,),
                    {"dtype": corresponding_dtype(value.data)},
                    meta,
                )
            else:
                return value

        args, kwargs = map_args(op, try_coerce, args, kwargs)

        return super().call_operator(op, args, kwargs, meta)
