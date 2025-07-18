# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from typing import Callable, Dict, Tuple

import executorch.backends.cortex_m.ops.operators  # noqa
import torch

from executorch.exir.dialects._ops import ops as exir_ops
from executorch.exir.dialects.edge._ops import EdgeOpOverload
from executorch.exir.pass_base import ExportPass, NodeMetadata, ProxyValue


class ReplaceQuantNodesPass(ExportPass):
    """
    Replace quantize and dequantize nodes with the corresponding
    cortex_m.quantize_per_tensor and cortex_m.dequantize_per_tensor nodes.
    """

    @staticmethod
    def _is_qualified_int8_node(args) -> bool:
        return (
            args[3] >= torch.iinfo(torch.int8).min  # qmin
            and args[4] <= torch.iinfo(torch.int8).max  # qmax
            and args[5] == torch.int8  # dtype
        )

    def __init__(self):
        super().__init__()
        self.op_replacements = {
            exir_ops.edge.add: {
                "new_target": exir_ops.edge.cortex_m.add,
                "qualifier": lambda args: True,
            },
            exir_ops.edge.aten.add.Tensor: {
                "new_target": exir_ops.edge.cortex_m.add.Tensor,
                "qualifier": lambda args: True,
            },
            exir_ops.edge.aten._softmax.out: {
                "new_target": exir_ops.edge.cortex_m.softmax.out,
                "qualifier": lambda args: True,
            },
            exir_ops.edge.aten._softmax.default: {
                "new_target": exir_ops.edge.cortex_m.softmax,  # or .softmax if you have an out variant
                "qualifier": lambda args: True,
            },
            exir_ops.edge.quantized_decomposed.quantize_per_tensor.default: {
                "new_target": exir_ops.edge.cortex_m.quantize_per_tensor.default,
                "qualifier": self._is_qualified_int8_node,
            },
            exir_ops.edge.quantized_decomposed.dequantize_per_tensor.default: {
                "new_target": exir_ops.edge.cortex_m.dequantize_per_tensor.default,
                "qualifier": self._is_qualified_int8_node,
            },
        }

    def call_operator(
        self,
        op: Callable[..., object],
        args: Tuple[object, ...],
        kwargs: Dict[str, object],
        meta: NodeMetadata,
    ) -> ProxyValue:
        assert isinstance(
            op, EdgeOpOverload
        ), "Op must be an EdgeOpOverload. Run this pass after to_edge()."
        print(f"[ReplaceQuantNodesPass] Operator called: {op}, Args: {args}")

        if op in self.op_replacements and self.op_replacements[op]["qualifier"](args):            
            print(f"[ReplaceQuantNodesPass] Replacing {op} with {self.op_replacements[op]['new_target']}")       
            return super().call_operator(
                    self.op_replacements[op]["new_target"],
                    args,
                    kwargs,
                    meta,
                )
        return super().call_operator(op, args, kwargs, meta)
