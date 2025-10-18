# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
# Copyright 2025 Arm Limited and/or its affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from typing import Callable, Dict, Tuple

import executorch.backends.cortex_m.ops.operators  # noqa
import torch

from executorch.exir.dialects._ops import ops as exir_ops
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
            exir_ops.edge.quantized_decomposed.quantize_per_tensor.default: {
                "new_target": exir_ops.edge.cortex_m.quantize_per_tensor.default,
                "qualifier": self._is_qualified_int8_node,
            },
            exir_ops.edge.quantized_decomposed.dequantize_per_tensor.default: {
                "new_target": exir_ops.edge.cortex_m.dequantize_per_tensor.default,
                "qualifier": self._is_qualified_int8_node,
            },
        }
        self.disallowed_targets = {
            torch.ops.quantized_decomposed.dequantize_per_tensor.default,
            torch.ops.quantized_decomposed.quantize_per_tensor.default,
        }

    def call_operator(
        self,
        op: Callable[..., object],
        args: Tuple[object, ...],
        kwargs: Dict[str, object],
        meta: NodeMetadata,
    ) -> ProxyValue:
        if op in self.disallowed_targets:
            raise RuntimeError(
                f"Found unexpected aten op '{op}'. Make sure you run this pass after lowering to edge."
            )

        if op in self.op_replacements and self.op_replacements[op]["qualifier"](args):
            return super().call_operator(
                self.op_replacements[op]["new_target"],
                args,
                kwargs,
                meta,
            )
        return super().call_operator(op, args, kwargs, meta)
