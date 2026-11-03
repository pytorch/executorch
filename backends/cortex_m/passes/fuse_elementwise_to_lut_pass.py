# Copyright 2026 Arm Limited and/or its affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from collections.abc import Callable

import torch
from executorch.backends.arm._passes.insert_table_ops import InsertTableOpsPass
from executorch.backends.arm._passes.quant_args import QuantArgs
from executorch.backends.cortex_m.utils import _ACTIVATION_FNS, build_activation_lut
from executorch.exir.dialects._ops import ops as exir_ops
from torch.export import ExportedProgram


class FuseQuantizedElementwiseToLUTPass(InsertTableOpsPass):
    """Use the shared region search with Cortex-M table and rounding semantics."""

    def __init__(self, exported_program: ExportedProgram) -> None:
        super().__init__(exported_program)
        self.table_op = exir_ops.edge.cortex_m.quantized_activation.default
        self.table_targets = set(_ACTIVATION_FNS)
        self.supported_ops = (
            self.table_targets
            | self.arithmetic_ops
            | {self.quantize_op, self.dequantize_op}
        )
        self.supported_dtypes = (torch.int8, torch.float32, torch.float64)

    def generate_table_values(
        self,
        torch_op: Callable[[torch.Tensor], torch.Tensor],
        in_quantargs: QuantArgs,
        out_quantargs: QuantArgs,
    ) -> tuple[torch.Tensor, int]:
        return (
            build_activation_lut(
                None,
                in_quantargs.get_scale_per_tensor(),
                in_quantargs.get_zp_per_tensor(),
                out_quantargs.get_scale_per_tensor(),
                out_quantargs.get_zp_per_tensor(),
                activation_fn=torch_op,
            ),
            0,
        )
