# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

import executorch.exir.dialects.fused_quant.ops  # noqa: F401
import torch
from executorch.backends.transforms.fused_quant.graph_utils import (
    get_qparams_flat,
    get_qparams_from_node,
    get_zero_point,
)
from executorch.backends.transforms.permute_pass_utils import get_arg
from executorch.backends.transforms.utils import add_constant, get_constant
from executorch.exir.dialects._ops import ops as exir_ops
from executorch.exir.pass_base import ExportedProgramPassBase, ExportedProgramPassResult
from executorch.exir.passes.constant_prop_pass import constant_prop_pass
from torch import fx
from torch.export import ExportedProgram
from torch.export.graph_signature import InputKind


class ReplaceSubConstantWithAdd(ExportedProgramPassBase):
    """Rewrite ``fused_quant.sub(x, c)`` with a quantized constant ``c`` as
    ``fused_quant.add(x, -c)``, negating ``c`` in the integer domain.
    """

    def call(self, exported_program: ExportedProgram) -> ExportedProgramPassResult:
        graph = exported_program.graph_module.graph
        modified = False
        for node in graph.find_nodes(
            op="call_function", target=exir_ops.edge.fused_quant.sub.default
        ):
            other_qparams = get_qparams_from_node(node, "other")
            if other_qparams is None or not other_qparams.is_per_tensor():
                continue
            codes = get_constant(exported_program, get_arg(node, "other", fx.Node))
            if codes is None or codes.is_floating_point():
                continue

            # Reflecting codes and zero point across [quant_min, quant_max] negates
            # the dequantized value exactly at the same scale.
            reflection = other_qparams.quant_min + other_qparams.quant_max
            negated_codes = add_constant(
                exported_program,
                f"{node.name}_negated_other",
                reflection - codes,
                node,
                InputKind.CONSTANT_TENSOR,
            )
            negated_zero_point = add_constant(
                exported_program,
                f"{node.name}_negated_other_zero_point",
                torch.tensor(
                    reflection - get_zero_point(exported_program, other_qparams)
                ),
                node,
                InputKind.CONSTANT_TENSOR,
            )

            with graph.inserting_before(node):
                add_node = graph.call_function(
                    exir_ops.edge.fused_quant.add.default,
                    args=(
                        get_arg(node, "inp"),
                        negated_codes,
                        *get_qparams_flat(node, "inp"),
                        other_qparams.scale,
                        negated_zero_point,
                        other_qparams.dtype,
                        other_qparams.quant_min,
                        other_qparams.quant_max,
                        *get_qparams_flat(node, "out"),
                        get_arg(node, "alpha"),
                    ),
                )
                add_node.meta = node.meta.copy()

            node.replace_all_uses_with(add_node)
            graph.erase_node(node)
            modified = True

        if modified:
            exported_program = constant_prop_pass(exported_program)
        return ExportedProgramPassResult(exported_program, modified)
