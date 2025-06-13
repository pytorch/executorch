# Copyright 2025 Arm Limited and/or its affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.


import torch

import torch.fx as fx
from executorch.backends.arm.operator_support.tosa_supported_operators import (
    register_tosa_support_check,
    SupportedTOSAOperatorCheck,
)
from executorch.backends.arm.tosa_specification import TosaSpecification
from executorch.exir.dialects._ops import ops as exir_ops


@register_tosa_support_check
class EmbeddingSupported(SupportedTOSAOperatorCheck):
    targets = [exir_ops.edge.aten.embedding.default]

    tosa_specs = [
        TosaSpecification.create_from_string("TOSA-0.80+BI"),
        TosaSpecification.create_from_string("TOSA-0.80+MI"),
        TosaSpecification.create_from_string("TOSA-1.0+INT"),
        TosaSpecification.create_from_string("TOSA-1.0+FP"),
    ]

    def is_node_tosa_supported(
        self, node: fx.Node, tosa_spec: TosaSpecification
    ) -> bool:  # type: ignore[override, misc]
        # Note aten.embedding.default requires int64 indices and TOSA does not support it.
        # Int32 indices here for aten.embedding.default is ok since it will be decomposed into ops that can handle it.
        assert (
            len(node.all_input_nodes) == 2
        ), "Number of inputs to aten.embedding is not 2"
        indices_val = node.all_input_nodes[1].meta["val"]
        indices_dtype = indices_val.dtype

        if indices_dtype != torch.int32:
            self.reporter.report_reject(
                node,
                f"Indices dtype {indices_val.dtype} is not supported in {node.target}.",
            )
            return False

        return True
