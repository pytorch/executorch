# Copyright (c) Qualcomm Innovation Center, Inc.
# All rights reserved
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import torch

from executorch.backends.qualcomm.passes.insert_io_qdq import InsertIOQDQ
from executorch.exir.dialects._ops import ops as exir_ops


class InsertRequantize(InsertIOQDQ):
    """
    This pass inserts dq/q nodes for non-arithmetic operators which have
    different quantization specs in input and activation
    """

    # Storing ops that has multi output but run _single_output_annotation logic
    # instead of _multi_output_annotation. Ops might be added into this set because
    # we don't use the 2nd output, 2nd output is an integer, etc.
    multi_output_op_ignore_set = {
        exir_ops.edge.aten._native_batch_norm_legit_no_training.default,
    }

    def __init__(
        self,
        edge_program: torch.export.ExportedProgram,
        insert_requantize: bool = False,
    ):
        super().__init__(edge_program)
        self.insert_requantize = insert_requantize

    # TODO: Implement this function when we have an op with
    # multiple outputs that requires quant attributes.
    def _multi_output_annotation(self) -> None:
        raise NotImplementedError("requant is not implemented for multi output yet")

    def _single_output_annotation(
        self, gm: torch.fx.GraphModule, n: torch.fx.node
    ) -> None:
        dq_attr = n.meta["quant_attrs"]
        q_attr = n.meta["requantize"]
        # insert dq with given quantization attribute in input node
        dq = self._insert_quant_node(
            gm, n, InsertIOQDQ.q_dq_map[q_attr["encoding"]], dq_attr
        )
        dq.meta["quant_attrs"] = dq_attr
        # insert q with given quantization attribute in current node
        q = self._insert_quant_node(gm, dq, q_attr["encoding"], q_attr)
        q.meta["quant_attrs"] = q_attr

    def _insert(self, graph_module: torch.fx.GraphModule) -> torch.fx.GraphModule:
        for n in graph_module.graph.nodes:
            if "requantize" in n.meta:
                (
                    self._single_output_annotation(graph_module, n)
                    if len(n.meta["val"]) == 1
                    or n.target in self.multi_output_op_ignore_set
                    else self._multi_output_annotation()
                )
