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

    def __init__(
        self,
        edge_program: torch.export.ExportedProgram,
        insert_requantize: bool = False,
    ):
        super().__init__(edge_program)
        # add non-arithmetic operators here if condition met
        self.op_map = {
            exir_ops.edge.aten.permute_copy.default: self._single_io_annotation,
        }
        self.insert_requantize = insert_requantize

    def _single_io_annotation(self, gm: torch.fx.GraphModule, n: torch.fx.node) -> None:
        in_q_attr = n.args[0].meta.get("quant_attrs")
        out_q_attr = n.meta["quant_attrs"]
        if in_q_attr is not None and in_q_attr["dtype"] != out_q_attr["dtype"]:
            if self.insert_requantize:
                dq_attr = n.meta["requantize"]["dq_attrs"]
                q_attr = n.meta["requantize"]["q_attrs"]
                # insert dq with given quantization attribute in input node
                dq = self._insert_quant_node(gm, n, dq_attr["encoding"], dq_attr)
                dq.meta["quant_attrs"] = dq_attr
                # insert q with given quantization attribute in current node
                q = self._insert_quant_node(gm, dq, q_attr["encoding"], q_attr)
                q.meta["quant_attrs"] = q_attr
            else:
                dq_attr = in_q_attr.copy()
                dq_attr["encoding"] = self.q_dq_map[out_q_attr["encoding"]]
                q_attr = out_q_attr.copy()
                n.meta["requantize"] = {"dq_attrs": dq_attr, "q_attrs": q_attr}

    def _insert(self, graph_module: torch.fx.GraphModule) -> torch.fx.GraphModule:
        for n in graph_module.graph.nodes:
            if (
                n.op == "call_function"
                and n.meta.get("quant_attrs")
                and n.target in self.op_map
            ):
                self.op_map[n.target](graph_module, n)
