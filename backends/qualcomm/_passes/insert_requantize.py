# Copyright (c) Qualcomm Innovation Center, Inc.
# All rights reserved
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import torch

from executorch.backends.qualcomm.utils.constants import (
    QCOM_QUANT_ATTRS,
    QCOM_QUANTIZED_IO,
    QCOM_REQUANTIZE,
)

from executorch.exir.dialects._ops import ops as exir_ops
from executorch.exir.pass_base import ExportPass, PassResult


class InsertRequantize(ExportPass):
    """
    This pass inserts convert op for operators which have
    different quantization specs in input and activation.
    Convert OP is a specific op which helps to requantize in Qnn backend
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
    ):
        super(InsertRequantize, self).__init__()
        self.edge_program = edge_program

    # TODO: Implement this function when we have an op with
    # multiple outputs that requires quant attributes.
    def _multi_output_annotation(self) -> None:
        raise NotImplementedError("requant is not implemented for multi output yet")

    def _single_output_annotation(
        self, gm: torch.fx.GraphModule, n: torch.fx.node
    ) -> None:
        with gm.graph.inserting_after(n):
            users = list(n.users.keys())
            inserted_n = gm.graph.create_node(
                "call_function",
                exir_ops.edge.aten._to_copy.default,
                (n,),
            )

            inserted_n.meta["val"] = n.meta["val"]
            inserted_n.meta[QCOM_QUANT_ATTRS] = n.meta.pop(QCOM_REQUANTIZE)
            if n.meta.get(QCOM_QUANTIZED_IO):
                inserted_n.meta[QCOM_QUANTIZED_IO] = n.meta[QCOM_QUANTIZED_IO]

            for user in users:
                user.replace_input_with(n, inserted_n)

    def _insert(self, graph_module: torch.fx.GraphModule) -> torch.fx.GraphModule:
        for n in graph_module.graph.nodes:
            if QCOM_REQUANTIZE in n.meta:
                (
                    self._single_output_annotation(graph_module, n)
                    if isinstance(
                        n.meta["val"], torch._subclasses.fake_tensor.FakeTensor
                    )
                    or n.target in self.multi_output_op_ignore_set
                    else self._multi_output_annotation()
                )

    def call(self, graph_module: torch.fx.GraphModule):
        self._insert(graph_module)
        graph_module.graph.eliminate_dead_code()
        graph_module.recompile()
        return PassResult(graph_module, True)
