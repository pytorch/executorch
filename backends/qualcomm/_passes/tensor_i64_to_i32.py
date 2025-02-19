# Copyright (c) Qualcomm Innovation Center, Inc.
# All rights reserved
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import logging

import torch
from executorch.backends.qualcomm.builders.utils import is_graph_output
from executorch.backends.qualcomm.utils.constants import QCOM_ORIG_DTYPE
from executorch.exir import ExirExportedProgram
from executorch.exir.dialects._ops import ops as exir_ops
from executorch.exir.pass_base import ExportPass, PassResult
from executorch.exir.program._program import _get_updated_graph_signature
from torch._subclasses.fake_tensor import FakeTensor


class TensorI64toI32(ExportPass):
    """
    Insert a cast node to cast dtype from int64 to int32.
    This will only be applied on fake tensors.
    """

    cast_ops = {
        torch.ops.aten.argmin.default,
    }

    def __init__(self, edge_program):
        super(TensorI64toI32, self).__init__()
        self.edge_program = edge_program

    # pyre-ignore[2]
    def _is_tensor_of_dtype(self, node_val, dtype: torch.dtype) -> bool:
        return isinstance(node_val, FakeTensor) and node_val.dtype == dtype

    def _cast_to_int32(self, core_ep: ExirExportedProgram):
        copy_op = torch.ops.aten._to_copy.default
        for n in core_ep.exported_program.graph.nodes:
            # Keep track of original output dtype so we ensure the dtype of the graph is consistent with nn.Module
            if is_graph_output(n):
                if isinstance(n.meta["val"], (tuple, list)):
                    dtype_list = [tensor.dtype for tensor in n.meta["val"]]
                    n.meta[QCOM_ORIG_DTYPE] = dtype_list
                else:
                    n.meta[QCOM_ORIG_DTYPE] = n.meta["val"].dtype
                continue
            if n.target in self.cast_ops:
                node_val = n.meta["val"]
                if self._is_tensor_of_dtype(node_val, torch.int64):
                    with core_ep.exported_program.graph.inserting_after(n):
                        users = list(n.users.keys())
                        args = (n,)
                        cast_node = core_ep.exported_program.graph.create_node(
                            "call_function",
                            copy_op,
                            args,
                            {"dtype": torch.int32},
                        )
                        cast_node.meta["val"] = node_val.to(torch.int32)
                        cast_node.args = args

                        for user in users:
                            user.replace_input_with(n, cast_node)

        core_ep.exported_program._graph_signature = _get_updated_graph_signature(
            core_ep.exported_program._graph_signature,
            core_ep.exported_program.graph_module,
        )
        core_ep.exported_program._validate()

    def _preserve_output_dtype(
        self, exported_program: torch.export.exported_program.ExportedProgram
    ):
        graph_module = exported_program.graph_module
        copy_op = exir_ops.edge.aten._to_copy.default
        for n in graph_module.graph.nodes:
            if is_graph_output(n) and QCOM_ORIG_DTYPE in n.meta:
                if isinstance(n.meta["val"], (tuple, list)):
                    for i, dtype in enumerate(n.meta[QCOM_ORIG_DTYPE]):
                        # TODO: Enable this in future to support OP such as topK
                        if n.meta["val"][i].dtype != dtype:
                            raise AssertionError(
                                "Multi output nodes currently don't support casting dtype back."
                            )
                elif n.meta["val"].dtype != n.meta[QCOM_ORIG_DTYPE]:
                    if n.meta[QCOM_ORIG_DTYPE] != torch.int64:
                        logging.warning(
                            "This pass is intended to maintain output as int64 when nn.Module outputs int64. Other dtype modification is detected. Please ensure this is desired."
                        )
                    with graph_module.graph.inserting_after(n):
                        orig_dtype = n.meta[QCOM_ORIG_DTYPE]
                        node_val = n.meta["val"]
                        args = (n,)
                        users = list(n.users.keys())
                        output_users = [
                            user for user in users if user.target == "output"
                        ]
                        cast_node = graph_module.graph.create_node(
                            "call_function",
                            copy_op,
                            args,
                            {"dtype": orig_dtype},
                        )
                        cast_node.meta["val"] = node_val.to(orig_dtype)
                        cast_node.args = args
                        for user in output_users:
                            user.replace_input_with(n, cast_node)

    def call(self, graph_module: torch.fx.GraphModule):
        # Stage 1: _cast_to_int32
        # We add to_copy after the desired operations during this stage because the data type only propagates before to_edge.
        # If we don't add to_copy here but do it after to_edge, the next operation after to_copy() will still expect int64 as its output.
        # Stage 2: _preserve_output_dtype
        # We will tag the output dtype during  stage 1, and we will ensure that if user expects int64 as output,
        # we need to convert the output back to int64 if it is casted from int64->int32 during stage 1.
        if isinstance(self.edge_program, ExirExportedProgram):
            self._cast_to_int32(self.edge_program)
            self.edge_program.exported_program.graph_module.recompile()
        elif isinstance(
            self.edge_program, torch.export.exported_program.ExportedProgram
        ):
            self._preserve_output_dtype(self.edge_program)
        else:
            raise AssertionError(
                "Should be ExirExportedProgram at stage 1 and torch.export.exported_program.ExportedProgram at stage 2"
            )
        return PassResult(graph_module, True)
