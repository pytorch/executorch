# Copyright (c) Qualcomm Innovation Center, Inc.
# All rights reserved
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import logging
from typing import final, List

import executorch.backends.qualcomm.python.PyQnnManagerAdaptor as PyQnnManager
from executorch.backends.qualcomm.builders.node_visitor import get_node_visitors

from executorch.backends.qualcomm.passes.convert_addmm_back_to_linear import (
    ConvertAddmmmmWithLinear,
)
from executorch.backends.qualcomm.passes.insert_io_qdq import InsertIOQDQ
from executorch.backends.qualcomm.passes.layout_transform import LayoutTransform
from executorch.backends.qualcomm.utils.utils import generate_qnn_executorch_option
from executorch.exir.backend.backend_details import (
    BackendDetails,
    CompileSpec,
    PreprocessResult,
)
from executorch.exir.passes import PassManager
from torch._export.exported_program import ExportedProgram

DEFAULT_DEBUG_HANDLE = 65535

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)


@final
class QnnBackend(BackendDetails):
    @staticmethod
    def preprocess(
        edge_program: ExportedProgram,
        compile_specs: List[CompileSpec],
    ) -> PreprocessResult:
        option = generate_qnn_executorch_option(compile_specs)
        qnn_manager = PyQnnManager.QnnManager(option)
        qnn_manager.Init()

        # QNN Delegate Specific Passes
        qnn_compiler_passes = PassManager(
            passes=[
                ConvertAddmmmmWithLinear(),
                InsertIOQDQ(edge_program),
                LayoutTransform(edge_program, insert_permute=True),
            ]
        )

        pass_result = qnn_compiler_passes(edge_program.graph_module)
        assert pass_result is not None

        nodes_to_wrappers = {}
        node_visitors = get_node_visitors(edge_program)
        py_op_wrapper_list = []
        for node in pass_result.graph_module.graph.nodes:
            if node.op == "call_function":
                logger.info(f"Visiting: {node}, {node.target.__name__}")
                if node.target.__name__ in node_visitors:
                    py_op_wrapper = node_visitors[node.target.__name__].define_node(
                        node, nodes_to_wrappers
                    )
                    if py_op_wrapper is not None:
                        py_op_wrapper_list.append(py_op_wrapper)
                else:
                    raise RuntimeError(
                        f"For {node}, {node.op}:{node.target.__name__} is not supported in Qnn Delegate"
                    )
            elif node.op in [
                "get_attr",
                "placeholder",
                "output",
            ]:
                continue
            else:
                raise RuntimeError(f"{node.op} is not supported in Qnn")

        qnn_context_binary = qnn_manager.Compile(
            [py_op_wrapper.GetOpWrapper() for py_op_wrapper in py_op_wrapper_list]
        )
        assert len(qnn_context_binary) != 0, "Failed to generate Qnn context binary."
        qnn_manager.Destroy()

        return PreprocessResult(bytes(qnn_context_binary))
