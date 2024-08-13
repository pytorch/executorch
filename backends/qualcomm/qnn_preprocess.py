# Copyright (c) Qualcomm Innovation Center, Inc.
# All rights reserved
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import logging
from collections import defaultdict
from typing import final, List

import executorch.backends.qualcomm.python.PyQnnManagerAdaptor as PyQnnManager

import torch  # noqa: F401
from executorch.backends.qualcomm.builders.node_visitor import get_node_visitors
from executorch.backends.qualcomm.builders.qnn_constants import OpContextLoader
from executorch.backends.qualcomm.passes.convert_to_linear import ConvertToLinear
from executorch.backends.qualcomm.passes.fuse_consecutive_transpose import (
    FuseConsecutiveTranspose,
)
from executorch.backends.qualcomm.passes.insert_io_qdq import InsertIOQDQ
from executorch.backends.qualcomm.passes.insert_requantize import InsertRequantize
from executorch.backends.qualcomm.passes.layout_transform import LayoutTransform
from executorch.backends.qualcomm.utils.utils import generate_qnn_executorch_option
from executorch.exir.backend.backend_details import (
    BackendDetails,
    CompileSpec,
    PreprocessResult,
)
from executorch.exir.passes import PassManager
from torch.export.exported_program import ExportedProgram

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
                ConvertToLinear(),
                InsertRequantize(edge_program),
                InsertIOQDQ(edge_program),
                LayoutTransform(edge_program, insert_permute=True),
                FuseConsecutiveTranspose(),
            ]
        )

        pass_result = qnn_compiler_passes(edge_program.graph_module)
        assert pass_result is not None

        enable_tensor_dump = qnn_manager.IsTensorDump()
        nodes_to_wrappers = defaultdict(dict)
        node_visitors = get_node_visitors(
            edge_program, enable_tensor_dump=enable_tensor_dump
        )
        py_op_wrapper_list = []
        for node in pass_result.graph_module.graph.nodes:
            if node.op == "call_function":
                logger.info(f"Visiting: {node}, {node.target.__name__}")
                if node.target.__name__ in node_visitors:
                    py_op_wrapper = node_visitors[node.target.__name__].define_node(
                        node, nodes_to_wrappers
                    )
                    if py_op_wrapper is not None:
                        if isinstance(py_op_wrapper, List):
                            py_op_wrapper_list.extend(py_op_wrapper)
                        else:
                            py_op_wrapper_list.append(py_op_wrapper)
                else:
                    err_msg = (
                        f"For {node}, {node.op}:{node.target.__name__} "
                        "is not supported in Qnn Delegate"
                    )
                    try:
                        context_loader_target = eval(
                            f"torch.ops.{OpContextLoader.namespace}.{node.name}.default",
                            globals().update(torch.__dict__),
                        )
                        assert node.target == context_loader_target, err_msg
                        # if graph has context binary loader node, return directly
                        return PreprocessResult(
                            processed_bytes=node.meta[OpContextLoader.meta_ctx_bin],
                            debug_handle_map={},
                        )
                    except:
                        raise RuntimeError(err_msg)

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
        # For now, debug_handle_map is not used by QNN ExecuTorch
        return PreprocessResult(
            processed_bytes=bytes(qnn_context_binary), debug_handle_map={}
        )
