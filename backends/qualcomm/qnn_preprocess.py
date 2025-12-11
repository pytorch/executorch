# Copyright (c) Qualcomm Innovation Center, Inc.
# All rights reserved
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import logging
from collections import defaultdict
from typing import Dict, final, List

import executorch.backends.qualcomm.python.PyQnnManagerAdaptor as PyQnnManager

import torch  # noqa: F401
from executorch.backends.qualcomm._passes.qnn_pass_manager import QnnPassManager
from executorch.backends.qualcomm.builders.node_visitor_manager import get_node_visitors
from executorch.backends.qualcomm.builders.qnn_constants import OpContextLoader
from executorch.backends.qualcomm.partition.utils import generate_qnn_executorch_option
from executorch.backends.qualcomm.serialization.qc_schema import (
    QnnExecuTorchOpPackageInfo,
)
from executorch.backends.qualcomm.serialization.qc_schema_serialize import (
    flatbuffer_to_option,
    option_to_flatbuffer,
)
from executorch.exir.backend.backend_details import (
    BackendDetails,
    CompileSpec,
    PreprocessResult,
)
from torch.export.exported_program import ExportedProgram

DEFAULT_DEBUG_HANDLE = 65535

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)


@final
class QnnBackend(BackendDetails):
    @staticmethod
    def _build_op_wrappers(
        edge_program: ExportedProgram,
        enable_tensor_dump: bool,
        op_package_infos: List[QnnExecuTorchOpPackageInfo],
        use_mha2sha: bool,
    ):
        # QNN Delegate Specific Passes
        graph_module = QnnPassManager().transform_for_preprocess_pipeline(
            edge_program, use_mha2sha=use_mha2sha
        )
        assert graph_module is not None

        nodes_to_wrappers = defaultdict(dict)
        node_visitors = get_node_visitors(
            edge_program,
            enable_tensor_dump=enable_tensor_dump,
            op_package_infos=op_package_infos,
        )
        py_op_wrapper_list = []
        for node in graph_module.graph.nodes:
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
                            f"torch.ops.{OpContextLoader.namespace}.{node.target.__name__}",
                            globals().update(torch.__dict__),
                        )
                        assert node.target == context_loader_target, err_msg
                        # if graph has context binary loader node, return directly
                        return node.meta[OpContextLoader.meta_ctx_bin]
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

        return py_op_wrapper_list

    @staticmethod
    def preprocess(
        edge_program: ExportedProgram,
        compile_specs: List[CompileSpec],
    ) -> PreprocessResult:
        option = generate_qnn_executorch_option(compile_specs)
        qnn_manager = PyQnnManager.QnnManager(option)
        qnn_manager.Init()
        obj_options = flatbuffer_to_option(option)
        py_op_wrapper_list = QnnBackend._build_op_wrappers(
            edge_program,
            qnn_manager.IsTensorDump(),
            obj_options.op_package_options.op_package_infos,
            obj_options.use_mha2sha,
        )

        qnn_context_binary = qnn_manager.Compile(
            qnn_manager.GetGraphNames(),
            [[py_op_wrapper.GetOpWrapper() for py_op_wrapper in py_op_wrapper_list]],
        )

        if obj_options.saver:
            exit(
                f"Record all QNN API calls from saver backend at: {obj_options.saver_output_dir}"
            )
        assert len(qnn_context_binary) != 0, "Failed to generate Qnn context binary."
        qnn_manager.Destroy()
        # For now, debug_handle_map is not used by QNN ExecuTorch
        return PreprocessResult(
            processed_bytes=bytes(qnn_context_binary),
            debug_handle_map={},
        )

    @staticmethod
    def preprocess_multimethod(
        edge_programs: Dict[str, List[ExportedProgram]],
        compile_specs: Dict[str, List[List[CompileSpec]]],
    ) -> PreprocessResult:
        # TODO: refactor QnnManager to consume multiple compile_spec
        # take first compile_specs here for the same partitions
        graph_name = list(edge_programs.keys())
        compile_spec = list(compile_specs.values())[0][0]
        # gather all graph names
        option = flatbuffer_to_option(compile_spec[0].value)
        option.graph_name = graph_name
        compile_spec[0].value = option_to_flatbuffer(option)
        # check if each graph has equal number of partitions
        num_sub_graphs = set()
        for edge_program in edge_programs.values():
            num_sub_graphs.add(len(edge_program))
        # this constraint is dedicated to weight-sharing scenario
        assert (
            len(num_sub_graphs) == 1
        ), "Only graphs with the same number of partitions could be used"

        all_processed_results = {key: [] for key in edge_programs.keys()}
        num_sub_graphs = next(iter(num_sub_graphs))
        for i in range(num_sub_graphs):
            # e.g. 2 methods (x, y) with 3 partitions
            #      > context_binary_0: [x.subgraph_0, y.subgraph_0]
            #      > context_binary_1: [x.subgraph_1, y.subgraph_1]
            #      > context_binary_2: [x.subgraph_2, y.subgraph_2]
            qnn_manager = PyQnnManager.QnnManager(
                generate_qnn_executorch_option(compile_spec)
            )
            qnn_manager.Init()
            py_op_wrapper_list, ctx_binary_list = [], []
            for j, programs in enumerate(edge_programs.values()):
                logger.info(f"Processing Method({j}): ({i+1}/{num_sub_graphs})")
                py_op_wrappers = QnnBackend._build_op_wrappers(
                    programs[i],
                    qnn_manager.IsTensorDump(),
                    option.op_package_options.op_package_infos,
                    option.use_mha2sha,
                )
                if isinstance(py_op_wrappers, bytes):
                    ctx_binary_list.append(py_op_wrappers)
                else:
                    py_op_wrapper_list.append(
                        [
                            py_op_wrapper.GetOpWrapper()
                            for py_op_wrapper in py_op_wrappers
                        ]
                    )

            if len(py_op_wrapper_list) == len(edge_programs.values()):
                qnn_context_binary = qnn_manager.Compile(graph_name, py_op_wrapper_list)
                if option.saver:
                    # TODO: Currently, only the first method is saved. Update this logic if saving multiple methods becomes necessary in the future.
                    exit(
                        f"Record all QNN API calls from saver backend at: {option.saver_output_dir}"
                    )
                assert (
                    len(qnn_context_binary) != 0
                ), "Failed to generate Qnn context binary."
                qnn_manager.Destroy()
                # methods should share the same context binary for current partition
                for key in edge_programs.keys():
                    all_processed_results[key].append(
                        PreprocessResult(
                            processed_bytes=bytes(qnn_context_binary),
                            debug_handle_map={},
                        )
                    )
            elif len(ctx_binary_list) == len(edge_programs.values()):
                for i, key in enumerate(edge_programs.keys()):
                    all_processed_results[key].append(
                        PreprocessResult(processed_bytes=ctx_binary_list[i])
                    )
            else:
                raise RuntimeError("Hybrid compilation is not supported")

        return all_processed_results
