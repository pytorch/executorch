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
from executorch.backends.qualcomm._passes.qnn_pass_manager import (
    get_qnn_pass_manager_cls,
)
from executorch.backends.qualcomm.builders.node_visitor_manager import get_node_visitors
from executorch.backends.qualcomm.builders.qnn_constants import OpContextLoader
from executorch.backends.qualcomm.partition.utils import generate_qnn_executorch_option
from executorch.backends.qualcomm.serialization.qc_schema import (
    QnnExecuTorchBackendType,
    QnnExecuTorchOpPackageInfo,
)
from executorch.backends.qualcomm.serialization.qc_schema_serialize import (
    flatbuffer_to_option,
)
from executorch.backends.qualcomm.utils.constants import (
    QCOM_AXIS_ORDER,
    QCOM_TENSOR_NAME,
)
from executorch.backends.qualcomm.utils.qnn_manager_lifecycle import (
    get_current_qnn_manager,
)
from executorch.exir.backend.backend_details import (
    BackendDetails,
    CompileSpec,
    PreprocessResult,
)
from executorch.exir.backend.utils import DelegateMappingBuilder
from executorch.exir.debug_handle_utils import DEBUG_HANDLE_KEY
from executorch.exir.operator.convert import unwrap_op_overload
from torch.export.exported_program import ExportedProgram

DEFAULT_DEBUG_HANDLE = 65535
DEFAULT_GRAPH_NAME = "forward"

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)


def _check_io_binding(edge_program: ExportedProgram, nodes_to_wrappers) -> None:
    """Fail here if QNN's graph I/O will not line up with the delegate signature.

    At runtime the delegate binds its arguments positionally: it walks the tensor
    lists recovered from the context binary and consumes one argument per tensor
    the name prefixes mark as bindable (QnnExecuTorchBackend::execute). Nothing
    reconciles that walk with the number of arguments ExecuTorch actually passes,
    so a graph that publishes extra I/O reads past the end of the argument list on
    device. Catching it here costs one pass over the wrappers and reports the
    offending tensor names instead of a register dump.
    """
    qnn_inputs, qnn_outputs = set(), set()
    for wrappers in nodes_to_wrappers.values():
        for wrapper in wrappers.values():
            name = PyQnnManager.PyQnnTensorWrapper(wrapper).GetName()
            # Mutable buffers are threaded through separately and never consume a
            # delegate argument; the runtime skips them by the same marker.
            if "mutbuf_" in name:
                continue
            if name.startswith("input_"):
                qnn_inputs.add(name)
            elif name.startswith("output_"):
                qnn_outputs.add(name)

    signature = edge_program.graph_signature
    num_inputs = len(signature.user_inputs)
    num_outputs = len(signature.user_outputs)
    if len(qnn_inputs) == num_inputs and len(qnn_outputs) == num_outputs:
        return

    raise RuntimeError(
        "QNN graph I/O does not match the delegated program signature. QNN "
        f"declares {len(qnn_inputs)} graph inputs and {len(qnn_outputs)} graph "
        f"outputs; the signature declares {num_inputs} user inputs and "
        f"{num_outputs} user outputs. The runtime binds delegate arguments "
        "positionally, so this reads past the end of the argument list on device."
        f"\n  qnn inputs        : {sorted(qnn_inputs)}"
        f"\n  qnn outputs       : {sorted(qnn_outputs)}"
        f"\n  signature inputs  : {list(signature.user_inputs)}"
        f"\n  signature outputs : {list(signature.user_outputs)}"
    )


@final
class QnnBackend(BackendDetails):
    @staticmethod
    def _build_op_wrappers(
        edge_program: ExportedProgram,
        enable_tensor_dump: bool,
        op_package_infos: List[QnnExecuTorchOpPackageInfo],
        use_mha2sha: bool,
        backend_type: QnnExecuTorchBackendType,
    ):
        for node in edge_program.graph_module.graph.nodes:
            if hasattr(node, "meta"):
                # pop certain keys in meta for not affecting the passes in compilation
                node.meta.pop(QCOM_AXIS_ORDER, "")
        # QNN Delegate Specific Passes
        graph_module = get_qnn_pass_manager_cls(
            backend_type
        )().transform_for_preprocess_pipeline(edge_program, use_mha2sha=use_mha2sha)
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
                        op = unwrap_op_overload(node.target)
                        context_loader_target = eval(
                            f"torch.ops.{OpContextLoader.namespace}.{op.__name__}",
                            {"torch": torch},
                        )
                        assert op == context_loader_target, err_msg
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

        _check_io_binding(edge_program, nodes_to_wrappers)
        return py_op_wrapper_list

    @staticmethod
    def preprocess(
        edge_program: ExportedProgram,
        compile_specs: List[CompileSpec],
    ) -> PreprocessResult:
        option = generate_qnn_executorch_option(compile_specs)
        obj_options = flatbuffer_to_option(option)
        qnn_manager = get_current_qnn_manager(compile_specs)
        qnn_manager.InitContext([DEFAULT_GRAPH_NAME])
        py_op_wrapper_list = QnnBackend._build_op_wrappers(
            edge_program,
            qnn_manager.IsTensorDump(),
            obj_options.op_package_options.op_package_infos,
            obj_options.use_mha2sha,
            obj_options.backend_options.backend_type,
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
        qnn_manager.DestroyContext()
        # For now, debug_handle_map is not used by QNN ExecuTorch
        return PreprocessResult(
            processed_bytes=qnn_context_binary,
            debug_handle_map={},
        )

    @staticmethod
    def _populate_delegate_mapping(
        debug_handle_builder: DelegateMappingBuilder,
        program: ExportedProgram,
    ):
        for node in program.graph.nodes:
            # Skip multi-output nodes: devtools only supports
            # single-output intermediate capture (len == 1).
            if (
                (handle_id := node.meta.get(DEBUG_HANDLE_KEY))
                and QCOM_TENSOR_NAME in node.meta
                and len(node.meta[QCOM_TENSOR_NAME]) == 1
            ):
                debug_handle_builder.insert_delegate_mapping_entry(
                    handles=handle_id,
                    identifier=node.meta[QCOM_TENSOR_NAME][0],
                )

    @staticmethod
    def _get_compile_func(qnn_manager: PyQnnManager.QnnManager):
        def compile_func(graph_names, op_wrapper_list):
            qnn_manager.InitContext(graph_names)
            try:
                qnn_context_binary = qnn_manager.Compile(graph_names, op_wrapper_list)
            finally:
                qnn_manager.DestroyContext()
            return qnn_context_binary

        return compile_func

    @staticmethod
    def _get_compile_func_fcb(qnn_managers: List[PyQnnManager.QnnManager]):
        def compile_func(graph_names, op_wrapper_list):
            with qnn_managers[0].CreateDlc() as dlc_handle:
                for qnn_manager in qnn_managers:
                    qnn_manager.InitContext(graph_names)
                    try:
                        qnn_manager.CompileToDlc(
                            graph_names, op_wrapper_list, dlc_handle
                        )
                    finally:
                        qnn_manager.DestroyContext()
                dlc_binary = bytes(qnn_managers[0].GetDlcBinary(dlc_handle))
            return dlc_binary

        return compile_func

    @staticmethod
    def preprocess_multimethod(  # noqa: C901
        edge_programs: Dict[str, List[ExportedProgram]],
        compile_specs: Dict[str, List[List[CompileSpec]]],
    ) -> Dict[str, List[PreprocessResult]]:
        # In `edge_programs: Dict[str, List[ExportedProgram]]`,
        # `edge_programs[method][i]` is the i-th ExecuTorch partition of one
        # method. In the QNN graph path, each partition contributes one QNN graph.
        #
        # Weight sharing compiles the i-th partitions from all methods into one
        # multi-graph context binary. For methods x and y:
        #
        #   context_binary_0 = compile([x.partition_0, y.partition_0])
        #   context_binary_1 = compile([x.partition_1, y.partition_1])
        #   context_binary_2 = compile([x.partition_2, y.partition_2])
        #
        # The returned `all_processed_results: Dict[str, List[PreprocessResult]]`
        # stores `context_binary_i` for every method at partition index i. This
        # requires every method to have the same number of partitions.

        # TODO: refactor QnnManager to consume multiple compile_spec
        # take first compile_specs here for the same partitions
        graph_names = list(edge_programs.keys())
        compile_spec = list(compile_specs.values())[0][0]
        option = flatbuffer_to_option(compile_spec[0].value)

        debug_handle_builder = DelegateMappingBuilder(generated_identifiers=False)

        is_fcb = option.target_options is not None
        if is_fcb:
            qnn_managers = [
                get_current_qnn_manager(compile_spec, target.soc_info.soc_model)
                for target in option.target_options.targets
            ]
            compile_func = QnnBackend._get_compile_func_fcb(qnn_managers)
        else:
            qnn_manager = get_current_qnn_manager(compile_spec)
            compile_func = QnnBackend._get_compile_func(qnn_manager)

        all_processed_results = {key: [] for key in edge_programs}

        # check if each graph has equal number of partitions
        num_partitions_per_method = {
            len(programs) for programs in edge_programs.values()
        }
        # this constraint is dedicated to weight-sharing scenario
        assert (
            len(num_partitions_per_method) == 1
        ), "Only graphs with the same number of partitions could be used"
        num_partitions_per_method = next(iter(num_partitions_per_method))

        for i in range(num_partitions_per_method):
            method_to_ith_partition_wrapper = {
                key: QnnBackend._build_op_wrappers(
                    programs[i],
                    option.dump_intermediate_outputs,
                    option.op_package_options.op_package_infos,
                    option.use_mha2sha,
                    option.backend_options.backend_type,
                )
                for key, programs in edge_programs.items()
            }

            # QNN preprocessing assigns the final QCOM_TENSOR_NAME metadata used as
            # delegate identifiers, so build this mapping only after wrappers exist.
            if option.dump_intermediate_outputs:
                for edge_program in edge_programs.values():
                    QnnBackend._populate_delegate_mapping(
                        debug_handle_builder, edge_program[i]
                    )

            # ensure not mixed
            wrapper_types = set(map(type, method_to_ith_partition_wrapper.values()))
            if len(wrapper_types) != 1:
                raise RuntimeError("Hybrid compilation is not supported")
            wrapper_type = next(iter(wrapper_types))

            if wrapper_type == bytes:
                for key, wrapper in method_to_ith_partition_wrapper.items():
                    all_processed_results[key].append(
                        PreprocessResult(
                            processed_bytes=wrapper,
                            debug_handle_map=debug_handle_builder.get_delegate_mapping(),
                        )
                    )
            elif wrapper_type == list:
                # List[PyQnnManager.PyQnnOpWrapper]
                op_wrapper_list = [
                    [wrapper.GetOpWrapper() for wrapper in wrappers]
                    for wrappers in method_to_ith_partition_wrapper.values()
                ]
                context_binary = compile_func(graph_names, op_wrapper_list)
                if option.saver:
                    # TODO: Currently, only the first method is saved. Update this logic if saving multiple methods becomes necessary in the future.
                    exit(
                        f"Record all QNN API calls from saver backend at: {option.saver_output_dir}"
                    )
                assert (
                    len(context_binary) != 0
                ), "Failed to generate Qnn context binary."
                for key in edge_programs:
                    all_processed_results[key].append(
                        PreprocessResult(
                            processed_bytes=context_binary,
                            debug_handle_map=debug_handle_builder.get_delegate_mapping(),
                        )
                    )
            else:
                raise ValueError(
                    f"Unexpected preprocessing op wrapper type {wrapper_type}"
                )
        return all_processed_results
