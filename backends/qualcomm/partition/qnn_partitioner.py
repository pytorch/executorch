# Copyright (c) Qualcomm Innovation Center, Inc.
# All rights reserved
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.
import copy
import functools
import logging
import operator
from collections import defaultdict
from typing import Any, Callable, Dict, List, Optional, Set, Tuple

import torch
from executorch.backends.qualcomm.builders import node_visitor_manager
from executorch.backends.qualcomm.builders.node_visitor import (
    PER_CHANNEL_GROUP_ENCODING,
)
from executorch.backends.qualcomm.builders.qnn_constants import OpContextLoader
from executorch.backends.qualcomm.builders.utils import is_parameter
from executorch.backends.qualcomm.qnn_preprocess import QnnBackend
from executorch.backends.qualcomm.serialization.qc_schema import (
    HtpArch,
    QnnExecuTorchBackendType,
    QnnExecuTorchHtpPrecision,
)
from executorch.backends.qualcomm.serialization.qc_schema_serialize import (
    flatbuffer_to_option,
)
from executorch.backends.qualcomm.utils.constants import (
    QCOM_BYPASS_NODE,
    QCOM_FALLBACK_NODE,
)

from executorch.backends.qualcomm.utils.qnn_manager_lifecycle import (
    get_current_qnn_manager,
)
from executorch.exir.backend.backend_details import CompileSpec
from executorch.exir.backend.canonical_partitioners.pattern_op_partitioner import (
    generate_partitions_from_list_of_nodes,
)
from executorch.exir.backend.partitioner import (
    DelegationSpec,
    Partitioner,
    PartitionResult,
)
from executorch.exir.backend.utils import tag_constant_data, tag_mutated_buffer
from executorch.exir.dialects._ops import ops as exir_ops
from executorch.exir.tracer import _default_decomposition_table
from torch.export.exported_program import ExportedProgram
from torch.export.graph_signature import InputKind
from torch.fx.passes.infra.partitioner import Partition
from torch.fx.passes.operator_support import OperatorSupportBase
from torch.utils import _pytree as pytree

from .common_defs import (
    allow_list_operator,
    constant_operator,
    not_supported_operator,
    to_be_implemented_operator,
)
from .utils import (
    filter_fn,
    generate_qnn_executorch_option,
    get_skip_decomp_table,
    has_data_dependent_shape,
    has_unsupported_dtype,
    is_quantized,
    must_stay_on_cpu,
    warn_if_dilated_conv_may_not_fit_vtcm,
)

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

# dequantize ops with a CPU out variant; others cannot fall back
CPU_DQ_OPS = {
    exir_ops.edge.quantized_decomposed.dequantize_per_tensor.default,
    exir_ops.edge.quantized_decomposed.dequantize_per_tensor.tensor,
    exir_ops.edge.quantized_decomposed.dequantize_per_channel.default,
}


class QnnOperatorSupport(OperatorSupportBase):
    def __init__(
        self,
        edge_program: torch.export.ExportedProgram,
        compiler_specs,
        skip_node_id_set: set = None,
        skip_node_op_set: set = None,
        is_qnn_partitioner=True,
        phase="QnnPartitioner",
    ):
        option = generate_qnn_executorch_option(compiler_specs)
        python_options = flatbuffer_to_option(option)
        self.node_visitors = node_visitor_manager.get_node_visitors(
            edge_program,
            op_package_infos=python_options.op_package_options.op_package_infos,
            is_qnn_partitioner=is_qnn_partitioner,
            backend_type=python_options.backend_options.backend_type,
        )

        self.skip_node_op_set = skip_node_op_set
        self.skip_node_id_set = skip_node_id_set
        self.is_qnn_partitioner = is_qnn_partitioner
        # Label prefixed to op-support log lines so callers that reuse this
        # checker (e.g. the LPAI fallback pass) are distinguishable in the logs.
        self.phase = phase
        self.soc_info = python_options.soc_info
        self.backend_type = python_options.backend_options.backend_type
        htp_options = python_options.backend_options.htp_options
        # Gate for the VTCM diagnostic in is_node_supported. Only fp16 HTP: the
        # quantized path rejects an untileable dilated conv at finalize, so it
        # needs no warning. Not gated on log level, unlike the backend's other
        # diagnostics -- the failure it points at is a silent unresponsive DSP
        # with no other clue in the log, so it has to be visible by default.
        self.warn_dilated_conv_vtcm = (
            self.backend_type == QnnExecuTorchBackendType.kHtpBackend
            and htp_options is not None
            and htp_options.precision == QnnExecuTorchHtpPrecision.kHtpFp16
        )
        self.nodes_to_wrappers = defaultdict(dict)
        target_socs = (
            [
                target.soc_info.soc_model
                for target in python_options.target_options.targets
            ]
            if python_options.target_options is not None
            else [python_options.soc_info.soc_model]
        )
        self.qnn_managers = [
            get_current_qnn_manager(compiler_specs, soc_model)
            for soc_model in target_socs
        ]

    def _passes_backend_capability_checks(self, node: torch.fx.Node) -> bool:
        # Scoped to HTP: other backends reusing this checker have no htp_arch.
        if (
            self.backend_type != QnnExecuTorchBackendType.kHtpBackend
            or self.soc_info.htp_info.htp_arch >= HtpArch.V69
        ):
            return True

        # Per-channel-group lowers to blockwise expansion (the encoding LPBQ
        # uses), which requires HTP >= V69. Pre-quantized weights bypass the
        # quantizer's validate_lpbq_support check, so gate the delegation here.
        if node.target in PER_CHANNEL_GROUP_ENCODING:
            logger.warning(
                f"[{self.phase}] {node.target.__name__} | Skipped: blockwise "
                f"(per-channel-group) encoding requires HTP arch >= V69, got "
                f"{self.soc_info.htp_info.htp_arch!r}"
            )
            return False

        # TransposeConv2d generated for convolution_backward is unsupported by
        # the V68 compiler despite passing the QNN op-support query.
        if node.target == exir_ops.edge.aten.convolution_backward.default:
            logger.warning(
                f"[{self.phase}] {node.target.__name__} | Skipped: requires HTP "
                f"arch >= V69, got {self.soc_info.htp_info.htp_arch!r}"
            )
            return False

        return True

    def is_node_supported(self, _, node: torch.fx.Node) -> bool:  # noqa: C901
        if node.op != "call_function" or node.target in not_supported_operator:
            return False

        if not self._passes_backend_capability_checks(node):
            return False

        if node.target in to_be_implemented_operator:
            logger.info(
                f"[{self.phase}] {node.target.__name__} | Skipped, this op can be supported, please report an issue in https://github.com/pytorch/executorch/issues"
            )
            return False

        if node.meta.get(QCOM_FALLBACK_NODE, False):
            logger.info(f"[{self.phase}] {node.target.__name__} | Forced Fallback")
            return False

        if (
            node.target in allow_list_operator
            # bypass if custom op appears
            or OpContextLoader.namespace == node.target.namespace
            # bypass dequantize op for parameters & buffers
            or node.meta.get(QCOM_BYPASS_NODE, False)
        ):
            logger.info(f"[{self.phase}] {node.target.__name__} | Forced Passed")
            return True

        if (
            node.name in self.skip_node_id_set
            or node.target.__name__ in self.skip_node_op_set
        ):
            logger.info(f"[{self.phase}] {node.target.__name__} | Skipped")
            return False

        if has_unsupported_dtype(node):
            logger.info(
                f"[{self.phase}] {node.target.__name__} | Skipped, unsupported dtype"
            )
            return False

        if has_data_dependent_shape(node):
            logger.info(
                f"[{self.phase}] {node.target.__name__} | Skipped, data-dependent shape"
            )
            return False

        # TODO: workaround for https://github.com/pytorch/executorch/issues/23096.
        # Remove this once the fp16 path performs the TCM-fit check the quantized
        # path does.
        if self.warn_dilated_conv_vtcm:
            warn_if_dilated_conv_may_not_fit_vtcm(
                node, self.soc_info.htp_info.vtcm_size_in_mb, self.phase
            )

        supported = False
        if node.target.__name__ not in self.node_visitors:
            logger.info(
                f"[{self.phase}] {node.target.__name__} | No node visitor, unsupported"
            )
            return False
        op_wrapper = self.node_visitors[node.target.__name__].define_node(
            node, self.nodes_to_wrappers
        )
        if node.target in constant_operator:
            return True

        op_wrapper_list = []
        if isinstance(op_wrapper, List):
            op_wrapper_list.extend(op_wrapper)
        else:
            op_wrapper_list.append(op_wrapper)

        if op_wrapper is not None:
            wrappers = [op.GetOpWrapper() for op in op_wrapper_list]
            supported = all(
                manager.IsNodeSupportedByBackend(wrappers)
                for manager in self.qnn_managers
            )

        self.nodes_to_wrappers.clear()
        logger.info(f"[{self.phase}] {node.target.__name__} | {supported}")
        return supported


class QnnPartitioner(Partitioner):
    """
    QnnPartitioner identifies subgraphs that can be lowered to QNN backend, by tagging nodes for delegation,
    and manages special cases such as mutable buffers and consumed constants.
    """

    def __init__(
        self,
        compiler_specs: List[CompileSpec],
        skip_node_id_set: set = None,
        skip_node_op_set: set = None,
        skip_mutable_buffer: bool = False,
    ):
        """
        Args:
            compiler_specs (List[CompileSpec]): Backend compiler specifications.
            skip_node_id_set (set, optional): Set of node IDs to exclude from partitioning.
            skip_node_op_set (set, optional): Set of OpOverload to exclude from partitioning.
            skip_mutable_buffer (bool, optional): If True, mutable buffers are not delegated to QNN.
        """
        self.compiler_specs_snapshot = copy.deepcopy(compiler_specs)
        self.backend = flatbuffer_to_option(
            generate_qnn_executorch_option(self.compiler_specs_snapshot)
        ).backend_options.backend_type

        self.delegation_spec = DelegationSpec(
            QnnBackend.__name__, self.compiler_specs_snapshot
        )
        self.partition_tags: Dict[str, DelegationSpec] = {}
        self.skip_node_id_set = set() if skip_node_id_set is None else skip_node_id_set
        self.skip_node_op_set = set() if skip_node_op_set is None else skip_node_op_set
        self.skip_mutable_buffer = skip_mutable_buffer

    def generate_partitions(
        self, edge_program: torch.export.ExportedProgram
    ) -> List[Any]:
        self.op_support_checker = QnnOperatorSupport(
            edge_program,
            self.compiler_specs_snapshot,
            self.skip_node_id_set,
            self.skip_node_op_set,
        )
        partitions = generate_partitions_from_list_of_nodes(
            edge_program.graph_module,
            op_support=self.op_support_checker,
        )
        partitions = self._drop_orphaned_parameter_dequants(partitions, edge_program)
        # LPAI places its CPU fallback boundaries before partitioning, and GPU
        # accepts graphs without inputs, so only HTP partitions are dropped.
        if self.backend == QnnExecuTorchBackendType.kHtpBackend:
            partitions = self._drop_partitions_without_runtime_inputs(
                partitions, edge_program
            )
        return partitions

    @staticmethod
    def _drop_orphaned_parameter_dequants(
        partitions: List[Partition], edge_program: torch.export.ExportedProgram
    ) -> List[Partition]:
        """
        A parameter dequantize whose consumer is not delegated stays on CPU. A QNN
        Dequantize alone rejects per-channel encodings.
        """
        for partition in partitions:
            stack = [
                node
                for node in partition.nodes
                if node.target in CPU_DQ_OPS
                and node.meta.get(QCOM_BYPASS_NODE, False)
                and is_parameter(node.args[0], edge_program)
            ]
            while stack:
                node = stack.pop()
                if node not in partition.nodes or any(
                    user in partition.nodes for user in node.users
                ):
                    continue
                partition.remove_node(node)
                stack.extend(
                    arg
                    for arg in node.args
                    if isinstance(arg, torch.fx.Node)
                    and arg.meta.get(QCOM_BYPASS_NODE, False)
                )
        return [partition for partition in partitions if partition.nodes]

    @staticmethod
    def _constant_inputs(edge_program: torch.export.ExportedProgram) -> Set[str]:
        """
        Names of the inputs that hold the same value on every run: parameters,
        buffers that are never mutated, and lifted constants.
        """
        signature = edge_program.graph_signature
        mutated = set(signature.buffers_to_mutate.values())
        return {
            spec.arg.name
            for spec in signature.input_specs
            if spec.kind
            in (InputKind.PARAMETER, InputKind.BUFFER, InputKind.CONSTANT_TENSOR)
            and spec.target not in mutated
        }

    @staticmethod
    def _is_runtime_input(node: torch.fx.Node, constant_inputs: Set[str]) -> bool:
        return node.op != "get_attr" and node.name not in constant_inputs

    @staticmethod
    def _drop_partitions_without_runtime_inputs(
        partitions: List[Partition], edge_program: torch.export.ExportedProgram
    ) -> List[Partition]:
        """
        HTP refuses a graph that reads nothing at runtime, e.g. a subgraph fed
        only by constants once a CPU op splits it from the rest. Such a
        partition stays on CPU.
        """
        constant_inputs = QnnPartitioner._constant_inputs(edge_program)
        return [
            partition
            for partition in partitions
            if any(
                QnnPartitioner._is_runtime_input(arg, constant_inputs)
                for node in partition.nodes
                for arg in node.all_input_nodes
                if arg not in partition.nodes
            )
        ]

    @staticmethod
    def _may_run_on_qnn(node: torch.fx.Node) -> bool:
        """
        Whether QNN's op check may accept the node, as far as can be told
        before the kept-whole ops are decomposed. Ops whose dtypes or shapes
        rule QNN out, ops without a tensor output, such as the runtime
        asserts export adds, and higher-order ops, which have no node visitor,
        stay on CPU. Every getitem is accepted, as in QNN's op check.
        """
        if node.op != "call_function":
            return False
        if node.target in allow_list_operator:
            return True
        if not isinstance(node.target, torch._ops.OpOverload):
            return False
        # Retracing can drop the "val" of an assert, but not its schema.
        if not node.target._schema.returns:
            return False
        if "val" in node.meta and not any(
            isinstance(val, torch.Tensor)
            for val in pytree.tree_leaves(node.meta["val"])
        ):
            return False
        return not must_stay_on_cpu(node)

    @staticmethod
    def _left_on_cpu(ep: ExportedProgram) -> Set[torch.fx.Node]:
        """
        Estimates, on the ATen program, which ops HTP leaves on CPU, so that
        the kept-whole ones among them can be decomposed.
        """

        # Partitioning here would take time quadratic in the graph size, so
        # only the ops that _may_run_on_qnn rules out count as CPU ops. A QNN
        # op that reads no runtime value is left on CPU when:
        # - it reaches a QNN op through a CPU op before any QNN op that reads
        #   a runtime value, since the partitioner may split it off to avoid a
        #   cycle;
        # - the only QNN ops that read a runtime value are getitems of CPU
        #   ops, since the partitioner moves those getitems out of their
        #   partitions;
        # - such a getitem follows a QNN op, and no chain of QNN ops links the
        #   op to a QNN op that reads a runtime value.
        # Skips are left out, since a skip that changed what is decomposed
        # would rename later nodes, so a skip_node_id_set entry could name a
        # different node.
        # Ops that QNN rejects or removes later are not known here either. So
        # an op cut off only by them still fails as before.
        # An op can also be decomposed although a partition that reads a
        # runtime value takes it.
        nodes = list(ep.graph.nodes)
        qnn = set(filter(QnnPartitioner._may_run_on_qnn, nodes))
        constant_inputs = QnnPartitioner._constant_inputs(ep)
        runtime, after_qnn = set(), set()
        for node in nodes:
            if node in qnn or not after_qnn.isdisjoint(node.all_input_nodes):
                after_qnn.add(node)
            if node in qnn and any(
                (
                    arg in runtime
                    if arg in qnn
                    else QnnPartitioner._is_runtime_input(arg, constant_inputs)
                )
                for arg in node.all_input_nodes
            ):
                runtime.add(node)
        to_qnn, before_cpu = set(), set()
        for node in reversed(nodes):
            if node in qnn or not to_qnn.isdisjoint(node.users):
                to_qnn.add(node)
            if any(
                (user in before_cpu and user not in runtime)
                or (user in to_qnn and user not in qnn)
                for user in node.users
            ):
                before_cpu.add(node)
        left_on_cpu = (qnn - runtime) & before_cpu
        cpu_getitems = {
            node
            for node in runtime
            if node.target is operator.getitem and node.args[0] not in qnn
        }
        if runtime <= cpu_getitems:
            return qnn - runtime
        if not any(node.args[0] in after_qnn for node in cpu_getitems):
            return left_on_cpu
        linked, stack = set(runtime), list(runtime)
        while stack:
            node = stack.pop()
            for neighbor in (*node.all_input_nodes, *node.users):
                if neighbor in qnn and neighbor not in linked:
                    linked.add(neighbor)
                    stack.append(neighbor)
        return left_on_cpu | (qnn - runtime - linked)

    def tag_nodes(
        self, partitions: List[Partition], edge_program: torch.export.ExportedProgram
    ) -> None:
        """
        Tags nodes in the given partitions and the edge program's graph with delegation tags for QNN partitioning.
        """
        for partition in partitions:
            for node in partition.nodes:
                delegation_tag = f"qnn_{partition.id}"
                node.meta["delegation_tag"] = delegation_tag
                self.partition_tags[delegation_tag] = self.delegation_spec

        # Dropping partitions can leave none, and there is no tag to reuse.
        if not partitions:
            return

        # need to take care of consumed constants
        consumed_constants = (
            *edge_program.graph_signature.inputs_to_buffers,
            *edge_program.graph_signature.inputs_to_parameters,
        )
        for node in edge_program.graph_module.graph.nodes:
            # find placeholders as lifted_constants
            if node.op != "placeholder" or len(node.users) != 0:
                continue

            if node.name in consumed_constants:
                # does no harm to merge them into last partition,
                # since they will all be removed in following stage
                node.meta["delegation_tag"] = delegation_tag

    # override
    def partition(self, edge_program: torch.export.ExportedProgram) -> PartitionResult:
        # Generate partitions by QNN op_support checker
        partitions = self.generate_partitions(edge_program)
        del self.op_support_checker
        # If partitions are found, handle tagging of nodes, constant data, and mutated buffers for delegation
        self.tag_nodes(partitions, edge_program)
        tag_constant_data(edge_program)
        if not self.skip_mutable_buffer:
            logger.info(
                "Qnn partitioner will delegate torch mutable buffer with the same I/O address during the runtime, "
                "so if your model contains mutable buffer, "
                "then you can get the better performance with skip_mutable_buffer=False. "
                "If you encounter accuracy issue during the runtime, "
                "then please set `skip_mutable_buffer=True` and try again."
            )
            tag_mutated_buffer(edge_program)

        return PartitionResult(
            tagged_exported_program=edge_program, partition_tags=self.partition_tags
        )

    # override
    def ops_to_not_decompose(
        self, ep: ExportedProgram
    ) -> Tuple[List[torch._ops.OpOverload], Optional[Callable[[torch.fx.Node], bool]]]:
        """
        Determines which op should not be decomposed during partitioning.
        The list of operators is obtained from `get_skip_decomp_table()`.
        The filter function (`filter_fn`) can be used to further refine which nodes are not decomposed. (advanced use case)
        On HTP, kept-whole ops that `_drop_partitions_without_runtime_inputs`
        would leave on CPU, as far as can be told before they are decomposed,
        are decomposed too. Quantized ops stay whole, and exir cannot
        decompose some ops, such as elu and the scatter variants, so such an op
        still fails as before.
        """
        do_not_decompose = get_skip_decomp_table()
        # On the lowered program exir reports every kept-whole op left there
        # that the filter accepts, so it stays plain filter_fn.
        if self.backend != QnnExecuTorchBackendType.kHtpBackend or ep.dialect == "EDGE":
            return (do_not_decompose, filter_fn)

        # exir calls this method several times on the ATen program but uses
        # only one of the filters, so the estimate runs on first use. That is
        # before exir renames the targets of the ops it keeps whole, which
        # would hide the dequantize ops.
        # Quantized ops stay whole, in case the estimate is wrong: their pieces
        # would have no quantization encodings, so QNN would reject them where
        # a partition would take them.
        @functools.cache
        def decomposed():
            nodes = {node for node in self._left_on_cpu(ep) if not is_quantized(node)}
            decompositions = _default_decomposition_table()
            names = sorted(
                node.name
                for node in nodes
                if node.target in do_not_decompose
                and node.target in decompositions
                and filter_fn(node)
            )
            message = f"[QnnPartitioner] Decomposing kept-whole ops HTP would leave on CPU: {names}"
            if names and self.skip_node_id_set:
                logger.warning(
                    f"{message}. Decomposing renames later nodes, so check that "
                    "skip_node_id_set still names the intended nodes."
                )
            elif names:
                logger.info(message)
            return nodes

        return (
            do_not_decompose,
            lambda node: filter_fn(node) and node not in decomposed(),
        )
