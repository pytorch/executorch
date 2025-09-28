# Copyright © 2023 Apple Inc. All rights reserved.
#
# Please refer to the license found in the LICENSE file in the root directory of the source tree.

import logging
from typing import Callable, List, Optional, Tuple

import coremltools as ct

import torch

from executorch.backends.apple.coreml.compiler import CoreMLBackend
from executorch.backends.apple.coreml.compiler.coreml_preprocess import (
    COMPILE_SPEC_KEYS,
)

from executorch.backends.apple.coreml.logging import get_coreml_log_level
from executorch.exir.backend.compile_spec_schema import CompileSpec

from executorch.exir.backend.partitioner import (
    DelegationSpec,
    Partitioner,
    PartitionResult,
)
from executorch.exir.backend.utils import tag_constant_data, tag_mutated_buffer
from executorch.exir.dialects._ops import ops as exir_ops
from torch.export.exported_program import ExportedProgram
from torch.fx.passes.infra.partitioner import CapabilityBasedPartitioner
from torch.fx.passes.operator_support import OperatorSupportBase
from torch.fx.passes.infra.partitioner import Partition

logger = logging.getLogger(__name__)
logger.setLevel(get_coreml_log_level(default_level=logging.INFO))


def _is_view_op(op: torch._ops.OpOverload) -> bool:
    schema = op._schema
    if len(schema.arguments) == 0:
        return False
    alias_info = schema.arguments[0].alias_info
    return (alias_info is not None) and (not alias_info.is_write)


class _OperatorsSupportedForCoreMLBackend(OperatorSupportBase):
    def __init__(
        self,
        skip_ops_for_coreml_delegation: Optional[List[str]] = None,
        lower_full_graph: bool = False,
        log: bool = False,
    ) -> None:
        if skip_ops_for_coreml_delegation is None:
            skip_ops_for_coreml_delegation = []
        super().__init__()
        self.skip_ops_for_coreml_delegation = skip_ops_for_coreml_delegation
        self.lower_full_graph = lower_full_graph
        self._logged_msgs = set()
        self._log = log

    def log_once(self, msg: str) -> None:
        if self._log and msg not in self._logged_msgs:
            logger.info(msg)
            self._logged_msgs.add(msg)

    def should_skip_op_for_delegation(self, node_target_name: str) -> bool:
        skipped_ops = self.skip_ops_for_coreml_delegation or []
        if node_target_name in skipped_ops:
            assert (
                not self.lower_full_graph
            ), f"Cannot skip {node_target_name} because lower_full_graph is True.  Please set skip_ops_for_coreml_delegation=None or lower_full_graph=False in the CoreMLPartitioner"
            self.log_once(
                "Skipping op for CoreML delegation because it is in skip_ops_for_coreml_delegation: "
                + node_target_name
            )
            return True
        return False

    def should_override_support(self, node) -> bool:
        # https://github.com/apple/coremltools/issues/2573
        if (
            node.target
            in [
                torch.ops.aten.sub.Tensor,
                exir_ops.edge.aten.sub.Tensor,
                torch.ops.aten.add.Tensor,
                exir_ops.edge.aten.add.Tensor,
            ]
            and "alpha" in node.kwargs
            and node.kwargs["alpha"] != 1
        ):
            self.log_once(
                "torch.ops.aten.{sub, add}.Tensor with alpha != 1 is not supported by CoreML.  Overriding support."
            )
            return True

        # https://github.com/apple/coremltools/issues/2565
        if node.target in [
            torch.ops.aten.diagonal.default,
            torch.ops.aten.diagonal_copy.default,
            exir_ops.edge.aten.diagonal.default,
            exir_ops.edge.aten.diagonal_copy.default,
        ]:
            self.log_once(
                "torch.ops.aten.diagonal.default has a bug in CoreML.  Overriding op support."
            )
            return True

        # https://github.com/apple/coremltools/issues/2569
        if node.target in [
            torch.ops.aten.acosh.default,
            exir_ops.edge.aten.acosh.default,
            torch.ops.aten.asinh.default,
            exir_ops.edge.aten.asinh.default,
        ]:
            self.log_once(
                "torch.ops.aten.{acosh, asinh}.default is not supported by CoreML.  Overriding op support."
            )
            return True

        # TODO: enable this after bugs in ExecuTorch's partitioner are fixed
        # # If lower_full_graph=False, do not partition nodes with symbolic args because it can result in symbolic args
        # # in the placeholders due to partitioning, which CoreML does not support
        # if not self.lower_full_graph and any(
        #     isinstance(arg, torch.fx.Node)
        #     and isinstance(
        #         arg.meta.get("val", None),
        #         (torch.SymInt, torch.SymBool, torch.SymFloat),
        #     )
        #     for arg in node.args
        # ):
        #     self.log_once(
        #         "Skipping op for CoreML delegation because it contains symbolic args: "
        #         + node_target_name
        #     )
        #     return True

        return False

    def is_node_supported(self, submodules, node: torch.fx.Node) -> bool:
        # get_attr node can always be supported on any backend
        if node.op == "get_attr":
            return True
        # check if the PyTorch op get called is supported in Core ML
        elif node.op == "call_function":
            # skip ops if specified by user
            node_target_name = getattr(node.target, "__name__", "").lower()

            if self.should_skip_op_for_delegation(node_target_name):
                return False

            # query coremltools to see if node is supported
            is_supported = ct.converters.mil.frontend.torch.is_torch_fx_node_supported(
                node
            )
            if self.should_override_support(node):
                is_supported = False

            if not is_supported:
                if self.lower_full_graph:
                    raise NotImplementedError(
                        f"""CoreML does not support the op {node_target_name}, but you have set lower_full_graph=True in the CoreMLPartitioner.

Please set lower_full_graph=False in the CoreMLPartitioner to allow running unsupported ops outside of CoreML.  Note that setting lower_full_graph=False may affect performance of CoreML and the available features.
As an alternative to setting lower_full_graph=False, you can try rewriting your model to avoid using this op.

Also consider filing an issue with Apple's coremltools repo to request support for the op: https://github.com/apple/coremltools/issues
Do not file an issue with ExecuTorch for op support.
"""
                    )
                self.log_once(
                    "Skipping op for CoreML delegation because it is not supported by CoreML: "
                    + node_target_name
                )
            return is_supported
        # cowardly refuse to support all other types of node:
        # 1. placeholder / output nodes should not be tagged
        #    reference: https://github.com/pytorch/executorch/pull/1398
        # 2. call_module / call_method should have been replaced with call_function?
        else:
            self.log_once(
                "Skipping op for CoreML delegation because it is not get_attr or call_function: "
                + node.op
            )
            return False


class SingleOpCoreMLPartitioner(Partitioner):
    """
    CoreML partitioner that creates individual call_delegate nodes for each operation,
    with special handling for 4-bit weight-only quantization patterns (dequantize_affine + linear).
    """

    def __init__(
        self,
        *,
        skip_ops_for_coreml_delegation: Optional[List[str]] = None,
        compile_specs: Optional[List[CompileSpec]] = None,
        take_over_mutable_buffer: Optional[bool] = True,
        take_over_constant_data: bool = True,
    ) -> None:
        if skip_ops_for_coreml_delegation is None:
            skip_ops_for_coreml_delegation = []
        self.skip_ops_for_coreml_delegation = skip_ops_for_coreml_delegation

        self.delegation_spec = DelegationSpec(
            backend_id=CoreMLBackend.__name__,
            compile_specs=compile_specs if compile_specs is not None else [],
        )
        self.take_over_mutable_buffer = take_over_mutable_buffer
        self.take_over_constant_data = take_over_constant_data
        self._logged_msgs = set()

    def _is_dequantize_affine_node(self, node: torch.fx.Node) -> bool:
        """Check if node is a dequantize_affine operation."""
        if node.op != "call_function":
            return False

        # Check for the actual torchao.dequantize_affine operation
        return str(node.target) == "torchao.dequantize_affine.default"

    def _is_linear_node(self, node: torch.fx.Node) -> bool:
        """Check if node is a linear operation."""
        if node.op != "call_function":
            return False

        # Check for the actual aten.linear operation
        return str(node.target) == "aten.linear.default"

    def _is_embedding_node(self, node: torch.fx.Node) -> bool:
        """Check if node is an embedding operation."""
        if node.op != "call_function":
            return False

        # Check for the actual aten.embedding operation
        return str(node.target) == "aten.embedding.default"

    def _is_4bit_weight_only_pattern(self, dequant_node: torch.fx.Node, consumer_node: torch.fx.Node) -> bool:
        """
        Check if the dequantize_affine + consumer pattern represents 4-bit weight-only quantization.
        This checks the quant_min and quant_max parameters of the dequantize_affine node.
        Consumer can be either linear or embedding.
        """
        if not self._is_dequantize_affine_node(dequant_node):
            return False

        # Check if consumer is either linear or embedding
        if not (self._is_linear_node(consumer_node) or self._is_embedding_node(consumer_node)):
            return False

        # Check if dequantize_affine output feeds into consumer input
        if dequant_node not in consumer_node.all_input_nodes:
            return False

        # Check for 4-bit quantization parameters (quant_min=-8, quant_max=7)
        if len(dequant_node.args) >= 6:
            try:
                quant_min = dequant_node.args[5]
                quant_max = dequant_node.args[6] if len(dequant_node.args) > 6 else None

                # Handle case where parameters might be nodes vs constants
                if hasattr(quant_min, 'meta') and 'val' in quant_min.meta:
                    quant_min = quant_min.meta['val']
                if hasattr(quant_max, 'meta') and 'val' in quant_max.meta:
                    quant_max = quant_max.meta['val']

                return quant_min == -8 and quant_max == 7
            except (IndexError, AttributeError, TypeError):
                return False

        return False

    def _find_4bit_patterns(self, graph_module: torch.fx.GraphModule) -> List[Tuple[torch.fx.Node, torch.fx.Node]]:
        """Find all dequantize_affine + consumer patterns that represent 4-bit weight-only quantization."""
        patterns = []

        for node in graph_module.graph.nodes:
            if self._is_dequantize_affine_node(node):
                # Look for linear or embedding nodes that use this dequantize_affine output
                for user in node.users:
                    if (self._is_linear_node(user) or self._is_embedding_node(user)) and self._is_4bit_weight_only_pattern(node, user):
                        patterns.append((node, user))

        return patterns

    def _create_single_op_partitions(self, exported_program: ExportedProgram) -> List[Partition]:
        """Create individual partitions for each supported operation."""
        op_support = _OperatorsSupportedForCoreMLBackend(
            self.skip_ops_for_coreml_delegation,
            lower_full_graph=False,
            log=True,
        )

        # Find 4-bit quantization patterns first
        patterns_4bit = self._find_4bit_patterns(exported_program.graph_module)
        pattern_nodes = set()
        for dequant_node, consumer_node in patterns_4bit:
            pattern_nodes.add(dequant_node)
            pattern_nodes.add(consumer_node)

        partitions = []
        partition_id = 0

        # Create combined partitions for 4-bit patterns (dequantize_affine + linear/embedding)
        for dequant_node, consumer_node in patterns_4bit:
            partition = Partition(id=partition_id, nodes=[dequant_node, consumer_node])
            partitions.append(partition)
            partition_id += 1

        # Create single-node partitions for all other supported operations
        for node in exported_program.graph_module.graph.nodes:
            if node in pattern_nodes:
                continue  # Skip nodes that are part of 4-bit patterns

            if op_support.is_node_supported(None, node):
                # Check if the node actually has tensor inputs/outputs to avoid empty delegates
                # Skip operations that don't have meaningful computation (like constants)
                if node.op == "get_attr":
                    continue

                partition = Partition(id=partition_id, nodes=[node])
                partitions.append(partition)
                partition_id += 1

        return partitions

    def partition(self, exported_program: ExportedProgram) -> PartitionResult:
        """Partition the graph into single-operation delegates with special 4-bit quantization handling."""
        logger.info("SingleOpCoreMLPartitioner::partition")
        partition_tags = {}

        # Create single-op partitions with special 4-bit pattern handling
        partition_list = self._create_single_op_partitions(exported_program)

        for partition in partition_list:
            for node in partition.nodes:
                tag = f"tag{partition.id}"
                node.meta["delegation_tag"] = tag
                partition_tags[tag] = self.delegation_spec

        if self.take_over_constant_data:
            tag_constant_data(exported_program)
        if self.take_over_mutable_buffer:
            logger.info(
                "Core ML partitioner will take over torch mutable buffer as Core ML state, "
                "so if your model contains mutable buffer, "
                "then you will need MacOS15+/iOS18+ to execute. "
                "If you want your mutable buffer model to be compatible with older OS, "
                "then please set `take_over_mutable_buffer=False`"
            )
            tag_mutated_buffer(exported_program)

        return PartitionResult(
            tagged_exported_program=exported_program, partition_tags=partition_tags
        )

    def log_once(self, msg: str) -> None:
        if msg not in self._logged_msgs:
            logging.info(msg)
            self._logged_msgs.add(msg)

    def ops_to_not_decompose(
        self, ep: ExportedProgram
    ) -> Tuple[List[torch._ops.OpOverload], Optional[Callable[[torch.fx.Node], bool]]]:
        """Reuse the same logic as the original CoreMLPartitioner."""
        do_not_decompose = []
        op_support = _OperatorsSupportedForCoreMLBackend(
            self.skip_ops_for_coreml_delegation,
            lower_full_graph=False,
            log=False,
        )

        do_not_decompose_blocklist = [
            torch.ops.aten.triu.default,
            torch.ops.aten.tril.default,
            torch.ops.aten.repeat_interleave.self_int,
            torch.ops.aten.repeat_interleave.self_Tensor,
        ]
        for node in ep.graph.nodes:
            if node.op == "call_function" and isinstance(
                node.target, torch._ops.OpOverload
            ):
                try:
                    if (
                        op_support.is_node_supported(None, node)
                        and node.target not in do_not_decompose_blocklist
                        and not _is_view_op(node.target)
                    ):
                        do_not_decompose.append(node.target)
                except Exception as e:
                    self.log_once(
                        f"Encountered exception when checking node support, treating node as unsupported: {e}"
                    )
        return do_not_decompose, None


class CoreMLPartitioner(Partitioner):
    def __init__(
        self,
        *,
        skip_ops_for_coreml_delegation: Optional[List[str]] = None,
        compile_specs: Optional[List[CompileSpec]] = None,
        take_over_mutable_buffer: Optional[bool] = True,
        lower_full_graph: bool = False,
        take_over_constant_data: bool = True,
    ) -> None:
        if skip_ops_for_coreml_delegation is None:
            skip_ops_for_coreml_delegation = []
        self.skip_ops_for_coreml_delegation = skip_ops_for_coreml_delegation

        for compile_spec in compile_specs or []:
            if compile_spec.key == COMPILE_SPEC_KEYS.ENUMERATED_SHAPES.value:
                assert (
                    lower_full_graph
                ), "lower_full_graph must be True in the CoreMLPartitioner when using an enumerated shape compile spec"

        self.delegation_spec = DelegationSpec(
            backend_id=CoreMLBackend.__name__,
            compile_specs=compile_specs if compile_specs is not None else [],
        )
        self.take_over_mutable_buffer = take_over_mutable_buffer
        self.lower_full_graph = lower_full_graph
        self.take_over_constant_data = take_over_constant_data
        self._logged_msgs = set()

        if self.lower_full_graph:
            assert (
                len(self.skip_ops_for_coreml_delegation) == 0
            ), "When lower_full_graph=True, you cannot set skip_ops_for_coreml_delegation"
            assert (
                self.take_over_constant_data
            ), "When lower_full_graph=True, you must set take_over_constant_data=True"
            assert (
                self.take_over_mutable_buffer
            ), "When lower_full_graph=True, you must set take_over_mutable_buffer=True"

    def partition(self, exported_program: ExportedProgram) -> PartitionResult:
        # Run the CapabilityBasedPartitioner to return the largest possible
        # subgraphs containing the nodes with the tags
        logger.info("CoreMLPartitioner::partition")
        partition_tags = {}

        capability_partitioner = CapabilityBasedPartitioner(
            exported_program.graph_module,
            _OperatorsSupportedForCoreMLBackend(
                self.skip_ops_for_coreml_delegation,
                self.lower_full_graph,
                log=True,
            ),
            allows_single_node_partition=True,
        )
        partition_list = capability_partitioner.propose_partitions()
        for partition in partition_list:
            for node in partition.nodes:
                tag = f"tag{partition.id}"
                node.meta["delegation_tag"] = tag
                partition_tags[tag] = self.delegation_spec

        if self.take_over_constant_data:
            tag_constant_data(exported_program)
        if self.take_over_mutable_buffer:
            logger.info(
                "Core ML partitioner will take over torch mutable buffer as Core ML state, "
                "so if your model contains mutable buffer, "
                "then you will need MacOS15+/iOS18+ to execute. "
                "If you want your mutable buffer model to be compatible with older OS, "
                "then please set `take_over_mutable_buffer=False`"
            )
            tag_mutated_buffer(exported_program)

        return PartitionResult(
            tagged_exported_program=exported_program, partition_tags=partition_tags
        )

    def log_once(self, msg: str) -> None:
        if msg not in self._logged_msgs:
            logging.info(msg)
            self._logged_msgs.add(msg)

    def ops_to_not_decompose(
        self, ep: ExportedProgram
    ) -> Tuple[List[torch._ops.OpOverload], Optional[Callable[[torch.fx.Node], bool]]]:
        do_not_decompose = []
        op_support = _OperatorsSupportedForCoreMLBackend(
            self.skip_ops_for_coreml_delegation,
            self.lower_full_graph,
            log=False,
        )

        # CoreML prevents certain ops (like triu) from lowering to CoreML when put in the ExecuTorch op namespace
        # TODO: upstream fixes, but pending ET consuming a new published version of coremltools with the
        # desired changes, we need to manually block them here
        do_not_decompose_blocklist = [
            # https://github.com/apple/coremltools/blob/release/8.3/coremltools/converters/mil/frontend/torch/ops.py#L6965-L6966
            torch.ops.aten.triu.default,
            # https://github.com/apple/coremltools/blob/release/8.3/coremltools/converters/mil/frontend/torch/ops.py#L6997-L6998
            torch.ops.aten.tril.default,
            # CoreML's translation of repeat_interleave has poor perf
            torch.ops.aten.repeat_interleave.self_int,
            torch.ops.aten.repeat_interleave.self_Tensor,
        ]
        for node in ep.graph.nodes:
            if node.op == "call_function" and isinstance(
                node.target, torch._ops.OpOverload
            ):
                try:
                    if (
                        op_support.is_node_supported(None, node)
                        and node.target not in do_not_decompose_blocklist
                        and not _is_view_op(node.target)
                    ):
                        do_not_decompose.append(node.target)
                except Exception as e:
                    # CoreML's op_support.is_node_supported will sometimes throw
                    # for unsupported ops, rather than returning False
                    self.log_once(
                        f"Encountered exception when checking node support, treating node as unsupported: {e}"
                    )
        return do_not_decompose, None
