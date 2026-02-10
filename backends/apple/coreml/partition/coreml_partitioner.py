# Copyright © 2023 Apple Inc. All rights reserved.
#
# Please refer to the license found in the LICENSE file in the root directory of the source tree.

import inspect
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

    def _check_if_called_from_to_backend(self) -> bool:
        """
        Check if the partition method is being called from the deprecated to_backend workflow.
        Returns True if called from deprecated direct to_backend, False if called from to_edge_transform_and_lower.
        """
        stack = inspect.stack()

        for frame_info in stack:
            if frame_info.function == "to_edge_transform_and_lower":
                return False

        for frame_info in stack:
            if frame_info.function == "to_backend":
                filename = frame_info.filename
                if "program/_program.py" in filename:
                    return True
        return False

    def partition(self, exported_program: ExportedProgram) -> PartitionResult:
        """
        Override partition to add deprecation warning when called from to_backend.
        """
        # Check if we're being called from the deprecated to_backend workflow
        if self._check_if_called_from_to_backend():
            logger.warning(
                "\nDEPRECATION WARNING: You are using the deprecated 'to_edge() + to_backend()' workflow. "
                "This may result in decreased performance because ExecuTorch decomposes ops (e.g., SDPA) "
                "that CoreML has optimized implementations for. "
                "Please consider migrating to 'to_edge_transform_and_lower()' for better performance. "
                "See: https://docs.pytorch.org/executorch/main/backends/coreml/coreml-overview.html#using-the-core-ml-backend"
            )

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
