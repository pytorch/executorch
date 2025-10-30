# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from typing import Dict, final, List, Tuple

import torch

from executorch.exir.backend.backend_details import BackendDetails, PreprocessResult
from executorch.exir.backend.canonical_partitioners.pattern_op_partitioner import (
    generate_pattern_op_partitions,
)

from executorch.exir.backend.compile_spec_schema import CompileSpec
from executorch.exir.backend.partitioner import (
    DelegationSpec,
    Partitioner,
    PartitionResult,
)
from executorch.exir.dialects._ops import ops as exir_ops
from executorch.exir.graph_module import get_control_flow_submodules
from torch._export.utils import is_buffer, is_lifted_tensor_constant, is_param
from torch.export.exported_program import ExportedProgram
from torch.fx.passes.operator_support import any_chain, OperatorSupportBase


def is_param_node(exp_prog: ExportedProgram, node: torch.fx.Node) -> bool:
    return (
        is_param(exp_prog, node)
        or is_buffer(exp_prog, node)
        or is_lifted_tensor_constant(exp_prog, node)
    )


def get_total_num_ops_in_ep(edge_programs, supported_ops):
    total_number_of_ops = 0
    for edge_program in edge_programs.values():
        for partitioned_program in edge_program:
            for node in partitioned_program.graph.nodes:
                if node.op == "call_function":
                    if node.target in supported_ops:
                        total_number_of_ops += 1
    return total_number_of_ops


def _preprocess_multimethod(
    edge_programs: Dict[str, List[ExportedProgram]],
    compile_specs: Dict[str, List[List[CompileSpec]]],
    supported_ops: List[torch._ops.OpOverload],
    backend_name: str,
) -> Dict[str, List[PreprocessResult]]:
    """
    Helper function to abstract out the logic to be shared between the two backends:
    FirstBackendWithPreprocessAll and SecondBackendWithPreprocessAll. This will be used
    in testing for a partitioner which tags different partitions for different backends
    to be lowered to
    """
    total_number_of_ops = get_total_num_ops_in_ep(edge_programs, supported_ops)
    all_processed_results = {key: [] for key in edge_programs.keys()}

    for method_name, partitioned_programs in edge_programs.items():
        compile_specs_for_method = compile_specs[method_name]

        assert len(compile_specs_for_method) == len(partitioned_programs)
        for compile_spec_for_partition, partitioned_program in zip(
            compile_specs_for_method, partitioned_programs
        ):
            debug_handle_map = {}
            processed_bytes = f"{backend_name}#{total_number_of_ops}#"
            for node in partitioned_program.graph.nodes:
                if node.op == "call_function":
                    if node.target in supported_ops:
                        op_name = node.target.__name__
                        processed_bytes += f"{op_name}:"
                        original_debug_id = node.meta["debug_handle"]
                        new_debug_id = original_debug_id
                        debug_handle_map[new_debug_id] = (original_debug_id,)
                    else:
                        raise RuntimeError(
                            f"{node.op} {node.target.__name__} is not supported in backend {backend_name}"
                        )
                if is_param_node(partitioned_program, node):
                    processed_bytes += f"CONST{node.name}:"

            processed_bytes += "#"
            for cs in compile_spec_for_partition:
                processed_bytes += f"{cs.key}:{cs.value};"

            all_processed_results[method_name].append(
                PreprocessResult(
                    processed_bytes=bytes(processed_bytes, encoding="utf8"),
                    debug_handle_map=debug_handle_map,
                )
            )

    return all_processed_results


@final
class FirstBackendWithPreprocessAll(BackendDetails):
    """
    Backend used to test the preprocess_multimethod for multi methods lowering.
    lowered modules are returned in the format:
    FirstBackendWithPreprocessAll#<all number of ops>#<op1>:<op2>:<op3>#<compile_spec.key>;<compile_spec.value>:


    lowered blobs are not functional, and are purely used for testing purposes
    """

    @staticmethod
    def preprocess(
        edge_program: ExportedProgram,
        compile_specs: List[CompileSpec],
    ) -> PreprocessResult:
        """
        Not used for testing
        """
        return PreprocessResult(
            processed_bytes=bytes(b"\x00"),
            debug_handle_map={},
        )

    @staticmethod
    def preprocess_multimethod(
        edge_programs: Dict[str, List[ExportedProgram]],
        compile_specs: Dict[str, List[List[CompileSpec]]],
    ) -> Dict[str, list[PreprocessResult]]:
        """
        Preprocess all the edge programs in the given dictionary and return a dictionary
        of preprocess results. The preprocess result is a tuple of processed bytes and
        a map from the node name to the original debug handle.
        """
        match_ops = [
            exir_ops.edge.aten.sin.default,
            exir_ops.edge.aten.add.Tensor,
        ]

        return _preprocess_multimethod(
            edge_programs, compile_specs, match_ops, "FirstBackendWithPreprocessAll"
        )


@final
class SecondBackendWithPreprocessAll(BackendDetails):
    """
    Backend used to test the preprocess_multimethod for multi methods lowering.
    lowered modules are returned in the format:
    SecondBackendWithPreprocessAll#<all number of ops>#<op1>:<op2>:<op3>#<compile_spec.key>;<compile_spec.value>:


    lowered blobs are not functional, and are purely used for testing purposes
    """

    @staticmethod
    def preprocess(
        edge_program: ExportedProgram,
        compile_specs: List[CompileSpec],
    ) -> PreprocessResult:
        """
        Not used for testing
        """
        return PreprocessResult(
            processed_bytes=bytes(b"\x00"),
            debug_handle_map={},
        )

    @staticmethod
    def preprocess_multimethod(
        edge_programs: Dict[str, List[ExportedProgram]],
        compile_specs: Dict[str, List[List[CompileSpec]]],
    ) -> Dict[str, list[PreprocessResult]]:
        """
        Preprocess all the edge programs in the given dictionary and return a dictionary
        of preprocess results. The preprocess result is a tuple of processed bytes and
        a map from the node name to the original debug handle.
        """
        match_ops = [
            exir_ops.edge.aten.cos.default,
            exir_ops.edge.aten.sub.Tensor,
        ]

        return _preprocess_multimethod(
            edge_programs, compile_specs, match_ops, "SecondBackendWithPreprocessAll"
        )


class AddSinOperatorSupport(OperatorSupportBase):
    def __init__(self, original_program):
        self.original_program = original_program
        super().__init__()

    def is_node_supported(self, submodules, node: torch.fx.Node) -> bool:
        supported_targets = [
            exir_ops.edge.aten.add.Tensor,
            exir_ops.edge.aten.sin.default,
        ]
        if node.op == "call_function" and node.target in supported_targets:
            return True

        if node.op == "placeholder" and is_param_node(self.original_program, node):
            for user in node.users.keys():
                if user.target in supported_targets:
                    return True
        return False


class SubCosOperatorSupport(OperatorSupportBase):
    def __init__(self, original_program):
        self.original_program = original_program
        super().__init__()

    def is_node_supported(self, submodules, node: torch.fx.Node) -> bool:
        return node.op == "call_function" and node.target in [
            exir_ops.edge.aten.sub.Tensor,
            exir_ops.edge.aten.cos.default,
        ]


@final
class BackendWithPreprocessAllPartitioner(Partitioner):
    """
    Partitioner that partitions for both FirstBackendWithPreprocessAll
    and SecondBackendWithPreprocessAll.

    - The partitioner tags all sin and add nodes for delegation to
      FirstBackendWithPreprocessAll
    - The partitioner tags all cos and sub nodes for delegation to
      SecondBackendWithPreprocessAll
    """

    def __init__(self) -> None:
        self.sub_cos_backend_id = SecondBackendWithPreprocessAll.__name__
        self.add_sin_backend_id = FirstBackendWithPreprocessAll.__name__

    def _partition_graph_module(
        self,
        graph_module: torch.fx.GraphModule,
        id_start=0,
    ) -> Tuple[Dict[str, DelegationSpec], int]:
        partition_tags: Dict[str, DelegationSpec] = {}

        num_partitions_in_gm = 0
        for op_support, backend_id, tag_prefix in [
            (self.add_sin_support, self.add_sin_backend_id, "first"),
            (self.sub_cos_support, self.sub_cos_backend_id, "second"),
        ]:
            partition_list = generate_pattern_op_partitions(
                graph_module, op_support=op_support
            )
            num_partitions_in_gm = num_partitions_in_gm + len(partition_list)
            for partition in partition_list:
                compile_specs = []
                delegation_tag = f"{tag_prefix}_tag{id_start + partition.id}"
                for node in partition.nodes:
                    node.meta["delegation_tag"] = delegation_tag
                    if (
                        node.op == "call_function"
                        and node.target == exir_ops.edge.aten.add.Tensor
                    ):
                        compile_specs.append(CompileSpec("add", bytes(b"\x00")))
                    if (
                        node.op == "call_function"
                        and node.target == exir_ops.edge.aten.sin.default
                    ):
                        compile_specs.append(CompileSpec("sin", bytes(b"\x01")))
                    if (
                        node.op == "call_function"
                        and node.target == exir_ops.edge.aten.sub.Tensor
                    ):
                        compile_specs.append(CompileSpec("sub", bytes(b"\x02")))
                    if (
                        node.op == "call_function"
                        and node.target == exir_ops.edge.aten.cos.default
                    ):
                        compile_specs.append(CompileSpec("cos", bytes(b"\x03")))

                delegation_spec = DelegationSpec(backend_id, compile_specs)
                partition_tags[delegation_tag] = delegation_spec

        start_idx_for_submodules = num_partitions_in_gm
        for _, submodule, _ in get_control_flow_submodules(graph_module):
            ret_partition_tags, start_idx_for_submodules = self._partition_graph_module(
                submodule, id_start=start_idx_for_submodules
            )
            partition_tags.update(ret_partition_tags)

        return partition_tags, start_idx_for_submodules

    def partition(self, exported_program: ExportedProgram) -> PartitionResult:
        self.add_sin_support = any_chain(AddSinOperatorSupport(exported_program))
        self.sub_cos_support = any_chain(SubCosOperatorSupport(exported_program))
        partition_tags, _ = self._partition_graph_module(exported_program.graph_module)
        return PartitionResult(
            tagged_exported_program=exported_program, partition_tags=partition_tags
        )
