# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Dict, List, NamedTuple, TypeVar

from executorch.exir.backend.backend_details import enforcedmethod
from executorch.exir.backend.compile_spec_schema import CompileSpec
from torch.export import ExportedProgram


class DelegationSpec(NamedTuple):
    backend_id: str
    compile_specs: List[CompileSpec]


@dataclass
class PartitionResult:
    """
    tagged_exported_program: the graph with nodes that intend to be delegated containing a "DelegationSpec" metadata
    partition_tags: A dictionary that will be used to keep track of the tags and it's corresponding DelegationSpec. The tag is defined by users and used
    in the node.meta.
    """

    tagged_exported_program: ExportedProgram
    partition_tags: Dict[str, DelegationSpec]


class Partitioner(ABC):
    """
    Defines a callable interface for partitioning an exported program for
    backend delegation.
    A partitioner implementation would receive an exported program, determine what portions of
    the it can be delegated to certain backend (though a partitioner can target multiple
    backends as well), and return the PartitionResult including:
    - the same input module with specific nodes in the input graph tagged for delegation
    - the "partition_tags" to indicate how the tag is mapped to Delegation Spec.

    The nodes that intend to be delegated must be tagged (by setting
    node.meta["delegation_tag"]) and this tag must be provided in the
    `partition_tags` dictionary mapping to an instance of
    DelegationSpec(backend_id, method_compilation_spec). Each tag must represent
    a distinct submodule that we intend on lowering and should be fully contained.

    For details on method_compilation_spec see the to_backend API, as these objects follow
    the same format.

    Args:
        exported_program: An ExportedProgram in Edge dialect to be partitioned for backend delegation.
    """

    def __call__(self, exported_program: ExportedProgram) -> PartitionResult:
        return self.partition(exported_program)

    @enforcedmethod
    @abstractmethod
    def partition(self, exported_program: ExportedProgram) -> PartitionResult:
        """
        Returns the input exported program with newly created sub-Modules encapsulating
        specific portions of the input "tagged" for delegation.

        The specific implementation is free to decide how existing computation in the
        input exported program should be delegated to one or even more than one specific
        backends.

        The contract is stringent in that:
        * Each node that is intended to be delegated must be tagged
        * No change in the original input exported program (ExportedProgram) representation can take
        place other than adding sub-Modules for encapsulating existing portions of the
        input exported program and the associated metadata for tagging.

        Args:
            exported_program: An ExportedProgram in Edge dialect to be partitioned for backend delegation.

        Returns:
            PartitionResult: includes the tagged graph and the delegation spec to indicate what backend_id and compile_spec is used for each node and the tag created by the backend developers.
        """
        pass


# Define Type variables to allow instantiate an instance a subclass of Partitioner
# in to_backend(edge_exported_program: ExportedProgram, partitioner: Type[TPartitioner])
TPartitioner = TypeVar("TPartitioner", bound=Partitioner)
