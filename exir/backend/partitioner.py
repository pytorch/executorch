# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from abc import ABC, abstractmethod
from dataclasses import dataclass
from types import MappingProxyType
from typing import Callable, Dict, List, Mapping, NamedTuple, Optional, Tuple, Union

import torch

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

    def __init__(
        self,
        spec: Mapping[Union[str, int, float, bool], object] = MappingProxyType({}),
    ):
        self._spec = spec

    def __call__(self, exported_program: ExportedProgram) -> PartitionResult:
        return self.partition(exported_program)

    @property
    def spec(self) -> Mapping[Union[str, int, float, bool], object]:
        return self._spec

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

    def ops_to_not_decompose(
        self,
        ep: ExportedProgram,
    ) -> Tuple[List[torch._ops.OpOverload], Optional[Callable[[torch.fx.Node], bool]]]:
        """
        Returns a list of operator names that should not be decomposed. When these ops are
        registered and the `to_backend` is invoked through to_edge_transform_and_lower it will be
        guaranteed that the program that the backend receives will not have any of these ops
        decomposed.

        Returns:
            List[torch._ops.OpOverload]: a list of operator names that should not be decomposed.
            Optional[Callable[[torch.fx.Node], bool]]]: an optional callable, acting as a filter, that users can provide
            which will be called for each node in the graph that users can use as a filter for certain
            nodes that should be continued to be decomposed even though the op they correspond to is
            in the list returned by ops_to_not_decompose.
        """
        return ([], None)
