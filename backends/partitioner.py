# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from abc import ABC, abstractmethod
from typing import Dict, List, NamedTuple, TypeVar

import torch.fx as fx

from executorch.backends.backend_details import enforcedmethod
from executorch.backends.compile_spec_schema import CompileSpec


class DelegationSpec(NamedTuple):
    backend_id: str
    compile_specs: List[CompileSpec]


class Partitioner(ABC):
    """
    Defines a callable interface for partitioning an exported Module (i.e. a program) for
    backend delegation.
    A partitioner implementation would receive an exported Module, determine what portions of
    the it can be delegated to certain backend (though a partitioner can target multiple
    backends as well), and return the same input Module with specific nodes in
    the input graph tagged for delegation.

    The nodes that intend to be delegated must be tagged (by setting
    node.meta["delegation_tag"]) and this tag must be provided in the
    `partition_tags` dictionary mapping to an instance of
    DelegationSpec(backend_id, method_compilation_spec). Each tag must represent
    a distinct submodule that we intend on lowering and should be fully contained.

    For details on method_compilation_spec see the to_backend API, as these objects follow
    the same format.

    Args:
        edge_graph_module: A module in Edge dialect to be partitioned for backend delegation.
    """

    partition_tags: Dict[str, DelegationSpec]

    def __call__(self, edge_graph_module: fx.GraphModule) -> fx.GraphModule:
        return self.partition(edge_graph_module)

    @enforcedmethod
    @abstractmethod
    def partition(self, edge_graph_module: fx.GraphModule) -> fx.GraphModule:
        """
        Returns the input exported program with newly created sub-Modules encapsulating
        specific portions of the input "tagged" for delegation.

        The specific implementation is free to decide how existing computation in the
        input Module should be delegated to one or even more than one specific
        backends.

        The contract is stringent in that:
        * Each node that is intended to be delegated must be tagged
        * No change in the original input Module (GraphModule) representation can take
        place other than adding sub-Modules for encapsulating existing portions of the
        input Module and the associated metadata for tagging.

        Args:
            edge_graph_module: A module in Edge dialect to be partitioned for backend delegation.

        Returns:
            GraphModule: Returns the input exported program with nodes that
            intend to be delegated containing a "delegate_spec" metadata
        """
        pass


# Define Type variables to allow instantiate an instance a subclass of Partitioner
# in to_backend(edge_graph_module: torch.fx.GraphModule, partitioner: Type[TPartitioner])
TPartitioner = TypeVar("TPartitioner", bound=Partitioner)
