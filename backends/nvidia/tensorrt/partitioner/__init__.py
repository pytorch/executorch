# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""TensorRT partitioner for ExecuTorch."""

from typing import Callable, Dict, List, Optional, Tuple

import torch
from torch.export import ExportedProgram
from torch.fx.passes.infra.partitioner import CapabilityBasedPartitioner

from executorch.backends.nvidia.tensorrt.backend import TensorRTBackend
from executorch.backends.nvidia.tensorrt.partitioner.operator_support import (
    TensorRTOperatorSupport,
)
from executorch.exir.backend.compile_spec_schema import CompileSpec
from executorch.exir.backend.partitioner import (
    DelegationSpec,
    Partitioner,
    PartitionResult,
)
from executorch.exir.backend.utils import tag_constant_data


class TensorRTPartitioner(Partitioner):
    """Partitioner for TensorRT backend.
    """

    def __init__(
        self,
        compile_specs: Optional[List[CompileSpec]] = None,
    ) -> None:
        super().__init__()
        self.compile_specs = compile_specs or []
        self.delegation_spec = DelegationSpec(
            backend_id=TensorRTBackend.__name__,
            compile_specs=self.compile_specs,
        )

    def partition(self, exported_program: ExportedProgram) -> PartitionResult:
        """Partition the graph for TensorRT delegation.

        Identifies subgraphs that can be lowered to the TensorRT backend.
        """

        # Save SymInt expressions in node metadata BEFORE partitioning
        # concretizes them.  After partitioning, partition-boundary
        # placeholder nodes lose their free symbols (expr becomes a
        # constant).  By stashing the original expressions in the node
        # meta, the TRT backend can evaluate exact ranges for derived
        # dimensions (e.g., (s18-1)//8+1) instead of inaccurate
        # proportional scaling.  Node meta survives partitioning via
        # shallow copy.
        for node in exported_program.graph_module.graph.nodes:
            if node.op != "call_function":
                continue
            val = node.meta.get("val")
            if val is None:
                continue
            exprs: Dict[int, Any] = {}  # dim → sympy expr
            if hasattr(val, "shape"):
                for i, s in enumerate(val.shape):
                    sn = getattr(s, "node", None) if hasattr(s, "node") else None
                    if sn is not None:
                        expr = getattr(sn, "expr", None)
                        if expr is not None and getattr(expr, "free_symbols", set()):
                            exprs[i] = expr
            elif hasattr(val, "node"):
                sn = val.node
                expr = getattr(sn, "expr", None)
                if expr is not None and getattr(expr, "free_symbols", set()):
                    exprs[-1] = expr
            if exprs:
                node.meta["trt_symint_exprs"] = exprs

        capability_partitioner = CapabilityBasedPartitioner(
            exported_program.graph_module,
            TensorRTOperatorSupport(),
            allows_single_node_partition=True,
        )
        partition_list = capability_partitioner.propose_partitions()

        partition_tags: Dict[str, DelegationSpec] = {}
        for partition in partition_list:
            tag = f"tensorrt_{partition.id}"
            for node in partition.nodes:
                node.meta["delegation_tag"] = tag
            partition_tags[tag] = self.delegation_spec

        tag_constant_data(exported_program)

        return PartitionResult(
            tagged_exported_program=exported_program,
            partition_tags=partition_tags,
        )

    def ops_to_not_decompose(
        self, ep: ExportedProgram
    ) -> Tuple[List[torch._ops.OpOverload], Optional[Callable[[torch.fx.Node], bool]]]:
        """Return operators that should not be decomposed.
        
        This prevents certain operations from being decomposed during edge transform,
        which can cause partition boundaries when intermediate dim_order operations
        are inserted. By keeping these operations intact, we ensure the entire model
        stays in a single TRT partition.
        
        Args:
            ep: The exported program being partitioned.
            
        Returns:
            Tuple of (list of ops to not decompose, optional filter function).
        """
        # pixel_shuffle decomposes into view + permute + view, and the edge transform
        # inserts dim_order_ops._clone_dim_order operations between them. By preventing
        # decomposition, we keep pixel_shuffle as a single operation that our converter
        # handles directly.
        ops_not_decompose = [
            torch.ops.aten.pixel_shuffle.default,
        ]
        
        return (ops_not_decompose, None)
