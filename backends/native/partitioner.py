# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
Native backend partitioner.

Claims core ATen ops (torch.Tag.core) plus an explicit opt-in set, and delegates
whole torch.cond control-flow ops whose branches are fully supported. Graph
cleanup (CSE) runs before lowering via transform_passes, since ExecuTorch forbids
a partitioner from mutating the graph module.
"""

from typing import Callable, final, List, Mapping, Optional, Tuple

import torch

from executorch.exir.backend.compile_spec_schema import CompileSpec
from executorch.exir.backend.partitioner import (
    DelegationSpec,
    Partitioner,
    PartitionResult,
)
from executorch.exir.backend.utils import tag_constant_data, tag_mutated_buffer

from torch.export.exported_program import ExportedProgram
from torch.fx import GraphModule, Node
from torch.fx.passes.infra.partitioner import CapabilityBasedPartitioner
from torch.fx.passes.operator_support import OperatorSupportBase

# Non-core ops the native backend supports. Also preserved (not decomposed).
_SUPPORTED_NON_CORE_OPS = [
    torch.ops.aten.matmul.default,
    torch.ops.aten.linear.default,
    torch.ops.aten.addmm.default,
    torch.ops.aten.scaled_dot_product_attention.default,
]

# Maps a control-flow higher-order op to the arg indices of its branch submodule
# get_attr nodes (cond's true and false fns).
_HOP_SUBMODULE_ARG_INDICES = {
    torch.ops.higher_order.cond: (1, 2),
}

EXTERNAL_CONSTANTS_TAG_KEY = "external_constants_tag"
# Opt preprocess into handing constants back on NativeDelegateInfo instead of
# copying them into a NamedDataStore. Set by to_native; a program lowered this way
# has no constants to serialize into a PTE, so it must not reach to_executorch.
PTN_SERIALIZATION_KEY = "serialize_as_ptn"


class NativeSupportedOperators(OperatorSupportBase):
    _NON_CORE = set(_SUPPORTED_NON_CORE_OPS)

    def is_node_supported(
        self, submodules: Mapping[str, torch.nn.Module], node: Node
    ) -> bool:
        if node.op in ("placeholder", "output", "get_attr"):
            return False
        if node.op != "call_function":
            return False
        if isinstance(node.target, torch._ops.HigherOrderOperator):
            return False

        from executorch.exir.dialects.edge._ops import EdgeOpOverload

        target = node.target
        if isinstance(target, EdgeOpOverload):
            target = target._op
        if isinstance(target, torch._ops.OpOverload):
            if target in self._NON_CORE:
                return True
            return torch.Tag.core in target.tags or torch.Tag.view_copy in target.tags
        return False


def _branch_graph_module(gm: GraphModule, node: Node) -> Optional[GraphModule]:
    """Resolve a get_attr node to the branch/body GraphModule it names."""
    if node.op != "get_attr":
        return None
    try:
        sub = gm.get_submodule(str(node.target))
    except AttributeError:
        return None
    return sub if isinstance(sub, GraphModule) else None


def _hop_branch_nodes(node: Node) -> List[Node]:
    """The get_attr arg nodes carrying a control-flow HOP's branch submodules."""
    indices = _HOP_SUBMODULE_ARG_INDICES.get(node.target)
    if indices is None:
        return []
    out: List[Node] = []
    for i in indices:
        if i < len(node.args):
            arg = node.args[i]
            if isinstance(arg, Node) and arg.op == "get_attr":
                out.append(arg)
    return out


def _submodule_fully_supported(
    gm: GraphModule, op_support: OperatorSupportBase
) -> bool:
    """True if every op in a branch (recursing into nested HOPs) is supported."""
    for node in gm.graph.nodes:
        if node.op != "call_function":
            continue
        if isinstance(node.target, torch._ops.HigherOrderOperator):
            branch_nodes = _hop_branch_nodes(node)
            if not branch_nodes:
                return False
            for b in branch_nodes:
                sub = _branch_graph_module(gm, b)
                if sub is None or not _submodule_fully_supported(sub, op_support):
                    return False
        elif not op_support.is_node_supported({}, node):
            return False
    return True


@final
class NativePartitioner(Partitioner):
    def __init__(
        self,
        external_constants_tag: Optional[str] = "native_weights",
        *,
        _serialize_as_ptn: bool = False,
    ) -> None:
        """Partition for the native backend.

        Args:
            external_constants_tag: NamedDataStore tag for constant data, or None
                to embed it in the PTE.
            _serialize_as_ptn: Private, for ``to_native`` only. A program lowered
                with this set has no constants to serialize into a PTE, so
                ``to_executorch`` would emit an incomplete one. Public callers get
                the ordinary ExecuTorch delegate path.
        """
        from executorch.backends.native.preprocess import NativeBackend

        if _serialize_as_ptn and external_constants_tag is not None:
            raise ValueError(
                "NativePartitioner: external_constants_tag only applies to the "
                "NamedDataStore path; pass external_constants_tag=None alongside "
                "_serialize_as_ptn=True."
            )

        compile_specs: List[CompileSpec] = []
        if external_constants_tag is not None:
            compile_specs.append(
                CompileSpec(
                    EXTERNAL_CONSTANTS_TAG_KEY,
                    external_constants_tag.encode("utf-8"),
                )
            )
        if _serialize_as_ptn:
            compile_specs.append(CompileSpec(PTN_SERIALIZATION_KEY, b"1"))
        self.delegation_spec = DelegationSpec(NativeBackend.__name__, compile_specs)

    def ops_to_not_decompose(
        self, ep: ExportedProgram
    ) -> Tuple[List[torch._ops.OpOverload], Optional[Callable[[Node], bool]]]:
        # Already-partitioned graph -> nothing to preserve.
        from executorch.exir.lowered_backend_module import executorch_call_delegate

        for node in ep.graph.nodes:
            if node.op == "call_function" and node.target is executorch_call_delegate:
                return ([], None)

        present: List[torch._ops.OpOverload] = []
        seen = set()
        for node in ep.graph.nodes:
            if node.op != "call_function":
                continue
            if not isinstance(node.target, torch._ops.OpOverload):
                continue
            if node.target in _SUPPORTED_NON_CORE_OPS and node.target not in seen:
                present.append(node.target)
                seen.add(node.target)

        return (present, None)

    def partition(self, exported_program: ExportedProgram) -> PartitionResult:
        partition_tags = {}
        op_support = NativeSupportedOperators()

        capability_partitioner = CapabilityBasedPartitioner(
            exported_program.graph_module,
            op_support,
            allows_single_node_partition=True,
        )

        partition_list = capability_partitioner.propose_partitions()

        for partition in partition_list:
            for node in partition.nodes:
                tag = f"tag{partition.id}"
                node.meta["delegation_tag"] = tag
                partition_tags[tag] = self.delegation_spec

        # Delegate each control-flow HOP whose branches are fully supported. HOPs
        # are unsupported by is_node_supported (so the capability partitioner skips
        # them); tagging the HOP node and its branch get_attr args together makes
        # the fuser pull the branch submodules into one delegated subgraph.
        gm = exported_program.graph_module
        hop_id = 0
        for node in gm.graph.nodes:
            if node.op != "call_function":
                continue
            if node.target not in _HOP_SUBMODULE_ARG_INDICES:
                continue
            branch_nodes = _hop_branch_nodes(node)
            if not branch_nodes:
                continue
            if not all(
                (sub := _branch_graph_module(gm, b)) is not None
                and _submodule_fully_supported(sub, op_support)
                for b in branch_nodes
            ):
                continue
            tag = f"hop_tag{hop_id}"
            hop_id += 1
            node.meta["delegation_tag"] = tag
            for b in branch_nodes:
                b.meta["delegation_tag"] = tag
            partition_tags[tag] = self.delegation_spec

        tag_constant_data(exported_program)
        tag_mutated_buffer(exported_program)

        return PartitionResult(
            tagged_exported_program=exported_program,
            partition_tags=partition_tags,
        )
