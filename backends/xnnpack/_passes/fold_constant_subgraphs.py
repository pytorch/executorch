# Copyright 2026 Arm Limited and/or its affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import copy
import operator

import torch

from executorch.backends.xnnpack._passes.constant_folding import fold_constants

from executorch.exir.passes.constant_prop_pass import (
    _nodes_written_in_place,
    _storages,
    constant_prop_pass,
    get_constant_placeholder_dict,
)
from executorch.exir.passes.insert_write_back_for_buffers_pass import _aliasing_inputs
from torch.export import ExportedProgram
from torch.export.graph_signature import InputKind, OutputKind
from torch.multiprocessing.reductions import StorageWeakRef

_CONSTANT_SUBGRAPH_OPS = {
    torch.ops.aten.cat.default,
    torch.ops.aten.chunk.default,
    torch.ops.aten.split.Tensor,
    torch.ops.aten.split_with_sizes.default,
    operator.getitem,
    torch.ops.aten._weight_norm.default,
}


def _excluded_nodes(exported_program: ExportedProgram) -> set[torch.fx.Node]:
    mutated_targets = {
        spec.target
        for spec in exported_program.graph_signature.output_specs
        if spec.kind in (OutputKind.BUFFER_MUTATION, OutputKind.USER_INPUT_MUTATION)
    }
    mutated_inputs = {
        spec.arg.name
        for spec in exported_program.graph_signature.input_specs
        if spec.target in mutated_targets or spec.arg.name in mutated_targets
    }
    excluded = _nodes_written_in_place(exported_program.graph_module)
    for node in exported_program.graph.nodes:
        if (
            node.name in mutated_inputs
            or node.meta.get("custom", {}).get("delegate_constant_tag") is not None
        ):
            excluded.add(node)
        if isinstance(node.target, torch._ops.HigherOrderOperator):
            # Unknown operand effects must not freeze data read by the region.
            excluded.update(node.all_input_nodes)

    # Writes and external tags also exclude aliases, including lifted inputs
    # sharing storage. Independent constant branches remain eligible to fold.
    tensors = dict(exported_program.state_dict) | exported_program.constants
    targets = {
        spec.arg.name: spec.target
        for spec in exported_program.graph_signature.input_specs
    }
    aliases = {node: set() for node in exported_program.graph.nodes}
    storage_owners = {}
    for node in exported_program.graph.nodes:
        for base in _aliasing_inputs(node):
            aliases[node].add(base)
            aliases[base].add(node)
        storages = _storages(node)
        tensor = tensors.get(targets.get(node.name))
        if isinstance(tensor, torch.Tensor) and tensor.layout == torch.strided:
            storages.add(StorageWeakRef(tensor.untyped_storage()))
        for storage in storages:
            owner = storage_owners.setdefault(storage, node)
            aliases[node].add(owner)
            aliases[owner].add(node)

    pending = list(excluded)
    while pending:
        node = pending.pop()
        for alias in aliases[node] - excluded:
            excluded.add(alias)
            pending.append(alias)
    return excluded


class FoldConstantSubgraphsPass:
    """
    Fold selected constant subgraphs into direct constants before decomposition.

    This pass folds constant-only subgraphs such as:

    bias_0  bias_1  bias_2       # constant parameters
     \\    ||      //
         cat
          |
      split/chunk
          |
        getitem
          |
     addmm / linear

    Although the resulting bias slices are constant, the intermediate
    cat/split/getitem chain can prevent later lowering and delegation from
    recognizing them as constant inputs.

    Constant propagation evaluates eligible operations ahead of time and replaces
    each projection with a direct constant:

    constant bias_0 -> addmm / linear
    constant bias_1 -> addmm / linear
    constant bias_2 -> addmm / linear

    Mutation, aliases, higher-order operands and external storage tags exclude
    affected paths from folding. Independent constant paths remain eligible.
    """

    def __call__(self, exported_program: ExportedProgram) -> ExportedProgram:
        excluded = _excluded_nodes(exported_program)
        constants = set(get_constant_placeholder_dict(exported_program)) - excluded
        candidates = set()
        for node in exported_program.graph.nodes:
            if (
                node.op == "call_function"
                and node.target in _CONSTANT_SUBGRAPH_OPS
                and node not in excluded
                and not node.is_impure()
                and all(arg in constants for arg in node.all_input_nodes)
            ):
                constants.add(node)
                candidates.add(node)
        if not candidates:
            return exported_program

        candidate_names = {node.name for node in candidates}
        excluded_names = {node.name for node in excluded}
        exported_program = copy.deepcopy(exported_program)
        values = {
            node: value
            for node, value in get_constant_placeholder_dict(exported_program).items()
            if node.name not in excluded_names
        }
        # Early folding uses an allowlist before decomposition; late folding
        # relies on GEMM eligibility established by the partitioner.
        fold_constants(
            exported_program,
            {
                node
                for node in exported_program.graph.nodes
                if node.name in candidate_names
            },
            values,
            InputKind.CONSTANT_TENSOR,
        )
        return constant_prop_pass(
            exported_program,
            custom_skip_targets={
                node.target
                for node in exported_program.graph.nodes
                if node.op == "call_function"
            },
        )
