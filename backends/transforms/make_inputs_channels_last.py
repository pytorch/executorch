# Copyright 2026 NXP
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from typing import Sequence

import executorch.backends.transforms.channels_last_ops  # noqa: F401

import torch

from executorch.exir.dialects._ops import ops as exir_ops
from executorch.exir.pass_base import ExportPass
from torch.export import ExportedProgram

from torch.export.graph_signature import InputKind
from torch.fx.passes.infra.pass_base import PassResult


def _is_4d_contiguous(dim_order: Sequence[int]) -> bool:
    return list(dim_order) == [0, 1, 2, 3]


def _user_input_names(ep: ExportedProgram) -> frozenset[str]:
    return frozenset(
        spec.arg.name
        for spec in ep.graph_signature.input_specs
        if spec.kind == InputKind.USER_INPUT
    )


class MakeInputsChannelsLast(ExportPass):
    """Transforms a model with contiguous 4D user inputs to accept channels-last inputs.

    Parameters/buffers that are also lowered as placeholders are intentionally left untouched. Only nodes whose name
     appears in the graph signature as `InputKind.USER_INPUT` are considered.
    """

    def __init__(self, ep: ExportedProgram) -> None:
        super().__init__()
        self._user_input_names = _user_input_names(ep)

    def call(self, graph_module: torch.fx.GraphModule) -> PassResult:
        modified = False
        graph = graph_module.graph

        for node in list(graph.nodes):
            if node.op != "placeholder":
                continue

            # Skip lifted parameters / buffers.
            if node.name not in self._user_input_names:
                continue

            val = node.meta.get("val")
            if not isinstance(val, torch.Tensor):
                continue

            if val.dim() != 4:
                continue

            if not _is_4d_contiguous(val.dim_order()):
                continue

            # Mark the placeholder as channels-last so callers supply CL data.
            node.meta["val"] = val.to(memory_format=torch.channels_last)

            # Insert a clone immediately after to restore contiguous dim order
            # for the rest of the model.
            with graph.inserting_after(node):
                clone_node = graph.call_function(
                    exir_ops.edge.dim_order_ops._clone_dim_order.default,
                    args=(node,),
                    kwargs={"dim_order": [0, 1, 2, 3]},
                )
                clone_node.meta["val"] = val

            node.replace_all_uses_with(clone_node)
            clone_node.args = (node,)  # restore after replace_all_uses_with

            modified = True

        if modified:
            graph.lint()
            graph_module.recompile()

        return PassResult(graph_module, modified)
