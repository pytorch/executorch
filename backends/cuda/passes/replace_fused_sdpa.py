# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
Graph Transformation Pass for Fused Attention Replacement.

Decomposing ``scaled_dot_product_attention`` on CUDA, as ``torch.export``'s
default decompositions and Torch-TensorRT do, turns it into one of its fused
CUDA variants. The backend has no kernel for those, so this pass turns them
back into ``scaled_dot_product_attention``, which the Triton SDPA kernel or
Inductor's math decomposition lowers.
"""

import logging
import operator
from typing import Any, Dict, Optional

from executorch.exir.dialects._ops import ops as exir_ops
from torch.export import ExportedProgram
from torch.export.graph_signature import ExportGraphSignature
from torch.fx import GraphModule, Node
from torch.fx.experimental.symbolic_shapes import statically_known_true

logger = logging.getLogger(__name__)

_FUSED_SDPA_OPS = (
    exir_ops.edge.aten._scaled_dot_product_flash_attention.default,
    exir_ops.edge.aten._scaled_dot_product_efficient_attention.default,
    exir_ops.edge.aten._scaled_dot_product_cudnn_attention.default,
)


class ReplaceFusedSdpaWithSdpaPass:
    """
    Replace each fused SDPA node with ``scaled_dot_product_attention`` when
    only its attention output is used and ``scaled_dot_product_attention``
    computes the same result. Any other node stays as it is.
    """

    requires_exported_program = True

    def __call__(self, exported_program: ExportedProgram) -> None:
        program_module = exported_program.graph_module
        for module in program_module.modules():
            if not isinstance(module, GraphModule):
                continue
            # The signature names the outputs of the program's own graph only.
            signature = (
                exported_program.graph_signature if module is program_module else None
            )
            modified = False
            for node in list(module.graph.nodes):
                if node.op != "call_function" or node.target not in _FUSED_SDPA_OPS:
                    continue
                values = _arguments(node)
                reason = _reason_to_keep(node, values)
                if reason is None:
                    _replace_with_sdpa(module, node, values, signature)
                    modified = True
                else:
                    logger.warning(
                        "The CUDA backend cannot lower %s, since %s",
                        node.format_node(),
                        reason,
                    )
            if modified:
                module.recompile()


def _arguments(node: Node) -> Dict[str, Any]:
    arguments = node.target._schema.arguments
    values = {a.name: a.default_value for a in arguments}
    values.update(zip((a.name for a in arguments), node.args))
    values.update(node.kwargs)
    return values


def _reason_to_keep(node: Node, values: Dict[str, Any]) -> Optional[str]:
    if any(
        user.target is not operator.getitem or user.args[1] != 0 for user in node.users
    ):
        return "an output other than the attention output is used"
    if not values["is_causal"]:
        return None
    if values.get("attn_bias") is not None:
        return "scaled_dot_product_attention takes no mask together with is_causal"
    # Flash attention aligns the causal mask to the bottom right, the other
    # variants and scaled_dot_product_attention to the top left, and the Triton
    # decode kernel ignores it.
    query, key = values["query"].meta["val"], values["key"].meta["val"]
    if not statically_known_true(query.shape[-2] == key.shape[-2]):
        return "its causal mask may have different query and key lengths"
    return None


def _replace_with_sdpa(
    module: GraphModule,
    node: Node,
    values: Dict[str, Any],
    signature: Optional[ExportGraphSignature],
) -> None:
    query, key = values["query"], values["key"]
    kwargs = {}
    # The Triton kernels take a float scale, so leave out the default.
    if values["scale"] is not None:
        kwargs["scale"] = values["scale"]
    # The fused variants that accept fewer key heads than query heads group
    # them as enable_gqa does.
    if not statically_known_true(
        query.meta["val"].shape[-3] == key.meta["val"].shape[-3]
    ):
        kwargs["enable_gqa"] = True
    with module.graph.inserting_before(node):
        sdpa = module.graph.call_function(
            exir_ops.edge.aten.scaled_dot_product_attention.default,
            args=(
                query,
                key,
                values["value"],
                # Only the flash variant has no bias.
                values.get("attn_bias"),
                values["dropout_p"],
                values["is_causal"],
            ),
            kwargs=kwargs,
        )
    sdpa.meta = {**node.meta, "val": node.meta["val"][0]}
    for user in list(node.users):
        user.replace_all_uses_with(sdpa)
        if signature is not None:
            signature.replace_all_uses(user.name, sdpa.name)
        module.graph.erase_node(user)
    module.graph.erase_node(node)
