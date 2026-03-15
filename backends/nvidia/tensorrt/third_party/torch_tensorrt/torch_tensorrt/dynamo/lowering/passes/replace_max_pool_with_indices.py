import logging
import operator

import torch
from torch_tensorrt.dynamo._settings import CompilationSettings
from torch_tensorrt.dynamo.lowering.passes.pass_utils import (
    clean_up_graph_after_modifications,
)

logger = logging.getLogger(__name__)


def replace_max_pool_with_indices(
    gm: torch.fx.GraphModule, settings: CompilationSettings
) -> torch.fx.GraphModule:
    """Replace MaxPool nodes which return unused indices"""
    replacement_dict = {
        torch.ops.aten.max_pool1d_with_indices.default: torch.ops.aten.max_pool1d.default,
        torch.ops.aten.max_pool2d_with_indices.default: torch.ops.aten.max_pool2d.default,
        torch.ops.aten.max_pool3d_with_indices.default: torch.ops.aten.max_pool3d.default,
    }

    modified_graph = False

    for node in gm.graph.nodes:
        # If the node is a placeholder and its only user is a clone node
        # it was modified by the input alias-fixing pass, and the change
        # needs to be undone
        if (
            node.target in replacement_dict
            and len(node.users) == 1
            and list(node.users)[0].target == operator.getitem
            and list(node.users)[0].args[1] == 0
        ):
            modified_graph = True

            # Replace all uses of the clone with the placeholder, delete the clone
            getitem_node = list(node.users)[0]

            with gm.graph.inserting_after(getitem_node):
                maxpool_fused = gm.graph.call_function(
                    replacement_dict[node.target],
                    args=node.args,
                    kwargs=node.kwargs,
                )
                maxpool_fused.meta = node.meta
                # The metadata for this node should exclude the indices metadata
                maxpool_fused.meta["val"] = maxpool_fused.meta["val"][0]
            logger.debug(
                f"Replacing all uses of nodes {node}, {getitem_node} with fused maxpool node {maxpool_fused} "
                f"is the only user of placeholder {node} and was inserted by the compiler."
            )

            getitem_node.replace_all_uses_with(maxpool_fused)
            gm.graph.erase_node(getitem_node)
            gm.graph.erase_node(node)

    if modified_graph:
        gm = clean_up_graph_after_modifications(gm)
        logger.debug(f"Graph after fusing maxpool operators with indices:\n{gm.graph}")

    return gm
