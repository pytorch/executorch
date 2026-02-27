import logging

import torch
from torch_tensorrt.dynamo._settings import CompilationSettings
from torch_tensorrt.dynamo.lowering.passes.pass_utils import (
    clean_up_graph_after_modifications,
)

logger = logging.getLogger(__name__)


def remove_num_users_is_0_nodes(
    gm: torch.fx.GraphModule, settings: CompilationSettings
) -> torch.fx.GraphModule:
    """Remove ops that [num_users=0] in the graph"""
    nodes = list(gm.graph.nodes)
    output_node = nodes[-1]

    for node in nodes[::-1]:
        if (
            node != output_node
            and len(node.users) == 0
            and len(node.all_input_nodes) > 0
        ):
            gm.graph.erase_node(node)

    gm = clean_up_graph_after_modifications(gm)

    logger.debug(f"Removed ops that [num_users=0] nodes:\n{gm.graph}")

    return gm
