import logging

import torch
from torch_tensorrt.dynamo._settings import CompilationSettings

logger = logging.getLogger(__name__)


def remove_detach(
    gm: torch.fx.GraphModule, settings: CompilationSettings
) -> torch.fx.GraphModule:
    """Remove detach ops in the graph"""
    count = 0
    for node in gm.graph.nodes:
        # node.target = "detach" in torch.compile workflow
        if node.target == torch.ops.aten.detach.default or node.target == "detach":
            # Detach node has only one input
            node_input = node.all_input_nodes[0]
            node.replace_all_uses_with(node_input)
            gm.graph.erase_node(node)
            count += 1

    logger.debug(f"Removed {count} detach nodes:\n{gm.graph}")

    return gm
