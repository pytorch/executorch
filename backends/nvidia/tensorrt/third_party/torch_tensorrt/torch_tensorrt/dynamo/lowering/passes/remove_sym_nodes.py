import logging
from typing import Any, Sequence

import torch
from torch_tensorrt.dynamo._settings import CompilationSettings

logger = logging.getLogger(__name__)


def remove_sym_nodes(
    gm: torch.fx.GraphModule,
    sample_inputs: Sequence[Any],
    settings: CompilationSettings,
) -> torch.fx.GraphModule:
    """Remove sym_int placeholders which get inserted due to torch.compile's
    dynamic=True behavior
    """
    # Extract SymInt placeholder Tensors
    placeholder_idx_sym_ints = [
        (idx, node)
        for idx, node in enumerate(gm.graph.nodes)
        if (
            node.op == "placeholder"
            and isinstance(node.type, type)
            and issubclass(node.type, torch.SymInt)
            and not node.users
        )
    ]

    for idx, node in placeholder_idx_sym_ints:
        gm.graph.erase_node(node)
        sample_inputs.pop(idx)

    gm.graph.lint()
    gm.recompile()
    logger.debug(f"Removed SymInt placeholders:\n{gm.graph}")

    return gm
