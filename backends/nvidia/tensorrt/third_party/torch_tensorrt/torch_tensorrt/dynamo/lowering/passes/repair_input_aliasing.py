import logging

import torch
from torch_tensorrt.dynamo._settings import CompilationSettings
from torch_tensorrt.dynamo.lowering.passes.pass_utils import get_tensor_placeholders

logger = logging.getLogger(__name__)


def repair_input_aliasing(
    gm: torch.fx.GraphModule, settings: CompilationSettings
) -> torch.fx.GraphModule:
    """Inserts clone operators temporarily ahead of every placeholder

    See: https://github.com/pytorch/pytorch/issues/108079
    Undone by `remove_input_alias_fixing_clones` after tracing
    """
    # Extract graph placeholder Tensors
    placeholders = get_tensor_placeholders(gm)

    for node in placeholders:
        # Insert clones for placeholder nodes to avoid
        # input aliasing or mutation
        with gm.graph.inserting_after(placeholders[-1]):
            cloned_input = gm.graph.call_function(
                torch.ops.aten.clone.default,
                args=(node,),
            )

        # Replace all uses of the placeholder except the cloned node
        # with the cloned placeholder
        node.replace_all_uses_with(
            cloned_input,
            delete_user_cb=lambda node: node != cloned_input,
        )

    gm.graph.lint()
    gm.recompile()
    logger.debug(f"Inserted auxiliary clone nodes for placeholders:\n{gm.graph}")

    return gm
