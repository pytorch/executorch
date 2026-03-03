import logging

import torch
from torch_tensorrt.dynamo._settings import CompilationSettings
from torch_tensorrt.dynamo.lowering.passes.pass_utils import (
    clean_up_graph_after_modifications,
    get_tensor_placeholders,
)

logger = logging.getLogger(__name__)


def repair_input_as_output(
    gm: torch.fx.GraphModule, settings: CompilationSettings
) -> torch.fx.GraphModule:
    """Repair scenarios where inputs are also outputs of the graph

    TRT does not allow such cases, so we insert a clone (identity) layer
    """
    modified_graph = False

    # Extract graph placeholder Tensors
    placeholders = get_tensor_placeholders(gm)

    for placeholder in placeholders:
        # If any placeholder has any users which are direct graph outputs
        if len(placeholder.users) >= 1 and any(
            user.op == "output" for user in placeholder.users
        ):
            modified_graph = True

            # Get direct graph outputs which are direct uses of placeholders
            direct_outputs = [user for user in placeholder.users if user.op == "output"]

            # Insert clone node for placeholder to ensure placeholder is not a direct output
            with gm.graph.inserting_after(placeholders[-1]):
                cloned_placeholder = gm.graph.call_function(
                    torch.ops.aten.clone.default,
                    args=(placeholder,),
                )

            # Replace placeholder as output with cloned version
            for output in direct_outputs:
                output.replace_input_with(placeholder, cloned_placeholder)

    if modified_graph:
        gm = clean_up_graph_after_modifications(gm)
        logger.debug(f"Graph after repair_input_as_output:\n{gm.graph}")

    return gm
