import logging

import torch
from torch_tensorrt.dynamo._settings import CompilationSettings
from torch_tensorrt.dynamo.lowering.passes.pass_utils import (
    clean_up_graph_after_modifications,
)

logger = logging.getLogger(__name__)


# TODO: Add relevant prims to this fusion
def fuse_prims_broadcast(
    gm: torch.fx.GraphModule, settings: CompilationSettings
) -> torch.fx.GraphModule:
    """Fuses prim nodes which are effectively the ATen equivalents with keep_dim=True"""
    modified_graph = False

    for node in gm.graph.nodes:
        # If the node is a sum prims operator, with broadcast_in_dim being the only consumer
        # it is a candidate for fusing
        if (
            node.target in (torch.ops.prims.sum.default,)
            and len(node.users) == 1
            and list(node.users)[0].target == torch.ops.prims.broadcast_in_dim.default
        ):
            # Get broadcasted shape, reduced dimensions, and original tensor shape
            broadcast_node = list(node.users)[0]
            broadcasted_shape = broadcast_node.args[1]
            reduced_dims = node.args[1]
            original_shape = node.args[0].meta["tensor_meta"].shape

            # If the rank of the broadcasted shape is the same as the original
            # and the broadcasts are all singletons for the reduced dimensions
            # and all of the non-reduced dimensions are identical to the originals

            # Then the broadcast is effectively performing a "keep_dim=True" operation
            if (
                len(broadcasted_shape) == len(original_shape)
                and all(broadcasted_shape[i] == 1 for i in reduced_dims)
                and all(
                    broadcasted_shape[j] == original_shape[j]
                    for j in range(len(original_shape))
                    if j not in reduced_dims
                )
            ):
                # Fuse the operator to its convertible alternative
                with gm.graph.inserting_after(broadcast_node):
                    modified_graph = True

                    if node.target == torch.ops.prims.sum.default:
                        fused_node = gm.graph.call_function(
                            torch.ops.aten.sum.dim_IntList,
                            args=(node.args[0], reduced_dims, True),
                        )

                # Replace all uses of the placeholder except the cloned node
                # with the cloned placeholder
                broadcast_node.replace_all_uses_with(
                    fused_node,
                )

                # Erase uses of the broadcast node and original
                gm.graph.erase_node(broadcast_node)
                gm.graph.erase_node(node)

    if modified_graph:
        gm = clean_up_graph_after_modifications(gm)
        logger.debug(f"Graph after fusing prims-broadcast paradigm:\n{gm.graph}")

    return gm
