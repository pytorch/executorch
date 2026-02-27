import logging
import operator
from typing import Any

import torch
from torch_tensorrt._enums import dtype
from torch_tensorrt.dynamo._settings import CompilationSettings

from .nodeclassifier import NodeClassifier
from .pass_utils import clean_up_graph_after_modifications

logger = logging.getLogger(__name__)


def is_tensor_node(n: torch.fx.Node) -> bool:
    val = n.meta.get("val", None)
    if hasattr(val, "dtype"):
        return True
    return False


def rule_based_autocast(
    gm: torch.fx.GraphModule, settings: CompilationSettings
) -> torch.fx.GraphModule:
    """Rule-based autocast"""
    if not settings.enable_autocast:
        logger.debug("Autocast is not enabled, skipping rule-based autocast.")
        return gm

    # get config from settings
    autocast_low_precision_type = settings.autocast_low_precision_type
    if autocast_low_precision_type is None:
        return gm
    if isinstance(autocast_low_precision_type, dtype):
        autocast_low_precision_type = autocast_low_precision_type.to(torch.dtype)
    autocast_high_precision_type = torch.float32
    autocast_excluded_nodes = settings.autocast_excluded_nodes
    autocast_excluded_ops = settings.autocast_excluded_ops
    autocast_max_output_threshold = settings.autocast_max_output_threshold
    autocast_max_depth_of_reduction = settings.autocast_max_depth_of_reduction
    reference_data: dict[str, torch.Tensor] = (
        settings.autocast_intermediate_node_outputs
    )
    del settings.autocast_intermediate_node_outputs

    node_classifier = NodeClassifier(
        gm.graph.nodes,
        excluded_nodes=autocast_excluded_nodes,
        excluded_ops=autocast_excluded_ops,
        max_output_threshold=autocast_max_output_threshold,
        max_depth_of_reduction=autocast_max_depth_of_reduction,
    )
    low_precision_nodes, high_precision_nodes = node_classifier.run(reference_data)

    def _cast_all_tensor_args_to_dtype(
        node: torch.fx.Node, arg: Any, dtype: torch.dtype
    ) -> Any:
        """Cast all tensor args to the given dtype

        Args:
            node: The node to insert the cast before
            arg: The argument to cast
            dtype: The dtype to cast to

        Returns:
            The casted argument
        """
        if isinstance(arg, torch.fx.Node) and is_tensor_node(arg):
            val = arg.meta.get("val", None)
            if isinstance(val, torch.Tensor):
                if val.dtype == dtype:
                    return arg
                else:
                    with gm.graph.inserting_before(node):
                        cast = gm.graph.call_function(
                            torch.ops.aten.to.dtype, args=(arg, dtype)
                        )
                    # copy the meta of the original tensor to the casted tensor
                    cast.meta.update(arg.meta)
                    # update the dtype of the casted tensor
                    cast.meta["val"] = cast.meta["val"].to(dtype)
                    return cast
        elif isinstance(arg, (tuple, list)):
            return type(arg)(
                _cast_all_tensor_args_to_dtype(node, a, dtype) for a in arg
            )
        elif isinstance(arg, dict):
            return {
                k: _cast_all_tensor_args_to_dtype(node, v, dtype)
                for k, v in arg.items()
            }
        else:
            return arg

    for node in list(gm.graph.nodes):
        if node.op == "call_function":
            if (
                node.target == torch.ops.higher_order.wrap_with_autocast
                or node.target == operator.getitem
            ):
                continue

            if node.name in low_precision_nodes:
                node.args = _cast_all_tensor_args_to_dtype(
                    node, node.args, autocast_low_precision_type
                )
                node.kwargs = _cast_all_tensor_args_to_dtype(
                    node, node.kwargs, autocast_low_precision_type
                )
                node.meta["val"] = node.meta["val"].to(autocast_low_precision_type)
            elif node.name in high_precision_nodes:
                node.args = _cast_all_tensor_args_to_dtype(
                    node, node.args, autocast_high_precision_type
                )
                node.kwargs = _cast_all_tensor_args_to_dtype(
                    node, node.kwargs, autocast_high_precision_type
                )
                node.meta["val"] = node.meta["val"].to(autocast_high_precision_type)

    gm = clean_up_graph_after_modifications(gm)
    logger.debug("Graph after Autocast based on the rules:\n%s", gm.graph)

    return gm
