import logging
import operator
from typing import Any, Callable, Optional, Sequence, Union

import torch
from torch_tensorrt._utils import is_tegra_platform
from torch_tensorrt.dynamo._settings import CompilationSettings
from torch_tensorrt.dynamo.lowering.passes._FakeTensorUpdater import FakeTensorUpdater
from torch_tensorrt.dynamo.lowering.passes.pass_utils import (
    trace_intermediate_node_outputs,
)

from .complex_graph_rewrite import complex_graph_detection
from .constant_folding import constant_fold
from .fuse_prims_broadcast import fuse_prims_broadcast
from .pass_manager import DynamoPassManager
from .remove_assert_nodes import remove_assert_nodes
from .remove_detach import remove_detach
from .remove_input_alias_fixing_clones import remove_input_alias_fixing_clones
from .remove_num_users_is_0_nodes import remove_num_users_is_0_nodes
from .repair_input_as_output import repair_input_as_output
from .replace_fused_rms_norm import replace_fused_rms_norm
from .replace_max_pool_with_indices import replace_max_pool_with_indices
from .rule_based_autocast import rule_based_autocast

pre_lowering_pass_list = [
    remove_detach,
    remove_assert_nodes,
    rule_based_autocast,
]

post_lowering_pass_list = [
    replace_fused_rms_norm,
    remove_input_alias_fixing_clones,
    constant_fold,
    repair_input_as_output,
    fuse_prims_broadcast,
    replace_max_pool_with_indices,
    remove_assert_nodes,
    remove_num_users_is_0_nodes,
    complex_graph_detection,
]

if not is_tegra_platform():
    from .fuse_distributed_ops import fuse_distributed_ops

    post_lowering_pass_list.append(fuse_distributed_ops)

ATEN_POST_LOWERING_PASSES = DynamoPassManager.build_from_passlist(
    post_lowering_pass_list
)

ATEN_PRE_LOWERING_PASSES = DynamoPassManager.build_from_passlist(pre_lowering_pass_list)

logger = logging.getLogger(__name__)


LoweringPassSignature = Callable[
    [torch.fx.GraphModule, Sequence[torch.Tensor]], torch.fx.GraphModule
]


def _aten_lowering_pass(
    *args: LoweringPassSignature,
    index: Optional[int] = None,
    **kwargs: Any,
) -> Union[
    LoweringPassSignature, Callable[[LoweringPassSignature], LoweringPassSignature]
]:
    """Adds a lowering pass to the registry, at a specified index if desired

    If no index is specified, the lowering pass is inserted at the end of the list

    Additional keyword arguments can be passed to configure the lowering pass behavior.
    These will be stored as metadata on the pass function.
    """

    def add_lowering_pass(
        lowering_pass: LoweringPassSignature,
    ) -> LoweringPassSignature:
        # Store additional parameters as metadata on the function
        if kwargs:
            lowering_pass._lowering_pass_config = kwargs

        ATEN_POST_LOWERING_PASSES.add_pass_with_index(lowering_pass, index)
        logger.debug(
            f"Added lowering pass {lowering_pass} to list at index {index} with config {kwargs}, current passlist: {ATEN_POST_LOWERING_PASSES}"
        )
        return lowering_pass

    # If there are arguments specified, the decorator may have been called as-is
    if args:
        # The decorator may only be called with the lowering pass
        # The index must be specified as a keyword argument
        if len(args) == 1 and callable(args[0]):
            return add_lowering_pass(args[0])
        else:
            raise AssertionError(
                f"aten_lowering_pass decorator called with invalid arguments {args} "
                "To specify an index to insert the pass, use the keyword 'index='"
            )
    # If no arguments are specified, the decorator was called with keyword arguments
    else:
        return add_lowering_pass


def _remove_lowering_pass(*, index: int) -> None:
    """Removes a lowering pass at a specific index from the registry"""
    ATEN_POST_LOWERING_PASSES.remove_pass_with_index(index)
    logger.debug(
        f"Removed lowering pass at index {index}, current passlist: {ATEN_POST_LOWERING_PASSES}"
    )
    return


def get_lowering_pass_config(lowering_pass: LoweringPassSignature) -> dict[str, Any]:
    """Get the configuration parameters for a lowering pass function

    Args:
        lowering_pass: The lowering pass function

    Returns:
        Dictionary containing the configuration parameters, or empty dict if none
    """
    return getattr(lowering_pass, "_lowering_pass_config", {})


def post_lowering(
    gm: torch.fx.GraphModule, settings: CompilationSettings = CompilationSettings()
) -> torch.fx.GraphModule:
    """Applies the lowering passes to a graph module after torch.export/ torch.compile and their decompositions, returns the modified GraphModule"""
    logging.debug(
        f"Invoking DynamoPassManager and applying lowering passes: {ATEN_POST_LOWERING_PASSES}"
    )
    fake_mode = torch._export.utils._detect_fake_mode_from_gm(gm)
    fake_tensor_updater = FakeTensorUpdater(gm.graph)
    gm = ATEN_POST_LOWERING_PASSES(gm, settings)
    if fake_mode is not None:
        fake_tensor_updater.incremental_update(fake_mode)

    return gm


def pre_export_lowering(
    ep: torch.export.ExportedProgram,
    settings: CompilationSettings = CompilationSettings(),
) -> torch.fx.GraphModule:
    """Applies the lowering passes to a graph module after torch.export/ torch.compile and their decompositions, returns the modified GraphModule"""
    logging.debug(
        f"Invoking DynamoPassManager and applying lowering passes: {ATEN_PRE_LOWERING_PASSES}"
    )

    # Only for rule-based autocast to collect the intermediate node outputs
    if settings.enable_autocast:
        settings.autocast_intermediate_node_outputs = trace_intermediate_node_outputs(
            ep.module(),
            settings.autocast_calibration_dataloader,
            [torch.ops.higher_order.wrap_with_autocast, operator.getitem],
        )
    gm = ep.graph_module
    gm = ATEN_PRE_LOWERING_PASSES(gm, settings)
    return ep


def dump_lowering_passes() -> str:
    """Returns a string containing the lowering passes"""
    return str(ATEN_POST_LOWERING_PASSES)
