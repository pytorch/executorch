import logging
from typing import Any, Set

import torch
from torch_tensorrt._utils import sanitized_torch_version
from torch_tensorrt.dynamo._settings import CompilationSettings
from torch_tensorrt.dynamo.lowering.passes.pass_utils import (
    clean_up_graph_after_modifications,
)

from packaging import version

# Modify import location of utilities based on Torch version
if version.parse(sanitized_torch_version()) < version.parse("2.1.1"):
    from torch._inductor.freezing import ConstantFolder
else:
    from torch._inductor.constant_folding import ConstantFolder

logger = logging.getLogger(__name__)


@torch.utils._python_dispatch._disable_current_modes()  # type: ignore
def constant_fold(
    gm: torch.fx.GraphModule, settings: CompilationSettings
) -> torch.fx.GraphModule:
    """Adapted from:
    https://github.com/pytorch/pytorch/blob/3a79621c9dce17f77fbddc06aab21f6bc477f313/torch/_inductor/freezing.py#L178-L197

    Folds constants in the graph module, not skipping constructors

    Modifies the graph in-place and replaces node with constants
    """
    cf = _TorchTensorRTConstantFolder(gm, skip_constructors=False)
    cf.run()

    # The constants are created on CPU to save GPU memory for TensorRT compilation.
    # For TRT INetwork construction the constants are moved to CPU in get_attr call.
    for node, constant in cf.node_replacements.items():
        if settings.offload_module_to_cpu:
            replace_node_with_constant(
                gm,
                node,
                torch.nn.Parameter(constant.cpu().contiguous(), requires_grad=False),
            )
        else:
            replace_node_with_constant(
                gm, node, torch.nn.Parameter(constant, requires_grad=False)
            )

    erased_params = []
    for node in gm.graph.nodes:
        # If get_attr node has no users, mark it for deletion
        if node.op == "get_attr" and len(node.users) == 0:
            erased_params.append(node)

    # Remove unused nodes from the graph
    for node in erased_params:
        gm.graph.erase_node(node)

    gm = clean_up_graph_after_modifications(gm)
    # Delete the constant folder instance which holds GPU memory
    del cf

    logger.debug(f"Graph after constant folding:\n{gm.graph}")
    return gm


def replace_node_with_constant(
    gm: torch.fx.GraphModule, node: torch.fx.Node, constant: torch.Tensor
) -> None:
    """Adapted from:
    https://github.com/pytorch/pytorch/blob/bcf35c6ae62bb6560befa3550e37a8283944e5f4/torch/_inductor/constant_folding.py#L17-L43

    Modified to register parameters, instead of buffers for frozen constants
    """
    g = gm.graph

    if not hasattr(gm, "_frozen_param_count"):
        gm._frozen_param_count = 0

    i = gm._frozen_param_count

    while True:
        qualname = f"_frozen_param{i}"
        if not hasattr(gm, qualname):
            break
        i += 1

    gm._frozen_param_count = i + 1

    with g.inserting_before(node):
        new_input_node = g.create_node("get_attr", qualname, (), {})
        node.replace_all_uses_with(new_input_node)
        new_input_node.meta.update(node.meta)
        g.erase_node(node)

    # Needed to suppress `does not reference an nn.Module, nn.Parameter, or buffer` warning
    gm.register_parameter(qualname, constant)
    setattr(gm, qualname, constant)


# TODO: Delete this class when the following code is fixed in nightly:
# https://github.com/pytorch/pytorch/blob/4b881b0da390c1290bb12850ef9daad6f6eb2cb6/torch/_inductor/constant_folding.py#L53-L63
class _TorchTensorRTConstantFolder(ConstantFolder):  # type: ignore[misc]
    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        # Set of known quantization ops to be excluded from constant folding.
        # Currently, we exclude all quantization ops coming from modelopt library.
        self.quantization_ops: Set[torch._ops.OpOverload] = set()
        try:
            # modelopt import ensures torch.ops.tensorrt.quantize_op.default is registered
            import modelopt.torch.quantization as mtq  # noqa: F401

            assert torch.ops.tensorrt.quantize_op.default
            assert torch.ops.tensorrt.dynamic_block_quantize_op.default
            self.quantization_ops.add(torch.ops.tensorrt.quantize_op.default)
            self.quantization_ops.add(
                torch.ops.tensorrt.dynamic_block_quantize_op.default
            )
        except Exception as e:
            pass

    # TODO: Update this function when quantization is added
    def is_impure(self, node: torch.fx.node.Node) -> bool:

        if node.target in self.quantization_ops:
            return True
        return False
