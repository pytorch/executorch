# Copyright 2025-2026 Arm Limited and/or its affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Fold TOSA TRANSPOSE operations on constant tensors at compile time.

This pass identifies TRANSPOSE operations where the input is a static tensor
(parameter, buffer, or lifted tensor constant) and folds the transpose at
compile time by:
1. Actually permuting the tensor data
2. Creating a new constant placeholder with the permuted data
3. Removing the transpose node and rewiring users

This eliminates runtime transpose operations on static tensors like weights,
which is especially important for Ethos-U55 where Vela implements transposes
as expensive NPU_OP_POOL (1x1 AvgPool) sequences.
"""

import logging
from typing import Sequence, Set, Type

import torch
import torch.fx
from executorch.backends.arm._passes.arm_pass import ArmPass
from executorch.backends.arm._passes.arm_pass_utils import (
    get_constant_placeholder_kind,
    get_param_tensor,
    is_param_node,
    is_persistent_buffer,
)
from executorch.backends.transforms.utils import (
    create_constant_placeholder,
    delete_constant_placeholder,
)
from executorch.exir import ExportedProgram
from executorch.exir.dialects._ops import ops as exir_ops
from executorch.exir.pass_base import ExportPass, PassResult

logger = logging.getLogger(__name__)


def _get_transpose_perm(node: torch.fx.Node) -> list[int] | None:
    """Extract the permutation order from a TRANSPOSE node.

    Args:
        node: A node with target exir_ops.backend.tosa.TRANSPOSE.default

    Returns:
        The permutation order as a list of ints, or None if not extractable.
    """
    if node.target != exir_ops.backend.tosa.TRANSPOSE.default:
        return None

    if len(node.args) < 2:
        return None

    perm = node.args[1]
    if isinstance(perm, (list, tuple)):
        return list(perm)

    return None


class FoldConstantTransposePass(ArmPass):
    """Folds TOSA TRANSPOSE operations on constant tensors at compile time.

    This pass transforms patterns like:
        static_weight (placeholder) -> TRANSPOSE [0,2,3,1] -> Conv2D

    Into:
        static_weight_transposed (placeholder with pre-permuted data) -> Conv2D

    This eliminates runtime transposes on static tensors, which is especially
    beneficial for Ethos-U55 where Vela implements transposes as expensive
    NPU_OP_POOL (1x1 AvgPool) sequences.

    Example:
        Before: weight (NCHW) -> TRANSPOSE -> Conv2D (expects NHWC)
        After: weight_nhwc (already NHWC) -> Conv2D

    Note:
        This pass only folds transposes on constant/static tensors. Transposes
        on dynamic activations (runtime data) cannot be folded and must be
        executed at runtime.
    """

    _passes_required_after: Set[Type[ExportPass]] = set()

    def __init__(self, exported_program: ExportedProgram, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.exported_program = exported_program

    def _fold_transpose(
        self,
        graph_module: torch.fx.GraphModule,
        transpose_node: torch.fx.Node,
        input_node: torch.fx.Node,
        perm: Sequence[int],
    ) -> bool:
        """Fold a single transpose operation on a constant tensor.

        Args:
            graph_module: The graph module being transformed.
            transpose_node: The TRANSPOSE node to fold.
            input_node: The constant input node (parameter/buffer/lifted constant).
            perm: The permutation order.

        Returns:
            True if the transpose was successfully folded, False otherwise.
        """
        # Get the original tensor data
        tensor = get_param_tensor(self.exported_program, input_node)
        if tensor is None:
            logger.debug(
                f"FoldConstantTransposePass: Could not get tensor for {input_node.name}"
            )
            return False

        # Validate permutation
        if len(perm) != tensor.dim():
            logger.warning(
                f"FoldConstantTransposePass: Permutation length {len(perm)} does not "
                f"match tensor rank {tensor.dim()} for {transpose_node.name}"
            )
            return False

        # Actually permute the data at compile time
        try:
            permuted_tensor = tensor.permute(perm).contiguous()
        except Exception as e:
            logger.warning(
                f"FoldConstantTransposePass: Failed to permute tensor for "
                f"{transpose_node.name}: {e}"
            )
            return False

        # Determine the kind and persistence of the original constant
        try:
            input_kind = get_constant_placeholder_kind(self.exported_program, input_node)
        except RuntimeError:
            logger.debug(
                f"FoldConstantTransposePass: {input_node.name} is not a constant placeholder"
            )
            return False

        persistent = is_persistent_buffer(self.exported_program, input_node)

        # Create new constant placeholder with permuted data
        with graph_module.graph.inserting_before(input_node):
            const_node = create_constant_placeholder(
                exp_program=self.exported_program,
                graph=graph_module.graph,
                kind=input_kind,
                name=f"{input_node.name}_transposed",
                data=permuted_tensor,
                persistent_buffer=persistent if persistent is not None else True,
            )

        # Copy relevant metadata from the transpose output
        if "tosa_dim_order" in transpose_node.meta:
            const_node.meta["tosa_dim_order"] = transpose_node.meta["tosa_dim_order"]
        if "tosa_spatial_rank" in transpose_node.meta:
            const_node.meta["tosa_spatial_rank"] = transpose_node.meta["tosa_spatial_rank"]

        # Replace all uses of the transpose node with the new constant
        transpose_node.replace_all_uses_with(const_node)

        logger.debug(
            f"FoldConstantTransposePass: Folded transpose {transpose_node.name} "
            f"on constant {input_node.name} with perm={perm}"
        )

        return True

    def call(self, graph_module: torch.fx.GraphModule) -> PassResult:
        modified = False
        nodes_to_remove: Set[torch.fx.Node] = set()
        input_nodes_to_check: Set[torch.fx.Node] = set()

        # Process TOSA transpose nodes
        for node in list(graph_module.graph.nodes):
            if node.target != exir_ops.backend.tosa.TRANSPOSE.default:
                continue

            # Get permutation
            perm = _get_transpose_perm(node)
            if perm is None:
                continue

            # Get input node
            input_nodes = node.all_input_nodes
            if len(input_nodes) == 0:
                continue
            input_node = input_nodes[0]

            # Check if input is a constant tensor
            if not is_param_node(self.exported_program, input_node):
                continue

            # Skip if input node has multiple users (other than this transpose)
            # to avoid duplicating the constant
            if len(input_node.users) > 1:
                logger.debug(
                    f"FoldConstantTransposePass: Skipping {node.name} because input "
                    f"{input_node.name} has multiple users"
                )
                continue

            # Fold the transpose
            if self._fold_transpose(graph_module, node, input_node, perm):
                modified = True
                nodes_to_remove.add(node)
                input_nodes_to_check.add(input_node)

        # Clean up removed transpose nodes
        if modified:
            graph_module.graph.eliminate_dead_code()

            # Try to clean up orphaned input nodes
            for input_node in input_nodes_to_check:
                if len(input_node.users) == 0:
                    try:
                        delete_constant_placeholder(self.exported_program, input_node)
                    except Exception as e:
                        logger.debug(
                            f"FoldConstantTransposePass: Could not delete orphaned "
                            f"placeholder {input_node.name}: {e}"
                        )

            graph_module.recompile()
            graph_module = super().call(graph_module).graph_module

        return PassResult(graph_module, modified)
