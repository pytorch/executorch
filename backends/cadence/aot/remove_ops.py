# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

from typing import cast, List, Optional, Set, Type

# Import these for the cadence function signatures.
import executorch.backends.cadence.aot.ops_registrations  # noqa: F401
import torch
import torch.fx
from executorch.backends.cadence.aot.fuse_ops import FuseTransposeOrPermuteOpPairsPass
from executorch.backends.cadence.aot.pass_utils import (
    get_arg,
    RemoveOrReplacePassInterface,
    set_arg,
)
from executorch.backends.cadence.aot.simplify_ops import SimplifySliceOpPass
from executorch.backends.transforms.remove_alias_copy_op import (
    RemoveAliasCopyOpPass as _SharedRemoveAliasCopyOpPass,
)
from executorch.backends.transforms.remove_branched_quant_dequant import (
    RemoveBranchedQuantDequant as _SharedRemoveBranchedQuantDequant,
)
from executorch.backends.transforms.remove_cat_from_slice_copy import (
    RemoveCatFromSliceCopyPass as _SharedRemoveCatFromSliceCopyPass,
)
from executorch.backends.transforms.remove_clone_ops_transform_imported import (
    RemoveCloneOpsTransformImported as _SharedRemoveCloneOpsTransformImported,
)
from executorch.backends.transforms.remove_nop_as_strided_copy_op import (
    RemoveNopAsStridedCopyOpPass as _SharedRemoveNopAsStridedCopyOpPass,
)
from executorch.backends.transforms.remove_nop_expand_op import (
    RemoveNopExpandOpPass as _SharedRemoveNopExpandOpPass,
)
from executorch.backends.transforms.remove_nop_slice_or_view_op import (
    RemoveNopSliceOrViewOpPass as _SharedRemoveNopSliceOrViewOpPass,
)
from executorch.backends.transforms.remove_permute_before_mean import (
    RemovePermuteBeforeMeanPass as _SharedRemovePermuteBeforeMeanPass,
)
from executorch.backends.transforms.remove_permutes_around_elementwise_ops import (
    RemovePermutesAroundElementwiseOps as _SharedRemovePermutesAroundElementwiseOps,
)
from executorch.backends.transforms.remove_zero_sized_cat_args import (
    RemoveZeroSizedCatArgsPass as _SharedRemoveZeroSizedCatArgsPass,
)
from executorch.backends.transforms.replace_squeeze_unsqueeze_with_view import (
    ReplaceSqueezeAndUnsqueezeWithViewPass as _SharedReplaceSqueezeAndUnsqueezeWithViewPass,
)
from executorch.exir.dialects._ops import ops as exir_ops
from executorch.exir.dialects.edge._ops import EdgeOpOverload, EdgeOpOverloadPacket
from executorch.exir.pass_base import (
    ExportedProgramPassBase,
    ExportedProgramPassResult,
    PassResult,
)
from torch.export import ExportedProgram
from torch.export.graph_signature import InputKind, OutputKind
from torch.fx.node import Node
from torch.fx.passes.infra.pass_base import PassBase
from torch.utils import _pytree as pytree


class RemoveCloneOpsTransformImported(_SharedRemoveCloneOpsTransformImported):
    pass


class RemoveDetachCopyPass(RemoveOrReplacePassInterface):
    @property
    def targets(self) -> list[EdgeOpOverload]:
        return [exir_ops.edge.aten.detach_copy.default]

    def maybe_remove_or_replace(self, node: Node) -> bool:
        input_node = node.args[0]
        assert isinstance(input_node, Node)
        node.replace_all_uses_with(input_node)
        return True


# The following class consolidates passes to remove ops that are redundant:
# either by the virtue of the operation they perform, or redundant in the
# context of inference.
class RemoveRedundantOps:
    passes = [
        RemoveDetachCopyPass,
    ]


class RemoveZeroSizedCatArgsPass(_SharedRemoveZeroSizedCatArgsPass):
    pass


class RemoveNopExpandOpPass(_SharedRemoveNopExpandOpPass):
    pass


class RemoveToOpsPass(RemoveOrReplacePassInterface):
    # aten.to.* as of now are all nops
    @property
    def targets(self) -> list[EdgeOpOverload]:
        return [
            exir_ops.edge.aten.to.dtype,
            exir_ops.edge.aten.to.dtype_layout,
        ]

    def maybe_remove_or_replace(self, node: Node) -> bool:
        input_node = node.args[0]
        assert isinstance(input_node, Node)
        node.replace_all_uses_with(input_node)
        return True


class RemoveZeroSizedConstantPadNd(RemoveOrReplacePassInterface):
    @property
    def targets(self) -> list[EdgeOpOverload]:
        return [exir_ops.edge.aten.constant_pad_nd.default]

    def maybe_remove_or_replace(self, node: Node) -> bool:
        # Get padding argument (second argument)
        if len(node.args) < 2:
            return False

        padding = node.args[1]
        if not isinstance(padding, (list, tuple)):
            return False

        # If any padding value is non-zero, keep the node
        if any(x != 0 for x in padding):
            return False

        # All padding is zero, replace with input
        input_node = node.args[0]
        assert isinstance(input_node, Node)
        node.replace_all_uses_with(input_node)
        return True


class RemoveNopSliceOrViewOpPass(_SharedRemoveNopSliceOrViewOpPass):
    pass


class RemoveNopAsStridedCopyOpPass(_SharedRemoveNopAsStridedCopyOpPass):
    pass


class RemoveNopLinalgVectorNormOpPass(RemoveOrReplacePassInterface):
    """
    If the norm is applied over a dimension that is size 1, it can be eliminated.
    """

    @property
    def targets(self) -> list[EdgeOpOverload]:
        return [exir_ops.edge.aten.linalg_vector_norm.default]

    def maybe_remove_or_replace(self, node: Node) -> bool:
        # If the op has three args or less, it can't be a nop
        if len(node.args) <= 3:
            return False
        # If dim is None, or keepdim is False, it is not a nop
        dim = cast(Optional[tuple[int, ...]], node.args[2])
        keepdim = cast(bool, node.args[3])
        if dim is None or not keepdim:
            return False

        # If the norm has 4 args and keepdim is True, check if dim is not None
        # and if the dimensions in dim are size 1. If not, the norm is not a nop.
        input_node = node.args[0]
        assert isinstance(input_node, Node)
        shape = input_node.meta["val"].shape
        if len(node.args) < 4:
            for d in dim:
                if shape[d] != 1:
                    return False

        node.replace_all_uses_with(input_node)
        return True


class RemoveContiguousOpPass(RemoveOrReplacePassInterface):
    """
    This is based on the assumption that all tensors are contiguous in ExecuTorch
    and after cadence passes, and we should revisit this if that assumption is no longer true.
    This causes the model to not be runnable with the arguments given to the
    original graph module.
    """

    @property
    def targets(self) -> list[EdgeOpOverload]:
        return [exir_ops.edge.aten.contiguous.default]

    def maybe_remove_or_replace(self, node: Node) -> bool:
        input_node = node.args[0]
        assert isinstance(input_node, Node)
        node.replace_all_uses_with(input_node)
        return True


class RemoveAliasCopyOpPass(_SharedRemoveAliasCopyOpPass):
    pass


class RemoveNopRequantizeOpPass(RemoveOrReplacePassInterface):
    """
    For a requantize op, if the following three conditions are satisfied:
    1. the in_scale matches the out_scale
    2. the in_zero_point matches the out_zero_point
    3. the dtypes of the input and output tensors are the same
    then the requantize op is redundant, and can be eliminated
    """

    @property
    def targets(self) -> list[EdgeOpOverload]:
        return [exir_ops.edge.cadence.requantize.per_tensor]

    def maybe_remove_or_replace(self, node: Node) -> bool:
        input_node = node.args[0]
        assert isinstance(input_node, Node)
        in_scale = node.args[1]
        in_zero_point = node.args[2]
        out_scale = node.args[3]
        out_zero_point = node.args[4]
        out_dtype = node.args[5]
        in_dtype = input_node.meta["val"].dtype
        # Check the three conditions
        if (
            in_scale == out_scale
            and in_zero_point == out_zero_point
            and in_dtype == out_dtype
        ):
            node.replace_all_uses_with(input_node)
            return True
        return False


class RemoveNopMulOpPass(RemoveOrReplacePassInterface):
    """
    If a mul op is multiplying two tensors with the same shape and one
    of those tensors is all zeros, return the zero tensor instead.
    """

    @property
    def targets(self) -> list[EdgeOpOverload]:
        return [exir_ops.edge.aten.mul.Tensor]

    def maybe_remove_or_replace(self, node: Node) -> bool:
        input1 = node.args[0]
        input2 = node.args[1]
        assert isinstance(input1, Node)
        assert isinstance(input2, Node)

        # Check if both inputs have the same shape
        if input1.meta["val"].shape != input2.meta["val"].shape:
            return False

        # Check if one of the inputs is a zero tensor
        if input1.target == exir_ops.edge.aten.full.default:
            if input1.args[1] == 0:
                node.replace_all_uses_with(input1)
                return True
        elif input2.target == exir_ops.edge.aten.full.default:
            if input2.args[1] == 0:
                node.replace_all_uses_with(input2)
                return True

        return False


class RemoveNopAddOpPass(RemoveOrReplacePassInterface):
    """
    If an add op is adding two tensors with the same shape and one
    of those tensors is all zeros, return the other tensor instead.
    """

    @property
    def targets(self) -> list[EdgeOpOverload]:
        return [exir_ops.edge.aten.add.Tensor]

    def maybe_remove_or_replace(self, node: Node) -> bool:
        input1 = node.args[0]
        input2 = node.args[1]
        assert isinstance(input1, Node)
        assert isinstance(input2, Node)

        # Check if both inputs have the same shape
        if input1.meta["val"].shape != input2.meta["val"].shape:
            return False

        # Check if one of the inputs is a zero tensor
        if input1.target == exir_ops.edge.aten.full.default:
            if input1.args[1] == 0:
                node.replace_all_uses_with(input2)
                return True
        elif input2.target == exir_ops.edge.aten.full.default:
            if input2.args[1] == 0:
                node.replace_all_uses_with(input1)
                return True

        return False


class RemovePermuteBeforeMeanPass(_SharedRemovePermuteBeforeMeanPass):
    _UNARY_TARGETS: frozenset[EdgeOpOverload] = (
        _SharedRemovePermuteBeforeMeanPass._UNARY_TARGETS
        | {
            exir_ops.edge.cadence.dequantize_per_tensor.default,
            exir_ops.edge.cadence.quantize_per_tensor.default,
        }
    )


class ReplaceSqueezeAndUnsqueezeWithViewPassImported(
    _SharedReplaceSqueezeAndUnsqueezeWithViewPass
):
    pass


class RemovePermutesAroundElementwiseOps(_SharedRemovePermutesAroundElementwiseOps):
    def __init__(self) -> None:
        super().__init__(
            extra_permutable_ops={
                exir_ops.edge.cadence.quantize_per_tensor.default,
                exir_ops.edge.cadence.dequantize_per_tensor.default,
                exir_ops.edge.cadence.quantized_relu.per_tensor,
                exir_ops.edge.cadence.requantize.per_tensor,
                exir_ops.edge.cadence.quantized_add.per_tensor,
            }
        )


class RemoveSqueezeViewBeforeElementwiseOps(PassBase):
    """
    Looks for subgraphs of the form:
    squeeze -> [elementwise ops] -> view
    and removes the squeeze node by reshaping the intermediate ops. If the final view
    is a corresponding unsqueeze it should also get eliminated by noop view elimination
    later. Only handles simple chain of intermediates now.

    The pass works on view ops instead of squeeze directly, thus it should be run after
    the squeeze/unsqueeze->view lowering.
    """

    intermediate_ops: set[EdgeOpOverload] = {
        exir_ops.edge.quantized_decomposed.quantize_per_tensor.default,
        exir_ops.edge.quantized_decomposed.dequantize_per_tensor.default,
        exir_ops.edge.cadence.quantize_per_tensor.default,
        exir_ops.edge.cadence.dequantize_per_tensor.default,
        # Ops that require special handling:
        exir_ops.edge.aten.slice_copy.Tensor,
    }

    def get_squeeze_indices(self, view_node: Node) -> List[int]:
        """
        Returns the indices of the input dimensions that are squeezed in the output if
        view node is a squeeze. Returns an empty list otherwise.
        """
        input_node = get_arg(view_node, "input", Node)
        input_shape = input_node.meta["val"].shape
        output_shape = view_node.meta["val"].shape

        if len(input_shape) <= len(output_shape):
            return []

        squeeze_indices = []
        out_idx = 0
        for idx, dim in enumerate(input_shape):
            if out_idx >= len(output_shape):
                return []
            if dim == output_shape[out_idx]:
                out_idx += 1
            else:
                # If there's a mismatch between the input and output dimensions, input
                # dimension has to be 1.
                if dim == 1:
                    squeeze_indices.append(idx)
                else:
                    return []

        # Check if all the output dimensions are consumed.
        if out_idx != len(output_shape):
            return []

        return squeeze_indices

    def recompute_meta(self, nodes: List[Node]) -> None:
        for node in nodes:
            args, kwargs = pytree.tree_map_only(
                Node,
                lambda arg: arg.meta["val"],
                (node.args, node.kwargs),
            )
            assert callable(node.target)
            node.meta["val"] = node.target(*args, **kwargs)
            node.meta["tensor_meta"] = None

    def handle_squeeze(self, view_node: Node, visited_view_nodes: Set[Node]) -> bool:
        if view_node in visited_view_nodes:
            return False

        squeeze_indices = self.get_squeeze_indices(view_node)
        if not squeeze_indices:
            return False

        # Only handle simple chains for now.
        if len(view_node.users) != 1:
            return False
        node = next(iter(view_node.users))

        # Traverse down from the node until finding another view op.
        intermediate_nodes = []
        intermediate_slices = []
        while node.target != exir_ops.edge.aten.view_copy.default:
            # Only handle simple chains for now
            if len(node.users) != 1:
                return False
            if node.target not in self.intermediate_ops:
                return False
            intermediate_nodes.append(node)
            if node.target == exir_ops.edge.aten.slice_copy.Tensor:
                intermediate_slices.append(node)
            node = next(iter(node.users))

        # View node found. We can't optimize this view_node again since the
        # input shape is invalid now so add it to the visited set.
        visited_view_nodes.add(node)

        # Update the intermediate slices.
        for slice_node in intermediate_slices:
            slice_rank = len(slice_node.meta["val"].shape)
            slice_dim = get_arg(slice_node, "dim", int)
            if slice_dim < 0:
                slice_dim += slice_rank
            for squeeze_dim in squeeze_indices:
                if slice_dim >= squeeze_dim:
                    slice_dim += 1
            set_arg(slice_node, "dim", slice_dim)

        # Skip the initial view node.
        input_node = get_arg(view_node, "input", Node)
        view_node.replace_all_uses_with(input_node)
        self.recompute_meta(intermediate_nodes)
        return True

    def call(self, graph_module: torch.fx.GraphModule) -> PassResult:
        visited_view_nodes = set()
        modified = False
        for view_node in graph_module.graph.find_nodes(
            op="call_function", target=exir_ops.edge.aten.view_copy.default, sort=True
        ):
            modified |= self.handle_squeeze(view_node, visited_view_nodes)

        if modified:
            graph_module.graph.eliminate_dead_code()
            graph_module.recompile()
            return PassResult(graph_module, True)

        return PassResult(graph_module, False)


class RemoveBranchedQuantDequant(_SharedRemoveBranchedQuantDequant):
    quantize_op_packets: set[EdgeOpOverloadPacket] = {
        exir_ops.edge.cadence.quantize_per_tensor,
        exir_ops.edge.quantized_decomposed.quantize_per_tensor,
    }
    dequantize_op_packets: set[EdgeOpOverloadPacket] = {
        exir_ops.edge.cadence.dequantize_per_tensor,
        exir_ops.edge.quantized_decomposed.dequantize_per_tensor,
    }


class RemoveCatFromSliceCopyPass(_SharedRemoveCatFromSliceCopyPass):
    pass


class CommonRemovePasses:
    passes: List[Type[PassBase]] = [
        # Canonicalise squeeze/unsqueeze to view_copy first: the nop-view and
        # permute passes below both reason about view_copy only.
        ReplaceSqueezeAndUnsqueezeWithViewPassImported,
        RemoveAliasCopyOpPass,
        RemoveNopExpandOpPass,
        RemoveNopSliceOrViewOpPass,
        RemoveNopAsStridedCopyOpPass,
        RemoveToOpsPass,
        RemoveZeroSizedCatArgsPass,
        RemovePermuteBeforeMeanPass,
        RemovePermutesAroundElementwiseOps,
        FuseTransposeOrPermuteOpPairsPass,
        RemoveSqueezeViewBeforeElementwiseOps,
        RemoveCatFromSliceCopyPass,
        RemoveCloneOpsTransformImported,
    ]


class RemoveBNTrackingMutationsPass(ExportedProgramPassBase):
    """Remove num_batches_tracked buffer mutations from an ExportedProgram.

    run_decompositions() re-introduces num_batches_tracked mutable buffer
    outputs even when batch_norm uses training=False. These mutations are
    dead (the counter is never read in eval mode) but inflate the PTE.

    Removes both the mutation outputs AND the dead input placeholders,
    along with their corresponding graph signature entries and state dict
    tensors.
    """

    def call(self, exported_program: ExportedProgram) -> ExportedProgramPassResult:
        ep = exported_program
        nbt_fqns = {
            fqn
            for fqn in ep.graph_signature.buffers_to_mutate.values()
            if "num_batches_tracked" in fqn
        }
        if not nbt_fqns:
            return ExportedProgramPassResult(ep, False)

        nbt_output_names = {
            name
            for name, fqn in ep.graph_signature.buffers_to_mutate.items()
            if fqn in nbt_fqns
        }
        # buffers_to_mutate / inputs_to_buffers are keyed by the FX node name
        # (arg.name), which can differ from node.target when export
        # uniquifies or sanitizes placeholder names. Match on node.name.
        nbt_input_names = {
            name
            for name, fqn in ep.graph_signature.inputs_to_buffers.items()
            if fqn in nbt_fqns
        }

        gm = ep.graph_module

        # Remove mutation outputs
        output_node = gm.graph.output_node()
        output_args = list(output_node.args[0])
        for idx in sorted(
            (
                i
                for i, n in enumerate(output_args)
                if isinstance(n, torch.fx.Node) and n.name in nbt_output_names
            ),
            reverse=True,
        ):
            output_args.pop(idx)
        output_node.args = (tuple(output_args),)

        gm.graph.eliminate_dead_code()

        removed_nbt_fqns: Set[str] = set()

        # Remove dead input placeholders
        for node in list(gm.graph.nodes):
            if (
                node.op == "placeholder"
                and node.name in nbt_input_names
                and len(node.users) == 0
            ):
                removed_nbt_fqns.add(ep.graph_signature.inputs_to_buffers[node.name])
                gm.graph.erase_node(node)

        gm.recompile()

        # Update output specs
        ep.graph_signature.output_specs = [
            s
            for s in ep.graph_signature.output_specs
            if not (
                s.kind == OutputKind.BUFFER_MUTATION
                and s.target is not None
                and s.target in nbt_fqns
            )
        ]

        ep.graph_signature.input_specs = [
            s
            for s in ep.graph_signature.input_specs
            if not (
                s.kind == InputKind.BUFFER
                and s.target is not None
                and s.target in removed_nbt_fqns
            )
        ]

        # Remove state for buffers whose placeholders were removed.
        for fqn in removed_nbt_fqns:
            ep.state_dict.pop(fqn, None)
            ep.constants.pop(fqn, None)

        return ExportedProgramPassResult(ep, True)


class CadenceRemoveNops:
    passes: List[Type[PassBase]] = CommonRemovePasses.passes + [
        SimplifySliceOpPass,
        RemoveNopRequantizeOpPass,
        RemoveZeroSizedConstantPadNd,
        RemoveContiguousOpPass,
        RemoveNopMulOpPass,
        RemoveNopAddOpPass,
        RemoveNopLinalgVectorNormOpPass,
        RemoveBranchedQuantDequant,
    ]
