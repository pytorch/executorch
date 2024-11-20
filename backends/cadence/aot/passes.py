# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

from typing import Any, cast, Dict, List, Optional, Sequence, Tuple, Type

import torch
import torch.fx
import torch.utils._pytree as pytree
from executorch.backends.cadence.aot.fuse_ops import CadenceFuseOpsInGraph
from executorch.backends.cadence.aot.pass_utils import (
    CadencePassAttribute,
    create_cadence_pass_filter,
    register_cadence_pass,
)
from executorch.backends.cadence.aot.simplify_ops import CadenceSimplifyOpsInGraph
from executorch.backends.cadence.aot.utils import get_edge_overload_packet
from executorch.backends.transforms.remove_clone_ops import RemoveCloneOpsTransform
from executorch.exir.dialects._ops import ops as exir_ops
from executorch.exir.pass_base import ExportPass, NodeMetadata, PassResult, ProxyValue
from executorch.exir.pass_manager import PassManager, PassType
from executorch.exir.passes import dead_code_elimination_pass
from executorch.exir.passes.scalar_to_tensor_pass import ScalarToTensorPass
from executorch.exir.passes.spec_prop_pass import SpecPropPass


@register_cadence_pass(CadencePassAttribute(opt_level=0))
class RemoveCloneOpsTransformImported(ExportPass):
    def call(self, graph_module: torch.fx.GraphModule) -> PassResult:
        finalize_passes: List[PassType] = [
            RemoveCloneOpsTransform(),
        ]
        result = PassManager(passes=finalize_passes)(graph_module)
        dead_code_elimination_pass(result.graph_module)
        return result


@register_cadence_pass(CadencePassAttribute(opt_level=0))
class InitializePipeline(ExportPass):
    """
    Initialize the Jarvis pipeline. This should invariably be the first pass to
    run.
    """

    def call(self, graph_module: torch.fx.GraphModule) -> PassResult:
        dead_code_elimination_pass(graph_module)
        result = SpecPropPass()(graph_module)
        assert result is not None
        return result


@register_cadence_pass(CadencePassAttribute(opt_level=0))
class FinalizePipeline(ExportPass):
    """
    The final cleanup pass after running the Jarvis pipeline.
    """

    def call(self, graph_module: torch.fx.GraphModule) -> PassResult:
        finalize_passes: List[PassType] = [
            ScalarToTensorPass(),
            SpecPropPass(),
        ]
        result = PassManager(passes=finalize_passes)(graph_module)
        dead_code_elimination_pass(result.graph_module)
        return result


# Similar to what's done in executorch/exir/pass_base.py
Argument = Any  # pyre-ignore


@register_cadence_pass(CadencePassAttribute(opt_level=0))
class ReplacePT2QuantWithCadenceQuantPass(ExportPass):
    """
    Replace the pt2 quantization ops with cadence quantization ops.
    We do not link kernels to the PT2 quantization ops, so we need to
    replace them with cadence ops at all optimization levels.
    """

    def call_operator(
        self,
        op,  # pyre-ignore
        args: Tuple[Argument, ...],
        kwargs: Dict[str, Argument],
        meta: NodeMetadata,
    ) -> ProxyValue:
        if op not in {exir_ops.edge.quantized_decomposed.quantize_per_tensor.default}:
            return super().call_operator(op, args, kwargs, meta)

        return super().call_operator(
            exir_ops.edge.cadence.quantize_per_tensor.default,
            args,
            kwargs,
            meta,
        )


@register_cadence_pass(CadencePassAttribute(opt_level=0))
class ReplacePT2DequantWithCadenceDequantPass(ExportPass):
    """
    Replace the pt2 dequantization ops with cadence dequantization ops.
    We do not link kernels to the PT2 quantization ops, so we need to
    replace them with cadence ops at all optimization levels.
    """

    def call_operator(
        self,
        op,  # pyre-ignore
        args: Tuple[Argument, ...],
        kwargs: Dict[str, Argument],
        meta: NodeMetadata,
    ) -> ProxyValue:
        if op not in {exir_ops.edge.quantized_decomposed.dequantize_per_tensor.default}:
            return super().call_operator(op, args, kwargs, meta)

        return super().call_operator(
            exir_ops.edge.cadence.dequantize_per_tensor.default,
            args,
            kwargs,
            meta,
        )


@register_cadence_pass(CadencePassAttribute(opt_level=0))
class ReplaceScalarTensorWithFullPass(ExportPass):
    """
    aten.scalar_tensor can be replaced by aten.full with a shape of [1].
    """

    def call_operator(
        self,
        op,  # pyre-ignore
        args: Tuple[Argument, ...],
        kwargs: Dict[str, Argument],
        meta: NodeMetadata,
    ) -> ProxyValue:
        if op not in {
            exir_ops.edge.aten.scalar_tensor.default,
            torch.ops.aten.scalar_tensor.default,
        }:
            return super().call_operator(op, args, kwargs, meta)

        return super().call_operator(
            exir_ops.edge.aten.full.default,
            (
                [1],
                args[0],
            ),
            {"dtype": torch.float32},
            meta,
        )


@register_cadence_pass(CadencePassAttribute(opt_level=0))
class ReplaceSqueezeAndUnsqueezeWithViewPass(ExportPass):
    """
    When the shape is static, replace squeeze_copy and unsqueeze_copy ops with
    view_copy op
    """

    def call_operator(
        self,
        op,  # pyre-ignore
        args: Tuple[Argument, ...],
        kwargs: Dict[str, Argument],
        meta: NodeMetadata,
    ) -> ProxyValue:
        # Instead of testing EdgeOpOverload, test EdgeOpOverloadPacket,
        # which allows us to cover all overloads.
        if get_edge_overload_packet(op) not in {
            exir_ops.edge.aten.squeeze_copy,
            exir_ops.edge.aten.unsqueeze_copy,
        }:
            return super().call_operator(op, args, kwargs, meta)
        # Get the output tensor shape
        out_shape = meta["val"].shape

        # Bail out if any dim is not an int (dynamic shape)
        for dim in list(out_shape):
            if not isinstance(dim, int):
                return super().call_operator(op, args, kwargs, meta)

        # Return a view op with the new shape
        view_args = (args[0], list(out_shape))
        return super().call_operator(
            exir_ops.edge.aten.view_copy.default, view_args, kwargs, meta
        )


@register_cadence_pass(CadencePassAttribute(opt_level=0))
class RemoveZeroSizedCatArgsPass(ExportPass):
    def call_operator(
        self,
        op,  # pyre-ignore
        args: tuple[Argument, ...],
        kwargs: dict[str, Argument],
        meta: NodeMetadata,
    ) -> ProxyValue:
        if op != exir_ops.edge.aten.cat.default:
            return super().call_operator(op, args, kwargs, meta)

        # Remove any zero-sized tensor arg to form a new args list.
        cat_inputs: list[ProxyValue] = []
        for arg in cast(Sequence[ProxyValue], args[0]):
            if arg.to_tensor().numel() > 0:
                cat_inputs.append(arg)

        # If all the tensors were empty, we just return an empty tensor with
        # the right shape.
        if not cat_inputs:
            empty_shape = meta["val"].shape
            dtype = meta["val"].dtype
            return super().call_operator(
                exir_ops.edge.aten.full.default,
                (tuple(empty_shape), 0),
                {"dtype": dtype},
                meta,
            )

        # If there was only one tensor in the cat_inputs list,
        # we can safely erase this cat op.
        if len(cat_inputs) == 1:
            return cat_inputs[0]

        # Otherwise, we replace args[0] with cat_inputs.
        new_args = list(args)
        new_args[0] = cat_inputs
        return super().call_operator(op, tuple(new_args), kwargs, meta)


@register_cadence_pass(CadencePassAttribute(opt_level=0))
class RemoveNopExpandOpPass(ExportPass):
    """
    For an expand op, if the operator shape matches the expand shape, then the
    expand is a nop.
    """

    def call_operator(
        self,
        op,  # pyre-ignore
        args: tuple[Argument, ...],
        kwargs: dict[str, Argument],
        meta: NodeMetadata,
    ) -> ProxyValue:
        if get_edge_overload_packet(op) not in {
            exir_ops.edge.aten.expand_copy,
            exir_ops.edge.aten.expand,
        }:
            return super().call_operator(op, args, kwargs, meta)

        # Parse the args, and check for nop condition
        arg0 = cast(ProxyValue, args[0])
        arg1 = cast(Sequence[int], args[1])
        in_tensor = arg0.to_tensor()
        if list(in_tensor.shape) == list(arg1):
            return arg0

        return super().call_operator(op, args, kwargs, meta)


@register_cadence_pass(CadencePassAttribute(opt_level=0))
class ReplaceLogicalNotBooleanWhereWithWherePass(ExportPass):
    """
    A where op with a logical_not and a boolean tensor can be replaced
    by a where op with flipped inputs and the initial boolean tensor.
    """

    def replace_logical_nop_where_with_where(
        self, graph_module: torch.fx.GraphModule
    ) -> None:
        graph = graph_module.graph
        for node in graph.nodes:
            # We are only interested in where nodes
            if node.target != exir_ops.edge.aten.where.self:
                continue

            # If the third arg is not a logical_not, bail.
            if node.args[0].target != exir_ops.edge.aten.logical_not.default:
                continue

            # Get the third arg node and its input
            logical_not_node = node.args[0]
            logical_not_input_tensor = (
                logical_not_node.args[0].to_tensor()
                if isinstance(logical_not_node.args[0], ProxyValue)
                else logical_not_node.args[0]
            )

            # If the logical_not input is not a boolean tensor, bail.
            if logical_not_input_tensor.meta["spec"].dtype != torch.bool:
                continue

            # Replace the where op with another one, flipping the inputs and using the boolean
            # tensor from logical_not.
            with graph.inserting_before(node):
                linear_node = graph.call_function(
                    exir_ops.edge.aten.where.self,
                    args=(logical_not_node.args[0], node.args[2], node.args[1]),
                )
            # Replace all the uses
            node.replace_all_uses_with(linear_node)

        graph_module.recompile()
        graph_module.graph.eliminate_dead_code()

    def call(self, graph_module: torch.fx.GraphModule) -> PassResult:
        self.replace_logical_nop_where_with_where(graph_module)
        result = super().call(graph_module)
        return result


@register_cadence_pass(CadencePassAttribute(opt_level=0))
class ReplaceSafeSoftmaxWithSoftmax(ExportPass):  # keep
    """
    Replace _safe_softmax with _softmax
    """

    def call_operator(
        self,
        op,  # pyre-ignore
        args: tuple[Argument, ...],
        kwargs: dict[str, Argument],
        meta: NodeMetadata,
    ) -> ProxyValue:
        if op != torch.ops.aten._safe_softmax.default:
            return super().call_operator(op, args, kwargs, meta)

        # Add False for the half_to_float argument of softmax
        softmax_args = list(args) + [False]

        return super().call_operator(
            torch.ops.aten._softmax.default,
            tuple(softmax_args),
            kwargs,
            meta,
        )


def get_passes_in_default_order() -> List[Type[PassType]]:
    passes = [
        InitializePipeline,
        RemoveZeroSizedCatArgsPass,
        ReplaceLogicalNotBooleanWhereWithWherePass,
        ReplaceScalarTensorWithFullPass,
        RemoveCloneOpsTransformImported,
        RemoveNopExpandOpPass,
        CadenceFuseOpsInGraph.passes,
        ReplaceSqueezeAndUnsqueezeWithViewPass,
        ReplacePT2QuantWithCadenceQuantPass,
        ReplacePT2DequantWithCadenceDequantPass,
        CadenceSimplifyOpsInGraph.passes,
        # TODO: add the rest of the passes here.
        # InitializePipeline,
        # RemoveRedundantOps.passes,
        # ReorderOpsInGraph.passes,
        # RemoveJarvisNops.passes,
        # CadenceFuseOpsInGraph.passes,
        # ReplaceOpsInGraph.passes,
        # SimplifyOpsInGraph.passes,
        # FinalizePipeline,
        # FuseFullThenReshapePass,
        # FuseTransposeOpPairsPass,
        # RemoveNopSliceOrViewOpPass,
    ]
    return pytree.tree_flatten(passes)[0]


def get_cadence_passes(
    opt_level: int,
) -> List[Optional[PassResult]]:
    passes = get_passes_in_default_order()
    pass_filter = create_cadence_pass_filter(opt_level)
    filtered_passes = [
        # pyre-fixme[20]: Call `torch.fx.passes.infra.pass_base.PassBase.__call__` expects argument `graph_module`.
        filtered_pass()
        # pyre-fixme[6]: In call `filter.__new__` ... got `List[Type[typing.Callable[[GraphModule], Optional[PassResult]]]]`.
        for filtered_pass in list(filter(pass_filter, passes))
    ]
    return filtered_passes
