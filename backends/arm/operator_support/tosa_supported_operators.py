# Copyright 2024-2025 Arm Limited and/or its affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-unsafe

import itertools
import operator
import typing
from typing import final, Optional, Sequence, Type

import torch
import torch.fx as fx

from executorch.backends.arm._passes.arm_pass_utils import get_first_fake_tensor
from executorch.backends.arm._passes.fuse_quantized_activation_pass import (
    FuseQuantizedActivationPass,
)
from executorch.backends.arm.tosa_specification import Tosa_0_80, TosaSpecification
from executorch.exir.dialects._ops import ops as exir_ops
from torch.fx.passes.operator_support import any_chain, chain, OperatorSupportBase
from torch.fx.passes.utils.source_matcher_utils import get_source_partitions


class SupportedTOSAOperatorCheck(OperatorSupportBase):
    """
    Supported OP for TOSA lowering
    """

    def __init__(self, tosa_spec: TosaSpecification):
        self.tosa_spec = tosa_spec

    # Should be populated by subclass implementation
    tosa_specs: list[TosaSpecification] = []
    targets: list[str] = []

    @final
    def is_node_supported(
        self, submodules: typing.Mapping[str, torch.nn.Module], node: fx.Node
    ) -> bool:
        if node.target not in self.targets:
            return False
        return self.is_node_tosa_supported(node, self.tosa_spec)

    def is_node_tosa_supported(
        self, node: fx.Node, tosa_spec: TosaSpecification
    ) -> bool:
        """
        Checks if the fx.Node node is lowerable using the TOSA specification defined by tosa_spec.
        """
        raise NotImplementedError("SupportedTOSAOperatorCheck must be extended.")


# container for all SupportedTosaOperatorCheck classes
_tosa_spec_support: dict[TosaSpecification, list[Type[SupportedTOSAOperatorCheck]]] = {
    TosaSpecification.create_from_string("TOSA-0.80+BI"): [],
    TosaSpecification.create_from_string("TOSA-0.80+MI"): [],
}


def register_tosa_support_check(checker: Type[SupportedTOSAOperatorCheck]):
    """
    Decorator to mark a subclass implmentation of SupportedTosaOperatorCheck
    to be registered for checking if a torch.fx.Node is lowerable given
    a TOSA specification.
    """
    for tosa_spec in checker.tosa_specs:
        _tosa_spec_support[tosa_spec].append(checker)
    return checker


def get_registered_tosa_support_checks(
    tosa_spec: TosaSpecification,
) -> list[Type[SupportedTOSAOperatorCheck]]:

    if tosa_spec not in _tosa_spec_support:
        raise RuntimeError(
            f"TOSA specification not valid: {tosa_spec} not in {list(_tosa_spec_support.keys())}"
        )

    return _tosa_spec_support[tosa_spec]


def tosa_support_factory(
    tosa_spec: TosaSpecification,
    additional_checks: Optional[Sequence[OperatorSupportBase]] = None,
) -> OperatorSupportBase:
    negative_checks: list[OperatorSupportBase] = []
    if not tosa_spec.support_float():
        negative_checks.append(NeedsDecompositionCheck())
        negative_checks.append(CheckProperQuantization())
        negative_checks.append(EthosU55NotSupported(tosa_spec))
    return chain(
        any_chain(
            BaseTOSASupportList(),
            *(
                check(tosa_spec)
                for check in get_registered_tosa_support_checks(tosa_spec)
            ),
        ),
        *negative_checks,
        *additional_checks if additional_checks else [],
    )


class BaseTOSASupportList(OperatorSupportBase):

    def is_node_supported(
        self, submodules: typing.Mapping[str, torch.nn.Module], node: fx.Node
    ) -> bool:
        supported = node.op == "call_function" and node.target in [
            exir_ops.edge.aten.abs.default,
            exir_ops.edge.aten.add.Tensor,
            exir_ops.edge.aten.logical_and.default,
            exir_ops.edge.aten.logical_or.default,
            exir_ops.edge.aten.logical_xor.default,
            exir_ops.edge.aten.logical_not.default,
            exir_ops.edge.aten.bitwise_and.Tensor,
            exir_ops.edge.aten.bitwise_or.Tensor,
            exir_ops.edge.aten.bitwise_xor.Tensor,
            exir_ops.edge.aten.expand_copy.default,
            exir_ops.edge.aten.cat.default,
            exir_ops.edge.aten.clamp.default,
            exir_ops.edge.aten.bmm.default,
            exir_ops.edge.aten.permute_copy.default,
            exir_ops.edge.aten.hardsigmoid.default,
            exir_ops.edge.aten.hardtanh.default,
            exir_ops.edge.aten.hardswish.default,
            exir_ops.edge.aten.div.Tensor,
            exir_ops.edge.aten.eq.Tensor,
            exir_ops.edge.aten.exp.default,
            exir_ops.edge.aten.log.default,
            exir_ops.edge.aten.linear.default,
            exir_ops.edge.aten.split_with_sizes_copy.default,
            exir_ops.edge.aten.floor.default,
            exir_ops.edge.aten.full.default,
            exir_ops.edge.aten.full_like.default,
            exir_ops.edge.aten.ge.Tensor,
            exir_ops.edge.aten.gt.Tensor,
            exir_ops.edge.aten.le.Tensor,
            exir_ops.edge.aten.lt.Tensor,
            exir_ops.edge.aten.mul.Tensor,
            exir_ops.edge.aten.add.Scalar,
            exir_ops.edge.aten.sub.Scalar,
            exir_ops.edge.aten.mul.Scalar,
            exir_ops.edge.aten.div.Scalar,
            exir_ops.edge.aten._native_batch_norm_legit_no_training.default,
            exir_ops.edge.aten.native_layer_norm.default,
            exir_ops.edge.aten.sigmoid.default,
            exir_ops.edge.aten.mean.dim,
            exir_ops.edge.aten.mm.default,
            exir_ops.edge.aten.minimum.default,
            exir_ops.edge.aten.maximum.default,
            exir_ops.edge.aten.repeat.default,
            exir_ops.edge.aten.reciprocal.default,
            exir_ops.edge.aten.relu.default,
            exir_ops.edge.aten.rsqrt.default,
            exir_ops.edge.aten._softmax.default,
            exir_ops.edge.aten.select_copy.int,
            exir_ops.edge.aten._log_softmax.default,
            exir_ops.edge.aten.slice_copy.Tensor,
            exir_ops.edge.aten.sub.Tensor,
            exir_ops.edge.aten.tanh.default,
            exir_ops.edge.aten.upsample_nearest2d.vec,
            exir_ops.edge.aten.var.correction,
            exir_ops.edge.aten.var.dim,
            exir_ops.edge.aten.view_copy.default,
            exir_ops.edge.aten.clone.default,
            exir_ops.edge.aten.unsqueeze_copy.default,
            exir_ops.edge.aten.squeeze_copy.dims,
            operator.getitem,
            exir_ops.edge.quantized_decomposed.quantize_per_tensor.default,
            exir_ops.edge.quantized_decomposed.dequantize_per_tensor.default,
            exir_ops.edge.aten.constant_pad_nd.default,
            exir_ops.edge.aten.amax.default,
            exir_ops.edge.aten.amin.default,
        ]

        return supported


class EthosU55NotSupported(OperatorSupportBase):
    """
    Certain operators are not supported on U55. These are listed in `unsupported` in
    is_node_supported().
    """

    def __init__(self, tosa_spec: TosaSpecification):
        self.tosa_spec = tosa_spec

    def is_node_supported(
        self, submodules: typing.Mapping[str, torch.nn.Module], node: fx.Node
    ) -> bool:
        if isinstance(self.tosa_spec, Tosa_0_80) and self.tosa_spec.is_U55_subset:
            unsupported_ops = [
                exir_ops.edge.aten.bitwise_and.Tensor,
                exir_ops.edge.aten.bitwise_or.Tensor,
                exir_ops.edge.aten.bitwise_xor.Tensor,
                exir_ops.edge.aten.logical_and.default,
                exir_ops.edge.aten.logical_or.default,
                exir_ops.edge.aten.logical_xor.default,
                exir_ops.edge.aten.logical_not.default,
                exir_ops.edge.aten.amax.default,
                exir_ops.edge.aten.amin.default,
                exir_ops.edge.aten.eq.Tensor,
                exir_ops.edge.aten.ge.Tensor,
                exir_ops.edge.aten.gt.Tensor,
                exir_ops.edge.aten.le.Tensor,
                exir_ops.edge.aten.lt.Tensor,
            ]

            if node.target in unsupported_ops:
                return False

        return True


class NeedsDecompositionCheck(OperatorSupportBase):
    """
    Targeted operators need to be decomposed prior to quantization in order to get a pair of q-dq-nodes surrounding
    the operator, and to get optimal quantization parameters for each operator. This check will reject operators
    that need to be decomposed.
    """

    def is_node_supported(
        self, submodules: typing.Mapping[str, torch.nn.Module], node: fx.Node
    ) -> bool:

        if node.op != "call_function":
            return True
        if node.target == exir_ops.edge.aten.mean.dim:
            dim = node.args[1]
            return dim == [-1, -2]
        needs_decomp = node.target in [
            exir_ops.edge.aten.div.Tensor,
            exir_ops.edge.aten._native_batch_norm_legit_no_training.default,
            exir_ops.edge.aten.native_layer_norm.default,
            exir_ops.edge.aten.mean.dim,
            exir_ops.edge.aten._softmax.default,
            exir_ops.edge.aten._log_softmax.default,
            exir_ops.edge.aten.var.correction,
            exir_ops.edge.aten.var.dim,
        ]
        return not needs_decomp


class CheckProperQuantization(OperatorSupportBase):
    """
    For targeted nodes, check that it has been quantized as expected. In most cases this means that a pair of quantize
    and dequantize nodes surrounds the node. This is neccessary for table operators and operators that need to rescale
    activations.
    """

    dq_op = exir_ops.edge.quantized_decomposed.dequantize_per_tensor.default
    q_op = exir_ops.edge.quantized_decomposed.quantize_per_tensor.default

    def _is_matmul_node_supported(
        self, submodules: typing.Mapping[str, torch.nn.Module], node: fx.Node
    ):
        """
        Find the matmul source partition containing this node and check that all its inputs and outputs are quantized.
        """
        for graph_module in submodules.values():
            graph_module = typing.cast(fx.GraphModule, graph_module)
            matmul_partitions = get_source_partitions(
                graph_module.graph,
                [
                    torch.matmul,
                ],
                None,
            )
            matmul_partitions = list(
                itertools.chain.from_iterable(matmul_partitions.values())
            )
            matched_partition = None
            for partition in matmul_partitions:
                if node in partition.nodes:
                    matched_partition = partition
            if matched_partition is not None:
                input_quantized = all(
                    input_node.target == self.dq_op
                    for input_node in matched_partition.input_nodes
                )
                if not input_quantized:
                    return False
                output_quantized = all(
                    output_node_user.target == self.q_op
                    for output_node_user in matched_partition.output_nodes[0].users
                )
                if not output_quantized:
                    return False
            else:
                return False

        return True

    def is_node_supported(
        self, submodules: typing.Mapping[str, torch.nn.Module], node: fx.Node
    ) -> bool:
        output_quantized = False
        input_quantized = False
        if node.target not in (
            exir_ops.edge.aten.add.Tensor,
            exir_ops.edge.aten.avg_pool2d.default,
            exir_ops.edge.aten.bmm.default,
            exir_ops.edge.aten.convolution.default,
            exir_ops.edge.aten.exp.default,
            exir_ops.edge.aten.hardtanh.default,
            exir_ops.edge.aten.linear.default,
            exir_ops.edge.aten.log.default,
            exir_ops.edge.aten.max_pool2d_with_indices.default,
            exir_ops.edge.aten.mm.default,
            exir_ops.edge.aten.mul.Tensor,
            exir_ops.edge.aten.reciprocal.default,
            exir_ops.edge.aten.relu.default,
            exir_ops.edge.aten.rsqrt.default,
            exir_ops.edge.aten.sigmoid.default,
            exir_ops.edge.aten.sub.Tensor,
            exir_ops.edge.aten.tanh.default,
            exir_ops.edge.aten.upsample_nearest2d.vec,
        ):
            return True
        elif node.target in (
            exir_ops.edge.aten.bmm.default,
            exir_ops.edge.aten.mm.default,
        ):
            source_fn_stack: tuple[typing.Any] = node.meta.get("source_fn_stack", [])
            if len(source_fn_stack) > 0:
                if source_fn_stack[-1][1] in (torch.matmul,):
                    return self._is_matmul_node_supported(submodules, node)

        elif node.target in (exir_ops.edge.aten.max_pool2d_with_indices.default,):
            users = node.users
            output_quantized = all(
                user.target == operator.getitem
                and all(user_user.target == self.q_op for user_user in user.users)
                for user in users
            )
        elif FuseQuantizedActivationPass._is_fuseable_input(node):
            users = node.users
            output_quantized = all(
                FuseQuantizedActivationPass._is_fuseable_quantized_activation(user)
                for user in users
            )
        elif FuseQuantizedActivationPass._is_fuseable_quantized_activation(node):
            input_node = node.all_input_nodes[0]
            input_quantized = FuseQuantizedActivationPass._is_fuseable_input(input_node)

        input_quantized = input_quantized or all(
            (input_node.target == self.dq_op)
            or (not get_first_fake_tensor(input_node).dtype.is_floating_point)
            for input_node in node.all_input_nodes
        )

        if not input_quantized:
            return False

        all_q_users = all(
            (output_node.target == self.q_op) for output_node in node.users
        )
        is_floating_point = get_first_fake_tensor(node).dtype.is_floating_point
        output_quantized = output_quantized or all_q_users or not is_floating_point

        if not output_quantized:
            return False
        return True
