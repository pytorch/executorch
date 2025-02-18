# Copyright 2024-2025 Arm Limited and/or its affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-unsafe

import operator
from typing import final, Optional, Sequence, Type

import torch.fx as fx
from executorch.backends.arm.tosa_specification import TosaSpecification
from executorch.exir.dialects._ops import ops as exir_ops
from torch.fx.passes.operator_support import any_chain, chain, OperatorSupportBase


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
    def is_node_supported(self, submodules, node: fx.Node) -> bool:
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
    return chain(
        any_chain(
            BaseTOSASupportList(),
            *(
                check(tosa_spec)
                for check in get_registered_tosa_support_checks(tosa_spec)
            ),
        ),
        *additional_checks if additional_checks else [],
    )


class BaseTOSASupportList(OperatorSupportBase):

    def is_node_supported(self, submodules, node: fx.Node) -> bool:
        supported = node.op == "call_function" and node.target in [
            exir_ops.edge.aten.add.Tensor,
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
        ]

        return supported
