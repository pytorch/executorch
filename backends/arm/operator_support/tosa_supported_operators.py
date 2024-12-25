# Copyright 2024 Arm Limited and/or its affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-unsafe

import operator
from typing import Type

import torch.fx as fx
from executorch.backends.arm.tosa_specification import TosaSpecification
from executorch.exir.dialects._ops import ops as exir_ops
from torch.fx.passes.operator_support import OperatorSupportBase


class SupportedTOSAOperatorCheck:
    """
    Supported OP for TOSA lowering
    """

    # Should be populated by subclass implementation
    tosa_specs: list[TosaSpecification] = []
    targets: list[str] = []

    def is_node_supported(self, node: fx.Node, tosa_spec: TosaSpecification) -> bool:
        """
        Checks if the fx.Node node is lowerable using the TOSA specification defined by tosa_spec.
        To be implemented by subclasses targeting
        """
        raise NotImplementedError("NodeVisitor must be extended.")


# container for all SupportedTosaOperatorCheck classes
_tosa_spec_dicts: dict[
    TosaSpecification, dict[str, Type[SupportedTOSAOperatorCheck]]
] = {
    TosaSpecification.create_from_string("TOSA-0.80+BI"): {},
    TosaSpecification.create_from_string("TOSA-0.80+MI"): {},
}


def register_tosa_support_check(checker):
    """
    Decorator to mark a subclass implmentation of SupportedTosaOperatorCheck
    to be registered for checking if a torch.fx.Node is lowerable given
    a TOSA specification.
    """
    for tosa_spec in checker.tosa_specs:
        for target in checker.targets:
            _tosa_spec_dicts[tosa_spec][target] = checker
    return checker


def get_registered_tosa_support_checks(
    tosa_spec: TosaSpecification,
) -> dict[str, SupportedTOSAOperatorCheck]:

    if tosa_spec not in _tosa_spec_dicts:
        raise RuntimeError

    tosa_support_checks = {}
    for target, tosa_check in _tosa_spec_dicts[tosa_spec].items():
        tosa_support_checks[target] = tosa_check()

    return tosa_support_checks


class TOSASupportedOperators(OperatorSupportBase):
    def __init__(self, tosa_spec: TosaSpecification):
        super().__init__()
        self.tosa_spec = tosa_spec

    def is_node_supported(self, submodules, node: fx.Node) -> bool:
        supported = node.op == "call_function" and node.target in [
            exir_ops.edge.aten.add.Tensor,
            exir_ops.edge.aten.expand_copy.default,
            exir_ops.edge.aten.cat.default,
            exir_ops.edge.aten.bmm.default,
            exir_ops.edge.aten.permute_copy.default,
            exir_ops.edge.aten.hardtanh.default,
            exir_ops.edge.aten.convolution.default,
            exir_ops.edge.aten.div.Tensor,
            exir_ops.edge.aten.exp.default,
            exir_ops.edge.aten.log.default,
            exir_ops.edge.aten.linear.default,
            exir_ops.edge.aten.split_with_sizes_copy.default,
            exir_ops.edge.aten.full.default,
            exir_ops.edge.aten.mul.Tensor,
            exir_ops.edge.aten._native_batch_norm_legit_no_training.default,
            exir_ops.edge.aten.native_layer_norm.default,
            exir_ops.edge.aten.avg_pool2d.default,
            exir_ops.edge.aten.max_pool2d_with_indices.default,
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
            exir_ops.edge.aten.sum.dim_IntList,
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
        ]

        if not supported:
            supported = self.is_node_supported_custom(node)

        # Override partitioning based on pre partition passes
        if "arm_override_partition" in node.meta:
            supported = supported & node.meta["arm_override_partition"]
            node.meta.pop("arm_override_partition")

        return supported

    def is_node_supported_custom(self, node: fx.Node) -> bool:
        tosa_checks = get_registered_tosa_support_checks(self.tosa_spec)
        if node.target in tosa_checks.keys():
            return tosa_checks[node.target].is_node_supported(node, self.tosa_spec)
        return False
