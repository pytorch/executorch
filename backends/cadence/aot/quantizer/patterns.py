# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import List, Optional, Tuple, Union

import torch
from executorch.backends.cadence.aot.quantizer.utils import get_bias_qparams

from torch import fx
from torch._ops import OpOverload
from torch.ao.quantization.quantizer import (
    DerivedQuantizationSpec,
    SharedQuantizationSpec,
)


@dataclass
class PartitionAnchors:
    """
    All fields except output are lists of (node, args_index) pair, where node is from
    the given partition and node.args[args_index] is an input to the partition. Assumes
    a single output.

    Quantizer uses inputs, weights and biases for quantization annotation. The others
    field contains tensor inputs that aren't quantized, and the literals fields contains
    is used for other types of input values as well as handling default parameters.
    """

    inputs: List[Tuple[fx.Node, int]] = field(default_factory=list)
    weights: List[Tuple[fx.Node, int]] = field(default_factory=list)
    biases: List[
        Union[Tuple[fx.Node, int], Tuple[fx.Node, int, DerivedQuantizationSpec]]
    ] = field(default_factory=list)
    others: List[Tuple[fx.Node, int]] = field(default_factory=list)
    literals: List[Tuple[fx.Node, int]] = field(default_factory=list)
    output: List[Union[Tuple[fx.Node], Tuple[fx.Node, SharedQuantizationSpec]]] = field(
        default_factory=list
    )


class QuantizationPattern(ABC):
    @abstractmethod
    def partition_types(self) -> list[OpOverload]:
        """
        List of types to be passed to find_sequential_partitions_aten.
        """
        pass

    @abstractmethod
    def get_anchors(
        self, gm: torch.fx.GraphModule, fused_partition: List[fx.GraphModule]
    ) -> Optional[PartitionAnchors]:
        pass

    @abstractmethod
    def replacement_op(self) -> OpOverload:
        """
        Operator (most likely a custom one) that this partition should be fused into in
        the backend. Refer to the QuantFusion pass for examples.
        """
        pass


class AddmmPattern(QuantizationPattern):
    def partition_types(self) -> List[OpOverload]:
        return [torch.ops.aten.addmm.default]

    def get_anchors(
        self, gm: fx.GraphModule, fused_partition: List[fx.GraphModule]
    ) -> PartitionAnchors:
        addmm_node = fused_partition[0].nodes[-1]

        bias_qspec = DerivedQuantizationSpec(
            derived_from=[
                (addmm_node.args[1], addmm_node),
                (addmm_node.args[2], addmm_node),
            ],
            derive_qparams_fn=get_bias_qparams,
            dtype=torch.int32,
            quant_min=-(2**31),
            quant_max=2**31 - 1,
            qscheme=torch.per_tensor_affine,
        )

        return PartitionAnchors(
            inputs=[(addmm_node, 1)],
            weights=[(addmm_node, 2)],
            biases=[(addmm_node, 0, bias_qspec)],
            output=[(addmm_node,)],
        )

    def replacement_op(self) -> OpOverload:
        return torch.ops.cadence.quantized_linear


class BmmPattern(QuantizationPattern):
    def partition_types(self) -> List[OpOverload]:
        return [torch.ops.aten.bmm.default]

    def get_anchors(
        self, gm: fx.GraphModule, fused_partition: List[fx.GraphModule]
    ) -> PartitionAnchors:
        bmm_node = fused_partition[0].nodes[-1]

        return PartitionAnchors(
            inputs=[(bmm_node, 0), (bmm_node, 1)],
            weights=[],
            biases=[],
            output=[(bmm_node,)],
        )

    def replacement_op(self) -> OpOverload:
        return torch.ops.cadence.quantized_matmul.default


class Conv1dPattern(QuantizationPattern):
    def partition_types(self) -> List[OpOverload]:
        return [torch.ops.aten.conv1d.default]

    def get_anchors(
        self, gm: fx.GraphModule, fused_partition: List[fx.GraphModule]
    ) -> PartitionAnchors:
        conv1d_node = fused_partition[0].nodes[-1]

        bias_qspec = DerivedQuantizationSpec(
            derived_from=[
                (conv1d_node.args[0], conv1d_node),
                (conv1d_node.args[1], conv1d_node),
            ],
            derive_qparams_fn=get_bias_qparams,
            dtype=torch.int32,
            quant_min=-(2**31),
            quant_max=2**31 - 1,
            qscheme=torch.per_tensor_affine,
        )

        # Keep bias empty if not supplied
        bias = []
        if len(conv1d_node.args) > 2 and conv1d_node.args[2] is not None:
            bias = [(conv1d_node, 2, bias_qspec)]

        return PartitionAnchors(
            inputs=[(conv1d_node, 0)],
            weights=[(conv1d_node, 1)],
            # pyre-fixme[6]: Incompatible parameter type
            biases=bias,
            output=[(conv1d_node,)],
        )

    def replacement_op(self) -> OpOverload:
        return torch.ops.cadence.quantized_conv.default


class Conv2dPattern(QuantizationPattern):
    def partition_types(self) -> List[OpOverload]:
        return [torch.ops.aten.conv2d.default]

    def get_anchors(
        self, gm: fx.GraphModule, fused_partition: List[fx.GraphModule]
    ) -> PartitionAnchors:
        conv2d_node = fused_partition[0].nodes[-1]

        bias_qspec = DerivedQuantizationSpec(
            derived_from=[
                (conv2d_node.args[0], conv2d_node),
                (conv2d_node.args[1], conv2d_node),
            ],
            derive_qparams_fn=get_bias_qparams,
            dtype=torch.int32,
            quant_min=-(2**31),
            quant_max=2**31 - 1,
            qscheme=torch.per_tensor_affine,
        )

        # Keep bias empty if not supplied
        bias = []
        if len(conv2d_node.args) > 2 and conv2d_node.args[2] is not None:
            bias = [(conv2d_node, 2, bias_qspec)]

        return PartitionAnchors(
            inputs=[(conv2d_node, 0)],
            weights=[(conv2d_node, 1)],
            # pyre-fixme[6]: Incompatible parameter type
            biases=bias,
            output=[(conv2d_node,)],
        )

    def replacement_op(self) -> OpOverload:
        return torch.ops.cadence.quantized_conv.default


class LayerNormPattern(QuantizationPattern):
    def partition_types(self) -> List[OpOverload]:
        return [torch.ops.aten.layer_norm.default]

    def get_anchors(
        self, gm: fx.GraphModule, fused_partition: List[fx.GraphModule]
    ) -> PartitionAnchors:
        layer_norm_node = fused_partition[0].nodes[-1]

        others = [(layer_norm_node, 1)]

        # Add weights if supplied
        if len(layer_norm_node.args) > 2 and layer_norm_node.args[2]:
            others.append((layer_norm_node, 2))

        # Add bias if supplied
        if len(layer_norm_node.args) > 3 and layer_norm_node.args[3]:
            others.append((layer_norm_node, 3))

        # Weights are used in quantized mode by our kernel, so they are
        # passed in as others here along with the normalized shape.
        return PartitionAnchors(
            inputs=[(layer_norm_node, 0)],
            weights=[],
            biases=[],
            # Ordering: normalized_shape, weights, bias
            others=others,
            output=[(layer_norm_node,)],
        )

    def replacement_op(self) -> OpOverload:
        return torch.ops.cadence.quantized_layer_norm.default


class LinearPattern(QuantizationPattern):
    def partition_types(self) -> List[OpOverload]:
        return [torch.ops.aten.linear.default]

    def get_anchors(
        self, gm: fx.GraphModule, fused_partition: List[fx.GraphModule]
    ) -> PartitionAnchors:
        linear_node = fused_partition[0].nodes[-1]

        bias_qspec = DerivedQuantizationSpec(
            derived_from=[
                (linear_node.args[0], linear_node),
                (linear_node.args[1], linear_node),
            ],
            derive_qparams_fn=get_bias_qparams,
            dtype=torch.int32,
            quant_min=-(2**31),
            quant_max=2**31 - 1,
            qscheme=torch.per_tensor_affine,
        )

        # Keep bias empty if not supplied
        bias = []
        if len(linear_node.args) > 2:
            bias = [(linear_node, 2, bias_qspec)]

        return PartitionAnchors(
            inputs=[(linear_node, 0)],
            weights=[(linear_node, 1)],
            # pyre-fixme[6]: Incompatible parameter type
            biases=bias,
            output=[(linear_node,)],
        )

    def replacement_op(self) -> OpOverload:
        return torch.ops.cadence.quantized_linear.default


class MatmulPattern(QuantizationPattern):
    def partition_types(self) -> List[OpOverload]:
        return [torch.ops.aten.matmul.default]

    def get_anchors(
        self, gm: fx.GraphModule, fused_partition: List[fx.GraphModule]
    ) -> PartitionAnchors:
        matmul_node = fused_partition[0].nodes[-1]

        return PartitionAnchors(
            inputs=[(matmul_node, 0), (matmul_node, 1)],
            weights=[],
            biases=[],
            output=[(matmul_node,)],
        )

    def replacement_op(self) -> OpOverload:
        return torch.ops.cadence.quantized_matmul.default


# This is a base class for ReLU, since it can be used with two different aten ops
class ReluBasePattern(QuantizationPattern):
    @abstractmethod
    def partition_types(self) -> List[OpOverload]:
        pass

    def get_anchors(
        self, gm: fx.GraphModule, fused_partition: List[fx.GraphModule]
    ) -> PartitionAnchors:
        relu_node = fused_partition[0].nodes[-1]

        return PartitionAnchors(
            inputs=[(relu_node, 0)],
            weights=[],
            biases=[],
            output=[(relu_node,)],
        )

    def replacement_op(self) -> OpOverload:
        return torch.ops.cadence.quantized_relu.default


# Regular relu op
class ReluPattern0(ReluBasePattern):
    def partition_types(self) -> List[OpOverload]:
        return [torch.ops.aten.relu.default]


# Alternate relu op
class ReluPattern1(ReluBasePattern):
    def partition_types(self) -> List[OpOverload]:
        return [torch.ops.aten.relu_.default]
