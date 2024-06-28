# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Callable, List, Optional, Tuple, Type, Union

import torch
from executorch.backends.cadence.aot.quantizer.utils import get_bias_qparams

from torch import fx
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
    def partition_types(self):
        """
        List of types to be passed to find_sequential_partitions.
        """
        pass

    @abstractmethod
    def get_anchors(self, gm, fused_partition) -> Optional[PartitionAnchors]:
        pass

    @abstractmethod
    def replacement_op(self) -> Callable[..., Any]:
        """
        Operator (most likely a custom one) that this partition should be fused into in
        the backend. Refer to the QuantFusion pass for examples.
        """
        pass


class AddmmPattern(QuantizationPattern):
    def partition_types(self) -> List[Type[torch.nn.Module]]:
        return [torch.addmm]

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

    def replacement_op(self):
        return torch.ops.cadence.quantized_linear


class Conv1dPattern(QuantizationPattern):
    def partition_types(self) -> List[Type[torch.nn.Module]]:
        return [torch.nn.Conv1d]

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

    def replacement_op(self):
        return torch.ops.cadence.quantized_conv.default


class Conv2dPattern(QuantizationPattern):
    def partition_types(self) -> List[Type[torch.nn.Module]]:
        return [torch.nn.Conv2d]

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

    def replacement_op(self):
        return torch.ops.cadence.quantized_conv.default


class LayerNormPattern(QuantizationPattern):
    def partition_types(self):
        return [torch.nn.LayerNorm]

    def get_anchors(self, gm, fused_partition) -> PartitionAnchors:
        layer_norm_node = fused_partition[0].nodes[-1]

        # Weights and biases are used as fp32 by our kernel, so they are
        # passed in as others here along with the normalized shape.
        return PartitionAnchors(
            inputs=[(layer_norm_node, 0)],
            weights=[],
            biases=[],
            # Ordering: normalized_shape, weights, bias
            others=[(layer_norm_node, 1), (layer_norm_node, 2), (layer_norm_node, 3)],
            output=[(layer_norm_node,)],
        )

    def replacement_op(self):
        return torch.ops.cadence.quantized_layer_norm.default


class LayerNormFunctionalPattern(QuantizationPattern):
    def partition_types(self):
        return [torch.nn.functional.layer_norm]

    def get_anchors(self, gm, fused_partition) -> PartitionAnchors:
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

    def replacement_op(self):
        return torch.ops.cadence.quantized_layer_norm.default


class LinearPattern(QuantizationPattern):
    def partition_types(self) -> List[Type[torch.nn.Module]]:
        return [torch.nn.Linear]

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

    def replacement_op(self):
        return torch.ops.cadence.quantized_linear.default


class LinearFunctionalPattern(QuantizationPattern):
    def partition_types(self):
        return [torch.nn.functional.linear]

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
        if len(linear_node.args) > 2 and linear_node.args[2] is not None:
            bias = [(linear_node, 2, bias_qspec)]

        return PartitionAnchors(
            inputs=[(linear_node, 0)],
            weights=[(linear_node, 1)],
            # pyre-fixme[6]: Incompatible parameter type
            biases=bias,
            output=[(linear_node,)],
        )

    def replacement_op(self):
        return torch.ops.cadence.quantized_linear.default


class MatmulPattern(QuantizationPattern):
    def partition_types(self):
        return [torch.matmul]

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

    def replacement_op(self):
        return torch.ops.cadence.quantized_matmul.default


class ReluPattern(QuantizationPattern):
    def partition_types(self) -> List[Type[torch.nn.Module]]:
        return [torch.nn.ReLU]

    def get_anchors(
        self, gm: fx.GraphModule, fused_partition: List[fx.GraphModule]
    ) -> PartitionAnchors:
        relu_node = fused_partition[0].nodes[-1]

        return PartitionAnchors(
            inputs=[(relu_node, 0)],
            weights=[],
            biases=[],
            output=[
                (relu_node, SharedQuantizationSpec((relu_node.args[0], relu_node)))
            ],
        )

    def replacement_op(self):
        return torch.ops.cadence.quantized_relu.default
