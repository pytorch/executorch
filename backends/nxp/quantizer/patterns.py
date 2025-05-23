# Copyright (c) Meta Platforms, Inc. and affiliates.
# Copyright 2025 NXP
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import List, Optional, Tuple, Type, Union

import torch

from executorch.backends.nxp.quantizer.utils import get_bias_qparams
from torch import fx
from torch._ops import OpOverload
from torchao.quantization.pt2e.quantizer import (
    DerivedQuantizationSpec,
    FixedQParamsQuantizationSpec,
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

    # Inputs can share quantization parameters
    inputs: List[
        Union[
            Tuple[fx.Node, Union[int, Tuple[int, int]]],
            Tuple[
                fx.Node,
                Union[int, Tuple[int, int]],
                SharedQuantizationSpec,
            ],
        ]
    ] = field(default_factory=list)
    weights: List[Tuple[fx.Node, int]] = field(default_factory=list)
    biases: List[
        Union[Tuple[fx.Node, int], Tuple[fx.Node, int, DerivedQuantizationSpec]]
    ] = field(default_factory=list)
    others: List[Tuple[fx.Node, int]] = field(default_factory=list)
    literals: List[Tuple[fx.Node, int]] = field(default_factory=list)
    output: List[Union[Tuple[fx.Node], Tuple[fx.Node, SharedQuantizationSpec]]] = field(
        default_factory=list
    )
    empty: bool = False


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


class SharedSpecPattern(QuantizationPattern):
    """
    Quantization pattern for shared quantization.

    The quantization is derived from the previous node quantization and the input and output shares the same
    quantization parameters (scale and zero-point).
    """

    def partition_types(self) -> List[Type[torch.nn.Module]]:
        pass

    def get_anchors(
        self, gm: fx.GraphModule, fused_partition: List[fx.GraphModule]
    ) -> PartitionAnchors | None:
        node = fused_partition[0].nodes[-1]
        assert len(fused_partition[0].input_nodes) == 1
        prev_node = fused_partition[0].input_nodes[0]

        # Previous node was not quantized => we are not able to share q-params
        if "quantization_annotation" not in prev_node.meta:
            return None

        qspec = SharedQuantizationSpec(prev_node)

        return PartitionAnchors(
            inputs=[(node, 0)],
            weights=[],
            biases=[],
            output=[
                (node, qspec),
            ],
        )


class AddmmPattern(QuantizationPattern):
    def partition_types(self) -> List[OpOverload]:
        return [torch.ops.aten.addmm.default]

    def get_anchors(
        self, gm: fx.GraphModule, fused_partition: List[fx.GraphModule]
    ) -> PartitionAnchors:
        # pyre-fixme[29]: `Union[BoundMethod[typing.Callable(torch._C.TensorBase.__ge...
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


class AvgPoolPattern(SharedSpecPattern):
    """
    Quantizer for AvgPool2D operator.
    """

    def partition_types(self):
        return [torch.ops.aten.avg_pool2d.default]


class Conv1dPattern(QuantizationPattern):
    def partition_types(self) -> List[OpOverload]:
        return [torch.ops.aten.conv1d.default]

    def get_anchors(
        self, gm: fx.GraphModule, fused_partition: List[fx.GraphModule]
    ) -> PartitionAnchors:
        # pyre-fixme[29]: `Union[BoundMethod[typing.Callable(torch._C.TensorBase.__ge...
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


class Conv2dPattern(QuantizationPattern):
    def partition_types(self) -> List[OpOverload]:
        return [torch.ops.aten.conv2d.default]

    def get_anchors(
        self, gm: fx.GraphModule, fused_partition: List[fx.GraphModule]
    ) -> PartitionAnchors:
        # pyre-fixme[29]: `Union[BoundMethod[typing.Callable(torch._C.TensorBase.__ge...
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


class LinearPattern(QuantizationPattern):
    def partition_types(self) -> List[OpOverload]:
        return [torch.ops.aten.linear.default]

    def get_anchors(
        self, gm: fx.GraphModule, fused_partition: List[fx.GraphModule]
    ) -> PartitionAnchors:
        # pyre-fixme[29]: `Union[BoundMethod[typing.Callable(torch._C.TensorBase.__ge...
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


class MaxPoolPattern(SharedSpecPattern):
    """
    Quantizer for MaxPool2D operator.
    """

    def partition_types(self):
        return [torch.ops.aten.max_pool2d.default]


class PadPattern(SharedSpecPattern):
    """
    Quantizer for Pad operator.
    """

    def partition_types(self):
        return [torch.ops.aten.pad.default]


class PermutePattern(SharedSpecPattern):
    """
    Quantizer for Permute operator.
    """

    def partition_types(self):
        return [torch.ops.aten.permute.default]


class ReluPattern(SharedSpecPattern):
    """
    Quantizer for Relu operator. Shared quantization spec is selected, as ReLU usually follows computation layer.
    """

    def partition_types(self):
        return [torch.ops.aten.relu.default]


class ReluInPlacePattern(SharedSpecPattern):
    """
    Quantizer for Relu operator with param inplace=True. Shared quantization spec is selected, as ReLU usually
    follows computation layer.
    """

    def partition_types(self):
        return [torch.ops.aten.relu_.default]


class ReshapePattern(SharedSpecPattern):
    """
    Quantizer for Reshape operator.
    """

    def partition_types(self):
        return [torch.ops.aten.reshape.default]


class SoftMaxPattern(QuantizationPattern):
    """
    Quantizer for Softmax operator.

    The quantization of Softmax output is fixed to scale 1/256, zero point -128, dtype int8.
    """

    def partition_types(self) -> List[OpOverload]:
        return [torch.ops.aten.softmax.int]

    def get_anchors(
        self, gm: fx.GraphModule, fused_partition: List[fx.GraphModule]
    ) -> PartitionAnchors:
        node = fused_partition[0].nodes[-1]
        assert len(fused_partition[0].input_nodes) == 1

        qspec = FixedQParamsQuantizationSpec(
            dtype=torch.int8,
            scale=1.0 / 256.0,
            zero_point=-128,
            quant_min=-128,
            quant_max=127,
            qscheme=torch.per_tensor_affine,
        )

        return PartitionAnchors(
            inputs=[(node, 0)],
            weights=[],
            biases=[],
            output=[
                (node, qspec),
            ],
        )
