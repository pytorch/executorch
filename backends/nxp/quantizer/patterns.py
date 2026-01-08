# Copyright (c) Meta Platforms, Inc. and affiliates.
# Copyright 2025 NXP
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from abc import ABC, abstractmethod
from dataclasses import dataclass, field

import torch

from executorch.backends.nxp.quantizer.utils import get_bias_qparams
from torch import fx
from torch._ops import OpOverload
from torch.fx import Node
from torchao.quantization.pt2e import (
    FakeQuantize,
    MovingAveragePerChannelMinMaxObserver,
    PerChannelMinMaxObserver,
)
from torchao.quantization.pt2e.quantizer import (
    DerivedQuantizationSpec,
    FixedQParamsQuantizationSpec,
    QuantizationSpec,
    SharedQuantizationSpec,
)

from torchao.quantization.pt2e.quantizer.quantizer import Q_ANNOTATION_KEY


@dataclass
class NodeArgsIdx:
    """
    Specifies indexes to args paramater of Node in node input annotation.


    Attributes:
        idx (int): Index to Node's args paramater (list). Selects an input Node or a list of Nodes at the index.
        inner_idx (int): If specified, index to a list pointed by 'idx' attribute. Selects an input Node at the index.
                         Default: None.
    """

    idx: int
    inner_idx: int = None


@dataclass
class PartitionAnchors:
    """
    All fields except output are lists of (node, node_args_idx) or (node, node_args_idx, quantization_spec) tuples,
    where node is from the given partition and node.args[node_args_idx] is an input to the partition. Assumes
    a single output.

    Quantizer uses inputs, weights and biases for quantization annotation. The others
    field contains tensor inputs that aren't quantized, and the literals fields contains
    is used for other types of input values as well as handling default parameters.
    """

    # Inputs can share quantization parameters
    inputs: list[
        tuple[fx.Node, NodeArgsIdx]
        | tuple[fx.Node, NodeArgsIdx, SharedQuantizationSpec],
    ] = field(default_factory=list)
    weights: list[
        tuple[fx.Node, NodeArgsIdx]
        | tuple[fx.Node, NodeArgsIdx, QuantizationSpec | FakeQuantize],
    ] = field(default_factory=list)
    biases: list[
        tuple[fx.Node, NodeArgsIdx]
        | tuple[fx.Node, NodeArgsIdx, DerivedQuantizationSpec],
    ] = field(default_factory=list)
    others: list[tuple[fx.Node, NodeArgsIdx]] = field(default_factory=list)
    literals: list[tuple[fx.Node, NodeArgsIdx]] = field(default_factory=list)
    output: list[
        tuple[fx.Node]
        | tuple[
            fx.Node,
            FixedQParamsQuantizationSpec | SharedQuantizationSpec,
        ],
    ] = field(default_factory=list)
    empty: bool = False


class QuantizationPattern(ABC):
    def __init__(self, is_qat: bool = False):
        self.is_qat = is_qat

    @abstractmethod
    def partition_types(self) -> list[OpOverload]:
        """
        List of types to be passed to find_sequential_partitions_aten.
        """
        pass

    @abstractmethod
    def get_anchors(
        self, gm: torch.fx.GraphModule, fused_partition: list[fx.GraphModule]
    ) -> PartitionAnchors | None:
        pass


class SharedSpecPattern(QuantizationPattern):
    """
    Quantization pattern for shared quantization.

    The quantization is derived from the previous node quantization and the input and output shares the same
    quantization parameters (scale and zero-point).
    """

    @abstractmethod
    def partition_types(self) -> list[torch.nn.Module]:
        pass

    def get_anchors(
        self, gm: fx.GraphModule, fused_partition: list[fx.GraphModule]
    ) -> PartitionAnchors | None:
        node = fused_partition[0].nodes[-1]
        assert len(fused_partition[0].input_nodes) == 1
        prev_node = fused_partition[0].input_nodes[0]

        # Previous node was not quantized => we are not able to share q-params
        if Q_ANNOTATION_KEY not in prev_node.meta:
            return None

        qspec = SharedQuantizationSpec(prev_node)

        return PartitionAnchors(
            inputs=[(node, NodeArgsIdx(0))],
            weights=[],
            biases=[],
            output=[
                (node, qspec),
            ],
        )


class SingleInputBasicPattern(QuantizationPattern):
    @abstractmethod
    def partition_types(self) -> list[OpOverload]:
        pass

    def get_anchors(
        self, gm: fx.GraphModule, fused_partition: list[fx.GraphModule]
    ) -> PartitionAnchors | None:
        node = fused_partition[0].nodes[-1]

        return PartitionAnchors(
            inputs=[(node, NodeArgsIdx(0))],
            weights=[],
            biases=[],
            output=[(node,)],
        )


def get_anchors_for_fixed_quant_specs(
    fused_partition: list[fx.GraphModule],
    scale: float,
    zero_point: int,
    quant_min: int = -128,
    quant_max: int = 127,
    is_qat: bool = False,
) -> PartitionAnchors:
    node = fused_partition[0].nodes[-1]
    assert len(fused_partition[0].input_nodes) == 1

    qspec_or_fake_quantize = FixedQParamsQuantizationSpec(
        dtype=torch.int8,
        scale=scale,
        zero_point=zero_point,
        quant_min=quant_min,
        quant_max=quant_max,
        qscheme=torch.per_tensor_affine,
    )

    return PartitionAnchors(
        inputs=[(node, NodeArgsIdx(0))],
        weights=[],
        biases=[],
        output=[
            (node, qspec_or_fake_quantize),
        ],
    )


class AbsPattern(SharedSpecPattern):
    """
    Quantizer for Abs operator.
    """

    def partition_types(self):
        return [torch.ops.aten.abs.default]


class AdaptiveAvgPoolPattern(SharedSpecPattern):
    """
    Quantizer for AdaptiveAvgPool2D operator.
    """

    def partition_types(self):
        return [torch.ops.aten.adaptive_avg_pool2d.default]


class AddmmPattern(QuantizationPattern):
    def __init__(self, neutron_quantizer, is_qat: bool):
        super().__init__(is_qat=is_qat)

        self.neutron_quantizer = neutron_quantizer
        self.neutron_target_info = (
            self.neutron_quantizer.neutron_target_spec.neutron_target_info
        )

    def partition_types(self) -> list[OpOverload]:
        return [torch.ops.aten.addmm.default]

    def get_anchors(
        self, gm: fx.GraphModule, fused_partition: list[fx.GraphModule]
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

        # If the following node is a fusable activation, quantize together with activation
        output = [(addmm_node,)]
        if len(
            addmm_node.users
        ) == 1 and self.neutron_target_info.is_supported_fused_activation__aten(
            activation := next(iter(addmm_node.users))
        ):
            activation_quantizer = self.neutron_quantizer.op_to_quantizer[
                activation.target
            ]
            activation_quantizer.annotate(gm)
            output = []
            activation.meta["quantization_annotation"].input_qspec_map = {}

        return PartitionAnchors(
            inputs=[(addmm_node, NodeArgsIdx(1))],
            weights=[(addmm_node, NodeArgsIdx(2))],
            biases=[(addmm_node, NodeArgsIdx(0), bias_qspec)],
            output=output,
        )


class AddTensorPattern(QuantizationPattern):
    """
    Quantization pattern for Add Tensor quantization. Accepts 1 or 2 input nodes.

    Basic quantization for all inputs and output.
    """

    def partition_types(self) -> list[torch.nn.Module]:
        return [torch.ops.aten.add.Tensor]

    def get_anchors(
        self, gm: fx.GraphModule, fused_partition: list[fx.GraphModule]
    ) -> PartitionAnchors | None:
        node = fused_partition[0].nodes[-1]
        inputs = [(node, NodeArgsIdx(0))]
        if len(fused_partition[0].input_nodes) == 2:
            inputs = [(node, NodeArgsIdx(0)), (node, NodeArgsIdx(1))]

        return PartitionAnchors(
            inputs=inputs,
            weights=[],
            biases=[],
            output=[(node,)],
        )


class SubTensorPattern(QuantizationPattern):
    """
    Quantization pattern for Sub Tensor quantization. Accepts 1 or 2 input nodes.

    Basic quantization for all inputs and output.
    """

    def partition_types(self) -> list[torch.nn.Module]:
        return [torch.ops.aten.sub.Tensor]

    def get_anchors(
        self, gm: fx.GraphModule, fused_partition: list[fx.GraphModule]
    ) -> PartitionAnchors | None:
        node = fused_partition[0].nodes[-1]
        inputs = [(node, NodeArgsIdx(0))]
        if len(fused_partition[0].input_nodes) == 2:
            inputs = [(node, NodeArgsIdx(0)), (node, NodeArgsIdx(1))]

        return PartitionAnchors(
            inputs=inputs,
            weights=[],
            biases=[],
            output=[(node,)],
        )


class AvgPoolPattern(SharedSpecPattern):
    """
    Quantizer for AvgPool2D operator.
    """

    def partition_types(self):
        return [torch.ops.aten.avg_pool2d.default]


class CatPattern(QuantizationPattern):
    """
    Quantizer for the Cat operator. The pattern is designed for the `NeutronAtenQuantizer`.

    The node can have an arbitrary number of inputs, which are all quantized.
    """

    def partition_types(self) -> list[OpOverload]:
        return [torch.ops.aten.cat.default]

    def get_anchors(
        self, gm: fx.GraphModule, fused_partition: list[fx.GraphModule]
    ) -> PartitionAnchors | None:
        node = fused_partition[0].nodes[-1]

        quantized_input = None
        for prev_node in node.args[0]:
            if "quantization_annotation" in prev_node.meta:
                quantized_input = prev_node
                break

        if quantized_input is not None:
            inputs = []
            for idx, _ in enumerate(node.args[0]):
                inputs.append(
                    (node, NodeArgsIdx(0, idx), SharedQuantizationSpec(quantized_input))
                )
            outputs = [(node, SharedQuantizationSpec(quantized_input))]

        else:
            # No previous node was quantized => we are not able to share q-params. The conversion to IR will have to
            #  re-quantize the inputs if necessary.
            inputs = [(node, NodeArgsIdx(0, idx)) for idx in range(len(node.args[0]))]
            outputs = [(node,)]

        return PartitionAnchors(
            inputs=inputs,
            weights=[],
            biases=[],
            output=outputs,
        )


class ConvPattern(QuantizationPattern):
    @abstractmethod
    def partition_types(self) -> list[OpOverload]:
        pass

    def get_anchors(
        self, gm: fx.GraphModule, fused_partition: list[fx.GraphModule]
    ) -> PartitionAnchors:
        conv_node = fused_partition[0].nodes[-1]

        bias_quantization_qspec = DerivedQuantizationSpec(
            derived_from=[
                (conv_node.args[0], conv_node),
                (conv_node.args[1], conv_node),
            ],
            derive_qparams_fn=get_bias_qparams,
            dtype=torch.int32,
            quant_min=-(2**31) + 1,
            quant_max=2**31 - 1,
            qscheme=torch.per_channel_symmetric,
            ch_axis=0,
        )

        weight_observer_or_fake_quant_ctr = (
            FakeQuantize.with_args(observer=MovingAveragePerChannelMinMaxObserver)
            if self.is_qat
            else PerChannelMinMaxObserver
        )
        weight_quantization_spec = QuantizationSpec(
            dtype=torch.int8,
            observer_or_fake_quant_ctr=weight_observer_or_fake_quant_ctr,
            quant_min=-127,
            quant_max=127,
            qscheme=torch.per_channel_symmetric,
            ch_axis=0,
        )

        # Keep bias empty if not supplied
        bias = []
        if len(conv_node.args) > 2 and conv_node.args[2] is not None:
            bias = [(conv_node, NodeArgsIdx(2), bias_quantization_qspec)]

        return PartitionAnchors(
            inputs=[(conv_node, NodeArgsIdx(0))],
            weights=[(conv_node, NodeArgsIdx(1), weight_quantization_spec)],
            biases=bias,
            output=[(conv_node,)],
        )


class Conv1dPattern(ConvPattern):
    def partition_types(self) -> list[OpOverload]:
        return [torch.ops.aten.conv1d.default]


class ConvTranspose1dPattern(ConvPattern):
    def partition_types(self) -> list[OpOverload]:
        return [torch.ops.aten.conv_transpose1d.default]


class Conv2dPattern(ConvPattern):
    def __init__(self, neutron_quantizer, is_qat: bool = False):
        super().__init__(is_qat=is_qat)

        self.neutron_quantizer = neutron_quantizer
        self.neutron_target_info = (
            self.neutron_quantizer.neutron_target_spec.neutron_target_info
        )

    def partition_types(self) -> list[OpOverload]:
        return [torch.ops.aten.conv2d.default]

    def get_anchors(
        self, gm: fx.GraphModule, fused_partition: list[fx.GraphModule]
    ) -> PartitionAnchors:
        conv_node = fused_partition[0].nodes[-1]

        bias_quantization_qspec = DerivedQuantizationSpec(
            derived_from=[
                (conv_node.args[0], conv_node),
                (conv_node.args[1], conv_node),
            ],
            derive_qparams_fn=get_bias_qparams,
            dtype=torch.int32,
            quant_min=-(2**31) + 1,
            quant_max=2**31 - 1,
            qscheme=torch.per_channel_symmetric,
            ch_axis=0,
        )

        weight_observer_or_fake_quant_ctr = (
            FakeQuantize.with_args(observer=MovingAveragePerChannelMinMaxObserver)
            if self.is_qat
            else PerChannelMinMaxObserver
        )
        weight_quantization_spec = QuantizationSpec(
            dtype=torch.int8,
            observer_or_fake_quant_ctr=weight_observer_or_fake_quant_ctr,
            quant_min=-127,
            quant_max=127,
            qscheme=torch.per_channel_symmetric,
            ch_axis=0,
        )

        # Keep bias empty if not supplied
        bias = []
        if len(conv_node.args) > 2 and conv_node.args[2] is not None:
            bias = [(conv_node, NodeArgsIdx(2), bias_quantization_qspec)]

        # If the following node is a fusable activation, quantize together with activation
        output = [(conv_node,)]
        if len(
            conv_node.users
        ) == 1 and self.neutron_target_info.is_supported_fused_activation__aten(
            activation := next(iter(conv_node.users))
        ):
            activation_quantizer = self.neutron_quantizer.op_to_quantizer[
                activation.target
            ]
            activation_quantizer.annotate(gm)
            output = []
            activation.meta["quantization_annotation"].input_qspec_map = {}

        return PartitionAnchors(
            inputs=[(conv_node, NodeArgsIdx(0))],
            weights=[(conv_node, NodeArgsIdx(1), weight_quantization_spec)],
            biases=bias,
            output=output,
        )


class ConvTranspose2dPattern(QuantizationPattern):
    def partition_types(self) -> list[OpOverload]:
        return [torch.ops.aten.conv_transpose2d.input]

    def get_anchors(
        self, gm: fx.GraphModule, fused_partition: list[fx.GraphModule]
    ) -> PartitionAnchors:
        conv_node = fused_partition[0].nodes[-1]

        bias_quantization_qspec = DerivedQuantizationSpec(
            derived_from=[
                (conv_node.args[0], conv_node),
                (conv_node.args[1], conv_node),
            ],
            derive_qparams_fn=get_bias_qparams,
            dtype=torch.int32,
            quant_min=-(2**31) + 1,
            quant_max=2**31 - 1,
            qscheme=torch.per_channel_symmetric,
            ch_axis=0,
        )

        weight_observer_or_fake_quant_ctr = PerChannelMinMaxObserver
        weight_quantization_spec = QuantizationSpec(
            dtype=torch.int8,
            observer_or_fake_quant_ctr=weight_observer_or_fake_quant_ctr,
            quant_min=-127,
            quant_max=127,
            qscheme=torch.per_channel_symmetric,
            ch_axis=1,
        )

        # Keep bias empty if not supplied
        bias = []
        if len(conv_node.args) > 2 and conv_node.args[2] is not None:
            bias = [(conv_node, NodeArgsIdx(2), bias_quantization_qspec)]

        return PartitionAnchors(
            inputs=[(conv_node, NodeArgsIdx(0))],
            weights=[(conv_node, NodeArgsIdx(1), weight_quantization_spec)],
            biases=bias,
            output=[(conv_node,)],
        )


class DropoutPattern(SharedSpecPattern):
    """
    Quantizer for Dropout operator.
    """

    def partition_types(self):
        return [torch.ops.aten.dropout.default]


class FlattenPattern(SharedSpecPattern):
    """
    Quantizer for Flatten operator.
    """

    def partition_types(self):
        return [torch.ops.aten.flatten.using_ints]


class HardTanhPattern(SingleInputBasicPattern):
    """
    Quantizer for HardTanh operator.
    """

    def partition_types(self):
        return [torch.ops.aten.hardtanh.default]

    def replacement_op(self):
        raise AssertionError()


class HardTanhInPlacePattern(SingleInputBasicPattern):
    """
    Quantizer for HardTanh operator with param inplace=True.
    """

    def partition_types(self):
        return [torch.ops.aten.hardtanh_.default]

    def get_anchors(
        self, gm: fx.GraphModule, fused_partition: list[fx.GraphModule]
    ) -> PartitionAnchors | None:
        node = fused_partition[0].nodes[-1]

        return PartitionAnchors(
            inputs=[(node, NodeArgsIdx(0))],
            weights=[],
            biases=[],
            output=[(node,)],
        )

    def replacement_op(self):
        raise AssertionError()


class LinearPattern(QuantizationPattern):
    def __init__(self, neutron_quantizer, is_qat: bool = False):
        super().__init__(is_qat=is_qat)

        self.neutron_quantizer = neutron_quantizer
        self.neutron_target_info = (
            self.neutron_quantizer.neutron_target_spec.neutron_target_info
        )

    def partition_types(self) -> list[OpOverload]:
        return [torch.ops.aten.linear.default]

    def get_anchors(
        self, gm: fx.GraphModule, fused_partition: list[fx.GraphModule]
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
            bias = [(linear_node, NodeArgsIdx(2), bias_qspec)]

        # If the following node is a fusable activation, quantize together with activation
        output = [(linear_node,)]
        if (
            len(linear_node.users) == 1
            and len(linear_node.meta["val"].shape) <= 2
            and self.neutron_target_info.is_supported_fused_activation__aten(
                activation := next(iter(linear_node.users))
            )
        ):
            activation_quantizer = self.neutron_quantizer.op_to_quantizer[
                activation.target
            ]
            activation_quantizer.annotate(gm)
            output = []
            activation.meta["quantization_annotation"].input_qspec_map = {}

        return PartitionAnchors(
            inputs=[(linear_node, NodeArgsIdx(0))],
            weights=[(linear_node, NodeArgsIdx(1))],
            biases=bias,
            output=output,
        )


class MaxPoolPattern(SharedSpecPattern):
    """
    Quantizer for MaxPool2D operator.
    """

    def partition_types(self):
        return [torch.ops.aten.max_pool2d.default]


class MeanDimPattern(SharedSpecPattern):
    """
    Quantizer for Mean Dim operator.
    """

    def partition_types(self):
        return [torch.ops.aten.mean.dim]


class MmPattern(QuantizationPattern):
    def __init__(self, neutron_quantizer, is_qat: bool = False):
        super().__init__(is_qat=is_qat)

        self.neutron_quantizer = neutron_quantizer
        self.neutron_target_info = (
            self.neutron_quantizer.neutron_target_spec.neutron_target_info
        )

    def partition_types(self) -> list[OpOverload]:
        return [torch.ops.aten.mm.default]

    def get_anchors(
        self, gm: fx.GraphModule, fused_partition: list[fx.GraphModule]
    ) -> PartitionAnchors:
        mm_node = fused_partition[0].nodes[-1]

        # If the following node is a fusable activation, quantize together with activation
        output = [(mm_node,)]
        if len(
            mm_node.users
        ) == 1 and self.neutron_target_info.is_supported_fused_activation__aten(
            activation := next(iter(mm_node.users))
        ):
            activation_quantizer = self.neutron_quantizer.op_to_quantizer[
                activation.target
            ]
            activation_quantizer.annotate(gm)
            output = []
            activation.meta["quantization_annotation"].input_qspec_map = {}

        return PartitionAnchors(
            inputs=[(mm_node, NodeArgsIdx(0))],
            weights=[(mm_node, NodeArgsIdx(1))],
            biases=[],
            output=output,
        )


class MulTensorPattern(QuantizationPattern):
    """
    Quantization pattern for Mul Tensor quantization. Accepts 1 or 2 input nodes.

    Basic quantization for all inputs and output.
    """

    def partition_types(self) -> list[torch.nn.Module]:
        return [torch.ops.aten.mul.Tensor]

    def get_anchors(
        self, gm: fx.GraphModule, fused_partition: list[fx.GraphModule]
    ) -> PartitionAnchors | None:
        node = fused_partition[0].nodes[-1]
        input_nodes = node.all_input_nodes

        qspec = FixedQParamsQuantizationSpec(
            dtype=torch.int8,
            scale=1.0 / 256.0,
            zero_point=0,
            quant_min=-128,
            quant_max=127,
            qscheme=torch.per_tensor_affine,
        )

        # The "Mul" operator in Neutron IR requires a specific scale and zero_point
        # (defined above) for its inputs.
        # Since these input nodes have already been annotated by their own patterns
        # which didn't take the requirements of "Mul" into account, we need to overwrite
        # the existing "quantization_annotation".
        for input_node in input_nodes:
            input_node.meta["quantization_annotation"].output_qspec = qspec

        return PartitionAnchors(
            inputs=[(node, NodeArgsIdx(0), qspec), (node, NodeArgsIdx(1), qspec)],
            weights=[],
            biases=[],
            output=[
                (node,),
            ],
        )


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


class TransposeIntPattern(SharedSpecPattern):
    """
    Quantizer for Transpose Int operator.
    """

    def partition_types(self) -> list[OpOverload]:
        return [torch.ops.aten.transpose.int]


class ReluPattern(SingleInputBasicPattern):
    """
    Quantizer for Relu operator.
    """

    def partition_types(self):
        return [torch.ops.aten.relu.default]


class ReluInPlacePattern(SingleInputBasicPattern):
    """
    Quantizer for Relu operator with param inplace=True.
    """

    def partition_types(self):
        return [torch.ops.aten.relu_.default]


class ReshapePattern(SharedSpecPattern):
    """
    Quantizer for Reshape operator.
    """

    def partition_types(self):
        return [torch.ops.aten.reshape.default]


class ViewPattern(SharedSpecPattern):
    """
    Quantizer for View operator.
    """

    def partition_types(self):
        return [torch.ops.aten.view.default]


class SliceTensorPattern(SharedSpecPattern):
    """
    Quantizer for Slice operator.
    """

    def partition_types(self):
        return [torch.ops.aten.slice.Tensor]


class SoftMaxPattern(QuantizationPattern):
    """
    Quantizer for Softmax operator.

    The quantization of Softmax output is fixed to scale 1/256, zero point -128, dtype int8.
    """

    def partition_types(self) -> list[OpOverload]:
        return [torch.ops.aten.softmax.int]

    def get_anchors(
        self, gm: fx.GraphModule, fused_partition: list[fx.GraphModule]
    ) -> PartitionAnchors:
        return get_anchors_for_fixed_quant_specs(
            fused_partition, scale=1.0 / 256.0, zero_point=-128, is_qat=self.is_qat
        )


class SigmoidPattern(QuantizationPattern):
    """
    Quantizer for Sigmoid operator.

    The quantization of Sigmoid output is fixed to scale 1/256, zero point -128, dtype int8.
    """

    def partition_types(self) -> list[OpOverload]:
        return [torch.ops.aten.sigmoid.default]

    def get_anchors(
        self, gm: fx.GraphModule, fused_partition: list[fx.GraphModule]
    ) -> PartitionAnchors:
        return get_anchors_for_fixed_quant_specs(
            fused_partition, scale=1.0 / 256.0, zero_point=-128, is_qat=self.is_qat
        )


class TanhPattern(QuantizationPattern):
    """
    Quantizer for Tanh operator.

    The quantization of Tanh output is fixed to scale 1/128, zero point 0, dtype int8.
    """

    def partition_types(self):
        return [torch.ops.aten.tanh.default]

    def get_anchors(
        self, gm: fx.GraphModule, fused_partition: list[fx.GraphModule]
    ) -> PartitionAnchors:
        return get_anchors_for_fixed_quant_specs(
            fused_partition, scale=1.0 / 128.0, zero_point=0, is_qat=self.is_qat
        )


class TanhInPlacePattern(QuantizationPattern):
    """
    Quantizer for inplace version of Tanh operator (torch.tanh_).

    The quantization of Tanh output is fixed to scale 1/128, zero point 0, dtype int8.
    """

    def partition_types(self):
        return [torch.ops.aten.tanh_.default]

    def get_anchors(
        self, gm: fx.GraphModule, fused_partition: list[fx.GraphModule]
    ) -> PartitionAnchors:
        return get_anchors_for_fixed_quant_specs(
            fused_partition, scale=1.0 / 128.0, zero_point=0, is_qat=self.is_qat
        )


class ActivationsConcatClusterPattern(QuantizationPattern):
    """
    Quantizer for activations concat cluster pattern.

    The quantizer matches a pattern where concat node is preceded by activation nodes preceded by Conv 2D or Linear.
    All activation nodes quantization parameters must be the same. Only activations, that have support for fusion
    to preceding compute node on Neutron are allowed. This cluster is usually produced by MoveActivationBeforeConcat
    pass. Cluster schema:

            │                     │
     ┌──────▼──────┐       ┌──────▼──────┐
     │ aten.conv2d │  ...  │ aten.conv2d │
     └──────┬──────┘       └──────┬──────┘
            │                     │
      ┌─────▼─────┐         ┌─────▼─────┐
      │ aten.relu │   ...   │ aten.relu │
      └─────┬─────┘         └─────┬─────┘
            └───────┐     ┌───────┘
                 ┌──▼─────▼─┐
                 │ aten.cat │
                 └────┬─────┘
                      │
    """

    def __init__(self, neutron_quantizer, is_qat: bool = False):
        super().__init__(is_qat=is_qat)

        self.neutron_quantizer = neutron_quantizer
        self.neutron_target_info = (
            self.neutron_quantizer.neutron_target_spec.neutron_target_info
        )

    @staticmethod
    def _all_activations_are_equal(activations: list[Node]) -> bool:
        first_input_node = activations[0]
        hardtanh_t = [
            torch.ops.aten.hardtanh.default,
            torch.ops.aten.hardtanh_.default,
        ]
        relu_t = [
            torch.ops.aten.relu.default,
            torch.ops.aten.relu_.default,
        ]
        tanh_t = [
            torch.ops.aten.tanh.default,
            torch.ops.aten.tanh_.default,
        ]

        def _activations_are_equal(activation1: Node, activation2: Node) -> bool:
            if (  # Targets are equal also with their inplace variants
                (activation1.target in hardtanh_t and activation2.target in hardtanh_t)
                or (activation1.target in relu_t and activation2.target in relu_t)
                or (activation1.target in tanh_t and activation2.target in tanh_t)
                or (
                    activation1.target == torch.ops.aten.sigmoid.default
                    and activation2.target == torch.ops.aten.sigmoid.default
                )
            ):
                return True
            elif (  # Hardtanh with min_val 0 and max_val 'inf' is equal to Relu
                activation1.target in hardtanh_t
                and activation1.args[1:] == (0.0, float("inf"))
                and activation2.target in relu_t
            ) or (
                activation1.target in relu_t
                and activation2.target in hardtanh_t
                and activation2.args[1:] == (0.0, float("inf"))
            ):
                return True
            else:
                return False

        return all(
            _activations_are_equal(activation, first_input_node)
            for activation in activations
        )

    def partition_types(self) -> list[OpOverload]:
        return [torch.ops.aten.cat.default]

    def get_anchors(
        self, gm: fx.GraphModule, fused_partition: list[fx.GraphModule]
    ) -> PartitionAnchors | None:
        cat_node = fused_partition[0].nodes[-1]

        # Check all cat inputs are supported activations
        if not all(
            self.neutron_target_info.is_supported_fused_activation__aten(input_node)
            for input_node in cat_node.all_input_nodes
        ):
            return None

        # Check all cat inputs are equal activations
        if not self._all_activations_are_equal(cat_node.all_input_nodes):
            return None

        # Check compute nodes are Conv 2D or Linear
        if not all(
            self.neutron_target_info.is_fusable_conv_or_linear__aten(compute_node)
            for input_node in cat_node.all_input_nodes
            for compute_node in input_node.all_input_nodes
        ):
            return None

        # Annotate compute nodes
        for input_node in cat_node.all_input_nodes:
            for compute_node in input_node.all_input_nodes:
                if compute_node.target not in self.neutron_quantizer.op_to_quantizer:
                    return None
                compute_node_quantizer = self.neutron_quantizer.op_to_quantizer[
                    compute_node.target
                ]
                compute_node_quantizer.annotate(gm)
                del compute_node.meta["quantization_annotation"].output_qspec

        # Annotate activations
        for input_node in cat_node.all_input_nodes:
            if input_node.target not in self.neutron_quantizer.op_to_quantizer:
                return None
            activation_quantizer = self.neutron_quantizer.op_to_quantizer[
                input_node.target
            ]
            activation_quantizer.annotate(gm)
            input_node.meta["quantization_annotation"].input_qspec_map = {}

        # Annotate cat node
        inputs = []
        first_input_node = cat_node.all_input_nodes[0]
        for idx in range(len(cat_node.all_input_nodes)):
            inputs.append(
                (
                    cat_node,
                    NodeArgsIdx(0, idx),
                    SharedQuantizationSpec(first_input_node),
                )
            )
        outputs = [(cat_node, SharedQuantizationSpec(first_input_node))]

        return PartitionAnchors(
            inputs=inputs,
            weights=[],
            biases=[],
            output=outputs,
        )
