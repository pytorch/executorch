# Copyright 2024 NXP
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# Quantizer for Neutron NPU.

from typing import List, Type

import torch
from torch import fx
from torch._ops import OpOverload
from torch.ao.quantization.observer import HistogramObserver, MinMaxObserver
from torch.ao.quantization.quantizer import (
    FixedQParamsQuantizationSpec,
    SharedQuantizationSpec,
)
from torch.ao.quantization.quantizer import QuantizationSpec
from torch.ao.quantization.quantizer.composable_quantizer import ComposableQuantizer
from torch.ao.quantization.quantizer.xnnpack_quantizer_utils import QuantizationConfig

from executorch.backends.cadence.aot.quantizer.patterns import (
    QuantizationPattern,
    PartitionAnchors,
    AddmmPattern,
    Conv1dPattern,
    Conv2dPattern,
    LinearPattern,
)
from executorch.backends.cadence.aot.quantizer.quantizer import CadenceAtenQuantizer

# Quantization Specification used by Neutron NPU
act_qspec = QuantizationSpec(
    dtype=torch.int8,
    quant_min=-128,
    quant_max=127,
    qscheme=torch.per_tensor_affine,
    is_dynamic=False,
    observer_or_fake_quant_ctr=HistogramObserver.with_args(eps=2 ** -12),
)

wgt_qspec = QuantizationSpec(
    dtype=torch.int8,
    quant_min=-128,
    quant_max=127,
    qscheme=torch.per_tensor_symmetric,
    is_dynamic=False,
    observer_or_fake_quant_ctr=MinMaxObserver,
    ch_axis=0
)

wgt_fc_qspec = QuantizationSpec(
    dtype=torch.int8,
    quant_min=-128,
    quant_max=127,
    qscheme=torch.per_tensor_symmetric,
    is_dynamic=False,
    observer_or_fake_quant_ctr=MinMaxObserver,
)
# Bias Quantization Specification is as follows:
#   dtype = torch.int32
#   quant_min, quant_max - full int32 range
#   qcheme = torch.per_channel_symetric (for Conv), torch.per_tensor_symetric for Addmmn ==> i.e. zero_point = 0
#   scale = input_scale * weight_scale
# Is set by the *PatternQuantizer directly.
bias_qspec = None


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

        # In the case of a node with shared quantization spec has no previous node, return None to not quantize the node
        if not hasattr(prev_node, "meta") or "quantization_annotation" not in prev_node.meta:
            return None
        else:
            qspec = SharedQuantizationSpec(prev_node)

            return PartitionAnchors(
                inputs=[(node, 0)],
                weights=[],
                biases=[],
                output=[(node, qspec), ],
            )

    def replacement_op(self):
        assert False


class MaxPoolPattern(SharedSpecPattern):
    """
    Quantizer for MaxPool2D operator.
    """

    def partition_types(self):
        return [torch.ops.aten.max_pool2d.default]


class AvgPoolPattern(SharedSpecPattern):
    """
    Quantizer for AvgPool2D operator.
    """

    def partition_types(self):
        return [torch.ops.aten.avg_pool2d.default]


class PadPattern(SharedSpecPattern):
    """
    Quantizer for Pad operator.
    """

    def partition_types(self):
        return [torch.ops.aten.pad.default]


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


class PermutePattern(SharedSpecPattern):
    """
    Quantizer for Permute operator.
    """

    def partition_types(self):
        return [torch.ops.aten.permute.default]


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
            output=[(node, qspec), ],
        )

    def replacement_op(self):
        assert False


class NeutronQuantizer(ComposableQuantizer):
    def __init__(self):
        static_qconfig = QuantizationConfig(
            act_qspec,
            act_qspec,
            wgt_qspec,
            None,
        )
        static_fc_qconfig = QuantizationConfig(
            act_qspec,
            act_qspec,
            wgt_fc_qspec,
            None
        )
        super().__init__(
            [
                CadenceAtenQuantizer(AddmmPattern(), static_fc_qconfig),
                CadenceAtenQuantizer(Conv1dPattern(), static_qconfig),
                CadenceAtenQuantizer(Conv2dPattern(), static_qconfig),
                CadenceAtenQuantizer(LinearPattern(), static_fc_qconfig),
                CadenceAtenQuantizer(MaxPoolPattern(), static_qconfig),
                CadenceAtenQuantizer(SoftMaxPattern(), static_qconfig),
                CadenceAtenQuantizer(ReshapePattern(), static_qconfig),
                CadenceAtenQuantizer(PermutePattern(), static_qconfig),
                CadenceAtenQuantizer(PadPattern(), static_qconfig),
                CadenceAtenQuantizer(ReluPattern(), static_qconfig),
                CadenceAtenQuantizer(ReluInPlacePattern(), static_qconfig),
                CadenceAtenQuantizer(AvgPoolPattern(), static_qconfig),
            ]
        )

    def transform_for_annotation(
            self, model: torch.fx.GraphModule
    ) -> torch.fx.GraphModule:
        return model

    def annotate(self, model: torch.fx.GraphModule) -> torch.fx.GraphModule:
        return super().annotate(model)

    def validate(self, model: torch.fx.GraphModule) -> None:
        return super().validate(model)
