# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from typing import List

import torch
from executorch.backends.cadence.aot.quantizer.patterns import (
    AddmmPattern,
    BmmPattern,
    Conv1dPattern,
    Conv2dPattern,
    LayerNormFunctionalPattern,
    LayerNormPattern,
    LinearFunctionalPattern,
    LinearPattern,
    MatmulPattern,
    ReluPattern,
)
from executorch.backends.cadence.aot.quantizer.utils import (
    is_annotated,
    no_outside_users,
)

from torch import fx

from torch.ao.quantization.observer import HistogramObserver, MinMaxObserver
from torch.ao.quantization.pt2e.graph_utils import find_sequential_partitions
from torch.ao.quantization.quantizer import Quantizer
from torch.ao.quantization.quantizer.composable_quantizer import ComposableQuantizer
from torch.ao.quantization.quantizer.xnnpack_quantizer_utils import (
    OperatorConfig,
    QuantizationAnnotation,
    QuantizationConfig,
    QuantizationSpec,
)


act_qspec = QuantizationSpec(
    dtype=torch.uint8,
    quant_min=0,
    quant_max=255,
    qscheme=torch.per_tensor_affine,
    is_dynamic=False,
    observer_or_fake_quant_ctr=HistogramObserver.with_args(eps=2**-12),
)

wgt_qspec = QuantizationSpec(
    dtype=torch.uint8,
    quant_min=0,
    quant_max=255,
    qscheme=torch.per_tensor_affine,
    is_dynamic=False,
    observer_or_fake_quant_ctr=MinMaxObserver,
)

bias_qspec = None


class CadenceGenericQuantizer(Quantizer):
    def __init__(self, pattern, quantization_config):
        super().__init__()
        self.pattern = pattern
        self.quantization_config = quantization_config

    def annotate(self, model):
        fused_partitions = find_sequential_partitions(
            model,
            self.pattern.partition_types(),
        )

        input_act_qspec = self.quantization_config.input_activation
        weight_qspec = self.quantization_config.weight
        bias_qspec = self.quantization_config.bias
        output_act_qspec = self.quantization_config.output_activation

        for fused_partition in fused_partitions:
            if not no_outside_users(fused_partition):
                continue

            anchors = self.pattern.get_anchors(model, fused_partition)
            if not anchors:
                continue
            if is_annotated(
                [
                    x[0]
                    for x in anchors.inputs
                    + anchors.weights
                    + anchors.biases
                    + anchors.output
                ]
            ):
                continue

            for output, *custom_spec in anchors.output:
                output.meta["quantization_annotation"] = QuantizationAnnotation(
                    output_qspec=custom_spec[0] if custom_spec else output_act_qspec,
                    _annotated=True,
                )

            def annotate_inputs(inputs, spec):
                for node, idx, *custom_spec in inputs:
                    annotation = node.meta.get(
                        "quantization_annotation",
                        QuantizationAnnotation(_annotated=True),
                    )
                    annotation.input_qspec_map[node.args[idx]] = (
                        custom_spec[0] if custom_spec else spec
                    )
                    node.meta["quantization_annotation"] = annotation

            annotate_inputs(anchors.inputs, input_act_qspec)
            annotate_inputs(anchors.weights, weight_qspec)
            annotate_inputs(anchors.biases, bias_qspec)

    def validate(self, model: fx.GraphModule) -> None:
        pass

    @classmethod
    def get_supported_operators(cls) -> List[OperatorConfig]:
        return []


class CadenceQuantizer(ComposableQuantizer):
    def __init__(self):
        static_qconfig = QuantizationConfig(
            act_qspec,
            act_qspec,
            wgt_qspec,
            None,
        )
        super().__init__(
            [
                CadenceGenericQuantizer(AddmmPattern(), static_qconfig),
                CadenceGenericQuantizer(BmmPattern(), static_qconfig),
                CadenceGenericQuantizer(Conv1dPattern(), static_qconfig),
                CadenceGenericQuantizer(Conv2dPattern(), static_qconfig),
                CadenceGenericQuantizer(LayerNormPattern(), static_qconfig),
                CadenceGenericQuantizer(LayerNormFunctionalPattern(), static_qconfig),
                CadenceGenericQuantizer(LinearPattern(), static_qconfig),
                CadenceGenericQuantizer(LinearFunctionalPattern(), static_qconfig),
                CadenceGenericQuantizer(MatmulPattern(), static_qconfig),
                CadenceGenericQuantizer(ReluPattern(), static_qconfig),
            ]
        )
