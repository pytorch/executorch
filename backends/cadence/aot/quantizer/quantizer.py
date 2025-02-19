# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

from dataclasses import dataclass
from typing import List, Optional, Tuple, Union

import torch
from executorch.backends.cadence.aot.quantizer.patterns import (
    AddmmPattern,
    BmmPattern,
    Conv1dPattern,
    Conv2dPattern,
    LayerNormPattern,
    LinearPattern,
    MatmulPattern,
    QuantizationPattern,
    ReluPattern0,
    ReluPattern1,
)
from executorch.backends.cadence.aot.quantizer.utils import (
    find_sequential_partitions_aten,
    is_annotated,
    no_outside_users,
)
from executorch.backends.xnnpack.quantizer.xnnpack_quantizer_utils import (
    OperatorConfig,
    QuantizationAnnotation,
    QuantizationConfig,
    QuantizationSpec,
)

from torch import fx

from torch.ao.quantization.observer import HistogramObserver, MinMaxObserver
from torch.ao.quantization.quantizer import DerivedQuantizationSpec, Quantizer
from torch.ao.quantization.quantizer.composable_quantizer import ComposableQuantizer


act_qspec_asym8u = QuantizationSpec(
    dtype=torch.int8,
    quant_min=-128,
    quant_max=127,
    qscheme=torch.per_tensor_affine,
    is_dynamic=False,
    observer_or_fake_quant_ctr=HistogramObserver.with_args(eps=2**-12),
)

wgt_qspec_asym8u = QuantizationSpec(
    dtype=torch.int8,
    quant_min=-128,
    quant_max=127,
    qscheme=torch.per_tensor_affine,
    is_dynamic=False,
    observer_or_fake_quant_ctr=MinMaxObserver,
)

wgt_qspec_asym8s = QuantizationSpec(
    dtype=torch.int8,
    quant_min=-128,
    quant_max=127,
    qscheme=torch.per_tensor_symmetric,
    is_dynamic=False,
    observer_or_fake_quant_ctr=MinMaxObserver,
)

bias_qspec: Optional[QuantizationSpec] = None

qconfig_A8uW8u = QuantizationConfig(
    act_qspec_asym8u,
    act_qspec_asym8u,
    wgt_qspec_asym8u,
    None,
)

qconfig_A8uW8s = QuantizationConfig(
    act_qspec_asym8u,
    act_qspec_asym8u,
    wgt_qspec_asym8s,
    None,
)


class CadenceAtenQuantizer(Quantizer):
    def __init__(
        self, pattern: QuantizationPattern, quantization_config: QuantizationConfig
    ) -> None:
        super().__init__()
        self.pattern = pattern
        self.quantization_config = quantization_config

    def annotate(self, model: torch.fx.GraphModule) -> torch.fx.GraphModule:
        fused_partitions = find_sequential_partitions_aten(
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
                # pyre-ignore[16]: no attribute
                output.meta["quantization_annotation"] = QuantizationAnnotation(
                    # pyre-ignore[6]: incompatible parameter type
                    output_qspec=(custom_spec[0] if custom_spec else output_act_qspec),
                    _annotated=True,
                )

            def annotate_inputs(
                inputs: Union[
                    List[Tuple[fx.Node, int]],
                    List[Tuple[fx.Node, int, DerivedQuantizationSpec],],
                ],
                spec: Optional[QuantizationSpec],
            ) -> None:
                for node, idx, *custom_spec in inputs:
                    # pyre-ignore[16]: no attribute
                    annotation = node.meta.get(
                        "quantization_annotation",
                        QuantizationAnnotation(_annotated=True),
                    )
                    # pyre-ignore[16]: no attribute
                    annotation.input_qspec_map[node.args[idx]] = (
                        custom_spec[0] if custom_spec else spec
                    )
                    # pyre-ignore[16]: no attribute
                    node.meta["quantization_annotation"] = annotation

            annotate_inputs(anchors.inputs, input_act_qspec)
            annotate_inputs(anchors.weights, weight_qspec)
            # pyre-ignore[6]: incompatible parameter type
            annotate_inputs(anchors.biases, bias_qspec)
        return model

    def validate(self, model: fx.GraphModule) -> None:
        pass

    @classmethod
    def get_supported_operators(cls) -> List[OperatorConfig]:
        return []


def get_cadence_default_quantizers() -> List[Quantizer]:
    return [
        CadenceAtenQuantizer(AddmmPattern(), qconfig_A8uW8u),
        CadenceAtenQuantizer(BmmPattern(), qconfig_A8uW8u),
        CadenceAtenQuantizer(Conv1dPattern(), qconfig_A8uW8s),
        CadenceAtenQuantizer(Conv2dPattern(), qconfig_A8uW8s),
        CadenceAtenQuantizer(LayerNormPattern(), qconfig_A8uW8u),
        CadenceAtenQuantizer(LinearPattern(), qconfig_A8uW8u),
        CadenceAtenQuantizer(MatmulPattern(), qconfig_A8uW8u),
        CadenceAtenQuantizer(ReluPattern0(), qconfig_A8uW8u),
        CadenceAtenQuantizer(ReluPattern1(), qconfig_A8uW8u),
    ]


# Note: need dataclass to be used in CI configs through OmegaConf and Hydra
@dataclass
class CadenceQuantizer(ComposableQuantizer):
    """
    Generic CadenceQuantizer. Although it can be used directly, it is typically a base
    class for explicitly defined quantizers (like CadenceDefaultQuantizer).
    """

    def __init__(self, quantizers: List[Quantizer]) -> None:
        super().__init__(quantizers)


class CadenceDefaultQuantizer(CadenceQuantizer):
    """
    Default quantizer for Cadence backend.
    """

    def __init__(self, quantizers: Optional[list[Quantizer]] = None) -> None:
        if quantizers is None:
            quantizers = get_cadence_default_quantizers()
        super().__init__(quantizers)


# Nop quantizer, used to run fp32 cases
# Calls an empty list of quantizers (no quantization). Note
# that we do not strictly need that class since we could call
# CadenceQuantizer([]), but this is more explicit and
# does not require knowledge of the internals of the base class.
class CadenceNopQuantizer(CadenceQuantizer):
    def __init__(
        self,
    ) -> None:
        super().__init__([])
