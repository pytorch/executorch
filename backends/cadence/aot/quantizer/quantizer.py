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
    AddPattern,
    BmmPattern,
    CatPattern,
    Conv1dPattern,
    Conv1dReluPattern0,
    Conv1dReluPattern1,
    Conv2dPattern,
    Conv2dReluPattern0,
    Conv2dReluPattern1,
    LayerNormPattern,
    LinearPattern,
    MatmulPattern,
    MixedW8A32LinearPattern,
    QuantizationPattern,
    ReluPattern0,
    ReluPattern1,
    SoftmaxPattern,
)
from executorch.backends.cadence.aot.quantizer.utils import (
    find_sequential_partitions_aten,
    is_annotated,
    no_outside_users,
)

from torch import fx

from torchao.quantization.pt2e import HistogramObserver, MinMaxObserver
from torchao.quantization.pt2e.quantizer import (
    ComposableQuantizer,
    DerivedQuantizationSpec,
    OperatorConfig,
    QuantizationAnnotation,
    QuantizationConfig,
    QuantizationSpec,
    Quantizer,
)
from torchao.quantization.pt2e.quantizer.quantizer import Q_ANNOTATION_KEY


act_qspec_asym8s = QuantizationSpec(
    dtype=torch.int8,
    quant_min=-128,
    quant_max=127,
    qscheme=torch.per_tensor_affine,
    is_dynamic=False,
    observer_or_fake_quant_ctr=HistogramObserver.with_args(eps=2**-12),
)

act_qspec_asym16s = QuantizationSpec(
    dtype=torch.int16,
    quant_min=-32768,
    quant_max=32767,
    qscheme=torch.per_tensor_affine,
    is_dynamic=False,
    observer_or_fake_quant_ctr=HistogramObserver.with_args(eps=2**-12),
)

wgt_qspec_asym8s = QuantizationSpec(
    dtype=torch.int8,
    quant_min=-128,
    quant_max=127,
    qscheme=torch.per_tensor_affine,
    is_dynamic=False,
    observer_or_fake_quant_ctr=MinMaxObserver,
)

wgt_qspec_sym8s = QuantizationSpec(
    dtype=torch.int8,
    quant_min=-128,
    quant_max=127,
    qscheme=torch.per_tensor_symmetric,
    is_dynamic=False,
    observer_or_fake_quant_ctr=MinMaxObserver,
)

bias_qspec: Optional[QuantizationSpec] = None

qconfig_A8W8 = QuantizationConfig(
    act_qspec_asym8s,
    act_qspec_asym8s,
    wgt_qspec_asym8s,
    None,
)

qconfig_A8W8sym = QuantizationConfig(
    act_qspec_asym8s,
    act_qspec_asym8s,
    wgt_qspec_sym8s,
    None,
)

qconfig_A16 = QuantizationConfig(
    act_qspec_asym16s,
    act_qspec_asym16s,
    wgt_qspec_asym8s,
    None,
)

qconfig_A32W8sym = QuantizationConfig(
    input_activation=None,
    output_activation=None,
    weight=wgt_qspec_sym8s,
    bias=wgt_qspec_sym8s,
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

            anchors, _ = self.pattern.get_anchors(model, fused_partition)
            if not anchors or anchors.empty:
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
                output.meta[Q_ANNOTATION_KEY] = QuantizationAnnotation(
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
                        Q_ANNOTATION_KEY,
                        QuantizationAnnotation(_annotated=True),
                    )
                    arg = (
                        # pyre-ignore[16]: no attribute
                        node.args[idx]
                        if isinstance(idx, int)
                        # pyre-ignore[16]: no attribute
                        else node.args[idx[0]][idx[1]]
                    )
                    annotation.input_qspec_map[arg] = (
                        custom_spec[0] if custom_spec else spec
                    )
                    # pyre-ignore[16]: no attribute
                    node.meta[Q_ANNOTATION_KEY] = annotation

            def annotate_weights_or_biases(
                weights_or_biases: List[Tuple[fx.Node, int]],
                spec: Optional[QuantizationSpec],
            ) -> None:
                for node, idx, *custom_spec in weights_or_biases:
                    annotation = node.meta.get(
                        Q_ANNOTATION_KEY,
                        QuantizationAnnotation(_annotated=True),
                    )
                    annotation.input_qspec_map[node.args[idx]] = (
                        custom_spec[0] if custom_spec else spec
                    )
                    node.meta[Q_ANNOTATION_KEY] = annotation

            # pyre-ignore[6]: incompatible parameter type
            annotate_inputs(anchors.inputs, input_act_qspec)
            annotate_weights_or_biases(anchors.weights, weight_qspec)
            # pyre-ignore[6]: incompatible parameter type
            annotate_weights_or_biases(anchors.biases, bias_qspec)
        return model

    def validate(self, model: fx.GraphModule) -> None:
        pass

    @classmethod
    def get_supported_operators(cls) -> List[OperatorConfig]:
        return []


def get_cadence_default_quantizers() -> List[Quantizer]:
    return [
        CadenceAtenQuantizer(AddmmPattern(), qconfig_A8W8),
        CadenceAtenQuantizer(BmmPattern(), qconfig_A8W8),
        CadenceAtenQuantizer(Conv1dPattern(), qconfig_A8W8sym),
        CadenceAtenQuantizer(Conv2dPattern(), qconfig_A8W8sym),
        CadenceAtenQuantizer(LinearPattern(), qconfig_A8W8),
        CadenceAtenQuantizer(MatmulPattern(), qconfig_A8W8),
        CadenceAtenQuantizer(ReluPattern0(), qconfig_A8W8),
        CadenceAtenQuantizer(ReluPattern1(), qconfig_A8W8),
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


class CadenceWithLayerNormQuantizer(CadenceQuantizer):
    """
    Quantizer including layer norm
    """

    def __init__(self, quantizers: Optional[list[Quantizer]] = None) -> None:
        if quantizers is None:
            quantizers = get_cadence_default_quantizers()
        quantizers.append(CadenceAtenQuantizer(LayerNormPattern(), qconfig_A8W8))
        super().__init__(quantizers)


class CadenceWakeWordQuantizer(CadenceQuantizer):
    """
    Quantizer for WakeWord, including add and cat
    """

    def __init__(self, quantizers: Optional[list[Quantizer]] = None) -> None:
        if quantizers is None:
            quantizers = get_cadence_default_quantizers()
        quantizers.append(CadenceAtenQuantizer(AddPattern(), qconfig_A8W8))
        quantizers.append(CadenceAtenQuantizer(CatPattern(), qconfig_A8W8))
        super().__init__(quantizers)


class CadenceFusedConvReluQuantizer(CadenceQuantizer):
    """
    Quantizer using fused conv+relu patterns, and including add and cat
    """

    def __init__(self, quantizers: Optional[list[Quantizer]] = None) -> None:
        if quantizers is None:
            quantizers = []
        # Order matters here, perform the "fused" patterns first
        quantizers.append(CadenceAtenQuantizer(Conv1dReluPattern0(), qconfig_A8W8sym))
        quantizers.append(CadenceAtenQuantizer(Conv1dReluPattern1(), qconfig_A8W8sym))
        quantizers.append(CadenceAtenQuantizer(Conv2dReluPattern0(), qconfig_A8W8sym))
        quantizers.append(CadenceAtenQuantizer(Conv2dReluPattern1(), qconfig_A8W8sym))
        quantizers = quantizers + get_cadence_default_quantizers()
        quantizers.append(CadenceAtenQuantizer(AddPattern(), qconfig_A8W8))
        quantizers.append(CadenceAtenQuantizer(CatPattern(), qconfig_A8W8))
        super().__init__(quantizers)


class CadenceW8A32MixedQuantizer(CadenceQuantizer):
    """
    Quantizer for mixed quantization, 8 bit weights and 32 bit activations
    TODO: Experimental quantizer, not yet well supported in OSS
    """

    def __init__(self) -> None:
        quantizers = []
        quantizers.append(
            CadenceAtenQuantizer(MixedW8A32LinearPattern(), qconfig_A32W8sym)
        )
        super().__init__(quantizers)


class CadenceWithSoftmaxQuantizer(CadenceQuantizer):
    """
    Quantizer including A16 softmax
    """

    def __init__(self, quantizers: Optional[list[Quantizer]] = None) -> None:
        if quantizers is None:
            quantizers = get_cadence_default_quantizers()
        quantizers.append(CadenceAtenQuantizer(SoftmaxPattern(), qconfig_A16))
        super().__init__(quantizers)
