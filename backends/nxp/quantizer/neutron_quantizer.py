# Copyright (c) Meta Platforms, Inc. and affiliates.
# Copyright 2024-2025 NXP
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from typing import List, Optional, Tuple, Union

import torch
from executorch.backends.nxp.aten_passes.neutron_aten_pass_manager import (
    NeutronAtenPassManager,
)

from executorch.backends.nxp.quantizer.patterns import (
    AddmmPattern,
    AvgPoolPattern,
    Conv1dPattern,
    Conv2dPattern,
    LinearPattern,
    MaxPoolPattern,
    PadPattern,
    PermutePattern,
    QuantizationPattern,
    ReluInPlacePattern,
    ReluPattern,
    ReshapePattern,
    SoftMaxPattern,
)
from executorch.backends.nxp.quantizer.utils import (
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


class NeutronAtenQuantizer(Quantizer):
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
                    node.meta["quantization_annotation"] = annotation

            def annotate_weights_or_biases(
                weights_or_biases: List[Tuple[fx.Node, int]],
                spec: Optional[QuantizationSpec],
            ) -> None:
                for node, idx, *custom_spec in weights_or_biases:
                    annotation = node.meta.get(
                        "quantization_annotation",
                        QuantizationAnnotation(_annotated=True),
                    )
                    annotation.input_qspec_map[node.args[idx]] = (
                        custom_spec[0] if custom_spec else spec
                    )
                    node.meta["quantization_annotation"] = annotation

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


# Quantization Specification used by Neutron NPU
act_qspec = QuantizationSpec(
    dtype=torch.int8,
    quant_min=-128,
    quant_max=127,
    qscheme=torch.per_tensor_affine,
    is_dynamic=False,
    observer_or_fake_quant_ctr=HistogramObserver.with_args(eps=2**-12),
)

wgt_qspec = QuantizationSpec(
    dtype=torch.int8,
    quant_min=-127,
    quant_max=127,
    qscheme=torch.per_tensor_symmetric,
    is_dynamic=False,
    observer_or_fake_quant_ctr=MinMaxObserver,
    ch_axis=0,
)

wgt_fc_qspec = QuantizationSpec(
    dtype=torch.int8,
    quant_min=-127,
    quant_max=127,
    qscheme=torch.per_tensor_symmetric,
    is_dynamic=False,
    observer_or_fake_quant_ctr=MinMaxObserver,
)

# Is set by the *PatternQuantizer directly.
bias_qspec = None


class NeutronQuantizer(ComposableQuantizer):
    def __init__(self):
        static_qconfig = QuantizationConfig(
            act_qspec,
            act_qspec,
            wgt_qspec,
            None,
        )
        static_fc_qconfig = QuantizationConfig(act_qspec, act_qspec, wgt_fc_qspec, None)
        super().__init__(
            [
                NeutronAtenQuantizer(AddmmPattern(), static_fc_qconfig),
                NeutronAtenQuantizer(Conv1dPattern(), static_qconfig),
                NeutronAtenQuantizer(Conv2dPattern(), static_qconfig),
                NeutronAtenQuantizer(LinearPattern(), static_fc_qconfig),
                NeutronAtenQuantizer(MaxPoolPattern(), static_qconfig),
                NeutronAtenQuantizer(SoftMaxPattern(), static_qconfig),
                NeutronAtenQuantizer(ReshapePattern(), static_qconfig),
                NeutronAtenQuantizer(PermutePattern(), static_qconfig),
                NeutronAtenQuantizer(PadPattern(), static_qconfig),
                NeutronAtenQuantizer(ReluPattern(), static_qconfig),
                NeutronAtenQuantizer(ReluInPlacePattern(), static_qconfig),
                NeutronAtenQuantizer(AvgPoolPattern(), static_qconfig),
            ]
        )

    def transform_for_annotation(
        self, model: torch.fx.GraphModule
    ) -> torch.fx.GraphModule:
        pass_runner = NeutronAtenPassManager()
        return pass_runner(model).graph_module
