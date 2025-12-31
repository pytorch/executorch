# Copyright (c) Meta Platforms, Inc. and affiliates.
# Copyright 2024-2025 NXP
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import torch
from executorch.backends.nxp.aten_passes.neutron_aten_pass_manager import (
    NeutronAtenPassManager,
)

from executorch.backends.nxp.backend.neutron_target_spec import NeutronTargetSpec
from executorch.backends.nxp.quantizer.patterns import (
    AbsPattern,
    ActivationsConcatClusterPattern,
    AdaptiveAvgPoolPattern,
    AddmmPattern,
    AddTensorPattern,
    AvgPoolPattern,
    CatPattern,
    Conv1dPattern,
    Conv2dPattern,
    ConvTranspose2dPattern,
    DropoutPattern,
    FlattenPattern,
    HardTanhInPlacePattern,
    HardTanhPattern,
    LinearPattern,
    MaxPoolPattern,
    MeanDimPattern,
    MmPattern,
    MulTensorPattern,
    NodeArgsIdx,
    PadPattern,
    PermutePattern,
    QuantizationPattern,
    ReluInPlacePattern,
    ReluPattern,
    ReshapePattern,
    SharedSpecPattern,
    SigmoidPattern,
    SliceTensorPattern,
    SoftMaxPattern,
    SubTensorPattern,
    TanhInPlacePattern,
    TanhPattern,
    TransposeIntPattern,
    ViewPattern,
)
from executorch.backends.nxp.quantizer.utils import (
    find_sequential_partitions_aten,
    is_annotated,
    no_outside_users,
)
from torch import fx
from torchao.quantization.pt2e import (
    FakeQuantize,
    FusedMovingAvgObsFakeQuantize,
    HistogramObserver,
    MinMaxObserver,
    MovingAverageMinMaxObserver,
)
from torchao.quantization.pt2e.quantizer import (
    annotate_output_qspec,
    ComposableQuantizer,
    DerivedQuantizationSpec,
    OperatorConfig,
    QuantizationAnnotation,
    QuantizationConfig,
    QuantizationSpec,
    Quantizer,
)
from torchao.quantization.pt2e.quantizer.quantizer import Q_ANNOTATION_KEY


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
                output.meta[Q_ANNOTATION_KEY] = QuantizationAnnotation(
                    # pyre-ignore[6]: incompatible parameter type
                    output_qspec=(custom_spec[0] if custom_spec else output_act_qspec),
                    _annotated=True,
                )

            def annotate_inputs(
                inputs: (
                    list[tuple[fx.Node, NodeArgsIdx]]
                    | list[tuple[fx.Node, NodeArgsIdx, DerivedQuantizationSpec]]
                ),
                spec: QuantizationSpec | None,
            ) -> None:
                for node, args_idx, *custom_spec in inputs:
                    # pyre-ignore[16]: no attribute
                    annotation = node.meta.get(
                        Q_ANNOTATION_KEY,
                        QuantizationAnnotation(_annotated=True),
                    )
                    arg = (
                        # pyre-ignore[16]: no attribute
                        node.args[args_idx.idx]
                        if args_idx.inner_idx is None
                        # pyre-ignore[16]: no attribute
                        else node.args[args_idx.idx][args_idx.inner_idx]
                    )
                    annotation.input_qspec_map[arg] = (
                        custom_spec[0] if custom_spec else spec
                    )
                    # pyre-ignore[16]: no attribute
                    node.meta[Q_ANNOTATION_KEY] = annotation

            # pyre-ignore[6]: incompatible parameter type
            annotate_inputs(anchors.inputs, input_act_qspec)
            annotate_inputs(anchors.weights, weight_qspec)
            # pyre-ignore[6]: incompatible parameter type
            annotate_inputs(anchors.biases, bias_qspec)
        return model

    def validate(self, model: fx.GraphModule) -> None:
        pass

    @classmethod
    def get_supported_operators(cls) -> list[OperatorConfig]:
        return []


# Quantization Specification used by Neutron NPU
def act_qspec(is_qat: bool):
    eps = 2**-12
    observer_or_fake_quant_ctr = (
        FusedMovingAvgObsFakeQuantize.with_args(
            observer=MovingAverageMinMaxObserver, eps=eps
        )
        if is_qat
        else HistogramObserver.with_args(eps=eps)
    )

    return QuantizationSpec(
        dtype=torch.int8,
        quant_min=-128,
        quant_max=127,
        qscheme=torch.per_tensor_affine,
        is_dynamic=False,
        observer_or_fake_quant_ctr=observer_or_fake_quant_ctr,
    )


def wgt_qspec(is_qat: bool):
    observer_or_fake_quant_ctr = (
        FakeQuantize.with_args(observer=MovingAverageMinMaxObserver)
        if is_qat
        else MinMaxObserver
    )

    return QuantizationSpec(
        dtype=torch.int8,
        quant_min=-127,
        quant_max=127,
        qscheme=torch.per_tensor_symmetric,
        is_dynamic=False,
        observer_or_fake_quant_ctr=observer_or_fake_quant_ctr,
        ch_axis=0,
    )


def wgt_fc_qspec(is_qat: bool):
    observer_or_fake_quant_ctr = (
        FakeQuantize.with_args(observer=MovingAverageMinMaxObserver)
        if is_qat
        else MinMaxObserver
    )

    return QuantizationSpec(
        dtype=torch.int8,
        quant_min=-127,
        quant_max=127,
        qscheme=torch.per_tensor_symmetric,
        is_dynamic=False,
        observer_or_fake_quant_ctr=observer_or_fake_quant_ctr,
    )


# Is set by the *PatternQuantizer directly.
bias_qspec = None


class NeutronQuantizer(ComposableQuantizer):
    def __init__(self, neutron_target_spec: NeutronTargetSpec, is_qat: bool = False):
        self.neutron_target_spec = neutron_target_spec
        self.is_qat = is_qat

        static_qconfig = QuantizationConfig(
            act_qspec(is_qat=is_qat),
            act_qspec(is_qat=is_qat),
            wgt_qspec(is_qat=is_qat),
            None,
        )
        static_fc_qconfig = QuantizationConfig(
            act_qspec(is_qat=is_qat),
            act_qspec(is_qat=is_qat),
            wgt_fc_qspec(is_qat=is_qat),
            None,
        )

        OpQuantizer = NeutronAtenQuantizer
        super().__init__(
            [
                OpQuantizer(AbsPattern(is_qat=is_qat), static_qconfig),
                OpQuantizer(AdaptiveAvgPoolPattern(is_qat=is_qat), static_qconfig),
                OpQuantizer(AddTensorPattern(is_qat=is_qat), static_qconfig),
                OpQuantizer(AddmmPattern(self, is_qat=is_qat), static_fc_qconfig),
                OpQuantizer(AvgPoolPattern(is_qat=is_qat), static_qconfig),
                OpQuantizer(CatPattern(is_qat=is_qat), static_qconfig),
                OpQuantizer(Conv1dPattern(is_qat=is_qat), static_qconfig),
                OpQuantizer(Conv2dPattern(self, is_qat=is_qat), static_qconfig),
                OpQuantizer(ConvTranspose2dPattern(is_qat=is_qat), static_qconfig),
                OpQuantizer(DropoutPattern(is_qat=is_qat), static_qconfig),
                OpQuantizer(FlattenPattern(is_qat=is_qat), static_qconfig),
                OpQuantizer(HardTanhPattern(is_qat=is_qat), static_qconfig),
                OpQuantizer(HardTanhInPlacePattern(is_qat=is_qat), static_qconfig),
                OpQuantizer(LinearPattern(self, is_qat=is_qat), static_fc_qconfig),
                OpQuantizer(MaxPoolPattern(is_qat=is_qat), static_qconfig),
                OpQuantizer(MeanDimPattern(is_qat=is_qat), static_qconfig),
                OpQuantizer(MmPattern(self, is_qat=is_qat), static_qconfig),
                OpQuantizer(MulTensorPattern(is_qat=is_qat), static_qconfig),
                OpQuantizer(PadPattern(is_qat=is_qat), static_qconfig),
                OpQuantizer(PermutePattern(is_qat=is_qat), static_qconfig),
                OpQuantizer(ReluPattern(is_qat=is_qat), static_qconfig),
                OpQuantizer(ReluInPlacePattern(is_qat=is_qat), static_qconfig),
                OpQuantizer(ReshapePattern(is_qat=is_qat), static_qconfig),
                OpQuantizer(SigmoidPattern(is_qat=is_qat), static_qconfig),
                OpQuantizer(SliceTensorPattern(is_qat=is_qat), static_qconfig),
                OpQuantizer(SoftMaxPattern(is_qat=is_qat), static_qconfig),
                OpQuantizer(SubTensorPattern(is_qat=is_qat), static_qconfig),
                OpQuantizer(TanhPattern(is_qat=is_qat), static_qconfig),
                OpQuantizer(TanhInPlacePattern(is_qat=is_qat), static_qconfig),
                OpQuantizer(TransposeIntPattern(is_qat=is_qat), static_qconfig),
                OpQuantizer(ViewPattern(is_qat=is_qat), static_qconfig),
            ]
        )

        # Mapping ops defined in quantizer partition types to its quantizer
        self.op_to_quantizer = {
            pt: q for q in self.quantizers for pt in q.pattern.partition_types()
        }
        # Mapping ops to the quantizer application state
        self.op_to_applied_quantizer = {
            pt: False for q in self.quantizers for pt in q.pattern.partition_types()
        }
        self.cluster_quantizers = [
            NeutronAtenQuantizer(
                ActivationsConcatClusterPattern(self, is_qat=is_qat), static_qconfig
            )
        ]

    def transform_for_annotation(
        self, model: torch.fx.GraphModule
    ) -> torch.fx.GraphModule:
        model.graph.eliminate_dead_code()  # Remove dead code to simplify the graph for the passes.

        model = NeutronAtenPassManager(self.neutron_target_spec)(model).graph_module

        model.graph.eliminate_dead_code()  # Remove dead code again, in case it was created by the passes.

        return model

    def annotate(self, model: torch.fx.GraphModule) -> torch.fx.GraphModule:
        self._annotate_inputs(model)

        # Annotate node clusters in model
        for cluster_quantizer in self.cluster_quantizers:
            cluster_quantizer.annotate(model)

        nodes = list(model.graph.nodes)
        for node in nodes:
            if (
                node.target not in self.op_to_quantizer
                or self.op_to_applied_quantizer[node.target]
            ):
                continue
            else:
                quantizer = self.op_to_quantizer[node.target]
                quantizer.annotate(model)
                if not isinstance(quantizer.pattern, SharedSpecPattern):
                    self.op_to_applied_quantizer[node.target] = True

        return model

    def _is_input_annotated(self, node: fx.Node) -> bool:
        return (
            "quantization_annotation" in node.meta
            and node.meta["quantization_annotation"]._annotated
        )

    def _mark_input_node_as_annotated(self, node: fx.Node) -> None:
        if "quantization_annotation" not in node.meta:
            node.meta["quantization_annotation"] = QuantizationAnnotation()
        node.meta["quantization_annotation"]._annotated = True

    def _annotate_inputs(self, model: fx.GraphModule):
        for node in model.graph.nodes:
            if self._is_input_annotated(node):
                continue

            if node.op == "placeholder" and len(node.users) > 0:
                annotate_output_qspec(node, act_qspec(self.is_qat))
                self._mark_input_node_as_annotated(node)

    def validate(self, model: torch.fx.GraphModule) -> None:
        return super().validate(model)
