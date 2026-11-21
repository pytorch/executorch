# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from dataclasses import replace

import torch
from executorch.backends.transforms.permute_pass_utils import get_arg, set_arg
from executorch.backends.xnnpack.quantizer.xnnpack_quantizer import (
    get_symmetric_quantization_config,
)
from executorch.frontends.fused_quant.quantizer.quantizer import (
    ConvolutionQuantizer,
    FusedQuantQuantizer,
    get_core_aten_quantizers,
    OpQuantizer,
)
from torchao.quantization.pt2e.quantizer import (
    DerivedQuantizationSpec,
    FixedQParamsQuantizationSpec,
    SharedQuantizationSpec,
)
from torchao.quantization.pt2e.quantizer.quantizer import Q_ANNOTATION_KEY
from torchao.quantization.pt2e.quantizer.utils import (
    annotate_input_qspec_map,
    annotate_output_qspec,
)


def _derive_bias_qparams(observers):
    input_scale, _ = observers[0].calculate_qparams()
    weight_scale, _ = observers[1].calculate_qparams()
    return input_scale * weight_scale, torch.zeros_like(weight_scale, dtype=torch.int32)


class _LinearQuantizer(OpQuantizer):
    def __init__(self, config):
        super().__init__(
            torch.ops.aten.linear.default,
            torch.ops.fused_quant.linear.default,
            config,
            weight_names=("weight",),
            other_names=("bias",),
        )

    def annotate_node(self, node: torch.fx.Node) -> None:
        # FQ fusion requires constant per-channel weight tensors.
        if get_arg(node, "weight", torch.fx.Node).op != "get_attr":
            return
        super().annotate_node(node)
        _annotate_bias(node)


class _ConvolutionQuantizer(ConvolutionQuantizer):
    def annotate_node(self, node: torch.fx.Node) -> None:
        super().annotate_node(node)
        shape = get_arg(node, "input", torch.fx.Node).meta["val"].shape
        if (
            not get_arg(node, "transposed")
            and len(shape) == 4
            and get_arg(node, "groups") in (1, shape[1])
        ):
            _annotate_bias(node)


def _annotate_bias(node: torch.fx.Node) -> None:
    bias = get_arg(node, "bias")
    if (
        not isinstance(bias, torch.fx.Node)
        or bias.op != "get_attr"
        or bias.meta["val"].dim() != 1
    ):
        return
    if len(bias.users) > 1:
        # PT2E's implicit sharing ignores DerivedQuantizationSpec inputs.
        with node.graph.inserting_before(node):
            bias_copy = node.graph.call_function(torch.ops.aten.clone.default, (bias,))
        bias_copy.meta = bias.meta.copy()
        bias_copy.meta.pop(Q_ANNOTATION_KEY, None)
        set_arg(node, "bias", bias_copy)
        bias = bias_copy
    annotate_input_qspec_map(
        node,
        bias,
        DerivedQuantizationSpec(
            derived_from=[
                (get_arg(node, "input"), node),
                (get_arg(node, "weight"), node),
            ],
            derive_qparams_fn=_derive_bias_qparams,
            dtype=torch.int32,
            quant_min=-(1 << 31),
            # INT32_MAX rounds to 2**31 in PT2E's float32 clamp before the cast.
            quant_max=(1 << 31) - 128,
            qscheme=torch.per_channel_symmetric,
            ch_axis=0,
        ),
    )


class _AvgPoolQuantizer(OpQuantizer):
    def annotate_node(self, node: torch.fx.Node) -> None:
        super().annotate_node(node)
        annotate_output_qspec(node, SharedQuantizationSpec((node.args[0], node)))


class _EspFusedQuantizer(FusedQuantQuantizer):
    """Fused Quant quantizer for the Espressif operator-library backend.

    Uses int8 per-tensor affine activations and symmetric weights.
    Use EspNNQuantizer for the EspNNBackend delegate instead.
    """

    def __init__(self) -> None:
        config = get_symmetric_quantization_config(is_per_channel=False)
        weighted_config = get_symmetric_quantization_config(is_per_channel=True)
        softmax_config = replace(
            config,
            output_activation=FixedQParamsQuantizationSpec(
                dtype=torch.int8,
                scale=1 / 256,
                zero_point=-128,
                quant_min=-128,
                quant_max=127,
                qscheme=torch.per_tensor_affine,
            ),
        )
        super().__init__(
            [
                OpQuantizer(
                    torch.ops.aten._softmax.default,
                    torch.ops.fused_quant._softmax.default,
                    softmax_config,
                ),
                OpQuantizer(
                    torch.ops.aten.hardswish.default,
                    torch.ops.fused_quant.hardswish.default,
                    config,
                ),
                OpQuantizer(
                    torch.ops.aten.hardsigmoid.default,
                    torch.ops.fused_quant.hardsigmoid.default,
                    config,
                ),
                OpQuantizer(
                    torch.ops.aten.silu.default,
                    torch.ops.fused_quant.silu.default,
                    config,
                ),
                _LinearQuantizer(weighted_config),
                _ConvolutionQuantizer(
                    weighted_config=weighted_config, grouped_transposed_config=config
                ),
                _AvgPoolQuantizer(
                    torch.ops.aten.avg_pool2d.default,
                    torch.ops.fused_quant.avg_pool2d.default,
                    config,
                    node_filter=lambda node: get_arg(node, "divisor_override") is None,
                ),
                *get_core_aten_quantizers(config, weighted_config),
            ]
        )

    def annotate(self, model: torch.fx.GraphModule) -> torch.fx.GraphModule:
        groups = []
        for activation in model.graph.nodes:
            if activation.target not in (
                torch.ops.aten.relu.default,
                torch.ops.aten.hardtanh.default,
            ):
                continue
            producer = activation.args[0]
            if (
                producer.target
                not in (
                    torch.ops.aten.add.Tensor,
                    torch.ops.aten.sub.Tensor,
                    torch.ops.aten.mul.Tensor,
                    torch.ops.aten.linear.default,
                    torch.ops.aten.convolution.default,
                )
                or len(producer.users) != 1
                or Q_ANNOTATION_KEY in producer.meta
                or Q_ANNOTATION_KEY in activation.meta
            ):
                continue
            if producer.target in (
                torch.ops.aten.add.Tensor,
                torch.ops.aten.sub.Tensor,
                torch.ops.aten.mul.Tensor,
            ) and (
                not isinstance(get_arg(producer, "other"), torch.fx.Node)
                or (
                    producer.target != torch.ops.aten.mul.Tensor
                    and get_arg(producer, "alpha") != 1
                )
            ):
                continue
            groups.append((producer, activation))

        super().annotate(model)
        for producer, activation in groups:
            producer_annotation = producer.meta.get(Q_ANNOTATION_KEY)
            activation_annotation = activation.meta.get(Q_ANNOTATION_KEY)
            if producer_annotation is not None and activation_annotation is not None:
                # Observe the pattern's output, without rounding its internal edge.
                producer_annotation.output_qspec = None
                activation_annotation.input_qspec_map[producer] = None
        return model


def get_esp_fused_quant_quantizer() -> FusedQuantQuantizer:
    """Create the quantizer for the Espressif Fused Quant operator-library flow."""
    return _EspFusedQuantizer()
