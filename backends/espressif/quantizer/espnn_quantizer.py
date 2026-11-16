# Copyright 2026 Mwaniki Njagi.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

from typing import Callable, Optional, Union

import torch

from executorch.backends.xnnpack.quantizer.xnnpack_quantizer import (
    _get_module_type_filter,
    get_symmetric_quantization_config,
    XNNPACKQuantizer,
)
from executorch.backends.xnnpack.quantizer.xnnpack_quantizer_utils import (
    _is_share_obs_or_fq_op,
)
from torch.fx import Node
from torchao.quantization.pt2e.quantizer import (
    FixedQParamsQuantizationSpec,
    get_bias_qspec,
    get_input_act_qspec,
    get_module_name_filter,
    get_output_act_qspec,
    get_weight_qspec,
    QuantizationAnnotation,
    QuantizationConfig,
    SharedQuantizationSpec,
)
from torchao.quantization.pt2e.quantizer.quantizer import Q_ANNOTATION_KEY


CONV_TARGETS = (torch.ops.aten.conv2d.default,)
LINEAR_TARGETS = (torch.ops.aten.linear.default,)
ADD_TARGETS = (torch.ops.aten.add.Tensor, torch.ops.aten.add_.Tensor)
RELU_TARGETS = (torch.ops.aten.relu.default, torch.ops.aten.relu_.default)
HARDTANH_TARGETS = (torch.ops.aten.hardtanh.default, torch.ops.aten.hardtanh_.default)
RELU6_TARGETS = (torch.ops.aten.relu6.default,)
LUT_TARGETS = (
    torch.ops.aten.hardswish.default,
    torch.ops.aten.hardswish_.default,
    torch.ops.aten.hardsigmoid.default,
    torch.ops.aten.hardsigmoid_.default,
    torch.ops.aten.sigmoid.default,
    torch.ops.aten.sigmoid_.default,
    torch.ops.aten.silu.default,
    torch.ops.aten.silu_.default,
    torch.ops.aten.tanh.default,
    torch.ops.aten.tanh_.default,
    torch.ops.aten.gelu.default,
    torch.ops.aten.elu.default,
    torch.ops.aten.elu_.default,
    torch.ops.aten.leaky_relu.default,
    torch.ops.aten.leaky_relu_.default,
)
SOFTMAX_TARGETS = (torch.ops.aten.softmax.int, torch.ops.aten._softmax.default)
SOFTMAX_OUTPUT_SCALE = 1.0 / 256.0
SOFTMAX_OUTPUT_ZERO_POINT = -128


def fused_activation_kind(node: Node) -> Optional[str]:
    if node.op != "call_function":
        return None
    if node.target in RELU_TARGETS:
        return "relu"
    if node.target in RELU6_TARGETS:
        return "relu6"
    if node.target in HARDTANH_TARGETS:
        minimum = float(
            node.args[1] if len(node.args) > 1 else node.kwargs.get("min_val", -1.0)
        )
        maximum = float(
            node.args[2] if len(node.args) > 2 else node.kwargs.get("max_val", 1.0)
        )
        if minimum == 0.0 and maximum == 6.0:
            return "relu6"
    return None


def _is_annotated(node: Node) -> bool:
    annotation = node.meta.get(Q_ANNOTATION_KEY)
    return annotation is not None and annotation._annotated


def _mark_annotated(nodes) -> None:
    for node in nodes:
        if node is None:
            continue
        if Q_ANNOTATION_KEY not in node.meta:
            node.meta[Q_ANNOTATION_KEY] = QuantizationAnnotation()
        node.meta[Q_ANNOTATION_KEY]._annotated = True


def _annotate_weighted_activation(
    producer: Node,
    activation: Node,
    config: QuantizationConfig,
) -> bool:
    input_act = producer.args[0]
    weight = producer.args[1]
    bias = producer.args[2] if len(producer.args) > 2 else None
    if not isinstance(input_act, Node) or not isinstance(weight, Node):
        return False
    nodes = [producer, activation, weight]
    if isinstance(bias, Node):
        nodes.append(bias)
    if any(_is_annotated(node) for node in nodes):
        return False

    input_qspec_map = {
        input_act: get_input_act_qspec(config),
        weight: get_weight_qspec(config),
    }
    if isinstance(bias, Node):
        input_qspec_map[bias] = get_bias_qspec(config)

    producer.meta[Q_ANNOTATION_KEY] = QuantizationAnnotation(
        input_qspec_map=input_qspec_map,
        _annotated=True,
    )
    activation.meta[Q_ANNOTATION_KEY] = QuantizationAnnotation(
        output_qspec=get_output_act_qspec(config),
        _annotated=True,
    )
    _mark_annotated(nodes)
    return True


def _annotate_add_activation(
    add: Node,
    activation: Node,
    config: QuantizationConfig,
) -> bool:
    lhs = add.args[0]
    rhs = add.args[1]
    if not isinstance(lhs, Node) or not isinstance(rhs, Node):
        return False
    if _is_annotated(add) or _is_annotated(activation):
        return False
    add.meta[Q_ANNOTATION_KEY] = QuantizationAnnotation(
        input_qspec_map={
            lhs: get_input_act_qspec(config),
            rhs: get_input_act_qspec(config),
        },
        _annotated=True,
    )
    activation.meta[Q_ANNOTATION_KEY] = QuantizationAnnotation(
        output_qspec=get_output_act_qspec(config),
        _annotated=True,
    )
    return True


ConfigSource = Union[
    QuantizationConfig, Callable[[Node], Optional[QuantizationConfig]], None
]


def _config_lookup(
    config: ConfigSource,
) -> Callable[[Node], Optional[QuantizationConfig]]:
    if callable(config) and not isinstance(config, QuantizationConfig):
        return config
    return lambda node: config


def annotate_fused_activations(
    model: torch.fx.GraphModule,
    config: ConfigSource,
) -> int:
    """
    Annotate conv/linear/add followed by ReLU or ReLU6 as one fused unit.
    `config` is a QuantizationConfig or a per-node lookup returning one, or
    None for nodes that must stay in float.
    """
    config_for = _config_lookup(config)
    fused = 0
    for node in model.graph.nodes:
        if fused_activation_kind(node) is None:
            continue
        producer = node.args[0]
        if not isinstance(producer, Node) or producer.op != "call_function":
            continue
        if len(producer.users) != 1:
            continue
        config = config_for(node)
        if config is None or config_for(producer) is not config:
            continue
        if producer.target in CONV_TARGETS or producer.target in LINEAR_TARGETS:
            fused += int(_annotate_weighted_activation(producer, node, config))
        elif producer.target in ADD_TARGETS:
            fused += int(_annotate_add_activation(producer, node, config))
    return fused


class EspNNQuantizer(XNNPACKQuantizer):
    """
    PT2E quantizer matching the ESP-NN int8 kernel contract.

    Activations: int8 per-tensor affine.
    Weights: int8 per-output-channel symmetric.
    ReLU and ReLU6 that directly follow conv2d, linear, or add are fused
    into the producer so that the clamp happens on the producer's
    requantized output, exactly as the ESP-NN activation_min/max arguments
    implement it. Without this fusion ReLU6 would inherit the producer's
    full-range scale and waste most of the int8 grid.
    """

    def __init__(self, per_channel: bool = True, is_qat: bool = False):
        super().__init__()
        self.set_global(
            get_symmetric_quantization_config(is_per_channel=per_channel, is_qat=is_qat)
        )
        self.fused_activation_count = 0
        self.rewritten_split_count = 0

    def node_config(self, node: Node) -> Optional[QuantizationConfig]:
        """
        The config that governs `node`, with the same precedence as
        XNNPACKQuantizer: module name, then module type, then global.
        None means the node stays in float.
        """
        for name, config in self.module_name_config.items():
            if get_module_name_filter(name)(node):
                return config
        for module_type, config in self.module_type_config.items():
            if _get_module_type_filter(module_type)(node):
                return config
        return self.global_config

    def transform_for_annotation(
        self, model: torch.fx.GraphModule
    ) -> torch.fx.GraphModule:
        model = super().transform_for_annotation(model)
        self.rewritten_split_count = rewrite_splits_as_slices(model)
        return model

    def annotate(self, model: torch.fx.GraphModule) -> torch.fx.GraphModule:
        self.fused_activation_count = annotate_fused_activations(
            model, self.node_config
        )
        model = self._annotate_for_quantization_config(model)
        annotate_unary_and_softmax(model, self.node_config)
        propagate_shared_quantization(model, self.node_config)
        return model


def annotate_unary_and_softmax(
    model: torch.fx.GraphModule, config: ConfigSource
) -> None:
    config_for = _config_lookup(config)
    softmax_output = FixedQParamsQuantizationSpec(
        dtype=torch.int8,
        scale=SOFTMAX_OUTPUT_SCALE,
        zero_point=SOFTMAX_OUTPUT_ZERO_POINT,
        quant_min=-128,
        quant_max=127,
        qscheme=torch.per_tensor_affine,
    )
    for node in model.graph.nodes:
        if node.op != "call_function" or _is_annotated(node):
            continue
        if node.target not in LUT_TARGETS and node.target not in SOFTMAX_TARGETS:
            continue
        source = node.args[0]
        config = config_for(node)
        if not isinstance(source, Node) or config is None:
            continue
        node.meta[Q_ANNOTATION_KEY] = QuantizationAnnotation(
            input_qspec_map={source: get_input_act_qspec(config)},
            output_qspec=(
                softmax_output
                if node.target in SOFTMAX_TARGETS
                else get_output_act_qspec(config)
            ),
            _annotated=True,
        )


EXTRA_SHARED_QPARAM_TARGETS = (
    torch.ops.aten.avg_pool2d.default,
    torch.ops.aten.relu6.default,
    torch.ops.aten.unsqueeze.default,
    torch.ops.aten.squeeze.default,
    torch.ops.aten.squeeze.dims,
    torch.ops.aten.slice.Tensor,
    torch.ops.aten.select.int,
    torch.ops.aten.expand.default,
    torch.ops.aten.unflatten.int,
    torch.ops.aten._unsafe_view.default,
    torch.ops.aten.clone.default,
    torch.ops.aten.contiguous.default,
    torch.ops.aten.upsample_nearest2d.vec,
    torch.ops.aten.upsample_nearest2d.default,
    torch.ops.aten.upsample_bilinear2d.vec,
    torch.ops.aten.upsample_bilinear2d.default,
    torch.ops.aten.constant_pad_nd.default,
    torch.ops.aten.pad.default,
    torch.ops.aten.dropout.default,
    torch.ops.aten.dropout_.default,
    torch.ops.aten.alias.default,
)

SPLIT_TARGETS = (
    torch.ops.aten.split.Tensor,
    torch.ops.aten.split_with_sizes.default,
    torch.ops.aten.chunk.default,
    torch.ops.aten.unbind.int,
)


def rewrite_splits_as_slices(model: torch.fx.GraphModule) -> int:
    """
    Replace split/chunk/unbind + getitem with slice/select so that every
    piece is an ordinary tensor that can carry a shared quantization spec.
    """
    import operator

    rewritten = 0
    for node in list(model.graph.nodes):
        if node.op != "call_function" or node.target not in SPLIT_TARGETS:
            continue
        source = node.args[0]
        value = source.meta.get("val") if isinstance(source, Node) else None
        if value is None:
            continue
        users = list(node.users)
        if any(user.target is not operator.getitem for user in users):
            continue
        rank = value.dim()
        if node.target is torch.ops.aten.unbind.int:
            dim = int(node.args[1]) if len(node.args) > 1 else 0
            dim %= rank
            bounds = None
        else:
            dim = (
                int(node.args[2])
                if len(node.args) > 2
                else int(node.kwargs.get("dim", 0))
            )
            dim %= rank
            size = int(value.shape[dim])
            if node.target is torch.ops.aten.chunk.default:
                chunks = int(node.args[1])
                step = (size + chunks - 1) // chunks
                sizes = [min(step, size - start) for start in range(0, size, step)]
            elif isinstance(node.args[1], (list, tuple)):
                sizes = [int(v) for v in node.args[1]]
            else:
                step = int(node.args[1])
                sizes = [min(step, size - start) for start in range(0, size, step)]
            starts = [sum(sizes[:i]) for i in range(len(sizes))]
            bounds = [(start, start + length) for start, length in zip(starts, sizes)]
        for user in users:
            index = int(user.args[1])
            with model.graph.inserting_before(user):
                if bounds is None:
                    replacement = model.graph.call_function(
                        torch.ops.aten.select.int, (source, dim, index)
                    )
                else:
                    start, end = bounds[index]
                    replacement = model.graph.call_function(
                        torch.ops.aten.slice.Tensor, (source, dim, start, end, 1)
                    )
            replacement.meta = dict(user.meta)
            user.replace_all_uses_with(replacement)
            model.graph.erase_node(user)
        model.graph.erase_node(node)
        rewritten += 1
    if rewritten:
        model.graph.lint()
        model.recompile()
    return rewritten


_ALWAYS = object()


def _shares_qparams(node: Node) -> bool:
    return node.op == "call_function" and (
        _is_share_obs_or_fq_op(node.target)
        or node.target in EXTRA_SHARED_QPARAM_TARGETS
    )


def propagate_shared_quantization(
    model: torch.fx.GraphModule, config: ConfigSource = _ALWAYS
) -> None:
    """
    Give shape and movement ops their input's quantization. With a per-node
    `config` lookup, nodes whose config is None are left in float.
    """
    config_for = _config_lookup(config)
    for node in model.graph.nodes:
        if not _shares_qparams(node):
            continue
        if _is_annotated(node) or config_for(node) is None:
            continue
        source = node.args[0]
        if not isinstance(source, Node):
            continue
        annotation = source.meta.get(Q_ANNOTATION_KEY)
        if annotation is None or annotation.output_qspec is None:
            continue
        shared = SharedQuantizationSpec(source)
        node.meta[Q_ANNOTATION_KEY] = QuantizationAnnotation(
            input_qspec_map={source: shared},
            output_qspec=shared,
            _annotated=True,
        )


__all__ = [
    "EspNNQuantizer",
    "annotate_fused_activations",
    "fused_activation_kind",
    "propagate_shared_quantization",
    "annotate_unary_and_softmax",
    "rewrite_splits_as_slices",
    "SOFTMAX_OUTPUT_SCALE",
    "SOFTMAX_OUTPUT_ZERO_POINT",
]
