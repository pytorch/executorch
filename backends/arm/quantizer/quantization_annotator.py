# Copyright 2024-2025 Arm Limited and/or its affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import operator
from dataclasses import dataclass
from typing import Callable, List, Optional

import torch
import torch.fx
from executorch.backends.arm.quantizer import arm_quantizer_utils
from executorch.backends.arm.quantizer.quantization_config import QuantizationConfig
from torch.ao.quantization.quantizer import QuantizationSpecBase, SharedQuantizationSpec
from torch.ao.quantization.quantizer.utils import (
    _annotate_input_qspec_map,
    _annotate_output_qspec,
)
from torch.fx import Node


@dataclass(frozen=True)
class _QuantProperty:
    """Specify how the input/output at 'index' must be quantized."""

    index: int
    qspec: type[QuantizationSpecBase] | List[type[QuantizationSpecBase]]
    optional: bool = False
    mark_annotated: bool = False


class _OpQuantProperties:
    def __init__(self):
        self.quant_inputs: List[_QuantProperty] = []
        self.quant_output: Optional[_QuantProperty] = None


def _as_list(x):
    if isinstance(x, list):
        return x
    else:
        return [
            x,
        ]


def _is_ok_for_quantization(
    node: Node, quant_property: _QuantProperty, gm: torch.fx.GraphModule
) -> bool:
    if quant_property.optional and (
        quant_property.index >= len(node.args)
        or node.args[quant_property.index] is None
    ):
        return True

    for n_arg in _as_list(node.args[quant_property.index]):
        assert isinstance(n_arg, Node)
        if not arm_quantizer_utils.is_ok_for_quantization(n_arg, gm):  # type: ignore[attr-defined]
            return False

    return True


def _annotate_input(node: Node, quant_property: _QuantProperty):
    assert not arm_quantizer_utils.is_annotated(node)
    if quant_property.optional and (
        quant_property.index >= len(node.args)
        or node.args[quant_property.index] is None
    ):
        return

    for n_arg, qspec in zip(
        _as_list(node.args[quant_property.index]),
        _as_list(quant_property.qspec),
        strict=True,
    ):
        assert isinstance(n_arg, Node)
        _annotate_input_qspec_map(node, n_arg, qspec)
        if quant_property.mark_annotated:
            arm_quantizer_utils.mark_node_as_annotated(n_arg)  # type: ignore[attr-defined]


def _annotate_output(node: Node, quant_property: _QuantProperty):
    assert not arm_quantizer_utils.is_annotated(node)
    assert not quant_property.mark_annotated
    assert not quant_property.optional
    assert quant_property.index == 0, "Only one output annotation supported currently"

    _annotate_output_qspec(node, quant_property.qspec)


def _match_pattern(
    node: Node, pattern: List[List], filter_fn: Optional[Callable[[Node], bool]] = None
) -> bool:
    """
    Check if there's a chain of node.ancestors? -> node -> node.descendant? that matches the
    chain provided in 'pattern'. If 'filter_fn' is provided, check that all the nodes in the
    chain pass the filtering.

    Each 'pattern' element is composed of a list of disjunctive nodes types.
    """
    assert len(pattern) == 2, "Only two-nodes patterns supported currently"

    if node.target in pattern[0]:
        assert len(node.users) != 0
        parent = node
        child = next(iter(node.users))
    elif node.target in pattern[1]:
        assert len(node.args) != 0
        parent = node.args[0]  # type: ignore[assignment]
        child = node
    else:
        return False

    if len(parent.users) != 1:
        return False

    if parent.target not in pattern[0] or child.target not in pattern[1]:
        return False

    if filter_fn is not None:
        return filter_fn(parent) and filter_fn(child)

    return True


_one_to_one = [
    torch.ops.aten.exp.default,
    torch.ops.aten.log.default,
    torch.ops.aten.reciprocal.default,
    torch.ops.aten.rsqrt.default,
    torch.ops.aten.sigmoid.default,
    torch.ops.aten.tanh.default,
    torch.ops.aten.sum.dim_IntList,
    torch.ops.aten.hardsigmoid.default,
    torch.ops.aten.hardswish.default,
]

_one_to_one_shared_input_qspec = [
    torch.ops.aten.squeeze.default,
    torch.ops.aten.squeeze_copy.default,
    torch.ops.aten.squeeze_copy.dim,
    torch.ops.aten.squeeze.dim,
    torch.ops.aten.squeeze.dims,
    torch.ops.aten.unsqueeze.default,
    torch.ops.aten.unsqueeze_copy.default,
    torch.ops.aten.reshape.default,
    torch.ops.aten.repeat.default,
    torch.ops.aten.expand_copy.default,
    torch.ops.aten.expand.default,
    # Disabling these as there seems to be an issue with support for complex
    # datatypes in torch:
    # torch.ops.aten.view_as_complex.default,
    # torch.ops.aten.view_as_complex_copy.default,
    # torch.ops.aten.view_as_real.default,
    # torch.ops.aten.view_as_real_copy.default,
    torch.ops.aten.view.default,
    torch.ops.aten.view_as.default,
    torch.ops.aten.view_copy.default,
    torch.ops.aten.select.int,
    torch.ops.aten.select_copy.int,
    torch.ops.aten.slice.Tensor,
    torch.ops.aten.slice_copy.Tensor,
    torch.ops.aten.split.Tensor,
    torch.ops.aten.split_with_sizes.default,
    torch.ops.aten.transpose.Dimname,
    torch.ops.aten.transpose.int,
    torch.ops.aten.transpose_copy.int,
    torch.ops.aten.tile.default,
    torch.ops.aten.flip.default,
    torch.ops.aten.chunk.default,
    torch.ops.aten.contiguous.default,
    torch.ops.aten.upsample_nearest2d.vec,
]

# Operators that can inherit the quantization specs from its parent node
# as SharedQuantizationSpec.
_parent_shared_qspec = [
    torch.ops.aten.hardtanh.default,
    torch.ops.aten.hardtanh_.default,
    torch.ops.aten.relu.default,
    torch.ops.aten.mean.default,
    torch.ops.aten.mean.dim,
    torch.ops.aten.permute.default,
    torch.ops.aten.permute_copy.default,
    torch.ops.aten.avg_pool2d.default,
    torch.ops.aten.max_pool2d.default,
    torch.ops.aten.full.default,
    torch.ops.aten.flatten.using_ints,
    torch.ops.aten.dropout.default,
    torch.ops.aten.clamp.default,
    torch.ops.aten.clamp.Tensor,
    operator.getitem,
]


def get_quant_properties(  # noqa: C901
    node: Node, gm: torch.fx.GraphModule, quantization_config
) -> _OpQuantProperties:
    input_act_qspec = quantization_config.get_input_act_qspec()
    weight_qspec = quantization_config.get_weight_qspec()
    output_act_qspec = quantization_config.get_output_act_qspec()
    bias_qspec = quantization_config.get_bias_qspec(node)

    quant_properties = _OpQuantProperties()

    def any_or_hardtanh_min_zero(n: Node):
        # Check that if the node is a hardtanh, its min_val is zero
        return n.target != torch.ops.aten.hardtanh.default or n.args[1] == 0

    if _match_pattern(
        node,
        [
            [
                torch.ops.aten.conv1d.default,
                torch.ops.aten.conv2d.default,
                torch.ops.aten.linear.default,
            ],
            [torch.ops.aten.relu.default, torch.ops.aten.hardtanh.default],
        ],
        any_or_hardtanh_min_zero,
    ):
        if node.target in (
            torch.ops.aten.conv1d.default,
            torch.ops.aten.conv2d.default,
            torch.ops.aten.linear.default,
        ):
            quant_properties.quant_inputs = [
                _QuantProperty(0, input_act_qspec),
                _QuantProperty(1, weight_qspec, mark_annotated=True),
                _QuantProperty(2, bias_qspec, optional=True, mark_annotated=True),
            ]
        else:
            quant_properties.quant_output = _QuantProperty(0, output_act_qspec)
    elif node.target in (
        torch.ops.aten.conv1d.default,
        torch.ops.aten.conv2d.default,
        torch.ops.aten.linear.default,
    ):
        quant_properties.quant_inputs = [
            _QuantProperty(0, input_act_qspec),
            _QuantProperty(1, weight_qspec, mark_annotated=True),
            _QuantProperty(2, bias_qspec, optional=True, mark_annotated=True),
        ]
        quant_properties.quant_output = _QuantProperty(0, output_act_qspec)
    elif node.target in (
        torch.ops.aten.matmul.default,
        torch.ops.aten.mm.default,
        torch.ops.aten.bmm.default,
        torch.ops.aten.mul.Tensor,
        torch.ops.aten.mul_.Tensor,
    ):
        quant_properties.quant_inputs = [
            _QuantProperty(0, input_act_qspec),
            _QuantProperty(1, input_act_qspec),
        ]
        quant_properties.quant_output = _QuantProperty(0, output_act_qspec)
    elif node.target in (
        torch.ops.aten.add.Tensor,
        torch.ops.aten.add_.Tensor,
        torch.ops.aten.sub.Tensor,
        torch.ops.aten.sub_.Tensor,
        torch.ops.aten.minimum.default,
        torch.ops.aten.maximum.default,
    ):
        shared_qspec = SharedQuantizationSpec((node.args[0], node))  # type: ignore[arg-type]
        quant_properties.quant_inputs = [
            _QuantProperty(0, input_act_qspec),
            _QuantProperty(
                1, input_act_qspec if node.args[0] == node.args[1] else shared_qspec  # type: ignore[arg-type]
            ),
        ]
        quant_properties.quant_output = _QuantProperty(0, shared_qspec)  # type: ignore[arg-type]
    elif node.target == torch.ops.aten.adaptive_avg_pool2d.default:
        input_qspec = (
            SharedQuantizationSpec(node.args[0])  # type: ignore[arg-type]
            if arm_quantizer_utils.is_output_annotated(node.args[0])  # type: ignore
            else input_act_qspec
        )
        quant_properties.quant_inputs = [_QuantProperty(0, input_qspec)]  # type: ignore[arg-type]
        quant_properties.quant_output = _QuantProperty(
            0, SharedQuantizationSpec((node.args[0], node))  # type: ignore[arg-type]
        )
    elif node.target in (
        torch.ops.aten.cat.default,
        torch.ops.aten.concatenate.default,
        torch.ops.aten.stack.default,
    ):
        assert isinstance(node.args[0], list)
        assert len(node.args[0]) != 0

        shared_qspec = SharedQuantizationSpec((node.args[0][0], node))
        quant_properties.quant_inputs = [
            _QuantProperty(
                0,
                [
                    input_act_qspec if n == node.args[0][0] else shared_qspec  # type: ignore[misc]
                    for n in node.args[0]
                ],
            )
        ]
        quant_properties.quant_output = _QuantProperty(0, shared_qspec)  # type: ignore[arg-type]
    elif node.target in _one_to_one:
        quant_properties.quant_inputs = [_QuantProperty(0, input_act_qspec)]
        quant_properties.quant_output = _QuantProperty(0, output_act_qspec)
    elif node.target in _one_to_one_shared_input_qspec:
        quant_properties.quant_inputs = [_QuantProperty(0, input_act_qspec)]
        quant_properties.quant_output = _QuantProperty(
            0, SharedQuantizationSpec((node.args[0], node))  # type: ignore[arg-type]
        )
    elif node.target in [
        torch.ops.aten.eq.Tensor,
        torch.ops.aten.ge.Tensor,
        torch.ops.aten.gt.Tensor,
        torch.ops.aten.le.Tensor,
        torch.ops.aten.lt.Tensor,
    ]:
        shared_qspec = SharedQuantizationSpec((node.args[0], node))  # type: ignore[arg-type]
        quant_properties.quant_inputs = [
            _QuantProperty(0, input_act_qspec),
            _QuantProperty(
                1, input_act_qspec if node.args[0] == node.args[1] else shared_qspec  # type: ignore[arg-type]
            ),
        ]
        quant_properties.quant_output = None
    elif node.target in _parent_shared_qspec:
        if not isinstance(node.args[0], Node):
            return None  # type: ignore[return-value]

        if not arm_quantizer_utils.is_output_annotated(node.args[0]):  # type: ignore[attr-defined]
            return None  # type: ignore[return-value]

        shared_qspec = SharedQuantizationSpec(node.args[0])
        quant_properties.quant_inputs = [_QuantProperty(0, shared_qspec)]  # type: ignore[arg-type]
        quant_properties.quant_output = _QuantProperty(0, shared_qspec)  # type: ignore[arg-type]
    else:
        return None  # type: ignore[return-value]

    # Don't check if operator.getitem is ok for quantization, it's always ok
    if node.target == operator.getitem:
        return quant_properties

    # Check that each inputs/outputs can be quantized properly with the
    # provided QuantProperties
    for quant_property in quant_properties.quant_inputs:
        if not _is_ok_for_quantization(node, quant_property, gm):
            return None  # type: ignore[return-value]

    if quant_properties.quant_output is not None:
        if not _is_ok_for_quantization(node, quant_properties.quant_output, gm):
            return None  # type: ignore[return-value]

    return quant_properties


def annotate_graph(  # type: ignore[return]
    gm: torch.fx.GraphModule,
    quantization_config: QuantizationConfig,
    filter_fn: Optional[Callable[[Node], bool]] = None,
) -> Optional[List[List[Node]]]:
    for node in gm.graph.nodes:
        if node.op != "call_function":
            continue

        if arm_quantizer_utils.is_annotated(node):
            continue

        if filter_fn is not None and not filter_fn(node):
            continue

        quant_properties = get_quant_properties(node, gm, quantization_config)
        if quant_properties is None:
            continue

        for quant_property in quant_properties.quant_inputs:
            _annotate_input(node, quant_property)

        if quant_properties.quant_output is not None:
            _annotate_output(node, quant_properties.quant_output)

        arm_quantizer_utils.mark_node_as_annotated(node)  # type: ignore[attr-defined]
