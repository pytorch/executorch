# Copyright 2024-2025 Arm Limited and/or its affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import logging
import operator
from dataclasses import dataclass
from typing import Callable, List, Optional

import torch
import torch.fx
import torch.nn.functional as F
from executorch.backends.arm.quantizer import QuantizationConfig
from executorch.backends.arm.tosa_utils import get_node_debug_info

from torch.fx import Node
from torchao.quantization.pt2e.quantizer import (
    annotate_input_qspec_map,
    annotate_output_qspec,
    QuantizationSpecBase,
    SharedQuantizationSpec,
)

from .arm_quantizer_utils import (
    is_annotated,
    is_ok_for_quantization,
    is_output_annotated,
    mark_node_as_annotated,
)

logger = logging.getLogger(__name__)


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
    node: Node, quant_properties: _OpQuantProperties, gm: torch.fx.GraphModule
) -> bool:
    """Check if a node can be quantized.

    A node can be quantized if:
    - All inputs that are required for quantization are of type `float32`
      and are not large scalar values.
    - The output of the node itself is of type `float32` and is not a large scalar.

    Args:
        node (Node): The node being analyzed.
        quant_properties (_OpQuantProperties): Contains quantization properties for
            the node, including input and output quantization specifications.
        gm (torch.fx.GraphModule): The graph module containing the computational graph.

    Returns:
        bool: `True` if the node can be quantized, otherwise `False`.
    """
    # Check output
    if quant_properties.quant_output is not None:
        if not is_ok_for_quantization(node, gm):  # type: ignore[attr-defined]
            logger.debug(
                f"Could not quantize node due to output: "
                f"{get_node_debug_info(node, gm)}"
            )

            return False

    # Check inputs
    for quant_property in quant_properties.quant_inputs:
        if quant_property.optional and (
            quant_property.index >= len(node.args)
            or node.args[quant_property.index] is None
        ):
            continue

        for n_arg in _as_list(node.args[quant_property.index]):
            if not isinstance(n_arg, Node):
                raise TypeError(
                    f"n_arg must be a Node instance, got {type(n_arg).__name__!r}"
                )
            if not is_ok_for_quantization(n_arg, gm):  # type: ignore[attr-defined]
                logger.debug(
                    f'could not quantize node due to input "{node}": '
                    f"{get_node_debug_info(node, gm)}"
                )

                return False

    return True


def _annotate_input(node: Node, quant_property: _QuantProperty):
    if is_annotated(node):
        raise RuntimeError(
            f"Cannot annotate input: node '{node.name}' is already annotated"
        )
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
        if not isinstance(n_arg, Node):
            raise TypeError(
                f"n_arg must be a Node instance, got {type(n_arg).__name__!r}"
            )
        annotate_input_qspec_map(node, n_arg, qspec)
        if quant_property.mark_annotated:
            mark_node_as_annotated(n_arg)  # type: ignore[attr-defined]


def _annotate_output(node: Node, quant_property: _QuantProperty):
    if is_annotated(node):
        raise RuntimeError(
            f"Cannot annotate output: node '{node.name}' is already annotated"
        )
    if quant_property.mark_annotated:
        raise ValueError(
            "quant_property.mark_annotated must be False for output annotation"
        )
    if quant_property.optional:
        raise ValueError("quant_property.optional must be False for output annotation")
    if quant_property.index != 0:
        raise ValueError("Only one output annotation supported currently")

    annotate_output_qspec(node, quant_property.qspec)


def _match_pattern(
    node: Node, pattern: List[List], filter_fn: Optional[Callable[[Node], bool]] = None
) -> bool:
    """
    Check if there's a chain of node.ancestors? -> node -> node.descendant? that matches the
    chain provided in 'pattern'. If 'filter_fn' is provided, check that all the nodes in the
    chain pass the filtering.

    Each 'pattern' element is composed of a list of disjunctive nodes types.
    """
    if len(pattern) < 1:
        raise ValueError("No pattern provided")

    if filter_fn is not None:
        if not filter_fn(node):
            return False
    if len(pattern) == 1:
        # Base case where it has passed the filter_fn. Simply look if node.target is in pattern.
        return node.target in pattern[0]
    if node.target not in [op for sub_pattern in pattern for op in sub_pattern]:
        # node.target not in pattern. No need to look at the rest of the pattern.
        return False
    # Find the index of this node's target in pattern
    idx = [node.target in sub_pattern for sub_pattern in pattern].index(True)
    left_pattern = pattern[:idx]
    # Exclude idx as this contains node.target which we have already matched
    right_pattern = pattern[idx + 1 :]
    left_condition = True
    right_condition = True
    # Recursively look at the rest of the pattern by calling this function for
    # node's input and user node with updated patterns.
    if len(left_pattern) > 0:
        parent = node.all_input_nodes[0]
        if len(parent.users) != 1:
            return False
        left_condition = _match_pattern(parent, left_pattern, filter_fn)
    if len(right_pattern) > 0:
        right_condition = _match_pattern(list(node.users)[0], right_pattern, filter_fn)
    return left_condition and right_condition


_one_to_one = [
    torch.ops.aten.abs.default,
    torch.ops.aten.ceil.default,
    torch.ops.aten.erf.default,
    torch.ops.aten.exp.default,
    torch.ops.aten.floor.default,
    torch.ops.aten.log.default,
    torch.ops.aten.reciprocal.default,
    torch.ops.aten.rsqrt.default,
    torch.ops.aten.sigmoid.default,
    torch.ops.aten.cos.default,
    torch.ops.aten.sin.default,
    torch.ops.aten.tanh.default,
    torch.ops.aten.sum.dim_IntList,
    torch.ops.aten.hardsigmoid.default,
    torch.ops.aten.hardswish.default,
    torch.ops.aten.hardswish_.default,
    torch.ops.aten.full_like.default,
    torch.ops.aten.pow.Tensor_Scalar,
    torch.ops.aten.gelu.default,
    torch.ops.aten.sinh.default,
    torch.ops.aten.atan.default,
    torch.ops.aten.acosh.default,
    torch.ops.aten.sign.default,
    torch.ops.aten.asin.default,
    torch.ops.aten.atanh.default,
]

_one_to_one_shared_input_qspec = [
    torch.ops.aten.squeeze.default,
    torch.ops.aten.squeeze_copy.default,
    torch.ops.aten.squeeze_copy.dim,
    torch.ops.aten.squeeze.dim,
    torch.ops.aten.squeeze.dims,
    torch.ops.aten.unbind.int,
    torch.ops.aten.unsqueeze.default,
    torch.ops.aten.unsqueeze_copy.default,
    torch.ops.aten.reshape.default,
    torch.ops.aten.repeat.default,
    torch.ops.aten.repeat_interleave.self_int,
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
    torch.ops.aten.upsample_bilinear2d.vec,
    torch.ops.aten.upsample_nearest2d.vec,
    torch.ops.aten.pad.default,
    torch.ops.aten.amax.default,
    torch.ops.aten.amin.default,
    torch.ops.aten.clamp.default,
    torch.ops.aten.clamp.Tensor,
    torch.ops.aten.unflatten.int,
    torch.ops.aten.index_select.default,
    torch.ops.aten.index.Tensor,
]

_one_to_one_shared_input_or_input_act_qspec = [
    torch.ops.aten.clone.default,
    torch.ops.aten.hardtanh.default,
    torch.ops.aten.hardtanh_.default,
    torch.ops.aten.relu.default,
    torch.ops.aten.relu_.default,
    torch.ops.aten.mean.default,
    torch.ops.aten.mean.dim,
    torch.ops.aten.permute.default,
    torch.ops.aten.permute_copy.default,
    torch.ops.aten.avg_pool2d.default,
    torch.ops.aten.max_pool2d.default,
    torch.ops.aten.full.default,
    torch.ops.aten.full,
    torch.ops.aten.flatten.using_ints,
    torch.ops.aten.dropout.default,
    torch.ops.aten.dropout_.default,
    torch.ops.aten.adaptive_avg_pool2d.default,
    torch.ops.aten.alias_copy.default,
]


def get_quant_properties(  # noqa: C901
    node: Node, gm: torch.fx.GraphModule, quantization_config
) -> _OpQuantProperties | None:
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
                torch.ops.aten.conv2d.padding,
            ],
            [torch.ops.aten.batch_norm.default, F.batch_norm],
            [torch.ops.aten.relu.default, torch.ops.aten.hardtanh.default],
        ],
        filter_fn=any_or_hardtanh_min_zero,
    ):
        if node.target in (
            torch.ops.aten.conv1d.default,
            torch.ops.aten.conv2d.default,
            torch.ops.aten.conv2d.padding,
        ):
            quant_properties.quant_inputs = [
                _QuantProperty(0, input_act_qspec),
                _QuantProperty(1, weight_qspec, mark_annotated=True),
                _QuantProperty(2, bias_qspec, optional=True, mark_annotated=True),
            ]
        elif node.target in (
            torch.ops.aten.relu.default,
            torch.ops.aten.hardtanh.default,
        ):
            quant_properties.quant_output = _QuantProperty(0, output_act_qspec)

    elif _match_pattern(
        node,
        [
            [
                torch.ops.aten.conv1d.default,
                torch.ops.aten.conv2d.default,
                torch.ops.aten.conv2d.padding,
            ],
            [torch.ops.aten.batch_norm.default, F.batch_norm],
        ],
    ):
        if node.target in (
            torch.ops.aten.conv1d.default,
            torch.ops.aten.conv2d.default,
            torch.ops.aten.conv2d.padding,
        ):
            quant_properties.quant_inputs = [
                _QuantProperty(0, input_act_qspec),
                _QuantProperty(1, weight_qspec, mark_annotated=True),
                _QuantProperty(2, bias_qspec, optional=True, mark_annotated=True),
            ]
        elif node.target in [torch.ops.aten.batch_norm.default, F.batch_norm]:
            quant_properties.quant_output = _QuantProperty(0, output_act_qspec)
    elif _match_pattern(
        node,
        [
            [
                torch.ops.aten.conv1d.default,
                torch.ops.aten.conv2d.default,
                torch.ops.aten.linear.default,
                torch.ops.aten.conv2d.padding,
            ],
            [torch.ops.aten.relu.default, torch.ops.aten.hardtanh.default],
        ],
        any_or_hardtanh_min_zero,
    ):
        if node.target in (
            torch.ops.aten.conv1d.default,
            torch.ops.aten.conv2d.default,
            torch.ops.aten.linear.default,
            torch.ops.aten.conv2d.padding,
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
        torch.ops.aten.conv2d.padding,
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
    elif node.target in (torch.ops.aten.where.self,):
        shared_qspec = SharedQuantizationSpec(node.args[1])  # type: ignore[arg-type]
        quant_properties.quant_inputs = [
            _QuantProperty(1, shared_qspec),  # type: ignore[arg-type]
            _QuantProperty(2, shared_qspec),  # type: ignore[arg-type]
        ]
        quant_properties.quant_output = _QuantProperty(0, shared_qspec)  # type: ignore[arg-type]
    elif node.target in _one_to_one_shared_input_or_input_act_qspec:
        if not isinstance(node.args[0], Node):
            return None

        input_qspec = (
            SharedQuantizationSpec(node.args[0])  # type: ignore[arg-type]
            if is_output_annotated(node.args[0])  # type: ignore
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
        # first argument should be a non-empty list of nodes
        if not isinstance(node.args[0], list):
            raise TypeError(
                "Expected node.args[0] to be a list, got "
                f"{type(node.args[0]).__name__!r}"
            )
        if len(node.args[0]) == 0:
            raise ValueError("Expected non-empty list for node.args[0]")

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
    elif node.target in (torch.ops.aten.neg.default,):
        quant_properties.quant_inputs = [_QuantProperty(0, input_act_qspec)]
        quant_properties.quant_output = _QuantProperty(0, input_act_qspec)
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
    elif node.target in [torch.ops.aten.scalar_tensor.default]:
        quant_properties.quant_inputs = []
        quant_properties.quant_output = _QuantProperty(0, output_act_qspec)
    elif node.target in [operator.getitem]:
        if not is_output_annotated(node.args[0]):  # type: ignore[attr-defined, arg-type]
            return None
        shared_qspec = SharedQuantizationSpec(node.args[0])  # type: ignore[arg-type]
        quant_properties.quant_inputs = [_QuantProperty(0, shared_qspec)]  # type: ignore[arg-type]
        quant_properties.quant_output = _QuantProperty(0, shared_qspec)  # type: ignore[arg-type]
    else:
        return None

    # Don't check if operator.getitem is ok for quantization, it's always ok
    if node.target == operator.getitem:
        return quant_properties

    # Check that each inputs/outputs can be quantized properly with the
    # provided quantization properties.
    if not _is_ok_for_quantization(node, quant_properties, gm):
        return None

    return quant_properties


def annotate_graph(  # type: ignore[return]
    gm: torch.fx.GraphModule,
    quantization_config: QuantizationConfig,
    filter_fn: Optional[Callable[[Node], bool]] = None,
) -> Optional[List[List[Node]]]:
    for node in gm.graph.nodes:
        if node.op != "call_function":
            continue

        if is_annotated(node):
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

        mark_node_as_annotated(node)  # type: ignore[attr-defined]

        # Quantization does not allow kwargs for some reason.
        # Remove from ops we know have and where we know it does not break anything.
        if node.target in [
            torch.ops.aten.full_like.default,
            torch.ops.aten.full.default,
            torch.ops.aten.full,
            torch.ops.aten.scalar_tensor.default,
        ]:
            node.kwargs = {}
