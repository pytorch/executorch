# Copyright 2024-2025 Arm Limited and/or its affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.
"""Provide quantization annotation logic for Arm backends.

This module computes per-node quantization properties and applies input/output
annotations to FX graphs using TorchAO qspecs.

"""

import logging
import operator
from dataclasses import dataclass
from typing import Callable, cast, List, Optional, Sequence

import torch
import torch.fx
from executorch.backends.arm.common.debug import get_node_debug_info
from executorch.backends.arm.common.type import ensure_type
from executorch.backends.arm.quantizer import QuantizationConfig
from torch._subclasses import FakeTensor

from torch.fx import Node
from torchao.quantization.pt2e.quantizer import (
    annotate_input_qspec_map,
    annotate_output_qspec,
    QuantizationSpecBase,
    SharedQuantizationSpec,
)

from .arm_quantizer_utils import (
    is_annotated,
    is_output_annotated,
    mark_node_as_annotated,
)

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class _QuantProperty:
    """Specify how the input/output at 'index' must be quantized."""

    index: int
    qspec: QuantizationSpecBase | List[QuantizationSpecBase]
    optional: bool = False
    mark_annotated: bool = False


class _OpQuantProperties:
    """Collect input/output quantization properties for a node.

    Attributes:
        quant_inputs (List[_QuantProperty]): Quantization specs for inputs
            indexed by argument positions.
        quant_output (Optional[_QuantProperty]): Quantization spec for the
            node's output when applicable.

    """

    def __init__(self):
        self.quant_inputs: List[_QuantProperty] = []
        self.quant_output: Optional[_QuantProperty] = None


def _as_list(x):
    """Return ``x`` wrapped as a list if needed.

    Args:
        x: Value or list of values.

    Returns:
        list: ``x`` if already a list; otherwise ``[x]``.

    """
    if isinstance(x, (list, tuple)):
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
    - The output of the node itself is of type `float32` and is not a large
      scalar.

    Args:
        node (Node): The node being analyzed.
        quant_properties (_OpQuantProperties): Contains quantization properties
            for the node, including input and output quantization specifications.
        gm (torch.fx.GraphModule): The graph module containing the computational
            graph.

    Returns:
        bool: `True` if the node can be quantized, otherwise `False`.

    """
    # Check output
    if quant_properties.quant_output is not None:
        if _is_non_float_tensor(node):
            logger.debug(
                "Could not quantize non float tensor for the following output node: "
                f"{get_node_debug_info(node, gm)}"
            )

            return False
        elif _is_large_scalar(node, gm):
            logger.debug(
                "Could not quantize large scalar node for the following output node: "
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

            if _is_non_float_tensor(n_arg):
                logger.debug(
                    "Could not quantize non float tensor for the following input "
                    f"node: {get_node_debug_info(node, gm)}"
                )

                return False
            elif _is_large_scalar(n_arg, gm):
                logger.debug(
                    "Could not quantize large scalar node for the following input "
                    f"node: {get_node_debug_info(node, gm)}"
                )

                return False

    return True


def _get_node_target(module: torch.nn.Module | torch.fx.GraphModule, target_str: str):
    """Get an attribute from a module by dotted path.

    Args:
        module (torch.nn.Module | torch.fx.GraphModule): Root module.
        target_str (str): Dotted attribute path, e.g., ``"sub.weight"``.

    Returns:
        Any: Resolved attribute on the module.

    """
    targets = target_str.split(".")
    for target in targets[:-1]:
        module = module.get_submodule(target)
    return getattr(module, targets[-1])


def _is_large_scalar(node: Node, gm: torch.fx.GraphModule):
    """Return True if input is a large scalar value.

    Large scalars are skipped because ``torch.histc`` supports values only up
    to a certain upper bound.

    """
    HISTC_UPPER_BOUND = 3.4028235e15
    if node.op == "get_attr" and isinstance(node.target, str):
        tensor = _get_node_target(gm, node.target)
        # torch.histc works until this upper bound
        return tensor.numel() == 1 and abs(tensor.item()) > HISTC_UPPER_BOUND
    if node.op == "call_function" and node.target in (
        torch.ops.aten.full.default,
        torch.ops.aten.full,
        torch.ops.aten.fill_.Scalar,
    ):
        fill_value = cast(float, node.args[1])
        return abs(fill_value) > HISTC_UPPER_BOUND
    return False


def _is_non_float_tensor(node: Node) -> bool:
    """Check if the output of a node has a data type other than `torch.float32`.

    If the output is not `torch.float32`, quantization cannot be performed, as
    observers only work with floating-point tensors.

    Args:
        node (Node): The node to check the output(s) for.

    Returns:
        bool: `True` if the data type is not float32, otherwise `False`.

    Note:
        - If `node.meta["val"]` is a `list`, the function returns `True` if
          any element is not an instance of `FakeTensor` or does not have
          `torch.float32` as its data type.
        - If node.meta["val"] is missing or is not an instance of `FakeTensor`,
          the function returns True.

    """
    if "val" in node.meta and isinstance(node.meta["val"], Sequence):
        return any(
            not isinstance(fake_tensor, FakeTensor)
            or fake_tensor.dtype != torch.float32
            for fake_tensor in node.meta["val"]
        )

    if "val" not in node.meta or not isinstance(node.meta["val"], FakeTensor):
        return True

    return node.meta["val"].dtype != torch.float32


def _annotate_input(node: Node, quant_property: _QuantProperty):
    """Annotate a node's input with the given qspec.

    Maps the specified input argument(s) to the provided quantization spec and
    optionally marks the input node(s) as annotated.

    Args:
        node (Node): Node whose input should be annotated.
        quant_property (_QuantProperty): Input index and qspec(s).

    Raises:
        RuntimeError: If the node is already annotated.
        TypeError: If an input argument is not a ``Node`` instance.

    """
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
    """Annotate a node's output with the given qspec.

    Args:
        node (Node): Node whose output should be annotated.
        quant_property (_QuantProperty): Output index and qspec.

    Raises:
        RuntimeError: If the node is already annotated.
        ValueError: If ``mark_annotated`` is True, ``optional`` is True, or
            ``index`` is not zero.

    """
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
    """Check whether a node chain matches a pattern.

    Verify a chain of ancestors -> node -> descendants matches the provided
    ``pattern``. If ``filter_fn`` is provided, require all nodes in the chain
    to pass the filter. Each pattern element is a list of disjunctive node
    targets.

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


_conv_ops = [
    torch.ops.aten.conv1d.default,
    torch.ops.aten.conv2d.default,
    torch.ops.aten.conv2d.padding,
    torch.ops.aten.conv3d.default,
    torch.ops.aten.conv3d.padding,
]

_one_to_one = [
    torch.ops.aten.abs.default,
    torch.ops.aten.ceil.default,
    torch.ops.aten.erf.default,
    torch.ops.aten.exp.default,
    torch.ops.aten.expm1.default,
    torch.ops.aten.elu.default,
    torch.ops.aten.floor.default,
    torch.ops.aten.log.default,
    torch.ops.aten.reciprocal.default,
    torch.ops.aten.rsqrt.default,
    torch.ops.aten.sigmoid.default,
    torch.ops.aten.cos.default,
    torch.ops.aten.sin.default,
    torch.ops.aten.tanh.default,
    torch.ops.aten.sum.dim_IntList,
    torch.ops.aten.sum.default,
    torch.ops.aten.hardsigmoid.default,
    torch.ops.aten.hardswish.default,
    torch.ops.aten.hardswish_.default,
    torch.ops.aten.full_like.default,
    torch.ops.aten.zeros_like.default,
    torch.ops.aten.pow.Tensor_Scalar,
    torch.ops.aten.gelu.default,
    torch.ops.aten.sinh.default,
    torch.ops.aten.atan.default,
    torch.ops.aten.acosh.default,
    torch.ops.aten.sign.default,
    torch.ops.aten.asin.default,
    torch.ops.aten.atanh.default,
    torch.ops.aten.asinh.default,
    torch.ops.aten.cosh.default,
    torch.ops.aten.acos.default,
    torch.ops.aten.cumsum.default,
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
    torch.ops.aten._unsafe_view.default,
    torch.ops.aten.select.int,
    torch.ops.aten.select_copy.int,
    torch.ops.aten.slice.Tensor,
    torch.ops.aten.slice_copy.Tensor,
    torch.ops.aten.split.Tensor,
    torch.ops.aten.split_with_sizes.default,
    torch.ops.aten.split_copy.Tensor,
    torch.ops.aten.transpose.Dimname,
    torch.ops.aten.transpose.int,
    torch.ops.aten.transpose_copy.int,
    torch.ops.aten.t_copy.default,
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
    # Neg operator flips the range, but keps the magnitude the same.
    # That is why we force it to use the same qparams and avoid
    # dequant -> neg -> requant chain.
    torch.ops.aten.neg.default,
]

_one_to_one_shared_input_or_input_act_qspec = [
    torch.ops.aten.alias.default,
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
    torch.ops.aten.flatten.using_ints,
    torch.ops.aten.dropout.default,
    torch.ops.aten.dropout_.default,
    torch.ops.aten.adaptive_avg_pool2d.default,
    torch.ops.aten.alias_copy.default,
    torch.ops.aten.pixel_shuffle.default,
    torch.ops.aten.pixel_unshuffle.default,
]


def get_quant_properties(  # noqa: C901
    node: Node, gm: torch.fx.GraphModule, quantization_config
) -> _OpQuantProperties | None:
    """Compute quantization properties for a node.

    Determine which inputs and/or outputs should be annotated for quantization
    based on the node's operator and surrounding pattern.

    Args:
        node (Node): Node to analyze.
        gm (torch.fx.GraphModule): Owning graph module.
        quantization_config: Source for activation/weight/bias qspecs.

    Returns:
        _OpQuantProperties | None: Properties to apply, or ``None`` if the
            node is unsupported or not suitable for quantization.

    """
    input_act_qspec = quantization_config.get_input_act_qspec()
    weight_qspec = quantization_config.get_weight_qspec()
    output_act_qspec = quantization_config.get_output_act_qspec()
    bias_qspec = quantization_config.get_bias_qspec(node)

    quant_properties = _OpQuantProperties()

    def any_or_hardtanh_min_zero(n: Node):
        """Return True for any op or hardtanh with ``min_val == 0``."""
        # Check that if the node is a hardtanh, its min_val is zero
        return (
            n.target
            not in (torch.ops.aten.hardtanh.default, torch.ops.aten.hardtanh_.default)
            or n.args[1] == 0
        )

    if _match_pattern(
        node,
        [
            _conv_ops,
            [torch.ops.aten.batch_norm.default],
            [
                torch.ops.aten.relu.default,
                torch.ops.aten.relu_.default,
                torch.ops.aten.hardtanh.default,
                torch.ops.aten.hardtanh_.default,
            ],
        ],
        filter_fn=any_or_hardtanh_min_zero,
    ):
        if node.target in _conv_ops:
            quant_properties.quant_inputs = [
                _QuantProperty(0, input_act_qspec),
                _QuantProperty(1, weight_qspec, mark_annotated=True),
                _QuantProperty(2, bias_qspec, optional=True, mark_annotated=True),
            ]
        elif node.target in (
            torch.ops.aten.relu.default,
            torch.ops.aten.relu_.default,
            torch.ops.aten.hardtanh.default,
            torch.ops.aten.hardtanh_.default,
        ):
            quant_properties.quant_output = _QuantProperty(0, output_act_qspec)

    elif _match_pattern(
        node,
        [
            _conv_ops,
            [torch.ops.aten.batch_norm.default],
        ],
    ):
        if node.target in _conv_ops:
            quant_properties.quant_inputs = [
                _QuantProperty(0, input_act_qspec),
                _QuantProperty(1, weight_qspec, mark_annotated=True),
                _QuantProperty(2, bias_qspec, optional=True, mark_annotated=True),
            ]
        elif node.target in [
            torch.ops.aten.batch_norm.default,
        ]:
            quant_properties.quant_output = _QuantProperty(0, output_act_qspec)
    elif _match_pattern(
        node,
        [
            [
                *_conv_ops,
                torch.ops.aten.linear.default,
            ],
            [
                torch.ops.aten.relu.default,
                torch.ops.aten.relu_.default,
                torch.ops.aten.hardtanh.default,
                torch.ops.aten.hardtanh_.default,
            ],
        ],
        any_or_hardtanh_min_zero,
    ):
        if node.target in (
            *_conv_ops,
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
        *_conv_ops,
        torch.ops.aten.linear.default,
    ):
        quant_properties.quant_inputs = [
            _QuantProperty(0, input_act_qspec),
            _QuantProperty(1, weight_qspec, mark_annotated=True),
            _QuantProperty(2, bias_qspec, optional=True, mark_annotated=True),
        ]
        quant_properties.quant_output = _QuantProperty(0, output_act_qspec)
    elif node.target in (
        torch.ops.aten.add.Tensor,
        torch.ops.aten.add_.Tensor,
        torch.ops.aten.sub.Tensor,
        torch.ops.aten.sub_.Tensor,
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
        torch.ops.aten.minimum.default,
        torch.ops.aten.maximum.default,
    ):
        lhs_node = ensure_type(Node, node.args[0])
        shared_qspec = SharedQuantizationSpec((lhs_node, node))
        quant_properties.quant_inputs = [
            _QuantProperty(0, input_act_qspec),
            _QuantProperty(
                1,
                input_act_qspec if node.args[0] == node.args[1] else shared_qspec,
            ),
        ]
        quant_properties.quant_output = _QuantProperty(0, shared_qspec)
    elif node.target in (torch.ops.aten.where.self,):
        true_node = ensure_type(Node, node.args[1])
        input_qspec = (
            SharedQuantizationSpec(true_node)
            if is_output_annotated(true_node)
            else input_act_qspec
        )
        quant_properties.quant_inputs = [
            _QuantProperty(1, input_qspec),
            _QuantProperty(2, SharedQuantizationSpec((true_node, node))),
        ]
        quant_properties.quant_output = _QuantProperty(
            0,
            SharedQuantizationSpec((true_node, node)),
        )
    elif node.target in _one_to_one_shared_input_or_input_act_qspec:
        input_node = ensure_type(Node, node.args[0])
        input_qspec = (
            SharedQuantizationSpec(input_node)
            if is_output_annotated(input_node)
            else input_act_qspec
        )
        quant_properties.quant_inputs = [_QuantProperty(0, input_qspec)]
        quant_properties.quant_output = _QuantProperty(
            0,
            SharedQuantizationSpec((input_node, node)),
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
        inputs = [ensure_type(Node, element) for element in node.args[0]]
        shared_qspec = SharedQuantizationSpec((inputs[0], node))
        quant_properties.quant_inputs = [
            _QuantProperty(
                0,
                [input_act_qspec if n == inputs[0] else shared_qspec for n in inputs],
            )
        ]
        quant_properties.quant_output = _QuantProperty(0, shared_qspec)
    elif node.target in _one_to_one:
        quant_properties.quant_inputs = [_QuantProperty(0, input_act_qspec)]
        quant_properties.quant_output = _QuantProperty(0, output_act_qspec)
    elif node.target in _one_to_one_shared_input_qspec:
        input_node = ensure_type(Node, node.args[0])
        quant_properties.quant_inputs = [_QuantProperty(0, input_act_qspec)]
        quant_properties.quant_output = _QuantProperty(
            0,
            SharedQuantizationSpec((input_node, node)),
        )
    elif node.target in [torch.ops.aten.copy_.default]:
        input_node = ensure_type(Node, node.args[1])
        quant_properties.quant_inputs = [
            _QuantProperty(0, input_act_qspec),
            _QuantProperty(1, input_act_qspec),
        ]
        quant_properties.quant_output = _QuantProperty(
            0,
            SharedQuantizationSpec((input_node, node)),
        )
    elif node.target in [
        torch.ops.aten.eq.Tensor,
        torch.ops.aten.ge.Tensor,
        torch.ops.aten.gt.Tensor,
        torch.ops.aten.le.Tensor,
        torch.ops.aten.lt.Tensor,
    ]:
        input_node = ensure_type(Node, node.args[0])
        shared_qspec = SharedQuantizationSpec((input_node, node))
        quant_properties.quant_inputs = [
            _QuantProperty(0, input_act_qspec),
            _QuantProperty(
                1,
                input_act_qspec if node.args[0] == node.args[1] else shared_qspec,
            ),
        ]
        quant_properties.quant_output = None
    elif node.target in [
        torch.ops.aten.full.default,
        torch.ops.aten.full,
        torch.ops.aten.zeros.default,
        torch.ops.aten.ones.default,
        torch.ops.aten.fill_.Scalar,
        torch.ops.aten.scalar_tensor.default,
    ]:
        quant_properties.quant_inputs = []
        quant_properties.quant_output = _QuantProperty(0, output_act_qspec)
    elif node.target in [operator.getitem]:
        input_node = ensure_type(Node, node.args[0])
        if not is_output_annotated(input_node):
            return None
        shared_qspec = SharedQuantizationSpec(input_node)
        quant_properties.quant_inputs = [_QuantProperty(0, shared_qspec)]
        quant_properties.quant_output = _QuantProperty(0, shared_qspec)
    elif node.target in (
        torch.ops.higher_order.cond,
        torch.ops.higher_order.while_loop,
    ):
        submodule_args_pos = -1 if node.target == torch.ops.higher_order.cond else -2
        submodule_args = node.args[submodule_args_pos]
        output_qspec = output_act_qspec
        if len(submodule_args) > 0:  # type: ignore[arg-type]
            # The way the TOSA backend handles quantized inputs, arrays of input tensors (such as the input to a
            # conditional graph) need shared quantization.
            shared_qspec = SharedQuantizationSpec(
                (cast(list[Node], submodule_args)[0], node)
            )
            quant_properties.quant_inputs = [
                _QuantProperty(
                    submodule_args_pos,
                    [
                        input_act_qspec,
                        *([shared_qspec] * (len(submodule_args) - 1)),  # type: ignore[arg-type]
                    ],
                )
            ]
            if node.target == torch.ops.higher_order.while_loop:
                # The output of the while loop body can either re-enter the body, or exit the while loop.
                # Therefore, A and B in the diagram below need to share the same quantization parameters.
                # A -> while ( RESCALE -> ... RESCALE -> ) -> B
                output_qspec = shared_qspec

        quant_properties.quant_output = _QuantProperty(0, output_qspec)

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
    """Annotate supported nodes in a graph with quantization specs.

    Iterate through call_function nodes, computes quantization properties, and
    apply input/output annotations. A filter can restrict which nodes are
    considered.

    Args:
        gm (torch.fx.GraphModule): Graph to annotate.
        quantization_config (QuantizationConfig): Default qspecs for nodes.
        filter_fn (Optional[Callable[[Node], bool]]): Optional node predicate.

    Returns:
        Optional[List[List[Node]]]: Reserved for future use; currently None.

    """
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
            torch.ops.aten.fill_.Scalar,
            torch.ops.aten.scalar_tensor.default,
            torch.ops.aten.zeros.default,
            torch.ops.aten.ones.default,
        ]:
            node.kwargs = {}
