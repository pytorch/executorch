# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from math import frexp, isclose, trunc
from typing import Any, Callable, Dict, List, Optional, Tuple, Type

import torch
from executorch.exir.pass_base import ExportPass

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
from torch.fx import GraphModule
from torch.fx.passes.infra.pass_base import PassResult
from torch.fx.passes.utils.fuser_utils import legalize_graph


def quantize_tensor_multiplier(
    requantize_scale_tensor: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Given requantize_scale_tensor with values in the interval (0, 1),
    produce a pair of tensors (out_multiplier, right_shift) where out_multiplier
    is an int32 tensor representing fixed-point values in the interval [-1, 1),
    and right_shift is an amount to shift right by, so that the floating-point
    multiplication of some int32 input with each value of requantize_scale_tensor:
        result = int32_value * requantize_scale_tensors[i]
    is best approximated by the integer-arithmetic-only code:
        result = RoundingRightShift(FixedPointMultiplication(int32_value,
                                    out_multiplier[i]), right_shift[i])
    """
    # This is identical to C++11 std::round(). The general python round rounds
    # down, and C++ rounds away from zero.
    def round_away_zero(f) -> int:
        r = -0.5 if (f < 0) else 0.5
        return trunc(f + r)

    def quantize_scalar_multiplier(requantize_scale: float) -> Tuple[int, int]:
        significand, exponent = frexp(requantize_scale)
        significand_q31 = int(round_away_zero(significand * (1 << 31)))
        # Handle the special case when the real multiplier was so close to 1
        # that its fixed-point approximation was indistinguishable from 1.
        # We handle this by dividing it by two, incrementing exponent by 1.
        # the right shift amount.
        if significand_q31 == (1 << 31):
            significand_q31 //= 2
            exponent += 1

        # Verify that the decomposition of requantize_scale into significand
        # and exponent is correct.
        reconstructed = significand_q31 / (1 << 31) * pow(2, exponent)
        assert isclose(
            requantize_scale, reconstructed, rel_tol=1e-4, abs_tol=1e-4
        ), "computation of significand and exponent from requantize_scale is not accurate"

        return (significand_q31, exponent)

    # Flatten the input scale tensor so that we can operate on individual values
    orig_shape = requantize_scale_tensor.shape
    flattened_tensor = requantize_scale_tensor.flatten().to(torch.float32)
    out_multiplier = torch.zeros(flattened_tensor.shape, dtype=torch.int32)
    right_shift = torch.zeros(flattened_tensor.shape, dtype=torch.int32)

    # Iterate over the flattened scale tensor and compute the decomposition of
    # each value in scale tensor into significand(out_multiplier) and
    # exponent(right_shift)
    for idx, scale in enumerate(flattened_tensor):
        (si, ex) = quantize_scalar_multiplier(scale)
        out_multiplier[idx], right_shift[idx] = si, ex

    # Reshape the tensors back to the original shape
    out_multiplier = out_multiplier.reshape(orig_shape)
    right_shift = right_shift.reshape(orig_shape)

    return (out_multiplier, right_shift)


def _is_annotated(nodes: List[fx.Node]) -> bool:
    annotated = False
    for node in nodes:
        annotated = annotated or (
            "quantization_annotation" in node.meta
            and node.meta["quantization_annotation"]._annotated
        )
    return annotated


def _no_outside_users(fused_partition) -> bool:
    """
    Checks if each partition other than the last does not have any outside users.
    """
    for source_partition in fused_partition[:-1]:
        if len(source_partition.output_nodes) != 1:
            return False
        if len(source_partition.output_nodes[0].users) != 1:
            return False
    return True


# Helper function to get the weight node for both quantized and unquantized weights
# TODO(matthiascremon): get a better test!
def get_weight_node(weights_inputs: fx.Node, dequants_weights: fx.Node) -> fx.Node:
    """
    Returns the weight node.
    """
    weight_node = (
        weights_inputs
        if weights_inputs.name.endswith("_frozen_param")
        else dequants_weights
    )
    return weight_node


# Helper function to get the args and kwargs for the linear replacement op
def get_args_and_kwargs_linear(
    graph_module: GraphModule,
    inputs_inputs: List[fx.Node],
    dequants_inputs: List[fx.Node],
    other_inputs: List[fx.Node],
    weights_inputs: List[fx.Node],
    dequants_weights: List[fx.Node],
    bias_inputs: List[fx.Node],
    quant_node: fx.Node,
) -> Tuple[Tuple[Any], Dict[str, Any]]:
    """
    Returns the args and kwargs for the linear replacement op.
    """
    weight_scale = get_weight_node(weights_inputs[0], dequants_weights[0]).args[1]
    # pyre-fixme[58]: Unsupported operand types
    bias_scale = inputs_inputs[0].args[1] * weight_scale
    requantize_scale = bias_scale / quant_node.args[1]
    requantize_scale_t = torch.tensor([requantize_scale])

    (out_multiplier, out_shift) = quantize_tensor_multiplier(requantize_scale_t)
    bias_shape = weights_inputs
    weight_node = get_weight_node(weights_inputs[0], dequants_weights[0]).args[0]
    # pyre-fixme[16]: Undefined attribute
    attr_node = getattr(graph_module, weight_node.target)
    weight_shape = list(attr_node.shape)
    bias_shape = weight_shape[0]
    bias = (
        bias_inputs[0]
        if bias_inputs
        else graph_module.graph.call_function(
            torch.ops.aten.full.default, ([bias_shape], 0.0)
        )
    )
    bias_int32_quant = graph_module.graph.call_function(
        torch.ops.quantized_decomposed.quantize_per_tensor.default,
        (
            bias,
            bias_scale,
            0,
            -(2**31),
            (2**31) - 1,
            torch.int32,
        ),
    )

    # Create single element tensors for weight_zero_point, out_multiplier, out_shift.
    # Note that the function expects int32_t, when it would default to int64_t, so
    # we explicitly require that type.
    weight_zero_point_ = graph_module.graph.call_function(
        torch.ops.aten.full.default,
        ([1], dequants_weights[0].args[2]),
        {"dtype": torch.int32},
    )
    out_multiplier_ = graph_module.graph.call_function(
        torch.ops.aten.full.default,
        ([1], out_multiplier[0].item()),
        {"dtype": torch.int32},
    )
    out_shift_ = graph_module.graph.call_function(
        torch.ops.aten.full.default,
        ([1], out_shift[0].item()),
        {"dtype": torch.int32},
    )

    args = tuple(inputs_inputs + weights_inputs + other_inputs + [bias_int32_quant])
    kwargs = {
        "src_zero_point": dequants_inputs[0].args[2],
        "weight_zero_point": weight_zero_point_,
        "out_multiplier": out_multiplier_,
        "out_shift": out_shift_,
        "out_zero_point": quant_node.args[2],
        "offset": None,
    }
    return args, kwargs


def get_args_and_kwargs_conv1d(
    graph_module: GraphModule,
    inputs_inputs: List[fx.Node],
    dequants_inputs: List[fx.Node],
    other_inputs: List[fx.Node],
    weights_inputs: List[fx.Node],
    dequants_weights: List[fx.Node],
    bias_inputs: List[fx.Node],
    quant_node: fx.Node,
    op_node: fx.Node,
):
    weight_scale = get_weight_node(weights_inputs[0], dequants_weights[0]).args[1]
    weight_zero_point = get_weight_node(weights_inputs[0], dequants_weights[0]).args[2]
    # pyre-fixme[58]: Unsupported operand types
    bias_scale = inputs_inputs[0].args[1] * weight_scale
    # pyre-fixme[16]: Undefined attribute
    stride = [1, 1] if len(op_node.args) < 4 else [1, op_node.args[3][0]]
    # pyre-fixme[16]: Undefined attribute
    padding = [0, 0] if len(op_node.args) < 5 else [0, op_node.args[4][0]]
    # pyre-fixme[16]: Undefined attribute
    dilation = [1, 1] if len(op_node.args) < 6 else [1, op_node.args[5][0]]
    groups = 1 if len(op_node.args) < 7 else op_node.args[6]
    weight_node = get_weight_node(weights_inputs[0], dequants_weights[0]).args[0]
    # pyre-fixme[16]: Undefined attribute
    attr_node = getattr(graph_module, weight_node.target)
    weight_shape = list(attr_node.shape)
    bias_shape = weight_shape[0]

    # If the bias is empty, create a tensor of zeros with the appropriate shape
    bias = (
        bias_inputs[0]
        if bias_inputs
        else graph_module.graph.call_function(
            torch.ops.aten.full.default, ([bias_shape], 0.0)
        )
    )

    # The bias is quantized to int32_t
    bias_int32_quant = graph_module.graph.call_function(
        torch.ops.quantized_decomposed.quantize_per_tensor.default,
        (
            bias,
            bias_scale,
            0,
            -(2**31),
            (2**31) - 1,
            torch.int32,
        ),
    )

    # Compute the out multiplier and out shift. They are used when the conv op is
    # replaced by quantized linear, we compute them a priori for simplicity but
    # may revisit the decision.
    requantize_scale = bias_scale / quant_node.args[1]
    requantize_scale_t = torch.tensor([requantize_scale])

    (out_multiplier, out_shift) = quantize_tensor_multiplier(requantize_scale_t)

    out_multiplier_ = graph_module.graph.call_function(
        torch.ops.aten.full.default,
        ([1], out_multiplier[0].item()),
        {"dtype": torch.int32},
    )
    out_shift_ = graph_module.graph.call_function(
        torch.ops.aten.full.default,
        ([1], out_shift[0].item()),
        {"dtype": torch.int32},
    )

    # Create a single element tensor for the weight zero point
    weight_zero_point_tensor = graph_module.graph.call_function(
        torch.ops.aten.full.default,
        ([1], weight_zero_point),
        {"dtype": torch.int32},
    )

    # Create a single element tensor for the bias scale
    bias_scale_tensor = graph_module.graph.call_function(
        torch.ops.aten.full.default,
        ([1], bias_scale),
        {"dtype": torch.float32},
    )

    # Make the args and kwargs for the replacement op
    args = tuple(inputs_inputs + weights_inputs + other_inputs + [bias_int32_quant])
    kwargs = {
        "stride": stride,
        "padding": padding,
        "dilation": dilation,
        "groups": groups,
        "input_zero_point": dequants_inputs[0].args[2],
        "weight_zero_point": weight_zero_point_tensor,
        "bias_scale": bias_scale_tensor,
        "out_scale": quant_node.args[1],
        "out_zero_point": quant_node.args[2],
        "out_multiplier": out_multiplier_,
        "out_shift": out_shift_,
        "channel_last": False,
    }
    return args, kwargs


@dataclass
class PartitionAnchors:
    """
    All fields except output are lists of (node, args_index) pair, where node is from
    the given partition and node.args[args_index] is an input to the partition. Assumes
    a single output.

    Quantizer uses inputs, weights and biases for quantization annotation. The others
    field contains tensor inputs that aren't quantized, and the literals fields contains
    is used for other types of input values as well as handling default parameters.
    """

    inputs: List[Tuple[fx.Node, int]] = field(default_factory=list)
    weights: List[Tuple[fx.Node, int]] = field(default_factory=list)
    biases: List[Tuple[fx.Node, int]] = field(default_factory=list)
    others: List[Tuple[fx.Node, int]] = field(default_factory=list)
    literals: List[Tuple[fx.Node, int]] = field(default_factory=list)
    output: Optional[fx.Node] = None


class QuantizationPattern(ABC):
    @abstractmethod
    def partition_types(self) -> List[Any]:
        """
        List of types to be passed to find_sequential_partitions.
        """
        pass

    @abstractmethod
    def get_anchors(self, gm, fused_partition) -> Optional[PartitionAnchors]:
        pass

    @abstractmethod
    def replacement_op(self) -> Callable[..., Any]:
        """
        Operator (most likely a custom one) that this partition should be fused into in
        the backend. Refer to the QuantFusion pass for examples.
        """
        pass


class LinearPattern(QuantizationPattern):
    def partition_types(self) -> List[Type[torch.nn.Module]]:
        return [torch.nn.Linear]

    def get_anchors(
        self, gm: GraphModule, fused_partition: List[GraphModule]
    ) -> PartitionAnchors:
        linear_node = fused_partition[0].nodes[-1]

        # Keep bias empty if not supplied
        bias = []
        if len(linear_node.args) > 2:
            bias = [(linear_node, 2)]

        return PartitionAnchors(
            inputs=[(linear_node, 0)],
            weights=[(linear_node, 1)],
            biases=bias,
            output=linear_node,
        )

    def replacement_op(self):
        return torch.ops.xtensa.quantized_linear.default


class Conv1dPattern(QuantizationPattern):
    def partition_types(self) -> List[Type[torch.nn.Module]]:
        return [torch.nn.Conv1d]

    def get_anchors(
        self, gm: GraphModule, fused_partition: List[GraphModule]
    ) -> PartitionAnchors:
        conv1d_node = fused_partition[0].nodes[-1]

        # If bias is None, replace it with an empty list.
        bias = (
            [(conv1d_node, 2)]
            if len(conv1d_node.args) > 2 and conv1d_node.args[2]
            else []
        )

        return PartitionAnchors(
            inputs=[(conv1d_node, 0)],
            weights=[(conv1d_node, 1)],
            biases=bias,
            output=conv1d_node,
        )

    def replacement_op(self):
        return torch.ops.jarvis.quantized_conv.default


class GenericQuantizer(Quantizer):
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
            if not _no_outside_users(fused_partition):
                continue

            anchors = self.pattern.get_anchors(model, fused_partition)
            if not anchors:
                continue
            if _is_annotated(
                [x[0] for x in anchors.inputs + anchors.weights + anchors.biases]
                + [anchors.output]
            ):
                continue

            anchors.output.meta["quantization_annotation"] = QuantizationAnnotation(
                output_qspec=output_act_qspec,
                _annotated=True,
            )

            def annotate_inputs(inputs, spec):
                for node, idx in inputs:
                    annotation = node.meta.get(
                        "quantization_annotation",
                        QuantizationAnnotation(_annotated=True),
                    )
                    annotation.input_qspec_map[node.args[idx]] = spec
                    node.meta["quantization_annotation"] = annotation

            annotate_inputs(anchors.inputs, input_act_qspec)
            annotate_inputs(anchors.weights, weight_qspec)
            annotate_inputs(anchors.biases, bias_qspec)

    def validate(self, model: fx.GraphModule) -> None:
        pass

    @classmethod
    def get_supported_operators(cls) -> List[OperatorConfig]:
        return []


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


class JarvisQuantizer(ComposableQuantizer):
    def __init__(self):
        static_qconfig = QuantizationConfig(
            act_qspec,
            act_qspec,
            wgt_qspec,
            None,
        )
        super().__init__(
            [
                GenericQuantizer(LinearPattern(), static_qconfig),
                GenericQuantizer(Conv1dPattern(), static_qconfig),
            ]
        )


class QuantFusion(ExportPass):
    def __init__(self, patterns):
        super().__init__()
        self.patterns = patterns

    def call(self, graph_module: fx.GraphModule) -> PassResult:
        for pattern in self.patterns:
            fused_partitions = find_sequential_partitions(
                graph_module,
                pattern.partition_types(),
            )
            for fused_partition in fused_partitions:
                anchors = pattern.get_anchors(graph_module, fused_partition)
                if not anchors:
                    continue
                if any(self.is_fused(p.nodes) for p in fused_partition):
                    continue

                for p in fused_partition:
                    self.mark_fused(p.nodes)

                dequants_inputs = []
                for node, idx in anchors.inputs:
                    if (
                        node.args[idx].target
                        == torch.ops.quantized_decomposed.dequantize_per_tensor.default
                    ):
                        dequants_inputs.append(node.args[idx])
                dequants_weights = []
                for node, idx in anchors.weights:
                    if (
                        node.args[idx].target
                        == torch.ops.quantized_decomposed.dequantize_per_tensor.default
                    ):
                        dequants_weights.append(node.args[idx])

                inputs_inputs = [node.args[0] for node in dequants_inputs]
                weights_inputs = [node.args[0] for node in dequants_weights]
                bias_inputs = [node.args[idx] for node, idx in anchors.biases]
                other_inputs = [node.args[idx] for node, idx in anchors.others]

                assert len(anchors.output.users) == 1
                quant_node = list(anchors.output.users.keys())[0]

                op_node = anchors.output

                with graph_module.graph.inserting_after(anchors.output):
                    args = tuple(
                        inputs_inputs + weights_inputs + other_inputs + bias_inputs
                    )
                    kwargs = {}
                    if isinstance(pattern, LinearPattern):
                        args, kwargs = get_args_and_kwargs_linear(
                            graph_module,
                            inputs_inputs,
                            dequants_inputs,
                            other_inputs,
                            weights_inputs,
                            dequants_weights,
                            bias_inputs,
                            quant_node,
                        )
                    elif isinstance(pattern, Conv1dPattern):
                        args, kwargs = get_args_and_kwargs_conv1d(
                            graph_module,
                            inputs_inputs,
                            dequants_inputs,
                            other_inputs,
                            weights_inputs,
                            dequants_weights,
                            bias_inputs,
                            quant_node,
                            op_node,
                        )
                    fused = graph_module.graph.call_function(
                        pattern.replacement_op(),
                        args,
                        kwargs,
                    )
                    fused.meta = quant_node.meta
                    quant_node.replace_all_uses_with(fused)

            legalize_graph(graph_module)
            graph_module.graph.eliminate_dead_code()
            # pyre-fixme[7]: Incompatible return type
            graph_module.recompile()

    @classmethod
    def is_fused(cls, nodes) -> bool:
        return any(cls.__qualname__ in n.meta for n in nodes)

    @classmethod
    def mark_fused(cls, nodes) -> bool:
        for n in nodes:
            # pyre-fixme[7]: Incompatible return type
            n.meta["QuantFusion"] = True
