# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

from typing import Any, Dict, List, Tuple

import torch
from executorch.backends.cadence.aot.quantizer.patterns import (
    AddmmPattern,
    BmmPattern,
    Conv1dPattern,
    Conv2dPattern,
    LayerNormPattern,
    LinearPattern,
    MatmulPattern,
    ReluPattern0,
    ReluPattern1,
)
from executorch.backends.cadence.aot.quantizer.utils import (
    create_zero_bias_int32,
    find_sequential_partitions_aten,
    get_conv_args,
    quantize_tensor_multiplier,
)
from executorch.exir.pass_base import ExportPass
from torch import fx
from torch.fx import GraphModule
from torch.fx.passes.infra.pass_base import PassResult
from torch.fx.passes.utils.fuser_utils import legalize_graph


# Use this to avoid pyre errors
# pyre-ignore[33]: `_ModelInputsType` cannot alias to `Any`.
ArgsType = Any

# Use this part for patterns with multiple aten ops
ReluPatterns = (ReluPattern0, ReluPattern1)


# Helper function to get the args and kwargs for the linear replacement op
def get_args_and_kwargs_linear(
    graph_module: GraphModule,
    inputs_inputs: List[fx.Node],
    dequants_inputs: List[fx.Node],
    weights_inputs: List[fx.Node],
    dequants_weights: List[fx.Node],
    bias_inputs: List[fx.Node],
    quant_node: fx.Node,
) -> Tuple[Tuple[ArgsType], Dict[str, ArgsType]]:
    """
    Returns the args and kwargs for the linear replacement op.
    """
    weight_scale = dequants_weights[0].args[1]
    # pyre-fixme[58]: Unsupported operand types
    bias_scale = dequants_inputs[0].args[1] * weight_scale
    requantize_scale = bias_scale / quant_node.args[1]
    requantize_scale_t = torch.tensor([requantize_scale])

    (out_multiplier, out_shift) = quantize_tensor_multiplier(requantize_scale_t)

    # If bias is not available, create a bias tensor with the shape of weight[0]
    if not bias_inputs:
        weight_node = dequants_weights[0].args[0]
        assert isinstance(weight_node, fx.Node)
        bias = create_zero_bias_int32(graph_module, weight_node, bias_scale)
    else:
        bias = bias_inputs[0]

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

    args = tuple(inputs_inputs + weights_inputs + [bias])
    kwargs = {
        "src_zero_point": dequants_inputs[0].args[2],
        "weight_zero_point": weight_zero_point_,
        "out_multiplier": out_multiplier_,
        "out_shift": out_shift_,
        "out_zero_point": quant_node.args[2],
        "offset": None,
    }
    return args, kwargs


# Helper function to get the args and kwargs for the layer norm replacement op
def get_args_and_kwargs_layer_norm(
    graph_module: GraphModule,
    inputs_inputs: List[fx.Node],
    dequants_inputs: List[fx.Node],
    other_inputs: List[fx.Node],
    quant_node: fx.Node,
) -> Tuple[Tuple[ArgsType], Dict[str, ArgsType]]:
    """
    Returns the args and kwargs for the layer norm replacement op.
    """
    # Check if the input is per-channel quantized
    # TODO(matthiascremon): add proper support and testing for per-channel quantization
    assert isinstance(dequants_inputs[0].args[1], float) and isinstance(
        dequants_inputs[0].args[2], int
    ), "per-channel quantization is not supported for layer norm, both scale and zero_point should be scalars"

    # Make the scale and zero_point tensors
    scale_tensor = graph_module.graph.call_function(
        torch.ops.aten.full.default,
        (
            [1],
            dequants_inputs[0].args[1],
        ),
        {"dtype": torch.float32},
    )
    zero_point_tensor = graph_module.graph.call_function(
        torch.ops.aten.full.default,
        (
            [1],
            dequants_inputs[0].args[2],
        ),
        {"dtype": torch.int32},
    )

    weight = other_inputs[1] if len(other_inputs) > 1 else None

    if not weight:
        weight = graph_module.graph.call_function(
            torch.ops.aten.full.default,
            (
                other_inputs[0],
                1,
            ),
            {"dtype": torch.float32},
        )

    bias = other_inputs[2] if len(other_inputs) > 2 else None

    if not bias:
        bias = graph_module.graph.call_function(
            torch.ops.aten.full.default,
            (
                other_inputs[0],
                0,
            ),
            {"dtype": torch.float32},
        )

    # Make the args and kwargs for the replacement op
    args = tuple(inputs_inputs + [scale_tensor] + [zero_point_tensor])
    kwargs = {
        "normalized_shape": other_inputs[0],
        "weight": weight,
        "bias": bias,
        "eps": 1e-05,
        "output_scale": quant_node.args[1],
        "output_zero_point": quant_node.args[2],
    }
    return args, kwargs


def get_args_and_kwargs_matmul(
    inputs_inputs: List[fx.Node],
    dequants_inputs: List[fx.Node],
    quant_node: fx.Node,
) -> Tuple[Tuple[ArgsType, ...], Dict[str, ArgsType]]:
    requantize_scale = (
        # pyre-ignore[58]: Unsupported operand
        dequants_inputs[0].args[1]
        * dequants_inputs[1].args[1]
    ) / quant_node.args[1]
    requantize_scale_t = torch.tensor([requantize_scale])

    (out_multiplier, out_shift) = quantize_tensor_multiplier(requantize_scale_t)

    args = (
        inputs_inputs[0],
        dequants_inputs[0].args[2],
        inputs_inputs[1],
        dequants_inputs[1].args[2],
        None,
    )

    kwargs = {
        "out_multiplier": out_multiplier[0].item(),
        "out_shift": out_shift[0].item(),
        "out_zero_point": quant_node.args[2],
        "transposed": False,
    }
    return args, kwargs


def get_args_and_kwargs_conv(
    graph_module: GraphModule,
    inputs_inputs: List[fx.Node],
    dequants_inputs: List[fx.Node],
    weights_inputs: List[fx.Node],
    dequants_weights: List[fx.Node],
    bias_inputs: List[fx.Node],
    quant_node: fx.Node,
    op_node: fx.Node,
) -> Tuple[Tuple[ArgsType], Dict[str, ArgsType]]:
    weight_scale = dequants_weights[0].args[1]
    weight_zero_point = dequants_weights[0].args[2]
    # pyre-fixme[58]: Unsupported operand types
    bias_scale = dequants_inputs[0].args[1] * weight_scale
    stride = [1, 1] if len(op_node.args) < 4 else get_conv_args(op_node.args[3], 1)
    padding = [0, 0] if len(op_node.args) < 5 else get_conv_args(op_node.args[4], 0)
    dilation = [1, 1] if len(op_node.args) < 6 else get_conv_args(op_node.args[5], 1)
    groups = 1 if len(op_node.args) < 7 else op_node.args[6]

    # If bias is not available, create a bias tensor with the shape of weight[0]
    if not bias_inputs:
        weight_node = dequants_weights[0].args[0]
        assert isinstance(weight_node, fx.Node)
        bias = create_zero_bias_int32(graph_module, weight_node, bias_scale)
    else:
        bias = bias_inputs[0]

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
    args = tuple(inputs_inputs + weights_inputs + [bias])
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


def get_args_and_kwargs_relu(
    graph_module: GraphModule,
    inputs_inputs: List[fx.Node],
    dequants_inputs: List[fx.Node],
    quant_node: fx.Node,
) -> Tuple[Tuple[ArgsType], Dict[str, ArgsType]]:
    input_scale = dequants_inputs[0].args[1]
    # pyre-fixme[58]: Unsupported operand types
    requantize_scale = input_scale / quant_node.args[1]
    requantize_scale_t = torch.tensor([requantize_scale])

    (out_multiplier, out_shift) = quantize_tensor_multiplier(requantize_scale_t)

    # Make the args and kwargs for the replacement op
    args = tuple(inputs_inputs)

    X_zero_point = graph_module.graph.call_function(
        torch.ops.aten.full.default,
        ([1], dequants_inputs[0].args[2]),
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

    kwargs = {
        "X_zero_point": X_zero_point,
        "out_zero_point": quant_node.args[2],
        "out_multiplier": out_multiplier_,
        "out_shift": out_shift_,
    }
    return args, kwargs


class QuantFusion(ExportPass):
    # pyre-ignore[2]: Parameter `patterns` has no type specified
    def __init__(self, patterns) -> None:
        super().__init__()
        # pyre-ignore[4]: Parameter `patterns` of class `QuantFusion` has no type specified
        self.patterns = patterns

    def call(self, graph_module: fx.GraphModule) -> PassResult:  # noqa: C901
        for pattern in self.patterns:
            fused_partitions = find_sequential_partitions_aten(
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
                dequants_biases = []
                for node, idx, *_spec in anchors.biases:
                    if (
                        node.args[idx].target
                        == torch.ops.quantized_decomposed.dequantize_per_tensor.default
                    ):
                        dequants_biases.append(node.args[idx])

                inputs_inputs = [node.args[0] for node in dequants_inputs]
                weights_inputs = [node.args[0] for node in dequants_weights]
                bias_inputs = [node.args[0] for node in dequants_biases]
                other_inputs = [node.args[idx] for node, idx in anchors.others]

                # The node is the first index of the list and first of the tuple
                op_node = anchors.output[0][0]

                assert len(op_node.users) == 1
                quant_node = list(op_node.users.keys())[0]

                with graph_module.graph.inserting_after(op_node):
                    args = tuple(
                        inputs_inputs + weights_inputs + other_inputs + bias_inputs
                    )
                    kwargs = {}
                    if isinstance(pattern, (Conv1dPattern, Conv2dPattern)):
                        args, kwargs = get_args_and_kwargs_conv(
                            graph_module,
                            inputs_inputs,
                            dequants_inputs,
                            weights_inputs,
                            dequants_weights,
                            bias_inputs,
                            quant_node,
                            op_node,
                        )
                    elif isinstance(pattern, LinearPattern):
                        args, kwargs = get_args_and_kwargs_linear(
                            graph_module,
                            inputs_inputs,
                            dequants_inputs,
                            weights_inputs,
                            dequants_weights,
                            bias_inputs,
                            quant_node,
                        )
                    elif isinstance(pattern, LayerNormPattern):
                        args, kwargs = get_args_and_kwargs_layer_norm(
                            graph_module,
                            inputs_inputs,
                            dequants_inputs,
                            other_inputs,
                            quant_node,
                        )
                    elif isinstance(pattern, (BmmPattern, MatmulPattern)):
                        args, kwargs = get_args_and_kwargs_matmul(
                            inputs_inputs,
                            dequants_inputs,
                            quant_node,
                        )
                    elif isinstance(pattern, AddmmPattern):
                        # Transpose the weight tensor
                        transposed_weights = graph_module.graph.call_function(
                            torch.ops.aten.transpose.int,
                            (weights_inputs[0], 0, 1),
                        )
                        # Call linear with transposed weight
                        args, kwargs = get_args_and_kwargs_linear(
                            graph_module,
                            inputs_inputs,
                            dequants_inputs,
                            [transposed_weights],
                            dequants_weights,
                            bias_inputs,
                            quant_node,
                        )
                    elif isinstance(pattern, ReluPatterns):
                        args, kwargs = get_args_and_kwargs_relu(
                            graph_module,
                            inputs_inputs,
                            dequants_inputs,
                            quant_node,
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
    # pyre-ignore[2]: Parameter `nodes` has no type specified
    def is_fused(cls, nodes) -> bool:
        return any(cls.__qualname__ in n.meta for n in nodes)

    @classmethod
    # pyre-ignore[2]: Parameter `nodes` has no type specified
    def mark_fused(cls, nodes) -> bool:
        for n in nodes:
            # pyre-fixme[7]: Incompatible return type
            n.meta["QuantFusion"] = True
