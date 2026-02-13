# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.


import torch
from executorch.exir.dialects._ops import ops as exir_ops
from executorch.exir.pass_base import ExportPass, PassResult
from torch import fx
from torch._export.utils import (
    get_buffer,
    get_lifted_tensor_constant,
    get_param,
    is_lifted_tensor_constant,
    is_param,
)
from torch._guards import detect_fake_mode
from torch.export.exported_program import InputKind, InputSpec, TensorArgument


def _set_param(exported_program, node_or_name, tensor):
    """Set or create a parameter in an exported program.

    If node_or_name is a Node, updates the existing parameter or constant value.
    If node_or_name is a string, creates a new parameter placeholder.
    """
    fake_mode = detect_fake_mode(
        tuple(
            node.meta["val"]
            for node in exported_program.graph.nodes
            if node.op == "placeholder"
        )
    )

    if isinstance(node_or_name, fx.Node):
        node = node_or_name
        if node.name in exported_program.graph_signature.inputs_to_parameters:
            name = exported_program.graph_signature.inputs_to_parameters[node.name]
            exported_program.state_dict[name] = tensor
        elif (
            node.name
            in exported_program.graph_signature.inputs_to_lifted_tensor_constants
        ):
            name = exported_program.graph_signature.inputs_to_lifted_tensor_constants[
                node.name
            ]
            exported_program.constants[name] = tensor
        else:
            raise ValueError(
                f"Node {node.name} is not a parameter or lifted tensor constant"
            )
        node.meta["val"] = fake_mode.from_tensor(tensor, static_shapes=True)
        node.meta["val"].constant = tensor
        return node

    # Create a new parameter from string name
    name = node_or_name
    graph = exported_program.graph_module.graph
    placeholders = [n for n in graph.nodes if n.op == "placeholder"]
    input_name = f"arg_{name}"
    with graph.inserting_before(placeholders[0]):
        new_placeholder = graph.placeholder(input_name)
    exported_program.graph_signature.input_specs.insert(
        0,
        InputSpec(
            kind=InputKind.PARAMETER,
            arg=TensorArgument(name=input_name),
            target=name,
            persistent=None,
        ),
    )
    exported_program.state_dict[name] = tensor
    new_placeholder.meta["val"] = fake_mode.from_tensor(tensor, static_shapes=True)
    new_placeholder.meta["val"].constant = tensor
    return new_placeholder


class QuantizeFusedConvBnBiasPass(ExportPass):
    """
    BatchNorm fusion or QAT would introduce a bias that is not quantized if user
    specified bias=False because it's not there yet when the quantizer runs. This pass
    quantizes these biases so downstream passes can run.

    Supports both aten and edge dialect operators.
    """

    def __init__(self, exported_program, default_zero_bias=False) -> None:
        super().__init__()
        self.exported_program = exported_program
        self.default_zero_bias = default_zero_bias

    def _is_conv_node(self, node):
        """Check if node is a convolution operation."""
        return node.target in (
            exir_ops.edge.aten.convolution.default,
            torch.ops.aten.convolution.default,
            torch.ops.aten.conv2d.default,
        )

    def _is_edge_dialect(self, node):
        """Check if node uses edge dialect operators."""
        return node.target == exir_ops.edge.aten.convolution.default

    def _get_or_create_bias_node(self, node):
        """Get existing bias node or create a default zero bias if enabled."""
        input_dequant, weight_dequant, bias_node, *_ = node.args
        if bias_node is None:
            if self.default_zero_bias:
                channel = node.meta["val"].shape[1]
                bias_node = _set_param(
                    self.exported_program,
                    node.name + "_default_zero_bias",
                    torch.zeros(channel),
                )
                args = list(node.args)
                args[2] = bias_node
                node.args = tuple(args)
                return input_dequant, weight_dequant, bias_node
            return None, None, None
        return input_dequant, weight_dequant, bias_node

    def _get_bias_tensor(self, bias_node):
        """Extract bias tensor from parameter or lifted constant."""
        if is_param(self.exported_program, bias_node):
            return get_param(self.exported_program, bias_node)
        elif is_lifted_tensor_constant(self.exported_program, bias_node):
            return get_lifted_tensor_constant(self.exported_program, bias_node)
        return None

    def _unwrap_unsqueeze(self, input_dequant, is_edge):
        """Unwrap unsqueeze operations from input dequantize node."""
        if is_edge:
            unsqueeze_targets = (exir_ops.edge.aten.unsqueeze_copy.default,)
        else:
            unsqueeze_targets = (
                torch.ops.aten.unsqueeze_copy.default,
                torch.ops.aten.unsqueeze.default,
            )
        if input_dequant.target in unsqueeze_targets:
            return input_dequant.args[0]
        return input_dequant

    def _create_dequant_val(self, bias_node, bias):
        """Create fake tensor value for dequantized bias output."""
        bias_val = bias_node.meta.get("val")
        if bias_val is not None:
            return bias_val.to(torch.float32)
        return torch.empty(bias.shape, dtype=torch.float32)

    def _quantize_bias_per_channel(
        self, graph_module, node, bias, bias_node, bias_scale, dequant_val, is_edge
    ):
        """Quantize bias per-channel and insert dequantize node."""
        qbias = torch.ops.quantized_decomposed.quantize_per_channel.default(
            bias,
            bias_scale,
            torch.zeros(bias_scale.shape, dtype=torch.int32),
            0,
            -(2**31),
            2**31 - 1,
            torch.int32,
        )
        _set_param(self.exported_program, bias_node, qbias)

        dq_per_channel = (
            exir_ops.edge.quantized_decomposed.dequantize_per_channel.default
            if is_edge
            else torch.ops.quantized_decomposed.dequantize_per_channel.default
        )

        with graph_module.graph.inserting_before(node):
            bias_dequant = graph_module.graph.call_function(
                dq_per_channel,
                (
                    bias_node,
                    bias_scale,
                    torch.zeros(bias_scale.shape, dtype=torch.int32),
                    0,
                    -(2**31),
                    2**31 - 1,
                    torch.int32,
                ),
            )
            bias_dequant.meta["val"] = dequant_val
            node.replace_input_with(bias_node, bias_dequant)

    def _quantize_bias_per_tensor(
        self, graph_module, node, bias, bias_node, bias_scale, dequant_val, is_edge
    ):
        """Quantize bias per-tensor and insert dequantize node."""
        qbias = torch.ops.quantized_decomposed.quantize_per_tensor.default(
            bias, bias_scale, 0, -(2**31), 2**31 - 1, torch.int32
        )
        _set_param(self.exported_program, bias_node, qbias)

        dq_per_tensor = (
            exir_ops.edge.quantized_decomposed.dequantize_per_tensor.default
            if is_edge
            else torch.ops.quantized_decomposed.dequantize_per_tensor.default
        )

        with graph_module.graph.inserting_before(node):
            bias_dequant = graph_module.graph.call_function(
                dq_per_tensor,
                (bias_node, bias_scale, 0, -(2**31), 2**31 - 1, torch.int32),
            )
            bias_dequant.meta["val"] = dequant_val
            node.replace_input_with(bias_node, bias_dequant)

    def call(self, graph_module: fx.GraphModule) -> PassResult:
        modified = False
        for node in graph_module.graph.nodes:
            if not self._is_conv_node(node):
                continue

            is_edge = self._is_edge_dialect(node)

            input_dequant, weight_dequant, bias_node = self._get_or_create_bias_node(
                node
            )
            if bias_node is None:
                continue

            bias = self._get_bias_tensor(bias_node)
            if bias is None or bias.dtype == torch.int32:
                continue

            input_dequant = self._unwrap_unsqueeze(input_dequant, is_edge)

            dq_per_tensor = (
                exir_ops.edge.quantized_decomposed.dequantize_per_tensor.default
                if is_edge
                else torch.ops.quantized_decomposed.dequantize_per_tensor.default
            )
            assert (
                input_dequant.target == dq_per_tensor
            ), f"Expected dequantize_per_tensor, got {input_dequant.target}"

            dequant_val = self._create_dequant_val(bias_node, bias)

            if isinstance(weight_dequant.args[1], torch.fx.node.Node):
                weight_scale = get_buffer(self.exported_program, weight_dequant.args[1])
                bias_scale = input_dequant.args[1] * weight_scale
                self._quantize_bias_per_channel(
                    graph_module,
                    node,
                    bias,
                    bias_node,
                    bias_scale,
                    dequant_val,
                    is_edge,
                )
            else:
                weight_scale = weight_dequant.args[1]
                bias_scale = input_dequant.args[1] * weight_scale
                self._quantize_bias_per_tensor(
                    graph_module,
                    node,
                    bias,
                    bias_node,
                    bias_scale,
                    dequant_val,
                    is_edge,
                )

            modified = True
        graph_module.recompile()
        return PassResult(graph_module, modified)
