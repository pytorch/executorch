# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.


import torch
from executorch.exir.dialects._ops import ops as exir_ops
from executorch.exir.pass_base import ExportPass
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
from torch.fx.passes.infra.pass_base import PassBase, PassResult


# --- ExportedProgram param helpers ---


def _set_param_ep(exported_program, node_or_name, tensor, insert_before=None):
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
            exported_program.state_dict[name] = torch.nn.Parameter(
                tensor, requires_grad=False
            )
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
    exported_program.state_dict[name] = torch.nn.Parameter(tensor, requires_grad=False)
    new_placeholder.meta["val"] = fake_mode.from_tensor(tensor, static_shapes=True)
    new_placeholder.meta["val"].constant = tensor
    return new_placeholder


def _get_bias_tensor_ep(exported_program, bias_node):
    """Extract bias tensor from parameter or lifted constant in an ExportedProgram."""
    if is_param(exported_program, bias_node):
        return get_param(exported_program, bias_node)
    elif is_lifted_tensor_constant(exported_program, bias_node):
        return get_lifted_tensor_constant(exported_program, bias_node)
    return None


# --- GraphModule param helpers ---


def _get_tensor_from_node(graph_module, node):
    """Get tensor from a get_attr node on a GraphModule."""
    if node is None or node.op != "get_attr":
        return None
    target_atoms = node.target.split(".")
    attr = graph_module
    for atom in target_atoms:
        if not hasattr(attr, atom):
            return None
        attr = getattr(attr, atom)
    return attr


def _set_param_gm(graph_module, node_or_name, tensor, insert_before=None):
    """Set or create a parameter on a GraphModule using get_attr nodes.

    If node_or_name is a Node, updates the existing parameter tensor.
    If node_or_name is a string, creates a new get_attr node.
    """
    if isinstance(node_or_name, fx.Node):
        node = node_or_name
        target_atoms = node.target.split(".")
        parent = graph_module
        for atom in target_atoms[:-1]:
            parent = getattr(parent, atom)
        setattr(
            parent,
            target_atoms[-1],
            torch.nn.Parameter(tensor, requires_grad=False),
        )
        if "val" in node.meta:
            fake_mode = detect_fake_mode(
                tuple(
                    n.meta["val"] for n in graph_module.graph.nodes if "val" in n.meta
                )
            )
            if fake_mode is not None:
                node.meta["val"] = fake_mode.from_tensor(tensor, static_shapes=True)
            else:
                node.meta["val"] = tensor
        return node

    # Create new get_attr node
    name = node_or_name
    graph_module.register_parameter(
        name, torch.nn.Parameter(tensor, requires_grad=False)
    )
    with graph_module.graph.inserting_before(insert_before):
        new_node = graph_module.graph.get_attr(name)
    fake_mode = detect_fake_mode(
        tuple(n.meta["val"] for n in graph_module.graph.nodes if "val" in n.meta)
    )
    if fake_mode is not None:
        new_node.meta["val"] = fake_mode.from_tensor(tensor, static_shapes=True)
    else:
        new_node.meta["val"] = tensor
    return new_node


# --- Shared core logic ---


def _quantize_fused_conv_bias(
    graph_module,
    conv_targets,
    unsqueeze_targets,
    dq_per_tensor,
    dq_per_channel,
    get_bias_tensor,
    set_param,
    get_weight_scale_tensor,
    default_zero_bias=False,
):
    """Core logic for quantizing biases introduced by BatchNorm fusion/QAT.

    BatchNorm fusion or QAT introduces a bias to conv layers that originally had
    bias=False. Since the bias is added after the quantizer runs, it lacks proper
    quantize->dequantize nodes. This function adds them.

    Args:
        graph_module: The graph module to transform.
        conv_targets: Tuple of conv op targets to match.
        unsqueeze_targets: Tuple of unsqueeze op targets to unwrap.
        dq_per_tensor: The dequantize_per_tensor op for this dialect.
        dq_per_channel: The dequantize_per_channel op for this dialect.
        get_bias_tensor: Callable(node) -> Optional[Tensor].
        set_param: Callable(node_or_name, tensor, insert_before=None) -> Node.
        get_weight_scale_tensor: Callable(node) -> Tensor.
        default_zero_bias: If True, create zero bias for conv nodes without bias.

    Returns:
        True if any modifications were made.
    """
    modified = False
    for node in graph_module.graph.nodes:
        if node.target not in conv_targets:
            continue

        input_dequant, weight_dequant, bias_node, *_ = node.args

        if bias_node is None:
            if default_zero_bias:
                channel = node.meta["val"].shape[1]
                bias_node = set_param(
                    node.name + "_default_zero_bias",
                    torch.zeros(channel),
                    insert_before=node,
                )
                args = list(node.args)
                args[2] = bias_node
                node.args = tuple(args)
            else:
                continue

        bias = get_bias_tensor(bias_node)
        if bias is None or bias.dtype == torch.int32:
            continue

        if input_dequant.target in unsqueeze_targets:
            input_dequant = input_dequant.args[0]

        assert (
            input_dequant.target == dq_per_tensor
        ), f"Expected dequantize_per_tensor, got {input_dequant.target}"

        bias_val = bias_node.meta.get("val")
        dequant_val = (
            bias_val.to(torch.float32)
            if bias_val is not None
            else torch.empty(bias.shape, dtype=torch.float32)
        )

        if isinstance(weight_dequant.args[1], torch.fx.node.Node):
            weight_scale = get_weight_scale_tensor(weight_dequant.args[1])
            bias_scale = input_dequant.args[1] * weight_scale

            bias_zp = torch.zeros(bias_scale.shape, dtype=torch.int32)
            qbias = torch.ops.quantized_decomposed.quantize_per_channel.default(
                bias,
                bias_scale,
                bias_zp,
                0,
                -(2**31),
                2**31 - 1,
                torch.int32,
            )
            set_param(bias_node, qbias)

            scale_node = set_param(
                node.name + "_bias_scale", bias_scale, insert_before=node
            )
            zp_node = set_param(
                node.name + "_bias_zero_point", bias_zp, insert_before=node
            )

            with graph_module.graph.inserting_before(node):
                bias_dequant = graph_module.graph.call_function(
                    dq_per_channel,
                    (
                        bias_node,
                        scale_node,
                        zp_node,
                        0,
                        -(2**31),
                        2**31 - 1,
                        torch.int32,
                    ),
                )
                bias_dequant.meta["val"] = dequant_val
                node.replace_input_with(bias_node, bias_dequant)
        else:
            weight_scale = weight_dequant.args[1]
            bias_scale = input_dequant.args[1] * weight_scale

            qbias = torch.ops.quantized_decomposed.quantize_per_tensor.default(
                bias, bias_scale, 0, -(2**31), 2**31 - 1, torch.int32
            )
            set_param(bias_node, qbias)

            with graph_module.graph.inserting_before(node):
                bias_dequant = graph_module.graph.call_function(
                    dq_per_tensor,
                    (bias_node, bias_scale, 0, -(2**31), 2**31 - 1, torch.int32),
                )
                bias_dequant.meta["val"] = dequant_val
                node.replace_input_with(bias_node, bias_dequant)

        modified = True

    graph_module.recompile()
    return modified


# --- Pass classes ---


class QuantizeFusedConvBnBiasPass(ExportPass):
    """Quantize biases introduced by BatchNorm fusion/QAT on edge dialect graphs.

    Works on ExportedPrograms after to_edge() conversion.
    """

    def __init__(self, exported_program, default_zero_bias=False) -> None:
        super().__init__()
        self.exported_program = exported_program
        self.default_zero_bias = default_zero_bias

    def call(self, graph_module: fx.GraphModule) -> PassResult:
        ep = self.exported_program
        modified = _quantize_fused_conv_bias(
            graph_module,
            conv_targets=(exir_ops.edge.aten.convolution.default,),
            unsqueeze_targets=(exir_ops.edge.aten.unsqueeze_copy.default,),
            dq_per_tensor=exir_ops.edge.quantized_decomposed.dequantize_per_tensor.default,
            dq_per_channel=exir_ops.edge.quantized_decomposed.dequantize_per_channel.default,
            get_bias_tensor=lambda node: _get_bias_tensor_ep(ep, node),
            set_param=lambda n, t, insert_before=None: _set_param_ep(ep, n, t),
            get_weight_scale_tensor=lambda node: get_buffer(ep, node),
            default_zero_bias=self.default_zero_bias,
        )
        return PassResult(graph_module, modified)


class QuantizeFusedConvBnBiasAtenPass(PassBase):
    """Quantize biases introduced by BatchNorm fusion/QAT on aten dialect graphs.

    Operates on a GraphModule. If the graph_module came from an ExportedProgram
    (params are placeholder nodes), pass the exported_program so params can be
    resolved. If operating on a plain GraphModule (params are get_attr nodes),
    exported_program can be omitted.
    """

    def __init__(self, exported_program=None, default_zero_bias=False) -> None:
        self.exported_program = exported_program
        self.default_zero_bias = default_zero_bias

    def call(self, graph_module: fx.GraphModule) -> PassResult:
        ep = self.exported_program
        if ep is not None:
            get_bias = lambda node: _get_bias_tensor_ep(ep, node)
            set_param = lambda n, t, insert_before=None: _set_param_ep(ep, n, t)
            get_scale = lambda node: get_buffer(ep, node)
        else:
            get_bias = lambda node: _get_tensor_from_node(graph_module, node)
            set_param = lambda n, t, insert_before=None: _set_param_gm(
                graph_module, n, t, insert_before
            )
            get_scale = lambda node: _get_tensor_from_node(graph_module, node)

        modified = _quantize_fused_conv_bias(
            graph_module,
            conv_targets=(
                torch.ops.aten.convolution.default,
                torch.ops.aten.conv2d.default,
                torch.ops.aten.conv_transpose2d.input,
            ),
            unsqueeze_targets=(
                torch.ops.aten.unsqueeze_copy.default,
                torch.ops.aten.unsqueeze.default,
            ),
            dq_per_tensor=torch.ops.quantized_decomposed.dequantize_per_tensor.default,
            dq_per_channel=torch.ops.quantized_decomposed.dequantize_per_channel.default,
            get_bias_tensor=get_bias,
            set_param=set_param,
            get_weight_scale_tensor=get_scale,
            default_zero_bias=self.default_zero_bias,
        )
        return PassResult(graph_module, modified)
