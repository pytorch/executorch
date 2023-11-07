import numpy as np
import serializer.tosa_serializer as ts
import torch
from executorch.backends.arm.tosa_mapping import TosaArg
from executorch.backends.arm.tosa_quant_utils import (
    dq_q_ops,
    getQuantNodeArgs,
    isQuantArg,
    q_op,
)
from executorch.backends.arm.tosa_utils import getNodeArgs
from executorch.exir.dialects._ops import ops as exir_ops
from torch._export.exported_program import ExportedProgram


def process_placeholder(
    node: torch.fx.Node, tosa_graph: ts.TosaSerializer, edge_program: ExportedProgram
):
    assert node.name == node.target, "Expect placeholder name and target to match"
    assert 0 == len(node.args), "Can't handle default input values"
    inputs = [TosaArg(node)]
    out = node.name

    if out in edge_program.graph_signature.inputs_to_parameters:
        parameter_name = edge_program.graph_signature.inputs_to_parameters[node.name]
        p_data = edge_program.state_dict[parameter_name]

        assert isinstance(p_data, torch.Tensor), "Expect Attr to be tensor"
        parameter_values = p_data.detach().numpy()

        # Check if they're for quantized nodes
        consumer_node = list(node.users)[0]
        if consumer_node.target in dq_q_ops:
            _, weight_node_scale, weight_node_zp, _, _, _ = getNodeArgs(consumer_node)

            parameter_values_quantized = (
                (parameter_values / weight_node_scale.number) + weight_node_zp.number
            ).astype(np.int8)
            tosa_graph.addConst(
                inputs[0].shape,
                ts.DType.INT8,
                parameter_values_quantized,
                name=out,
            )
        elif (
            consumer_node.target == exir_ops.edge.aten.addmm.default
            and list(consumer_node.users)[0].target == q_op
        ):
            (
                _,
                input_node,
                weight_node_permuted,
            ) = consumer_node.all_input_nodes
            weight_node = weight_node_permuted.all_input_nodes[0]

            input_node_scale, _ = getQuantNodeArgs(input_node)
            weight_node_scale, weight_node_zp = getQuantNodeArgs(weight_node)

            parameter_values_quantized = (
                parameter_values / (input_node_scale * weight_node_scale)
            ).astype(np.int32)

            tosa_graph.addConst(
                inputs[0].shape,
                ts.DType.INT32,
                parameter_values_quantized,
                name=out,
            )
        elif (
            consumer_node.target == exir_ops.edge.aten.convolution.default
            and list(consumer_node.users)[0].target == q_op
        ):
            (
                input_node,
                weight_node,
                bias_node,
            ) = consumer_node.all_input_nodes

            input_node_scale, _ = getQuantNodeArgs(input_node)
            weight_node_scale, _ = getQuantNodeArgs(weight_node)

            bias_scales = input_node_scale * weight_node_scale
            parameter_values_quantized = (parameter_values / bias_scales).astype(
                np.int32
            )

            tosa_graph.addConst(
                inputs[0].shape,
                ts.DType.INT32,
                parameter_values_quantized,
                name=out,
            )
        else:
            tosa_graph.addConst(
                inputs[0].shape, inputs[0].dtype, parameter_values, name=out
            )

    elif out in edge_program.graph_signature.inputs_to_buffers:
        parameter_name = edge_program.graph_signature.inputs_to_buffers[node.name]
        p_data = edge_program.state_dict[parameter_name]

        assert isinstance(p_data, torch.Tensor), "Expect Attr to be tensor"
        buffer_values = p_data.detach().numpy()
        tosa_graph.addConst(inputs[0].shape, inputs[0].dtype, buffer_values, name=out)
    else:
        tensor = ts.TosaSerializerTensor(
            inputs[0].name,
            inputs[0].shape,
            ts.DType.INT8 if isQuantArg(node) else inputs[0].dtype,
            data=None,
            placeholderFilename=inputs[0].name + ".npy",
        )
        tosa_graph.addInputTensor(tensor)
