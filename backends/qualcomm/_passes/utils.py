# Copyright (c) Qualcomm Innovation Center, Inc.
# All rights reserved
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import torch
from executorch.backends.qualcomm.builders.utils import get_parameter
from executorch.backends.qualcomm.utils.constants import QCOM_ENCODING
from executorch.exir.dialects._ops import ops as exir_ops


q_ops = {
    exir_ops.edge.quantized_decomposed.quantize_per_channel.default,
    exir_ops.edge.quantized_decomposed.quantize_per_tensor.default,
    exir_ops.edge.quantized_decomposed.quantize_per_tensor.tensor,
}

dq_ops = {
    exir_ops.edge.quantized_decomposed.dequantize_per_tensor.default,
    exir_ops.edge.quantized_decomposed.dequantize_per_tensor.tensor,
    exir_ops.edge.quantized_decomposed.dequantize_per_channel.default,
}


def get_quant_attrs(
    edge_program: torch.export.ExportedProgram, quant_node: torch.fx.Node
):
    quant_attr_keys = [arg.name for arg in quant_node.target._schema.arguments][1:]
    quant_attrs = dict.fromkeys(quant_attr_keys)

    for i in range(1, len(quant_node.args)):
        attr_n = quant_node.args[i]

        value = attr_n
        if isinstance(attr_n, torch.fx.node.Node):
            # could be a commonly shared attribute between q & dq
            if attr_n.target == exir_ops.edge.aten._to_copy.default:
                value = get_parameter(attr_n.args[0], edge_program)
            else:
                value = get_parameter(attr_n, edge_program)
        quant_attrs[quant_attr_keys[i - 1]] = value

    quant_attrs[QCOM_ENCODING] = quant_node.target
    return quant_attrs


def get_passes_dependency_for_capture_program():
    """
    This function records the dependencies for passes used in the capture_program.

    It returns a dictionary where the keys are pass classes and the values are lists of
    dependencies required by each pass. This helps in managing and organizing the sequence
    of passes needed for the capture_program to function correctly.

    Returns:
        dict: A dictionary mapping each pass to its corresponding list of dependencies.
    """
    from executorch.backends.qualcomm._passes import (
        AnnotateAndQuantScalar,
        AnnotateDecomposed,
        AnnotateQuantAttrs,
        ConvertBmmToMatmul,
        ConvertInterpolateWithUpsample2D,
        ConvertPReLU,
        ConvertToLinear,
        ExpandBroadcastTensorShape,
        FoldQDQ,
        I64toI32,
        LayoutTransform,
        RecomposePixelUnshuffle,
        RecomposeRmsNorm,
        RemoveRedundancy,
        ReplaceIndexPutInput,
    )

    return {
        RecomposePixelUnshuffle: [RemoveRedundancy],
        RecomposeRmsNorm: [RemoveRedundancy],
        ConvertToLinear: [RecomposePixelUnshuffle],
        ConvertPReLU: [RemoveRedundancy],
        ConvertBmmToMatmul: [ConvertToLinear],
        ConvertInterpolateWithUpsample2D: [RemoveRedundancy],
        I64toI32: [RemoveRedundancy],
        AnnotateQuantAttrs: [
            RecomposePixelUnshuffle,
            RecomposeRmsNorm,
            ConvertToLinear,
            ConvertPReLU,
            ConvertBmmToMatmul,
            ConvertInterpolateWithUpsample2D,
        ],
        AnnotateAndQuantScalar: [
            AnnotateQuantAttrs,
        ],
        AnnotateDecomposed: [RemoveRedundancy],
        FoldQDQ: [AnnotateQuantAttrs, AnnotateAndQuantScalar, AnnotateDecomposed],
        ExpandBroadcastTensorShape: [RemoveRedundancy],
        LayoutTransform: [
            AnnotateQuantAttrs,
            AnnotateAndQuantScalar,
            ExpandBroadcastTensorShape,
        ],
        ReplaceIndexPutInput: [LayoutTransform],
    }
