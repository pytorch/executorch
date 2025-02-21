# Copyright (c) Qualcomm Innovation Center, Inc.
# All rights reserved
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from typing import Dict

import torch
from executorch.backends.qualcomm.builders.utils import get_parameter
from executorch.backends.qualcomm.utils.constants import QCOM_ENCODING
from executorch.exir.dialects._ops import ops as exir_ops
from torch._subclasses import FakeTensor


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


def copy_meta(meta: Dict):
    copied = {}
    for k, v in meta.items():
        copied[k] = v
    return copied


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
        AnnotateDecomposed,
        AnnotateQuantAttrs,
        AnnotateStack,
        ConstantI64toI32,
        ConvertBmmToMatmul,
        ConvertConv1dToConv2d,
        ConvertInterpolateWithUpsample2D,
        ConvertToLinear,
        DecomposeAny,
        DecomposeLinalgVectorNorm,
        ExpandBroadcastTensorShape,
        FoldQDQ,
        LayoutTransform,
        RecomposePixelUnshuffle,
        RecomposePReLU,
        RecomposeRmsNorm,
        RemoveRedundancy,
        ReplaceIndexPutInput,
        TensorI64toI32,
    )

    return {
        AnnotateDecomposed: [RemoveRedundancy],
        AnnotateQuantAttrs: [
            RecomposePixelUnshuffle,
            RecomposeRmsNorm,
            ConvertToLinear,
            RecomposePReLU,
            ConvertBmmToMatmul,
            ConvertInterpolateWithUpsample2D,
        ],
        AnnotateStack: [FoldQDQ],
        ConstantI64toI32: [ConvertInterpolateWithUpsample2D],
        ConvertBmmToMatmul: [ConvertToLinear],
        ConvertConv1dToConv2d: [FoldQDQ],
        ConvertInterpolateWithUpsample2D: [RemoveRedundancy],
        ConvertToLinear: [RecomposePixelUnshuffle],
        DecomposeAny: [RemoveRedundancy],
        DecomposeLinalgVectorNorm: [RemoveRedundancy],
        ExpandBroadcastTensorShape: [RemoveRedundancy],
        FoldQDQ: [AnnotateQuantAttrs, AnnotateDecomposed],
        LayoutTransform: [
            AnnotateQuantAttrs,
            ConvertConv1dToConv2d,
            ExpandBroadcastTensorShape,
        ],
        RecomposePixelUnshuffle: [RemoveRedundancy],
        RecomposePReLU: [RemoveRedundancy],
        RecomposeRmsNorm: [RemoveRedundancy],
        ReplaceIndexPutInput: [LayoutTransform],
        TensorI64toI32: [RemoveRedundancy],
    }


def is_float_tensor(node: torch.fx.Node) -> bool:
    if "val" not in node.meta or not isinstance(node.meta["val"], FakeTensor):
        return False
    return node.meta["val"].dtype == torch.float32
