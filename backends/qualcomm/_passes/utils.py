# Copyright (c) Qualcomm Innovation Center, Inc.
# All rights reserved
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from typing import Dict

import torch
from executorch.backends.qualcomm.builders.utils import get_parameter
from executorch.backends.qualcomm.utils.constants import QCOM_DTYPE, QCOM_ENCODING
from executorch.exir.dialects._ops import ops as exir_ops
from torch._subclasses import FakeTensor


def copy_meta(meta: Dict, callback=None):
    copied = {}
    for k, v in meta.items():
        copied[k] = v
    if callback:
        copied = callback(copied)
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

    # remap key for compatibility - block quantization only
    if dtype := quant_attrs.get("input_dtype", None):
        quant_attrs[QCOM_DTYPE] = dtype

    quant_attrs[QCOM_ENCODING] = quant_node.target
    return quant_attrs


def get_passes_dependency_for_capture_program():
    """
    This function records the dependencies for passes used in the to_edge_transform_and_lower_to_qnn.

    It returns a dictionary where the keys are pass classes and the values are lists of
    dependencies required by each pass. This helps in managing and organizing the sequence
    of passes needed for the to_edge_transform_and_lower_to_qnn to function correctly.

    Returns:
        dict: A dictionary mapping each pass to its corresponding list of dependencies.
    """
    from executorch.backends.qualcomm._passes import (
        AnnotateAdaptiveAvgPool1D,
        AnnotateQuantAttrs,
        AnnotateStack,
        AnnotateUnbind,
        ConvertBmmToMatmul,
        ConvertConv1dToConv2d,
        DecomposeAny,
        DecomposeColIm,
        DecomposeLinalgVectorNorm,
        ExpandBroadcastTensorShape,
        FixedLinearKeepDim,
        FoldQDQ,
        I64toI32,
        LayoutTransform,
        RecomposePixelUnshuffle,
        RecomposeRmsNorm,
        RemoveRedundancy,
        ReplaceIndexPutInput,
        TagQuantIO,
    )

    return {
        AnnotateAdaptiveAvgPool1D: [RemoveRedundancy],
        AnnotateQuantAttrs: [
            RecomposePixelUnshuffle,
            ConvertBmmToMatmul,
            RemoveRedundancy,
        ],
        AnnotateStack: [RemoveRedundancy],
        AnnotateUnbind: [RemoveRedundancy],
        ConvertBmmToMatmul: [RecomposePixelUnshuffle],
        DecomposeAny: [RemoveRedundancy],
        DecomposeColIm: [FoldQDQ],
        DecomposeLinalgVectorNorm: [RemoveRedundancy],
        ExpandBroadcastTensorShape: [FoldQDQ],
        FixedLinearKeepDim: [FoldQDQ],
        FoldQDQ: [AnnotateQuantAttrs, AnnotateStack, AnnotateUnbind],
        I64toI32: [RemoveRedundancy],
        LayoutTransform: [
            AnnotateQuantAttrs,
            ConvertConv1dToConv2d,
            ExpandBroadcastTensorShape,
            FixedLinearKeepDim,
        ],
        RecomposePixelUnshuffle: [RemoveRedundancy],
        RecomposeRmsNorm: [RemoveRedundancy],
        ReplaceIndexPutInput: [LayoutTransform],
        TagQuantIO: [ReplaceIndexPutInput],
    }


def copy_nn_module_stack(src, target):
    """
    Copy meta["nn_module_stack"] from src node to target node if existing.
    """
    if value := src.meta.get("nn_module_stack"):
        target.meta["nn_module_stack"] = value


def is_float_tensor(node: torch.fx.Node) -> bool:
    if "val" not in node.meta or not isinstance(node.meta["val"], FakeTensor):
        return False
    return node.meta["val"].dtype == torch.float32
