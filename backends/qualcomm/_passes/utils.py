# Copyright (c) Qualcomm Innovation Center, Inc.
# All rights reserved
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from typing import Callable, Dict, List

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
        CanonicalizeConv,
        ConvertBmmToMatmul,
        DecomposeAny,
        DecomposeColIm,
        DecomposeLinalgVectorNorm,
        DecomposeMaxPool3d,
        ExpandBroadcastTensorShape,
        FixedLinearKeepDim,
        FoldQDQ,
        I64toI32,
        LayoutTransform,
        RecomposePixelUnshuffle,
        RecomposeRmsNorm,
        RemoveRedundancy,
        TagQuantIO,
    )

    return {
        AnnotateAdaptiveAvgPool1D: [RemoveRedundancy],
        AnnotateQuantAttrs: [
            ConvertBmmToMatmul,
            RecomposePixelUnshuffle,
            RemoveRedundancy,
        ],
        AnnotateStack: [RemoveRedundancy],
        AnnotateUnbind: [RemoveRedundancy],
        ConvertBmmToMatmul: [RecomposePixelUnshuffle],
        DecomposeAny: [RemoveRedundancy],
        DecomposeColIm: [FoldQDQ],
        DecomposeLinalgVectorNorm: [RemoveRedundancy],
        DecomposeMaxPool3d: [RemoveRedundancy],
        ExpandBroadcastTensorShape: [FoldQDQ],
        FixedLinearKeepDim: [FoldQDQ],
        FoldQDQ: [AnnotateQuantAttrs, AnnotateStack, AnnotateUnbind],
        I64toI32: [RemoveRedundancy],
        LayoutTransform: [
            AnnotateQuantAttrs,
            CanonicalizeConv,
            ExpandBroadcastTensorShape,
            FixedLinearKeepDim,
        ],
        RecomposePixelUnshuffle: [RemoveRedundancy],
        RecomposeRmsNorm: [RemoveRedundancy],
        TagQuantIO: [LayoutTransform],
    }


def copy_nn_module_stack(src, target):
    """
    Copy meta["nn_module_stack"] from src node to target node if existing.
    """
    if value := src.meta.get("nn_module_stack"):
        target.meta["nn_module_stack"] = value


def merge_decomposed_graph(
    remap: Dict[str, torch.fx.Node],
    target_node: torch.fx.Node,
    target_graph: torch.fx.GraphModule,
    decomposed_graph_module: torch.fx.GraphModule,
    predicate: Callable[[torch.fx.Node], None] = None,
    # target_node, decomposed_output_node, remap
    output_processor: Callable[
        [torch.fx.Node, torch.fx.Node, Dict[str, torch.fx.Node]], None
    ] = None,
) -> None:
    def default_output_process(node):
        for user in node.users.copy():
            # remap
            user.replace_input_with(
                node,
                remap[decomposed_node.args[0][0]],
            )

    for decomposed_node in decomposed_graph_module.graph.nodes:
        copy_nn_module_stack(target_node, decomposed_node)
        if predicate is None or predicate(decomposed_node):
            # no need to copy existent 'output'
            if decomposed_node.op == "output":
                if output_processor is None:
                    default_output_process(target_node)
                else:
                    output_processor(target_node, decomposed_node, remap)
            # no need to copy existent placeholders
            elif decomposed_node.op == "placeholder":
                # replace node map from string to graph node
                remap[decomposed_node] = remap.pop(decomposed_node.name)
            else:
                remap[decomposed_node] = target_graph.node_copy(
                    decomposed_node,
                    arg_transform=lambda x, remap=remap: remap[x],
                )


def is_float_tensor(node: torch.fx.Node) -> bool:
    if "val" not in node.meta or not isinstance(node.meta["val"], FakeTensor):
        return False
    return node.meta["val"].dtype == torch.float32


def _is_node(node):
    return isinstance(node, torch.fx.Node)


def _pred(node, pat):
    return isinstance(pat, Callable) and pat(node)


def _next(node, from_args=True):
    if from_args:
        yield from [i for i in node.args if _is_node(i)]
    else:
        yield from list(node.users)


def find_pattern(
    node: torch.fx.Node,
    pattern: List[Callable[[torch.fx.Node], bool] | str],
    from_args: bool = True,
    max_wildcard_life: int = 3,
    verbose: bool = False,
):
    """
    Implement wildcard pattern matching
        - node: fx.Node
        - pattern: predicate list, can contain followings
            Callable(fx.node): predicate
            '*': wildcard
            '?': any single node
        - from_args: if True find from node.args, otherwise from node.users
        - max_wildcard_life: max number of skips for wildcard

    If not matched, return None.
    Otherwise, return list of matched node list, which is the same length as pattern
    """

    asterisk, question = "*", "?"

    def _probe(
        cur, hist, pat_idx, asterisk_life_count=max_wildcard_life, verbose=verbose
    ):
        if pat_idx == len(pattern):
            # Expected len(hist) is equal to pat_idx
            assert len(hist) == len(pattern)
            if list(hist) not in matched:
                matched.append(list(hist))
            return
        if verbose:
            print(
                f"cur:{cur}, idx:{pat_idx}, life={asterisk_life_count}, pattern:{pattern[pat_idx]} hist={hist}"
            )
        if pattern[pat_idx] == question or _pred(cur, pattern[pat_idx]):
            hist.append(cur)
            for child in _next(cur, from_args):
                _probe(child, hist, pat_idx + 1)
            hist.pop()
        elif pattern[pat_idx] == asterisk and asterisk_life_count > 0:
            # 3 cases: ignore/consume/keep asterisk
            # 1, Ignore asterisk
            hist.append(None)
            _probe(cur, hist, pat_idx + 1)
            hist.pop()

            # 2. Consume asterisk
            hist.append(None)
            for child in _next(cur, from_args):
                _probe(child, hist, pat_idx + 1)
            hist.pop()

            # 3. keep asterisk and skip to next node
            for child in _next(cur, from_args):
                _probe(child, hist, pat_idx, asterisk_life_count - 1)

    # Check if pattern is valid
    assert all(
        isinstance(i, Callable) or (isinstance(i, str) and (i == "*" or i == "?"))
        for i in pattern
    ), f"Invalid pattern: {pattern}"

    # Start probing
    matched = []
    _probe(node, [], 0)
    return matched if matched else None


def find_patterns(node, patterns, **kwargs):
    assert isinstance(patterns, list) and isinstance(patterns[0], list)
    results = []
    for pattern in patterns:
        result = find_pattern(node, pattern, **kwargs)
        results.append(result)
    return results


def append_qdq(
    graph_module: torch.fx.GraphModule,
    node: torch.fx.Node,
    qdq_node: torch.fx.Node,
):
    q_op = torch.ops.quantized_decomposed.quantize_per_tensor.default
    dq_op = torch.ops.quantized_decomposed.dequantize_per_tensor.default

    if qdq_node.target not in {q_op, dq_op}:
        return node

    with graph_module.graph.inserting_after(node):
        q_args = (node, *qdq_node.args[1:])
        q_node = graph_module.graph.create_node("call_function", q_op, q_args)
        q_node.meta = copy_meta(node.meta)
        q_node.meta["val"] = q_node.meta["val"].to(q_args[-1])
        with graph_module.graph.inserting_after(q_node):
            dq_args = (q_node, *qdq_node.args[1:])
            dq_node = graph_module.graph.create_node("call_function", dq_op, dq_args)
            dq_node.meta = copy_meta(node.meta)
    return dq_node
