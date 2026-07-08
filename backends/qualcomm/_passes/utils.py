# Copyright (c) Qualcomm Innovation Center, Inc.
# All rights reserved
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from typing import Callable, Dict, List

import torch
from executorch.backends.qualcomm.builders.node_visitor import q_ops
from executorch.backends.qualcomm.builders.utils import get_parameter
from executorch.backends.qualcomm.utils.constants import (
    QCOM_DTYPE,
    QCOM_ENCODING,
    QCOM_QUANT_ATTRS,
)
from executorch.exir.dialects._ops import ops as exir_ops
from torch._subclasses import FakeTensor


def _create_q_or_dq_node(
    graph_module: torch.fx.GraphModule,
    node: torch.fx.node,
    target: torch.fx.node.Target,
    quant_attrs: Dict = None,
    pop_quant_attrs: bool = True,
) -> torch.fx.node:
    # pop_quant_attrs: Most cases it makes sense to pop quant attributes from source node.
    # e.g., input(quant_attrs) -> q -> node -> ...
    # In this case, pop input quant attr makes sense so node_visitor does not interpret input as quantized input.
    # However, LPAI with partition case needs to keep the node as quantized. Check lpai_partition_fallback_support.py for more info.

    def create_args(target: torch.fx.node.Target, quant_attrs: Dict):
        ret = []

        arg_schemas = list(target._schema.arguments)[1:]
        for arg_schema in arg_schemas:
            name = arg_schema.name
            # TODO: Due to the new parameter "out_dtype" in the dequantize node,
            # it could not be found in the quant_attrs of other nodes,
            # and it will cause a key error. For now, the output type
            # of our dequantize node is only float. (by default in pytorch)
            if name == "out_dtype":
                continue
            value = quant_attrs[name]
            if isinstance(arg_schema.type, torch.Tensor) and (
                isinstance(value, int) or isinstance(value, float)
            ):
                value = torch.tensor(value)
            ret.append(value)
        return ret

    # check if there has a specified quant_attrs
    # if not, use the existent info. from current node
    if quant_attrs is None:
        quant_attrs = node.meta.get(QCOM_QUANT_ATTRS)

    inserted_node = graph_module.graph.create_node(
        "call_function",
        target,
        (node, *create_args(target, quant_attrs)),
    )
    meta_val = node.meta["val"]
    if target in q_ops:
        inserted_node.meta[QCOM_QUANT_ATTRS] = (
            node.meta.pop(QCOM_QUANT_ATTRS)
            if pop_quant_attrs
            else node.meta.get(QCOM_QUANT_ATTRS)
        )
        meta_val = meta_val.to(quant_attrs["dtype"])

    inserted_node.meta["val"] = meta_val
    return inserted_node


def insert_quant_node(
    graph_module: torch.fx.GraphModule,
    input_node: torch.fx.node,
    output_node: torch.fx.node,
    target: torch.fx.node.Target,
    quant_attrs: Dict = None,
    pop_quant_attrs: bool = True,
) -> torch.fx.Node:
    with graph_module.graph.inserting_after(input_node):
        inserted_node = _create_q_or_dq_node(
            graph_module=graph_module,
            node=input_node,
            target=target,
            quant_attrs=quant_attrs,
            pop_quant_attrs=pop_quant_attrs,
        )
        # If we found mix quantization pattern and reuse the existing q_node, we skip adding a new q node.
        if output_node.target not in q_ops:
            output_node.replace_input_with(input_node, inserted_node)
    return inserted_node


def insert_dequant_node(
    graph_module: torch.fx.GraphModule,
    input_node: torch.fx.node,
    output_node: torch.fx.node,
    target: torch.fx.node.Target,
) -> None:
    with graph_module.graph.inserting_after(input_node):
        inserted_node = _create_q_or_dq_node(
            graph_module=graph_module, node=input_node, target=target
        )
        output_node.replace_input_with(input_node, inserted_node)
    return inserted_node


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


def copy_nn_module_stack(src, target):
    """
    Copy meta["nn_module_stack"] from src node to target node if existing.
    """
    if value := src.meta.get("nn_module_stack"):
        target.meta["nn_module_stack"] = value


def _unify_fake_mode(node: torch.fx.Node, fake_mode) -> None:
    val = node.meta.get("val")
    if val is None:
        return
    if isinstance(val, FakeTensor) and val.fake_mode is not fake_mode:
        node.meta["val"] = fake_mode.from_tensor(val)
    elif isinstance(val, (list, tuple)):
        unified = []
        for v in val:
            if isinstance(v, FakeTensor) and v.fake_mode is not fake_mode:
                unified.append(fake_mode.from_tensor(v))
            else:
                unified.append(v)
        node.meta["val"] = type(val)(unified)


def merge_decomposed_graph(  # noqa: C901
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
    target_fake_mode = None
    target_val = target_node.meta.get("val")
    if isinstance(target_val, FakeTensor):
        target_fake_mode = target_val.fake_mode
    elif isinstance(target_val, (list, tuple)):
        for v in target_val:
            if isinstance(v, FakeTensor):
                target_fake_mode = v.fake_mode
                break

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
                copied = target_graph.node_copy(
                    decomposed_node,
                    arg_transform=lambda x, remap=remap: remap[x],
                )
                if target_fake_mode is not None:
                    _unify_fake_mode(copied, target_fake_mode)
                remap[decomposed_node] = copied


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


def create_const_node(
    graph: torch.fx.Graph,
    graph_module: torch.fx.GraphModule,
    attr_name: str,
    value,
    source_node: torch.fx.Node,
) -> torch.fx.Node:
    """
    Register a scalar constant as a named buffer on the graph module and return a get_attr node referencing it.
    Used in edge dialect op decomposition passes where raw scalar arguments are not accepted by QNN op builders which need the inputs to be graph nodes.
    """
    dtype = source_node.meta["val"].dtype
    tensor = torch.tensor(value, dtype=dtype)
    graph_module.register_buffer(attr_name, tensor)

    fake_mode = source_node.meta["val"].fake_mode
    with graph.inserting_before(next(iter(graph.nodes))):
        const_node = graph.get_attr(attr_name)
        const_node.meta["val"] = fake_mode.from_tensor(tensor)
    return const_node


def create_node(graph, target, args, meta, callback=None):
    node = graph.create_node("call_function", target, args)
    node.meta = copy_meta(meta, callback)
    return node
