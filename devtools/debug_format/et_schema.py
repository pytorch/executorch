# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-unsafe

"""
Intermediate Representation of ExecuTorch Concepts in Developer Tools
"""

from __future__ import annotations

import operator
import warnings

from collections import defaultdict

from enum import Enum
from types import NoneType
from typing import Any, Dict, List, Optional, Set, Tuple

import torch
from executorch import exir
from executorch.devtools.debug_format.base_schema import (
    Node,
    OperatorGraph,
    OperatorNode,
    ValueNode,
)
from torch._subclasses import FakeTensor


# Keywords used in debug_format Metadata
class RESERVED_METADATA_ARG(Enum):
    DEBUG_HANDLE = "debug_handle"
    MODULE_STACK = "nn_module_stack"
    SOURCE_FN_STACK = "source_fn_stack"
    MODULE_TYPE = "module_type"
    PROFILE_START_TIME = "profile_start_time"
    PROFILE_END_TIME = "profile_end_time"
    LOAD_START_TIME = "load_start_time"
    LOAD_END_TIME = "load_end_time"
    MEMORY_USAGE = "memory_usage"
    DEBUG_ENTRY = "debug_entry"
    STACK_TRACE = "stack_trace"
    DEBUG_DATA = "debug_data"

    METRICS_KEYWORD = "metrics"
    PROFILE_SUMMARY_COLDSTART = "Coldstart"
    PROFILE_SUMMARY_AVERAGE = "Average"
    PROFILE_SUMMARY_P90 = "P90"
    PROFILE_SUMMARY_P10 = "P10"
    PROFILE_SUMMARY_MIN = "Min"
    PROFILE_SUMMARY_MAX = "Max"

    AGGREGATED_OP_TABLE = "Aggregated op stats"
    RUN_SUMMARY_INDIVIDUAL_RUNS_TABLE = "Run summary individual stats"
    RUN_SUMMARY_TABLE = "Aggregated run summary stats"
    OP_INSTANCE_SUMMARY_TABLE = "Individual op stats"

    TABLES_KEYWORD = "tables"
    KV_KEYWORD = "kv"
    MODEL_LOAD_TIME_KEY = "Model load time (ms)"


# Representation of an FX GraphModule as an OperatorGraph
class FXOperatorGraph(OperatorGraph):
    @staticmethod
    def _get_node_name(node: torch.fx.Node) -> str:
        if node.target == operator.getitem:
            # pyre-ignore[9]: Incompatible variable type
            node = node.args[0]
            assert isinstance(
                node, torch.fx.Node
            ), f"First argument of getitem must be a torch fx node. Got {node.args[0]}"

        # Adding the "_" to the node name prevents TensorBoard from collapsing
        # nodes with similar names that only differ by an integer at the end of
        # their name.
        return node.name + "_"

    @staticmethod
    def _get_op_name(node: torch.fx.Node) -> str:
        # pyre-ignore[16]: has no attribute `__name__`.
        return node.target.__name__

    # Given a node and its metadata (if containing module stack), update the provided module mappings
    @staticmethod
    def _update_module_mapping(
        node: Node,
        module_mapping: Dict[Tuple[str, str], List[Node]],
        metadata: Dict[str, Any],
    ):
        if (
            source_fn_stack := metadata.get("source_fn_stack")
        ) is not None and "nn_module_stack" in metadata:
            # (module name, module type)
            source_fn = source_fn_stack[-1]
            module_type = (
                source_fn[1] if isinstance(source_fn[1], str) else source_fn[1].__name__
            )
            module_mapping[(source_fn[0], module_type)].append(node)

    @staticmethod
    def _parse_args(  # noqa: C901
        node: torch.fx.Node,
        nodes: Dict[str, Node],
        const_count: int,
        module_mapping: Dict[Tuple[str, str], List[Node]],
        enable_module_hierarchy: bool,
    ) -> Tuple[List[Node], int]:
        inputs = []
        op = node.op
        name = node.name
        args = node.args
        kwargs = node.kwargs
        named_args = None
        if node.op == "call_function" and hasattr(node.target, "_schema"):
            # pyre-ignore
            named_args = node.target._schema.arguments

        for index, arg in enumerate(args):
            if isinstance(arg, torch.fx.node.Node):
                if arg.target == exir.memory.alloc:
                    continue
                arg_name = FXOperatorGraph._get_node_name(arg)
            elif isinstance(arg, (int, float, torch.dtype, str)):
                # e.g. The "0" from node.args of squeeze_copy (mm_default, 0)
                if named_args and len(named_args) > index:
                    arg_name = named_args[index].name + "_" + str(const_count)
                else:
                    arg_name = "CONST_" + str(const_count)
                const_count += 1
                const_node = ValueNode(arg_name, val=str(arg))
                nodes[arg_name] = const_node
                if enable_module_hierarchy:
                    FXOperatorGraph._update_module_mapping(
                        const_node, module_mapping, node.meta
                    )
            elif isinstance(arg, list):
                arg_name: List[str] = []
                for list_arg in arg:
                    if isinstance(list_arg, (int, float)):
                        # Consider the whole list of ints/floats as a single constant and
                        # stringify that.
                        if named_args and len(named_args) > index:
                            arg_name = [named_args[index].name + "_" + str(const_count)]
                        else:
                            arg_name = ["CONST_" + str(const_count)]
                        const_count += 1
                        const_node = ValueNode(arg_name[0], val=arg)
                        nodes[arg_name[0]] = const_node
                        if enable_module_hierarchy:
                            FXOperatorGraph._update_module_mapping(
                                const_node, module_mapping, node.meta
                            )
                        break
                    elif isinstance(list_arg, torch.fx.node.Node):
                        arg_name += [FXOperatorGraph._get_node_name(list_arg)]
                    elif list_arg is None:
                        arg_name += ["CONST_NONE" + str(const_count)]
                        const_count += 1
                        const_node = ValueNode(arg_name[-1], val=str(arg))
                        nodes[arg_name[-1]] = const_node
                        if enable_module_hierarchy:
                            FXOperatorGraph._update_module_mapping(
                                const_node, module_mapping, node.meta
                            )
                    else:
                        raise Exception(
                            f"Unsupported argument encountered in list {arg}, {type(arg[0])}"
                        )
            elif isinstance(arg, NoneType):
                continue
            else:
                raise Exception(
                    f"Unsupported argument encountered {op}, {name}, {arg}, {type(arg)}"
                )

            if isinstance(arg_name, list):
                for val in arg_name:
                    inputs.append(nodes[val])
            else:
                inputs.append(nodes[arg_name])
        for _, node in kwargs.items():
            # We can ignore the out kwarg as that's mostly used to pass in the output tensor
            # which has been memory planned. The same value is also returned by the operator
            # which is then consumed by other nodes in the graph.
            if (
                isinstance(node, torch.fx.node.Node)
                and node.target == exir.memory.alloc
            ):
                continue
            else:
                warnings.warn(
                    f"Unsupported kwarg encountered: {name}, {kwargs}", stacklevel=1
                )

        return inputs, const_count

    # Given an FX GraphModule, parse it into an OperatorGraph
    @staticmethod
    def gen_operator_graph(
        model: torch.fx.GraphModule,
        skip_stack_trace: Optional[bool] = False,
        enable_module_hierarchy: bool = False,
    ) -> FXOperatorGraph:
        graph: torch.fx.Graph = model.graph

        nodes = {}
        input_nodes = {}
        output_nodes = {}
        out_variant_output_nodes = set()
        module_mapping = defaultdict(list)

        const_count = 0
        for fx_node in graph.nodes:
            if (
                fx_node.target == exir.memory.alloc
                or fx_node.target == operator.getitem
            ):
                continue
            op = fx_node.op
            name = FXOperatorGraph._get_node_name(fx_node)
            target = fx_node.target
            args = fx_node.args
            kwargs = fx_node.kwargs
            metadata = FXOperatorGraph._extract_metadata(fx_node.meta, skip_stack_trace)
            output_shapes = FXOperatorGraph._extract_output_shapes(
                fx_node.meta.get("val")
            )
            dtype = FXOperatorGraph._extract_output_dtype(fx_node.meta.get("val")) or ""

            assert (
                op != "call_module"
            ), f"Module call not yet supported in edge model graph [.toEdge()]: {name}, {str(target)}"
            assert (
                op != "call_method"
            ), f"Call Method not yet supported in edge model graph [.toEdge()]: {name}, {str(target)}"

            # Input
            if op == "placeholder":
                node = ValueNode(
                    name,
                    output_shapes=output_shapes,
                    metadata=metadata,
                    dtype=str(dtype),
                )  # val is default arg
                input_nodes[name] = node
            # Constants
            elif op == "get_attr":
                node = ValueNode(
                    name,
                    output_shapes=output_shapes,
                    metadata=metadata,
                    dtype=str(dtype),
                )
            # Output
            elif op == "output":
                assert len(args) == 1
                # Args of op=='output' is a wrapped list of return nodes ([ret_1, ret_2, ...], )
                in_nodes = [
                    (
                        nodes[FXOperatorGraph._get_node_name(ret)]
                        if ret is not None
                        else []
                    )
                    for ret in args[0]
                ]
                node = ValueNode(
                    name,
                    inputs=in_nodes,
                    output_shapes=output_shapes,
                    metadata=metadata,
                    dtype=str(dtype),
                )
                output_nodes[name] = node
            # Op Calls
            elif op == "call_function":
                inputs, const_count = FXOperatorGraph._parse_args(
                    fx_node, nodes, const_count, module_mapping, enable_module_hierarchy
                )
                named_args = []
                if fx_node.op == "call_function" and hasattr(fx_node.target, "_schema"):
                    named_args = [arg.name for arg in fx_node.target._schema.arguments]
                node = OperatorNode(
                    name,
                    inputs=inputs,
                    output_shapes=output_shapes,
                    metadata=metadata,
                    op=FXOperatorGraph._get_op_name(fx_node),
                    named_args=named_args,
                )
                if enable_module_hierarchy:
                    FXOperatorGraph._update_module_mapping(
                        node, module_mapping, fx_node.meta
                    )

                for kwarg_name, kwarg in kwargs.items():
                    if (
                        isinstance(kwarg, torch.fx.node.Node)
                        and kwarg.target == exir.memory.alloc
                        and kwarg_name == "out"
                    ):
                        nodes[FXOperatorGraph._get_node_name(kwarg)] = node
                        out_variant_output_nodes.add(
                            FXOperatorGraph._get_node_name(kwarg)
                        )
            else:
                raise Exception(f"Unsupported op type encountered {op}, {name}")

            nodes[name] = node
        return FXOperatorGraph._compose_op_graph(
            "base",
            nodes,
            input_nodes,
            output_nodes,
            out_variant_output_nodes,
            module_mapping,
        )

    @staticmethod
    def _compose_op_graph(
        name: str,
        nodes: Dict[str, Node],
        input_nodes: Dict[
            str, Node | OperatorGraph
        ],  # Never OperatorGraph, annotated for Pyre
        output_nodes: Dict[
            str, Node | OperatorGraph
        ],  # Never OperatorGraph, annotated for Pyre
        out_variant_output_nodes: Set[str],
        module_mapping: Dict[
            Tuple[str, str], List[Any]
        ],  # Any used here for Pyre, list of Nodes
    ):
        # Generate Module Graphs
        module_graphs: List[OperatorGraph] = []
        for module_key, module_nodes in module_mapping.items():
            module_element = OperatorGraph(
                graph_name=module_key[0],
                elements=module_nodes,
                metadata={"module_type": module_key[1]},
            )
            module_graphs.append(module_element)

            # Remove module modes from main graph
            for node in module_nodes:
                nodes.pop(node.name)

        main_nodes = [
            node
            for name, node in nodes.items()
            if name not in input_nodes
            and name not in output_nodes
            and name not in out_variant_output_nodes
        ]
        main_graph = FXOperatorGraph(
            graph_name="forward", elements=main_nodes + module_graphs
        )
        input_graph = FXOperatorGraph(
            graph_name="inputs", elements=list(input_nodes.values())
        )
        output_graph = FXOperatorGraph(
            graph_name="outputs", elements=list(output_nodes.values())
        )

        return FXOperatorGraph(
            graph_name=name,
            elements=[input_graph, main_graph, output_graph],
        )

    # Given a dict, extract only the utilized metadata
    @staticmethod
    def _extract_metadata(
        metadata: Dict[str, Any], skip_stack_trace: Optional[bool] = False
    ) -> Dict[str, Any]:
        ret = {}
        if RESERVED_METADATA_ARG.DEBUG_HANDLE.value in metadata:
            ret[RESERVED_METADATA_ARG.DEBUG_HANDLE.value] = metadata[
                RESERVED_METADATA_ARG.DEBUG_HANDLE.value
            ]
        if not skip_stack_trace and RESERVED_METADATA_ARG.STACK_TRACE.value in metadata:
            ret[RESERVED_METADATA_ARG.STACK_TRACE.value] = metadata[
                RESERVED_METADATA_ARG.STACK_TRACE.value
            ]
        if RESERVED_METADATA_ARG.MODULE_STACK.value in metadata:
            ret[RESERVED_METADATA_ARG.MODULE_STACK.value] = metadata[
                RESERVED_METADATA_ARG.MODULE_STACK.value
            ]
        return ret

    @staticmethod
    def _extract_output_shapes(val: Any) -> Optional[List[List[int]]]:
        if isinstance(val, (FakeTensor, torch.Tensor)):
            # If val is a single tensor
            return [list(val.shape)]
        elif isinstance(val, tuple) and all(
            isinstance(tensor, (FakeTensor, torch.Tensor)) for tensor in val
        ):
            # If val is a tuple of tensors
            shapes = [list(fake_tensor.shape) for fake_tensor in val]
            return shapes
        else:
            return None

    @staticmethod
    def _extract_output_dtype(val: Any) -> Optional[List[torch.dtype]]:
        if isinstance(val, (FakeTensor, torch.Tensor)):
            # If val is a single tensor
            return [val.dtype]
        elif isinstance(val, tuple) and all(
            isinstance(tensor, (FakeTensor, torch.Tensor)) for tensor in val
        ):
            # If val is a tuple of tensors
            dtypes = [fake_tensor.dtype for fake_tensor in val]
            return dtypes
        else:
            return None
