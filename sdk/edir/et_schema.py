"""
Intermediate Representation of Executorch Concepts in Productivity SDK
"""

from __future__ import annotations

import operator
import warnings

from collections import defaultdict

from dataclasses import dataclass
from enum import Enum
from typing import Any, Dict, List, Optional, Set, Tuple

import numpy as np
import torch
from executorch import exir
from executorch.exir import schema
from executorch.exir.schema import KernelCall, Program, TensorList
from executorch.exir.serialize import deserialize_from_flatbuffer
from executorch.sdk.edir.base_schema import Node, OperatorGraph, OperatorNode, ValueNode
from executorch.sdk.etdump.schema import ETDump, PROFILE_EVENT_ENUM
from executorch.sdk.etdump.serialize import deserialize_from_etdump


# Keywords used in EDIR Metadata
class RESERVED_METADATA_ARG(Enum):
    DEBUG_HANDLE = "debug_handle"
    MODULE_STACK = "module_stack"
    SOURCE_FN = "source_fn"
    MODULE_TYPE = "module_type"
    PROFILE_START_TIME = "profile_start_time"
    PROFILE_END_TIME = "profile_end_time"
    LOAD_START_TIME = "load_start_time"
    LOAD_END_TIME = "load_end_time"
    MEMORY_USAGE = "memory_usage"
    DEBUG_ENTRY = "debug_entry"
    STACK_TRACE = "stack_trace"

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


# Column Headers for Summary Tables
class PROFILE_STAT_HEADER(Enum):
    NAME = "Name"
    COUNT = "Count"
    COLD_START_MS = "Coldstart (ms)"
    MEAN_MS = "Mean (ms)"
    MIN_MS = "Min (ms)"
    P10_MS = "P10 (ms)"
    P90_MS = "P90 (ms)"
    MAX_MS = "Max (ms)"


# A single inference run extracted from Executorch SDK ET Dump
@dataclass
class InferenceRun:
    # Maps from a node identifier (e.g. Debug Handle) to its associated metadata
    # TODO: Pyre Doesn't play nice with nested dicts, use [Any, Any] for now
    node_metadata: Dict[Any, Any]

    # Run level metadata
    run_metadata: Dict[str, Any]

    # Given an ET Dump object, extract all inference runs
    # Note: This current supports ET Dump files containing only one ET Dump Run
    @staticmethod
    def _extract_runs_from_etdump(et_dump: ETDump) -> List[InferenceRun]:
        node_metadata = defaultdict(lambda: defaultdict(list))
        run_metadata = {}
        for run in et_dump.run_data:

            # Debug Blocks
            # Note: Debug Blocks are not currently being generated in live ET Dumps
            for block in run.debug_blocks:
                for event in block.debug_events:
                    debug_handle = event.debug_handle
                    node_metadata[debug_handle][
                        RESERVED_METADATA_ARG.DEBUG_ENTRY.value
                    ].append(str(event.debug_entries))

            # Profile Blocks
            for block in run.profile_blocks:
                for idx, event in enumerate(block.profile_events):
                    if event.name == PROFILE_EVENT_ENUM.LOAD_MODEL.value:
                        run_metadata[
                            RESERVED_METADATA_ARG.LOAD_START_TIME.value
                        ] = event.start_time
                        run_metadata[
                            RESERVED_METADATA_ARG.LOAD_END_TIME.value
                        ] = event.end_time

                    # Parse Model Run time
                    if event.name == PROFILE_EVENT_ENUM.RUN_MODEL.value:
                        run_metadata.setdefault(
                            RESERVED_METADATA_ARG.PROFILE_START_TIME.value, []
                        ).append(event.start_time)
                        run_metadata.setdefault(
                            RESERVED_METADATA_ARG.PROFILE_END_TIME.value, []
                        ).append(event.end_time)

                    def extract_start_end_time(event):
                        debug_handle = event.debug_handle
                        node_metadata[debug_handle][
                            RESERVED_METADATA_ARG.PROFILE_START_TIME.value
                        ].append(event.start_time)
                        node_metadata[debug_handle][
                            RESERVED_METADATA_ARG.PROFILE_END_TIME.value
                        ].append(event.end_time)

                    # Parse Operator Call times (logged entry after OPERATOR_CALL event)
                    prev_event = block.profile_events[idx - 1].name if idx > 0 else ""
                    if (
                        prev_event == PROFILE_EVENT_ENUM.OPERATOR_CALL.value
                        or event.name == PROFILE_EVENT_ENUM.DELEGATE_CALL.value
                    ):
                        extract_start_end_time(event)

        return [InferenceRun(node_metadata, run_metadata)]

    # Given the path of an ET Dump file, extract all inference runs
    # Note: This current supports ET Dump files containing only one ET Dump Run
    @staticmethod
    def extract_runs_from_path(file_path: str) -> List[InferenceRun]:
        with open(file_path, "rb") as buff:
            et_dump = deserialize_from_etdump(buff.read())
            return InferenceRun._extract_runs_from_etdump(et_dump)


class OperatorGraphWithStats(OperatorGraph):

    # Generate model load time (a single number in string form)
    def _gen_model_load_time(self) -> str:
        metadata = self.metadata
        if metadata is not None:
            return str(
                (
                    metadata[RESERVED_METADATA_ARG.LOAD_END_TIME.value]
                    - metadata[RESERVED_METADATA_ARG.LOAD_START_TIME.value]
                )
                / (1000 * 1000)
            )

        return "Data not available"

    # Generate summary stats across profiling runs
    def _gen_run_summary_stats(self) -> List[Any]:
        header_row = [
            [
                PROFILE_STAT_HEADER.NAME.value,
                PROFILE_STAT_HEADER.COUNT.value,
                PROFILE_STAT_HEADER.MEAN_MS.value,
                PROFILE_STAT_HEADER.MIN_MS.value,
                PROFILE_STAT_HEADER.P10_MS.value,
                PROFILE_STAT_HEADER.P90_MS.value,
                PROFILE_STAT_HEADER.MAX_MS.value,
            ]
        ]
        data_rows = []
        metadata = self.metadata
        if metadata is not None:
            run_durations = [
                end - start
                for start, end in zip(
                    metadata[RESERVED_METADATA_ARG.PROFILE_START_TIME.value],
                    metadata[RESERVED_METADATA_ARG.PROFILE_END_TIME.value],
                )
            ]
            summary = self._gen_summary_stats(run_durations)
            data_rows.append(
                [
                    "Execution Time",
                    len(run_durations),
                    summary[RESERVED_METADATA_ARG.PROFILE_SUMMARY_AVERAGE.value],
                    summary[RESERVED_METADATA_ARG.PROFILE_SUMMARY_MIN.value],
                    summary[RESERVED_METADATA_ARG.PROFILE_SUMMARY_P10.value],
                    summary[RESERVED_METADATA_ARG.PROFILE_SUMMARY_P90.value],
                    summary[RESERVED_METADATA_ARG.PROFILE_SUMMARY_MAX.value],
                ]
            )
        return header_row + sorted(data_rows, key=lambda x: x[2], reverse=True)

    # Generate summary stats across profiling runs
    def _gen_run_summary_individual_runs_stats(self) -> List[Any]:
        metadata = self.metadata
        if metadata is None:
            return []

        run_durations = [
            (end - start) / (1000 * 1000)
            for start, end in zip(
                metadata[RESERVED_METADATA_ARG.PROFILE_START_TIME.value],
                metadata[RESERVED_METADATA_ARG.PROFILE_END_TIME.value],
            )
        ]

        header_row = [["Run #", "Run Duration (ms)"]]
        data_rows = [(index, value) for index, value in enumerate(run_durations)]

        return header_row + data_rows

    # Generate summary stats grouped by operator type
    def _gen_op_summary_stats(self) -> List[Any]:
        grouped_ops = {}

        def gen_stats(node):
            if not isinstance(node, OperatorNode):
                return

            metadata = node.metadata

            if metadata is None:
                return

            metrics = metadata.get(RESERVED_METADATA_ARG.METRICS_KEYWORD.value)
            if metrics is None:
                return
            if metadata is not None:
                run_durations = [
                    end - start
                    for start, end in zip(
                        metadata[RESERVED_METADATA_ARG.PROFILE_START_TIME.value],
                        metadata[RESERVED_METADATA_ARG.PROFILE_END_TIME.value],
                    )
                ]
                if node.op in grouped_ops:
                    grouped_ops[node.op] = (
                        grouped_ops[node.op][0] + run_durations,
                        grouped_ops[node.op][1] + 1,
                    )
                else:
                    grouped_ops[node.op] = (np.array(run_durations), 1)

        for element in self.elements:
            if (
                not isinstance(element, OperatorGraph)
                or element.graph_name != "forward"
            ):
                continue

            for node in element.elements:
                if isinstance(node, OperatorGraphWithStats):
                    for node_with_stats in node.elements:
                        gen_stats(node_with_stats)
                else:
                    gen_stats(node)

        header_row = [
            [
                PROFILE_STAT_HEADER.NAME.value,
                PROFILE_STAT_HEADER.COUNT.value,
                PROFILE_STAT_HEADER.MEAN_MS.value,
                PROFILE_STAT_HEADER.MIN_MS.value,
                PROFILE_STAT_HEADER.P10_MS.value,
                PROFILE_STAT_HEADER.P90_MS.value,
                PROFILE_STAT_HEADER.MAX_MS.value,
            ]
        ]
        data_rows = []
        for op, (durations, count) in grouped_ops.items():
            summary = self._gen_summary_stats(durations)
            data_rows.append(
                [
                    op,
                    count,
                    summary[RESERVED_METADATA_ARG.PROFILE_SUMMARY_AVERAGE.value],
                    summary[RESERVED_METADATA_ARG.PROFILE_SUMMARY_MIN.value],
                    summary[RESERVED_METADATA_ARG.PROFILE_SUMMARY_P10.value],
                    summary[RESERVED_METADATA_ARG.PROFILE_SUMMARY_P90.value],
                    summary[RESERVED_METADATA_ARG.PROFILE_SUMMARY_MAX.value],
                ]
            )
        return header_row + sorted(data_rows, key=lambda x: x[2], reverse=True)

    # Given a list of (ns) run_durations, return a dict of summary stats in (ms)
    def _gen_summary_stats(self, run_durations: List[Any]) -> Dict[str, Any]:
        nano_in_milli = 1000000
        return {
            RESERVED_METADATA_ARG.PROFILE_SUMMARY_COLDSTART.value: int(run_durations[0])
            / nano_in_milli,
            RESERVED_METADATA_ARG.PROFILE_SUMMARY_AVERAGE.value: int(
                np.average(run_durations)
            )
            / nano_in_milli,
            RESERVED_METADATA_ARG.PROFILE_SUMMARY_P90.value: int(
                np.percentile(run_durations, 90)
            )
            / nano_in_milli,
            RESERVED_METADATA_ARG.PROFILE_SUMMARY_P10.value: int(
                np.percentile(run_durations, 10)
            )
            / nano_in_milli,
            RESERVED_METADATA_ARG.PROFILE_SUMMARY_MIN.value: int(np.min(run_durations))
            / nano_in_milli,
            RESERVED_METADATA_ARG.PROFILE_SUMMARY_MAX.value: int(np.max(run_durations))
            / nano_in_milli,
        }

    # Extract the summary stats from each node instance into the top level metadata
    def _extract_node_instance_stats(self) -> List[Any]:
        header_row = [
            [
                PROFILE_STAT_HEADER.NAME.value,
                PROFILE_STAT_HEADER.COLD_START_MS.value,
                PROFILE_STAT_HEADER.MEAN_MS.value,
                PROFILE_STAT_HEADER.MIN_MS.value,
                PROFILE_STAT_HEADER.P10_MS.value,
                PROFILE_STAT_HEADER.P90_MS.value,
                PROFILE_STAT_HEADER.MAX_MS.value,
            ]
        ]

        def extract_node_stats(node):
            if not isinstance(node, OperatorNode):
                return

            metadata = node.metadata
            if metadata is not None:
                metrics = metadata.get(RESERVED_METADATA_ARG.METRICS_KEYWORD.value)
                if metrics is None:
                    return
                data_rows.append(
                    [
                        node.name,
                        metrics[RESERVED_METADATA_ARG.PROFILE_SUMMARY_COLDSTART.value],
                        metrics[RESERVED_METADATA_ARG.PROFILE_SUMMARY_AVERAGE.value],
                        metrics[RESERVED_METADATA_ARG.PROFILE_SUMMARY_MIN.value],
                        metrics[RESERVED_METADATA_ARG.PROFILE_SUMMARY_P10.value],
                        metrics[RESERVED_METADATA_ARG.PROFILE_SUMMARY_P90.value],
                        metrics[RESERVED_METADATA_ARG.PROFILE_SUMMARY_MAX.value],
                    ]
                )

        data_rows = []
        for element in self.elements:
            if (
                not isinstance(element, OperatorGraph)
                or element.graph_name != "forward"
            ):
                continue

            for node in element.elements:
                if isinstance(node, OperatorGraphWithStats):
                    for node_with_stats in node.elements:
                        extract_node_stats(node_with_stats)
                else:
                    extract_node_stats(node)

        return header_row + data_rows

    # Populate and associate node and graph level metadatas based on ET Dump Run
    def attach_metadata(
        self, inference_run: InferenceRun, include_summaries: bool = True
    ) -> None:
        for element in self.elements:
            # Recursively attach metadata to nodes
            if isinstance(element, OperatorGraphWithStats):
                element.attach_metadata(
                    inference_run, include_summaries=include_summaries
                )
            if isinstance(element, OperatorNode) and element.metadata is not None:
                metadata = element.metadata
                debug_handle = metadata.get(RESERVED_METADATA_ARG.DEBUG_HANDLE.value)
                if debug_handle is not None:
                    inference_metadata = inference_run.node_metadata.get(debug_handle)
                    if inference_metadata is not None:
                        metadata.update(inference_metadata)

                        # Write Node Instance Summary across Runs
                        if include_summaries:
                            run_durations = [
                                end - start
                                for start, end in zip(
                                    metadata[
                                        RESERVED_METADATA_ARG.PROFILE_START_TIME.value
                                    ],
                                    metadata[
                                        RESERVED_METADATA_ARG.PROFILE_END_TIME.value
                                    ],
                                )
                            ]
                            metadata.update(
                                {
                                    RESERVED_METADATA_ARG.METRICS_KEYWORD.value: self._gen_summary_stats(
                                        run_durations
                                    )
                                }
                            )

        # Attach Run level metadata
        if self.graph_name == "base":
            if self.metadata is None:
                self.metadata = inference_run.run_metadata
            else:
                self.metadata.update(inference_run.run_metadata)

            if include_summaries:
                tables = {
                    RESERVED_METADATA_ARG.TABLES_KEYWORD.value: {
                        RESERVED_METADATA_ARG.OP_INSTANCE_SUMMARY_TABLE.value: self._extract_node_instance_stats(),
                        RESERVED_METADATA_ARG.AGGREGATED_OP_TABLE.value: self._gen_op_summary_stats(),
                        RESERVED_METADATA_ARG.RUN_SUMMARY_TABLE.value: self._gen_run_summary_stats(),
                        RESERVED_METADATA_ARG.RUN_SUMMARY_INDIVIDUAL_RUNS_TABLE.value: self._gen_run_summary_individual_runs_stats(),
                    },
                    RESERVED_METADATA_ARG.KV_KEYWORD.value: {
                        RESERVED_METADATA_ARG.MODEL_LOAD_TIME_KEY.value: self._gen_model_load_time()
                    },
                }

                # Pyre is unable to infer the type of self.metadata here
                assert self.metadata is not None
                self.metadata.update(tables)


# Representation of an FX GraphModule as an OperatorGraph
class FXOperatorGraph(OperatorGraphWithStats):
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
            source_fn := metadata.get("source_fn")
        ) is not None and "nn_module_stack" in metadata:
            # (module name, module type)
            module_type = (
                source_fn[1] if isinstance(source_fn[1], str) else source_fn[1].__name__
            )
            module_mapping[(source_fn[0], module_type)].append(node)

    @staticmethod
    def _parse_args(
        node: torch.fx.Node,
        nodes: Dict[str, Node],
        const_count: int,
        module_mapping: Dict[Tuple[str, str], List[Node]],
    ) -> Tuple[List[Node], int]:
        inputs = []
        op = node.op
        name = node.name
        args = node.args
        kwargs = node.kwargs

        for arg in args:
            if isinstance(arg, torch.fx.node.Node):
                arg_name = FXOperatorGraph._get_node_name(arg)
            elif isinstance(arg, (int, float, torch.dtype)):
                # e.g. The "0" from node.args of squeeze_copy (mm_default, 0)
                arg_name = "CONST_" + str(const_count)
                const_count += 1
                const_node = ValueNode(arg_name, val=str(arg))
                nodes[arg_name] = const_node
                FXOperatorGraph._update_module_mapping(
                    const_node, module_mapping, node.meta
                )
            elif isinstance(arg, list):
                arg_name: List[str] = []
                for list_arg in arg:
                    if isinstance(list_arg, (int, float)):
                        # Consider the whole list of ints/floats as a single constant and
                        # stringify that.
                        arg_name += ["CONST_" + str(const_count)]
                        const_count += 1
                        const_node = ValueNode(arg_name[-1], val=str(arg))
                        nodes[arg_name[-1]] = const_node
                        FXOperatorGraph._update_module_mapping(
                            const_node, module_mapping, node.meta
                        )
                    elif isinstance(list_arg, torch.fx.node.Node):
                        arg_name += [FXOperatorGraph._get_node_name(list_arg)]
                    else:
                        raise Exception(
                            f"Unsupported argument encountered in list {arg}, {type(arg[0])}"
                        )
            else:
                raise Exception(f"Unsupported argument encountered {op}, {name}, {arg}")

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
                warnings.warn(f"Unsupported kwarg encountered: {name}, {kwargs}")

        return inputs, const_count

    # Given an FX GraphModule, parse it into an OperatorGraph
    @staticmethod
    def gen_operator_graph(
        model: torch.fx.GraphModule, skip_stack_trace: Optional[bool] = False
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
            dtype = fx_node.type
            target = fx_node.target
            args = fx_node.args
            kwargs = fx_node.kwargs
            metadata = FXOperatorGraph._extract_metadata(fx_node.meta, skip_stack_trace)
            output_shapes = FXOperatorGraph._extract_output_shapes(fx_node.meta)

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
                    dtype=dtype,
                    val=args,
                )  # val is default arg
                input_nodes[name] = node
            # Constants
            elif op == "get_attr":
                node = ValueNode(
                    name, output_shapes=output_shapes, metadata=metadata, dtype=dtype
                )
            # Output
            elif op == "output":
                assert len(args) == 1
                # Args of op=='output' is a wrapped list of return nodes ([ret_1, ret_2, ...], )
                in_nodes = [
                    nodes[FXOperatorGraph._get_node_name(ret)] for ret in args[0]
                ]
                node = ValueNode(
                    name,
                    inputs=in_nodes,
                    output_shapes=output_shapes,
                    metadata=metadata,
                    dtype=dtype,
                )
                output_nodes[name] = node
            # Op Calls
            elif op == "call_function":
                inputs, const_count = FXOperatorGraph._parse_args(
                    fx_node, nodes, const_count, module_mapping
                )
                node = OperatorNode(
                    name,
                    inputs=inputs,
                    output_shapes=output_shapes,
                    metadata=metadata,
                    op=FXOperatorGraph._get_op_name(fx_node),
                )
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
            module_element = OperatorGraphWithStats(
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
        return ret

    # Not yet implemented
    @staticmethod
    def _extract_output_shapes(metadata: Dict[str, Any]) -> Optional[List[List[int]]]:
        return None


# Representation of an Exported ExecuTorch Program as an OperatorGraph
class ExportedETOperatorGraph(OperatorGraphWithStats):
    # Given the path to an Executorch Program, parse it into an OperatorGraph
    # `include_constant_nodes`: Whether to include constant nodes in output graph
    @staticmethod
    def gen_operator_graph_from_path(
        file_path: str, include_constant_nodes=True
    ) -> ExportedETOperatorGraph:
        with open(file_path, "rb") as fd:
            program = deserialize_from_flatbuffer(fd.read())
            return ExportedETOperatorGraph.gen_operator_graph(
                program, include_constant_nodes
            )

    # Given a list of Kernel arguments and Evals, unpack the TensorList arguments
    # Then, discern the input and outputs indices
    @staticmethod
    def _resolve_kernel_args(args, values):
        # Unpack TensorList arguments
        unpacked_args = []
        for arg in args[:-1]:
            val = values[arg].val
            if isinstance(val, TensorList):
                unpacked_args += val.items
            else:
                unpacked_args.append(arg)

        # Remove output indices from the input args
        # This is due to duplication of output args in arg lists
        if args[-1] in unpacked_args:
            unpacked_args.remove(args[-1])

        out_vals = values[args[-1]].val
        if isinstance(out_vals, TensorList):
            # De-dup based on unpacked output args in arg lists
            unpacked_args = [idx for idx in unpacked_args if idx not in out_vals.items]
            out_indices = out_vals.items
        else:
            out_indices = [args[-1]]

        return unpacked_args, out_indices

    # Given an Executorch Program, parse it into an OperatorGraph
    # `include_constant_nodes`: Whether to include constant nodes in output graph
    @staticmethod
    def gen_operator_graph(
        model: Program, include_constant_nodes=True
    ) -> ExportedETOperatorGraph:
        plan = model.execution_plan[0]
        chain = plan.chains[0]
        operators = plan.operators
        instructions = chain.instructions
        values = plan.values

        # Maps from plan.Values index to Node representation
        index_to_node = {}
        const_count = 0

        # Input Nodes
        for i, arg_index in enumerate(plan.inputs):
            node = ExportedETOperatorGraph.gen_constant_node(
                "in_" + str(i), arg_index, values[arg_index]
            )
            index_to_node[arg_index] = node

        # Op and Value Nodes
        dup_nodes = []
        for index, instruction in list(enumerate(instructions)):
            if isinstance(instruction.instr_args, KernelCall):
                kernel = instruction.instr_args
                op = operators[kernel.op_index].name
                args = kernel.args

                # Get the true Kernel Arg inputs and outputs
                (
                    unpacked_args,
                    out_indices,
                ) = ExportedETOperatorGraph._resolve_kernel_args(args, values)

                inputs = []
                for arg in unpacked_args:
                    in_node = index_to_node.get(arg, None)

                    # Create constant nodes if not previously defined
                    if in_node is None:
                        in_node = ExportedETOperatorGraph.gen_constant_node(
                            "const_" + str(const_count), arg, values[arg]
                        )
                        const_count += 1
                        index_to_node[arg] = in_node

                    inputs.append(in_node)

                output_shapes = []
                for out_index in out_indices:
                    value = values[out_index].val
                    if isinstance(value, schema.Tensor):
                        output_shapes += [value.sizes]
                    else:
                        output_shapes += []

                node = OperatorNode(
                    name=op + "_" + str(index),
                    op=op,
                    inputs=inputs,
                    output_shapes=output_shapes,
                    metadata={
                        "arg": str(args),
                        RESERVED_METADATA_ARG.DEBUG_HANDLE.value: index,
                    },
                )
                for out_index in out_indices:
                    index_to_node[out_index] = node

                # When there are multiple output idx, track duplicates to remove later
                if len(out_indices) > 0:
                    dup_nodes += out_indices[1:]

        # Remove Duplicate Nodes
        for dup in dup_nodes:
            index_to_node.pop(dup)

        # Output Graph
        out_nodes = []
        for i, arg_index in enumerate(plan.outputs):
            node = ExportedETOperatorGraph.gen_constant_node(
                "out_" + str(i), arg_index, plan.values[arg_index]
            )
            node.inputs = [index_to_node[arg_index]]
            out_nodes.append(node)
        output_graph = ExportedETOperatorGraph(graph_name="outputs", elements=out_nodes)

        # Input Graph
        input_graph = ExportedETOperatorGraph(
            graph_name="inputs",
            elements=[index_to_node.pop(arg) for arg in plan.inputs],
        )

        # Plan Graph
        plan_nodes = [
            node
            for node in index_to_node.values()
            if (include_constant_nodes or not isinstance(node, ValueNode))
        ]
        plan_graph = ExportedETOperatorGraph(graph_name=plan.name, elements=plan_nodes)

        return ExportedETOperatorGraph(
            graph_name="base", elements=[input_graph, plan_graph, output_graph]
        )

    # Parser for https://www.internalfb.com/code/fbsource/fbcode/executorch/exir/schema.py?lines=115
    @staticmethod
    def parse_value(e_value: schema.EValue) -> Tuple[str, str]:
        value = e_value.val
        if isinstance(value, schema.Tensor):
            return ("Tensor<" + str(value.scalar_type) + ">", str(value.sizes))
        if isinstance(value, schema.Int):
            return ("Int", str(value.int_val))
        if isinstance(value, schema.Double):
            return ("Double", str(value.double_val))
        if isinstance(value, schema.Bool):
            return ("Bool", str(value.bool_val))
        if isinstance(value, schema.String):
            return ("String", value.string_val)
        if isinstance(value, schema.IntList):
            return ("IntList", str(value.items))
        if isinstance(value, schema.DoubleList):
            return ("DoubleList", str(value.items))
        if isinstance(value, schema.BoolList):
            return ("BoolList", str(value.items))
        if isinstance(value, schema.TensorList):
            return ("TensorList", str(value.items))
        return ("other_dtype", "other_val")

    # Create a generic ValueNode in an ExportedETOperatorGraph
    @staticmethod
    def gen_constant_node(
        name: str, arg_index: int, e_value: schema.EValue
    ) -> ValueNode:
        (dtype, val) = ExportedETOperatorGraph.parse_value(e_value)
        return ValueNode(
            name=name,
            dtype=dtype,
            val=val,
            metadata={"arg_index": arg_index},
        )

    # Value equality comparison
    def __eq__(self, other: ExportedETOperatorGraph) -> bool:
        if not isinstance(other, ExportedETOperatorGraph):
            return False

        if self.graph_name != other.graph_name or self.metadata != other.metadata:
            return False

        self_elements = self.elements
        other_elements = other.elements

        if len(self_elements) != len(other_elements):
            return False

        for node in self_elements:
            if node not in other_elements:
                return False

        return True
