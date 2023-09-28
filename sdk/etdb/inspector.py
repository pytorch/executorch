# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import dataclasses
import logging
from collections import defaultdict, OrderedDict
from dataclasses import dataclass
from typing import (
    Dict,
    List,
    Mapping,
    Optional,
    Sequence,
    Tuple,
    TypeAlias,
    TypedDict,
    Union,
)

import numpy as np
import pandas as pd
import torch
from executorch.exir import ExportedProgram

from executorch.sdk.edir.et_schema import OperatorGraphWithStats, OperatorNode
from executorch.sdk.etdb._inspector_utils import (
    create_debug_handle_to_op_node_mapping,
    EDGE_DIALECT_GRAPH_KEY,
    gen_etdump_object,
    gen_etrecord_object,
    gen_graphs_from_etrecord,
)
from executorch.sdk.etdump.schema_flatcc import ETDumpFlatCC, ProfileEvent

from tabulate import tabulate


FORWARD = "forward"
RESERVED_SPECIAL_EVENT_NAMES = [
    "Method::init",
    "Program::load_method",
    "Method::execute",
]
EXCLUDED_COLUMNS_WHEN_PRINTING = [
    "raw",
    "delegate_debug_identifier",
    "stack_traces",
    "module_hierarchy",
    "debug_data",
]


log: logging.Logger = logging.getLogger(__name__)

# Signature of a ProfileEvent
@dataclass(frozen=True, order=True)
class ProfileEventSignature:
    name: str
    instruction_id: Optional[int]
    delegate_id: Optional[int] = None
    delegate_id_str: Optional[str] = None

    @staticmethod
    def _gen_from_event(event: ProfileEvent) -> "ProfileEventSignature":
        """
        Given a ProfileEvent, extract the fields into a signature

        ProfileEvents from ETDump default to "" and -1 when the field is not populated
        The Signature will convert these back to the intended None value
        """
        return ProfileEventSignature(
            event.name or "",
            event.instruction_id if event.instruction_id != -1 else None,
            event.delegate_debug_id_int if event.delegate_debug_id_int != -1 else None,
            event.delegate_debug_id_str if event.delegate_debug_id_str != "" else None,
        )


# Signature of a RunData as defined by its ProfileEvents
RunSignature: TypeAlias = Tuple[ProfileEventSignature]


# Typing for mapping Event.delegate_debug_identifiers to debug_handle(s)
DelegateIdentifierDebugHandleMap: TypeAlias = Union[
    Mapping[int, Tuple[int, ...]], Mapping[str, Tuple[int, ...]]
]

# Typing for Dict containig delegate metadata
DelegateMetadata = TypedDict(
    "DelegateMetadata",
    {"name": str, "delegate_map": DelegateIdentifierDebugHandleMap},
)


@dataclass
class PerfData:
    def __init__(self, raw: List[float]):
        self.raw: List[float] = raw

    @property
    def p50(self) -> float:
        return np.percentile(self.raw, 50)

    @property
    def p90(self) -> float:
        return np.percentile(self.raw, 90)

    @property
    def avg(self) -> float:
        return np.mean(self.raw)

    @property
    def min(self) -> float:
        return min(self.raw)

    @property
    def max(self) -> float:
        return max(self.raw)

    @property
    def median(self) -> float:
        return np.median(self.raw)


# TODO: detailed documentation
@dataclass
class Event:
    """
    Corresponds to an op instance
    """

    name: str
    perf_data: PerfData
    op_types: List[str] = dataclasses.field(default_factory=list)

    # Instruction Id of the original profiling event
    instruction_id: Optional[int] = None

    # Supplemental Identifier used in combination with instruction_identifier
    delegate_debug_identifier: Optional[Union[int, str]] = None

    # Debug Handles in the model graph to which this event is correlated
    debug_handles: Optional[Union[int, Sequence[int]]] = None

    stack_traces: Dict[str, str] = dataclasses.field(default_factory=dict)
    module_hierarchy: Dict[str, Dict] = dataclasses.field(default_factory=dict)
    is_delegated_op: Optional[bool] = None
    delegate_backend_name: Optional[str] = None
    debug_data: List[torch.Tensor] = dataclasses.field(default_factory=list)

    @staticmethod
    def _gen_from_profile_events(
        signature: ProfileEventSignature, events: List[ProfileEvent]
    ) -> "Event":
        """
        Given a ProfileEventSignature and a list of ProfileEvents with that signature,
        return an Event object matching the ProfileEventSignature, with perf_data
        populated from the list of ProfileEvents
        """
        if signature.delegate_id is not None:  # 0 is a valid value
            delegate_debug_identifier = signature.delegate_id
        else:
            delegate_debug_identifier = signature.delegate_id_str or None

        # Use the delegate identifier as the event name if delegated
        is_delegated_op = delegate_debug_identifier is not None
        name = signature.name if not is_delegated_op else str(delegate_debug_identifier)

        perf_data = PerfData(
            [float(event.end_time - event.start_time) / 1000 for event in events]
        )

        return Event(
            name=name,
            perf_data=perf_data,
            instruction_id=signature.instruction_id,
            delegate_debug_identifier=delegate_debug_identifier,
            is_delegated_op=is_delegated_op,
        )

    def _associate_with_op_graph_nodes(
        self, debug_handle_to_op_node_map: Dict[int, OperatorNode]
    ) -> None:
        """
        Helper function to populate the stack_traces, module_hierarchy and op_types attributes
        based on the debug handles of this event
        """
        if (debug_handles := self.debug_handles) is None:
            return

        if isinstance(debug_handles, int):
            debug_handles = [debug_handles]

        for handle in debug_handles:
            node = debug_handle_to_op_node_map.get(handle)
            if node is not None and (metadata := node.metadata) is not None:
                self.stack_traces[node.name] = metadata.get("stack_trace")
                self.module_hierarchy[node.name] = metadata.get("nn_module_stack")
                if node.op:
                    # TODO: consider having this as a dict from node.name -> node.op
                    self.op_types += [node.op]


@dataclass
class EventBlock:
    """
    EventBlock contains a collection of events associated with a particular profiling/debugging block retrieved from the runtime.
    Attributes:
        name (str): Name of the profiling/debugging block
        events (List[Event]): List of events associated with the profiling/debugging block
    """

    name: str
    events: List[Event] = dataclasses.field(default_factory=list)

    def to_dataframe(self) -> pd.DataFrame:
        """
        Converts the EventBlock into a DataFrame with each row being an event instance
        """
        # TODO: push row generation down to Event
        data = {
            "event_block_name": [self.name] * len(self.events),
            "event_name": [event.name for event in self.events],
            "raw": [event.perf_data.raw for event in self.events],
            "p50": [event.perf_data.p50 for event in self.events],
            "p90": [event.perf_data.p90 for event in self.events],
            "avg": [event.perf_data.avg for event in self.events],
            "min": [event.perf_data.min for event in self.events],
            "max": [event.perf_data.max for event in self.events],
            "median": [event.perf_data.median for event in self.events],
            "op_types": [event.op_types for event in self.events],
            "delegate_debug_identifier": [
                event.delegate_debug_identifier for event in self.events
            ],
            "stack_traces": [event.stack_traces for event in self.events],
            "module_hierarchy": [event.module_hierarchy for event in self.events],
            "is_delegated_op": [event.is_delegated_op for event in self.events],
            "delegate_backend_name": [
                event.delegate_backend_name for event in self.events
            ],
            "debug_data": [event.debug_data for event in self.events],
        }
        df = pd.DataFrame(data)
        return df

    @staticmethod
    def _gen_from_etdump(etdump: ETDumpFlatCC) -> List["EventBlock"]:
        """
        Given an etdump, generate a list of EventBlocks corresponding to the
        contents
        """

        # Group all the RunData by the set of profile events
        profile_run_groups: Mapping[
            RunSignature,
            OrderedDict[ProfileEventSignature, List[ProfileEvent]],
        ] = defaultdict(OrderedDict)
        for run in etdump.run_data:
            if (run_events := run.events) is None:
                continue

            # Identify all the ProfileEventSignatures
            profile_events: OrderedDict[
                ProfileEventSignature, ProfileEvent
            ] = OrderedDict()
            for event in run_events:
                if (profile_event := event.profile_event) is not None:
                    signature = ProfileEventSignature._gen_from_event(profile_event)
                    profile_events[signature] = profile_event

            # Create a RunSignature from the ProfileEventSignature found
            run_signature: RunSignature = tuple(profile_events.keys())

            # Update the Profile Run Groups, indexed on the RunSignature
            run_signature_events: OrderedDict[
                ProfileEventSignature, List[ProfileEvent]
            ] = profile_run_groups[run_signature]
            for event_signature, event in profile_events.items():
                run_signature_events.setdefault(event_signature, []).append(event)

        # Create EventBlocks from the Profile Run Groups
        return [
            EventBlock(
                name=str(index),
                events=[
                    Event._gen_from_profile_events(signature, event)
                    for signature, event in profile_events.items()
                ],
            )
            for index, profile_events in enumerate(profile_run_groups.values())
        ]

    # TODO: Considering changing ETRecord deserialization logic to cast the ints in string format to actual ints
    def _gen_resolve_debug_handles(
        self,
        handle_map: Dict[str, List[int]],
        delegate_map: Optional[Dict[str, DelegateMetadata]] = None,
    ):
        """
        Given mappings from instruction id to debug handles, populate the
        debug_handles field of all underlying events

        If the event is delegated, index with the instruction_id and delegate_debug_identifier
        to obtain the debug_handle via the delegate map
        """
        for event in self.events:
            # Check if instruction_id is present in the event
            if event.instruction_id is None:
                continue

            # Check for the instruction_id in handle map
            if (instruction_id := str(event.instruction_id)) not in handle_map:
                continue

            # For non-delegated event, handles are found in handle_map
            if (delegate_debug_id := event.delegate_debug_identifier) is None:
                event.debug_handles = handle_map[instruction_id]
                continue

            # Check that the delegated event has a corresponding mapping
            if (
                delegate_map is None
                or (delegate_metadata := delegate_map.get(instruction_id)) is None
            ):
                event.debug_handles = handle_map[instruction_id]
                log.warning(
                    f" No delegate mapping found for delegate with instruction id {event.instruction_id}"
                )
                continue

            # For delegated events, handles are found via delegateMetadata
            event.delegate_backend_name = delegate_metadata.get("name", "")
            delegate_metadata_delegate_map = delegate_metadata.get("delegate_map", {})

            # delegate_debug_id can be either int based or string based, therefore we need to check both
            debug_handles = delegate_metadata_delegate_map.get(
                delegate_debug_id  # pyre-ignore
            )
            if debug_handles is not None:
                event.debug_handles = debug_handles
            else:
                event.debug_handles = delegate_metadata_delegate_map.get(
                    str(delegate_debug_id)  # pyre-ignore
                )


class Inspector:
    """
    APIs for examining model architecture and performance stats.

    Public Attributes:
        event_blocks: List["EventBlocks"]. Structured data accessible through Inspector for analysis.

    Private Attributes:
        _etrecord: Optional[ETRecord]. File under etrecord_path deserialized into an object.
        _op_graph_dict: Mapping[str, OperatorGraphWithStats]. Graph objects parsed from etrecord matched with user defined graph names.
    """

    def __init__(
        self, etdump_path: Optional[str] = None, etrecord_path: Optional[str] = None
    ) -> None:
        """
        Create an inspector instance from the provided ETDump/ETRecord
        """

        # TODO: etrecord_path can be optional, so need to support the case when it is not present
        self._etrecord = gen_etrecord_object(etrecord_path=etrecord_path)
        etdump = gen_etdump_object(etdump_path=etdump_path)
        self.event_blocks = EventBlock._gen_from_etdump(etdump)

        self._op_graph_dict: Mapping[
            str, OperatorGraphWithStats
        ] = gen_graphs_from_etrecord(etrecord=self._etrecord)

        # Use the delegate map from etrecord, associate debug handles with each event
        for event_block in self.event_blocks:
            event_block._gen_resolve_debug_handles(
                self._etrecord._debug_handle_map[FORWARD],
                self._etrecord._delegate_map[FORWARD]
                if self._etrecord._delegate_map is not None
                else None,
            )

        # Traverse the edge dialect op graph to create mapping from debug_handle to op node
        debug_handle_to_op_node_map = {}
        create_debug_handle_to_op_node_mapping(
            self._op_graph_dict[EDGE_DIALECT_GRAPH_KEY],
            debug_handle_to_op_node_map,
        )

        for event_block in self.event_blocks:
            for event in event_block.events:
                event._associate_with_op_graph_nodes(debug_handle_to_op_node_map)

    def print_data_tabular(self) -> None:
        """
        Prints the underlying EventBlocks (essentially all the performance data)
        """

        def style_text_size(val, size=12):
            return f"font-size: {size}px"

        df_list = [event_block.to_dataframe() for event_block in self.event_blocks]
        combined_df = pd.concat(df_list, ignore_index=True)
        # Filter out some columns for better readability when printing
        filtered_df = combined_df.drop(columns=EXCLUDED_COLUMNS_WHEN_PRINTING)
        try:
            from IPython.display import display

            styled_df = filtered_df.style.applymap(style_text_size)
            display(styled_df)
        except:
            # TODO: figure out how to trigger this path in python shell
            print(tabulate(filtered_df, headers="keys", tablefmt="fancy_grid"))

    # TODO: write unit test
    def find_total_for_module(self, module_name: str):
        total = 0.0
        for block in self.event_blocks:
            for event in block.events:
                module_hierarchy = event.module_hierarchy.values()
                for hierarchy in module_hierarchy:
                    if not hierarchy:
                        continue
                    found = any(module_name in key for key in hierarchy.keys())
                    if found:
                        total += event.perf_data.avg
                        break
        return total

    def get_event_blocks(self) -> List[EventBlock]:
        """
        Returns EventBlocks containing contents from ETDump (correlated to ETRecord if provided)
        """
        return self.event_blocks

    def get_op_list(
        self, event_block: str, show_delegated_ops: Optional[bool] = True
    ) -> Dict[str, List[Event]]:
        """
        Return a map of op_types to Events of that op_type
        """
        # TODO: implement
        return {}

    def write_tensorboard_artifact(self, path: str) -> None:
        """
        Write to the provided path, the artifacts required for visualization in TensorBoard
        """
        # TODO: implement
        pass

    def get_exported_program(self, graph: Optional[str] = None) -> ExportedProgram:
        """
        Access helper for ETRecord, defaults to returning Edge Dialect Program

        Args:
            graph: Name of the graph to access. If None, returns the Edge Dialect Program.
        """
        if graph is None:
            return self._etrecord.edge_dialect_program
        return self._etrecord.graph_map.get(graph)
