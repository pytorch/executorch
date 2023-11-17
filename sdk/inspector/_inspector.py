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

from executorch.sdk.debug_format.et_schema import OperatorGraph, OperatorNode
from executorch.sdk.etdump.schema_flatcc import ETDumpFlatCC, ProfileEvent
from executorch.sdk.etrecord import ETRecord, parse_etrecord
from executorch.sdk.inspector._inspector_utils import (
    create_debug_handle_to_op_node_mapping,
    EDGE_DIALECT_GRAPH_KEY,
    EXCLUDED_COLUMNS_WHEN_PRINTING,
    EXCLUDED_EVENTS_WHEN_PRINTING,
    FORWARD,
    gen_etdump_object,
    gen_graphs_from_etrecord,
    RESERVED_FRAMEWORK_EVENT_NAMES,
    TIME_SCALE_DICT,
    TimeScale,
)

from tabulate import tabulate


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
    def p10(self) -> float:
        return np.percentile(self.raw, 10)

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


# TODO: detailed documentation
@dataclass
class Event:
    """
    An Event corresponds to an operator instance with perf data retrieved from the runtime and other metadata from `ETRecord`.

    Args:
        name: Name of the profiling/debugging `Event`.
        perf_data: Performance data associated with the event retrived from the runtime (available attributes: p10, p50, p90, avg, min and max).
        op_type: List of op types corresponding to the event.
        delegate_debug_identifier: Supplemental identifier used in combination with instruction id.
        debug_handles: Debug handles in the model graph to which this event is correlated.
        stack_trace: A dictionary mapping the name of each associated op to its stack trace.
        module_hierarchy: A dictionary mapping the name of each associated op to its module hierarchy.
        is_delegated_op: Whether or not the event was delegated.
        delegate_backend_name: Name of the backend this event was delegated to.
        debug_data: Intermediate data collected during runtime.
        delegate_debug_metadatas: A list of delegate debug metadata in string, one for each profile event.
    """

    name: str
    perf_data: PerfData
    op_types: List[str] = dataclasses.field(default_factory=list)
    delegate_debug_identifier: Optional[Union[int, str]] = None
    debug_handles: Optional[Union[int, Sequence[int]]] = None
    stack_traces: Dict[str, str] = dataclasses.field(default_factory=dict)
    module_hierarchy: Dict[str, Dict] = dataclasses.field(default_factory=dict)
    is_delegated_op: Optional[bool] = None
    delegate_backend_name: Optional[str] = None
    debug_data: List[torch.Tensor] = dataclasses.field(default_factory=list)
    delegate_debug_metadatas: List[str] = dataclasses.field(default_factory=list)

    _instruction_id: Optional[int] = None

    @staticmethod
    def _gen_from_profile_events(
        signature: ProfileEventSignature,
        events: List[ProfileEvent],
        scale_factor: float = 1.0,
    ) -> "Event":
        """
        Given a ProfileEventSignature and a list of ProfileEvents with that signature,
        return an Event object matching the ProfileEventSignature, with perf_data
        populated from the list of ProfileEvents

        An optional inverse scale factor can be provided to adjust the event timestamps
        """
        if signature.delegate_id is not None:  # 0 is a valid value
            delegate_debug_identifier = signature.delegate_id
        else:
            delegate_debug_identifier = signature.delegate_id_str or None

        # Use the delegate identifier as the event name if delegated
        is_delegated_op = delegate_debug_identifier is not None
        name = signature.name if not is_delegated_op else str(delegate_debug_identifier)

        perf_data = PerfData(
            [
                float(event.end_time - event.start_time) / scale_factor
                for event in events
            ]
        )

        delegate_debug_metadatas = [
            event.delegate_debug_metadata if event.delegate_debug_metadata else ""
            for event in events
        ]

        return Event(
            name=name,
            perf_data=perf_data,
            delegate_debug_identifier=delegate_debug_identifier,
            is_delegated_op=is_delegated_op,
            delegate_debug_metadatas=delegate_debug_metadatas,
            _instruction_id=signature.instruction_id,
        )

    def _associate_with_op_graph_nodes(
        self,
        debug_handle_to_op_node_map: Dict[int, OperatorNode],
    ) -> None:
        """
        Helper function to populate the stack_traces, module_hierarchy and op_types attributes
        based on the debug handles of this event
        """

        # Framework events aren't logically associated with any nodes
        if self.name in RESERVED_FRAMEWORK_EVENT_NAMES:
            return

        if (debug_handles := self.debug_handles) is None:
            return

        if isinstance(debug_handles, int):
            debug_handles = [debug_handles]

        for handle in debug_handles:
            node = debug_handle_to_op_node_map.get(handle)
            # Attach node metadata including stack traces, module hierarchy and op_types to this event
            if node is not None and (metadata := node.metadata) is not None:
                self.stack_traces[node.name] = metadata.get("stack_trace")
                self.module_hierarchy[node.name] = metadata.get("nn_module_stack")
                if node.op:
                    # TODO: consider having this as a dict from node.name -> node.op
                    self.op_types += [node.op]


@dataclass
class EventBlock:
    r"""
    An `EventBlock` contains a collection of events associated with a particular profiling/debugging block retrieved from the runtime.
    Each `EventBlock` represents a pattern of execution. For example, model initiation and loading lives in a single `EventBlock`.
    If there's a control flow, each branch will be represented by a separate `EventBlock`.

    Args:
        name: Name of the profiling/debugging block.
        events: List of `Event`\ s associated with the profiling/debugging block.
    """

    name: str
    events: List[Event] = dataclasses.field(default_factory=list)
    source_time_scale: TimeScale = TimeScale.NS
    target_time_scale: TimeScale = TimeScale.MS

    def to_dataframe(self, include_units: bool = False) -> pd.DataFrame:
        """
        Converts the EventBlock into a DataFrame with each row being an event instance

        Note: Rows that have an event_name = OPERATOR_CALL correspond to the perf of the
            previous operator + framework tax of making said operator call.

        Args:
            include_units: Whether headers should include units (default false)

        Returns:
            A Pandas DataFrame containing the data of each Event instance in this EventBlock.
        """

        units = " (" + self.target_time_scale.value + ")" if include_units else ""

        # TODO: push row generation down to Event
        data = {
            "event_block_name": [self.name] * len(self.events),
            "event_name": [event.name for event in self.events],
            "raw": [event.perf_data.raw for event in self.events],
            "p10" + units: [event.perf_data.p10 for event in self.events],
            "p50" + units: [event.perf_data.p50 for event in self.events],
            "p90" + units: [event.perf_data.p90 for event in self.events],
            "avg" + units: [event.perf_data.avg for event in self.events],
            "min" + units: [event.perf_data.min for event in self.events],
            "max" + units: [event.perf_data.max for event in self.events],
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
    def _gen_from_etdump(
        etdump: ETDumpFlatCC,
        source_time_scale: TimeScale = TimeScale.NS,
        target_time_scale: TimeScale = TimeScale.MS,
    ) -> List["EventBlock"]:
        """
        Given an etdump, generate a list of EventBlocks corresponding to the
        contents.

        An optional (inverse) scale factor can be provided to adjust the
        etdump timestamps associated with each EventBlocks
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

        scale_factor = (
            TIME_SCALE_DICT[source_time_scale] / TIME_SCALE_DICT[target_time_scale]
        )
        # Create EventBlocks from the Profile Run Groups
        return [
            EventBlock(
                name=str(index),
                events=[
                    Event._gen_from_profile_events(signature, event, scale_factor)
                    for signature, event in profile_events.items()
                ],
                source_time_scale=source_time_scale,
                target_time_scale=target_time_scale,
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
            if event._instruction_id is None:
                continue

            # Check for the instruction_id in handle map
            if (instruction_id := str(event._instruction_id)) not in handle_map:
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
                    f" No delegate mapping found for delegate with instruction id {event._instruction_id}"
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
        event_blocks: List["EventBlocks"]. Structured data from ETDump (correlated with ETRecord if provided).

    Private Attributes:
        _etrecord: Optional[ETRecord]. File under etrecord_path deserialized into an object.
    """

    def __init__(
        self,
        etdump_path: Optional[str] = None,
        etrecord: Optional[Union[ETRecord, str]] = None,
        source_time_scale: TimeScale = TimeScale.NS,
        target_time_scale: TimeScale = TimeScale.MS,
    ) -> None:
        r"""
        Initialize an `Inspector` instance with the underlying `EventBlock`\ s populated with data from the provided ETDump path
        and optional ETRecord path.

        Args:
            etdump_path: Path to the ETDump file.
            etrecord: Optional ETRecord object or path to the ETRecord file.
            source_time_scale: The time scale of the performance data retrieved from the runtime. The default time hook implentation in the runtime returns NS.
            target_time_scale: The target time scale to which the users want their performance data converted to. Defaults to MS.

        Returns:
            None
        """

        if (source_time_scale == TimeScale.CYCLES) ^ (
            target_time_scale == TimeScale.CYCLES
        ):
            raise RuntimeError(
                "For TimeScale in cycles both the source and target time scale have to be in cycles."
            )
        self._source_time_scale = source_time_scale
        self._target_time_scale = target_time_scale

        if etrecord is None:
            self._etrecord = None
        elif isinstance(etrecord, ETRecord):
            self._etrecord = etrecord
        elif isinstance(etrecord, str):
            self._etrecord = parse_etrecord(etrecord_path=etrecord)
        else:
            raise TypeError("Unsupported ETRecord type")

        etdump = gen_etdump_object(etdump_path=etdump_path)
        self.event_blocks = EventBlock._gen_from_etdump(
            etdump, self._source_time_scale, self._target_time_scale
        )

        # Connect ETRecord to EventBlocks
        self.op_graph_dict: Optional[Mapping[str, OperatorGraph]] = None
        self._consume_etrecord()

    def _consume_etrecord(self) -> None:
        """
        If an ETRecord is provided, connect it to the EventBlocks and populate the Event metadata.

        Steps:
            1. Debug Handle Symbolification:
                For each Event, find the debug_handle counterparts using
                ETRecord's debug_handle_map and delegate_map

            2. Event Metadata Association:
                For each Event, populate its metadata from OperatorGraph Nodes,
                generated from ETRecord. The debug_handle is used to
                identify the corresponding OperatorGraph Nodes.
        """

        if self._etrecord is None:
            return

        # (1) Debug Handle Symbolification
        for event_block in self.event_blocks:
            event_block._gen_resolve_debug_handles(
                self._etrecord._debug_handle_map[FORWARD],
                self._etrecord._delegate_map[FORWARD]
                if self._etrecord._delegate_map is not None
                else None,
            )

        # (2) Event Metadata Association
        self.op_graph_dict = gen_graphs_from_etrecord(etrecord=self._etrecord)
        debug_handle_to_op_node_map = create_debug_handle_to_op_node_mapping(
            self.op_graph_dict[EDGE_DIALECT_GRAPH_KEY],
        )
        for event_block in self.event_blocks:
            for event in event_block.events:
                event._associate_with_op_graph_nodes(
                    debug_handle_to_op_node_map=debug_handle_to_op_node_map,
                )

    def print_data_tabular(self, include_units: bool = True) -> None:
        """
        Displays the underlying EventBlocks in a structured tabular format, with each row representing an Event.

        Args:
            include_units: Whether headers should include units (default true)

        Returns:
            None
        """

        def style_text_size(val, size=12):
            return f"font-size: {size}px"

        df_list = [
            event_block.to_dataframe(include_units=include_units)
            for event_block in self.event_blocks
        ]
        combined_df = pd.concat(df_list, ignore_index=True)

        # Filter out some columns and rows for better readability when printing
        filtered_column_df = combined_df.drop(columns=EXCLUDED_COLUMNS_WHEN_PRINTING)
        filtered_df = filtered_column_df[
            ~filtered_column_df["event_name"].isin(EXCLUDED_EVENTS_WHEN_PRINTING)
        ]
        try:
            from IPython import get_ipython
            from IPython.display import display

            if get_ipython() is not None:
                styled_df = filtered_df.style.applymap(style_text_size)
                display(styled_df)
            else:
                raise Exception(
                    "Environment unable to support IPython. Fall back to print()."
                )
        except:
            print(tabulate(filtered_df, headers="keys", tablefmt="fancy_grid"))

    # TODO: write unit test
    def find_total_for_module(self, module_name: str) -> float:
        """
        Returns the total average compute time of all operators within the specified module.

        Args:
            module_name: Name of the module to be aggregated against.

        Returns:
            Sum of the average compute time (in seconds) of all operators within the module with "module_name".
        """

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

    def get_exported_program(
        self, graph: Optional[str] = None
    ) -> Optional[ExportedProgram]:
        """
        Access helper for ETRecord, defaults to returning the Edge Dialect program.

        Args:
            graph: Optional name of the graph to access. If None, returns the Edge Dialect program.

        Returns:
            The ExportedProgram object of "graph".
        """
        if self._etrecord is None:
            log.warning(
                "Exported program is only available when a valid etrecord_path was provided at the time of Inspector construction"
            )
            return None
        return (
            self._etrecord.edge_dialect_program
            if graph is None
            else self._etrecord.graph_map.get(graph)
        )
