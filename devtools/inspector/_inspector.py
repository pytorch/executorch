# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-unsafe

import dataclasses
import logging
import sys
import warnings
from collections import defaultdict, OrderedDict
from dataclasses import dataclass
from functools import cached_property
from typing import (
    Any,
    Callable,
    Dict,
    IO,
    List,
    Mapping,
    Optional,
    Sequence,
    Tuple,
    TypeAlias,
    TypedDict,
    Union,
)

import executorch.devtools.etdump.schema_flatcc as flatcc

import numpy as np
import pandas as pd

from executorch.devtools.debug_format.et_schema import OperatorGraph, OperatorNode
from executorch.devtools.etdump.schema_flatcc import (
    DebugEvent,
    ETDumpFlatCC,
    ProfileEvent,
)
from executorch.devtools.etrecord import ETRecord, parse_etrecord
from executorch.devtools.inspector._inspector_utils import (
    calculate_time_scale_factor,
    create_debug_handle_to_op_node_mapping,
    EDGE_DIALECT_GRAPH_KEY,
    EXCLUDED_COLUMNS_WHEN_PRINTING,
    EXCLUDED_EVENTS_WHEN_PRINTING,
    find_populated_event,
    FORWARD,
    gen_etdump_object,
    gen_graphs_from_etrecord,
    inflate_runtime_output,
    is_debug_output,
    is_inference_output_equal,
    ProgramOutput,
    RESERVED_FRAMEWORK_EVENT_NAMES,
    TimeScale,
    verify_debug_data_equivalence,
)
from executorch.exir import ExportedProgram

from tabulate import tabulate


log: logging.Logger = logging.getLogger(__name__)


# Signature of an InstructionEvent
@dataclass(frozen=True, order=True)
class InstructionEventSignature:
    instruction_id: int
    chain_index: int
    delegate_id: Optional[int] = None
    delegate_id_str: Optional[str] = None


# Aggregated Runtime Events for a single instruction
@dataclass
class InstructionEvent:
    signature: InstructionEventSignature
    profile_events: Optional[List[ProfileEvent]] = None
    debug_events: Optional[List[DebugEvent]] = None

    @staticmethod
    def gen_from_events(run_events: List[flatcc.Event]) -> List["InstructionEvent"]:
        """
        Given a list of events from a run in ETDump, collate the ProfileEvent
        and DebugEvents by instruction id and return a list of InstructionEvents
        constructed from collated events (ignoring run_output events)
        """
        instruction_events: Dict[InstructionEventSignature, InstructionEvent] = (
            OrderedDict()
        )
        for event in run_events:
            # Find the event that was logged
            populated_event: Union[DebugEvent, ProfileEvent] = find_populated_event(
                event
            )

            # Get existing InstructionEvent or insert a new one
            signature = InstructionEventSignature(
                instruction_id=populated_event.instruction_id,
                chain_index=populated_event.chain_index,
                delegate_id=populated_event.delegate_debug_id_int,
                delegate_id_str=populated_event.delegate_debug_id_str,
            )

            instruction_event = instruction_events.setdefault(
                signature, InstructionEvent(signature=signature)
            )

            # Update InstructionEvent based on event type
            if isinstance(populated_event, ProfileEvent):
                if instruction_event.profile_events is None:
                    instruction_event.profile_events = []
                instruction_event.profile_events.append(populated_event)
            elif isinstance(populated_event, DebugEvent):
                # Ignore run_output events
                if not is_debug_output(populated_event.debug_entry):
                    if instruction_event.debug_events is None:
                        instruction_event.debug_events = []
                    instruction_event.debug_events.append(populated_event)

        return list(instruction_events.values())


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


# Signature of a DebugEvent
@dataclass(frozen=True, order=True)
class DebugEventSignature:
    name: str = ""
    instruction_id: Optional[int] = -1
    delegate_id: Optional[int] = None
    delegate_id_str: Optional[str] = None

    @staticmethod
    def _gen_from_event(event: DebugEvent) -> "DebugEventSignature":
        """
        Given a DebugEvent, extract the fields into a signature

        DebugEvents from ETDump default to "" and -1 when the field is not populated
        The Signature will convert these back to the intended None value
        """
        return DebugEventSignature(
            event.name or "",
            event.instruction_id if event.instruction_id != -1 else None,
            event.delegate_debug_id_int if event.delegate_debug_id_int != -1 else None,
            event.delegate_debug_id_str if event.delegate_debug_id_str != "" else None,
        )


# Signature of an Event inside of a Run
@dataclass(frozen=True, order=True)
class EventSignature:
    """
    Note that (profile_event_signature, debug_event_signature) are sufficient
    signature identifiers.

    instruction_id is extracted from the signatures (equivalent in both) and
    surfaced for convenience
    """

    instruction_id: int
    profile_event_signature: Optional[ProfileEventSignature] = None
    debug_event_signature: Optional[DebugEventSignature] = None

    @staticmethod
    def gen_from_instruction_event(
        instruction_event: InstructionEvent,
    ) -> List[Tuple["EventSignature", InstructionEvent]]:
        """
        Construct EventSignatures from the given InstructionEvent
        and return tuples of (1) EventSignature and (2) related subset
        InstructionEvent
        """

        # Generate the DebugEventSignature
        debug_events = instruction_event.debug_events
        debug_signature = (
            DebugEventSignature._gen_from_event(debug_events[0])
            if debug_events is not None and len(debug_events) > 0
            else None
        )

        # If no ProfileEvents, return a singleton EventSignature
        if (profile_events := instruction_event.profile_events) is None:
            return [
                (
                    EventSignature(
                        instruction_id=instruction_event.signature.instruction_id,
                        debug_event_signature=debug_signature,
                    ),
                    instruction_event,
                )
            ]

        # Generate the ProfileEventSignature
        return [
            (
                EventSignature(
                    instruction_id=instruction_event.signature.instruction_id,
                    profile_event_signature=ProfileEventSignature._gen_from_event(
                        profile_event
                    ),
                    debug_event_signature=debug_signature,
                ),
                dataclasses.replace(instruction_event, profile_events=[profile_event]),
            )
            for profile_event in profile_events
        ]


# Signature of a Run
@dataclass(frozen=True, order=True)
class RunSignature:
    """
    Args:
        name: Name of the run
        events: List of EventSignatures that correspond to the run
        bundled_input_index: Index of the bundled input used to generate the debug output
    """

    name: str
    events: Optional[Tuple[EventSignature]] = None
    bundled_input_index: Optional[int] = None


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


@dataclass
class Event:
    """
    An Event corresponds to an operator instance with perf data retrieved from the runtime and other metadata from `ETRecord`.

    Args:
        name: Name of the profiling `Event`, empty if no profiling event.
        perf_data: Performance data associated with the event retrived from the runtime (available attributes: p10, p50, p90, avg, min and max).
        op_type: List of op types corresponding to the event.
        delegate_debug_identifier: Supplemental identifier used in combination with instruction id.
        debug_handles: Debug handles in the model graph to which this event is correlated.
        stack_trace: A dictionary mapping the name of each associated op to its stack trace.
        module_hierarchy: A dictionary mapping the name of each associated op to its module hierarchy.
        is_delegated_op: Whether or not the event was delegated.
        delegate_backend_name: Name of the backend this event was delegated to.

        _delegate_debug_metadatas: A list of raw delegate debug metadata in string, one for each profile event.
            Available parsed (if parser provided) as Event.delegate_debug_metadatas
            Available as Event.raw_delegate_debug_metadatas

        debug_data: A list containing intermediate data collected.

        _instruction_id: Instruction Identifier for Symbolication
        _delegate_metadata_parser: Optional Parser for _delegate_debug_metadatas
    """

    name: str
    perf_data: Optional[PerfData] = None
    op_types: List[str] = dataclasses.field(default_factory=list)
    delegate_debug_identifier: Optional[Union[int, str]] = None
    debug_handles: Optional[Union[int, Sequence[int]]] = None
    stack_traces: Dict[str, str] = dataclasses.field(default_factory=dict)
    module_hierarchy: Dict[str, Dict] = dataclasses.field(default_factory=dict)
    is_delegated_op: Optional[bool] = None
    delegate_backend_name: Optional[str] = None
    _delegate_debug_metadatas: List[str] = dataclasses.field(default_factory=list)

    debug_data: ProgramOutput = dataclasses.field(default_factory=list)
    _instruction_id: Optional[int] = None

    _delegate_metadata_parser: Optional[Callable[[List[str]], Dict[str, Any]]] = None
    _delegate_time_scale_converter: Optional[
        Callable[[Union[int, str], Union[int, float]], Union[int, float]]
    ] = None

    @cached_property
    def delegate_debug_metadatas(self) -> Union[List[str], Dict[str, Any]]:
        """
        Returns the parsed _delegate_debug_metadatas if a parser is available
        Otherwise returns the raw _delegate_debug_metadatas
        """
        if not self.is_delegated_op or self._delegate_metadata_parser is None:
            return self._delegate_debug_metadatas
        return self._delegate_metadata_parser(self._delegate_debug_metadatas)

    @property
    def raw_delegate_debug_metadatas(self) -> List[str]:
        """
        Return the raw unparsed _delegate_debug_metadatas
        """
        return self._delegate_debug_metadatas

    def to_dataframe(self, _units="") -> pd.DataFrame:
        """
        Convert the Event into a pandas DataFrame

        Args:
            None

        Returns:
            A pandas DataFrame with the Event data
        """
        event_dict = self.asdict(_units=_units)
        return pd.DataFrame(event_dict)

    # Override the default implementation of dataclass.asdict to handle null perf data
    def asdict(self, _units="") -> dict:
        """
        Convert the Event into a dict

        Args:
            None

        Returns:
            A dict with the Event data
        """

        def truncated_list(long_list: List[str]) -> str:
            return f"['{long_list[0]}', '{long_list[1]}' ... '{long_list[-1]}'] ({len(long_list)} total)"

        return {
            "event_name": self.name,
            "raw": [self.perf_data.raw if self.perf_data else None],
            "p10" + _units: self.perf_data.p10 if self.perf_data else None,
            "p50" + _units: self.perf_data.p50 if self.perf_data else None,
            "p90" + _units: self.perf_data.p90 if self.perf_data else None,
            "avg" + _units: self.perf_data.avg if self.perf_data else None,
            "min" + _units: self.perf_data.min if self.perf_data else None,
            "max" + _units: self.perf_data.max if self.perf_data else None,
            "op_types": [
                (
                    self.op_types
                    if len(self.op_types) < 5
                    else truncated_list(self.op_types)
                )
            ],
            "delegate_debug_identifier": self.delegate_debug_identifier,
            "stack_traces": [self.stack_traces],
            "module_hierarchy": [self.module_hierarchy],
            "is_delegated_op": self.is_delegated_op,
            "delegate_backend_name": self.delegate_backend_name,
            "debug_data": [self.debug_data],
        }

    @staticmethod
    def _gen_from_inference_events(
        signature: EventSignature,
        events: List[InstructionEvent],
        scale_factor: float = 1.0,
        output_buffer: Optional[bytes] = None,
        delegate_metadata_parser: Optional[
            Callable[[List[str]], Dict[str, Any]]
        ] = None,
        delegate_time_scale_converter: Optional[
            Callable[[Union[int, str], Union[int, float]], Union[int, float]]
        ] = None,
    ) -> "Event":
        """
        Given an EventSignature and a list of Events with that signature,
        return an Event object matching the EventSignature, with perf_data
        populated from the list of ProfileEvents and debug_data populated from
        the list of DebugEvents.

        An optional inverse scale factor can be provided to adjust the event timestamps
        An optional buffer can be provided to inflate etdump references
        An optional delegate_metadata_parser can be provided to parse the delegate metadata
        """

        profile_event_signature = signature.profile_event_signature
        debug_event_signature = signature.debug_event_signature

        # Event is gradually populated in this function
        ret_event: Event = Event(
            name="",
            _instruction_id=signature.instruction_id,
            _delegate_metadata_parser=delegate_metadata_parser,
            _delegate_time_scale_converter=delegate_time_scale_converter,
        )

        # Populate fields from profile events
        Event._populate_profiling_related_fields(
            ret_event, profile_event_signature, events, scale_factor
        )

        # Populate fields from debug events
        Event._populate_debugging_related_fields(
            ret_event, debug_event_signature, events, output_buffer
        )

        return ret_event

    @staticmethod
    def _calculate_elapsed_time(start_time, end_time):
        # We're assuming if there's a wraparound in the time values, then
        # the time representation of that platform only contains 32 bits.
        # This should be fine for now, but ideally we should source the max
        # time value from the platform using etdump.
        max_uint32 = 2**32 - 1
        if start_time > end_time:
            if (start_time > max_uint32) or (end_time > max_uint32):
                raise ValueError(
                    f"Expected start_time ({start_time}) and end_time ({end_time}) to be less than {max_uint32} for cases where there is wrap-around of time values."
                )
            # Handle wraparound
            elapsed_time = (max_uint32 - start_time) + end_time
        else:
            # Normal case
            elapsed_time = end_time - start_time
        return elapsed_time

    @staticmethod
    def _populate_event_signature_fields(
        ret_event: "Event",
        event_signature: Optional[Union[ProfileEventSignature, DebugEventSignature]],
    ) -> None:
        """
        Given a partially constructed Event, populate the fields related to
        the profile event signature or debug event signature

        Fields Updated:
            name
            delegate_debug_identifier
            is_delegated_op
        """
        # TODO: T201347372 Push the None check to ealier in the stack.
        if event_signature is not None:
            if event_signature.delegate_id is not None:  # 0 is a valid value
                delegate_debug_identifier = event_signature.delegate_id
            else:
                delegate_debug_identifier = event_signature.delegate_id_str or None

            # Use the delegate identifier as the event name if delegated
            is_delegated_op = delegate_debug_identifier is not None
            name = (
                event_signature.name
                if not is_delegated_op
                else str(delegate_debug_identifier)
            )

            # Update fields
            # This is for older version of etdump that doesn't have the name field for debug events, we don't update the name field
            if name:
                ret_event.name = name
            ret_event.delegate_debug_identifier = delegate_debug_identifier
            ret_event.is_delegated_op = is_delegated_op

    @staticmethod
    def _populate_profiling_related_fields(
        ret_event: "Event",
        profile_event_signature: Optional[ProfileEventSignature],
        events: List[InstructionEvent],
        scale_factor: float,
    ) -> None:
        """
        Given a partially constructed Event, populate the fields related to
        the profile events

        Fields Updated:
            name
            delegate_debug_identifier
            is_delegated_op
            perf_data
            delegate_debug_metadatas
        """

        # Fill out fields from profile event signature
        Event._populate_event_signature_fields(ret_event, profile_event_signature)

        # Fill out fields from profile event
        data = []
        delegate_debug_metadatas = []
        for event in events:
            if (profile_events := event.profile_events) is not None:
                if len(profile_events) != 1:
                    raise ValueError(
                        f"Expected exactly one profile event per InstructionEvent when generating Inspector Event, but got {len(profile_events)}"
                    )

                profile_event = profile_events[0]

                # Scale factor should only be applied to non-delegated ops
                if (
                    ret_event.is_delegated_op
                    and (convert_time_scale := ret_event._delegate_time_scale_converter)
                    is not None
                ):
                    scaled_time = Event._calculate_elapsed_time(
                        convert_time_scale(ret_event.name, profile_event.start_time),
                        convert_time_scale(ret_event.name, profile_event.end_time),
                    )
                # If it's not a delegated op then we can just use the raw time values
                # and then scale them according to the scale factor that was passed in.
                elif not ret_event.is_delegated_op:
                    scaled_time = (
                        float(
                            Event._calculate_elapsed_time(
                                profile_event.start_time, profile_event.end_time
                            )
                        )
                        / scale_factor
                    )
                # If there was no scale factor passed in just take a difference of the
                # end and start times.
                else:
                    scaled_time = float(
                        Event._calculate_elapsed_time(
                            profile_event.start_time, profile_event.end_time
                        )
                    )

                data.append(scaled_time)
                delegate_debug_metadatas.append(
                    profile_event.delegate_debug_metadata
                    if profile_event.delegate_debug_metadata
                    else ""
                )

        # Update fields
        if len(data) > 0:
            ret_event.perf_data = PerfData(data)
        if any(delegate_debug_metadatas):
            ret_event._delegate_debug_metadatas = delegate_debug_metadatas

    @staticmethod
    def _populate_debugging_related_fields(
        ret_event: "Event",
        debug_event_signature: Optional[DebugEventSignature],
        events: List[InstructionEvent],
        output_buffer: Optional[bytes] = None,
    ) -> None:
        """
        Given a partially constructed Event, populate the fields related to
        the debug events

        Fields Updated:
            name
            delegate_debug_identifier
            is_delegated_op
            debug_data
        """

        # Fill out fields from debug event signature
        Event._populate_event_signature_fields(ret_event, debug_event_signature)

        debug_data: List[flatcc.Value] = []
        for event in events:
            if (debug_events := event.debug_events) is None:
                continue

            # Populate on the first iteration only, then verify equivalence for others
            if len(debug_data) == 0:
                debug_data = [debug_event.debug_entry for debug_event in debug_events]
            else:
                for debug_event, value in zip(debug_events, debug_data):
                    v1 = inflate_runtime_output(debug_event.debug_entry, output_buffer)
                    v2 = inflate_runtime_output(value, output_buffer)
                    assert is_inference_output_equal(
                        v1, v2
                    ), """Corresponding debug events in multiple iterations of the model
                    must have the same debug entry values. This is not the case for the
                    intermediate data present in this ETDump and indicates potential issues
                    with the model/runtime."""

        ret_event.debug_data = [
            inflate_runtime_output(debug_value, output_buffer)
            for debug_value in debug_data
        ]

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

        bundled_input_idx: Index of the Bundled Input that this EventBlock corresponds to.
        run_output: Run output extracted from the encapsulated Events
    """

    name: str
    events: List[Event] = dataclasses.field(default_factory=list)
    source_time_scale: TimeScale = TimeScale.NS
    target_time_scale: TimeScale = TimeScale.MS
    bundled_input_index: Optional[int] = None
    run_output: Optional[ProgramOutput] = None
    reference_output: Optional[ProgramOutput] = None

    def to_dataframe(
        self, include_units: bool = False, include_delegate_debug_data: bool = False
    ) -> pd.DataFrame:
        """
        Converts the EventBlock into a DataFrame with each row being an event instance

        Note: Rows that have an event_name = OPERATOR_CALL correspond to the perf of the
            previous operator + framework tax of making said operator call.

        Args:
            include_units: Whether headers should include units (default false)
            include_delegate_debug_data: Whether to show the delegate debug data

        Returns:
            A pandas DataFrame containing the data of each Event instance in this EventBlock.
        """

        units = " (" + self.target_time_scale.value + ")" if include_units else ""

        df = pd.concat([e.to_dataframe(units) for e in self.events], ignore_index=True)
        df.insert(
            0,
            "event_block_name",
            np.asarray([self.name for _ in range(len(self.events))]),
            allow_duplicates=True,
        )

        # Add Delegate Debug Metadata columns
        if include_delegate_debug_data:
            delegate_data = []
            for event in self.events:
                if (metadata := event.delegate_debug_metadatas) is not None and len(
                    metadata
                ) > 0:
                    if isinstance(metadata, list):
                        delegate_data.append(
                            pd.Series([metadata], index=["delegate_debug_metadata"])
                        )
                    elif isinstance(metadata, dict):
                        delegate_data.append(pd.Series(metadata))
                    else:
                        raise ValueError(
                            f"Unexpected type for delegate_debug_metadata: {type(metadata)}"
                        )
                else:
                    delegate_data.append(pd.Series())

            if any(not data.empty for data in delegate_data):
                df = pd.concat([df, pd.DataFrame(delegate_data)], axis=1)

        return df

    @staticmethod
    def _gen_from_etdump(
        etdump: ETDumpFlatCC,
        source_time_scale: TimeScale = TimeScale.NS,
        target_time_scale: TimeScale = TimeScale.MS,
        output_buffer: Optional[bytes] = None,
        delegate_metadata_parser: Optional[
            Callable[[List[str]], Dict[str, Any]]
        ] = None,
        delegate_time_scale_converter: Optional[
            Callable[[Union[int, str], Union[int, float]], Union[int, float]]
        ] = None,
    ) -> List["EventBlock"]:
        """
        Given an etdump, generate a list of EventBlocks corresponding to the
        contents.

        An optional (inverse) scale factor can be provided to adjust the
        etdump timestamps associated with each EventBlocks

        An optional buffer to inflate etdump references

        An optional delegate metadata parser function to parse delegate profiling metadata
        """

        # Map each RunSignatures to instances of its constituent events.
        #   The value of the map is a GroupedRunInstance which contains:
        #   (1) a map from each EventSignature to InstructionEvents with the signature
        #   (2) the run output for this RunSignature
        @dataclass
        class GroupedRunInstances:
            events: OrderedDict[EventSignature, List[InstructionEvent]]
            run_output: ProgramOutput

        run_groups: Mapping[RunSignature, GroupedRunInstances] = defaultdict(
            lambda: GroupedRunInstances(OrderedDict(), [])
        )

        # Collect all the run data
        for run in etdump.run_data:
            if (run_events := run.events) is None:
                continue

            # Collate the run_events into InstructionEvents
            instruction_events: List[InstructionEvent] = (
                InstructionEvent.gen_from_events(run_events)
            )

            # Map EventSignatures to the InstructionEvents
            event_signatures: Dict[EventSignature, InstructionEvent] = OrderedDict()
            for instruction_event in instruction_events:
                if (
                    instruction_event.debug_events is None
                    and instruction_event.profile_events is None
                ):
                    # Currently corresponds to run output
                    continue

                generated_event_signatures: List[
                    Tuple[EventSignature, InstructionEvent]
                ] = EventSignature.gen_from_instruction_event(instruction_event)
                for (
                    event_signature,
                    filtered_instruction_event,
                ) in generated_event_signatures:
                    event_signatures[event_signature] = filtered_instruction_event

            # Create a RunSignature from the EventSignatures
            run_signature = RunSignature(
                name=run.name,
                events=tuple(event_signatures.keys()),
                bundled_input_index=run.bundled_input_index,
            )

            # Update the Run Groups, indexed on the RunSignature
            run_signature_events: OrderedDict[
                EventSignature, List[InstructionEvent]
            ] = run_groups[run_signature].events
            for event_signature, event in event_signatures.items():
                run_signature_events.setdefault(event_signature, []).append(event)

            # Populate (or Verify if already populated) Run Outputs
            run_outputs: ProgramOutput = EventBlock._collect_run_outputs(
                run_events, output_buffer
            )
            if len(existing_run_outputs := run_groups[run_signature].run_output) == 0:
                existing_run_outputs.extend(run_outputs)
            else:
                verify_debug_data_equivalence(existing_run_outputs, run_outputs)

        # Construct the EventBlocks
        event_blocks = []
        scale_factor = calculate_time_scale_factor(source_time_scale, target_time_scale)
        for run_signature, grouped_run_instance in run_groups.items():
            run_group: OrderedDict[EventSignature, List[InstructionEvent]] = (
                grouped_run_instance.events
            )
            run_outputs: ProgramOutput = grouped_run_instance.run_output

            # Construct the Events
            events: List[Event] = [
                Event._gen_from_inference_events(
                    signature,
                    instruction_events,
                    scale_factor,
                    output_buffer,
                    delegate_metadata_parser,
                    delegate_time_scale_converter,
                )
                for signature, instruction_events in run_group.items()
            ]

            # Add the EventBlock to the return list
            event_blocks.append(
                EventBlock(
                    name=run_signature.name,
                    events=events,
                    source_time_scale=source_time_scale,
                    target_time_scale=target_time_scale,
                    bundled_input_index=run_signature.bundled_input_index,
                    run_output=run_outputs,
                )
            )

        return event_blocks

    @staticmethod
    def _collect_run_outputs(
        events: List[flatcc.Event], output_buffer: Optional[bytes] = None
    ) -> ProgramOutput:
        """
        Given a list of events, search the events for ProgramOutputs (aka lists of InferenceOutputs) marked
        as run outputs
        """

        output_events = []
        for event in events:
            if event.debug_event is None:
                continue
            if event.debug_event.debug_entry is None:
                raise RuntimeError(
                    "Debug entry inside debug event should not be empty!"
                )
            if is_debug_output(event.debug_event.debug_entry):
                output_events += [event]

        return [
            inflate_runtime_output(debug_event.debug_entry, output_buffer)
            for output_event in output_events
            if (debug_event := output_event.debug_event) is not None
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

                # DELEGATE_CALL is a special non-delegated event and benefits from having the name populated
                if (
                    event.name == "DELEGATE_CALL"
                    and delegate_map is not None
                    and (delegate_metadata := delegate_map.get(instruction_id))
                    is not None
                ):
                    event.delegate_backend_name = delegate_metadata.get("name", "")

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
            delegate_metadata_delegate_map = delegate_metadata.get("delegate_map") or {}

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
                for key, value in delegate_metadata_delegate_map.items():
                    if key in str(delegate_debug_id):
                        event.debug_handles = value


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
        debug_buffer_path: Optional[str] = None,
        delegate_metadata_parser: Optional[
            Callable[[List[str]], Dict[str, Any]]
        ] = None,
        delegate_time_scale_converter: Optional[
            Callable[[Union[int, str], Union[int, float]], Union[int, float]]
        ] = None,
        enable_module_hierarchy: bool = False,
    ) -> None:
        r"""
        Initialize an `Inspector` instance with the underlying `EventBlock`\ s populated with data from the provided ETDump path
        and optional ETRecord path.

        Args:
            etdump_path: Path to the ETDump file.
            etrecord: Optional ETRecord object or path to the ETRecord file.
            source_time_scale: The time scale of the performance data retrieved from the runtime. The default time hook implentation in the runtime returns NS.
            target_time_scale: The target time scale to which the users want their performance data converted to. Defaults to MS.
            debug_buffer_path: Debug buffer file path that contains the debug data referenced by ETDump for intermediate and program outputs.
            delegate_metadata_parser: Optional function to parse delegate metadata from an Profiling Event. Expected signature of the function is:
                    (delegate_metadata_list: List[bytes]) -> Union[List[str], Dict[str, Any]]
            delegate_time_scale_converter: Optional function to convert the time scale of delegate profiling data. If not given, use the conversion ratio of
                    target_time_scale/source_time_scale.
            enable_module_hierarchy: Enable submodules in the operator graph. Defaults to False.

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

        if delegate_time_scale_converter is None:
            scale_factor = calculate_time_scale_factor(
                source_time_scale, target_time_scale
            )
            delegate_time_scale_converter = (
                lambda event_name, input_time: input_time / scale_factor
            )

        if etrecord is None:
            self._etrecord = None
        elif isinstance(etrecord, ETRecord):
            self._etrecord = etrecord
        elif isinstance(etrecord, str):
            self._etrecord = parse_etrecord(etrecord_path=etrecord)
        else:
            raise TypeError("Unsupported ETRecord type")

        # Create EventBlocks from ETDump
        etdump = gen_etdump_object(etdump_path=etdump_path)
        if debug_buffer_path is not None:
            with open(debug_buffer_path, "rb") as f:
                output_buffer = f.read()
        else:
            output_buffer = None
            warnings.warn(
                "Output Buffer not found. Tensor Debug Data will not be available.",
                stacklevel=1,
            )

        self.event_blocks = EventBlock._gen_from_etdump(
            etdump=etdump,
            source_time_scale=self._source_time_scale,
            target_time_scale=self._target_time_scale,
            output_buffer=output_buffer,
            delegate_metadata_parser=delegate_metadata_parser,
            delegate_time_scale_converter=delegate_time_scale_converter,
        )

        # Connect ETRecord to EventBlocks
        self.op_graph_dict: Optional[Mapping[str, OperatorGraph]] = None

        # _consume_etrecord() will populate the _reference_outputs dict
        # Key str is method name; value is list of ProgramOutputs because of list of test cases
        self._reference_outputs: Dict[str, List[ProgramOutput]] = {}
        self._enable_module_hierarchy = enable_module_hierarchy
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

            3. Reference Outputs Extraction:
                If there're reference outputs saved in ETRecord, assign each reference output to the corresponding
                EventBlock based on the method name (currently assumes only "forward") and the
                bundled_input_index of the EventBlock.
        """

        if self._etrecord is None:
            return

        # (1) Debug Handle Symbolification
        for event_block in self.event_blocks:
            event_block._gen_resolve_debug_handles(
                self._etrecord._debug_handle_map[FORWARD],
                (
                    self._etrecord._delegate_map[FORWARD]
                    if self._etrecord._delegate_map is not None
                    else None
                ),
            )

        # (2) Event Metadata Association
        self.op_graph_dict = gen_graphs_from_etrecord(
            etrecord=self._etrecord,
            enable_module_hierarchy=self._enable_module_hierarchy,
        )
        debug_handle_to_op_node_map = create_debug_handle_to_op_node_mapping(
            self.op_graph_dict[EDGE_DIALECT_GRAPH_KEY],
        )
        for event_block in self.event_blocks:
            for event in event_block.events:
                event._associate_with_op_graph_nodes(
                    debug_handle_to_op_node_map=debug_handle_to_op_node_map,
                )

        # (3) Reference Outputs Extraction
        if self._etrecord._reference_outputs is not None:
            self._reference_outputs = self._etrecord._reference_outputs
            # Associate each reference output to the corresponding event block
            for event_block in self.event_blocks:
                index = event_block.bundled_input_index
                if index is not None:
                    event_block.reference_output = self._reference_outputs[FORWARD][
                        index
                    ]

    def to_dataframe(
        self,
        include_units: bool = True,
        include_delegate_debug_data: bool = False,
    ) -> pd.DataFrame:
        """
        Args:
            include_units: Whether headers should include units (default true)
            include_delegate_debug_data: Whether to include delegate debug metadata (default false)

        Returns:
            Returns a pandas DataFrame of the Events in each EventBlock in the inspector, with each row representing an Event.
        """

        df_list = [
            event_block.to_dataframe(
                include_units=include_units,
                include_delegate_debug_data=include_delegate_debug_data,
            )
            for event_block in self.event_blocks
        ]
        return pd.concat(df_list, ignore_index=True)

    def print_data_tabular(
        self,
        file: IO[str] = sys.stdout,
        include_units: bool = True,
        include_delegate_debug_data: bool = False,
    ) -> None:
        """
        Displays the underlying EventBlocks in a structured tabular format, with each row representing an Event.

        Args:
            file: Which IO stream to print to. Defaults to stdout.
                Not used if this is in an IPython environment such as a Jupyter notebook.
            include_units: Whether headers should include units (default true)
            include_delegate_debug_data: Whether to include delegate debug metadata (default false)

        Returns:
            None
        """
        combined_df = self.to_dataframe(include_units, include_delegate_debug_data)

        # Filter out some columns and rows for better readability when printing
        filtered_column_df = combined_df.drop(columns=EXCLUDED_COLUMNS_WHEN_PRINTING)
        for filter_name in EXCLUDED_EVENTS_WHEN_PRINTING:
            filtered_column_df = filtered_column_df[
                ~filtered_column_df["event_name"].str.contains(filter_name)
            ]
        filtered_column_df.reset_index(drop=True, inplace=True)

        try:
            from IPython import get_ipython
            from IPython.display import display

            def style_text_size(val, size=12):
                return f"font-size: {size}px"

            if get_ipython() is not None:
                styled_df = filtered_column_df.style.applymap(style_text_size)
                display(styled_df)
            else:
                raise Exception(
                    "Environment unable to support IPython. Fall back to print()."
                )
        except:
            print(
                tabulate(filtered_column_df, headers="keys", tablefmt="fancy_grid"),
                file=file,
            )

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
                        if event.perf_data is not None:
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
