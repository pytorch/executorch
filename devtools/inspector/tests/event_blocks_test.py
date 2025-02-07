# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict
import unittest
from typing import List, Optional, Tuple, Union

import executorch.devtools.etdump.schema_flatcc as flatcc
from executorch.devtools.etdump.schema_flatcc import ETDumpFlatCC, ProfileEvent
from executorch.devtools.inspector import Event, EventBlock, PerfData
from executorch.devtools.inspector._inspector import (
    DelegateMetadata,
    EventSignature,
    InstructionEvent,
    InstructionEventSignature,
    ProfileEventSignature,
)


class TestEventBlock(unittest.TestCase):

    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~ Test Helpers ~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    @staticmethod
    def _gen_sample_profile_event(
        name: str,
        instruction_id: int,
        time: Tuple[int, int],
        delegate_debug_id: Optional[Union[int, str]] = None,
        delegate_debug_metadata: Optional[str] = None,
    ) -> flatcc.ProfileEvent:
        """
        Helper for generating test ProfileEvents

        Notably:
        - the timestamp is specified as a tuple of two separate integers
        - delegate_debug_id takes either the str or int representation
        - chain_idx is auto-populated to 0
        """
        delegate_debug_id_int = (
            delegate_debug_id if isinstance(delegate_debug_id, int) else -1
        )
        delegate_debug_id_str = (
            delegate_debug_id if isinstance(delegate_debug_id, str) else ""
        )
        return flatcc.ProfileEvent(
            name,
            0,
            instruction_id,
            delegate_debug_id_int,
            delegate_debug_id_str,
            # pyre-fixme[6]: For 6th argument expected `Optional[bytes]` but got
            #  `Optional[str]`.
            delegate_debug_metadata,
            start_time=time[0],
            end_time=time[1],
        )

    @staticmethod
    def _gen_sample_debug_event(
        instruction_id: int,
        delegate_debug_id: Optional[Union[int, str]] = None,
        name: str = "test_debug_event",
    ) -> flatcc.DebugEvent:
        """
        Helper for generating test DebugEvents

        Notably:
        - delegate_debug_id takes either the str or int representation
        """
        delegate_debug_id_int = (
            delegate_debug_id if isinstance(delegate_debug_id, int) else -1
        )
        delegate_debug_id_str = (
            delegate_debug_id if isinstance(delegate_debug_id, str) else ""
        )

        return flatcc.DebugEvent(
            name=name,
            chain_index=0,
            instruction_id=instruction_id,
            delegate_debug_id_int=delegate_debug_id_int,
            delegate_debug_id_str=delegate_debug_id_str,
            debug_entry=flatcc.Value(
                val=flatcc.ValueType.TENSOR.value,
                tensor=flatcc.Tensor(
                    scalar_type=flatcc.ScalarType.INT,
                    sizes=[1],
                    strides=[1],
                    offset=12345,
                ),
                tensor_list=flatcc.TensorList(
                    [
                        flatcc.Tensor(
                            scalar_type=flatcc.ScalarType.INT,
                            sizes=[1],
                            strides=[1],
                            offset=12345,
                        )
                    ]
                ),
                int_value=flatcc.Int(1),
                float_value=flatcc.Float(1.0),
                double_value=flatcc.Double(1.0),
                bool_value=flatcc.Bool(False),
                output=None,
            ),
        )

    @staticmethod
    def _get_sample_etdump_flatcc() -> flatcc.ETDumpFlatCC:
        """
        Helper for getting a sample ETDumpFlatCC object with 3 RunData:
        - run_data_1 has signature_a with just profile_1
        - run_data_2 has the same signature with run_data_1, but different times
        - run_data_3 has signature_b with both (profile_1, profile_2)
        """
        profile_event_1 = TestEventBlock._gen_sample_profile_event(
            name="profile_1", instruction_id=1, time=(0, 1), delegate_debug_id=100
        )
        run_data_1 = flatcc.RunData(
            name="signature_a",
            bundled_input_index=-1,
            allocators=[],
            events=[
                flatcc.Event(
                    allocation_event=None,
                    debug_event=None,
                    profile_event=profile_event_1,
                )
            ],
        )
        profile_event_2 = TestEventBlock._gen_sample_profile_event(
            name="profile_1", instruction_id=1, time=(2, 4), delegate_debug_id=100
        )
        run_data_2 = flatcc.RunData(
            name="signature_a",
            bundled_input_index=-1,
            allocators=[],
            events=[
                flatcc.Event(
                    allocation_event=None,
                    debug_event=None,
                    profile_event=profile_event_2,
                )
            ],
        )

        profile_event_3 = TestEventBlock._gen_sample_profile_event(
            name="profile_1", instruction_id=1, time=(5, 6), delegate_debug_id=100
        )
        profile_event_4 = TestEventBlock._gen_sample_profile_event(
            name="profile_2", instruction_id=2, time=(7, 8), delegate_debug_id=100
        )
        run_data_3 = flatcc.RunData(
            name="signature_b",
            bundled_input_index=-1,
            allocators=[],
            events=[
                flatcc.Event(
                    allocation_event=None,
                    debug_event=None,
                    profile_event=profile_event_3,
                ),
                flatcc.Event(
                    allocation_event=None,
                    debug_event=None,
                    profile_event=profile_event_4,
                ),
            ],
        )

        return ETDumpFlatCC(version=0, run_data=[run_data_1, run_data_2, run_data_3])

    @staticmethod
    def _get_sample_etdump_flatcc_inconsistent_debug_data() -> flatcc.ETDumpFlatCC:
        debug_event_1 = TestEventBlock._gen_sample_debug_event(
            instruction_id=1, delegate_debug_id=100
        )
        run_data_1 = flatcc.RunData(
            name="signature_a",
            bundled_input_index=-1,
            allocators=[],
            events=[
                flatcc.Event(
                    allocation_event=None,
                    debug_event=debug_event_1,
                    profile_event=None,
                ),
            ],
        )

        debug_event_2 = TestEventBlock._gen_sample_debug_event(
            instruction_id=1, delegate_debug_id=100
        )
        # Modify this debug event so it's different from debug_event_1
        debug_event_2.debug_entry.tensor.sizes = [2]  # pyre-ignore
        run_data_2 = flatcc.RunData(
            name="signature_a",
            bundled_input_index=-1,
            allocators=[],
            events=[
                flatcc.Event(
                    allocation_event=None,
                    debug_event=debug_event_2,
                    profile_event=None,
                ),
            ],
        )
        return ETDumpFlatCC(version=0, run_data=[run_data_1, run_data_2])

    @staticmethod
    def _get_sample_etdump_flatcc_profiling_and_debugging() -> flatcc.ETDumpFlatCC:
        """
        Helper for getting a sample ETDumpFlatCC object with 3 RunData:
        - run_data_1 has signature_a with (debug_event_1, profile_event_1)
        - run_data_2 has the same signature with run_data_1 and same debug event, but different profiling times
        - run_data_3 has signature_b with (debug_event_3, profile_event_3) and (not debug event, profile_event_4)
        """
        profile_event_1 = TestEventBlock._gen_sample_profile_event(
            name="profile_1", instruction_id=1, time=(0, 1), delegate_debug_id=100
        )
        debug_event_1 = TestEventBlock._gen_sample_debug_event(
            instruction_id=1, delegate_debug_id=100
        )
        run_data_1 = flatcc.RunData(
            name="signature_a",
            bundled_input_index=-1,
            allocators=[],
            events=[
                flatcc.Event(
                    allocation_event=None,
                    debug_event=None,
                    profile_event=profile_event_1,
                ),
                flatcc.Event(
                    allocation_event=None,
                    debug_event=debug_event_1,
                    profile_event=None,
                ),
            ],
        )

        profile_event_2 = TestEventBlock._gen_sample_profile_event(
            name="profile_1", instruction_id=1, time=(2, 4), delegate_debug_id=100
        )
        debug_event_2 = TestEventBlock._gen_sample_debug_event(
            instruction_id=1, delegate_debug_id=100
        )
        run_data_2 = flatcc.RunData(
            name="signature_a",
            bundled_input_index=-1,
            allocators=[],
            events=[
                flatcc.Event(
                    allocation_event=None,
                    debug_event=None,
                    profile_event=profile_event_2,
                ),
                flatcc.Event(
                    allocation_event=None,
                    debug_event=debug_event_2,
                    profile_event=None,
                ),
            ],
        )

        profile_event_3 = TestEventBlock._gen_sample_profile_event(
            name="profile_3", instruction_id=1, time=(5, 6), delegate_debug_id=100
        )
        debug_event_3 = TestEventBlock._gen_sample_debug_event(
            instruction_id=1, delegate_debug_id=100
        )
        profile_event_4 = TestEventBlock._gen_sample_profile_event(
            name="profile_4", instruction_id=2, time=(7, 8), delegate_debug_id=100
        )
        run_data_3 = flatcc.RunData(
            name="signature_b",
            bundled_input_index=-1,
            allocators=[],
            events=[
                flatcc.Event(
                    allocation_event=None,
                    debug_event=debug_event_3,
                    profile_event=None,
                ),
                flatcc.Event(
                    allocation_event=None,
                    debug_event=None,
                    profile_event=profile_event_3,
                ),
                flatcc.Event(
                    allocation_event=None,
                    debug_event=None,
                    profile_event=profile_event_4,
                ),
            ],
        )

        return ETDumpFlatCC(version=0, run_data=[run_data_1, run_data_2, run_data_3])

    @staticmethod
    def _get_sample_etdump_flatcc_debug_events_only(
        event_name: str,
        delegate_debug_id: str,
    ) -> flatcc.ETDumpFlatCC:
        """
        Helper for getting a sample ETDumpFlatCC object with RunData signature_a
        and (debug_event_delegated, debug_event_non_delegated, no profile event)
        """

        debug_event_delegated = TestEventBlock._gen_sample_debug_event(
            instruction_id=1, delegate_debug_id=delegate_debug_id, name=event_name
        )
        debug_event_non_delegated = TestEventBlock._gen_sample_debug_event(
            instruction_id=1, name=event_name
        )
        run_data_1 = flatcc.RunData(
            name="signature_a",
            bundled_input_index=-1,
            allocators=[],
            events=[
                flatcc.Event(
                    allocation_event=None,
                    debug_event=debug_event_delegated,
                    profile_event=None,
                ),
                flatcc.Event(
                    allocation_event=None,
                    debug_event=debug_event_non_delegated,
                    profile_event=None,
                ),
            ],
        )

        return ETDumpFlatCC(version=0, run_data=[run_data_1])

    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~ Tests ~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    def test_gen_from_etdump(self) -> None:
        """
        Test "e2e" generation of EventBlocks given an ETDump
            - Generated via EventBlock.gen_from_etdump

        Specifically it tests for external correctness:
        - Correct number of EventBlocks
        - Correct number of Events and Raw Data values (iterations)
        """

        etdump: ETDumpFlatCC = TestEventBlock._get_sample_etdump_flatcc()
        blocks: List[EventBlock] = EventBlock._gen_from_etdump(etdump)

        self.assertEqual(len(blocks), 2, f"Expected 2 runs, got {len(blocks)}")

        # One EventBlock should have 1 event with 2 iterations
        # The other EventBlock should have 2 events with 1 iterations
        run_counts = set()
        for block in blocks:
            if (perf_data := block.events[0].perf_data) is not None:
                run_counts.add((len(block.events), len(perf_data.raw)))
        self.assertSetEqual(run_counts, {(1, 2), (2, 1)})

    def test_gen_from_etdump_profiling_and_debugging(self) -> None:
        """
        Test "e2e" generation of EventBlocks given an ETDump with both profiling and debugging events
            - Generated via EventBlock.gen_from_etdump

        Specifically it tests for external correctness:
        - Correct number of EventBlocks
        - Correct number of raw perf_data and debug_data for each Event
        """
        etdump: ETDumpFlatCC = (
            TestEventBlock._get_sample_etdump_flatcc_profiling_and_debugging()
        )
        blocks: List[EventBlock] = EventBlock._gen_from_etdump(etdump)

        self.assertEqual(len(blocks), 2, f"Expected 2 runs, got {len(blocks)}")

        # One EventBlock should have 1 event with 2 iterations
        # and 1 debug data (because we only populate debug data in the first iteration)
        self.assertEqual(len(blocks[0].events), 1)
        if (perf_data := blocks[0].events[0].perf_data) is not None:
            self.assertEqual(len(perf_data.raw), 2)
        self.assertEqual(len(blocks[0].events[0].debug_data), 1)

        # The other EventBlock should have 2 events with 1 iterations, and only the fist event has debug data
        self.assertEqual(len(blocks[1].events), 2)
        perf_data = blocks[1].events[0].perf_data
        self.assertIsNotNone(perf_data)
        self.assertEqual(len(perf_data.raw), 1)

        perf_data = blocks[1].events[1].perf_data
        self.assertIsNotNone(perf_data)
        self.assertEqual(len(perf_data.raw), 1)
        self.assertEqual(len(blocks[1].events[0].debug_data), 1)
        self.assertEqual(len(blocks[1].events[1].debug_data), 0)

    def test_gen_from_etdump_inconsistent_debug_data(self) -> None:
        """
        Make sure AssertionError is thrown when intermediate outputs are different across
        different iterations of a model run
        """
        etdump: ETDumpFlatCC = (
            TestEventBlock._get_sample_etdump_flatcc_inconsistent_debug_data()
        )
        with self.assertRaises(AssertionError):
            EventBlock._gen_from_etdump(etdump)

    def test_gen_from_etdump_debug_events_only(self) -> None:
        """
        Test generation of EventBlocks given an ETDump with only debugging events

        Specifically it tests:
        - Correct number of EventBlocks and Events
        - Correct name of each Event
        """
        event_name = "test_debug_event_only"
        delegate_debug_id = "debug_id"
        etdump: ETDumpFlatCC = (
            TestEventBlock._get_sample_etdump_flatcc_debug_events_only(
                event_name=event_name,
                delegate_debug_id=delegate_debug_id,
            )
        )
        event_blocks = EventBlock._gen_from_etdump(etdump)
        self.assertEqual(len(event_blocks), 1)
        self.assertEqual(len(event_blocks[0].events), 2)
        # Delegated event uses delegate_debug_id as event name
        self.assertEqual(event_blocks[0].events[0].name, delegate_debug_id)
        # Non delegated event uses event_name as event name
        self.assertEqual(event_blocks[0].events[1].name, event_name)

    def test_inspector_event_generation(self) -> None:
        """
        Test Inspector.Event derivation from various ProfileEvent cases
        - Non Delegated
        - Delegate with Int Debug ID
        - Delegate with String Debug ID
        """

        def _test_profile_event_generation(
            name: str,
            instruction_id: int,
            delegate_debug_id_int: Optional[int] = None,
            delegate_debug_id_str: Optional[str] = None,
            scale_factor: int = 1000,
        ) -> None:
            """
            Helper function for testing that the provided ProfileEvent fields are
            properly translated to Inspector.ProfileEventSignature and Inspector.Event
            """
            delegate_debug_id = delegate_debug_id_int or delegate_debug_id_str
            profile_event: flatcc.ProfileEvent = (
                TestEventBlock._gen_sample_profile_event(
                    name,
                    instruction_id,
                    (0, 1),
                    delegate_debug_id,
                )
            )

            # Test Signature Generation
            profile_signature = ProfileEventSignature._gen_from_event(profile_event)
            expected_signature = ProfileEventSignature(
                name,
                instruction_id,
                delegate_debug_id_int,
                delegate_debug_id_str,
            )
            self.assertEqual(profile_signature, expected_signature)

            event_signature = EventSignature(
                instruction_id=instruction_id,
                profile_event_signature=profile_signature,
            )

            # Test Event Generation
            durations = [10, 20, 30]
            delegate_debug_metadatas = ["metadata_0", "metadata_1", "metadata_2"]
            profile_events: List[flatcc.ProfileEvent] = [
                TestEventBlock._gen_sample_profile_event(
                    name,
                    instruction_id,
                    (0, time),
                    delegate_debug_id,
                    (
                        delegate_debug_metadatas[index]
                        if delegate_debug_id is not None
                        else None
                    ),
                )
                for index, time in enumerate(durations)
            ]
            instruction_events = [
                InstructionEvent(
                    signature=InstructionEventSignature(
                        instruction_id=instruction_id, chain_index=0
                    ),
                    profile_events=[profile_event],
                )
                for profile_event in profile_events
            ]
            event = Event._gen_from_inference_events(
                event_signature, instruction_events, scale_factor=scale_factor
            )

            is_delegated = delegate_debug_id is not None
            expected_event = Event(
                name=str(delegate_debug_id) if is_delegated else name,
                perf_data=PerfData(
                    [float(duration) / scale_factor for duration in durations]
                ),
                delegate_debug_identifier=delegate_debug_id,
                is_delegated_op=is_delegated,
                _delegate_debug_metadatas=(
                    delegate_debug_metadatas if is_delegated else []
                ),
                _instruction_id=event_signature.instruction_id,
            )
            self.assertEqual(event, expected_event)

            # Test delegate_debug_metadata_parsing
            if is_delegated:
                expected_event = Event(
                    name=str(delegate_debug_id) if is_delegated else name,
                    perf_data=PerfData(
                        [float(duration) / scale_factor for duration in durations]
                    ),
                    delegate_debug_identifier=delegate_debug_id,
                    is_delegated_op=is_delegated,
                    _delegate_debug_metadatas=delegate_debug_metadatas,
                    _instruction_id=event_signature.instruction_id,
                    _delegate_metadata_parser=lambda metadatas: {
                        "joined": "-".join(metadatas)
                    },
                )
                self.assertEqual(
                    expected_event.delegate_debug_metadatas,
                    {"joined": "-".join(delegate_debug_metadatas)},
                )

        # Non Delegated
        _test_profile_event_generation("non-delegate", 1)

        # Delegate with Int Debug ID
        _test_profile_event_generation("delegate", 1, 100)

        # Delegate with String Debug ID
        _test_profile_event_generation("delegate", 1, None, "identifier")

        # Manipulating the scale factor
        _test_profile_event_generation(
            "delegate", 1, None, "identifier", scale_factor=10000
        )

    def test_gen_resolve_debug_handles(self) -> None:
        """
        Test that gen_resolve_debug_handles() correctly populates the EventBlock
        """

        def _gen_event_helper(events: List[ProfileEvent]) -> Event:
            """
            Helper function to generate an Event given a set of ProfileEvents
            """
            profile_event = events[0]
            profile_signature = ProfileEventSignature._gen_from_event(profile_event)
            event_signature = EventSignature(
                profile_event.instruction_id, profile_signature
            )
            instruction_events = [
                InstructionEvent(
                    signature=InstructionEventSignature(
                        instruction_id=profile_event.instruction_id,
                        chain_index=profile_event.chain_index,
                    ),
                    profile_events=[event],
                )
                for event in events
            ]
            return Event._gen_from_inference_events(event_signature, instruction_events)

        # Create Test Data

        # Non-Delegated
        non_delegated_profile_events_1 = [
            TestEventBlock._gen_sample_profile_event("non_del_1", 0, (0, 1)),
            TestEventBlock._gen_sample_profile_event("non_del_1", 0, (0, 1)),
        ]
        non_delegated_profile_events_2 = [
            TestEventBlock._gen_sample_profile_event("non_del_2", 1, (0, 1)),
        ]
        non_delegated_event_1 = _gen_event_helper(non_delegated_profile_events_1)
        non_delegated_event_2 = _gen_event_helper(non_delegated_profile_events_2)

        # Delegated
        delegated_profile_events_1 = [
            TestEventBlock._gen_sample_profile_event("del_1", 0, (0, 1), 10),
            TestEventBlock._gen_sample_profile_event("del_1", 0, (0, 1), 10),
            TestEventBlock._gen_sample_profile_event("del_1", 0, (0, 1), 10),
        ]
        delegated_profile_events_2 = [
            TestEventBlock._gen_sample_profile_event("del_2", 2, (0, 1), 20),
        ]
        delegated_event_1 = _gen_event_helper(delegated_profile_events_1)
        delegated_event_2 = _gen_event_helper(delegated_profile_events_2)

        # Create Test EventBlock
        event_block = EventBlock(
            name="test_name_1",
            events=[
                non_delegated_event_1,
                non_delegated_event_2,
                delegated_event_1,
                delegated_event_2,
            ],
        )

        # Create Test Maps
        handle_map = {"0": [100], "1": [110], "2": [120]}
        delegate_map = {
            "0": DelegateMetadata(
                {
                    "name": "delegate",
                    "delegate_map": {10: (100, 1000)},
                }
            ),
            "2": DelegateMetadata(
                {
                    "name": "delegate_2",
                    "delegate_map": {20: (200,)},
                }
            ),
        }
        event_block._gen_resolve_debug_handles(handle_map, delegate_map)

        # Verify Results
        for event in event_block.events:
            # To satisfy type checker
            assert event._instruction_id is not None
            if (
                delegate_debug_identifier := event.delegate_debug_identifier
            ) is not None:
                # Delegated
                metadata = delegate_map[str(event._instruction_id)]
                self.assertEqual(event.delegate_backend_name, metadata["name"])
                self.assertEqual(
                    event.debug_handles,
                    metadata["delegate_map"][delegate_debug_identifier],  # pyre-ignore
                )
            else:
                # Non Delegated
                self.assertEqual(
                    event.debug_handles, handle_map[str(event._instruction_id)]
                )
