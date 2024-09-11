# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-unsafe

import random
import statistics
import tempfile
import unittest
from contextlib import redirect_stdout

from typing import Callable, List

from unittest.mock import patch

from executorch.devtools import generate_etrecord, parse_etrecord
from executorch.devtools.debug_format.et_schema import OperatorNode
from executorch.devtools.etdump.schema_flatcc import ProfileEvent
from executorch.devtools.etrecord.tests.etrecord_test import TestETRecord

from executorch.devtools.inspector import (
    _inspector,
    Event,
    EventBlock,
    Inspector,
    PerfData,
)
from executorch.devtools.inspector._inspector import (
    DebugEventSignature,
    flatcc,
    InstructionEvent,
    InstructionEventSignature,
    ProfileEventSignature,
    TimeScale,
)

from executorch.exir import ExportedProgram


OP_TYPE = "aten::add"
EVENT_BLOCK_NAME = "block_0"
EVENTS_SIZE = 5
RAW_DATA_SIZE = 10
ETDUMP_PATH = "unittest_etdump_path"
ETRECORD_PATH = "unittest_etrecord_path"


# TODO: write an E2E test: create an inspector instance, mock just the file reads, and then verify the external correctness
class TestInspector(unittest.TestCase):
    def test_perf_data(self) -> None:
        random_floats = self._gen_random_float_list()
        perfData = PerfData(random_floats)

        # Intentionally use a different way to calculate p50 from the implementation
        self.assertEqual(perfData.p50, statistics.median(random_floats))

    def test_event_block_to_dataframe(self) -> None:
        eventBlock = EventBlock(name=EVENT_BLOCK_NAME, events=self._gen_random_events())

        df = eventBlock.to_dataframe()
        # Check some fields of the returned dataframe
        self.assertEqual(len(df), EVENTS_SIZE)
        self.assertTrue("op_0" in df["event_name"].values)
        self.assertEqual(len(df["raw"].values[0]), RAW_DATA_SIZE)
        self.assertEqual(df["op_types"].values[0][0], OP_TYPE)

    def test_inspector_constructor(self):
        # Create a context manager to patch functions called by Inspector.__init__
        with patch.object(
            _inspector, "parse_etrecord", return_value=None
        ) as mock_parse_etrecord, patch.object(
            _inspector, "gen_etdump_object", return_value=None
        ) as mock_gen_etdump, patch.object(
            EventBlock, "_gen_from_etdump"
        ) as mock_gen_from_etdump, patch.object(
            _inspector, "gen_graphs_from_etrecord"
        ) as mock_gen_graphs_from_etrecord:
            # Call the constructor of Inspector
            Inspector(
                etdump_path=ETDUMP_PATH,
                etrecord=ETRECORD_PATH,
            )

            # Assert that expected functions are called
            mock_parse_etrecord.assert_called_once_with(etrecord_path=ETRECORD_PATH)
            mock_gen_etdump.assert_called_once_with(etdump_path=ETDUMP_PATH)
            mock_gen_from_etdump.assert_called_once()
            # Because we mocked parse_etrecord() to return None, this method shouldn't be called
            mock_gen_graphs_from_etrecord.assert_not_called()

    def test_default_delegate_time_scale_converter(self):
        # Create a context manager to patch functions called by Inspector.__init__
        with patch.object(
            _inspector, "parse_etrecord", return_value=None
        ), patch.object(
            _inspector, "gen_etdump_object", return_value=None
        ), patch.object(
            EventBlock, "_gen_from_etdump"
        ) as mock_gen_from_etdump, patch.object(
            _inspector, "gen_graphs_from_etrecord"
        ), patch.object(
            _inspector, "create_debug_handle_to_op_node_mapping"
        ):
            # Call the constructor of Inspector
            Inspector(
                etdump_path=ETDUMP_PATH,
                etrecord=ETRECORD_PATH,
                source_time_scale=TimeScale.US,
                target_time_scale=TimeScale.S,
            )

            # Verify delegate_time_scale_converter is set to be a callable
            self.assertIsInstance(
                mock_gen_from_etdump.call_args.get("delegate_time_scale_converter"),
                Callable,
            )

    def test_inspector_print_data_tabular(self):
        # Create a context manager to patch functions called by Inspector.__init__
        with patch.object(
            _inspector, "parse_etrecord", return_value=None
        ), patch.object(
            _inspector, "gen_etdump_object", return_value=None
        ), patch.object(
            EventBlock, "_gen_from_etdump"
        ), patch.object(
            _inspector, "gen_graphs_from_etrecord"
        ):
            # Call the constructor of Inspector
            inspector_instance = Inspector(
                etdump_path=ETDUMP_PATH,
                etrecord=ETRECORD_PATH,
            )

            # The mock inspector instance starts with having an empty event blocks list.
            # Add non-empty event blocks to test print_data_tabular().
            inspector_instance.event_blocks = [
                EventBlock(name=EVENT_BLOCK_NAME, events=self._gen_random_events())
            ]
            # Call print_data_tabular(), make sure it doesn't crash
            with redirect_stdout(None):
                inspector_instance.print_data_tabular()

    def test_inspector_associate_with_op_graph_nodes_single_debug_handle(self):
        # Test on an event with a single debug handle
        debug_handle = 111
        event_with_single_debug_handle = Event(
            name="event_with_single_debug_handle",
            perf_data=PerfData(raw=[]),
            debug_handles=debug_handle,
        )
        node_0 = OperatorNode(
            name="node_0",
            metadata={
                "debug_handle": debug_handle,
                "stack_trace": "stack_trace_relu",
                "nn_module_stack": "module_hierarchy_relu",
            },
            op="op",
        )

        # Call the method that's under testing and verify
        event_with_single_debug_handle._associate_with_op_graph_nodes(
            {debug_handle: node_0}
        )

        expected_stack_traces = {"node_0": "stack_trace_relu"}
        self.assertEqual(
            event_with_single_debug_handle.stack_traces, expected_stack_traces
        )
        expected_module_hierarchy = {"node_0": "module_hierarchy_relu"}
        self.assertEqual(
            event_with_single_debug_handle.module_hierarchy, expected_module_hierarchy
        )
        expected_ops = ["op"]
        self.assertEqual(event_with_single_debug_handle.op_types, expected_ops)

    def test_inspector_associate_with_op_graph_nodes_multiple_debug_handles(self):
        # Test on an event with a sequence of debug handles
        debug_handles = [222, 333]
        event_with_multiple_debug_handles = Event(
            name="event_with_multiple_debug_handles",
            perf_data=PerfData(raw=[]),
            debug_handles=debug_handles,
        )
        node_0 = OperatorNode(
            name="node_0",
            metadata={
                "debug_handle": debug_handles[0],
                "stack_trace": "stack_trace_relu",
                "nn_module_stack": "module_hierarchy_relu",
            },
            op="op_0",
        )
        node_1 = OperatorNode(
            name="node_1",
            metadata={
                "debug_handle": debug_handles[1],
                "stack_trace": "stack_trace_conv",
                "nn_module_stack": "module_hierarchy_conv",
            },
            op="op_1",
        )

        # Call the method that's under testing and verify
        event_with_multiple_debug_handles._associate_with_op_graph_nodes(
            {debug_handles[0]: node_0, debug_handles[1]: node_1}
        )

        expected_stack_traces = {
            "node_0": "stack_trace_relu",
            "node_1": "stack_trace_conv",
        }
        self.assertEqual(
            event_with_multiple_debug_handles.stack_traces, expected_stack_traces
        )
        expected_module_hierarchy = {
            "node_0": "module_hierarchy_relu",
            "node_1": "module_hierarchy_conv",
        }
        self.assertEqual(
            event_with_multiple_debug_handles.module_hierarchy,
            expected_module_hierarchy,
        )
        expected_ops = ["op_0", "op_1"]
        self.assertEqual(event_with_multiple_debug_handles.op_types, expected_ops)

    def test_inspector_delegate_time_scale_converter(self):
        def time_scale_converter(event_name, time):
            return time / 10

        event = Event(
            name="",
            _delegate_metadata_parser=None,
            _delegate_time_scale_converter=None,
        )
        event_signature = ProfileEventSignature(
            name="",
            instruction_id=0,
            delegate_id_str="test_event",
        )
        instruction_events = [
            InstructionEvent(
                signature=InstructionEventSignature(0, 0),
                profile_events=[
                    ProfileEvent(
                        name="test_event",
                        chain_index=0,
                        instruction_id=0,
                        delegate_debug_id_int=None,
                        delegate_debug_id_str="test_event_delegated",
                        start_time=100,
                        end_time=200,
                        delegate_debug_metadata=None,
                    )
                ],
            )
        ]
        Event._populate_profiling_related_fields(
            event, event_signature, instruction_events, 1
        )
        # Value of the perf data before scaling is done.
        self.assertEqual(event.perf_data.raw[0], 100)
        event._delegate_time_scale_converter = time_scale_converter
        Event._populate_profiling_related_fields(
            event, event_signature, instruction_events, 1
        )
        # Value of the perf data after scaling is done. 200/10 - 100/10.
        self.assertEqual(event.perf_data.raw[0], 10)

    def test_inspector_get_exported_program(self):
        # Create a context manager to patch functions called by Inspector.__init__
        with patch.object(
            _inspector, "parse_etrecord", return_value=None
        ), patch.object(
            _inspector, "gen_etdump_object", return_value=None
        ), patch.object(
            EventBlock, "_gen_from_etdump"
        ), patch.object(
            _inspector, "gen_graphs_from_etrecord"
        ), patch.object(
            _inspector, "create_debug_handle_to_op_node_mapping"
        ):
            # Call the constructor of Inspector
            inspector_instance = Inspector(
                etdump_path=ETDUMP_PATH,
                etrecord=ETRECORD_PATH,
            )

            # Gen a mock etrecord
            captured_output, edge_output, et_output = TestETRecord().get_test_model()
            with tempfile.TemporaryDirectory() as tmpdirname:
                generate_etrecord(
                    tmpdirname + "/etrecord.bin",
                    edge_output,
                    et_output,
                    {
                        "aten_dialect_output": captured_output,
                    },
                )

                inspector_instance._etrecord = parse_etrecord(
                    tmpdirname + "/etrecord.bin"
                )

                self.assertTrue(
                    isinstance(
                        inspector_instance.get_exported_program(), ExportedProgram
                    )
                )

    def test_populate_debugging_related_fields_raises_for_inconsistent_events(self):
        ret_event: Event = Event(
            name="event",
        )

        debug_event_0 = flatcc.DebugEvent(
            name="event",
            chain_index=1,
            instruction_id=0,
            delegate_debug_id_int=1,
            delegate_debug_id_str=None,
            debug_entry=flatcc.Value(
                val=flatcc.ValueType.TENSOR.value,
                tensor=flatcc.Tensor(
                    scalar_type=flatcc.ScalarType.INT,
                    sizes=[2],
                    strides=[1],
                    offset=12345,
                ),
                tensor_list=None,
                int_value=None,
                float_value=None,
                double_value=None,
                bool_value=None,
                output=None,
            ),
        )

        # Note the sizes of this tensor are different from the previous one
        debug_event_1 = flatcc.DebugEvent(
            name="event",
            chain_index=1,
            instruction_id=0,
            delegate_debug_id_int=1,
            delegate_debug_id_str=None,
            debug_entry=flatcc.Value(
                val=flatcc.ValueType.TENSOR.value,
                tensor=flatcc.Tensor(
                    scalar_type=flatcc.ScalarType.INT,
                    sizes=[1],
                    strides=[1],
                    offset=23456,
                ),
                tensor_list=None,
                int_value=None,
                float_value=None,
                double_value=None,
                bool_value=None,
                output=None,
            ),
        )

        instruction_event_0 = InstructionEvent(
            signature=InstructionEventSignature(1, 1), debug_events=[debug_event_0]
        )
        instruction_event_1 = InstructionEvent(
            signature=InstructionEventSignature(1, 1), debug_events=[debug_event_1]
        )

        events = [instruction_event_0, instruction_event_1]

        # Expect AssertionError because 2 tensors have different sizes
        with self.assertRaises(AssertionError):
            Event._populate_debugging_related_fields(
                ret_event=ret_event,
                debug_event_signature=DebugEventSignature(instruction_id=1),
                events=events,
            )

    def test_populate_debugging_related_fields_passes_for_consistent_events(self):
        ret_event: Event = Event(
            name="event",
        )

        debug_event_0 = flatcc.DebugEvent(
            name="event",
            chain_index=1,
            instruction_id=0,
            delegate_debug_id_int=1,
            delegate_debug_id_str=None,
            debug_entry=flatcc.Value(
                val=flatcc.ValueType.TENSOR.value,
                tensor=flatcc.Tensor(
                    scalar_type=flatcc.ScalarType.INT,
                    sizes=[1],
                    strides=[1],
                    offset=12345,
                ),
                tensor_list=None,
                int_value=None,
                float_value=None,
                double_value=None,
                bool_value=None,
                output=None,
            ),
        )

        # Same as the event above except for offset
        debug_event_1 = flatcc.DebugEvent(
            name="event",
            chain_index=1,
            instruction_id=0,
            delegate_debug_id_int=1,
            delegate_debug_id_str=None,
            debug_entry=flatcc.Value(
                val=flatcc.ValueType.TENSOR.value,
                tensor=flatcc.Tensor(
                    scalar_type=flatcc.ScalarType.INT,
                    sizes=[1],
                    strides=[1],
                    offset=23456,
                ),
                tensor_list=None,
                int_value=None,
                float_value=None,
                double_value=None,
                bool_value=None,
                output=None,
            ),
        )

        instruction_event_0 = InstructionEvent(
            signature=InstructionEventSignature(1, 1), debug_events=[debug_event_0]
        )
        instruction_event_1 = InstructionEvent(
            signature=InstructionEventSignature(1, 1), debug_events=[debug_event_1]
        )

        events = [instruction_event_0, instruction_event_1]

        with patch.object(_inspector, "is_inference_output_equal", return_value=True):
            # Expect it runs with no error because is_inference_output_equal() is mocked to return True
            Event._populate_debugging_related_fields(
                ret_event=ret_event,
                debug_event_signature=DebugEventSignature(instruction_id=1),
                events=events,
            )

    def _gen_random_float_list(self) -> List[float]:
        return [random.uniform(0, 10) for _ in range(RAW_DATA_SIZE)]

    def _gen_random_events(self) -> List[Event]:
        events = []
        for i in range(EVENTS_SIZE):
            events.append(
                Event(
                    name=f"op_{i}",
                    op_types=[OP_TYPE],
                    perf_data=PerfData(self._gen_random_float_list()),
                )
            )
        return events
