# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import random
import statistics
import tempfile
import unittest
from contextlib import redirect_stdout

from typing import List

from unittest.mock import patch

from executorch.exir import ExportedProgram
from executorch.sdk.debug_format.et_schema import OperatorNode
from executorch.sdk.etrecord import generate_etrecord, parse_etrecord
from executorch.sdk.etrecord.tests.etrecord_test import TestETRecord

from executorch.sdk.inspector import inspector

from executorch.sdk.inspector.inspector import Event, EventBlock, Inspector, PerfData


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
            inspector, "parse_etrecord", return_value=None
        ) as mock_parse_etrecord, patch.object(
            inspector, "gen_etdump_object", return_value=None
        ) as mock_gen_etdump, patch.object(
            EventBlock, "_gen_from_etdump"
        ) as mock_gen_from_etdump, patch.object(
            inspector, "gen_graphs_from_etrecord"
        ) as mock_gen_graphs_from_etrecord:
            # Call the constructor of Inspector
            Inspector(
                etdump_path=ETDUMP_PATH,
                etrecord_path=ETRECORD_PATH,
            )

            # Assert that expected functions are called
            mock_parse_etrecord.assert_called_once_with(etrecord_path=ETRECORD_PATH)
            mock_gen_etdump.assert_called_once_with(etdump_path=ETDUMP_PATH)
            mock_gen_from_etdump.assert_called_once()
            # Because we mocked parse_etrecord() to return None, this method shouldn't be called
            mock_gen_graphs_from_etrecord.assert_not_called()

    def test_inspector_print_data_tabular(self):
        # Create a context manager to patch functions called by Inspector.__init__
        with patch.object(inspector, "parse_etrecord", return_value=None), patch.object(
            inspector, "gen_etdump_object", return_value=None
        ), patch.object(EventBlock, "_gen_from_etdump"), patch.object(
            inspector, "gen_graphs_from_etrecord"
        ):
            # Call the constructor of Inspector
            inspector_instance = Inspector(
                etdump_path=ETDUMP_PATH,
                etrecord_path=ETRECORD_PATH,
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

    def test_inspector_get_exported_program(self):
        # Create a context manager to patch functions called by Inspector.__init__
        with patch.object(inspector, "parse_etrecord", return_value=None), patch.object(
            inspector, "gen_etdump_object", return_value=None
        ), patch.object(EventBlock, "_gen_from_etdump"), patch.object(
            inspector, "gen_graphs_from_etrecord"
        ), patch.object(
            inspector, "create_debug_handle_to_op_node_mapping"
        ):
            # Call the constructor of Inspector
            inspector_instance = Inspector(
                etdump_path=ETDUMP_PATH,
                etrecord_path=ETRECORD_PATH,
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
