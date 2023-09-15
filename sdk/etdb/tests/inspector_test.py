# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import random
import statistics
import unittest
from contextlib import redirect_stdout

from typing import List

from unittest.mock import patch

from executorch.sdk.etdb import inspector

from executorch.sdk.etdb.inspector import Event, EventBlock, Inspector, PerfData


OP_TYPE = "aten::add"
EVENT_BLOCK_NAME = "block_0"
EVENTS_SIZE = 5
RAW_DATA_SIZE = 10
ETDUMP_PATH = "unittest_etdump_path"
ETRECORD_PATH = "unittest_etrecord_path"


class TestInspector(unittest.TestCase):
    def test_perf_data(self) -> None:
        random_floats = self._gen_random_float_list()
        perfData = PerfData(random_floats)

        # Intentionally use a different way to calculate median from the implementation
        self.assertEqual(perfData.median, statistics.median(random_floats))

    def test_event_block_to_dataframe(self) -> None:
        eventBlock = EventBlock(name=EVENT_BLOCK_NAME, events=self._gen_random_events())

        df = eventBlock.to_dataframe()
        # Check some fields of the returned dataframe
        self.assertEqual(len(df), EVENTS_SIZE)
        self.assertTrue("op_0" in df["event_name"].values)
        self.assertEqual(len(df["raw"].values[0]), RAW_DATA_SIZE)
        self.assertEqual(df["op_type"].values[0][0], OP_TYPE)

    def test_inspector_constructor(self):
        # Create a context manager to patch Inspector.__init__
        with patch.object(
            inspector, "parse_etrecord"
        ) as mock_parse_etrecord, patch.object(
            inspector, "gen_graphs_from_etrecord", return_value=None
        ) as mock_gen_graphs:
            # Call the constructor of Inspector
            Inspector(
                etdump_path=ETDUMP_PATH,
                etrecord_path=ETRECORD_PATH,
            )

            # Assert that expected functions are called
            mock_parse_etrecord.assert_called_once_with(
                etrecord_path=ETRECORD_PATH,
            )
            mock_gen_graphs.assert_called_once()

    def test_inspector_methods(self):
        inspector_instance = Inspector()

        # Test get_event_blocks() method. The mock inspector instance should have an empty event blocks list
        self.assertEqual(inspector_instance.get_event_blocks(), [])
        # Then add a non-empty event block list to the mock inspector instance, and test the get_event_blocks() method again
        event_block_list = [
            EventBlock(name=EVENT_BLOCK_NAME, events=self._gen_random_events())
        ]
        inspector_instance.event_blocks = event_block_list
        self.assertEqual(inspector_instance.get_event_blocks(), event_block_list)

        # Call print_data_tabular(), make sure it doesn't crash
        with redirect_stdout(None):
            inspector_instance.print_data_tabular()

    def _gen_random_float_list(self) -> List[float]:
        return [random.uniform(0, 10) for _ in range(RAW_DATA_SIZE)]

    def _gen_random_events(self) -> List[Event]:
        events = []
        for i in range(EVENTS_SIZE):
            events.append(
                Event(
                    name=f"op_{i}",
                    op_type=[OP_TYPE],
                    perf_data=PerfData(self._gen_random_float_list()),
                )
            )
        return events
