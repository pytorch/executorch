# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-unsafe

import copy
import os
import random
import statistics
import tempfile
import unittest
from contextlib import redirect_stdout

from typing import Callable, List, Union

from unittest.mock import patch

import pandas as pd

import torch
import torch.fx
import torch.utils._pytree as pytree

from executorch.backends.xnnpack.partition.xnnpack_partitioner import XnnpackPartitioner
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
from executorch.devtools.inspector.tests.inspector_test_utils import (
    check_if_debug_handle_to_op_names_match,
    check_if_intermediate_outputs_match,
    model_registry,
)
from executorch.exir import (
    EdgeCompileConfig,
    EdgeProgramManager,
    ExecutorchProgramManager,
    to_edge,
    to_edge_transform_and_lower,
)
from executorch.extension.pybindings.portable_lib import (
    _load_for_executorch_from_buffer,
)
from torch.export import export, ExportedProgram


OP_TYPE = "aten::add"
EVENT_BLOCK_NAME = "block_0"
EVENTS_SIZE = 10
RAW_DATA_SIZE = 10
ETDUMP_PATH = "unittest_etdump_path"
ETRECORD_PATH = "unittest_etrecord_path"


# TODO: write an E2E test: create an inspector instance, mock just the file reads, and then verify the external correctness
class TestInspector(unittest.TestCase):
    def test_perf_data(self) -> None:
        random_floats = self._gen_random_float_list()
        perfData = PerfData(random_floats)

        # Intentionally use a different way to calculate p50 from the implementation
        self.assertAlmostEqual(perfData.p50, statistics.median(random_floats))

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
            mock_gen_etdump.assert_called_once_with(
                etdump_path=ETDUMP_PATH, etdump_data=None
            )
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
            {
                debug_handle: [
                    node_0,
                ]
            }
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
            {
                debug_handles[0]: [
                    node_0,
                ],
                debug_handles[1]: [
                    node_1,
                ],
            }
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
                    extra_recorded_export_modules={
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

    def test_etrecord_populates_correct_edge_dialect_aot_intermediate_outputs(self):
        with tempfile.NamedTemporaryFile(suffix=".bin") as tmp_file:
            etrecord_path = tmp_file.name
            mod = model_registry["ConvLinearModel"]()
            input_tensor = torch.tensor(
                [[[[1.0, 2.0], [3.0, 4.0]]]], requires_grad=True
            )
            aten_model: ExportedProgram = export(mod, (input_tensor,), strict=True)
            edge_program_manager: EdgeProgramManager = to_edge(
                aten_model, compile_config=EdgeCompileConfig(_check_ir_validity=True)
            )
            edge_program_manager_copy = copy.deepcopy(edge_program_manager)
            et_program_manager: ExecutorchProgramManager = (
                edge_program_manager.to_executorch()
            )
            # Generate ETRecord
            generate_etrecord(
                etrecord_path, edge_program_manager_copy, et_program_manager
            )
            with patch.object(
                Inspector, "_consume_etrecord", return_value=None
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
                    etrecord=etrecord_path,
                )

                inspector_instance._etrecord._representative_inputs = (
                    aten_model.example_inputs[0]
                )

                aot_intermediate_outputs, aot_debug_handle_to_op_names = (
                    inspector_instance._get_aot_intermediate_outputs_and_op_names()
                )
                self.assertTrue(
                    check_if_intermediate_outputs_match(
                        aot_intermediate_outputs,
                        mod.get_edge_dialect_expected_intermediate_outputs(),
                    )
                )

                self.assertTrue(
                    check_if_debug_handle_to_op_names_match(
                        aot_debug_handle_to_op_names,
                        mod.get_edge_dialect_expected_debug_handle_to_op_names(),
                    )
                )

    def test_etrecord_populates_correct_export_program_aot_intermediate_outputs(self):
        with tempfile.NamedTemporaryFile(suffix=".bin") as tmp_file:
            etrecord_path = tmp_file.name
            mod = model_registry["ConvLinearModel"]()
            input_tensor = mod.get_input()
            aten_model: ExportedProgram = export(mod, (input_tensor,), strict=True)
            edge_program_manager: EdgeProgramManager = to_edge(aten_model)
            edge_program_manager_copy = copy.deepcopy(edge_program_manager)
            et_program_manager: ExecutorchProgramManager = (
                edge_program_manager.to_executorch()
            )
            # Generate ETRecord with the exported program
            generate_etrecord(
                etrecord_path,
                edge_program_manager_copy,
                et_program_manager,
                exported_program=aten_model,
            )
            with patch.object(
                Inspector, "_consume_etrecord", return_value=None
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
                    etrecord=etrecord_path,
                )

                inspector_instance._etrecord._representative_inputs = (
                    aten_model.example_inputs[0]
                )

                aot_intermediate_outputs, aot_debug_handle_to_op_names = (
                    inspector_instance._get_aot_intermediate_outputs_and_op_names()
                )
                self.assertTrue(
                    check_if_intermediate_outputs_match(
                        aot_intermediate_outputs,
                        mod.get_exported_program_expected_intermediate_outputs(),
                    )
                )
                self.assertTrue(
                    check_if_debug_handle_to_op_names_match(
                        aot_debug_handle_to_op_names,
                        mod.get_exported_program_expected_debug_handle_to_op_names(),
                    )
                )

    def test_get_runtime_intermediate_outputs_and_op_names(self):
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
            # Add pre-defined event blocks to test _get_runtime_outputs().
            inspector_instance.event_blocks = [
                EventBlock(name=EVENT_BLOCK_NAME, events=self._gen_random_events())
            ]

            runtime_outputs, op_names = (
                inspector_instance._get_runtime_intermediate_outputs_and_op_names()
            )
            # These outputs and op_names dictionaries should all have 5 keys
            self.assertEqual(
                len(runtime_outputs),
                5,
            )
            self.assertEqual(
                len(op_names),
                5,
            )

            # Check that keys (0,) and (1,) are not in these two dictionaries(skip OPERATOR_CALL and op_types are empty)
            self.assertNotIn((0,), runtime_outputs)
            self.assertNotIn((1,), runtime_outputs)
            self.assertNotIn((0,), op_names)
            self.assertNotIn((1,), op_names)

            # Same debug_handle but different instruction_id, should record the last one
            self.assertIn((4,), runtime_outputs)
            self.assertIn((4,), op_names)
            self.assertTrue(
                torch.allclose(
                    runtime_outputs[(4,)][0][0], torch.tensor([4.0, 5.0, 6.0])
                )
            )
            self.assertEqual(op_names[(4,)], ["op_3"])

            # Check that keys (5,) to (8,) are in the dictionary and have values of the correct size
            for key in range(5, 9):
                self.assertIn((key,), runtime_outputs)
                self.assertIn((key,), op_names)

    def test_calculate_numeric_gap(self):
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

            aot_intermediate_outputs = {
                (0,): torch.tensor([1.0, 2.0, 3.0]),
                (1,): torch.tensor([4.0, 5.0, 6.0]),
            }

            runtime_intermediate_outputs = {
                (0,): ([torch.tensor([2.0, 1.0, 4.0])], 1),
                (1,): ([torch.tensor([3.0, 6.0, 5.0])], 1),
            }

            aot_debug_handle_to_op_name = {(0,): "op_0", (1,): "op_1"}
            runtime_debug_handle_to_op_name = {(0,): "op_0", (1,): "op_1"}

            inspector_instance._get_aot_intermediate_outputs_and_op_names = lambda x: (
                aot_intermediate_outputs,
                aot_debug_handle_to_op_name,
            )
            inspector_instance._get_runtime_intermediate_outputs_and_op_names = (
                lambda: (runtime_intermediate_outputs, runtime_debug_handle_to_op_name)
            )

            df = inspector_instance.calculate_numeric_gap(distance="L1")
            self.assertIsInstance(df, pd.DataFrame)
            self.assertEqual(len(df), 2)
            cols = set(df.columns)
            expected_cols = {
                "aot_ops",
                "aot_intermediate_output",
                "runtime_ops",
                "runtime_intermediate_output",
                "gap",
            }
            self.assertEqual(cols, expected_cols)
            for i, row in df.iterrows():
                # Dummpy key to get the expected aot/runtime internmediate outputs
                key = (i,)
                # aot_intermediate_output should equal aot_intermediate_outputs[key]
                self.assertTrue(
                    torch.allclose(
                        row["aot_intermediate_output"],
                        aot_intermediate_outputs[key],
                    )
                )
                # runtime_intermediate_output should equal runtime_intermediate_outputs[key]
                self.assertTrue(
                    torch.allclose(
                        row["runtime_intermediate_output"],
                        runtime_intermediate_outputs[key][0][0],
                    )
                )
                # gap should equal 3.0
                self.assertEqual(row["gap"][0], 3.0)

    @unittest.skip("ci config values are not propagated")
    def test_intermediate_tensor_comparison_with_torch_export(self):
        """Test intermediate tensor comparison using torch.export.export and to_edge_transform_and_lower."""

        class SimpleTestModel(torch.nn.Module):
            """A simple test model for demonstration purposes."""

            def __init__(self, hidden_size: int = 32, num_layers: int = 2):
                super().__init__()
                self.layers = torch.nn.ModuleList(
                    [
                        torch.nn.Linear(hidden_size, hidden_size)
                        for _ in range(num_layers)
                    ]
                )
                self.activation = torch.nn.ReLU()
                self.output_layer = torch.nn.Linear(hidden_size, 10)

            def forward(self, x: torch.Tensor) -> torch.Tensor:
                x = self.activation(self.layers[0](x))
                y = self.activation(self.layers[1](x))
                return y, self.output_layer(x)

        # Create test model and inputs
        model = SimpleTestModel(hidden_size=32, num_layers=2)
        model.eval()

        # Create representative inputs (smaller for faster testing)
        batch_size, seq_len, hidden_size = 1, 8, 32
        input_tensor = torch.randn(batch_size, seq_len, hidden_size)
        example_inputs = (input_tensor,)
        representative_inputs = [example_inputs]

        with tempfile.TemporaryDirectory() as tmp_dir:
            model_path = os.path.join(tmp_dir, "model.pte")
            etrecord_path = os.path.join(tmp_dir, "etrecord.bin")

            # Step 1: Export using torch.export.export
            exported_program = torch.export.export(model, example_inputs)
            self.assertIsNotNone(exported_program)

            # Step 2: Lower to XNNPACK with generate_etrecord=True
            edge_compile_config = EdgeCompileConfig(_check_ir_validity=False)
            edge_program_manager = to_edge_transform_and_lower(
                exported_program,
                partitioner=[XnnpackPartitioner()],
                compile_config=edge_compile_config,
                generate_etrecord=True,
            )
            self.assertIsNotNone(edge_program_manager)

            # Step 3: Generate ETRecord from edge program manager
            # Step 4: Convert to executorch and save as PTE
            executorch_program = edge_program_manager.to_executorch()
            et_record = executorch_program.get_etrecord()
            self.assertIsNotNone(et_record)

            # Update with representative inputs
            flattened_x = pytree.tree_flatten(representative_inputs[0])[0]
            et_record.update_representative_inputs(flattened_x)
            et_record.save(etrecord_path)

            with open(model_path, "wb") as f:
                executorch_program.write_to_file(f)

            # Step 5: Test intermediate output comparison using pybind APIs
            # Read the PTE file
            with open(model_path, "rb") as f:
                pte_buffer = f.read()

            etdump_path = os.path.join(tmp_dir, "etdump.etdp")
            debug_buffer_path = os.path.join(tmp_dir, "debug_buffer.bin")

            # Load the PTE file with ETDump enabled using pybind API
            executorch_module = _load_for_executorch_from_buffer(
                pte_buffer,
                enable_etdump=True,
                debug_buffer_size=1024 * 1024,  # 1MB for testing
            )
            self.assertIsNotNone(executorch_module)

            # Run the model with the given input using pybind API
            flattened_x = pytree.tree_flatten(representative_inputs[0])[0]
            executorch_module.run_method("forward", tuple(flattened_x))

            # Write the ETDump results to a file using pybind API
            executorch_module.write_etdump_result_to_file(
                etdump_path, debug_buffer_path
            )

            # Step 6: Use Inspector API to compare intermediate outputs
            try:
                inspector = Inspector(
                    etdump_path=etdump_path,
                    etrecord=etrecord_path,
                    debug_buffer_path=debug_buffer_path,
                )
            except FileNotFoundError as e:
                new_message = f"{e} You likely need to run the test with --config executorch.event_tracer_enabled=true"
                raise RuntimeError(new_message) from e
            self.assertIsNotNone(inspector)

            # Calculate numerical gap using SNR metric
            df = inspector.calculate_numeric_gap("SNR")

            # Verify that we got some intermediate tensor comparisons
            # The exact number will depend on the model structure and partitioning
            self.assertEqual(len(df), 2)

    def _gen_random_float_list(self) -> List[float]:
        return [random.uniform(0, 10) for _ in range(RAW_DATA_SIZE)]

    def _gen_random_runtime_output(
        self,
    ) -> List[Union[None, List[torch.Tensor], bool, float, int, str, torch.Tensor]]:
        return [torch.randn(RAW_DATA_SIZE)]

    def test_disable_debug_handle_validation_with_symbolic_shapes(self):
        """
        Test that demonstrates the issue with symbolic shape related nodes losing from_node info
        during dynamic shape based export, and shows how disable_debug_handle_valdiation parameter
        in propagate_back_debug_handle allows validation to be bypassed.
        """
        from executorch.devtools.inspector._inspector_utils import (
            propagate_back_debug_handle,
        )

        class SymbolicShapeModel(torch.nn.Module):
            """Model that will have symbolic shape related operations after export."""

            def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
                # This will create symbolic shape nodes during dynamic export
                batch_size = x.shape[0]
                x = x + torch.rand((batch_size, 1))
                # Masking operation that creates gt/lt nodes
                valid_mask = mask > 0.5
                x = torch.where(valid_mask, x, torch.zeros_like(x))
                return x

        # Create model and dynamic inputs
        model = SymbolicShapeModel()
        batch_size = 2
        seq_len = 4
        x = torch.randn(batch_size, seq_len)
        mask = torch.rand(batch_size, seq_len)
        example_inputs = (x, mask)

        # Export with dynamic shapes to create symbolic shape related nodes
        dynamic_shapes = {
            "x": {0: torch.export.Dim("batch_size", min=1, max=10)},
            "mask": {0: torch.export.Dim("batch_size", min=1, max=10)},
        }

        exported_program = torch.export.export(
            model, example_inputs, dynamic_shapes=dynamic_shapes, strict=True
        )

        """
        In this case origina aten graph has sym_size_int_2 node but when we look at
        nodes metadata in edge_program_manager, its sym_size node's from_node says
        sym_size_int_3 which is not in the original aten graph.
        """
        # Create edge program - this is where from_node info can be lost for symbolic shape nodes
        edge_program_manager: EdgeProgramManager = to_edge(exported_program)
        edge_program_manager_copy = copy.deepcopy(edge_program_manager)
        et_program_manager: ExecutorchProgramManager = (
            edge_program_manager.to_executorch()
        )

        with tempfile.NamedTemporaryFile(suffix=".bin") as tmp_file:
            etrecord_path = tmp_file.name

            # Generate ETRecord with the exported program (aten graph)
            generate_etrecord(
                etrecord_path,
                edge_program_manager_copy,
                et_program_manager,
                exported_program=exported_program,
            )

            # Create Inspector and get etrecord
            with patch.object(
                _inspector, "gen_etdump_object", return_value=None
            ), patch.object(EventBlock, "_gen_from_etdump"):
                inspector_instance = Inspector(
                    etdump_path=ETDUMP_PATH,
                    etrecord=etrecord_path,
                )

                # Extract the necessary values from the inspector's etrecord
                exported_program_from_etrecord = (
                    inspector_instance._etrecord.exported_program
                )
                export_graph_id = inspector_instance._etrecord.export_graph_id
                edge_dialect_program = inspector_instance._etrecord.edge_dialect_program

                # Ensure we have all the necessary components
                self.assertIsNotNone(exported_program_from_etrecord)
                self.assertIsNotNone(export_graph_id)
                self.assertIsNotNone(edge_dialect_program)

                # Test propagate_back_debug_handle with validation enabled (should fail or return False)
                # This demonstrates the issue with symbolic shape nodes losing from_node info
                validation_enabled_result = propagate_back_debug_handle(
                    exported_program_from_etrecord,
                    export_graph_id,
                    edge_dialect_program,
                    disable_debug_handle_valdiation=False,
                )

                # With validation enabled, it should return False when from_node info is lost
                self.assertFalse(
                    validation_enabled_result,
                    "propagate_back_debug_handle should return False when validation is enabled "
                    "and symbolic shape nodes lose from_node info",
                )

                # Test propagate_back_debug_handle with validation disabled (should succeed)
                # This shows how the disable_debug_handle_valdiation flag allows the function to work
                validation_disabled_result = propagate_back_debug_handle(
                    exported_program_from_etrecord,
                    export_graph_id,
                    edge_dialect_program,
                    disable_debug_handle_valdiation=True,
                )

                # With validation disabled, it should return True even when from_node info is lost
                self.assertTrue(
                    validation_disabled_result,
                    "propagate_back_debug_handle should return True when validation is disabled, "
                    "allowing best effort comparison even when from_node info is lost",
                )

    def _gen_random_events(self) -> List[Event]:
        events = []
        for i in range(2):
            events.append(
                # OPERATOR_CALL with debug_handle/instruction_id 0 and 2
                Event(
                    name="OPERATOR_CALL",
                    op_types=[OP_TYPE],
                    perf_data=PerfData(self._gen_random_float_list()),
                    debug_handles=i * 2,
                    _instruction_id=i * 2,
                    debug_data=self._gen_random_runtime_output(),
                )
            )
            events.append(
                # op_0/op_1 wiht empty op_types and with debug_handle/instruction_id 1 and 3
                Event(
                    name=f"op_{i}",
                    op_types=[],
                    perf_data=PerfData(self._gen_random_float_list()),
                    debug_handles=i * 2 + 1,
                    _instruction_id=i * 2 + 1,
                    debug_data=self._gen_random_runtime_output(),
                )
            )

        # op_2 with debug_handle/instruction_id 4
        events.append(
            Event(
                name="op_2",
                op_types=[OP_TYPE],
                perf_data=PerfData(self._gen_random_float_list()),
                debug_handles=4,
                debug_data=[torch.tensor([1.0, 2.0, 3.0])],
                _instruction_id=4,
            )
        )
        # op_3 also with debug_handle 4 but with instruction_id 5
        events.append(
            Event(
                name="op_3",
                op_types=[OP_TYPE],
                perf_data=PerfData(self._gen_random_float_list()),
                debug_handles=4,
                debug_data=[torch.tensor([4.0, 5.0, 6.0])],
                _instruction_id=5,
            )
        )

        # op_4 to op_7 with debug_handle 5 to 8 and instruction_id 6 to 9
        for i in range(4, EVENTS_SIZE - 2):
            events.append(
                Event(
                    name=f"op_{i}",
                    op_types=[OP_TYPE],
                    perf_data=PerfData(self._gen_random_float_list()),
                    debug_handles=i + 1,
                    debug_data=self._gen_random_runtime_output(),
                    _instruction_id=i + 2,
                )
            )
        return events
