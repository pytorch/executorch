# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-unsafe

import tempfile
import unittest
from typing import Dict, Tuple

import executorch.exir.tests.models as models

import torch
from executorch.devtools import generate_etrecord, parse_etrecord

from executorch.devtools.debug_format.base_schema import (
    OperatorGraph,
    OperatorNode,
    ValueNode,
)

from executorch.devtools.debug_format.et_schema import FXOperatorGraph
from executorch.devtools.etdump import schema_flatcc as flatcc

from executorch.devtools.etrecord.tests.etrecord_test import TestETRecord
from executorch.devtools.inspector._inspector_utils import (
    calculate_cosine_similarity,
    calculate_mse,
    calculate_snr,
    calculate_time_scale_factor,
    compare_intermediate_outputs,
    convert_to_float_tensor,
    create_debug_handle_to_op_node_mapping,
    EDGE_DIALECT_GRAPH_KEY,
    find_op_names,
    find_populated_event,
    gen_graphs_from_etrecord,
    get_aot_debug_handle_to_op_name_mapping,
    is_inference_output_equal,
    map_runtime_aot_intermediate_outputs,
    merge_runtime_overlapping_debug_handles,
    NodeFilter,
    propagate_back_debug_handle,
    TimeScale,
)
from executorch.devtools.inspector.numerical_comparator import L1Comparator
from executorch.exir import to_edge
from executorch.exir.debug_handle_utils import DEBUG_HANDLE_KEY, UNSET_DEBUG_HANDLE
from torch.export import export


class TestInspectorUtils(unittest.TestCase):
    def test_gen_graphs_from_etrecord(self):
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

            etrecord = parse_etrecord(tmpdirname + "/etrecord.bin")

            graphs = gen_graphs_from_etrecord(etrecord)

            self.assertTrue("aten_dialect_output/forward" in graphs)
            self.assertTrue(EDGE_DIALECT_GRAPH_KEY in graphs)

            self.assertTrue(
                isinstance(graphs["aten_dialect_output/forward"], FXOperatorGraph)
            )
            self.assertTrue(isinstance(graphs[EDGE_DIALECT_GRAPH_KEY], FXOperatorGraph))

    def test_create_debug_handle_to_op_node_mapping(self):
        graph, expected_mapping = gen_mock_operator_graph_with_expected_map()
        debug_handle_to_op_node_map = create_debug_handle_to_op_node_mapping(graph)

        self.assertEqual(debug_handle_to_op_node_map, expected_mapping)

    def test_find_populated_event(self):
        profile_event = flatcc.ProfileEvent(
            name="test_profile_event",
            chain_index=1,
            instruction_id=1,
            delegate_debug_id_str="",
            delegate_debug_id_int=-1,
            delegate_debug_metadata="",
            start_time=1001,
            end_time=2002,
        )
        debug_event = flatcc.DebugEvent(
            name="test_debug_event",
            chain_index=1,
            instruction_id=0,
            delegate_debug_id_str="56",
            delegate_debug_id_int=-1,
            debug_entry=flatcc.Value(
                val=flatcc.ValueType.TENSOR.value,
                tensor=flatcc.Tensor(
                    scalar_type=flatcc.ScalarType.INT,
                    sizes=[1],
                    strides=[1],
                    offset=12345,
                ),
                tensor_list=[
                    flatcc.TensorList(
                        tensors=[
                            flatcc.Tensor(
                                scalar_type=flatcc.ScalarType.INT,
                                sizes=[1],
                                strides=[1],
                                offset=12345,
                            )
                        ]
                    )
                ],
                int_value=flatcc.Int(1),
                float_value=flatcc.Float(1.0),
                double_value=flatcc.Double(1.0),
                bool_value=flatcc.Bool(False),
                output=flatcc.Bool(True),
            ),
        )

        # Profile Populated
        event = flatcc.Event(
            profile_event=profile_event, debug_event=None, allocation_event=None
        )
        self.assertEqual(find_populated_event(event), profile_event)

        # Debug Populated
        event = flatcc.Event(
            profile_event=None, debug_event=debug_event, allocation_event=None
        )
        self.assertEqual(find_populated_event(event), debug_event)

        # Neither Populated
        event = flatcc.Event(
            profile_event=None, debug_event=None, allocation_event=None
        )
        with self.assertRaises(ValueError):
            self.assertEqual(find_populated_event(event), profile_event)

        # Both Populated (Returns Profile Event)
        event = flatcc.Event(
            profile_event=profile_event, debug_event=debug_event, allocation_event=None
        )
        self.assertEqual(find_populated_event(event), profile_event)

    def test_is_inference_output_equal_returns_false_for_different_tensor_values(self):
        self.assertFalse(
            is_inference_output_equal(
                torch.tensor([[2, 1], [4, 3]]),
                torch.tensor([[5, 6], [7, 8]]),
            )
        )

    def test_is_inference_output_equal_returns_false_for_different_tensor_lists(self):
        tensor_list_1 = (
            [
                torch.tensor([[1, 2], [3, 4]]),
                torch.tensor([[1, 2], [3, 4]]),
                torch.tensor([[1, 2], [3, 4]]),
            ],
        )
        tensor_list_2 = [
            torch.tensor([[1, 2], [3, 4]]),
            torch.tensor([[1, 2], [3, 4]]),
        ]
        # Not equal because of different number of tensors
        self.assertFalse(is_inference_output_equal(tensor_list_1, tensor_list_2))

    def test_is_inference_output_equal_returns_true_for_same_tensor_values(self):
        self.assertTrue(
            is_inference_output_equal(
                torch.tensor([[2, 1], [4, 3]]),
                torch.tensor([[2, 1], [4, 3]]),
            )
        )

    def test_is_inference_output_equal_returns_true_for_same_strs(self):
        self.assertTrue(
            is_inference_output_equal(
                "value_string",
                "value_string",
            )
        )

    def test_calculate_time_scale_factor_second_based(self):
        self.assertEqual(
            calculate_time_scale_factor(TimeScale.NS, TimeScale.MS), 1000000
        )
        self.assertEqual(
            calculate_time_scale_factor(TimeScale.MS, TimeScale.NS), 1 / 1000000
        )

    def test_calculate_time_scale_factor_cycles(self):
        self.assertEqual(
            calculate_time_scale_factor(TimeScale.CYCLES, TimeScale.CYCLES), 1
        )

    def test_compare_results(self):
        a = torch.rand(4, 4)

        # Create tensor b which has very close value to tensor a
        b = a.clone()
        b[0, 0] += 1e-2
        b[1, 0] += 1e-2
        b[1, 3] -= 1e-2

        self.assertLess(calculate_mse([a], [b])[0], 0.5)
        self.assertGreater(calculate_snr([a], [b])[0], 30.0)
        self.assertAlmostEqual(calculate_cosine_similarity([a], [b])[0], 1.0)

    def test_compare_results_uint8(self):
        a = torch.randint(1, 255, (4, 4), dtype=torch.uint8)

        # Create tensor b which has very close value to tensor a
        b = a.clone()
        b[0, 0] += 1
        b[1, 0] += 1
        b[1, 3] -= 1

        self.assertLess(calculate_mse([a], [b])[0], 0.5)
        self.assertGreater(calculate_snr([a], [b])[0], 30.0)
        self.assertAlmostEqual(calculate_cosine_similarity([a], [b])[0], 1.0)

    def test_merge_overlapping_debug_handles_basic(self):
        big_tensor = torch.rand(100, 100)
        intermediate_outputs = {
            (1, 2, 3): (1, "val1", 1),
            (2, 3, 4, 5): (2, "val2", 1),
            (6, 7, 8): (3, "val3", 1),
            (10, 11): (4, "val4", 1),
            (11, 12): (5, big_tensor, 1),
        }
        # basic merge behavior
        intermediate_outputs = merge_runtime_overlapping_debug_handles(
            intermediate_outputs
        )
        expected_intermediate_outputs = {
            (1, 2, 3, 4, 5): (2, "val2", 1),
            (6, 7, 8): (3, "val3", 1),
            (10, 11, 12): (5, big_tensor, 1),
        }
        self.assertEqual(intermediate_outputs, expected_intermediate_outputs)
        self.assertIs(expected_intermediate_outputs[(10, 11, 12)][1], big_tensor)

    def test_merge_overlapping_debug_handles_non_continuous(self):
        tensor1 = torch.randn(3, 4)
        tensor2 = torch.randn(2, 3)
        tensor3 = torch.randn(4, 5)
        tensor4 = torch.randn(6, 7)
        tensor5 = torch.randn(8, 9)
        intermediate_outputs = {
            (1, 10): (1, tensor1, 1),
            (2, 5): (2, tensor2, 1),
            (1, 7, 9): (3, tensor3, 1),
            (11, 13): (4, tensor4, 1),
            (11, 15): (5, tensor5, 1),
        }
        intermediate_outputs = merge_runtime_overlapping_debug_handles(
            intermediate_outputs
        )
        expected_intermediate_outputs = {
            (2, 5): (2, tensor2),
            (10, 1, 7, 9): (3, tensor3),
            (13, 11, 15): (5, tensor5),
        }

        for key in expected_intermediate_outputs:
            expected_value = expected_intermediate_outputs[key][1]
            actual_value = intermediate_outputs[key][1]
            self.assertTrue(torch.allclose(expected_value, actual_value))

    def test_merge_overlapping_debug_handles_edge_cases(self):
        intermediate_outputs = {
            (9,): (1, "val1", 1),
            (
                9,
                9,
                9,
            ): (2, "val2", 1),
            (
                9,
                9,
            ): (3, "val3", 1),
        }
        intermediate_outputs = merge_runtime_overlapping_debug_handles(
            intermediate_outputs
        )
        expected_intermediate_outputs = {
            (9,): (3, "val3", 1),
        }
        self.assertEqual(intermediate_outputs, expected_intermediate_outputs)

    def test_map_runtime_aot_intermediate_outputs_empty_inputs(self):
        # When the inputs are empty, the output should also be empty
        aot_intermediate_outputs = {}
        runtime_intermediate_outputs = {}
        actual = map_runtime_aot_intermediate_outputs(
            aot_intermediate_outputs, runtime_intermediate_outputs
        )
        expected = {}
        self.assertEqual(actual, expected)

    def test_map_runtime_aot_intermediate_outputs_single_element_tuple(self):
        # Single element tuple
        aot_intermediate_outputs = {(0,): 100, (1,): 200, (2,): 300}
        runtime_intermediate_outputs = {(0,): (150, 1), (1,): (250, 1), (2,): (350, 1)}
        actual = map_runtime_aot_intermediate_outputs(
            aot_intermediate_outputs, runtime_intermediate_outputs
        )
        expected = {
            ((0,), 100): ((0,), 150),
            ((1,), 200): ((1,), 250),
            ((2,), 300): ((2,), 350),
        }
        self.assertEqual(actual, expected)

    def test_map_runtime_aot_intermediate_outputs_no_overlaps(self):
        # No overlaps between aot and runtime debug_handles
        aot_intermediate_outputs = {(0,): 100, (4,): 300}
        runtime_intermediate_outputs = {(2, 3): (200, 1), (8, 9): (300, 1)}
        actual = map_runtime_aot_intermediate_outputs(
            aot_intermediate_outputs, runtime_intermediate_outputs
        )
        expected = {}
        self.assertEqual(actual, expected)

    def test_map_runtime_aot_intermediate_outputs_partial_match(self):
        # Partial match between aot and runtime debug_handles will return empty
        aot_intermediate_outputs = {(2,): 100, (9,): 300}
        runtime_intermediate_outputs = {(2, 3): (200, 1), (8, 9): (300, 1)}
        actual = map_runtime_aot_intermediate_outputs(
            aot_intermediate_outputs, runtime_intermediate_outputs
        )
        expected = {}
        self.assertEqual(actual, expected)

    def test_map_runtime_aot_intermediate_outputs_multiple_aot_to_one_runtime(self):
        # Multiple aot debug_handles map to one runtime debug_handle
        aot_intermediate_outputs = {(0,): 100, (1,): 200, (2,): 300, (3,): 400}
        runtime_intermediate_outputs = {(2, 3, 1): (250, 1), (8, 9): (300, 1)}
        actual = map_runtime_aot_intermediate_outputs(
            aot_intermediate_outputs, runtime_intermediate_outputs
        )
        expected = {((2, 3, 1), 200): ((2, 3, 1), 250)}
        self.assertEqual(actual, expected)

    def test_map_runtime_aot_intermediate_outputs_delegated(self):
        # Currently, runtime_intermediate_output logs all delegate call arguments
        # Test that the map function correctly extracted out the delegated outputs
        aot_intermediate_outputs = {
            (1,): torch.tensor([1, 2, 3]),
            (2,): torch.tensor([4, 5]),
            (3,): torch.tensor([10, 10, 13]),
            (4,): torch.tensor([10, 11, 12]),
            (5,): torch.tensor([13, 14, 15, 16, 21]),
            (6,): torch.tensor([2]),
        }
        runtime_intermediate_outputs = {
            (1, 2): ([torch.tensor([1, 2, 3]), torch.tensor([4, 5])], 2),
            (3, 4): (
                [
                    torch.tensor([10, 10, 13]),
                    torch.tensor([10, 11, 12]),
                ],
                2,
            ),
            (5, 6): (
                [
                    torch.tensor([13, 14, 15, 16, 21]),
                    torch.tensor([2]),
                ],
                2,
            ),
        }
        actual = map_runtime_aot_intermediate_outputs(
            aot_intermediate_outputs, runtime_intermediate_outputs
        )
        expected = {
            ((1, 2), torch.tensor([1, 2, 3])): ((1, 2), torch.tensor([1, 2, 3])),
            ((1, 2), torch.tensor([4, 5])): ((1, 2), torch.tensor([4, 5])),
            ((3, 4), torch.tensor([10, 10, 13])): ((3, 4), torch.tensor([10, 10, 13])),
            ((3, 4), torch.tensor([10, 11, 12])): ((3, 4), torch.tensor([10, 11, 12])),
            ((5, 6), torch.tensor([13, 14, 15, 16, 21])): (
                (5, 6),
                torch.tensor([13, 14, 15, 16, 21]),
            ),
            ((5, 6), torch.tensor([2])): (
                (5, 6),
                torch.tensor([2]),
            ),
        }
        self.assertEqual(len(actual), len(expected))

        for (exp_aot_key, exp_aot_value), (
            exp_runtime_key,
            exp_runtime_value,
        ) in expected.items():
            found = False
            for (act_aot_key, act_aot_value), (
                act_runtime_key,
                act_runtime_value,
            ) in actual.items():
                if (
                    exp_aot_key == act_aot_key
                    and exp_aot_value.numel() == act_aot_value.numel()
                    and torch.allclose(exp_aot_value, act_aot_value)
                ):
                    found = True
                    self.assertEqual(exp_runtime_key, act_runtime_key)
                    self.assertTrue(
                        torch.allclose(exp_runtime_value, act_runtime_value)
                    )
                    break
            self.assertTrue(found)

    def test_convert_input_to_tensor_convertible_inputs(self):
        # Scalar -> tensor
        actual_output1 = convert_to_float_tensor(5)
        self.assertIsInstance(actual_output1, torch.Tensor)
        self.assertEqual(actual_output1.dtype, torch.float64)
        self.assertEqual(tuple(actual_output1.shape), ())
        self.assertTrue(
            torch.allclose(actual_output1, torch.tensor([5.0], dtype=torch.float64))
        )
        self.assertEqual(actual_output1.device.type, "cpu")

        # Tensor of ints -> float32 CPU
        t_int = torch.tensor([4, 5, 6], dtype=torch.int32)
        actual_output2 = convert_to_float_tensor(t_int)
        self.assertIsInstance(actual_output2, torch.Tensor)
        self.assertEqual(actual_output2.dtype, torch.float64)
        self.assertTrue(
            torch.allclose(
                actual_output2, torch.tensor([4.0, 5.0, 6.0], dtype=torch.float64)
            )
        )
        self.assertEqual(actual_output2.device.type, "cpu")

        # List of tensors -> AssertionError
        t_list = [torch.tensor([1, 2]), torch.tensor([2, 3]), torch.tensor([3, 4])]
        with self.assertRaises(AssertionError):
            convert_to_float_tensor(t_list)

    def test_convert_input_to_tensor_non_convertible_raises(self):
        class X:
            pass

        with self.assertRaises(ValueError) as cm:
            convert_to_float_tensor(X())
        msg = str(cm.exception)
        self.assertIn("Cannot convert value of type", msg)

    def test_get_aot_debug_handle_to_op_name_mapping_single_debug_handle(self):
        # Create a simple graph module with one node
        graph_module = torch.fx.GraphModule({}, torch.fx.Graph())
        node = graph_module.graph.create_node(
            "call_function", target=torch.mul, args=(), kwargs={}, name="op1"
        )
        node.meta["debug_handle"] = 1
        debug_handle_to_op_name = get_aot_debug_handle_to_op_name_mapping(graph_module)
        expected_result = {(1,): ["op1"]}
        self.assertEqual(debug_handle_to_op_name, expected_result)

    def test_get_aot_debug_handle_to_op_name_mapping_multiple_debug_handles(self):
        # Create a simple graph module with two nodes
        graph_module = torch.fx.GraphModule({}, torch.fx.Graph())
        node1 = graph_module.graph.create_node(
            "call_function", target=torch.mul, args=(), kwargs={}, name="op1"
        )
        node1.meta["debug_handle"] = (1, 2)
        node2 = graph_module.graph.create_node(
            "call_function", target=torch.mul, args=(), kwargs={}, name="op2"
        )
        node2.meta["debug_handle"] = 3
        debug_handle_to_op_name = get_aot_debug_handle_to_op_name_mapping(graph_module)
        expected_result = {
            (
                1,
                2,
            ): ["op1"],
            (3,): ["op2"],
        }
        self.assertEqual(debug_handle_to_op_name, expected_result)

    def test_get_aot_debug_handle_to_op_name_mapping_no_debug_handles(self):
        # Create a simple graph module with no nodes
        graph_module = torch.fx.GraphModule({}, torch.fx.Graph())
        debug_handle_to_op_name = get_aot_debug_handle_to_op_name_mapping(graph_module)
        expected_result = {}
        self.assertEqual(debug_handle_to_op_name, expected_result)

    def test_node_filter_match(self):
        node_filter = NodeFilter(
            "debug_handle", "call_function", exclude_ops=["getitem"]
        )

        # Create a mock node that matches the filter criteria
        mock_node = torch.fx.Node(
            graph=torch.fx.Graph(),
            name="mock_node",
            op="call_function",
            target=torch.nn.functional.relu,
            args=(),
            kwargs={},
        )
        mock_node.meta["debug_handle"] = (1, 2)
        # Test that the filter matches the mock node
        self.assertTrue(node_filter.matches(mock_node))

    def test_node_filter_key_mismatch(self):
        node_filter = NodeFilter(
            "debug_handle", "call_function", exclude_ops=["getitem"]
        )
        mock_node_metadata_key_mismatch = torch.fx.Node(
            graph=torch.fx.Graph(),
            name="mock_node_metadata_key_mismatch",
            op="call_function",
            target=torch.nn.functional.relu,
            args=(),
            kwargs={},
        )
        # Test that the filter doesn't match the mock node (meta doesn't have debug_handle key)
        self.assertFalse(node_filter.matches(mock_node_metadata_key_mismatch))

    def test_node_filter_ops_mismatch(self):
        node_filter = NodeFilter(
            "debug_handle", "call_function", exclude_ops=["getitem"]
        )

        mock_node_exclude_ops_mismatch = torch.fx.Node(
            graph=torch.fx.Graph(),
            name="getitem",
            op="call_function",
            target=torch.nn.functional.relu,
            args=(),
            kwargs={},
        )
        mock_node_exclude_ops_mismatch.meta["debug_handle"] = (1, 2)
        # Test that the filter doesn't match the mock node (exclude_ops mismatch)
        self.assertFalse(node_filter.matches(mock_node_exclude_ops_mismatch))

    def test_node_op_type_mismatch(self):
        node_filter = NodeFilter(
            "debug_handle", "call_function", exclude_ops=["getitem"]
        )

        mock_node_op_type_mismatch = torch.fx.Node(
            graph=torch.fx.Graph(),
            name="mock_node_op_type_mismatch",
            op="get_attr",
            target="torch.nn.functional.relu",
            args=(),
            kwargs={},
        )
        mock_node_op_type_mismatch.meta["debug_handle"] = (1, 2)
        # Test that the filter doesn't match the mock node (op_type mismatch)
        self.assertFalse(node_filter.matches(mock_node_op_type_mismatch))

    def test_find_op_names_empty_debug_handle(self):
        debug_handle = ()
        debug_handle_to_op_name = {(1, 2): ["op1"], (3, 4): ["op2"]}
        self.assertEqual(find_op_names(debug_handle, debug_handle_to_op_name), [])

    def test_find_op_names_no_matching_handles(self):
        debug_handle = (1, 2)
        debug_handle_to_op_name = {(3, 4): ["op1"], (5, 6): ["op2"]}
        self.assertEqual(find_op_names(debug_handle, debug_handle_to_op_name), [])

    def test_find_op_names_matching_handles(self):
        debug_handle = (1, 2, 3)
        debug_handle_to_op_name = {(1, 2): ["op1"], (2, 3): ["op2"], (4, 5, 6): ["op3"]}
        self.assertEqual(
            find_op_names(debug_handle, debug_handle_to_op_name), ["op1", "op2"]
        )

    def test_find_op_names_multiple_ops_single_handle(self):
        """Test when a single debug handle maps to multiple operator names"""
        debug_handle = (1, 2, 3)
        debug_handle_to_op_name = {(1, 2): ["op1", "op2", "op3"], (4, 5): ["op4"]}
        self.assertEqual(
            find_op_names(debug_handle, debug_handle_to_op_name), ["op1", "op2", "op3"]
        )

    def test_find_op_names_mixed_single_and_multiple_ops(self):
        """Test mix of handles with single and multiple operator names"""
        debug_handle = (1, 2, 3, 4, 5)
        debug_handle_to_op_name = {
            (1, 2): ["op1"],
            (3,): ["op2", "op3"],
            (4,): ["op4"],
            (5,): ["op5", "op6", "op7"],  # Multiple ops
        }
        self.assertEqual(
            find_op_names(debug_handle, debug_handle_to_op_name),
            ["op1", "op2", "op3", "op4", "op5", "op6", "op7"],
        )

    def test_compare_intermediate_outputs_sequences(self):
        a = [1.0, 2.0, 3.0]
        b = [1.0, 2.5, 3.5]
        result = compare_intermediate_outputs(a, b, L1Comparator())
        self.assertEqual(result, [0.0, 0.5, 0.5])

    def test_compare_intermediate_outputs_diff_len_sequences(self):
        a = [1.0, 2.0]
        b = [1.0, 2.0, 3.0]
        with self.assertRaises(ValueError):
            compare_intermediate_outputs(a, b, L1Comparator())

    def test_compare_intermediate_outputs_sequence_and_non_sequence(self):
        a = [1.0, 2.0]
        b = 1.0
        with self.assertRaises(ValueError):
            compare_intermediate_outputs(a, b, L1Comparator())

    def test_equip_debug_handle_to_export_program_success(self):
        """Test that propagate_back_debug_handle returns True and properly equips debug handles."""
        # Create a test model
        model = models.FeedForwardBlock(5, 10)
        inputs = (torch.rand(5, 5),)

        # Export the model
        exported_program = export(model, inputs)
        export_graph_id = id(exported_program.graph)

        # Convert to edge dialect
        edge_dialect_program = to_edge(exported_program).exported_program()

        # Call propagate_back_debug_handle
        result = propagate_back_debug_handle(
            exported_program, export_graph_id, edge_dialect_program
        )

        self.assertTrue(result)

        # Check that debug handles are properly equipped in the exported program
        exported_program_debug_handles = []
        for node in exported_program.graph.nodes:
            if node.op not in ("placeholder", "output"):
                self.assertIn(DEBUG_HANDLE_KEY, node.meta)
                self.assertIsNotNone(node.meta[DEBUG_HANDLE_KEY])
                exported_program_debug_handles.append(node.meta[DEBUG_HANDLE_KEY])

        edge_dialect_program_debug_handles = []
        for node in edge_dialect_program.graph.nodes:
            if node.op not in ("placeholder", "output"):
                self.assertIn(DEBUG_HANDLE_KEY, node.meta)
                self.assertIsNotNone(node.meta[DEBUG_HANDLE_KEY])
                edge_dialect_program_debug_handles.append(node.meta[DEBUG_HANDLE_KEY])

        # The 0th operator in the exported program (layer_norm) has been decomposed into 0th and 1st ops in edge dialect graph (native_layer_norm and getitem)
        # So they should have the same debug handle
        self.assertEqual(
            exported_program_debug_handles[0], edge_dialect_program_debug_handles[0]
        )
        self.assertEqual(
            exported_program_debug_handles[0], edge_dialect_program_debug_handles[1]
        )

    def test_equip_debug_handle_to_strict_export_program_success(self):
        """Test that propagate_back_debug_handle returns True and properly equips debug handles."""
        # Create a test model
        model = models.FeedForwardBlock(5, 10)
        inputs = (torch.rand(5, 5),)

        # Export the model
        exported_program = export(model, inputs, strict=True)
        export_graph_id = id(exported_program.graph)

        # Convert to edge dialect
        edge_dialect_program = to_edge(exported_program).exported_program()

        # Call propagate_back_debug_handle
        result = propagate_back_debug_handle(
            exported_program, export_graph_id, edge_dialect_program
        )

        self.assertTrue(result)

        # Check that debug handles are properly equipped in the exported program
        exported_program_debug_handles = []
        for node in exported_program.graph.nodes:
            if node.op not in ("placeholder", "output"):
                self.assertIn(DEBUG_HANDLE_KEY, node.meta)
                self.assertIsNotNone(node.meta[DEBUG_HANDLE_KEY])
                exported_program_debug_handles.append(node.meta[DEBUG_HANDLE_KEY])

        edge_dialect_program_debug_handles = []
        for node in edge_dialect_program.graph.nodes:
            if node.op not in ("placeholder", "output"):
                self.assertIn(DEBUG_HANDLE_KEY, node.meta)
                self.assertIsNotNone(node.meta[DEBUG_HANDLE_KEY])
                edge_dialect_program_debug_handles.append(node.meta[DEBUG_HANDLE_KEY])

        # The 0th operator in the exported program (layer_norm) has been decomposed into 0th and 1st ops in edge dialect graph (native_layer_norm and getitem)
        # So they should have the same debug handle
        self.assertEqual(
            exported_program_debug_handles[0], edge_dialect_program_debug_handles[0]
        )
        self.assertEqual(
            exported_program_debug_handles[0], edge_dialect_program_debug_handles[1]
        )

    def test_equip_debug_handle_to_reexport_program_success(self):
        """Test that propagate_back_debug_handle returns True and properly equips debug handles."""
        # Create a test model
        model = models.FeedForwardBlock(5, 10)
        inputs = (torch.rand(5, 5),)

        # Export the model
        init_export_program = export(model, inputs)
        exported_program = export(init_export_program.module(), inputs)
        export_graph_id = id(exported_program.graph)

        # Convert to edge dialect
        edge_dialect_program = to_edge(exported_program).exported_program()

        # Call propagate_back_debug_handle
        result = propagate_back_debug_handle(
            exported_program, export_graph_id, edge_dialect_program
        )

        self.assertTrue(result)

        # Check that debug handles are properly equipped in the exported program
        exported_program_debug_handles = []
        for node in exported_program.graph.nodes:
            if node.op not in ("placeholder", "output"):
                self.assertIn(DEBUG_HANDLE_KEY, node.meta)
                self.assertIsNotNone(node.meta[DEBUG_HANDLE_KEY])
                exported_program_debug_handles.append(node.meta[DEBUG_HANDLE_KEY])

        edge_dialect_program_debug_handles = []
        for node in edge_dialect_program.graph.nodes:
            if node.op not in ("placeholder", "output"):
                self.assertIn(DEBUG_HANDLE_KEY, node.meta)
                self.assertIsNotNone(node.meta[DEBUG_HANDLE_KEY])
                edge_dialect_program_debug_handles.append(node.meta[DEBUG_HANDLE_KEY])

        # The 0th operator in the exported program (layer_norm) has been decomposed into 0th and 1st ops in edge dialect graph (native_layer_norm and getitem)
        # So they should have the same debug handle
        self.assertEqual(
            exported_program_debug_handles[0], edge_dialect_program_debug_handles[0]
        )
        self.assertEqual(
            exported_program_debug_handles[0], edge_dialect_program_debug_handles[1]
        )

    def test_equip_debug_handle_to_export_program_failure(self):
        """Test that propagate_back_debug_handle returns False when there's a mismatch."""
        # Create a test model
        model = models.FeedForwardBlock(5, 10)
        inputs = (torch.rand(5, 5),)

        exported_program = export(model, inputs)
        edge_dialect_program = to_edge(exported_program).exported_program()

        # Create a different exported program (reexport) to cause mismatch
        reexported_program = export(model, inputs)
        reexport_graph_id = id(reexported_program.graph)

        # Call propagate_back_debug_handle with mismatched programs
        # This should return False because the reexported program has different node identifiers
        result = propagate_back_debug_handle(
            reexported_program, reexport_graph_id, edge_dialect_program
        )

        # Check that it returns False due to mismatch
        self.assertFalse(result)

    def test_equip_debug_handle_to_export_program_op_to_be_removed_in_to_edge(self):
        """Test that propagate_back_debug_handle returns True and properly equips debug handles when an op is removed in to_edge"""

        class M(torch.nn.Module):
            """
            Simple model with ops that will be removed in to_edge
            """

            def __init__(self) -> None:
                super().__init__()

            def forward(self, x: torch.Tensor) -> torch.Tensor:
                x = x + 1
                x = x.to(x.dtype)
                x = x + 1
                return x

        inputs = (torch.rand(5, 5),)
        exported_program = torch.export.export(M(), inputs)
        export_graph_id = id(exported_program.graph)
        edge_dialect_program = to_edge(exported_program).exported_program()

        self.assertTrue(
            propagate_back_debug_handle(
                exported_program, export_graph_id, edge_dialect_program
            )
        )

        n_removed_nodes = 0

        for node in exported_program.graph.nodes:
            if node.name == "add":
                self.assertEqual(node.meta[DEBUG_HANDLE_KEY], 1)
            elif node.name == "add_1":
                self.assertEqual(node.meta[DEBUG_HANDLE_KEY], 2)
            elif node.op not in ("placeholder", "output"):
                n_removed_nodes += 1
                self.assertEqual(node.meta[DEBUG_HANDLE_KEY], UNSET_DEBUG_HANDLE)

        self.assertEqual(n_removed_nodes, 2)


def gen_mock_operator_graph_with_expected_map() -> (
    Tuple[OperatorGraph, Dict[int, OperatorNode]]
):
    # Make a mock OperatorGraph instance for testing
    node_input = ValueNode("input")
    mapping = {}
    node_fused_conv_relu = OperatorNode(
        "fused_conv_relu",
        [node_input],
        None,
        metadata={
            "debug_handle": 111,
            "stack_trace": "stack_trace_relu",
            "nn_module_stack": "module_hierarchy_relu",
        },
    )
    mapping[111] = [
        node_fused_conv_relu,
    ]
    node_sin = OperatorNode(
        "sin",
        [node_fused_conv_relu],
        None,
        metadata={
            "debug_handle": 222,
            "stack_trace": "stack_trace_sin",
            "nn_module_stack": "module_hierarchy_sin",
        },
    )
    mapping[222] = [
        node_sin,
    ]
    node_cos = OperatorNode(
        "cos",
        [node_sin],
        None,
        metadata={
            "debug_handle": 333,
            "stack_trace": "stack_trace_cos",
            "nn_module_stack": "module_hierarchy_cos",
        },
    )
    mapping[333] = [
        node_cos,
    ]
    node_div = OperatorNode(
        "div",
        [node_cos],
        None,
        metadata={
            "debug_handle": 444,
            "stack_trace": "stack_trace_div",
            "nn_module_stack": "module_hierarchy_div",
        },
    )
    mapping[444] = [
        node_div,
    ]
    node_output = ValueNode("output", [node_div])
    return (
        OperatorGraph(
            "mock_et_model",
            [
                node_input,
                node_fused_conv_relu,
                node_sin,
                node_cos,
                node_div,
                node_output,
            ],
        ),
        mapping,
    )
