# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-unsafe

import tempfile
import unittest
from typing import Dict, Tuple

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
    convert_to_float_tensor,
    create_debug_handle_to_op_node_mapping,
    EDGE_DIALECT_GRAPH_KEY,
    find_op_names,
    find_populated_event,
    gen_graphs_from_etrecord,
    get_aot_debug_handle_to_op_name_mapping,
    is_inference_output_equal,
    map_runtime_aot_intermediate_outputs,
    merge_overlapping_debug_handles,
    NodeFilter,
    TimeScale,
)


class TestInspectorUtils(unittest.TestCase):
    def test_gen_graphs_from_etrecord(self):
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
            (1, 2, 3): "val1",
            (2, 3, 4, 5): "val2",
            (6, 7, 8): "val3",
            (10, 11): "val4",
            (11, 12): big_tensor,
        }
        # basic merge behavior
        intermediate_outputs = merge_overlapping_debug_handles(intermediate_outputs)
        expected_intermediate_outputs = {
            (1, 2, 3, 4, 5): "val2",
            (6, 7, 8): "val3",
            (10, 11, 12): big_tensor,
        }

        self.assertEqual(intermediate_outputs, expected_intermediate_outputs)
        self.assertIs(expected_intermediate_outputs[(10, 11, 12)], big_tensor)

    def test_merge_overlapping_debug_handles_non_continuous(self):
        tensor1 = (torch.randn(3, 4),)
        tensor2 = (torch.randn(2, 3),)
        tensor3 = (torch.randn(4, 5),)
        tensor4 = (torch.randn(6, 7),)
        tensor5 = (torch.randn(8, 9),)
        intermediate_outputs = {
            (1, 10): tensor1,
            (2, 5): tensor2,
            (1, 7, 9): tensor3,
            (11, 13): tensor4,
            (11, 15): tensor5,
        }
        intermediate_outputs = merge_overlapping_debug_handles(intermediate_outputs)
        expected_intermediate_outputs = {
            (2, 5): tensor2,
            (1, 7, 9, 10): tensor1,
            (11, 13, 15): tensor5,
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
        runtime_intermediate_outputs = {(0,): 150, (1,): 250, (2,): 350}
        actual = map_runtime_aot_intermediate_outputs(
            aot_intermediate_outputs, runtime_intermediate_outputs
        )
        expected = {
            ((0,), 100): ((0,), 150),
            ((1,), 200): ((1,), 250),
            ((2,), 300): ((2,), 350),
        }
        self.assertEqual(actual, expected)

    def test_map_runtime_aot_intermediate_outputs_exact_match(self):
        # Exact match between aot and runtime debug_handles
        aot_intermediate_outputs = {(0, 1): 100, (2, 3): 200, (4, 5): 300}
        runtime_intermediate_outputs = {(0, 1): 150, (2, 3): 200, (4, 5): 300}
        actual = map_runtime_aot_intermediate_outputs(
            aot_intermediate_outputs, runtime_intermediate_outputs
        )
        expected = {
            ((0, 1), 100): ((0, 1), 150),
            ((2, 3), 200): ((2, 3), 200),
            ((4, 5), 300): ((4, 5), 300),
        }
        self.assertEqual(actual, expected)

    def test_map_runtime_aot_intermediate_outputs_no_overlaps(self):
        # No overlaps between aot and runtime debug_handles
        aot_intermediate_outputs = {(0, 1): 100, (4, 5): 300}
        runtime_intermediate_outputs = {(2, 3): 200, (8, 9): 300}
        actual = map_runtime_aot_intermediate_outputs(
            aot_intermediate_outputs, runtime_intermediate_outputs
        )
        expected = {}
        self.assertEqual(actual, expected)

    def test_map_runtime_aot_intermediate_outputs_multiple_aot_to_one_runtime(self):
        # Multiple aot debug_handles map to one runtime debug_handle
        aot_intermediate_outputs = {(0, 1, 2): 100, (3, 4): 300}
        runtime_intermediate_outputs = {(1, 2, 3): 250, (8, 9): 300}
        actual = map_runtime_aot_intermediate_outputs(
            aot_intermediate_outputs, runtime_intermediate_outputs
        )
        expected = {((0, 1, 2, 3, 4), 300): ((1, 2, 3), 250)}
        self.assertEqual(actual, expected)

    def test_map_runtime_aot_intermediate_outputs_one_aot_to_multiple_runtime(self):
        # One aot debug_handle map to multiple runtime debug_handles
        aot_intermediate_outputs = {(0, 1, 2, 3, 4): 100, (8, 9): 300}
        runtime_intermediate_outputs = {(0, 1): 150, (2, 3): 200, (4, 5): 300}
        actual = map_runtime_aot_intermediate_outputs(
            aot_intermediate_outputs, runtime_intermediate_outputs
        )
        expected = {((0, 1, 2, 3, 4), 100): ((0, 1, 2, 3, 4, 5), 300)}
        self.assertEqual(actual, expected)

    def test_map_runtime_aot_intermediate_outputs_complex_chain(self):
        # Complex chain (N-to-N mapping)
        aot_intermediate_outputs = {(1, 2): 100, (3, 4): 200, (5, 6): 300}
        runtime_intermediate_outputs = {(2, 3): 150, (4, 5): 250, (6, 7): 350}
        actual = map_runtime_aot_intermediate_outputs(
            aot_intermediate_outputs, runtime_intermediate_outputs
        )
        expected = {((1, 2, 3, 4, 5, 6), 300): ((2, 3, 4, 5, 6, 7), 350)}
        self.assertEqual(actual, expected)

    def test_map_runtime_aot_intermediate_outputs_delegated(self):
        # Currently, runtime_intermediate_output logs all delegate call arguments
        # Test that the map function correctly extracted out the delegated outputs
        aot_intermediate_outputs = {
            (1, 2): torch.tensor([4, 5]),
            (3, 4): torch.tensor([10, 11, 12]),
            (5, 6): torch.tensor([13, 14, 15, 16, 17]),
        }
        runtime_intermediate_outputs = {
            (1, 2): [torch.tensor([1, 2, 3]), torch.tensor([4, 5])],
            (3, 4): [
                torch.tensor([6, 7, 8, 9]),
                torch.tensor(1),
                torch.tensor([10, 11, 12]),
            ],
            (5, 6): [
                torch.tensor([1]),
                torch.tensor([2]),
                torch.tensor([13, 14, 15, 16, 17]),
            ],
        }
        actual = map_runtime_aot_intermediate_outputs(
            aot_intermediate_outputs, runtime_intermediate_outputs
        )
        expected = {
            ((1, 2), torch.tensor([4, 5])): ((1, 2), torch.tensor([4, 5])),
            ((3, 4), torch.tensor([10, 11, 12])): ((3, 4), torch.tensor([10, 11, 12])),
            ((5, 6), torch.tensor([13, 14, 15, 16, 17])): (
                (5, 6),
                torch.tensor([13, 14, 15, 16, 17]),
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
                if exp_aot_key == act_aot_key and torch.allclose(
                    exp_aot_value, act_aot_value
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

        # List of tensors -> stacked tensor float32 CPU
        t_list = [torch.tensor([1, 2]), torch.tensor([2, 3]), torch.tensor([3, 4])]
        actual_output3 = convert_to_float_tensor(t_list)
        self.assertIsInstance(actual_output3, torch.Tensor)
        self.assertEqual(actual_output3.dtype, torch.float64)
        self.assertEqual(tuple(actual_output3.shape), (3, 2))
        self.assertTrue(
            torch.allclose(
                actual_output3,
                torch.tensor([[1.0, 2.0], [2.0, 3.0], [3.0, 4.0]], dtype=torch.float64),
            )
        )
        self.assertEqual(actual_output3.device.type, "cpu")

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
        expected_result = {(1,): "op1"}
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
            ): "op1",
            (3,): "op2",
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
        debug_handle_to_op_name = {(1, 2): "op1", (3, 4): "op2"}
        self.assertEqual(find_op_names(debug_handle, debug_handle_to_op_name), [])

    def test_find_op_names_no_matching_handles(self):
        debug_handle = (1, 2)
        debug_handle_to_op_name = {(3, 4): "op1", (5, 6): "op2"}
        self.assertEqual(find_op_names(debug_handle, debug_handle_to_op_name), [])

    def test_find_op_names_matching_handles(self):
        debug_handle = (1, 2, 3)
        debug_handle_to_op_name = {(1, 2): "op1", (2, 3): "op2", (4, 5, 6): "op3"}
        self.assertEqual(
            find_op_names(debug_handle, debug_handle_to_op_name), ["op1", "op2"]
        )


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
    mapping[111] = node_fused_conv_relu
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
    mapping[222] = node_sin
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
    mapping[333] = node_cos
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
    mapping[444] = node_div
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
