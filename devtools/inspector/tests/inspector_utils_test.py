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
    calculate_time_scale_factor,
    create_debug_handle_to_op_node_mapping,
    EDGE_DIALECT_GRAPH_KEY,
    find_populated_event,
    gen_graphs_from_etrecord,
    is_inference_output_equal,
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
