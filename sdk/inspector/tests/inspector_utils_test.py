# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import tempfile
import unittest
from typing import Dict, Tuple

from executorch.sdk import generate_etrecord, parse_etrecord

from executorch.sdk.debug_format.base_schema import (
    OperatorGraph,
    OperatorNode,
    ValueNode,
)

from executorch.sdk.debug_format.et_schema import FXOperatorGraph
from executorch.sdk.etdump import schema_flatcc as flatcc

from executorch.sdk.etrecord.tests.etrecord_test import TestETRecord
from executorch.sdk.inspector._inspector_utils import (
    create_debug_handle_to_op_node_mapping,
    EDGE_DIALECT_GRAPH_KEY,
    find_populated_event,
    gen_graphs_from_etrecord,
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
            chain_index=1,
            instruction_id=0,
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


def gen_mock_operator_graph_with_expected_map() -> Tuple[
    OperatorGraph, Dict[int, OperatorNode]
]:
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
