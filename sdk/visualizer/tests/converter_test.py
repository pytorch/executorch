#!/usr/bin/env fbpython
# (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

import unittest

from executorch.sdk.edir.et_schema import RESERVED_METADATA_ARG

from executorch.sdk.edir.tests.exported_op_graph_test import (
    generate_op_graph,
    TwoLinearModule,
)

from executorch.sdk.visualizer.converter import Converter

EXPECTED_RUN_METADATA_NODE_NAMES = {
    "forward/aten::unsqueeze_copy_7",
    "forward/aten::mm_8",
    "forward/aten::squeeze_copy_9",
    "forward/aten::add_10",
    "forward/aten::unsqueeze_copy_1",
    "forward/aten::add_4",
    "forward/aten::t_copy_6",
    "forward/aten::t_copy_0",
    "forward/aten::squeeze_copy_3",
    "forward/aten::mm_2",
    "forward/aten::relu_5",
}

EXPECTED_GRAPH_DEF_NODE_NAMES = {
    "inputs/in_0",
    "forward/const_0",
    "forward/const_1",
    "forward/const_2",
    "forward/const_3",
    "forward/const_4",
    "forward/const_5",
    "forward/const_6",
    "forward/const_7",
    "forward/const_8",
    "forward/const_9",
    "outputs/out_0",
} | EXPECTED_RUN_METADATA_NODE_NAMES


class ConverterTest(unittest.TestCase):
    def test_convert(self) -> None:
        model = TwoLinearModule()
        op_graph = generate_op_graph(model, model.get_random_inputs())
        op_graph.attach_metadata(model.gen_inference_run())
        converter = Converter(op_graph=op_graph)

        graph_def, run_metadata = converter.convert()
        run_metadata_node_stats = (
            # Arbitrarily select a tag to test with
            # pyre-fixme[16]: `RunMetadata` has no attribute `__getitem__`.
            run_metadata[RESERVED_METADATA_ARG.PROFILE_SUMMARY_AVERAGE.value]
            .step_stats.dev_stats[0]
            .node_stats
        )

        # Too tedious to check comprehensively, so only check node names where mistakes are more likely
        node_names_of_run_metadata = {
            stats.node_name for stats in run_metadata_node_stats
        }
        self.assertEquals(node_names_of_run_metadata, EXPECTED_RUN_METADATA_NODE_NAMES)
        node_names_of_graph_def = {node.name for node in graph_def.node}
        self.assertEquals(node_names_of_graph_def, EXPECTED_GRAPH_DEF_NODE_NAMES)
