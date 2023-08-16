# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import copy
import json
import tempfile
import unittest

import executorch.exir.tests.models as models
from executorch import exir
from executorch.sdk.etrecord import generate_etrecord, parse_etrecord
from executorch.sdk.etrecord._etrecord import ETRecordReservedFileNames


# TODO : T154728484  Add test cases to cover multiple entry points
class TestETRecord(unittest.TestCase):
    def get_test_model(self):
        f = models.BasicSinMax()
        captured_output = exir.capture(f, f.get_random_inputs(), exir.CaptureConfig())
        captured_output_copy = copy.deepcopy(captured_output)
        edge_output = captured_output.to_edge(
            # TODO(gasoon): Remove _use_edge_ops=False once serde is fully migrated to Edge ops
            exir.EdgeCompileConfig(_check_ir_validity=False, _use_edge_ops=False)
        )
        edge_output_copy = copy.deepcopy(edge_output)
        et_output = edge_output.to_executorch()
        buffer = et_output.buffer
        return (captured_output_copy, edge_output_copy, et_output, buffer)

    # Serialized and deserialized graph modules are not completely the same, so we check
    # that they are close enough and match especially on the parameters we care about in the SDK.
    def check_graph_closeness(self, graph_a, graph_b):
        self.assertEqual(len(graph_a.graph.nodes), len(graph_b.graph.nodes))
        for node_a, node_b in zip(graph_a.graph.nodes, graph_b.graph.nodes):
            self.assertEqual(node_a.target, node_b.target)
            self.assertEqual(len(node_a.args), len(node_b.args))
            self.assertEqual(len(node_a.kwargs), len(node_b.kwargs))
            self.assertEqual(node_a.name, node_b.name)
            self.assertEqual(node_a.type, node_b.type)
            self.assertEqual(node_a.op, node_b.op)
            if node_a.op not in {"placeholder", "output"}:
                self.assertEqual(
                    node_a.meta.get("debug_handle"), node_b.meta.get("debug_handle")
                )

    def test_etrecord_generation(self):
        captured_output, edge_output, et_output, program_buffer = self.get_test_model()
        with tempfile.TemporaryDirectory() as tmpdirname:
            generate_etrecord(
                tmpdirname + "/etrecord.bin",
                et_output,
                {
                    "aten_dialect_output": captured_output,
                    "edge_dialect_output": edge_output,
                },
            )

            etrecord = parse_etrecord(tmpdirname + "/etrecord.bin")
            self.check_graph_closeness(
                etrecord.graph_map["aten_dialect_output/forward"],
                captured_output.exported_program.graph_module,
            )
            self.check_graph_closeness(
                etrecord.graph_map["edge_dialect_output/forward"],
                edge_output.exported_program.graph_module,
            )
            self.check_graph_closeness(
                etrecord.graph_map["et_dialect_graph_module/forward"],
                et_output.dump_exported_program(),
            )
            self.assertEqual(
                etrecord._debug_handle_map,
                json.loads(json.dumps(et_output.debug_handle_map)),
            )
            self.assertEqual(etrecord.program_buffer, program_buffer)

    def test_etrecord_invalid_input(self):
        captured_output, edge_output, et_output, program_buffer = self.get_test_model()
        with tempfile.TemporaryDirectory() as tmpdirname:
            with self.assertRaises(RuntimeError):
                generate_etrecord(
                    tmpdirname + "/etrecord.bin", {"fail_test_case": et_output}
                )

    def test_etrecord_reserved_name(self):
        captured_output, edge_output, et_output, program_buffer = self.get_test_model()
        with tempfile.TemporaryDirectory() as tmpdirname:
            for reserved_name in ETRecordReservedFileNames:
                with self.assertRaises(RuntimeError):
                    generate_etrecord(
                        tmpdirname + "/etrecord.bin",
                        {reserved_name: captured_output.exported_program.graph_module},
                    )
