# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-unsafe

import copy
import json
import tempfile
import unittest

import executorch.exir.tests.models as models
import torch
from executorch import exir
from executorch.devtools.bundled_program.config import MethodTestCase, MethodTestSuite
from executorch.devtools.bundled_program.core import BundledProgram
from executorch.devtools.etrecord import generate_etrecord, parse_etrecord
from executorch.devtools.etrecord._etrecord import (
    _get_reference_outputs,
    _get_representative_inputs,
    ETRecord,
    ETRecordReservedFileNames,
)
from executorch.exir import EdgeCompileConfig, EdgeProgramManager, to_edge
from torch.export import export


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
        return (captured_output_copy, edge_output_copy, et_output)

    def get_test_model_with_bundled_program(self):
        f = models.BasicSinMax()
        inputs = [f.get_random_inputs() for _ in range(2)]
        m_name = "forward"

        method_test_suites = [
            MethodTestSuite(
                method_name=m_name,
                test_cases=[
                    MethodTestCase(
                        inputs=inp, expected_outputs=getattr(f, m_name)(*inp)
                    )
                    for inp in inputs
                ],
            )
        ]
        captured_output = exir.capture(f, inputs[0], exir.CaptureConfig())
        captured_output_copy = copy.deepcopy(captured_output)
        edge_output = captured_output.to_edge(
            # TODO(gasoon): Remove _use_edge_ops=False once serde is fully migrated to Edge ops
            exir.EdgeCompileConfig(_check_ir_validity=False, _use_edge_ops=False)
        )
        edge_output_copy = copy.deepcopy(edge_output)
        et_output = edge_output.to_executorch()

        bundled_program = BundledProgram(et_output, method_test_suites)
        return (captured_output_copy, edge_output_copy, bundled_program)

    def get_test_model_with_manager(self):
        f = models.BasicSinMax()
        aten_dialect = export(f, f.get_random_inputs(), strict=True)
        edge_program: EdgeProgramManager = to_edge(
            aten_dialect, compile_config=EdgeCompileConfig(_check_ir_validity=False)
        )
        edge_program_copy = copy.deepcopy(edge_program)
        return (aten_dialect, edge_program_copy, edge_program.to_executorch())

    # Serialized and deserialized graph modules are not completely the same, so we check
    # that they are close enough and match especially on the parameters we care about in the Developer Tools.
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
                from_node_a = node_a.meta.get("from_node")
                from_node_b = node_b.meta.get("from_node")

                if from_node_a is None:
                    self.assertIsNone(from_node_b)
                else:
                    self.assertIsNotNone(from_node_b)
                    for node_source_a, node_source_b in zip(from_node_a, from_node_b):
                        self.assertEqual(
                            node_source_a.to_dict(), node_source_b.to_dict()
                        )

    def test_etrecord_generation(self):
        captured_output, edge_output, et_output = self.get_test_model()
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

            self.check_graph_closeness(
                etrecord.graph_map["aten_dialect_output/forward"],
                captured_output.exported_program.graph_module,
            )
            self.check_graph_closeness(
                etrecord.edge_dialect_program,
                edge_output.exported_program.graph_module,
            )
            self.assertEqual(
                etrecord._debug_handle_map,
                json.loads(json.dumps(et_output.debug_handle_map)),
            )

    def test_etrecord_generation_with_bundled_program(self):
        (
            captured_output,
            edge_output,
            bundled_program,
        ) = self.get_test_model_with_bundled_program()
        with tempfile.TemporaryDirectory() as tmpdirname:
            generate_etrecord(
                tmpdirname + "/etrecord.bin",
                edge_output,
                bundled_program,
                {
                    "aten_dialect_output": captured_output,
                },
            )
            etrecord = parse_etrecord(tmpdirname + "/etrecord.bin")

            expected_inputs = etrecord._representative_inputs
            actual_inputs = _get_representative_inputs(bundled_program)
            # assertEqual() gives "RuntimeError: Boolean value of Tensor with more than one value is ambiguous" when comparing tensors,
            # so we use torch.equal() to compare the tensors one by one.
            for expected, actual in zip(expected_inputs, actual_inputs):
                self.assertTrue(torch.equal(expected[0], actual[0]))
                self.assertTrue(torch.equal(expected[1], actual[1]))

            expected_outputs = etrecord._reference_outputs
            actual_outputs = _get_reference_outputs(bundled_program)
            self.assertTrue(
                torch.equal(
                    expected_outputs["forward"][0][0], actual_outputs["forward"][0][0]
                )
            )
            self.assertTrue(
                torch.equal(
                    expected_outputs["forward"][1][0], actual_outputs["forward"][1][0]
                )
            )

    def test_etrecord_generation_with_manager(self):
        captured_output, edge_output, et_output = self.get_test_model_with_manager()
        with tempfile.TemporaryDirectory() as tmpdirname:
            generate_etrecord(
                tmpdirname + "/etrecord.bin",
                edge_output,
                et_output,
            )

            etrecord = parse_etrecord(tmpdirname + "/etrecord.bin")
            self.check_graph_closeness(
                etrecord.edge_dialect_program,
                edge_output.exported_program().graph_module,
            )
            self.assertEqual(
                etrecord._debug_handle_map,
                json.loads(json.dumps(et_output.debug_handle_map)),
            )

    def test_etrecord_invalid_input(self):
        captured_output, edge_output, et_output = self.get_test_model()
        with tempfile.TemporaryDirectory() as tmpdirname:
            with self.assertRaises(RuntimeError):
                generate_etrecord(
                    tmpdirname + "/etrecord.bin",
                    edge_output,
                    et_output,
                    extra_recorded_export_modules={"fail_test_case": et_output},
                )

    def test_etrecord_reserved_name(self):
        captured_output, edge_output, et_output = self.get_test_model()
        with tempfile.TemporaryDirectory() as tmpdirname:
            for reserved_name in ETRecordReservedFileNames:
                with self.assertRaises(RuntimeError):
                    generate_etrecord(
                        tmpdirname + "/etrecord.bin",
                        edge_output,
                        et_output,
                        extra_recorded_export_modules={
                            reserved_name: captured_output.exported_program.graph_module
                        },
                    )

    def test_etrecord_generation_with_exported_program(self):
        """Test that exported program can be recorded and parsed back correctly."""
        captured_output, edge_output, et_output = self.get_test_model()
        original_exported_program = captured_output.exported_program
        expected_graph_id = id(original_exported_program.graph)

        with tempfile.TemporaryDirectory() as tmpdirname:
            # Generate ETRecord with exported program
            generate_etrecord(
                tmpdirname + "/etrecord.bin",
                edge_output,
                et_output,
                exported_program=original_exported_program,
            )

            # Parse ETRecord back
            etrecord = parse_etrecord(tmpdirname + "/etrecord.bin")

            # Validate that the parsed exported program matches the original
            self.assertIsNotNone(etrecord.exported_program)
            self.check_graph_closeness(
                etrecord.exported_program,
                original_exported_program.graph_module,
            )

            # Validate other components are still present
            self.check_graph_closeness(
                etrecord.edge_dialect_program,
                edge_output.exported_program.graph_module,
            )
            self.assertEqual(
                etrecord._debug_handle_map,
                json.loads(json.dumps(et_output.debug_handle_map)),
            )

            # Validate that export_graph_id matches the expected value
            self.assertEqual(etrecord.export_graph_id, expected_graph_id)

    def test_etrecord_class_constructor_and_save(self):
        """Test that ETRecord class constructor and save method work correctly."""
        captured_output, edge_output, et_output = self.get_test_model()
        original_exported_program = captured_output.exported_program
        expected_graph_id = id(original_exported_program.graph)

        # Create ETRecord instance directly using constructor
        etrecord = ETRecord(
            exported_program=original_exported_program,
            export_graph_id=expected_graph_id,
            edge_dialect_program=edge_output.exported_program,
            graph_map={"test_module/forward": original_exported_program},
            _debug_handle_map=et_output.debug_handle_map,
            _delegate_map=et_output.delegate_map,
        )

        with tempfile.TemporaryDirectory() as tmpdirname:
            etrecord_path = tmpdirname + "/etrecord_direct.bin"

            # Use the save method
            etrecord.save(etrecord_path)

            # Parse ETRecord back and verify
            parsed_etrecord = parse_etrecord(etrecord_path)

            # Validate that all components are preserved
            self.assertIsNotNone(parsed_etrecord.exported_program)
            self.check_graph_closeness(
                parsed_etrecord.exported_program,
                original_exported_program.graph_module,
            )

            self.assertIsNotNone(parsed_etrecord.edge_dialect_program)
            self.check_graph_closeness(
                parsed_etrecord.edge_dialect_program,
                edge_output.exported_program.graph_module,
            )

            # Validate graph map
            self.assertIsNotNone(parsed_etrecord.graph_map)
            self.assertIn("test_module/forward", parsed_etrecord.graph_map)
            self.check_graph_closeness(
                parsed_etrecord.graph_map["test_module/forward"],
                original_exported_program.graph_module,
            )

            # Validate debug and delegate maps
            self.assertEqual(
                parsed_etrecord._debug_handle_map,
                json.loads(json.dumps(et_output.debug_handle_map)),
            )
            self.assertEqual(
                parsed_etrecord._delegate_map,
                json.loads(json.dumps(et_output.delegate_map)),
            )

            # Validate export graph id
            self.assertEqual(parsed_etrecord.export_graph_id, expected_graph_id)

    def test_etrecord_class_with_bundled_program_data(self):
        """Test ETRecord class with bundled program data."""
        (
            captured_output,
            edge_output,
            bundled_program,
        ) = self.get_test_model_with_bundled_program()

        # Extract bundled program data
        reference_outputs = _get_reference_outputs(bundled_program)
        representative_inputs = _get_representative_inputs(bundled_program)

        # Create ETRecord instance with bundled program data
        etrecord = ETRecord(
            exported_program=captured_output.exported_program,
            export_graph_id=id(captured_output.exported_program.graph),
            edge_dialect_program=edge_output.exported_program,
            _debug_handle_map=bundled_program.executorch_program.debug_handle_map,
            _delegate_map=bundled_program.executorch_program.delegate_map,
            _reference_outputs=reference_outputs,
            _representative_inputs=representative_inputs,
        )

        with tempfile.TemporaryDirectory() as tmpdirname:
            etrecord_path = tmpdirname + "/etrecord_bundled.bin"

            # Save using the save method
            etrecord.save(etrecord_path)

            # Parse and verify
            parsed_etrecord = parse_etrecord(etrecord_path)

            # Validate bundled program specific data
            self.assertIsNotNone(parsed_etrecord._reference_outputs)
            self.assertIsNotNone(parsed_etrecord._representative_inputs)

            # Compare reference outputs
            expected_outputs = parsed_etrecord._reference_outputs
            self.assertTrue(
                torch.equal(
                    expected_outputs["forward"][0][0], reference_outputs["forward"][0][0]
                )
            )
            self.assertTrue(
                torch.equal(
                    expected_outputs["forward"][1][0], reference_outputs["forward"][1][0]
                )
            )

            # Compare representative inputs
            expected_inputs = parsed_etrecord._representative_inputs
            for expected, actual in zip(expected_inputs, representative_inputs):
                self.assertTrue(torch.equal(expected[0], actual[0]))
                self.assertTrue(torch.equal(expected[1], actual[1]))

    def test_etrecord_generation_with_exported_program_dict(self):
        """Test that exported program dictionary can be recorded and parsed back correctly."""
        captured_output, edge_output, et_output = self.get_test_model()
        original_exported_program = captured_output.exported_program
        exported_program_dict = {"forward": original_exported_program}
        expected_graph_id = id(original_exported_program.graph)

        with tempfile.TemporaryDirectory() as tmpdirname:
            # Generate ETRecord with exported program dictionary
            generate_etrecord(
                tmpdirname + "/etrecord.bin",
                edge_output,
                et_output,
                exported_program=exported_program_dict,
            )

            # Parse ETRecord back
            etrecord = parse_etrecord(tmpdirname + "/etrecord.bin")

            # Validate that the parsed exported program matches the original
            self.assertIsNotNone(etrecord.exported_program)
            self.check_graph_closeness(
                etrecord.exported_program,
                original_exported_program.graph_module,
            )

            # Validate other components are still present
            self.check_graph_closeness(
                etrecord.edge_dialect_program,
                edge_output.exported_program.graph_module,
            )
            self.assertEqual(
                etrecord._debug_handle_map,
                json.loads(json.dumps(et_output.debug_handle_map)),
            )

            # Validate that export_graph_id matches the expected value
            self.assertEqual(etrecord.export_graph_id, expected_graph_id)
