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
from typing import List

import executorch.exir.tests.models as models
import torch
from executorch import exir
from executorch.backends.xnnpack.partition.xnnpack_partitioner import XnnpackPartitioner
from executorch.devtools.bundled_program.config import MethodTestCase, MethodTestSuite
from executorch.devtools.bundled_program.core import BundledProgram
from executorch.devtools.etrecord import generate_etrecord, parse_etrecord
from executorch.devtools.etrecord._etrecord import (
    _get_reference_outputs,
    _get_representative_inputs,
    ETRecord,
    ETRecordReservedFileNames,
)
from executorch.exir import EdgeCompileConfig, EdgeProgramManager
from executorch.exir.program._program import to_edge, to_edge_transform_and_lower

from executorch.export import export as etexport, ExportRecipe, StageType
from torch.export import export


# TODO : T154728484  Add test cases to cover multiple entry points
class TestETRecord(unittest.TestCase):
    def assert_representative_inputs_equal(
        self,
        expected_inputs: List,
        actual_inputs: List,
        msg: str = "Representative inputs do not match",
    ) -> None:
        """
        Utility function to compare representative inputs.

        This function handles the comparison of representative inputs, which are lists of tuples
        containing tensors. It compares each input tuple element by element using torch.equal().

        Args:
            expected_inputs: List of expected input tuples
            actual_inputs: List of actual input tuples
            msg: Optional message to display on assertion failure
        """
        self.assertEqual(
            len(expected_inputs),
            len(actual_inputs),
            f"{msg}: Different number of input sets",
        )

        for i, (expected, actual) in enumerate(zip(expected_inputs, actual_inputs)):
            self.assertEqual(
                len(expected),
                len(actual),
                f"{msg}: Input set {i} has different number of tensors",
            )

            for j, (exp_tensor, act_tensor) in enumerate(zip(expected, actual)):
                self.assertTrue(
                    torch.equal(exp_tensor, act_tensor),
                    f"{msg}: Tensor {j} in input set {i} does not match",
                )

    def assert_etrecord_has_no_exported_program(self, etrecord: ETRecord) -> None:
        """Assert that ETRecord has no exported program data."""
        self.assertIsNone(etrecord.exported_program)
        self.assertIsNone(etrecord.export_graph_id)

    def assert_etrecord_has_no_edge_dialect_program(self, etrecord: ETRecord) -> None:
        """Assert that ETRecord has no edge dialect program data."""
        self.assertIsNone(etrecord.edge_dialect_program)

    def assert_etrecord_has_no_executorch_program(self, etrecord: ETRecord) -> None:
        """Assert that ETRecord has no executorch program data."""
        self.assertIsNone(etrecord._debug_handle_map)
        self.assertIsNone(etrecord._delegate_map)
        self.assertIsNone(etrecord._reference_outputs)
        self.assertIsNone(etrecord._representative_inputs)

    def assert_etrecord_is_empty(self, etrecord: ETRecord) -> None:
        """Assert that ETRecord has no data at all."""
        self.assert_etrecord_has_no_exported_program(etrecord)
        self.assert_etrecord_has_no_edge_dialect_program(etrecord)
        self.assert_etrecord_has_no_executorch_program(etrecord)
        self.assertIsNone(etrecord.graph_map)

    def assert_legal_etrecord_in_edge_program(self, etrecord: ETRecord) -> None:
        """Assert that ETRecord has all expected data after to_edge_transform_and_lower() or to_edge() stage"""
        self.assertIsNotNone(etrecord.exported_program)
        self.assertIsNotNone(etrecord.export_graph_id)
        self.assertIsNotNone(etrecord.edge_dialect_program)
        self.assert_etrecord_has_no_executorch_program(etrecord)

    def assert_etrecord_saveable(self, etrecord: ETRecord) -> None:
        """Assert ETRecord contains all essential information for saving"""
        self.assertIsNotNone(etrecord.exported_program)
        self.assertIsNotNone(etrecord.export_graph_id)
        self.assertIsNotNone(etrecord.edge_dialect_program)
        self.assertIsNotNone(etrecord._debug_handle_map)
        self.assertIsNotNone(etrecord._delegate_map)

    def get_test_model(self, generate_etrecord=False):
        f = models.BasicSinMax()
        aten_dialect = export(f, f.get_random_inputs(), strict=True)
        edge_program: EdgeProgramManager = to_edge(
            aten_dialect,
            compile_config=EdgeCompileConfig(_check_ir_validity=False),
            generate_etrecord=generate_etrecord,
        )
        edge_program_copy = copy.deepcopy(edge_program)
        return (aten_dialect, edge_program_copy, edge_program.to_executorch())

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
        aten_dialect, edge_program_copy, et_output = self.get_test_model()
        bundled_program = BundledProgram(et_output, method_test_suites)
        return (aten_dialect, edge_program_copy, bundled_program)

    def get_test_export_session(self, generate_etrecord=False, to_edge_flow=False):
        f = models.BasicSinMax()
        example_inputs = [f.get_random_inputs()]
        export_recipe = None

        if to_edge_flow:
            export_recipe = ExportRecipe(
                pipeline_stages=[
                    StageType.TORCH_EXPORT,
                    StageType.TO_EDGE,
                    StageType.TO_BACKEND,
                    StageType.TO_EXECUTORCH,
                ]
            )
        else:
            export_recipe = ExportRecipe()

        # Test with generate_etrecord=True
        export_session = etexport(
            model=f,
            example_inputs=example_inputs,
            export_recipe=export_recipe,
            generate_etrecord=generate_etrecord,
        )

        return export_session

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
                captured_output.graph_module,
            )
            self.check_graph_closeness(
                etrecord.edge_dialect_program,
                edge_output.exported_program().graph_module,
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
                            reserved_name: captured_output.graph_module
                        },
                    )

    def test_etrecord_generation_with_exported_program(self):
        """Test that exported program can be recorded and parsed back correctly."""
        captured_output, edge_output, et_output = self.get_test_model()
        original_exported_program = captured_output
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
                edge_output.exported_program().graph_module,
            )
            self.assertEqual(
                etrecord._debug_handle_map,
                json.loads(json.dumps(et_output.debug_handle_map)),
            )

            # Validate that export_graph_id matches the expected value
            self.assertEqual(etrecord.export_graph_id, expected_graph_id)

    def test_to_edge_transform_and_lower_with_etrecord_generation(self):
        """Test that to_edge_transform_and_lower generates ETRecord correctly."""
        f = models.BasicSinMax()
        aten_program = export(f, f.get_random_inputs(), strict=True)

        # Test with generate_etrecord=True
        edge_manager = to_edge_transform_and_lower(
            aten_program,
            generate_etrecord=True,
        )

        # Verify that ETRecord was generated and attached
        self.assertIsNotNone(edge_manager._etrecord)
        etrecord = edge_manager._etrecord
        self.assert_legal_etrecord_in_edge_program(etrecord)

        # Verify the exported program matches the input
        self.check_graph_closeness(
            etrecord.exported_program,
            aten_program.graph_module,
        )
        self.assertEqual(
            etrecord.export_graph_id,
            id(aten_program.graph),
        )

        # Verify the edge dialect program matches the edge manager
        self.check_graph_closeness(
            etrecord.edge_dialect_program,
            edge_manager.exported_program().graph_module,
        )

    def test_to_edge_transform_and_lower_without_etrecord_generation(self):
        """Test that to_edge_transform_and_lower works correctly without ETRecord generation."""
        f = models.BasicSinMax()
        aten_program = export(f, f.get_random_inputs(), strict=True)

        # Test with generate_etrecord=False (default)
        edge_manager = to_edge_transform_and_lower(aten_program)

        # Verify that no ETRecord was generated
        self.assertIsNone(edge_manager._etrecord)

        # Verify that the edge manager still works correctly
        self.assertIsNotNone(edge_manager.exported_program())

    def test_get_etrecord_from_executorch_program_manager(self):
        """Test getting ETRecord from ExecutorchProgramManager using get_etrecord() method."""
        f = models.BasicSinMax()
        aten_program = export(f, f.get_random_inputs(), strict=True)

        # Generate edge manager with ETRecord
        edge_manager = to_edge_transform_and_lower(
            aten_program,
            generate_etrecord=True,
        )

        # Convert to executorch
        et_manager = edge_manager.to_executorch()

        # Test get_etrecord method
        etrecord = et_manager.get_etrecord()
        self.assertIsNotNone(etrecord)
        self.assert_etrecord_saveable(etrecord)

        # Verify the data matches the original input
        self.check_graph_closeness(
            etrecord.exported_program,
            aten_program.graph_module,
        )
        self.assertEqual(
            etrecord.export_graph_id,
            id(aten_program.graph),
        )

        # Verify the executorch program data matches
        # ETRecord stores data directly (not JSON serialized), so compare with original data
        self.assertEqual(etrecord._debug_handle_map, et_manager.debug_handle_map)
        self.assertEqual(etrecord._delegate_map, et_manager.delegate_map)

    def test_get_etrecord_from_executorch_program_manager_with_partitioner(self):
        """Test getting ETRecord from ExecutorchProgramManager using get_etrecord() method."""
        f = models.BasicSinMax()
        aten_program = export(f, f.get_random_inputs(), strict=True)

        # Generate edge manager with ETRecord
        edge_manager = to_edge_transform_and_lower(
            aten_program,
            partitioner=[XnnpackPartitioner()],
            generate_etrecord=True,
        )

        # Convert to executorch
        et_manager = edge_manager.to_executorch()

        # Test get_etrecord method
        etrecord = et_manager.get_etrecord()
        self.assertIsNotNone(etrecord)
        self.assert_etrecord_saveable(etrecord)

        # Verify the data matches the original input
        self.check_graph_closeness(
            etrecord.exported_program,
            aten_program.graph_module,
        )
        self.assertEqual(
            etrecord.export_graph_id,
            id(aten_program.graph),
        )

        # Verify the executorch program data matches
        # ETRecord stores data directly (not JSON serialized), so compare with original data
        self.assertEqual(etrecord._debug_handle_map, et_manager.debug_handle_map)
        self.assertEqual(etrecord._delegate_map, et_manager.delegate_map)

    def test_get_etrecord_from_executorch_program_manager_without_generation(self):
        """Test getting ETRecord from ExecutorchProgramManager when ETRecord was not generated."""
        f = models.BasicSinMax()
        aten_program = export(f, f.get_random_inputs(), strict=True)

        # Generate edge manager without ETRecord
        edge_manager = to_edge_transform_and_lower(aten_program)

        # Verify no ETRecord on edge manager
        self.assertIsNone(edge_manager._etrecord)

        # Convert to executorch
        et_manager = edge_manager.to_executorch()

        # Verify no ETRecord on executorch manager
        self.assertIsNone(et_manager._etrecord)

        # Test get_etrecord method should raise RuntimeError
        with self.assertRaises(RuntimeError) as context:
            et_manager.get_etrecord()

        self.assertIn("ETRecord was not generated", str(context.exception))

    def test_to_edge_with_etrecord_generation(self):
        """Test that to_edge generates ETRecord correctly."""
        aten_program, edge_manager, _ = self.get_test_model(generate_etrecord=True)

        # Verify that ETRecord was generated and attached
        self.assertIsNotNone(edge_manager._etrecord)
        etrecord = edge_manager._etrecord
        self.assert_legal_etrecord_in_edge_program(etrecord)

        # Verify the exported program matches the input
        self.check_graph_closeness(
            etrecord.exported_program,
            aten_program.graph_module,
        )
        self.assertEqual(
            etrecord.export_graph_id,
            id(aten_program.graph),
        )

        # Verify the edge dialect program matches the edge manager
        self.check_graph_closeness(
            etrecord.edge_dialect_program,
            edge_manager.exported_program().graph_module,
        )

    def test_to_edge_without_etrecord_generation(self):
        """Test that to_edge works correctly without ETRecord generation."""
        # Test with generate_etrecord=False (default)
        _, edge_manager, et_manager = self.get_test_model()

        # Verify that no ETRecord was generated
        self.assertIsNone(edge_manager._etrecord)

        # Test get_etrecord method should raise RuntimeError
        with self.assertRaises(RuntimeError):
            et_manager.get_etrecord()

    def test_to_edge_etrecord_save_and_parse(self):
        """Test that ETRecord generated by to_edge can be saved and parsed."""
        aten_program, _, et_manager = self.get_test_model(generate_etrecord=True)

        etrecord = et_manager.get_etrecord()

        with tempfile.TemporaryDirectory() as tmpdirname:
            etrecord_path = tmpdirname + "/etrecord_to_edge.bin"

            etrecord.save(etrecord_path)

            # Parse ETRecord back and verify
            parsed_etrecord = parse_etrecord(etrecord_path)

            # Validate that all components are preserved
            # Note: Skip graph structure comparison due to transformation differences
            self.check_graph_closeness(
                etrecord.exported_program, parsed_etrecord.exported_program
            )
            self.check_graph_closeness(
                etrecord.edge_dialect_program, parsed_etrecord.edge_dialect_program
            )

            # Validate executorch program data
            self.assertEqual(
                parsed_etrecord._debug_handle_map,
                json.loads(json.dumps(et_manager.debug_handle_map)),
            )
            self.assertEqual(
                parsed_etrecord._delegate_map,
                json.loads(json.dumps(et_manager.delegate_map)),
            )

            # Validate export graph id
            self.assertEqual(
                parsed_etrecord.export_graph_id,
                id(aten_program.graph),
            )

    def test_to_edge_transform_and_lower_etrecord_save_and_parse(self):
        """Test that ETRecord generated by to_edge_transform_and_lower can be saved and parsed."""
        f = models.BasicSinMax()
        aten_program = export(f, f.get_random_inputs(), strict=True)

        # Generate edge manager with ETRecord
        edge_manager = to_edge_transform_and_lower(
            aten_program,
            partitioner=[XnnpackPartitioner()],
            generate_etrecord=True,
        )

        # Convert to executorch to get complete ETRecord
        et_manager = edge_manager.to_executorch()
        etrecord = et_manager.get_etrecord()

        with tempfile.TemporaryDirectory() as tmpdirname:
            etrecord_path = tmpdirname + "/etrecord_flow2.bin"

            etrecord.save(etrecord_path)

            # Parse ETRecord back and verify
            parsed_etrecord = parse_etrecord(etrecord_path)

            # Validate that all components are preserved
            # Note: Skip graph structure comparison due to transformation differences
            self.check_graph_closeness(
                etrecord.exported_program, parsed_etrecord.exported_program
            )
            self.check_graph_closeness(
                etrecord.edge_dialect_program, parsed_etrecord.edge_dialect_program
            )

            # Validate executorch program data
            self.assertEqual(
                parsed_etrecord._debug_handle_map,
                json.loads(json.dumps(et_manager.debug_handle_map)),
            )
            self.assertEqual(
                parsed_etrecord._delegate_map,
                json.loads(json.dumps(et_manager.delegate_map)),
            )

            # Validate export graph id
            self.assertEqual(
                parsed_etrecord.export_graph_id,
                id(aten_program.graph),
            )

    def test_add_extra_export_modules(self):
        """Test add_extra_export_modules when ETRecord already has a graph_map."""
        captured_output, edge_output, et_output = self.get_test_model()

        # Create an ETRecord instance with existing graph_map
        initial_graph_map = {"existing_module/forward": captured_output}
        etrecord = ETRecord(
            exported_program=captured_output,
            export_graph_id=id(captured_output.graph),
            edge_dialect_program=edge_output.exported_program(),
            graph_map=initial_graph_map,
            _debug_handle_map=et_output.debug_handle_map,
            _delegate_map=et_output.delegate_map,
        )

        # Verify initial state
        self.assertIsNotNone(etrecord.graph_map)
        self.assertIn("existing_module/forward", etrecord.graph_map)

        # Create additional module to add
        f2 = models.BasicSinMax()
        captured_output2 = exir.capture(
            f2, f2.get_random_inputs(), exir.CaptureConfig()
        )

        extra_modules = {
            "new_module": captured_output2.exported_program,
        }

        # Add extra export modules
        etrecord.add_extra_export_modules(extra_modules)

        # Verify both existing and new modules are present
        self.assertIn("existing_module/forward", etrecord.graph_map)
        self.assertIn("new_module/forward", etrecord.graph_map)

        # Verify the modules are correctly stored
        self.check_graph_closeness(
            etrecord.graph_map["existing_module/forward"],
            captured_output.graph_module,
        )
        self.check_graph_closeness(
            etrecord.graph_map["new_module/forward"],
            captured_output2.exported_program.graph_module,
        )

    def test_add_extra_export_modules_reserved_name_validation(self):
        """Test that add_extra_export_modules validates reserved names."""
        captured_output, edge_output, et_output = self.get_test_model()

        etrecord = ETRecord(
            exported_program=captured_output,
            export_graph_id=id(captured_output.graph),
            edge_dialect_program=edge_output.exported_program(),
            _debug_handle_map=et_output.debug_handle_map,
            _delegate_map=et_output.delegate_map,
        )

        # Test that reserved names are rejected
        for reserved_name in ETRecordReservedFileNames:
            with self.assertRaises(RuntimeError):
                etrecord.add_extra_export_modules({reserved_name: captured_output})

    def test_etrecord_class_constructor_and_save(self):
        """Test that ETRecord class constructor and save method work correctly."""
        captured_output, edge_output, et_output = self.get_test_model()
        original_exported_program = captured_output
        expected_graph_id = id(original_exported_program.graph)

        # Create ETRecord instance directly using constructor
        etrecord = ETRecord(
            exported_program=original_exported_program,
            export_graph_id=expected_graph_id,
            edge_dialect_program=edge_output.exported_program(),
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
                edge_output.exported_program().graph_module,
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
            exported_program=captured_output,
            export_graph_id=id(captured_output.graph),
            edge_dialect_program=edge_output.exported_program(),
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
                    expected_outputs["forward"][0][0],
                    reference_outputs["forward"][0][0],
                )
            )
            self.assertTrue(
                torch.equal(
                    expected_outputs["forward"][1][0],
                    reference_outputs["forward"][1][0],
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
        original_exported_program = captured_output
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
                edge_output.exported_program().graph_module,
            )
            self.assertEqual(
                etrecord._debug_handle_map,
                json.loads(json.dumps(et_output.debug_handle_map)),
            )

            # Validate that export_graph_id matches the expected value
            self.assertEqual(etrecord.export_graph_id, expected_graph_id)

    def test_add_executorch_program(self):
        """Test add_executorch_program when ETRecord has no existing executorch program data."""
        captured_output, edge_output, et_output = self.get_test_model()

        # Create an ETRecord instance without executorch program data
        etrecord = ETRecord(
            exported_program=captured_output,
            export_graph_id=id(captured_output.graph),
            edge_dialect_program=edge_output.exported_program(),
        )

        # Verify initial state - no executorch program data
        self.assert_etrecord_has_no_executorch_program(etrecord)

        # Add executorch program
        etrecord.add_executorch_program(et_output)

        # Verify executorch program data is now present
        self.assertIsNotNone(etrecord._debug_handle_map)
        self.assertIsNotNone(etrecord._delegate_map)
        self.assertEqual(etrecord._debug_handle_map, et_output.debug_handle_map)
        self.assertEqual(etrecord._delegate_map, et_output.delegate_map)
        # For regular ExecutorchProgram, reference_outputs and representative_inputs should be None
        self.assertIsNone(etrecord._reference_outputs)
        self.assertIsNone(etrecord._representative_inputs)

    def test_add_executorch_program_with_bundled_program(self):
        """Test add_executorch_program with BundledProgram."""
        (
            captured_output,
            edge_output,
            bundled_program,
        ) = self.get_test_model_with_bundled_program()

        # Create an ETRecord instance without executorch program data
        etrecord = ETRecord(
            exported_program=captured_output,
            export_graph_id=id(captured_output.graph),
            edge_dialect_program=edge_output.exported_program(),
        )

        # Verify initial state - no executorch program data
        self.assertIsNone(etrecord._debug_handle_map)
        self.assertIsNone(etrecord._delegate_map)
        self.assertIsNone(etrecord._reference_outputs)
        self.assertIsNone(etrecord._representative_inputs)

        # Add bundled program
        etrecord.add_executorch_program(bundled_program)

        # Verify executorch program data is now present
        self.assertIsNotNone(etrecord._debug_handle_map)
        self.assertIsNotNone(etrecord._delegate_map)
        self.assertIsNotNone(etrecord._reference_outputs)
        self.assertIsNotNone(etrecord._representative_inputs)

        # Verify the data matches expected values
        expected_reference_outputs = _get_reference_outputs(bundled_program)
        expected_representative_inputs = _get_representative_inputs(bundled_program)

        # Compare reference outputs
        self.assertTrue(
            torch.equal(
                etrecord._reference_outputs["forward"][0][0],
                expected_reference_outputs["forward"][0][0],
            )
        )
        self.assertTrue(
            torch.equal(
                etrecord._reference_outputs["forward"][1][0],
                expected_reference_outputs["forward"][1][0],
            )
        )

        # Compare representative inputs
        for expected, actual in zip(
            etrecord._representative_inputs, expected_representative_inputs
        ):
            self.assertTrue(torch.equal(expected[0], actual[0]))
            self.assertTrue(torch.equal(expected[1], actual[1]))

    def test_add_executorch_program_already_exists_exception(self):
        """Test that add_executorch_program raises exception when executorch program data already exists."""
        captured_output, edge_output, et_output = self.get_test_model()

        # Create an ETRecord instance with existing executorch program data
        etrecord = ETRecord(
            exported_program=captured_output,
            export_graph_id=id(captured_output.graph),
            edge_dialect_program=edge_output.exported_program(),
            _debug_handle_map=et_output.debug_handle_map,
            _delegate_map=et_output.delegate_map,
        )

        # Verify that adding executorch program raises RuntimeError
        with self.assertRaises(RuntimeError) as context:
            etrecord.add_executorch_program(et_output)

        self.assertIn(
            "Executorch program data already exists in the ETRecord",
            str(context.exception),
        )

    def test_add_executorch_program_partial_data_exists_exception(self):
        """Test that add_executorch_program raises exception when partial executorch program data exists."""
        captured_output, edge_output, et_output = self.get_test_model()

        # Create an ETRecord instance with only debug_handle_map (partial data)
        etrecord = ETRecord(
            exported_program=captured_output,
            export_graph_id=id(captured_output.graph),
            edge_dialect_program=edge_output.exported_program(),
            _debug_handle_map=et_output.debug_handle_map,
        )

        # Verify that adding executorch program raises RuntimeError even with partial data
        with self.assertRaises(RuntimeError) as context:
            etrecord.add_executorch_program(et_output)

        self.assertIn(
            "Executorch program data already exists in the ETRecord",
            str(context.exception),
        )

    def test_add_executorch_program_and_save(self):
        """Test that ETRecord with added executorch program can be saved and parsed correctly."""
        captured_output, edge_output, et_output = self.get_test_model()

        # Create an ETRecord instance without executorch program data
        etrecord = ETRecord(
            exported_program=captured_output,
            export_graph_id=id(captured_output.graph),
            edge_dialect_program=edge_output.exported_program(),
        )

        # Add executorch program
        etrecord.add_executorch_program(et_output)

        with tempfile.TemporaryDirectory() as tmpdirname:
            etrecord_path = tmpdirname + "/etrecord_with_added_program.bin"

            # Save the ETRecord
            etrecord.save(etrecord_path)

            # Parse ETRecord back and verify
            parsed_etrecord = parse_etrecord(etrecord_path)

            # Validate that all components are preserved
            self.assertIsNotNone(parsed_etrecord.exported_program)
            self.check_graph_closeness(
                parsed_etrecord.exported_program,
                captured_output.graph_module,
            )

            self.assertIsNotNone(parsed_etrecord.edge_dialect_program)
            self.check_graph_closeness(
                parsed_etrecord.edge_dialect_program,
                edge_output.exported_program().graph_module,
            )

            # Validate executorch program data
            self.assertEqual(
                parsed_etrecord._debug_handle_map,
                json.loads(json.dumps(et_output.debug_handle_map)),
            )
            self.assertEqual(
                parsed_etrecord._delegate_map,
                json.loads(json.dumps(et_output.delegate_map)),
            )

            # Validate export graph id
            self.assertEqual(
                parsed_etrecord.export_graph_id,
                id(captured_output.graph),
            )

    def test_add_exported_program(self):
        """Test add_exported_program when ETRecord has no existing exported program."""
        captured_output, edge_output, et_output = self.get_test_model()

        # Create an ETRecord instance without exported program
        etrecord = ETRecord(
            edge_dialect_program=edge_output.exported_program(),
            _debug_handle_map=et_output.debug_handle_map,
            _delegate_map=et_output.delegate_map,
        )

        # Verify initial state - no exported program
        self.assert_etrecord_has_no_exported_program(etrecord)

        # Add exported program
        etrecord.add_exported_program(captured_output)

        # Verify exported program is now present
        self.assertIsNotNone(etrecord.exported_program)
        self.assertIsNotNone(etrecord.export_graph_id)
        self.check_graph_closeness(
            etrecord.exported_program,
            captured_output.graph_module,
        )
        self.assertEqual(
            etrecord.export_graph_id,
            id(captured_output.graph),
        )

    def test_add_exported_program_with_dict(self):
        """Test add_exported_program with dictionary input."""
        captured_output, edge_output, et_output = self.get_test_model()

        # Create an ETRecord instance without exported program
        etrecord = ETRecord(
            edge_dialect_program=edge_output.exported_program(),
            _debug_handle_map=et_output.debug_handle_map,
            _delegate_map=et_output.delegate_map,
        )

        # Verify initial state - no exported program
        self.assertIsNone(etrecord.exported_program)
        self.assertIsNone(etrecord.export_graph_id)

        # Add exported program as dictionary
        exported_program_dict = {"forward": captured_output}
        etrecord.add_exported_program(exported_program_dict)

        # Verify exported program is now present
        self.assertIsNotNone(etrecord.exported_program)
        self.assertIsNotNone(etrecord.export_graph_id)
        self.check_graph_closeness(
            etrecord.exported_program,
            captured_output.graph_module,
        )
        self.assertEqual(
            etrecord.export_graph_id,
            id(captured_output.graph),
        )

    def test_add_exported_program_already_exists_exception(self):
        """Test that add_exported_program raises exception when exported program already exists."""
        captured_output, edge_output, et_output = self.get_test_model()

        # Create an ETRecord instance with existing exported program
        etrecord = ETRecord(
            exported_program=captured_output,
            export_graph_id=id(captured_output.graph),
            edge_dialect_program=edge_output.exported_program(),
            _debug_handle_map=et_output.debug_handle_map,
            _delegate_map=et_output.delegate_map,
        )

        # Create another exported program to try to add
        f2 = models.BasicSinMax()
        captured_output2 = exir.capture(
            f2, f2.get_random_inputs(), exir.CaptureConfig()
        )

        # Verify that adding exported program raises RuntimeError
        with self.assertRaises(RuntimeError) as context:
            etrecord.add_exported_program(captured_output2.exported_program)

        self.assertIn(
            "Exported program already exists in the ETRecord",
            str(context.exception),
        )

    def test_add_exported_program_partial_data_exists_exception(self):
        """Test that add_exported_program raises exception when partial exported program data exists."""
        captured_output, edge_output, et_output = self.get_test_model()

        # Create an ETRecord instance with only export_graph_id (partial data)
        etrecord = ETRecord(
            export_graph_id=id(captured_output.graph),
            edge_dialect_program=edge_output.exported_program(),
            _debug_handle_map=et_output.debug_handle_map,
            _delegate_map=et_output.delegate_map,
        )

        # Verify that adding exported program raises RuntimeError even with partial data
        with self.assertRaises(RuntimeError) as context:
            etrecord.add_exported_program(captured_output)

        self.assertIn(
            "Exported program already exists in the ETRecord",
            str(context.exception),
        )

    def test_add_exported_program_with_none(self):
        """Test add_exported_program with None input."""
        captured_output, edge_output, et_output = self.get_test_model()

        # Create an ETRecord instance without exported program
        etrecord = ETRecord(
            edge_dialect_program=edge_output.exported_program(),
            _debug_handle_map=et_output.debug_handle_map,
            _delegate_map=et_output.delegate_map,
        )

        # Verify initial state - no exported program
        self.assert_etrecord_has_no_exported_program(etrecord)

        # Add None exported program (should not raise error)
        etrecord.add_exported_program(None)

        # Verify exported program is still None
        self.assert_etrecord_has_no_exported_program(etrecord)

    def test_add_exported_program_and_save(self):
        """Test that ETRecord with added exported program can be saved and parsed correctly."""
        captured_output, edge_output, et_output = self.get_test_model()

        # Create an ETRecord instance without exported program
        etrecord = ETRecord(
            edge_dialect_program=edge_output.exported_program(),
            _debug_handle_map=et_output.debug_handle_map,
            _delegate_map=et_output.delegate_map,
        )

        # Add exported program
        etrecord.add_exported_program(captured_output)

        with tempfile.TemporaryDirectory() as tmpdirname:
            etrecord_path = tmpdirname + "/etrecord_with_added_exported_program.bin"

            # Save the ETRecord
            etrecord.save(etrecord_path)

            # Parse ETRecord back and verify
            parsed_etrecord = parse_etrecord(etrecord_path)

            # Validate that all components are preserved
            self.assertIsNotNone(parsed_etrecord.exported_program)
            self.check_graph_closeness(
                parsed_etrecord.exported_program,
                captured_output.graph_module,
            )

            self.assertIsNotNone(parsed_etrecord.edge_dialect_program)
            self.check_graph_closeness(
                parsed_etrecord.edge_dialect_program,
                edge_output.exported_program().graph_module,
            )

            # Validate export graph id
            self.assertEqual(
                parsed_etrecord.export_graph_id,
                id(captured_output.graph),
            )

    def test_add_edge_dialect_program(self):
        """Test add_edge_dialect_program when ETRecord has no existing edge dialect program."""
        captured_output, edge_output, et_output = self.get_test_model()

        # Create an ETRecord instance without edge dialect program
        etrecord = ETRecord(
            exported_program=captured_output,
            export_graph_id=id(captured_output.graph),
            _debug_handle_map=et_output.debug_handle_map,
            _delegate_map=et_output.delegate_map,
        )

        # Verify initial state - no edge dialect program
        self.assert_etrecord_has_no_edge_dialect_program(etrecord)

        # Add edge dialect program
        etrecord.add_edge_dialect_program(edge_output)

        # Verify edge dialect program is now present
        self.assertIsNotNone(etrecord.edge_dialect_program)
        self.check_graph_closeness(
            etrecord.edge_dialect_program,
            edge_output.exported_program().graph_module,
        )

    def test_add_edge_dialect_program_already_exists_exception(self):
        """Test that add_edge_dialect_program raises exception when edge dialect program already exists."""
        captured_output, edge_output, et_output = self.get_test_model()

        # Create an ETRecord instance with existing edge dialect program
        etrecord = ETRecord(
            exported_program=captured_output,
            export_graph_id=id(captured_output.graph),
            edge_dialect_program=edge_output.exported_program(),
            _debug_handle_map=et_output.debug_handle_map,
            _delegate_map=et_output.delegate_map,
        )

        # Create another edge program to try to add
        f2 = models.BasicSinMax()
        captured_output2 = exir.capture(
            f2, f2.get_random_inputs(), exir.CaptureConfig()
        )
        edge_output2 = captured_output2.to_edge(
            exir.EdgeCompileConfig(_check_ir_validity=False, _use_edge_ops=False)
        )

        # Verify that adding edge dialect program raises RuntimeError
        with self.assertRaises(RuntimeError) as context:
            etrecord.add_edge_dialect_program(edge_output2)

        self.assertIn(
            "Edge dialect program already exists in the ETRecord",
            str(context.exception),
        )

    def test_add_edge_dialect_program_and_save(self):
        """Test that ETRecord with added edge dialect program can be saved and parsed correctly."""
        captured_output, edge_output, et_output = self.get_test_model()

        # Create an ETRecord instance without edge dialect program
        etrecord = ETRecord(
            exported_program=captured_output,
            export_graph_id=id(captured_output.graph),
            _debug_handle_map=et_output.debug_handle_map,
            _delegate_map=et_output.delegate_map,
        )

        # Add edge dialect program
        etrecord.add_edge_dialect_program(edge_output)

        with tempfile.TemporaryDirectory() as tmpdirname:
            etrecord_path = tmpdirname + "/etrecord_with_added_edge_program.bin"

            # Save the ETRecord
            etrecord.save(etrecord_path)

            # Parse ETRecord back and verify
            parsed_etrecord = parse_etrecord(etrecord_path)

            # Validate that all components are preserved
            self.assertIsNotNone(parsed_etrecord.exported_program)
            self.check_graph_closeness(
                parsed_etrecord.exported_program,
                captured_output.graph_module,
            )

            self.assertIsNotNone(parsed_etrecord.edge_dialect_program)
            self.check_graph_closeness(
                parsed_etrecord.edge_dialect_program,
                edge_output.exported_program().graph_module,
            )

            # Validate export graph id
            self.assertEqual(
                parsed_etrecord.export_graph_id,
                id(captured_output.graph),
            )

    def test_add_all_programs_sequentially(self):
        """Test adding all programs sequentially to an empty ETRecord."""
        captured_output, edge_output, et_output = self.get_test_model()

        # Create an empty ETRecord instance
        etrecord = ETRecord()

        # Verify initial state - everything is None
        self.assert_etrecord_is_empty(etrecord)

        # Add exported program
        etrecord.add_exported_program(captured_output)

        # Add edge dialect program
        etrecord.add_edge_dialect_program(edge_output)

        # Add executorch program
        etrecord.add_executorch_program(et_output)

        # Verify all components are now present
        self.assertIsNotNone(etrecord.exported_program)
        self.assertIsNotNone(etrecord.export_graph_id)
        self.assertIsNotNone(etrecord.edge_dialect_program)
        self.assertIsNotNone(etrecord._debug_handle_map)
        self.assertIsNotNone(etrecord._delegate_map)

        # Verify the data matches expected values
        self.check_graph_closeness(
            etrecord.exported_program,
            captured_output.graph_module,
        )
        self.check_graph_closeness(
            etrecord.edge_dialect_program,
            edge_output.exported_program().graph_module,
        )
        self.assertEqual(
            etrecord.export_graph_id,
            id(captured_output.graph),
        )
        self.assertEqual(etrecord._debug_handle_map, et_output.debug_handle_map)
        self.assertEqual(etrecord._delegate_map, et_output.delegate_map)

        # Test that the complete ETRecord can be saved and parsed
        with tempfile.TemporaryDirectory() as tmpdirname:
            etrecord_path = tmpdirname + "/etrecord_complete.bin"

            # Save the ETRecord
            etrecord.save(etrecord_path)

            # Parse ETRecord back and verify
            parsed_etrecord = parse_etrecord(etrecord_path)

            # Validate that all components are preserved
            self.assertIsNotNone(parsed_etrecord.exported_program)
            self.check_graph_closeness(
                parsed_etrecord.exported_program,
                captured_output.graph_module,
            )

            self.assertIsNotNone(parsed_etrecord.edge_dialect_program)
            self.check_graph_closeness(
                parsed_etrecord.edge_dialect_program,
                edge_output.exported_program().graph_module,
            )

            # Validate all metadata
            self.assertEqual(
                parsed_etrecord.export_graph_id,
                id(captured_output.graph),
            )
            self.assertEqual(
                parsed_etrecord._debug_handle_map,
                json.loads(json.dumps(et_output.debug_handle_map)),
            )
            self.assertEqual(
                parsed_etrecord._delegate_map,
                json.loads(json.dumps(et_output.delegate_map)),
            )

    def test_executorch_export_with_etrecord_generation(self):
        """Test that executorch.export generates ETRecord correctly when generate_etrecord=True."""
        # Verify that ETRecord was generated and can be retrieved
        export_session = self.get_test_export_session(generate_etrecord=True)
        etrecord = export_session.get_etrecord()
        self.assertIsNotNone(etrecord)
        self.assert_etrecord_saveable(etrecord)

        # Verify the executorch program data matches
        et_manager = export_session.get_executorch_program_manager()
        self.assertEqual(etrecord._debug_handle_map, et_manager.debug_handle_map)
        self.assertEqual(etrecord._delegate_map, et_manager.delegate_map)

    def test_executorch_export_without_etrecord_generation(self):
        """Test that executorch.export works correctly without ETRecord generation."""
        # Test with generate_etrecord=False (default)
        export_session = self.get_test_export_session(generate_etrecord=False)

        # Verify that no ETRecord was generated
        with self.assertRaises(RuntimeError) as context:
            export_session.get_etrecord()

        self.assertIn("ETRecord was not generated", str(context.exception))

        # Verify that the export session still works correctly
        self.assertIsNotNone(export_session.get_executorch_program_manager())
        self.assertTrue(len(export_session.get_pte_buffer()) > 0)

    def test_executorch_export_etrecord_save_and_parse(self):
        """Test that ETRecord generated by executorch.export can be saved and parsed."""
        export_session = self.get_test_export_session(generate_etrecord=True)

        etrecord = export_session.get_etrecord()

        with tempfile.TemporaryDirectory() as tmpdirname:
            etrecord_path = tmpdirname + "/etrecord_export.bin"

            etrecord.save(etrecord_path)

            # Parse ETRecord back and verify
            parsed_etrecord = parse_etrecord(etrecord_path)

            # Validate that all components are preserved
            self.assertIsNotNone(parsed_etrecord.exported_program)
            self.assertIsNotNone(parsed_etrecord.edge_dialect_program)

            # Validate executorch program data
            et_manager = export_session.get_executorch_program_manager()
            self.assertEqual(
                parsed_etrecord._debug_handle_map,
                json.loads(json.dumps(et_manager.debug_handle_map)),
            )
            self.assertEqual(
                parsed_etrecord._delegate_map,
                json.loads(json.dumps(et_manager.delegate_map)),
            )

            # Validate export graph id is preserved
            self.assertIsNotNone(parsed_etrecord.export_graph_id)

    def test_executorch_export_with_to_edge_flow(self):
        """Test executorch.export with TO_EDGE flow and ETRecord generation."""
        export_session = self.get_test_export_session(
            generate_etrecord=True,
            to_edge_flow=True,
        )

        # Verify that ETRecord was generated
        etrecord = export_session.get_etrecord()
        self.assertIsNotNone(etrecord)
        self.assert_etrecord_saveable(etrecord)

    def test_executorch_export_etrecord_with_to_edge_flow_save_and_parse(self):
        """Test that ETRecord generated by executorch.export can be saved and parsed."""
        export_session = self.get_test_export_session(
            generate_etrecord=True,
            to_edge_flow=True,
        )

        etrecord = export_session.get_etrecord()

        with tempfile.TemporaryDirectory() as tmpdirname:
            etrecord_path = tmpdirname + "/etrecord_export.bin"

            etrecord.save(etrecord_path)

            # Parse ETRecord back and verify
            parsed_etrecord = parse_etrecord(etrecord_path)

            # Validate that all components are preserved
            self.assertIsNotNone(parsed_etrecord.exported_program)
            self.assertIsNotNone(parsed_etrecord.edge_dialect_program)

            # Validate executorch program data
            et_manager = export_session.get_executorch_program_manager()
            self.assertEqual(
                parsed_etrecord._debug_handle_map,
                json.loads(json.dumps(et_manager.debug_handle_map)),
            )
            self.assertEqual(
                parsed_etrecord._delegate_map,
                json.loads(json.dumps(et_manager.delegate_map)),
            )

            # Validate export graph id is preserved
            self.assertIsNotNone(parsed_etrecord.export_graph_id)

    def test_update_representative_inputs_with_list(self):
        """Test update_representative_inputs with a list of ProgramInput objects."""
        captured_output, edge_output, et_output = self.get_test_model()

        # Create an ETRecord instance
        etrecord = ETRecord(
            exported_program=captured_output,
            export_graph_id=id(captured_output.graph),
            edge_dialect_program=edge_output.exported_program(),
            _debug_handle_map=et_output.debug_handle_map,
            _delegate_map=et_output.delegate_map,
        )

        # Verify initial state - no representative inputs
        self.assertIsNone(etrecord._representative_inputs)

        # Create custom representative inputs
        f = models.BasicSinMax()
        custom_inputs = [f.get_random_inputs() for _ in range(3)]

        # Update representative inputs
        etrecord.update_representative_inputs(custom_inputs)

        # Verify representative inputs are now set
        self.assertIsNotNone(etrecord._representative_inputs)
        self.assertEqual(len(etrecord._representative_inputs), 3)

        # Compare the inputs using utility function
        self.assert_representative_inputs_equal(
            custom_inputs,
            etrecord._representative_inputs,
            "Custom inputs do not match ETRecord representative inputs",
        )

    def test_update_representative_inputs_with_bundled_program(self):
        """Test update_representative_inputs with a BundledProgram."""
        (
            captured_output,
            edge_output,
            bundled_program,
        ) = self.get_test_model_with_bundled_program()

        # Create an ETRecord instance
        etrecord = ETRecord(
            exported_program=captured_output,
            export_graph_id=id(captured_output.graph),
            edge_dialect_program=edge_output.exported_program(),
            _debug_handle_map=bundled_program.executorch_program.debug_handle_map,
            _delegate_map=bundled_program.executorch_program.delegate_map,
        )

        # Verify initial state - no representative inputs
        self.assertIsNone(etrecord._representative_inputs)

        # Update representative inputs using bundled program
        etrecord.update_representative_inputs(bundled_program)

        # Verify representative inputs are now set
        self.assertIsNotNone(etrecord._representative_inputs)

        # Compare with expected inputs from bundled program using utility function
        expected_inputs = _get_representative_inputs(bundled_program)
        self.assert_representative_inputs_equal(
            expected_inputs,
            etrecord._representative_inputs,
            "Bundled program inputs do not match ETRecord representative inputs",
        )

    def test_update_representative_inputs_overwrite_existing(self):
        """Test that update_representative_inputs overwrites existing inputs."""
        (
            captured_output,
            edge_output,
            bundled_program,
        ) = self.get_test_model_with_bundled_program()

        # Create an ETRecord instance with existing representative inputs
        initial_inputs = _get_representative_inputs(bundled_program)
        etrecord = ETRecord(
            exported_program=captured_output,
            export_graph_id=id(captured_output.graph),
            edge_dialect_program=edge_output.exported_program(),
            _debug_handle_map=bundled_program.executorch_program.debug_handle_map,
            _delegate_map=bundled_program.executorch_program.delegate_map,
            _representative_inputs=initial_inputs,
        )

        # Verify initial inputs are set
        self.assertIsNotNone(etrecord._representative_inputs)

        # Create new custom inputs
        f = models.BasicSinMax()
        new_inputs = [f.get_random_inputs() for _ in range(2)]

        # Update representative inputs with new inputs
        etrecord.update_representative_inputs(new_inputs)

        # Verify inputs are updated using utility function
        self.assertEqual(len(etrecord._representative_inputs), 2)
        self.assert_representative_inputs_equal(
            new_inputs,
            etrecord._representative_inputs,
            "New inputs do not match ETRecord representative inputs after overwrite",
        )

    def test_update_reference_outputs_with_dict(self):
        """Test update_reference_outputs with a dictionary of outputs."""
        captured_output, edge_output, et_output = self.get_test_model()

        # Create an ETRecord instance
        etrecord = ETRecord(
            exported_program=captured_output,
            export_graph_id=id(captured_output.graph),
            edge_dialect_program=edge_output.exported_program(),
            _debug_handle_map=et_output.debug_handle_map,
            _delegate_map=et_output.delegate_map,
        )

        # Verify initial state - no reference outputs
        self.assertIsNone(etrecord._reference_outputs)

        # Create custom reference outputs
        f = models.BasicSinMax()
        inputs = [f.get_random_inputs() for _ in range(2)]
        custom_outputs = {
            "forward": [f.forward(*inp) for inp in inputs],
            "custom_method": [torch.tensor([1.0, 2.0]), torch.tensor([3.0, 4.0])],
        }

        # Update reference outputs
        etrecord.update_reference_outputs(custom_outputs)

        # Verify reference outputs are now set
        self.assertIsNotNone(etrecord._reference_outputs)
        self.assertIn("forward", etrecord._reference_outputs)
        self.assertIn("custom_method", etrecord._reference_outputs)

        # Compare the outputs
        self.assertEqual(len(etrecord._reference_outputs["forward"]), 2)
        self.assertEqual(len(etrecord._reference_outputs["custom_method"]), 2)

        for expected, actual in zip(
            custom_outputs["forward"], etrecord._reference_outputs["forward"]
        ):
            self.assertTrue(torch.equal(expected[0], actual[0]))

        for expected, actual in zip(
            custom_outputs["custom_method"],
            etrecord._reference_outputs["custom_method"],
        ):
            self.assertTrue(torch.equal(expected, actual))

    def test_update_reference_outputs_with_list(self):
        """Test update_reference_outputs with a single list of outputs."""
        captured_output, edge_output, et_output = self.get_test_model()

        # Create an ETRecord instance
        etrecord = ETRecord(
            exported_program=captured_output,
            export_graph_id=id(captured_output.graph),
            edge_dialect_program=edge_output.exported_program(),
            _debug_handle_map=et_output.debug_handle_map,
            _delegate_map=et_output.delegate_map,
        )

        # Verify initial state - no reference outputs
        self.assertIsNone(etrecord._reference_outputs)

        # Create custom reference outputs as a single list
        f = models.BasicSinMax()
        inputs = [f.get_random_inputs() for _ in range(2)]
        custom_outputs_list = [f.forward(*inp) for inp in inputs]

        # Update reference outputs with a single list
        etrecord.update_reference_outputs(custom_outputs_list)

        # Verify reference outputs are now set and treated as "forward" method
        self.assertIsNotNone(etrecord._reference_outputs)
        self.assertIn("forward", etrecord._reference_outputs)
        self.assertEqual(len(etrecord._reference_outputs["forward"]), 2)

        # Compare the outputs
        for expected, actual in zip(
            custom_outputs_list, etrecord._reference_outputs["forward"]
        ):
            self.assertTrue(torch.equal(expected[0], actual[0]))

    def test_update_reference_outputs_with_bundled_program(self):
        """Test update_reference_outputs with a BundledProgram."""
        (
            captured_output,
            edge_output,
            bundled_program,
        ) = self.get_test_model_with_bundled_program()

        # Create an ETRecord instance
        etrecord = ETRecord(
            exported_program=captured_output,
            export_graph_id=id(captured_output.graph),
            edge_dialect_program=edge_output.exported_program(),
            _debug_handle_map=bundled_program.executorch_program.debug_handle_map,
            _delegate_map=bundled_program.executorch_program.delegate_map,
        )

        # Verify initial state - no reference outputs
        self.assertIsNone(etrecord._reference_outputs)

        # Update reference outputs using bundled program
        etrecord.update_reference_outputs(bundled_program)

        # Verify reference outputs are now set
        self.assertIsNotNone(etrecord._reference_outputs)
        self.assertIn("forward", etrecord._reference_outputs)

        # Compare with expected outputs from bundled program
        expected_outputs = _get_reference_outputs(bundled_program)
        self.assertTrue(
            torch.equal(
                etrecord._reference_outputs["forward"][0][0],
                expected_outputs["forward"][0][0],
            )
        )
        self.assertTrue(
            torch.equal(
                etrecord._reference_outputs["forward"][1][0],
                expected_outputs["forward"][1][0],
            )
        )

    def test_update_apis_and_save_parse(self):
        """Test that ETRecord with updated inputs/outputs can be saved and parsed correctly."""
        captured_output, edge_output, et_output = self.get_test_model()

        # Create an ETRecord instance
        etrecord = ETRecord(
            exported_program=captured_output,
            export_graph_id=id(captured_output.graph),
            edge_dialect_program=edge_output.exported_program(),
            _debug_handle_map=et_output.debug_handle_map,
            _delegate_map=et_output.delegate_map,
        )

        # Create custom inputs and outputs
        f = models.BasicSinMax()
        custom_inputs = [f.get_random_inputs() for _ in range(2)]
        custom_outputs = {
            "forward": [f.forward(*inp) for inp in custom_inputs],
        }

        # Update both inputs and outputs
        etrecord.update_representative_inputs(custom_inputs)
        etrecord.update_reference_outputs(custom_outputs)

        with tempfile.TemporaryDirectory() as tmpdirname:
            etrecord_path = tmpdirname + "/etrecord_with_custom_data.bin"

            # Save the ETRecord
            etrecord.save(etrecord_path)

            # Parse ETRecord back and verify
            parsed_etrecord = parse_etrecord(etrecord_path)

            # Verify representative inputs are preserved using utility function
            self.assertIsNotNone(parsed_etrecord._representative_inputs)
            self.assertEqual(len(parsed_etrecord._representative_inputs), 2)
            self.assert_representative_inputs_equal(
                custom_inputs,
                parsed_etrecord._representative_inputs,
                "Custom inputs do not match parsed ETRecord representative inputs",
            )

            # Verify reference outputs are preserved
            self.assertIsNotNone(parsed_etrecord._reference_outputs)
            self.assertIn("forward", parsed_etrecord._reference_outputs)
            self.assertEqual(len(parsed_etrecord._reference_outputs["forward"]), 2)
            for expected, actual in zip(
                custom_outputs["forward"], parsed_etrecord._reference_outputs["forward"]
            ):
                self.assertTrue(torch.equal(expected[0], actual[0]))

    def test_save_missing_essential_info(self):
        def expected_runtime_error(etrecord, etrecord_path):
            with self.assertRaises(RuntimeError) as context:
                etrecord.save(etrecord_path)

            self.assertIn(
                "ETRecord must contain edge dialect program and executorch program to be saved",
                str(context.exception),
            )

        """Test that save raises RuntimeError when essential info is missing."""
        _, edge_output, et_output = self.get_test_model()

        etrecord = ETRecord()

        with tempfile.TemporaryDirectory() as tmpdirname:
            etrecord_path = tmpdirname + "/etrecord_no_edge.bin"

            expected_runtime_error(etrecord, etrecord_path)
            etrecord.add_edge_dialect_program(edge_output)

            # Should raise runtime error due to  missing executorch program related info
            expected_runtime_error(etrecord, etrecord_path)

            etrecord.add_executorch_program(et_output)

            # All essential components are now present, so save should succeed
            etrecord.save(etrecord_path)
