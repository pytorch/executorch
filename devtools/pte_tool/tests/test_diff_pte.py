# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-unsafe

import struct
import unittest

from executorch.devtools.pte_tool.diff_pte import diff_pte, format_diff_result

from executorch.exir._serialize._program import PTEFile, serialize_pte_binary
from executorch.exir.schema import (
    Buffer,
    Chain,
    ContainerMetadata,
    EValue,
    ExecutionPlan,
    Instruction,
    Int,
    KernelCall,
    Null,
    Operator,
    Program,
    ScalarType,
    SubsegmentOffsets,
    Tensor,
    TensorShapeDynamism,
)


def _make_program_with_constant_tensor(
    weights: bytes,
    sizes: list,
    scalar_type: ScalarType = ScalarType.FLOAT,
    op_name: str = "aten::add",
    op_overload: str = "out",
) -> Program:
    """Create a minimal program with one constant tensor using given weight bytes."""
    return Program(
        version=0,
        execution_plan=[
            ExecutionPlan(
                name="forward",
                values=[
                    EValue(
                        val=Tensor(
                            scalar_type=scalar_type,
                            storage_offset=0,
                            sizes=sizes,
                            dim_order=list(range(len(sizes))),
                            requires_grad=False,
                            layout=0,
                            data_buffer_idx=1,
                            allocation_info=None,
                            shape_dynamism=TensorShapeDynamism.STATIC,
                        )
                    ),
                    EValue(Int(0)),
                ],
                inputs=[1],
                outputs=[0],
                chains=[
                    Chain(
                        inputs=[],
                        outputs=[],
                        instructions=[Instruction(KernelCall(op_index=0, args=[0, 1]))],
                        stacktrace=None,
                    )
                ],
                container_meta_type=ContainerMetadata(
                    encoded_inp_str="", encoded_out_str=""
                ),
                operators=[Operator(name=op_name, overload=op_overload)],
                delegates=[],
                non_const_buffer_sizes=[0, 64],
            )
        ],
        constant_buffer=[Buffer(storage=b""), Buffer(storage=weights)],
        backend_delegate_data=[],
        segments=[],
        constant_segment=SubsegmentOffsets(segment_index=0, offsets=[]),
        named_data=[],
    )


def _serialize_program(program: Program) -> bytes:
    return bytes(serialize_pte_binary(PTEFile(program=program)))


class DiffPteTest(unittest.TestCase):
    def test_bitwise_equal(self) -> None:
        weights = struct.pack("<4f", 1.0, 2.0, 3.0, 4.0)
        prog = _make_program_with_constant_tensor(weights, sizes=[2, 2])
        data = _serialize_program(prog)

        result = diff_pte(data, data, "a.pte", "b.pte")
        self.assertTrue(result.bitwise_equal)
        self.assertIsNone(result.error)
        self.assertEqual(result.plan_diffs, [])

    def test_different_weights(self) -> None:
        weights_a = struct.pack("<4f", 1.0, 2.0, 3.0, 4.0)
        weights_b = struct.pack("<4f", 1.0, 2.5, 3.0, 4.5)
        prog_a = _make_program_with_constant_tensor(weights_a, sizes=[2, 2])
        prog_b = _make_program_with_constant_tensor(weights_b, sizes=[2, 2])
        data_a = _serialize_program(prog_a)
        data_b = _serialize_program(prog_b)

        result = diff_pte(data_a, data_b, "a.pte", "b.pte")
        self.assertFalse(result.bitwise_equal)
        self.assertIsNone(result.error)
        self.assertEqual(len(result.plan_diffs), 1)

        pd = result.plan_diffs[0]
        self.assertEqual(len(pd.tensor_diffs), 1)

        td = pd.tensor_diffs[0]
        self.assertTrue(td.bytes_differ)
        self.assertEqual(td.num_elements, 4)
        self.assertEqual(td.num_differing, 2)
        self.assertGreater(td.max_abs_diff, 0.0)
        self.assertGreater(td.mean_abs_diff, 0.0)
        self.assertEqual(len(td.element_diffs), 2)

        self.assertEqual(td.element_diffs[0].flat_index, 1)
        self.assertAlmostEqual(td.element_diffs[0].value_a, 2.0, places=5)
        self.assertAlmostEqual(td.element_diffs[0].value_b, 2.5, places=5)

    def test_different_operators(self) -> None:
        weights = struct.pack("<4f", 1.0, 2.0, 3.0, 4.0)
        prog_a = _make_program_with_constant_tensor(
            weights, sizes=[2, 2], op_name="aten::add", op_overload="out"
        )
        prog_b = _make_program_with_constant_tensor(
            weights, sizes=[2, 2], op_name="aten::mul", op_overload="out"
        )
        data_a = _serialize_program(prog_a)
        data_b = _serialize_program(prog_b)

        result = diff_pte(data_a, data_b, "a.pte", "b.pte")
        self.assertFalse(result.bitwise_equal)
        self.assertIsNone(result.error)
        self.assertEqual(len(result.plan_diffs), 1)

        pd = result.plan_diffs[0]
        self.assertIn("aten::add.out", pd.operators_only_in_a)
        self.assertIn("aten::mul.out", pd.operators_only_in_b)

    def test_different_evalue_int(self) -> None:
        """Non-tensor EValues with different values should be reported."""
        weights = struct.pack("<4f", 1.0, 2.0, 3.0, 4.0)
        prog_a = _make_program_with_constant_tensor(weights, sizes=[2, 2])
        prog_b = _make_program_with_constant_tensor(weights, sizes=[2, 2])
        # Change the Int EValue in plan B
        prog_b.execution_plan[0].values[1] = EValue(Int(42))
        data_a = _serialize_program(prog_a)
        data_b = _serialize_program(prog_b)

        result = diff_pte(data_a, data_b, "a.pte", "b.pte")
        self.assertFalse(result.bitwise_equal)
        self.assertEqual(len(result.plan_diffs), 1)

        pd = result.plan_diffs[0]
        self.assertEqual(len(pd.evalue_diffs), 1)
        evd = pd.evalue_diffs[0]
        self.assertEqual(evd.evalue_index, 1)
        self.assertEqual(evd.type_a, "Int")
        self.assertEqual(evd.type_b, "Int")
        self.assertEqual(len(evd.field_diffs), 1)
        self.assertEqual(evd.field_diffs[0].field_name, "int_val")
        self.assertEqual(evd.field_diffs[0].value_a, 0)
        self.assertEqual(evd.field_diffs[0].value_b, 42)

    def test_different_evalue_type(self) -> None:
        """EValues that change type between A and B should be reported."""
        weights = struct.pack("<4f", 1.0, 2.0, 3.0, 4.0)
        prog_a = _make_program_with_constant_tensor(weights, sizes=[2, 2])
        prog_b = _make_program_with_constant_tensor(weights, sizes=[2, 2])
        # Change type from Int to Null
        prog_b.execution_plan[0].values[1] = EValue(Null())
        data_a = _serialize_program(prog_a)
        data_b = _serialize_program(prog_b)

        result = diff_pte(data_a, data_b, "a.pte", "b.pte")
        self.assertFalse(result.bitwise_equal)
        self.assertEqual(len(result.plan_diffs), 1)

        pd = result.plan_diffs[0]
        self.assertEqual(len(pd.evalue_diffs), 1)
        evd = pd.evalue_diffs[0]
        self.assertEqual(evd.type_a, "Int")
        self.assertEqual(evd.type_b, "Null")

    def test_tensor_metadata_diff(self) -> None:
        """Tensors with same data but different metadata fields should be reported."""
        weights = struct.pack("<4f", 1.0, 2.0, 3.0, 4.0)
        prog_a = _make_program_with_constant_tensor(weights, sizes=[2, 2])
        prog_b = _make_program_with_constant_tensor(weights, sizes=[4])
        data_a = _serialize_program(prog_a)
        data_b = _serialize_program(prog_b)

        result = diff_pte(data_a, data_b, "a.pte", "b.pte")
        self.assertFalse(result.bitwise_equal)
        self.assertEqual(len(result.plan_diffs), 1)

        pd = result.plan_diffs[0]
        self.assertEqual(len(pd.tensor_diffs), 1)

        td = pd.tensor_diffs[0]
        metadata_field_names = [md.field_name for md in td.metadata_diffs]
        self.assertIn("sizes", metadata_field_names)
        self.assertIn("dim_order", metadata_field_names)

    def test_tensor_allocation_info_diff(self) -> None:
        """Tensors with different allocation_info should be reported."""
        weights = struct.pack("<4f", 1.0, 2.0, 3.0, 4.0)
        prog_a = _make_program_with_constant_tensor(weights, sizes=[2, 2])
        prog_b = _make_program_with_constant_tensor(weights, sizes=[2, 2])
        prog_b.execution_plan[0].values[0].val.storage_offset = 16
        data_a = _serialize_program(prog_a)
        data_b = _serialize_program(prog_b)

        result = diff_pte(data_a, data_b, "a.pte", "b.pte")
        self.assertFalse(result.bitwise_equal)
        self.assertEqual(len(result.plan_diffs), 1)
        td = result.plan_diffs[0].tensor_diffs[0]
        metadata_field_names = [md.field_name for md in td.metadata_diffs]
        self.assertIn("storage_offset", metadata_field_names)

    def test_format_output_equal(self) -> None:
        weights = struct.pack("<4f", 1.0, 2.0, 3.0, 4.0)
        prog = _make_program_with_constant_tensor(weights, sizes=[2, 2])
        data = _serialize_program(prog)

        result = diff_pte(data, data, "a.pte", "b.pte")
        output = format_diff_result(result)
        self.assertIn("bitwise equal", output)
        self.assertIn("a.pte", output)
        self.assertIn("b.pte", output)

    def test_format_output_different(self) -> None:
        weights_a = struct.pack("<4f", 1.0, 2.0, 3.0, 4.0)
        weights_b = struct.pack("<4f", 1.0, 2.5, 3.0, 4.5)
        prog_a = _make_program_with_constant_tensor(weights_a, sizes=[2, 2])
        prog_b = _make_program_with_constant_tensor(weights_b, sizes=[2, 2])
        data_a = _serialize_program(prog_a)
        data_b = _serialize_program(prog_b)

        result = diff_pte(data_a, data_b, "a.pte", "b.pte")
        output = format_diff_result(result)
        self.assertIn("NOT bitwise equal", output)
        self.assertIn("Differing tensors", output)
        self.assertIn("elements differ", output)
        self.assertIn("max_abs_diff", output)

    def test_format_evalue_diff(self) -> None:
        weights = struct.pack("<4f", 1.0, 2.0, 3.0, 4.0)
        prog_a = _make_program_with_constant_tensor(weights, sizes=[2, 2])
        prog_b = _make_program_with_constant_tensor(weights, sizes=[2, 2])
        prog_b.execution_plan[0].values[1] = EValue(Int(42))
        data_a = _serialize_program(prog_a)
        data_b = _serialize_program(prog_b)

        result = diff_pte(data_a, data_b, "a.pte", "b.pte")
        output = format_diff_result(result)
        self.assertIn("Differing values", output)
        self.assertIn("int_val", output)

    def test_format_metadata_diff(self) -> None:
        weights = struct.pack("<4f", 1.0, 2.0, 3.0, 4.0)
        prog_a = _make_program_with_constant_tensor(weights, sizes=[2, 2])
        prog_b = _make_program_with_constant_tensor(weights, sizes=[4])
        data_a = _serialize_program(prog_a)
        data_b = _serialize_program(prog_b)

        result = diff_pte(data_a, data_b, "a.pte", "b.pte")
        output = format_diff_result(result)
        self.assertIn("sizes:", output)

    def test_empty_files(self) -> None:
        result = diff_pte(b"", b"not empty", "a.pte", "b.pte")
        self.assertFalse(result.bitwise_equal)
        self.assertIsNotNone(result.error)
        self.assertIn("Deserialization failed", result.error)

    def test_identical_programs_different_serialization(self) -> None:
        """Two independently serialized identical programs should be bitwise equal."""
        weights = struct.pack("<4f", 1.0, 2.0, 3.0, 4.0)
        prog_a = _make_program_with_constant_tensor(weights, sizes=[2, 2])
        prog_b = _make_program_with_constant_tensor(weights, sizes=[2, 2])
        data_a = _serialize_program(prog_a)
        data_b = _serialize_program(prog_b)

        result = diff_pte(data_a, data_b, "a.pte", "b.pte")
        self.assertTrue(result.bitwise_equal)

    def test_operator_usage_in_tensor_info(self) -> None:
        weights_a = struct.pack("<4f", 1.0, 2.0, 3.0, 4.0)
        weights_b = struct.pack("<4f", 5.0, 6.0, 7.0, 8.0)
        prog_a = _make_program_with_constant_tensor(weights_a, sizes=[2, 2])
        prog_b = _make_program_with_constant_tensor(weights_b, sizes=[2, 2])
        data_a = _serialize_program(prog_a)
        data_b = _serialize_program(prog_b)

        result = diff_pte(data_a, data_b, "a.pte", "b.pte")
        self.assertEqual(len(result.plan_diffs), 1)
        td = result.plan_diffs[0].tensor_diffs[0]
        self.assertGreater(len(td.tensor_a.operator_usages), 0)
        self.assertEqual(td.tensor_a.operator_usages[0].name, "aten::add.out")
        self.assertEqual(td.tensor_a.operator_usages[0].arg_index, 0)

    def test_version_difference(self) -> None:
        weights = struct.pack("<4f", 1.0, 2.0, 3.0, 4.0)
        prog_a = _make_program_with_constant_tensor(weights, sizes=[2, 2])
        prog_b = _make_program_with_constant_tensor(weights, sizes=[2, 2])
        prog_b.version = 1
        data_a = _serialize_program(prog_a)
        data_b = _serialize_program(prog_b)

        result = diff_pte(data_a, data_b, "a.pte", "b.pte")
        self.assertFalse(result.bitwise_equal)
        self.assertEqual(result.version_a, 0)
        self.assertEqual(result.version_b, 1)

        output = format_diff_result(result)
        self.assertIn("Version: 0 vs 1", output)

    def test_format_verbose(self) -> None:
        weights_a = struct.pack("<4f", 1.0, 2.0, 3.0, 4.0)
        weights_b = struct.pack("<4f", 1.0, 2.5, 3.0, 4.5)
        prog_a = _make_program_with_constant_tensor(weights_a, sizes=[2, 2])
        prog_b = _make_program_with_constant_tensor(weights_b, sizes=[2, 2])
        data_a = _serialize_program(prog_a)
        data_b = _serialize_program(prog_b)

        result = diff_pte(data_a, data_b, "a.pte", "b.pte")
        output = format_diff_result(result, verbose=True)
        self.assertIn("byte sizes:", output)
