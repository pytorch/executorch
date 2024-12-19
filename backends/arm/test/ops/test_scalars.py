import unittest

import torch

from executorch.backends.arm.test import common
from executorch.backends.arm.test.tester.arm_tester import ArmTester
from parameterized import parameterized

"""
Summary of non-working cases.
MI:
    Any case with int scalar: A to_copy is inserted to cast the value which we don't partition.
        This makes the constant end up outside our partition and the input to the delegate becomes
        a to_copy placeholder. In ArmTester, the placeholder is then interpreted as an input.
        Potential fix: partition int -> float to_copy-ops in ArmBackend.
        # MLETORCH-407
    Op(scalar, tensor):
        One issue is that lift_constant_tensor_pass looks for a fake_tensor in the meta of the first
        node which does not work the first node is a scalar.
        Fixing that, the lowering fails since edge_program.graph_signatures.inputs_to_buffers is changed from
        {"_lifted_tensor_constant0":"_lifted_tensor_constant0"} to {"x":"_lifted_tensor_constant0"}
        somewhere in _transform in the to_edge step. This makes ArmPartitioner miss tagging the
        data in tag_constant_data.
        # MLETORCH-408

BI:
    sub(Scalar, Tensor) becomes rsub, which either fails since the scalar does not become an attribute
        in scalars_to_attribute_pass, or, if added to targeted_ops in that pass, fails since rsub expects a
        Scalar.
        Potential fix: Create pass to convert rsub.Scalar to sub.Tensor
"""


class TestScalars(unittest.TestCase):
    """Tests various scalar cases for for"""

    class Add(torch.nn.Module):
        def forward(self, x, y):
            return x + y

    class Sub(torch.nn.Module):
        def forward(self, x, y):
            return x - y

    class Div(torch.nn.Module):
        def forward(self, x, y):
            return x / y

    class Mul(torch.nn.Module):
        def forward(self, x, y):
            return x * y

    class AddInplace(torch.nn.Module):
        def forward(self, x, y):
            x += y
            return x

    class SubInplace(torch.nn.Module):
        def forward(self, x, y):
            x -= y
            return x

    class DivInplace(torch.nn.Module):
        def forward(self, x, y):
            x /= y
            return x

    class MulInplace(torch.nn.Module):
        def forward(self, x, y):
            x *= y
            return x

    class AddConst(torch.nn.Module):
        def forward(self, x):
            x = 1.0 + x
            return x

    class ShiftInplaceSub(torch.nn.Module):
        def forward(self, x):
            x = x >> 4
            x -= 10
            return x

    # Inplace ops end with '_' (from aten naming)
    ops = [
        ("Add", Add()),
        ("Sub", Sub()),
        ("Mul", Mul()),
        ("Div", Div()),
        ("Add_", AddInplace()),
        ("Sub_", SubInplace()),
        ("Mul_", MulInplace()),
        ("Div_", DivInplace()),
    ]

    const_ops = [("Add", AddConst())]

    dtypes = [("int", 3), ("float", 3.0)]
    sizes = [("r1", (1)), ("r4", (2, 4, 5, 3))]

    # Create combinations of tests
    tensor_scalar_tests = []
    for op in ops:
        for dtype in dtypes:
            for size in sizes:
                test_name = f"{op[0]}_{dtype[0]}_{size[0]}"
                tensor = torch.rand(size[1])
                scalar = dtype[1]
                tensor_scalar_tests.append((test_name + "_ts", op[1], tensor, scalar))

                # Don't add (scalar, tensor) test case for inplace ops.
                if op[0][-1] == "_":
                    continue

                # sub(scalar, tensor) does not work in any case.
                if op[0][0:3] == "Sub":
                    continue
                tensor_scalar_tests.append((test_name + "_st", op[1], scalar, tensor))

    tensor_const_tests = []
    for op in const_ops:
        for size in sizes:
            test_name = f"{op[0]}_{size[0]}"
            tensor = torch.rand(size[1])
            tensor_const_tests.append((test_name, op[1], tensor))

    def _test_add_tosa_MI_pipeline(self, module: torch.nn.Module, test_data: tuple):
        (
            ArmTester(
                module,
                example_inputs=test_data,
                compile_spec=common.get_tosa_compile_spec("TOSA-0.80.0+MI"),
            )
            .export()
            .to_edge()
            .partition()
            .to_executorch()
            .run_method_and_compare_outputs(inputs=test_data)
        )

    def _test_add_tosa_BI_pipeline(self, module: torch.nn.Module, test_data: tuple):
        (
            ArmTester(
                module,
                example_inputs=test_data,
                compile_spec=common.get_tosa_compile_spec("TOSA-0.80.0+BI"),
            )
            .quantize()
            .export()
            .to_edge()
            .partition()
            .to_executorch()
            .run_method_and_compare_outputs(inputs=test_data)
        )

    @parameterized.expand(tensor_scalar_tests)
    def test_MI(self, test_name: str, op: torch.nn.Module, x, y):
        expected_exception = None
        if any(token in test_name for token in ("Sub_int", "Sub__int")):
            expected_exception = ValueError
        elif test_name.endswith("_st"):
            expected_exception = AttributeError

        if expected_exception:
            with self.assertRaises(
                expected_exception, msg=f"Test {test_name} is expected to fail."
            ):
                self._test_add_tosa_MI_pipeline(op, (x, y))
            return

        self._test_add_tosa_MI_pipeline(op, (x, y))

    # op(Scalar float, tensor) works if the scalar is constant.
    @parameterized.expand(tensor_const_tests)
    def test_MI_const(self, test_name: str, op: torch.nn.Module, x):
        self._test_add_tosa_MI_pipeline(op, (x,))

    @parameterized.expand(tensor_scalar_tests)
    def test_BI(self, test_name: str, op: torch.nn.Module, x, y):
        self._test_add_tosa_BI_pipeline(op, (x, y))

    def test_shift_sub_inplace_tosa_MI(self):
        self._test_add_tosa_MI_pipeline(self.ShiftInplaceSub(), (torch.IntTensor(5),))
