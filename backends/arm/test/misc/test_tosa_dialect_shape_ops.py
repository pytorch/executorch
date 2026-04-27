# Copyright 2026 Arm Limited and/or its affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.
#

import executorch.backends.arm.tosa.dialect  # noqa: F401
import pytest
import sympy  # type: ignore
import torch
from executorch.backends.arm.tosa.dialect.lib import TosaValueError
from executorch.backends.arm.tosa.specification import (
    TosaLoweringContext,
    TosaSpecification,
)
from executorch.exir.dialects._ops import ops as exir_ops
from torch._subclasses.fake_tensor import FakeTensorMode
from torch.fx.experimental.symbolic_shapes import ShapeEnv


def _make_symint(
    shape_env: ShapeEnv, symbol: str, hint: int, min: int = 1, max: int = 64
) -> torch.SymInt:
    """Create a symbolic dimension backed by the provided ShapeEnv."""
    symint = shape_env.create_symintnode(sympy.Symbol(symbol), hint=hint)
    symbol_expr = symint.node.expr
    shape_env.constrain_symbol_range(symbol_expr, compiler_min=min, compiler_max=max)
    return symint


def _expr(sym: torch.SymInt) -> str:
    """Return the SymPy expression backing a SymInt."""
    return str(sym.node._expr)


def _expr_equals(sym: torch.SymInt, expected: sympy.Expr) -> bool:
    """Return True if the SymPy expressions are equivalent."""
    actual = sympy.sympify(_expr(sym))
    expected_expr = sympy.sympify(expected)
    return sympy.simplify(actual - expected_expr) == 0


# Test that DIM can extract a symbolic dimension from a tensor when the TOSA specification supports the shape extension.
def test_dim_extracts_symbolic_dimension():
    shape_env = ShapeEnv()
    s0 = _make_symint(shape_env, "s0", hint=4)

    with TosaLoweringContext(
        TosaSpecification.create_from_string("TOSA-1.1+FP+shape"), shape_env
    ), FakeTensorMode(shape_env=shape_env) as mode:
        s0_tensor = torch.empty(size=(1, 3, s0))
        result = exir_ops.backend.tosa.DIM.default(mode.from_tensor(s0_tensor), axis=2)

    assert isinstance(result, list)
    assert len(result) == 1
    assert isinstance(result[0], torch.SymInt)
    assert _expr(result[0]) == "s0"


# Test that DIM raises an error when the TOSA specification doesn't support the shape extension, as DIM relies on shape
# expressions to return symbolic dimensions.
def test_dim_requires_shape_extension():
    spec_no_shape = TosaSpecification.create_from_string("TOSA-1.0+FP")
    shape_env = ShapeEnv()
    s0 = _make_symint(shape_env, "s0", hint=3)

    with TosaLoweringContext(
        spec_no_shape,
        shape_env,
    ), FakeTensorMode(shape_env=shape_env) as mode:
        s0_tensor = torch.empty(size=(1, 3, s0))
        with pytest.raises(TosaValueError, match="shape extension"):
            exir_ops.backend.tosa.DIM.default(mode.from_tensor(s0_tensor), axis=2)


# Test that CONST_SHAPE creates a constant shape tensor and returns the expected shape list.
def test_const_shape():
    with TosaLoweringContext(
        TosaSpecification.create_from_string("TOSA-1.1+FP+shape")
    ), FakeTensorMode():
        shape = exir_ops.backend.tosa.CONST_SHAPE.default([2, 3, 4])
    assert shape == [2, 3, 4]


# Test that CONCAT_SHAPE with constant shapes performs concatenation and returns a constant shape.
def test_concat_const_shapes():
    with TosaLoweringContext(
        TosaSpecification.create_from_string("TOSA-1.1+FP+shape")
    ), FakeTensorMode():
        const_shape_0 = exir_ops.backend.tosa.CONST_SHAPE.default([2, 3])
        const_shape_1 = exir_ops.backend.tosa.CONST_SHAPE.default([4])
        result = exir_ops.backend.tosa.CONCAT_SHAPE.default(
            [const_shape_0, const_shape_1]
        )
    assert result == [2, 3, 4]


# Test that CONCAT_SHAPE with symbolic shapes produces a symbolic expression concatenating the dimensions.
def test_concat_symbolic_shape():
    shape_env = ShapeEnv()
    s0 = _make_symint(shape_env, "s0", hint=2)
    s1 = _make_symint(shape_env, "s1", hint=3)

    with TosaLoweringContext(
        TosaSpecification.create_from_string("TOSA-1.1+FP+shape"), shape_env
    ), FakeTensorMode(shape_env=shape_env) as mode:
        s0_tensor = torch.empty(size=(1, 3, s0))
        s1_tensor = torch.empty(size=(1, 3, s1))

        dim_s0 = exir_ops.backend.tosa.DIM.default(mode.from_tensor(s0_tensor), axis=2)
        dim_s1 = exir_ops.backend.tosa.DIM.default(mode.from_tensor(s1_tensor), axis=2)
        result = exir_ops.backend.tosa.CONCAT_SHAPE.default([dim_s0, dim_s1])

    assert len(result) == 2
    assert _expr(result[0]) == "s0"
    assert _expr(result[1]) == "s1"


def test_concat_mixed_shape():
    shape_env = ShapeEnv()
    s0 = _make_symint(shape_env, "s0", hint=2)

    with TosaLoweringContext(
        TosaSpecification.create_from_string("TOSA-1.1+FP+shape")
    ), FakeTensorMode(shape_env=shape_env) as mode:
        const_shape = exir_ops.backend.tosa.CONST_SHAPE.default([4, 5])
        s0_tensor = torch.empty(size=(1, 3, s0))
        dim_s0 = exir_ops.backend.tosa.DIM.default(mode.from_tensor(s0_tensor), axis=2)
        result = exir_ops.backend.tosa.CONCAT_SHAPE.default([const_shape, dim_s0])

    assert len(result) == 3
    assert result[0] == 4
    assert result[1] == 5
    assert _expr(result[2]) == "s0"


# Test that CONCAT_SHAPE raises an error when given fewer than 2 shape tensors, as it requires at least 2 to
# concatenate.
def test_concat_shape_requires_arguments():
    with pytest.raises(
        TosaValueError, match="CONCAT_SHAPE expected 2 or more shape tensors"
    ):
        with TosaLoweringContext(
            TosaSpecification.create_from_string("TOSA-1.1+FP+shape")
        ), FakeTensorMode():
            exir_ops.backend.tosa.CONCAT_SHAPE.default([])


# Test ADD_SHAPE with constant values, which should perform elementwise addition and return a constant shape.
def test_add_const_shape():
    shape_env = ShapeEnv()
    with TosaLoweringContext(
        TosaSpecification.create_from_string("TOSA-1.1+FP+shape"), shape_env
    ), FakeTensorMode():
        const_0 = exir_ops.backend.tosa.CONST_SHAPE.default([2, 3])
        const_1 = exir_ops.backend.tosa.CONST_SHAPE.default([4, 5])
        result = exir_ops.backend.tosa.ADD_SHAPE.default(const_0, const_1)
    assert len(result) == 2
    assert result == [6, 8]


# Test ADD_SHAPE with symbolic values, which should produce a symbolic expression adding the two dimensions.
def test_add_symbolic_shape():
    shape_env = ShapeEnv()
    s0 = _make_symint(shape_env, "s0", hint=2)
    s1 = _make_symint(shape_env, "s1", hint=3)

    with TosaLoweringContext(
        TosaSpecification.create_from_string("TOSA-1.1+FP+shape"), shape_env
    ), FakeTensorMode(shape_env=shape_env) as mode:
        s0_tensor = torch.empty(size=(1, 3, s0))
        s1_tensor = torch.empty(size=(1, 3, s1))

        dim_s0 = exir_ops.backend.tosa.DIM.default(mode.from_tensor(s0_tensor), axis=2)
        dim_s1 = exir_ops.backend.tosa.DIM.default(mode.from_tensor(s1_tensor), axis=2)
        result = exir_ops.backend.tosa.ADD_SHAPE.default(dim_s0, dim_s1)
    assert len(result) == 1
    assert isinstance(result[0], torch.SymInt)
    assert _expr_equals(result[0], sympy.Symbol("s0") + sympy.Symbol("s1"))


def test_add_mixed_shape():
    shape_env = ShapeEnv()
    s0 = _make_symint(shape_env, "s0", hint=2)

    with TosaLoweringContext(
        TosaSpecification.create_from_string("TOSA-1.1+FP+shape"), shape_env
    ), FakeTensorMode(shape_env=shape_env) as mode:
        const_shape = exir_ops.backend.tosa.CONST_SHAPE.default([4])
        s0_tensor = torch.empty(size=(1, 3, s0))
        dim_s0 = exir_ops.backend.tosa.DIM.default(mode.from_tensor(s0_tensor), axis=2)
        result = exir_ops.backend.tosa.ADD_SHAPE.default(const_shape, dim_s0)

    assert len(result) == 1
    assert isinstance(result[0], torch.SymInt)
    assert _expr_equals(result[0], sympy.Symbol("s0") + sympy.Integer(4))


# Test SUB_SHAPE with constant values, which should perform subtraction and return a constant shape.
def test_sub_const_shape():
    shape_env = ShapeEnv()
    with TosaLoweringContext(
        TosaSpecification.create_from_string("TOSA-1.1+FP+shape"), shape_env
    ), FakeTensorMode():
        const_0 = exir_ops.backend.tosa.CONST_SHAPE.default([6, 5])
        const_1 = exir_ops.backend.tosa.CONST_SHAPE.default([2, 3])
        result = exir_ops.backend.tosa.SUB_SHAPE.default(const_0, const_1)
    assert len(result) == 2
    assert result == [4, 2]


# Test SUB_SHAPE with symbolic values, which should produce a Sub expression.
def test_sub_symbolic_shape():
    shape_env = ShapeEnv()
    s0 = _make_symint(shape_env, "s0", hint=2)
    s1 = _make_symint(shape_env, "s1", hint=3)

    with TosaLoweringContext(
        TosaSpecification.create_from_string("TOSA-1.1+FP+shape"),
        shape_env,
    ), FakeTensorMode(shape_env=shape_env) as mode:
        s0_tensor = torch.empty(size=(1, 3, s0))
        s1_tensor = torch.empty(size=(1, 3, s1))

        dim_s0 = exir_ops.backend.tosa.DIM.default(mode.from_tensor(s0_tensor), axis=2)
        dim_s1 = exir_ops.backend.tosa.DIM.default(mode.from_tensor(s1_tensor), axis=2)
        result = exir_ops.backend.tosa.SUB_SHAPE.default(dim_s0, dim_s1)
    assert len(result) == 1
    assert isinstance(result[0], torch.SymInt)
    assert _expr_equals(result[0], sympy.Symbol("s0") - sympy.Symbol("s1"))


def test_sub_mixed_shape():
    shape_env = ShapeEnv()
    s0 = _make_symint(shape_env, "s0", hint=3)

    with TosaLoweringContext(
        TosaSpecification.create_from_string("TOSA-1.1+FP+shape"),
        shape_env,
    ), FakeTensorMode(shape_env=shape_env) as mode:
        const_shape = exir_ops.backend.tosa.CONST_SHAPE.default([6])
        s0_tensor = torch.empty(size=(1, 3, s0))
        dim_s0 = exir_ops.backend.tosa.DIM.default(mode.from_tensor(s0_tensor), axis=2)
        result = exir_ops.backend.tosa.SUB_SHAPE.default(const_shape, dim_s0)

    assert len(result) == 1
    assert isinstance(result[0], torch.SymInt)
    assert _expr_equals(result[0], sympy.Integer(6) - sympy.Symbol("s0"))


# Test MUL_SHAPE with constant values, which should perform multiplication and return a constant shape.
def test_mul_const_shape():
    shape_env = ShapeEnv()
    with TosaLoweringContext(
        TosaSpecification.create_from_string("TOSA-1.1+FP+shape"), shape_env
    ), FakeTensorMode():
        const_0 = exir_ops.backend.tosa.CONST_SHAPE.default([2, 3])
        const_1 = exir_ops.backend.tosa.CONST_SHAPE.default([4, 5])
        result = exir_ops.backend.tosa.MUL_SHAPE.default(const_0, const_1)
    assert len(result) == 2
    assert result == [8, 15]


# Test MUL_SHAPE with symbolic values, which should produce a Mul expression.
def test_mul_symbolic_shape():
    shape_env = ShapeEnv()
    s0 = _make_symint(shape_env, "s0", hint=2)
    s1 = _make_symint(shape_env, "s1", hint=3)

    with TosaLoweringContext(
        TosaSpecification.create_from_string("TOSA-1.1+FP+shape"), shape_env
    ), FakeTensorMode(shape_env=shape_env) as mode:
        s0_tensor = torch.empty(size=(1, 3, s0))
        s1_tensor = torch.empty(size=(1, 3, s1))

        dim_s0 = exir_ops.backend.tosa.DIM.default(mode.from_tensor(s0_tensor), axis=2)
        dim_s1 = exir_ops.backend.tosa.DIM.default(mode.from_tensor(s1_tensor), axis=2)
        result = exir_ops.backend.tosa.MUL_SHAPE.default(dim_s0, dim_s1)
    assert len(result) == 1
    assert isinstance(result[0], torch.SymInt)
    assert _expr_equals(result[0], sympy.Symbol("s0") * sympy.Symbol("s1"))


def test_mul_mixed_shape():
    shape_env = ShapeEnv()
    s0 = _make_symint(shape_env, "s0", hint=4)

    with TosaLoweringContext(
        TosaSpecification.create_from_string("TOSA-1.1+FP+shape"), shape_env
    ), FakeTensorMode(shape_env=shape_env) as mode:
        const_shape = exir_ops.backend.tosa.CONST_SHAPE.default([3])
        s0_tensor = torch.empty(size=(1, 3, s0))
        dim_s0 = exir_ops.backend.tosa.DIM.default(mode.from_tensor(s0_tensor), axis=2)
        result = exir_ops.backend.tosa.MUL_SHAPE.default(const_shape, dim_s0)

    assert len(result) == 1
    assert isinstance(result[0], torch.SymInt)
    assert _expr_equals(result[0], sympy.Integer(3) * sympy.Symbol("s0"))


# Test MOD_SHAPE with constant values, which should perform modulo and return a constant shape.
def test_mod_const_shape_no_target():
    shape_env = ShapeEnv()
    with TosaLoweringContext(
        TosaSpecification.create_from_string("TOSA-1.1+FP+shape"), shape_env
    ), FakeTensorMode():
        const_0 = exir_ops.backend.tosa.CONST_SHAPE.default([8, 21])
        const_1 = exir_ops.backend.tosa.CONST_SHAPE.default([3, 5])
        result = exir_ops.backend.tosa.MOD_SHAPE.default(const_0, const_1)
    assert len(result) == 2
    assert result == [2, 1]


# Test MOD_SHAPE with symbolic values, which should produce a Mod expression.
def test_mod_symbolic_shape_no_target():
    shape_env = ShapeEnv()
    s0 = _make_symint(shape_env, "s0", hint=8)
    s1 = _make_symint(shape_env, "s1", hint=3)

    with TosaLoweringContext(
        TosaSpecification.create_from_string("TOSA-1.1+FP+shape"), shape_env
    ), FakeTensorMode(shape_env=shape_env) as mode:
        s0_tensor = torch.empty(size=(1, 3, s0))
        s1_tensor = torch.empty(size=(1, 3, s1))
        dim_s0 = exir_ops.backend.tosa.DIM.default(mode.from_tensor(s0_tensor), axis=2)
        dim_s1 = exir_ops.backend.tosa.DIM.default(mode.from_tensor(s1_tensor), axis=2)
        result = exir_ops.backend.tosa.MOD_SHAPE.default(dim_s0, dim_s1)
    assert len(result) == 1
    assert isinstance(result[0], torch.SymInt)
    assert _expr_equals(result[0], sympy.Mod(sympy.Symbol("s0"), sympy.Symbol("s1")))


def test_mod_mixed_shape_no_target():
    shape_env = ShapeEnv()
    s0 = _make_symint(shape_env, "s0", hint=4)

    with TosaLoweringContext(
        TosaSpecification.create_from_string("TOSA-1.1+FP+shape"), shape_env
    ), FakeTensorMode(shape_env=shape_env) as mode:
        const_shape = exir_ops.backend.tosa.CONST_SHAPE.default([8])
        s0_tensor = torch.empty(size=(1, 3, s0))
        dim_s0 = exir_ops.backend.tosa.DIM.default(mode.from_tensor(s0_tensor), axis=2)
        result = exir_ops.backend.tosa.MOD_SHAPE.default(const_shape, dim_s0)

    assert len(result) == 1
    assert isinstance(result[0], torch.SymInt)
    assert _expr_equals(result[0], sympy.Mod(sympy.Integer(8), sympy.Symbol("s0")))


# Test DIV_FLOOR_SHAPE with constant values, which should perform floor division and return a constant shape.
def test_div_floor_const_shape():
    shape_env = ShapeEnv()
    with TosaLoweringContext(
        TosaSpecification.create_from_string("TOSA-1.1+FP+shape"), shape_env
    ), FakeTensorMode():
        const_0 = exir_ops.backend.tosa.CONST_SHAPE.default([8, 21])
        const_1 = exir_ops.backend.tosa.CONST_SHAPE.default([2, 4])
        result = exir_ops.backend.tosa.DIV_FLOOR_SHAPE.default(const_0, const_1)
    assert len(result) == 2
    assert result == [4, 5]


# Test DIV_FLOOR_SHAPE with symbolic values, which should produce a FloorDiv expression.
def test_div_floor_symbolic_shape():
    shape_env = ShapeEnv()
    s0 = _make_symint(shape_env, "s0", hint=8)
    s1 = _make_symint(shape_env, "s1", hint=3)

    with TosaLoweringContext(
        TosaSpecification.create_from_string("TOSA-1.1+FP+shape"), shape_env
    ), FakeTensorMode(shape_env=shape_env) as mode:
        s0_tensor = torch.empty(size=(1, 3, s0))
        s1_tensor = torch.empty(size=(1, 3, s1))
        dim_s0 = exir_ops.backend.tosa.DIM.default(mode.from_tensor(s0_tensor), axis=2)
        dim_s1 = exir_ops.backend.tosa.DIM.default(mode.from_tensor(s1_tensor), axis=2)
        result = exir_ops.backend.tosa.DIV_FLOOR_SHAPE.default(dim_s0, dim_s1)
    assert len(result) == 1
    assert isinstance(result[0], torch.SymInt)
    assert _expr_equals(result[0], sympy.sympify("(s0//s1)"))


def test_div_floor_mixed_shape():
    shape_env = ShapeEnv()
    s0 = _make_symint(shape_env, "s0", hint=4)

    with TosaLoweringContext(
        TosaSpecification.create_from_string("TOSA-1.1+FP+shape"), shape_env
    ), FakeTensorMode(shape_env=shape_env) as mode:
        const_shape = exir_ops.backend.tosa.CONST_SHAPE.default([8])
        s0_tensor = torch.empty(size=(1, 3, s0))
        dim_s0 = exir_ops.backend.tosa.DIM.default(mode.from_tensor(s0_tensor), axis=2)
        result = exir_ops.backend.tosa.DIV_FLOOR_SHAPE.default(const_shape, dim_s0)

    assert len(result) == 1
    assert isinstance(result[0], torch.SymInt)
    assert _expr_equals(result[0], sympy.sympify("8//s0"))
