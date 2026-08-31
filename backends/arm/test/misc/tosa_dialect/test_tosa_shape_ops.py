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
from executorch.backends.arm.tosa.dialect.ops.shape_ops import (
    ASSERT_EQUAL_SHAPE as assert_equal_shape_impl,
)
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
    assert isinstance(symint, torch.SymInt)
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


# Test that DIM rejects unsupported tensor dtypes for the active TOSA profile and extensions.
def test_dim_rejects_unsupported_dtype() -> None:
    with TosaLoweringContext(
        TosaSpecification.create_from_string("TOSA-1.1+FP+shape")
    ), FakeTensorMode() as mode:
        x = mode.from_tensor(torch.empty((2, 3), dtype=torch.float64))
        with pytest.raises(TosaValueError, match="Unsupported dtype"):
            exir_ops.backend.tosa.DIM.default(x, axis=1)


# Test that DIM rejects known non-positive dimensions, as required by the TOSA specification.
def test_dim_rejects_zero_dimension() -> None:
    with TosaLoweringContext(
        TosaSpecification.create_from_string("TOSA-1.1+FP+shape")
    ), FakeTensorMode() as mode:
        x = mode.from_tensor(torch.empty((2, 0, 3), dtype=torch.float32))
        with pytest.raises(TosaValueError, match=r"shape\[axis\] > 0"):
            exir_ops.backend.tosa.DIM.default(x, axis=1)


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


# Test that CONCAT_SHAPE raises an error when given no shape tensors.
def test_concat_shape_requires_arguments():
    with pytest.raises(TosaValueError, match="requires at least one shape tensor"):
        with TosaLoweringContext(
            TosaSpecification.create_from_string("TOSA-1.1+FP+shape")
        ), FakeTensorMode():
            exir_ops.backend.tosa.CONCAT_SHAPE.default([])


# Test that CONCAT_SHAPE allows a single input shape.
def test_concat_shape_allows_single_argument():
    with TosaLoweringContext(
        TosaSpecification.create_from_string("TOSA-1.1+FP+shape")
    ), FakeTensorMode():
        result = exir_ops.backend.tosa.CONCAT_SHAPE.default([[2, 3]])

    assert result == [2, 3]


# Test that CONCAT_SHAPE rejects empty member shapes.
def test_concat_shape_rejects_empty_member_shape():
    with TosaLoweringContext(
        TosaSpecification.create_from_string("TOSA-1.1+FP+shape")
    ), FakeTensorMode():
        with pytest.raises(TosaValueError, match="disallows empty input shapes"):
            exir_ops.backend.tosa.CONCAT_SHAPE.default([[2], []])


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


# Test SLICE_SHAPE with a constant input shape.
def test_slice_shape_constants() -> None:
    shape_env = ShapeEnv()
    with TosaLoweringContext(
        TosaSpecification.create_from_string("TOSA-1.1+FP+shape"),
        shape_env,
    ), FakeTensorMode(shape_env=shape_env):
        input_shape = exir_ops.backend.tosa.CONST_SHAPE.default([8, 16, 7])
        assert exir_ops.backend.tosa.SLICE_SHAPE.default(input_shape, [1], [2]) == [
            16,
            7,
        ]


# Test SLICE_SHAPE rejects invalid start and size values.
def test_slice_shape_rejects_invalid_bounds() -> None:
    with TosaLoweringContext(
        TosaSpecification.create_from_string("TOSA-1.1+FP+shape")
    ), FakeTensorMode():
        input_shape = [8, 16, 7]
        with pytest.raises(TosaValueError, match="start >= 0"):
            exir_ops.backend.tosa.SLICE_SHAPE.default(input_shape, [-1], [1])
        with pytest.raises(TosaValueError, match="size > 0"):
            exir_ops.backend.tosa.SLICE_SHAPE.default(input_shape, [0], [0])
        with pytest.raises(TosaValueError, match="within input bounds"):
            exir_ops.backend.tosa.SLICE_SHAPE.default(input_shape, [2], [2])


# Test SLICE_SHAPE supports bounded symbolic start values when size is known.
def test_slice_shape_bounded_symbolic_start() -> None:
    shape_env = ShapeEnv()
    s0 = _make_symint(shape_env, "s0", hint=0, min=0, max=1)
    d0 = _make_symint(shape_env, "d0", hint=8)
    d1 = _make_symint(shape_env, "d1", hint=16)
    d2 = _make_symint(shape_env, "d2", hint=7)

    with TosaLoweringContext(
        TosaSpecification.create_from_string("TOSA-1.1+FP+shape"),
        shape_env,
    ), FakeTensorMode(shape_env=shape_env):
        result = exir_ops.backend.tosa.SLICE_SHAPE.default([d0, d1, d2], [s0], [2])

    assert len(result) == 2
    assert _expr_equals(
        result[0],
        sympy.Piecewise(
            (sympy.Symbol("d0"), sympy.Eq(sympy.Symbol("s0"), 0)),
            (sympy.Symbol("d1"), sympy.Eq(sympy.Symbol("s0"), 1)),
        ),
    )
    assert _expr_equals(
        result[1],
        sympy.Piecewise(
            (sympy.Symbol("d1"), sympy.Eq(sympy.Symbol("s0"), 0)),
            (sympy.Symbol("d2"), sympy.Eq(sympy.Symbol("s0"), 1)),
        ),
    )


# Test SLICE_SHAPE accepts symbolic sizes that are provably singleton.
def test_slice_shape_singleton_symbolic_size() -> None:
    shape_env = ShapeEnv()
    size = _make_symint(shape_env, "size", hint=2, min=2, max=2)

    with TosaLoweringContext(
        TosaSpecification.create_from_string("TOSA-1.1+FP+shape"),
        shape_env,
    ), FakeTensorMode(shape_env=shape_env):
        result = exir_ops.backend.tosa.SLICE_SHAPE.default([8, 16, 7], [1], [size])

    assert result == [16, 7]


# Test SLICE_SHAPE rejects bounded symbolic starts with any out-of-bounds value.
def test_slice_shape_rejects_out_of_bounds_symbolic_start() -> None:
    shape_env = ShapeEnv()
    start = _make_symint(shape_env, "start", hint=1, min=1, max=2)

    with TosaLoweringContext(
        TosaSpecification.create_from_string("TOSA-1.1+FP+shape"),
        shape_env,
    ), FakeTensorMode(shape_env=shape_env):
        with pytest.raises(TosaValueError, match="within input bounds"):
            exir_ops.backend.tosa.SLICE_SHAPE.default([8, 16, 7], [start], [2])


# Test EXP2_SHAPE with constant values.
def test_exp2_shape_constants() -> None:
    shape_env = ShapeEnv()
    with TosaLoweringContext(
        TosaSpecification.create_from_string("TOSA-1.1+FP+shape"),
        shape_env,
    ), FakeTensorMode(shape_env=shape_env):
        assert exir_ops.backend.tosa.EXP2_SHAPE.default([0, 3, 4]) == [1, 8, 16]


# Test EXP2_SHAPE preserves symbolic expressions.
def test_exp2_shape_symbolic() -> None:
    shape_env = ShapeEnv()
    s0 = _make_symint(shape_env, "s0", hint=3, min=0, max=6)

    with TosaLoweringContext(
        TosaSpecification.create_from_string("TOSA-1.1+FP+shape"),
        shape_env,
    ), FakeTensorMode(shape_env=shape_env):
        result = exir_ops.backend.tosa.EXP2_SHAPE.default([s0])

    assert isinstance(result[0], torch.SymInt)
    assert _expr_equals(result[0], sympy.Integer(2) ** sympy.Symbol("s0"))


# Test that EXP2_SHAPE enforces the TOSA MAX_LOG2_SIZE bound.
def test_exp2_shape_rejects_max_log2_size() -> None:
    with TosaLoweringContext(
        TosaSpecification.create_from_string("TOSA-1.1+FP+shape")
    ), FakeTensorMode():
        with pytest.raises(TosaValueError, match=r"input < 63"):
            exir_ops.backend.tosa.EXP2_SHAPE.default([63])


# Test that EXP2_SHAPE uses the stricter 8k-level MAX_LOG2_SIZE bound.
def test_exp2_shape_rejects_max_log2_size_at_8k_level() -> None:
    with TosaLoweringContext(
        TosaSpecification.create_from_string("TOSA-1.1+FP+shape+8k")
    ), FakeTensorMode():
        with pytest.raises(TosaValueError, match=r"input < 31"):
            exir_ops.backend.tosa.EXP2_SHAPE.default([31])


# Test LOG2_CEIL_SHAPE with constant values.
def test_log2_ceil_shape_constants() -> None:
    shape_env = ShapeEnv()
    with TosaLoweringContext(
        TosaSpecification.create_from_string("TOSA-1.1+FP+shape"),
        shape_env,
    ), FakeTensorMode(shape_env=shape_env):
        assert exir_ops.backend.tosa.LOG2_CEIL_SHAPE.default([1, 3, 8]) == [0, 2, 3]


# Test LOG2_CEIL_SHAPE rejects non-positive inputs.
def test_log2_ceil_shape_rejects_zero_input() -> None:
    with TosaLoweringContext(
        TosaSpecification.create_from_string("TOSA-1.1+FP+shape")
    ), FakeTensorMode():
        with pytest.raises(TosaValueError, match=r"input > 0"):
            exir_ops.backend.tosa.LOG2_CEIL_SHAPE.default([0])


# Test LOG2_FLOOR_SHAPE with constant values.
def test_log2_floor_shape_constants() -> None:
    shape_env = ShapeEnv()
    with TosaLoweringContext(
        TosaSpecification.create_from_string("TOSA-1.1+FP+shape"),
        shape_env,
    ), FakeTensorMode(shape_env=shape_env):
        assert exir_ops.backend.tosa.LOG2_FLOOR_SHAPE.default([1, 3, 8]) == [0, 1, 3]


# Test LOG2_FLOOR_SHAPE rejects non-positive inputs.
def test_log2_floor_shape_rejects_zero_input() -> None:
    with TosaLoweringContext(
        TosaSpecification.create_from_string("TOSA-1.1+FP+shape")
    ), FakeTensorMode():
        with pytest.raises(TosaValueError, match=r"input > 0"):
            exir_ops.backend.tosa.LOG2_FLOOR_SHAPE.default([0])


# Test MAX_SHAPE with constant values.
def test_max_shape_constants() -> None:
    shape_env = ShapeEnv()
    with TosaLoweringContext(
        TosaSpecification.create_from_string("TOSA-1.1+FP+shape"),
        shape_env,
    ), FakeTensorMode(shape_env=shape_env):
        assert exir_ops.backend.tosa.MAX_SHAPE.default([2, 9], [4, 3]) == [4, 9]


# Test MAX_SHAPE with symbolic values.
def test_max_shape_symbolic() -> None:
    shape_env = ShapeEnv()
    s0 = _make_symint(shape_env, "s0", 4)
    s1 = _make_symint(shape_env, "s1", 8)

    with TosaLoweringContext(
        TosaSpecification.create_from_string("TOSA-1.1+FP+shape"),
        shape_env,
    ), FakeTensorMode(shape_env=shape_env):
        max_shape = exir_ops.backend.tosa.MAX_SHAPE.default([s0], [s1])

    assert _expr(max_shape[0]) == "Max(s0, s1)"


# Test MAX_SHAPE with mixed constant and symbolic values.
def test_max_shape_mixed() -> None:
    shape_env = ShapeEnv()
    s0 = _make_symint(shape_env, "s0", 4)

    with TosaLoweringContext(
        TosaSpecification.create_from_string("TOSA-1.1+FP+shape"),
        shape_env,
    ), FakeTensorMode(shape_env=shape_env):
        max_shape = exir_ops.backend.tosa.MAX_SHAPE.default([s0], [5])

    assert _expr_equals(max_shape[0], sympy.Max(sympy.Symbol("s0"), sympy.Integer(5)))


# Test MIN_SHAPE with constant values.
def test_min_shape_constants() -> None:
    shape_env = ShapeEnv()
    with TosaLoweringContext(
        TosaSpecification.create_from_string("TOSA-1.1+FP+shape"),
        shape_env,
    ), FakeTensorMode(shape_env=shape_env):
        assert exir_ops.backend.tosa.MIN_SHAPE.default([2, 9], [4, 3]) == [2, 3]


# Test MIN_SHAPE with symbolic values.
def test_min_shape_symbolic() -> None:
    shape_env = ShapeEnv()
    s0 = _make_symint(shape_env, "s0", 4)
    s1 = _make_symint(shape_env, "s1", 8)

    with TosaLoweringContext(
        TosaSpecification.create_from_string("TOSA-1.1+FP+shape"),
        shape_env,
    ), FakeTensorMode(shape_env=shape_env):
        min_shape = exir_ops.backend.tosa.MIN_SHAPE.default([s0], [s1])

    assert _expr(min_shape[0]) == "Min(s0, s1)"


# Test MIN_SHAPE with mixed constant and symbolic values.
def test_min_shape_mixed() -> None:
    shape_env = ShapeEnv()
    s0 = _make_symint(shape_env, "s0", 4)

    with TosaLoweringContext(
        TosaSpecification.create_from_string("TOSA-1.1+FP+shape"),
        shape_env,
    ), FakeTensorMode(shape_env=shape_env):
        min_shape = exir_ops.backend.tosa.MIN_SHAPE.default([s0], [5])

    assert _expr_equals(min_shape[0], sympy.Min(sympy.Symbol("s0"), sympy.Integer(5)))


# Test DIV_CEIL_SHAPE with constant values.
def test_div_ceil_shape_constants() -> None:
    shape_env = ShapeEnv()
    with TosaLoweringContext(
        TosaSpecification.create_from_string("TOSA-1.1+FP+shape"),
        shape_env,
    ), FakeTensorMode(shape_env=shape_env):
        assert exir_ops.backend.tosa.DIV_CEIL_SHAPE.default([9, 16], [4, 8]) == [3, 2]


# Test DIV_CEIL_SHAPE preserves symbolic expressions.
def test_div_ceil_shape_symbolic() -> None:
    shape_env = ShapeEnv()
    s0 = _make_symint(shape_env, "s0", hint=8)
    s1 = _make_symint(shape_env, "s1", hint=3)

    with TosaLoweringContext(
        TosaSpecification.create_from_string("TOSA-1.1+FP+shape"),
        shape_env,
    ), FakeTensorMode(shape_env=shape_env) as mode:
        s0_tensor = torch.empty(size=(1, 3, s0))
        s1_tensor = torch.empty(size=(1, 3, s1))
        dim_s0 = exir_ops.backend.tosa.DIM.default(mode.from_tensor(s0_tensor), axis=2)
        dim_s1 = exir_ops.backend.tosa.DIM.default(mode.from_tensor(s1_tensor), axis=2)
        result = exir_ops.backend.tosa.DIV_CEIL_SHAPE.default(dim_s0, dim_s1)

    assert len(result) == 1
    assert isinstance(result[0], torch.SymInt)
    assert _expr_equals(
        result[0],
        sympy.floor(
            (sympy.Symbol("s0") + sympy.Symbol("s1") - sympy.Integer(1))
            / sympy.Symbol("s1")
        ),
    )


# Test DIV_CEIL_SHAPE with mixed constant and symbolic values.
def test_div_ceil_shape_mixed() -> None:
    shape_env = ShapeEnv()
    s0 = _make_symint(shape_env, "s0", hint=4)

    with TosaLoweringContext(
        TosaSpecification.create_from_string("TOSA-1.1+FP+shape"),
        shape_env,
    ), FakeTensorMode(shape_env=shape_env) as mode:
        const_shape = exir_ops.backend.tosa.CONST_SHAPE.default([8])
        s0_tensor = torch.empty(size=(1, 3, s0))
        dim_s0 = exir_ops.backend.tosa.DIM.default(mode.from_tensor(s0_tensor), axis=2)
        result = exir_ops.backend.tosa.DIV_CEIL_SHAPE.default(const_shape, dim_s0)

    assert len(result) == 1
    assert isinstance(result[0], torch.SymInt)
    assert _expr_equals(
        result[0],
        sympy.floor(
            (sympy.Integer(8) + sympy.Symbol("s0") - sympy.Integer(1))
            / sympy.Symbol("s0")
        ),
    )


# Test DIV_CEIL_SHAPE rejects invalid operands.
def test_div_ceil_shape_rejects_invalid_operands() -> None:
    with TosaLoweringContext(
        TosaSpecification.create_from_string("TOSA-1.1+FP+shape")
    ), FakeTensorMode():
        with pytest.raises(TosaValueError, match=r"input1 >= 0"):
            exir_ops.backend.tosa.DIV_CEIL_SHAPE.default([-1], [4])
        with pytest.raises(TosaValueError, match=r"input2 > 0"):
            exir_ops.backend.tosa.DIV_CEIL_SHAPE.default([8], [0])


# Test ASSERT_EQUAL_SHAPE accepts same-rank shapes without comparing values.
def test_assert_equal_shape_allows_same_rank() -> None:
    with TosaLoweringContext(
        TosaSpecification.create_from_string("TOSA-1.1+FP+shape")
    ), FakeTensorMode():
        result = assert_equal_shape_impl(
            [4, 1],
            [3, 7],
            allow_broadcast=False,
        )

    assert result is None


# Test ASSERT_EQUAL_SHAPE accepts symbolic same-rank shapes without SymBool checks.
def test_assert_equal_shape_allows_symbolic_same_rank() -> None:
    shape_env = ShapeEnv()
    s0 = _make_symint(shape_env, "s0", hint=2, min=2, max=4)
    s1 = _make_symint(shape_env, "s1", hint=5, min=5, max=8)

    with TosaLoweringContext(
        TosaSpecification.create_from_string("TOSA-1.1+FP+shape"),
        shape_env,
    ), FakeTensorMode(shape_env=shape_env):
        result = assert_equal_shape_impl(
            [s0, 1],
            [s1, 7],
            allow_broadcast=True,
        )

    assert result is None


# Test ASSERT_EQUAL_SHAPE rejects mismatched ranks.
def test_assert_equal_shape_rejects_rank_mismatch() -> None:
    with TosaLoweringContext(
        TosaSpecification.create_from_string("TOSA-1.1+FP+shape")
    ), FakeTensorMode():
        with pytest.raises(TosaValueError, match="requires equal lengths"):
            assert_equal_shape_impl(
                [4, 1],
                [4, 1, 7],
                allow_broadcast=True,
            )


def test_const_shape_allows_non_shape_specs() -> None:
    with TosaLoweringContext(
        TosaSpecification.create_from_string("TOSA-1.0+FP")
    ), FakeTensorMode():
        assert exir_ops.backend.tosa.CONST_SHAPE.default([2, 3]) == [2, 3]


def test_slice_shape_requires_shape_extension() -> None:
    with TosaLoweringContext(
        TosaSpecification.create_from_string("TOSA-1.0+FP")
    ), FakeTensorMode():
        with pytest.raises(TosaValueError, match="shape extension"):
            exir_ops.backend.tosa.SLICE_SHAPE.default([2, 3], [0], [1])
