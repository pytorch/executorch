# Copyright 2026 Arm Limited and/or its affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import pytest
import torch
from executorch.backends.arm._passes.cast_int_comparison_inputs_pass import (
    CastIntComparisonInputsPass,
)
from executorch.backends.arm.tosa.specification import (
    TosaLoweringContext,
    TosaSpecification,
)
from executorch.exir import to_edge
from executorch.exir.dialects._ops import ops as exir_ops


class Equal(torch.nn.Module):
    def forward(self, x, y):
        return x == y


class Greater(torch.nn.Module):
    def forward(self, x, y):
        return x > y


@pytest.mark.parametrize(
    ("profile", "dtypes", "expected_casts"),
    (
        ("INT", (torch.int8, torch.int16), [torch.int32, torch.int32]),
        ("INT", (torch.int32, torch.int16), [torch.int32]),
        ("INT", (torch.bool, torch.int32), [torch.int32]),
        ("FP", (torch.int8, torch.int16), [torch.float32, torch.float32]),
        ("FP+INT", (torch.int16, torch.float32), [torch.float32]),
        ("FP", (torch.int32, torch.float32), [torch.float32]),
        ("FP+INT", (torch.int32, torch.float32), [torch.float32]),
        ("FP+INT", (torch.float32, torch.int32), [torch.float32]),
        ("FP+INT", (torch.int8, torch.float16), [torch.float16]),
        ("FP", (torch.int16, torch.float16), [torch.float16]),
        ("FP+INT", (torch.int16, torch.float16), [torch.float16]),
        ("FP+bf16", (torch.int16, torch.bfloat16), [torch.bfloat16]),
        ("FP+INT+bf16", (torch.int16, torch.bfloat16), [torch.bfloat16]),
        ("FP+bf16", (torch.int8, torch.bfloat16), [torch.bfloat16]),
        ("FP", (torch.bool, torch.bool), []),
    ),
)
def test_comparison_casts_follow_profile(profile, dtypes, expected_casts):
    inputs = tuple(torch.tensor([0, 1], dtype=dtype) for dtype in dtypes)
    graph_module = (
        to_edge(torch.export.export(Equal(), inputs)).exported_program().graph_module
    )

    with TosaLoweringContext(
        TosaSpecification.create_from_string(f"TOSA-1.0+{profile}")
    ):
        result = CastIntComparisonInputsPass()(graph_module)

    cast_target = exir_ops.edge.dim_order_ops._to_dim_order_copy.default
    casts = [
        node.kwargs["dtype"]
        for node in result.graph_module.graph.nodes
        if node.target == cast_target
    ]
    assert casts == expected_casts


@pytest.mark.parametrize(
    ("integer_dtype", "floating_dtype", "integer_value", "rounded_value"),
    (
        (torch.int16, torch.float16, 2049, 2048),
        (torch.int16, torch.bfloat16, 257, 256),
        (torch.int32, torch.float32, 2**24 + 1, 2**24),
        (torch.int32, torch.float16, 2049, 2048),
        (torch.int32, torch.bfloat16, 257, 256),
    ),
)
@pytest.mark.parametrize("module", (Equal(), Greater()))
def test_comparison_casts_match_torch_rounding(
    integer_dtype, floating_dtype, integer_value, rounded_value, module
):
    inputs = (
        torch.tensor([integer_value], dtype=integer_dtype),
        torch.tensor([rounded_value], dtype=floating_dtype),
    )
    graph_module = (
        to_edge(torch.export.export(module, inputs)).exported_program().graph_module
    )

    spec = "TOSA-1.0+FP+bf16" if floating_dtype == torch.bfloat16 else "TOSA-1.0+FP"
    with TosaLoweringContext(TosaSpecification.create_from_string(spec)):
        result = CastIntComparisonInputsPass()(graph_module)

    torch.testing.assert_close(result.graph_module(*inputs)[0], module(*inputs))


@pytest.mark.parametrize(
    ("vector_dtype", "scalar_value", "rounded_value"),
    (
        (torch.float16, 2049, 2048),
        (torch.bfloat16, 257, 256),
    ),
)
def test_comparison_uses_result_type_for_zero_dim_tensor(
    vector_dtype, scalar_value, rounded_value
):
    inputs = (
        torch.tensor([rounded_value], dtype=vector_dtype),
        torch.tensor(scalar_value, dtype=torch.float32),
    )
    graph_module = (
        to_edge(torch.export.export(Equal(), inputs)).exported_program().graph_module
    )

    spec = "TOSA-1.0+FP+bf16" if vector_dtype == torch.bfloat16 else "TOSA-1.0+FP"
    with TosaLoweringContext(TosaSpecification.create_from_string(spec)):
        result = CastIntComparisonInputsPass()(graph_module)

    cast_target = exir_ops.edge.dim_order_ops._to_dim_order_copy.default
    casts = [
        node.kwargs["dtype"]
        for node in result.graph_module.graph.nodes
        if node.target == cast_target
    ]
    assert casts == [vector_dtype]
    torch.testing.assert_close(result.graph_module(*inputs)[0], Equal()(*inputs))


def test_bool_greater_casts_to_int32():
    inputs = (
        torch.tensor([False, False, True, True]),
        torch.tensor([False, True, False, True]),
    )
    graph_module = (
        to_edge(torch.export.export(Greater(), inputs)).exported_program().graph_module
    )

    with TosaLoweringContext(TosaSpecification.create_from_string("TOSA-1.0+FP+INT")):
        result = CastIntComparisonInputsPass()(graph_module)

    cast_target = exir_ops.edge.dim_order_ops._to_dim_order_copy.default
    casts = [
        node.kwargs["dtype"]
        for node in result.graph_module.graph.nodes
        if node.target == cast_target
    ]
    assert casts == [torch.int32, torch.int32]
    torch.testing.assert_close(result.graph_module(*inputs)[0], Greater()(*inputs))


@pytest.mark.parametrize(
    ("module", "spec", "dtypes", "message"),
    (
        (
            Equal(),
            "TOSA-1.0+FP+INT",
            (torch.bool, torch.float32),
            "cannot cast",
        ),
        (
            Equal(),
            "TOSA-1.0+FP",
            (torch.bool, torch.float32),
            "cannot cast",
        ),
        (
            Equal(),
            "TOSA-1.0+FP",
            (torch.bool, torch.int8),
            "cannot cast",
        ),
        (
            Equal(),
            "TOSA-1.1+INT+int64",
            (torch.bool, torch.int64),
            "cannot cast",
        ),
        (Equal(), "TOSA-1.0+INT", (torch.int64, torch.int32), "cannot cast"),
        (Equal(), "TOSA-1.0+FP", (torch.int64, torch.float32), "cannot cast"),
        (Equal(), "TOSA-1.0+FP", (torch.int32, torch.int32), "cannot cast"),
        (Equal(), "TOSA-1.0+FP+INT", (torch.uint8, torch.float32), "cannot cast"),
        (Equal(), "TOSA-1.0+FP+INT", (torch.uint8, torch.int32), "cannot cast"),
        (Equal(), "TOSA-1.1+INT+int64", (torch.int64, torch.int64), "cannot cast"),
        (Equal(), "TOSA-1.0+FP", (torch.complex64, torch.complex64), "cannot cast"),
        (Equal(), "TOSA-1.0+FP+INT", (torch.complex64, torch.float32), "cannot cast"),
        (Equal(), "TOSA-1.0+FP", (torch.float64, torch.float32), "cannot cast"),
    ),
)
def test_comparison_rejects_unsupported_inputs(module, spec, dtypes, message):
    inputs = tuple(torch.tensor([0, 1], dtype=dtype) for dtype in dtypes)
    graph_module = (
        to_edge(torch.export.export(module, inputs)).exported_program().graph_module
    )

    with TosaLoweringContext(TosaSpecification.create_from_string(spec)):
        with pytest.raises(ValueError, match=message):
            CastIntComparisonInputsPass()(graph_module)
