# Copyright 2026 Arm Limited and/or its affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import pytest
import torch

from executorch.backends.arm._passes import DecomposeRollPass
from executorch.backends.arm._passes.decompose_roll_pass import can_decompose_roll
from executorch.backends.arm.tosa.compile_spec import TosaCompileSpec
from executorch.backends.arm.util._factory import create_partitioner
from executorch.backends.arm.vgf.compile_spec import VgfCompileSpec
from executorch.exir import EdgeCompileConfig, to_edge
from executorch.exir.dialects._ops import ops as exir_ops


class Roll(torch.nn.Module):
    def __init__(self, shifts: tuple[int, ...], dims: tuple[int, ...]) -> None:
        super().__init__()
        self.shifts = shifts
        self.dims = dims

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.roll(x, self.shifts, self.dims)


roll_cases = (
    ((2, 5), (1,), (1,), 1),
    ((2, 3, 5), (-2,), (-1,), 1),
    ((2, 5), (12,), (1,), 1),
    ((1, 8, 8, 4), (-2, -2), (1, 2), 2),
    ((1, 5, 3), (1, 2), (1, 1), 2),
    ((5, 4), (5, 1), (0, 1), 1),
)


@pytest.mark.parametrize(
    "shape,shifts,dims,expected_cats",
    roll_cases,
    ids=(
        "positive",
        "negative_shift_and_dim",
        "oversized_shift",
        "multiple_dims",
        "repeated_dim",
        "mixed_zero_shift",
    ),
)
def test_decompose_roll(
    shape: tuple[int, ...],
    shifts: tuple[int, ...],
    dims: tuple[int, ...],
    expected_cats: int,
) -> None:
    model = Roll(shifts, dims)
    inputs = (torch.randn(shape),)
    eager_output = model(*inputs)
    edge = to_edge(
        torch.export.export(model, inputs, strict=True),
        compile_config=EdgeCompileConfig(
            _check_ir_validity=False,
            preserve_ops=[torch.ops.aten.roll.default],
        ),
    )

    edge = edge.transform([DecomposeRollPass()])
    graph = edge.exported_program().graph
    targets = [node.target for node in graph.nodes if node.op == "call_function"]

    assert exir_ops.edge.aten.roll.default not in targets
    assert targets.count(exir_ops.edge.aten.slice_copy.Tensor) == 2 * expected_cats
    assert targets.count(exir_ops.edge.aten.cat.default) == expected_cats
    assert torch.equal(edge.exported_program().module()(*inputs), eager_output)


@pytest.mark.parametrize(
    "shape,shifts,dims",
    (
        ((2, 4), (1,), ()),
        ((2, 4), (4,), (1,)),
        ((2, 0), (1,), (1,)),
        ((2, 4), (1, 2), (1,)),
        ((2, 4), (1,), (2,)),
    ),
    ids=("flat", "no_op", "zero_size", "mismatched_args", "invalid_dim"),
)
def test_roll_decomposition_guards(
    shape: tuple[int, ...], shifts: tuple[int, ...], dims: tuple[int, ...]
) -> None:
    assert not can_decompose_roll(shape, shifts, dims)


@pytest.mark.parametrize(
    "dtype,compile_spec",
    (
        (torch.float32, TosaCompileSpec("TOSA-1.0+FP")),
        (torch.float16, TosaCompileSpec("TOSA-1.0+FP")),
        (torch.bfloat16, TosaCompileSpec("TOSA-1.0+FP+bf16")),
        (torch.float32, VgfCompileSpec()),
    ),
    ids=("tosa_fp32", "tosa_fp16", "tosa_bf16", "vgf_fp32"),
)
def test_partitioner_preserves_supported_roll(
    dtype: torch.dtype, compile_spec: TosaCompileSpec | VgfCompileSpec
) -> None:
    model = Roll((-2, -2), (1, 2))
    exported_program = torch.export.export(
        model, (torch.randn(1, 8, 8, 4, dtype=dtype),), strict=True
    )
    partitioner = create_partitioner(compile_spec)
    preserved_ops, filter_fn = partitioner.ops_to_not_decompose(exported_program)
    roll_node = next(
        node
        for node in exported_program.graph.nodes
        if node.target == torch.ops.aten.roll.default
    )

    assert torch.ops.aten.roll.default in preserved_ops
    assert filter_fn is not None and filter_fn(roll_node)


def test_partitioner_does_not_preserve_flat_roll() -> None:
    class FlatRoll(torch.nn.Module):
        def forward(self, x: torch.Tensor) -> torch.Tensor:
            return torch.roll(x, 2)

    exported_program = torch.export.export(
        FlatRoll(), (torch.randn(2, 4),), strict=True
    )
    partitioner = create_partitioner(VgfCompileSpec())
    _, filter_fn = partitioner.ops_to_not_decompose(exported_program)
    roll_node = next(
        node
        for node in exported_program.graph.nodes
        if node.target == torch.ops.aten.roll.default
    )

    assert filter_fn is not None and not filter_fn(roll_node)


def test_partitioner_does_not_preserve_unquantized_tosa_int_roll() -> None:
    model = Roll((1,), (1,))
    exported_program = torch.export.export(model, (torch.randn(2, 4),), strict=True)
    partitioner = create_partitioner(TosaCompileSpec("TOSA-1.0+INT"))
    _, filter_fn = partitioner.ops_to_not_decompose(exported_program)
    roll_node = next(
        node
        for node in exported_program.graph.nodes
        if node.target == torch.ops.aten.roll.default
    )

    assert filter_fn is not None and not filter_fn(roll_node)


@pytest.mark.parametrize(
    "dtype,compile_spec",
    (
        (torch.float64, VgfCompileSpec()),
        (torch.int32, VgfCompileSpec()),
        (torch.bool, VgfCompileSpec()),
        (torch.bfloat16, TosaCompileSpec("TOSA-1.0+FP")),
    ),
    ids=("float64", "int32", "bool", "bf16_without_extension"),
)
def test_partitioner_does_not_preserve_unsupported_dtype_roll(
    dtype: torch.dtype, compile_spec: TosaCompileSpec | VgfCompileSpec
) -> None:
    model = Roll((1,), (1,))
    exported_program = torch.export.export(
        model, (torch.zeros(2, 4, dtype=dtype),), strict=True
    )
    partitioner = create_partitioner(compile_spec)
    _, filter_fn = partitioner.ops_to_not_decompose(exported_program)
    roll_node = next(
        node
        for node in exported_program.graph.nodes
        if node.target == torch.ops.aten.roll.default
    )

    assert filter_fn is not None and not filter_fn(roll_node)
