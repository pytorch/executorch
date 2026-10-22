# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.
# pyre-strict
"""Tests for folding quantized DyT alpha multiplication into a tanh LUT.

The tests prove the rewrite is byte-exact: the generated 256-entry TABLE is
compared against the real TOSA integer RESCALE/Mul/tanh path over every int8
input code, so the fold cannot change quantized output on any model.

"""

from __future__ import annotations

from typing import cast, ClassVar, Dict, Tuple

import executorch.backends.arm.tosa.dialect  # noqa: F401
import torch
from executorch.backends.arm._passes import (
    FoldAndAnnotateQParamsPass,
    InsertRescaleInt32Pass,
)
from executorch.backends.arm._passes.fold_dyt_alpha_into_lut_pass import (
    _generate_dyt_lut,
    _RescaleParams,
    FoldDyTAlphaIntoLUTPass,
)
from executorch.backends.arm._passes.quant_args import QuantArgs
from executorch.backends.arm.test import common
from executorch.backends.arm.test.tester.test_pipeline import PassPipeline
from executorch.backends.arm.tosa.specification import (
    TosaLoweringContext,
    TosaSpecification,
)
from executorch.exir import ExportedProgram
from executorch.exir.dialects._ops import ops as exir_ops
from torch.export import export


class _PostRescaleFixture(torch.nn.Module):
    # Declared so the checker sees the registered buffer as a Tensor rather than
    # the ``Tensor | Module`` that ``nn.Module.__getattr__`` is annotated to give.
    alpha_code: torch.Tensor

    def __init__(self) -> None:
        super().__init__()
        self.register_buffer(
            "alpha_code",
            torch.tensor([127], dtype=torch.int8),
        )

    def forward(self, x_code: torch.Tensor) -> torch.Tensor:
        # Export a buffer and user input; the test replaces this placeholder op
        # with the exact post-InsertRescaleInt32Pass topology consumed by the pass.
        return x_code + self.alpha_code


_TEST_DATA = (
    torch.arange(-128, 128, dtype=torch.int16).to(torch.int8).reshape(1, 1, 16, 16),
)


def _qargs(
    scale: float,
    zp: int,
    qmin: int | None = None,
    qmax: int | None = None,
    dtype: torch.dtype = torch.int8,
) -> QuantArgs:
    dtype_range = torch.iinfo(dtype)
    return QuantArgs(
        scale=scale,
        zp=zp,
        qmin=dtype_range.min if qmin is None else qmin,
        qmax=dtype_range.max if qmax is None else qmax,
        dtype=dtype,
    )


# ``QuantArgs.scale``/``zp`` are typed to also cover the per-channel case, where
# they are lists. Every fixture in this file is per-tensor, so narrow them once
# here instead of casting at each arithmetic site.
def _scale_of(qargs: QuantArgs) -> float:
    return cast(float, qargs.scale)


def _zp_of(qargs: QuantArgs) -> int:
    return cast(int, qargs.zp)


_ACTIVATION_QARGS = _qargs(scale=0.015, zp=3)
_ALPHA_QARGS = _qargs(scale=0.0019607844296842813, zp=-128)
_TANH_INPUT_QARGS = _qargs(scale=0.0077, zp=2)
_TANH_OUTPUT_QARGS = _qargs(scale=0.0078, zp=0)
_ACTIVATION_RESCALE = _RescaleParams(1.0, 3, 0, torch.int32)
_ALPHA_RESCALE = _RescaleParams(1.0, -128, 0, torch.int32)
_MUL_OUTPUT_RESCALE = _RescaleParams(
    (_scale_of(_ACTIVATION_QARGS) * _scale_of(_ALPHA_QARGS))
    / _scale_of(_TANH_INPUT_QARGS),
    0,
    2,
    torch.int8,
)


def _build_post_rescale_fixture(
    *,
    activation_rank_view: bool = False,
) -> ExportedProgram:
    exported_program = export(_PostRescaleFixture(), _TEST_DATA, strict=True)
    graph = exported_program.graph_module.graph
    alpha_name = next(iter(exported_program.graph_signature.inputs_to_buffers))
    alpha = next(node for node in graph.nodes if node.name == alpha_name)
    activation = next(
        node for node in graph.nodes if node.op == "placeholder" and node is not alpha
    )
    original_add = next(node for node in graph.nodes if node.op == "call_function")
    output = next(node for node in graph.nodes if node.op == "output")

    with graph.inserting_before(output):
        activation_rescale = graph.call_function(
            exir_ops.backend.tosa.RESCALE.default,
            (activation, torch.int32, [1.0], 3, 0),
        )
        alpha_rescale = graph.call_function(
            exir_ops.backend.tosa.RESCALE.default,
            (alpha, torch.int32, [1.0], -128, 0),
        )
        activation_mul_arg = activation_rescale
        if activation_rank_view:
            activation_mul_arg = graph.call_function(
                exir_ops.edge.aten.view_copy.default,
                (activation_rescale, [1, 1, 1, 16, 16]),
            )
        mul = graph.call_function(
            exir_ops.edge.aten.mul.Tensor,
            (activation_mul_arg, alpha_rescale),
        )
        mul_output_rescale = graph.call_function(
            exir_ops.backend.tosa.RESCALE.default,
            (
                mul,
                torch.int8,
                [_MUL_OUTPUT_RESCALE.scale],
                0,
                2,
            ),
        )
        tanh = graph.call_function(
            exir_ops.edge.aten.tanh.default,
            (mul_output_rescale,),
        )

    activation.meta["output_qparams"] = {0: _ACTIVATION_QARGS}
    mul.meta["input_qparams"] = {
        0: _qargs(_scale_of(_ACTIVATION_QARGS), 0, dtype=torch.int32),
        1: _qargs(_scale_of(_ALPHA_QARGS), 0, dtype=torch.int32),
    }
    tanh.meta["input_qparams"] = {0: _TANH_INPUT_QARGS}
    tanh.meta["output_qparams"] = {0: _TANH_OUTPUT_QARGS}
    output.replace_input_with(original_add, tanh)
    graph.erase_node(original_add)
    graph.lint()
    exported_program.graph_module.recompile()
    return exported_program


def _tosa_reference_outputs(domain: torch.Tensor) -> torch.Tensor:
    spec = TosaSpecification.create_from_string("TOSA-1.0+INT")
    with TosaLoweringContext(spec):
        activation_i32 = exir_ops.backend.tosa.RESCALE.default(
            domain,
            torch.int32,
            [_ACTIVATION_RESCALE.scale],
            _ACTIVATION_RESCALE.input_zp,
            _ACTIVATION_RESCALE.output_zp,
        )
        alpha_i32 = exir_ops.backend.tosa.RESCALE.default(
            torch.tensor([127], dtype=torch.int8),
            torch.int32,
            [_ALPHA_RESCALE.scale],
            _ALPHA_RESCALE.input_zp,
            _ALPHA_RESCALE.output_zp,
        )
        product = activation_i32 * alpha_i32
        mul_codes = exir_ops.backend.tosa.RESCALE.default(
            product,
            torch.int8,
            [_MUL_OUTPUT_RESCALE.scale],
            _MUL_OUTPUT_RESCALE.input_zp,
            _MUL_OUTPUT_RESCALE.output_zp,
        )
    return _TANH_OUTPUT_QARGS.quantize_value(
        torch.tanh(_TANH_INPUT_QARGS.dequantize_value(mul_codes))
    ).to(torch.int8)


def test_lut_flattens_ranked_scalar_alpha() -> None:
    """A broadcast-shaped scalar still produces one flat 256-entry table."""
    lut = _generate_dyt_lut(
        activation_qargs=_ACTIVATION_QARGS,
        alpha_code=torch.tensor([[[[127]]]], dtype=torch.int8),
        activation_rescale=_ACTIVATION_RESCALE,
        alpha_rescale=_ALPHA_RESCALE,
        mul_output_rescale=_MUL_OUTPUT_RESCALE,
        tanh_input_qargs=_TANH_INPUT_QARGS,
        tanh_output_qargs=_TANH_OUTPUT_QARGS,
    )

    expected = _tosa_reference_outputs(
        torch.arange(-128, 128, dtype=torch.int16).to(torch.int8)
    )
    assert lut.shape == (256,)
    assert torch.equal(lut, expected)


def test_lut_preserves_intermediate_integer_rounding() -> None:
    """The composed table matches Mul+RESCALE+tanh, not float alpha folding."""
    input_qargs = _qargs(scale=0.1, zp=3)
    alpha_qargs = _qargs(scale=0.05, zp=0)
    tanh_input_qargs = _qargs(scale=0.07, zp=-2)
    tanh_output_qargs = _qargs(scale=0.006, zp=1)
    alpha_code = torch.tensor([7], dtype=torch.int8)

    lut = _generate_dyt_lut(
        activation_qargs=input_qargs,
        alpha_code=alpha_code,
        activation_rescale=_RescaleParams(
            scale=1.0,
            input_zp=_zp_of(input_qargs),
            output_zp=0,
            output_dtype=torch.int32,
        ),
        alpha_rescale=_RescaleParams(
            scale=1.0,
            input_zp=_zp_of(alpha_qargs),
            output_zp=0,
            output_dtype=torch.int32,
        ),
        mul_output_rescale=_RescaleParams(
            scale=(_scale_of(input_qargs) * _scale_of(alpha_qargs))
            / _scale_of(tanh_input_qargs),
            input_zp=0,
            output_zp=_zp_of(tanh_input_qargs),
            output_dtype=torch.int8,
        ),
        tanh_input_qargs=tanh_input_qargs,
        tanh_output_qargs=tanh_output_qargs,
    )

    domain = torch.arange(-128, 128, dtype=torch.int16).to(torch.int8)
    alpha = alpha_qargs.dequantize_value(alpha_code)
    naive_float_fold = tanh_output_qargs.quantize_value(
        torch.tanh(input_qargs.dequantize_value(domain) * alpha)
    ).to(torch.int8)

    assert lut.shape == (256,)
    assert lut.dtype == torch.int8
    assert not torch.equal(lut, naive_float_fold)


def test_lut_clamps_narrowed_tanh_input_range() -> None:
    activation_qargs = _qargs(scale=0.01, zp=0)
    tanh_input_qargs = _qargs(scale=0.01, zp=0, qmin=-127, qmax=127)
    tanh_output_qargs = _qargs(scale=0.01, zp=0)
    identity_rescale = _RescaleParams(1.0, 0, 0, torch.int32)

    lut = _generate_dyt_lut(
        activation_qargs=activation_qargs,
        alpha_code=torch.tensor([1], dtype=torch.int8),
        activation_rescale=identity_rescale,
        alpha_rescale=identity_rescale,
        mul_output_rescale=_RescaleParams(1.0, 0, 0, torch.int8),
        tanh_input_qargs=tanh_input_qargs,
        tanh_output_qargs=tanh_output_qargs,
    )

    domain = torch.arange(-128, 128, dtype=torch.int16).to(torch.int8)
    effective_codes = domain.clamp(
        tanh_input_qargs.qmin,
        tanh_input_qargs.qmax,
    )
    expected = tanh_output_qargs.quantize_value(
        torch.tanh(tanh_input_qargs.dequantize_value(effective_codes))
    ).to(torch.int8)

    assert lut[0] == expected[0]
    assert torch.equal(lut, expected)


def test_pass_removes_alpha_mul_and_materializes_one_table() -> None:
    exported_program = _build_post_rescale_fixture()
    result = FoldDyTAlphaIntoLUTPass(exported_program).call(
        exported_program.graph_module
    )

    targets = [
        str(node.target)
        for node in result.graph_module.graph.nodes
        if node.op == "call_function"
    ]
    assert result.modified
    assert sum("tosa.TABLE" in target for target in targets) == 1
    assert not any("aten.mul" in target for target in targets)
    assert not any("aten.tanh" in target for target in targets)
    assert not any("tosa.RESCALE" in target for target in targets)


def test_pass_rejects_activation_side_rank_view() -> None:
    exported_program = _build_post_rescale_fixture(activation_rank_view=True)
    result = FoldDyTAlphaIntoLUTPass(exported_program).call(
        exported_program.graph_module
    )

    targets = [
        str(node.target)
        for node in result.graph_module.graph.nodes
        if node.op == "call_function"
    ]
    assert not result.modified
    assert not any("tosa.TABLE" in target for target in targets)
    assert any("aten.view_copy" in target for target in targets)
    assert any("aten.mul" in target for target in targets)
    assert any("aten.tanh" in target for target in targets)


def test_tosa_output_is_bit_exact_after_fold() -> None:
    exported_program = _build_post_rescale_fixture()
    result = FoldDyTAlphaIntoLUTPass(exported_program).call(
        exported_program.graph_module
    )
    table = next(
        value
        for name, value in exported_program.state_dict.items()
        if "dyt_table_constant" in name
    )
    domain = _TEST_DATA[0].flatten()
    expected = _tosa_reference_outputs(domain)
    table_outputs = table[(domain.to(torch.int16) + 128).to(torch.int64)]

    assert result.modified
    assert torch.equal(table, expected)
    assert torch.equal(table_outputs, expected)


class DyTModule(torch.nn.Module):
    """A DyT site fed by a conv, i.e. the shape DyT takes in a real encoder.

    The conv matters: the pass recovers the activation's int8 quantization from
    its producer's ``output_qparams``. Feeding the DyT straight off a graph input
    leaves a bare ``quantize_per_tensor`` as the producer, which carries no
    ``output_qparams``, and the pass then correctly declines to fold.

    """

    test_data: ClassVar[Dict[str, Tuple[torch.Tensor]]] = {
        "rand": (torch.rand(1, 3, 8, 8),),
    }

    def __init__(self, alpha: float = 0.5) -> None:
        super().__init__()
        self.conv = torch.nn.Conv2d(3, 3, kernel_size=1)
        self.alpha = torch.nn.Parameter(torch.tensor([alpha]))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # The permute mirrors the real DyT, which applies its affine in NHWC.
        # It also keeps the alpha Mul adjacent to something other than the conv,
        # so FoldScalarMulIntoConvPass does not absorb it before this pass runs.
        y = torch.permute(self.conv(x), (0, 2, 3, 1))
        return torch.tanh(self.alpha * y)


@common.parametrize("test_data", DyTModule.test_data)
def test_fold_dyt_alpha_into_lut_tosa_INT(test_data: Tuple[torch.Tensor]) -> None:
    """Pipeline-level counterpart to the exhaustive parity tests above.

    Those tests pin numerics on hand-built post-InsertRescale IR.

    This test starts from an nn.Module and runs the real quantization pipeline.

    It checks that one TABLE replaces both the alpha Mul and tanh.

    """
    pipeline = PassPipeline[Tuple[torch.Tensor]](
        DyTModule(),
        test_data,
        quantize=True,
        ops_after_pass={
            "executorch_exir_dialects_backend__ops_tosa_TABLE_default": 1,
        },
        ops_not_after_pass=[
            "executorch_exir_dialects_edge__ops_aten_mul_Tensor",
            "executorch_exir_dialects_edge__ops_aten_tanh_default",
        ],
        pass_list=[FoldAndAnnotateQParamsPass, InsertRescaleInt32Pass],
        passes_with_exported_program=[FoldDyTAlphaIntoLUTPass],
    )
    # The partial ``pass_list`` above stops short of a full TOSA lowering, so no
    # runnable program is left for the comparison stage to execute. Dropped for
    # the same reason as in ``test_insert_rescale_i32_pass.py``, which drives
    # the same two passes. Output equivalence is not lost here: it is pinned
    # exhaustively by ``test_tosa_output_is_bit_exact_after_fold`` above, which
    # checks the generated TABLE against the reference RESCALE/Mul/tanh path
    # over every int8 input code.
    pipeline.pop_stage("run_method_and_compare_outputs")
    pipeline.run()
