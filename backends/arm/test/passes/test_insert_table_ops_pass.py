# Copyright 2025-2026 Arm Limited and/or its affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from typing import ClassVar, Dict, Tuple

import torch
from executorch.backends.arm._passes.fold_qdq_with_annotated_qparams_pass import (
    FoldAndAnnotateQParamsPass,
)
from executorch.backends.arm._passes.insert_table_ops import InsertTableOpsPass
from executorch.backends.arm._passes.quant_args import QuantArgs
from executorch.backends.arm.test import common
from executorch.backends.arm.test.tester.test_pipeline import PassPipeline

input_t = Tuple[torch.Tensor]  # Input x


class Sigmoid(torch.nn.Module):
    test_data: ClassVar[Dict[str, input_t]] = {
        "rand": (torch.rand(4),),
    }

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x.sigmoid()


@common.parametrize("test_data", Sigmoid.test_data)
def test_insert_table_ops_tosa_INT(test_data: input_t) -> None:
    module = Sigmoid()
    pipeline = PassPipeline[input_t](
        module,
        test_data,
        quantize=True,
        ops_before_pass={"executorch_exir_dialects_edge__ops_aten_sigmoid_default": 1},
        ops_after_pass={
            "executorch_exir_dialects_edge__ops_quantized_decomposed_quantize_per_tensor_default": 1,
            "executorch_exir_dialects_edge__ops_quantized_decomposed_dequantize_per_tensor_default": 1,
            "backend__ops_tosa_TABLE_default": 1,
        },
        ops_not_after_pass=["executorch_exir_dialects_edge__ops_aten_sigmoid_default"],
        pass_list=[FoldAndAnnotateQParamsPass],
        passes_with_exported_program=[InsertTableOpsPass],
    )
    pipeline.pop_stage(-1)  # Do not compare output

    pipeline.run()


def test_generate_8bit_table_domain_covers_full_int8_range() -> None:
    table_domain = InsertTableOpsPass._get_8bit_table_domain()
    expected_domain = torch.arange(-128, 128, dtype=torch.int16)

    assert table_domain.dtype == torch.int8
    assert table_domain.shape == torch.Size((256,))
    assert torch.equal(table_domain.to(dtype=torch.int16), expected_domain)


def test_generate_8bit_table_values_matches_reference_for_qmin_minus_127() -> None:
    input_qargs = QuantArgs(
        scale=0.15988604724407196,
        zp=-17,
        qmin=-127,
        qmax=127,
        dtype=torch.int8,
    )
    output_qargs = QuantArgs(
        scale=0.0039350856095552444,
        zp=-127,
        qmin=-127,
        qmax=127,
        dtype=torch.int8,
    )

    insert_table_ops_pass = object.__new__(InsertTableOpsPass)
    lut_values, lshift = insert_table_ops_pass.generate_8bit_table_values(
        torch.sigmoid,
        input_qargs,
        output_qargs,
    )

    expected_domain = (
        torch.arange(-128, 128, dtype=torch.int16)
        .clamp(input_qargs.qmin, input_qargs.qmax)
        .to(dtype=torch.int8)
    )
    expected_lut_values = output_qargs.quantize_value(
        torch.sigmoid(input_qargs.dequantize_value(expected_domain))
    ).to(dtype=torch.int8)
    zero_input_code = input_qargs.get_zp_per_tensor()
    zero_input_index = zero_input_code - torch.iinfo(torch.int8).min
    expected_zero_output = int(
        output_qargs.quantize_value(torch.tensor([0.5], dtype=torch.float32))[0]
    )

    assert lshift == 0
    assert torch.equal(lut_values, expected_lut_values)
    assert int(lut_values[0]) == int(lut_values[1])
    assert int(lut_values[zero_input_index]) == expected_zero_output


def test_generate_16bit_identity_table_is_antisymmetric() -> None:
    """Equal zero-point-zero qparams produce an antisymmetric full-range LUT."""
    qargs = QuantArgs(
        scale=1.0,
        zp=0,
        qmin=-32767,
        qmax=32767,
        dtype=torch.int16,
    )

    lut_values, lshift = InsertTableOpsPass.generate_16_bit_table_values(
        torch.nn.Identity(),
        qargs,
        qargs,
    )

    assert lshift == -7
    for i in range(len(lut_values) // 2):
        neg_value = int(lut_values[i])
        pos_value = int(lut_values[~i])
        assert neg_value == -pos_value, (
            "Expected a zero-point-zero identity LUT to be antisymmetric, but "
            f"lut_values[{i}]={neg_value} and "
            f"lut_values[{~i}]={pos_value}"
        )


def test_generate_16bit_full_range_identity_table_applies_zero_point_delta() -> None:
    """Changing only zero point offsets each LUT value by output_zp - input_zp."""
    input_qargs = QuantArgs(
        scale=1.0,
        zp=1,
        qmin=-32767,
        qmax=32767,
        dtype=torch.int16,
    )
    output_qargs = QuantArgs(
        scale=1.0,
        zp=-1,
        qmin=-32767,
        qmax=32767,
        dtype=torch.int16,
    )

    lut_values, lshift = InsertTableOpsPass.generate_16_bit_table_values(
        torch.nn.Identity(),
        input_qargs,
        output_qargs,
    )

    table_input_codes = torch.arange(-32768, 32769, 128, dtype=torch.int32)
    zero_point_delta = (
        output_qargs.get_zp_per_tensor() - input_qargs.get_zp_per_tensor()
    )
    expected_lut_values = (table_input_codes + zero_point_delta).clamp(
        output_qargs.qmin,
        output_qargs.qmax,
    )

    assert lshift == -7
    assert torch.equal(lut_values.to(torch.int32), expected_lut_values)


def test_generate_16bit_full_range_identity_table_applies_scale_ratio() -> None:
    """Changing only scale multiplies each LUT value by input / output scale."""
    input_qargs = QuantArgs(
        scale=0.5,
        zp=0,
        qmin=-32767,
        qmax=32767,
        dtype=torch.int16,
    )
    output_qargs = QuantArgs(
        scale=1.0,
        zp=0,
        qmin=-32767,
        qmax=32767,
        dtype=torch.int16,
    )

    lut_values, lshift = InsertTableOpsPass.generate_16_bit_table_values(
        torch.nn.Identity(),
        input_qargs,
        output_qargs,
    )

    table_input_codes = torch.arange(-32768, 32769, 128, dtype=torch.int32)
    scale_ratio = (
        input_qargs.get_scale_per_tensor() / output_qargs.get_scale_per_tensor()
    )
    expected_lut_values = (table_input_codes * scale_ratio).round().to(torch.int32)

    assert lshift == -7
    assert torch.equal(lut_values.to(torch.int32), expected_lut_values)


def test_generate_16bit_table_clamps_endpoint_for_restricted_range() -> None:
    """A restricted input qmax clamps the final LUT value."""
    input_qargs = QuantArgs(
        scale=1.0,
        zp=0,
        qmin=-32767,
        qmax=30080,
        dtype=torch.int16,
    )
    output_qargs = QuantArgs(
        scale=1.0,
        zp=0,
        qmin=-32767,
        qmax=32767,
        dtype=torch.int16,
    )

    lut_values, lshift = InsertTableOpsPass.generate_16_bit_table_values(
        torch.nn.Identity(),
        input_qargs,
        output_qargs,
    )

    assert lshift == -7
    assert int(lut_values[-1]) == input_qargs.qmax
