# Copyright 2025 Arm Limited and/or its affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import torch
from executorch.backends.arm._passes.insert_table_ops import InsertTableOpsPass
from executorch.backends.arm._passes.quant_args import QuantArgs
from executorch.backends.arm.quantizer.quantization_config import QuantizationConfig
from executorch.backends.arm.test import common
from executorch.backends.arm.test.tester.test_pipeline import (
    EthosU85PipelineINT,
    OpNotSupportedPipeline,
    TosaPipelineINT,
)
from torchao.quantization.pt2e import HistogramObserver
from torchao.quantization.pt2e.quantizer import QuantizationSpec


def _get_16_bit_quant_config():
    int16_spec = QuantizationSpec(
        dtype=torch.int16,
        observer_or_fake_quant_ctr=HistogramObserver,
        qscheme=torch.per_tensor_symmetric,
    )
    int32_spec = QuantizationSpec(
        dtype=torch.int32,
        observer_or_fake_quant_ctr=HistogramObserver,
        qscheme=torch.per_tensor_symmetric,
    )
    qconfig = QuantizationConfig(
        input_activation=int16_spec,
        output_activation=int32_spec,
        weight=None,
        bias=None,
    )
    return qconfig


def _get_32_bit_quant_config():
    int32_spec = QuantizationSpec(
        dtype=torch.int32,
        observer_or_fake_quant_ctr=HistogramObserver,
        qscheme=torch.per_tensor_symmetric,
    )
    qconfig = QuantizationConfig(
        input_activation=int32_spec,
        output_activation=int32_spec,
        weight=None,
        bias=None,
    )
    return qconfig


def configure_32bit_sigmoid_quantizer(pipeline):
    pipeline.quantizer.set_global(_get_32_bit_quant_config())
    pipeline.quantizer.set_module_type(
        torch.nn.modules.activation.Sigmoid, _get_16_bit_quant_config()
    )


input_t = tuple[torch.Tensor]
test_data_suite = {
    "ones": lambda: torch.ones(10, 10, 10),
    "rand": lambda: torch.rand(10, 10) - 0.5,
    "rand_4d": lambda: torch.rand(1, 10, 10, 10),
    "randn_pos": lambda: torch.randn(10) + 10,
    "randn_neg": lambda: torch.randn(10) - 10,
    "ramp": lambda: torch.arange(-16, 16, 0.2),
}


class Sigmoid(torch.nn.Module):
    aten_op = "torch.ops.aten.sigmoid.default"
    exir_op = "executorch_exir_dialects_edge__ops_aten_sigmoid_default"

    def __init__(self):
        super().__init__()
        self.sigmoid = torch.nn.Sigmoid()

    def forward(self, x):
        return self.sigmoid(x)


class SigmoidAddSigmoid(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.sigmoid = torch.nn.Sigmoid()

    def forward(self, x):
        return self.sigmoid((self.sigmoid(x) + self.sigmoid(x)))


@common.parametrize("test_data", test_data_suite)
def test_sigmoid_tosa_INT(test_data):
    pipeline = TosaPipelineINT(
        Sigmoid(),
        (test_data(),),
        Sigmoid.aten_op,
        Sigmoid.exir_op,
        qtol=1,
        tosa_extensions=["int16"],
    )
    configure_32bit_sigmoid_quantizer(pipeline)
    pipeline.run()


@common.parametrize("test_data", test_data_suite)
def test_sigmoid_tosa_INT_add_sigmoid(test_data):
    pipeline = TosaPipelineINT(
        SigmoidAddSigmoid(),
        (test_data(),),
        Sigmoid.aten_op,
        Sigmoid.exir_op,
        qtol=1,
        tosa_extensions=["int16"],
    )
    configure_32bit_sigmoid_quantizer(pipeline)
    pipeline.run()


@common.parametrize("test_data", test_data_suite)
@common.XfailIfNoCorstone300
def test_sigmoid_u55_INT(test_data):
    pipeline = OpNotSupportedPipeline(
        Sigmoid(),
        (test_data(),),
        {Sigmoid.exir_op: 1},
        quantize=True,
        u55_subset=True,
        tosa_extensions=["int16"],
    )
    configure_32bit_sigmoid_quantizer(pipeline)
    pipeline.run()


@common.parametrize("test_data", test_data_suite)
@common.XfailIfNoCorstone300
def test_sigmoid_u55_INT_add_sigmoid(test_data):
    pipeline = OpNotSupportedPipeline(
        SigmoidAddSigmoid(),
        (test_data(),),
        {Sigmoid.exir_op: 3},
        n_expected_delegates=1,
        quantize=True,
        u55_subset=True,
        tosa_extensions=["int16"],
    )
    configure_32bit_sigmoid_quantizer(pipeline)
    pipeline.run()


@common.parametrize("test_data", test_data_suite)
@common.XfailIfNoCorstone320
def test_sigmoid_u85_INT(test_data):
    pipeline = EthosU85PipelineINT(
        Sigmoid(),
        (test_data(),),
        Sigmoid.aten_op,
        Sigmoid.exir_op,
    )
    configure_32bit_sigmoid_quantizer(pipeline)
    pipeline.run()


@common.parametrize(
    "test_data",
    test_data_suite,
)
@common.XfailIfNoCorstone320
def test_sigmoid_u85_INT_add_sigmoid(test_data):
    pipeline = EthosU85PipelineINT(
        SigmoidAddSigmoid(),
        (test_data(),),
        Sigmoid.aten_op,
        Sigmoid.exir_op,
    )
    configure_32bit_sigmoid_quantizer(pipeline)
    pipeline.run()


def test_int16_table_small_output_range_is_not_degenerate():
    """Regression for the int16 TABLE negative-rshift bug.

    A TABLE op whose output uses fewer than 16 bits -- e.g. a sigmoid output in
    [0, 1] quantized with a small scale, so the max table value (here 4096) is
    well below 2**15 -- yields ``rshift < 0``. The generator then did
    ``lut_values >> rshift``, an undefined negative right-shift that zeroed the
    whole table on the host, turning the on-device activation into a constant.
    The table must remain a non-degenerate, monotonic ramp.
    """
    # qparams for a small-output-range sigmoid (input spans ~+-22.4,
    # output in [0, 1] quantized at 1/4096 -> max table value 4096 = 13 bits).
    in_quantargs = QuantArgs(
        scale=0.0006833122461102903, zp=0, qmin=-32767, qmax=32767, dtype=torch.int16
    )
    out_quantargs = QuantArgs(
        scale=1.0 / 4096, zp=0, qmin=-32767, qmax=32767, dtype=torch.int16
    )

    # generate_16_bit_table_values is a @staticmethod; call it on the class.
    lut, rescale_lshift = InsertTableOpsPass.generate_16_bit_table_values(
        torch.sigmoid, in_quantargs, out_quantargs
    )

    assert (
        torch.unique(lut).numel() > 1
    ), "int16 sigmoid table collapsed to a constant (negative-rshift bug)"
    assert bool(
        (lut[1:] >= lut[:-1]).all()
    ), "sigmoid table must be monotonically non-decreasing"
    assert int(lut.min()) == 0 and int(lut.max()) == 4096, (
        f"unexpected table range [{int(lut.min())}, {int(lut.max())}], "
        "expected a full [0, 4096] sigmoid ramp"
    )
    # Values already fit in int16, so no shift is applied: this is the documented
    # int16 case (rshift == 0 -> rescale_lshift == -7), not the buggy -10.
    assert rescale_lshift == -7, f"expected rescale_lshift == -7, got {rescale_lshift}"
