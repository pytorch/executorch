# Copyright 2025 Arm Limited and/or its affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from typing import List, Tuple

import pytest

import torch

from executorch.backends.arm.quantizer import (
    EthosUQuantizer,
    get_symmetric_quantization_config,
    TOSAQuantizer,
)
from executorch.backends.arm.test import common
from executorch.backends.arm.test.tester.test_pipeline import (
    EthosU85PipelineBI,
    OpNotSupportedPipeline,
    TosaPipelineBI,
    TosaPipelineMI,
)
from executorch.backends.xnnpack.test.tester.tester import Quantize

aten_op = "torch.ops.aten.where.self"
exir_op = "executorch_exir_dialects_edge__ops_aten_where_self"


class Where(torch.nn.Module):
    def __init__(
        self, shape: tuple | int, dtype: torch.dtype | Tuple[torch.dtype], condition
    ):
        super().__init__()
        self.shape = shape if isinstance(shape, tuple) else (shape,) * shape
        self.dtype = (dtype, dtype) if isinstance(dtype, torch.dtype) else dtype
        self.condition = condition

    def get_inputs(self):
        inputs: List = [0, 0]
        for i in range(2):
            if self.dtype[i] in [torch.int8, torch.int16, torch.int32]:
                inputs[i] = torch.randint(
                    torch.iinfo(self.dtype[i]).min,
                    torch.iinfo(self.dtype[i]).max,
                    self.shape,
                    dtype=self.dtype[i],
                )
            elif self.dtype[i] in [torch.float32]:
                inputs[i] = torch.randn(*self.shape).to(self.dtype[i])
            elif self.dtype[i] is torch.bool:
                inputs[i] = torch.randint(0, 1, self.shape, dtype=torch.bool)
            else:
                raise TypeError(
                    f"Input generation for dtype {self.dtype[i]} not implemented in "
                    "Where()"
                )

        return tuple(inputs)

    def forward(
        self,
        input_: torch.Tensor,
        other_: torch.Tensor,
    ):
        return torch.where(self.condition(input_), input_, other_)


def tensor_condition(input: torch.Tensor):
    return input > torch.zeros_like(input)


def scalar_condition(input: torch.Tensor):
    return input > 0


two_dim_tensor_cond = Where(
    2,
    torch.float32,
    tensor_condition,
)

three_dim_tensor_cond = Where(
    3,
    torch.float32,
    tensor_condition,
)

float32_tensor_cond = Where(
    1,
    torch.float32,
    tensor_condition,
)

float32_tensor_cond_tuple_dtype = Where(
    1,
    (torch.float32, torch.int8),
    tensor_condition,
)

float32_tensor_cond_tuple_dtype_bool = Where(
    1,
    (torch.float32, torch.bool),
    tensor_condition,
)

# Scalar tests
two_dim_scalar_cond = Where(
    2,
    torch.float32,
    scalar_condition,
)

three_dim_scalar_cond = Where(
    3,
    torch.float32,
    scalar_condition,
)

float32_scalar_cond = Where(
    1,
    torch.float32,
    scalar_condition,
)

test_modules_common = {
    "two_dim_tensor_cond": two_dim_tensor_cond,
    "three_dim_tensor_cond": three_dim_tensor_cond,
    "float32_tensor_cond": float32_tensor_cond,
    "two_dim_scalar_cond": two_dim_scalar_cond,
    "three_dim_scalar_cond": three_dim_scalar_cond,
    "float32_scalar_cond": float32_scalar_cond,
}

test_modules_MI = {
    **test_modules_common,
    "float32_tensor_cond_tuple_dtype": float32_tensor_cond_tuple_dtype,
    "float32_tensor_cond_tuple_dtype_bool": float32_tensor_cond_tuple_dtype_bool,
}

test_modules_BI = {
    **test_modules_common,
}

input_t = Tuple[torch.Tensor]


@common.parametrize("test_module", test_modules_MI)
def test_where_tosa_MI(test_module):
    pipeline = TosaPipelineMI[input_t](
        test_module, test_module.get_inputs(), aten_op, exir_op
    )
    pipeline.run()


@common.parametrize("test_module", test_modules_BI)
def test_where_tosa_BI(test_module):
    compile_spec = common.get_tosa_compile_spec("TOSA-0.80+BI")
    quantizer = TOSAQuantizer(compile_spec).set_io(get_symmetric_quantization_config())
    pipeline = TosaPipelineBI[input_t](
        test_module, test_module.get_inputs(), aten_op, exir_op
    )
    pipeline.change_args(
        "quantize", Quantize(quantizer, get_symmetric_quantization_config())
    )
    pipeline.run()


@common.parametrize("test_module", test_modules_BI)
def test_where_u55_BI(test_module):
    compile_spec = common.get_u55_compile_spec()
    quantizer = EthosUQuantizer(compile_spec).set_io(
        get_symmetric_quantization_config()
    )

    # There will be one full_like op which will be delegated.
    num_delegates = 1
    num_exir = 0

    pipeline = OpNotSupportedPipeline[input_t](
        test_module,
        test_module.get_inputs(),
        "TOSA-0.80+BI+u55",
        {
            exir_op: 1,
            "executorch_exir_dialects_edge__ops_aten_full_default": num_exir,
        },
        num_delegates,
    )

    pipeline.change_args(
        "quantize", Quantize(quantizer, get_symmetric_quantization_config())
    )
    pipeline.run()


@common.parametrize("test_module", test_modules_BI)
def test_where_u85_BI(test_module):
    compile_spec = common.get_u85_compile_spec()
    quantizer = EthosUQuantizer(compile_spec).set_io(
        get_symmetric_quantization_config()
    )
    pipeline = EthosU85PipelineBI[input_t](
        test_module, test_module.get_inputs(), aten_op, exir_op, run_on_fvp=False
    )
    pipeline.change_args(
        "quantize", Quantize(quantizer, get_symmetric_quantization_config())
    )
    pipeline.run()


@common.parametrize("test_module", test_modules_BI)
@pytest.mark.skip(reason="The same as test_where_u55_BI")
@common.XfailIfNoCorstone300
def test_where_u55_BI_on_fvp(test_module):
    compile_spec = common.get_u55_compile_spec()
    quantizer = EthosUQuantizer(compile_spec).set_io(
        get_symmetric_quantization_config()
    )

    # There will be one full_like op which will be delegated.
    num_delegates = 1
    num_exir = 0

    pipeline = OpNotSupportedPipeline[input_t](
        test_module,
        test_module.get_inputs(),
        "TOSA-0.80+BI+u55",
        {
            exir_op: 1,
            "executorch_exir_dialects_edge__ops_aten_full_default": num_exir,
        },
        num_delegates,
    )

    pipeline.change_args(
        "quantize", Quantize(quantizer, get_symmetric_quantization_config())
    )
    pipeline.run()


@common.parametrize("test_module", test_modules_BI)
@common.XfailIfNoCorstone320
def test_where_u85_BI_on_fvp(test_module):
    compile_spec = common.get_u85_compile_spec()
    quantizer = EthosUQuantizer(compile_spec).set_io(
        get_symmetric_quantization_config()
    )
    pipeline = EthosU85PipelineBI[input_t](
        test_module, test_module.get_inputs(), aten_op, exir_op, run_on_fvp=True
    )
    pipeline.change_args(
        "quantize", Quantize(quantizer, get_symmetric_quantization_config())
    )
    pipeline.run()
