# Copyright 2025 Arm Limited and/or its affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from typing import List, Tuple

import torch
from executorch.backends.arm.quantizer import (
    EthosUQuantizer,
    get_symmetric_quantization_config,
)

from executorch.backends.arm.test import common
from executorch.backends.arm.test.tester.test_pipeline import (
    EthosU85PipelineINT,
    OpNotSupportedPipeline,
    TosaPipelineFP,
    TosaPipelineINT,
    VgfPipeline,
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


class ConstWhere(torch.nn.Module):

    def __init__(self, buffer: torch.Tensor, dtype: torch.dtype):
        super().__init__()
        self.buffer = buffer
        self.dtype = dtype
        self.min = torch.nn.Buffer(torch.tensor(0.0, dtype=self.dtype))
        self.input_1 = torch.nn.Buffer(torch.tensor(-1.0, dtype=self.dtype))
        self.input_2 = torch.nn.Buffer(torch.tensor(1.0, dtype=self.dtype))

    def get_inputs(self):
        return (torch.rand(self.buffer.size(), dtype=self.dtype),)

    def forward(self, input: torch.Tensor):
        return (
            torch.where(
                self.buffer > self.min,
                self.input_1,
                self.input_2,
            )
            + input
        )


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

int32_scalar_cond = Where(
    1,
    torch.int32,
    scalar_condition,
)

const_float32 = ConstWhere(
    buffer=torch.tensor([[1.0, -1.0], [-1.0, 1.0]]),
    dtype=torch.float32,
)

test_modules_common = {
    "two_dim_tensor_cond": lambda: two_dim_tensor_cond,
    "three_dim_tensor_cond": lambda: three_dim_tensor_cond,
    "float32_tensor_cond": lambda: float32_tensor_cond,
    "two_dim_scalar_cond": lambda: two_dim_scalar_cond,
    "three_dim_scalar_cond": lambda: three_dim_scalar_cond,
    "float32_scalar_cond": lambda: float32_scalar_cond,
    "const_float32": lambda: const_float32,
}

test_modules_FP = {
    **test_modules_common,
    "float32_tensor_cond_tuple_dtype_bool": lambda: float32_tensor_cond_tuple_dtype_bool,
}

test_modules_FP_unsupported_dtype = {
    "float32_tensor_cond_tuple_dtype": lambda: float32_tensor_cond_tuple_dtype,
    "int32_scalar_cond": lambda: int32_scalar_cond,
}

test_modules_INT = {
    **test_modules_common,
}

input_t = Tuple[torch.Tensor]


@common.parametrize("test_module", test_modules_FP)
def test_where_self_tosa_FP(test_module):
    pipeline = TosaPipelineFP[input_t](
        test_module(),
        test_module().get_inputs(),
        aten_op,
        exir_op,
    )
    pipeline.run()


@common.parametrize("test_module", test_modules_FP_unsupported_dtype)
def test_where_self_tosa_FP_unsupported_dtype(test_module):
    pipeline = OpNotSupportedPipeline[input_t](
        test_module(),
        test_module().get_inputs(),
        {exir_op: 1},
        n_expected_delegates=1,  # condition can be delegated
    )
    pipeline.run()


@common.parametrize("test_module", test_modules_INT)
def test_where_self_tosa_INT(test_module):
    pipeline = TosaPipelineINT[input_t](
        test_module(),
        test_module().get_inputs(),
        aten_op,
        exir_op,
    )
    pipeline.run()


@common.parametrize("test_module", test_modules_INT)
@common.XfailIfNoCorstone300
def test_where_self_u55_INT_not_delegated(test_module):
    # There will be one full_like op which will be delegated.
    num_delegates = 1
    num_exir = 0

    compile_spec = common.get_u55_compile_spec()
    quantizer = EthosUQuantizer(compile_spec).set_io(
        get_symmetric_quantization_config()
    )

    pipeline = OpNotSupportedPipeline[input_t](
        test_module(),
        test_module().get_inputs(),
        {
            exir_op: 1,
            "executorch_exir_dialects_edge__ops_aten_full_default": num_exir,
        },
        num_delegates,
        quantize=True,
        u55_subset=True,
    )
    pipeline.change_args(
        "quantize", Quantize(quantizer, get_symmetric_quantization_config())
    )
    pipeline.run()


@common.parametrize("test_module", test_modules_INT)
@common.XfailIfNoCorstone320
def test_where_self_u85_INT(test_module):

    pipeline = EthosU85PipelineINT[input_t](
        test_module(),
        test_module().get_inputs(),
        aten_op,
        exir_op,
        symmetric_io_quantization=True,
    )
    pipeline.run()


@common.parametrize("test_module", test_modules_FP)
@common.SkipIfNoModelConverter
def test_where_self_vgf_no_quant(test_module):
    pipeline = VgfPipeline[input_t](
        test_module(),
        test_module().get_inputs(),
        aten_op,
        exir_op,
        quantize=False,
    )
    pipeline.run()


@common.parametrize("test_module", test_modules_INT)
@common.SkipIfNoModelConverter
def test_where_self_vgf_quant(test_module):
    pipeline = VgfPipeline[input_t](
        test_module(),
        test_module().get_inputs(),
        aten_op,
        exir_op,
        quantize=True,
    )
    pipeline.run()
