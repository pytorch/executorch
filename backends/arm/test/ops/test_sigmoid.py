# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
# Copyright 2024-2025 Arm Limited and/or its affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.


from typing import Tuple

import torch
from executorch.backends.arm.test import common, conftest
from executorch.backends.arm.test.tester.test_pipeline import (
    EthosU55PipelineBI,
    EthosU85PipelineBI,
    TosaPipelineBI,
    TosaPipelineMI,
)

aten_op = "torch.ops.aten.sigmoid.default"  # Used for checking that we do not have softmax in the graph after decompose
exir_op = "executorch_exir_dialects_edge__ops_aten_sigmoid_default"
input_t1 = Tuple[torch.Tensor]  # Input x

test_data_suite = {
    # (test_name, test_data)
    "zeros": lambda: torch.zeros(10, 10, 10, 10),
    "ones": lambda: torch.ones(10, 10, 10),
    "rand": lambda: torch.rand(10, 10) - 0.5,
    "randn_pos": lambda: torch.randn(10) + 10,
    "randn_neg": lambda: torch.randn(10) - 10,
    "ramp": lambda: torch.arange(-16, 16, 0.2),
}


class Sigmoid(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.sigmoid = torch.nn.Sigmoid()

    def forward(self, x):
        return self.sigmoid(x)


class AddSigmoid(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.sigmoid = torch.nn.Sigmoid()

    def forward(self, x):
        return self.sigmoid(x + x)


class SigmoidAdd(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.sigmoid = torch.nn.Sigmoid()

    def forward(self, x):
        return x + self.sigmoid(x)


class SigmoidAddSigmoid(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.sigmoid = torch.nn.Sigmoid()

    def forward(self, x, y):
        return self.sigmoid((self.sigmoid(y) + self.sigmoid(x)))


@common.parametrize("test_data", test_data_suite)
def test_sigmoid_tosa_MI(test_data: torch.Tensor):
    TosaPipelineMI[input_t1](Sigmoid(), (test_data(),), aten_op, exir_op).run()


@common.parametrize("test_data", test_data_suite)
def test_sigmoid_tosa_BI(test_data: torch.Tensor):
    TosaPipelineBI[input_t1](Sigmoid(), (test_data(),), aten_op, exir_op).run()


def test_sigmoid_tosa_MI_add():
    TosaPipelineMI[input_t1](
        AddSigmoid(),
        (test_data_suite["zeros"](),),
        aten_op,
        exir_op,
        tosa_version=conftest.get_option("tosa_version"),
    ).run()


def test_sigmoid_tosa_BI_add():
    TosaPipelineBI[input_t1](
        AddSigmoid(),
        (test_data_suite["ramp"](),),
        aten_op,
        exir_op,
        tosa_version=conftest.get_option("tosa_version"),
    ).run()


def test_sigmoid_tosa_MI_add_2():
    TosaPipelineMI[input_t1](
        SigmoidAdd(),
        (test_data_suite["zeros"](),),
        aten_op,
        exir_op,
        tosa_version=conftest.get_option("tosa_version"),
    ).run()


def test_sigmoid_tosa_BI_add_2():
    TosaPipelineBI[input_t1](
        SigmoidAdd(),
        (test_data_suite["zeros"](),),
        aten_op,
        exir_op,
        tosa_version=conftest.get_option("tosa_version"),
    ).run()


def test_sigmoid_tosa_MI_add_3():
    TosaPipelineMI[input_t1](
        SigmoidAddSigmoid(),
        (test_data_suite["randn_neg"](), test_data_suite["randn_pos"]()),
        aten_op,
        exir_op,
        tosa_version=conftest.get_option("tosa_version"),
    ).run()


def test_sigmoid_tosa_BI_3():
    TosaPipelineBI[input_t1](
        SigmoidAddSigmoid(),
        (test_data_suite["randn_neg"](), test_data_suite["randn_pos"]()),
        aten_op,
        exir_op,
        tosa_version=conftest.get_option("tosa_version"),
    ).run()


@common.parametrize("test_data", test_data_suite)
def test_sigmoid_u55_BI(test_data: Tuple):
    pipeline = EthosU55PipelineBI[input_t1](
        Sigmoid(),
        (test_data(),),
        aten_op,
        exir_op,
        run_on_fvp=False,
    )
    pipeline.run()


@common.parametrize("test_data", test_data_suite)
def test_sigmoid_u85_BI(test_data: Tuple):
    pipeline = EthosU85PipelineBI[input_t1](
        Sigmoid(),
        (test_data(),),
        aten_op,
        exir_op,
        run_on_fvp=False,
    )
    pipeline.run()
