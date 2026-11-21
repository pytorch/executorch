# Copyright 2026 Arm Limited and/or its affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from copy import deepcopy

import pytest
import torch
from executorch.backends.arm._passes.insert_table_ops import InsertTableOpsPass
from executorch.backends.arm.quantizer.quantization_config import QuantizationConfig
from executorch.backends.arm.test.tester.test_pipeline import TosaPipelineINT
from executorch.exir.dialects._ops import ops as exir_ops
from torchao.quantization.pt2e import HistogramObserver
from torchao.quantization.pt2e.quantizer import QuantizationSpec


class Unary(torch.nn.Module):
    def __init__(self, op):
        super().__init__()
        self.op = op

    def forward(self, x):
        return self.op(x)


class Sigmoid(torch.nn.Module):
    def __init__(self, with_add):
        super().__init__()
        self.sigmoid = torch.nn.Sigmoid()
        self.with_add = with_add

    def forward(self, x):
        value = self.sigmoid(x)
        return self.sigmoid(value + value) if self.with_add else value


class FloorDiv(torch.nn.Module):
    def forward(self, x, y):
        return torch.div(x, y, rounding_mode="floor")


@pytest.fixture
def table_graphs(monkeypatch):
    graphs = []
    original = InsertTableOpsPass.call

    def capture(self, graph_module):
        result = original(self, graph_module)
        graphs.append(deepcopy(result.graph_module))
        return result

    monkeypatch.setattr(InsertTableOpsPass, "call", capture)
    return graphs


@pytest.mark.parametrize(
    "op,values",
    [
        (torch.acosh, [1.01, 1.1, 2.0, 3.0]),
        (torch.asin, [0.0, 0.25, 0.5, 0.75]),
        (torch.asinh, [-1.0, 0.0, 1.0, 2.0]),
        (torch.atanh, [-0.5, -0.25, 0.0, 0.5]),
        (torch.acos, [0.0, 0.25, 0.5, 0.75]),
        (torch.expm1, [0.0, 0.25, 0.5, 1.0]),
        (torch.erfinv, [0.125, 0.25, 0.5, 0.75]),
        (torch.cosh, [0.5, 1.0, 1.5, 2.0]),
        (torch.log10, [0.5, 1.0, 1.5, 2.0]),
        (torch.tan, [-0.5, -0.25, 0.0, 0.5]),
    ],
    ids=lambda value: value.__name__ if callable(value) else None,
)
def test_integer_promoting_unary_lowers_to_int8_table(op, values, table_graphs):
    pipeline = TosaPipelineINT(
        Unary(op), (torch.tensor(values),), [], run_on_tosa_ref_model=False
    )
    pipeline.pop_stage("to_executorch")
    pipeline.count_tosa_ops({"TABLE": 1})
    pipeline.run()

    tables = [
        node
        for graph in table_graphs
        for node in graph.graph.nodes
        if node.target == exir_ops.backend.tosa.TABLE.default
    ]
    assert len(tables) == 1
    assert tables[0].meta["val"].dtype == torch.int8
    assert tables[0].args[0].meta["val"].dtype == torch.int8


@pytest.mark.parametrize("with_add", [False, True])
def test_int32_requantization_stays_outside_int16_sigmoid_table(with_add, table_graphs):
    pipeline = TosaPipelineINT(
        Sigmoid(with_add),
        (torch.linspace(-16, 16, 17),),
        [],
        tosa_extensions=["int16"],
        run_on_tosa_ref_model=False,
    )
    int16_spec, int32_spec = (
        QuantizationSpec(
            dtype=dtype,
            observer_or_fake_quant_ctr=HistogramObserver,
            qscheme=torch.per_tensor_symmetric,
        )
        for dtype in (torch.int16, torch.int32)
    )
    pipeline.quantizer.set_global(
        QuantizationConfig(
            input_activation=int32_spec,
            output_activation=int32_spec,
            weight=None,
            bias=None,
        )
    )
    pipeline.quantizer.set_module_type(
        torch.nn.Sigmoid,
        QuantizationConfig(
            input_activation=int16_spec,
            output_activation=int32_spec,
            weight=None,
            bias=None,
        ),
    )
    pipeline.pop_stage("to_executorch")
    pipeline.count_tosa_ops({"TABLE": 2 if with_add else 1, "SIGMOID": 0})
    pipeline.run()

    tables = [
        node
        for graph in table_graphs
        for node in graph.graph.nodes
        if node.target == exir_ops.backend.tosa.TABLE.default
    ]
    assert len(tables) == (2 if with_add else 1)
    for table in tables:
        quantize = table.args[0]
        assert quantize.target == InsertTableOpsPass.quantize_op
        assert quantize.meta["val"].dtype == torch.int16
        assert quantize.args[0].target == InsertTableOpsPass.dequantize_op
        assert quantize.args[0].args[5] == torch.int32
        assert table.meta["val"].dtype == torch.int32
        rescale = next(iter(table.users))
        assert rescale.target == exir_ops.backend.tosa.RESCALE.default
        assert rescale.meta["val"].dtype == torch.int32


def test_int32_floor_div_lowers_reciprocal_after_int8_requantization(table_graphs):
    pipeline = TosaPipelineINT(
        FloorDiv(),
        (
            torch.tensor([[-101, -29, 31, 199]], dtype=torch.int32),
            torch.tensor([[11, 13, 17, 19]], dtype=torch.int32),
        ),
        [],
        tosa_extensions=["u55"],
        run_on_tosa_ref_model=False,
    )
    pipeline.pop_stage("to_executorch")
    pipeline.count_tosa_ops({"TABLE": 2, "RECIPROCAL": 0})
    pipeline.run()

    reciprocal_tables = [
        node
        for graph in table_graphs
        for node in graph.graph.nodes
        if node.target == exir_ops.backend.tosa.TABLE.default
        and node.args[0].target == InsertTableOpsPass.quantize_op
    ]
    assert len(reciprocal_tables) == 1
    table = reciprocal_tables[0]
    assert table.meta["val"].dtype == torch.int8
    quantize = table.args[0]
    assert quantize.meta["val"].dtype == torch.int8
    dequantize = quantize.args[0]
    assert dequantize.target == InsertTableOpsPass.dequantize_op
    assert dequantize.args[5] == torch.int32
