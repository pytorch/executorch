# Copyright 2025-2026 Arm Limited and/or its affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from typing import Callable

import pytest

import torch
from executorch.backends.arm.test import common
from executorch.backends.arm.test.tester.test_pipeline import (
    EthosU55PipelineINT,
    EthosU85PipelineINT,
    OpNotSupportedPipeline,
    TosaPipelineFP,
    TosaPipelineINT,
    VgfPipeline,
)
from executorch.backends.arm.tosa.compile_spec import TosaCompileSpec
from executorch.backends.arm.tosa.partitioner import TOSAPartitioner
from executorch.exir import to_edge_transform_and_lower
from executorch.exir.dialects._ops import ops as exir_ops
from torch._subclasses.fake_tensor import FakeTensorMode

input_t = tuple[torch.Tensor]
test_data_t = tuple[Callable[[], input_t], tuple[float, float, float, torch.dtype]]


class ArangeAdd(torch.nn.Module):
    aten_op: str = "torch.ops.aten.arange.start_step"
    exir_op: str = "executorch_exir_dialects_edge__ops_aten_arange_start_step"

    def __init__(self, start: float, stop: float, step: float, dtype: torch.dtype):
        super().__init__()
        self.args = (start, stop, step)
        self.dtype = dtype

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.arange(*self.args, dtype=self.dtype) + x

    test_data: dict[str, test_data_t] = {
        "10": (lambda: (torch.randn(10, 1),), (0.0, 10.0, 1.0, torch.float32)),
        "15": (lambda: (torch.randn(10),), (0.0, 15.0, 1.5, torch.float32)),
        "100": (lambda: (torch.randn(10, 1),), (0.0, 10.0, 0.1, torch.float32)),
    }

    test_data_dtypes: dict[str, test_data_t] = {
        "fp32_int32": (lambda: (torch.randn(10),), (0.0, 10.0, 1.0, torch.int32)),
        "fp32_int64": (lambda: (torch.randn(10),), (0.0, 10.0, 1.0, torch.int64)),
        "int32_int32": (
            lambda: (torch.randint(0, 10, [10], dtype=torch.int32),),
            (0.0, 10.0, 1.0, torch.int32),
        ),
    }
    test_reject: dict[str, test_data_t] = {
        "int32_int64": (
            lambda: (torch.randint(0, 10, [10], dtype=torch.int32),),
            (0.0, 10.0, 1.0, torch.int64),
        ),
    }


@common.parametrize("test_data", ArangeAdd.test_data)
def test_arange_start_step_tosa_FP(test_data: test_data_t):
    input_data, init_data = test_data
    pipeline = TosaPipelineFP[input_t](
        ArangeAdd(*init_data),
        input_data(),
        ArangeAdd.aten_op,
        ArangeAdd.exir_op,
    )
    pipeline.run()


@common.parametrize("test_data", ArangeAdd.test_data_dtypes)
def test_arange_start_step_tosa_FP_dtypes(test_data: test_data_t):
    input_data, init_data = test_data
    pipeline = TosaPipelineFP[input_t](
        ArangeAdd(*init_data),
        input_data(),
        ArangeAdd.aten_op,
        ArangeAdd.exir_op,
    )
    pipeline.run()


@common.parametrize("test_data", ArangeAdd.test_reject)
def test_arange_start_step_tosa_FP_not_delegated(test_data: test_data_t):
    input_data, init_data = test_data
    pipeline = OpNotSupportedPipeline[input_t](
        ArangeAdd(*init_data), input_data(), non_delegated_ops={ArangeAdd.exir_op: 1}
    )
    pipeline.run()


@common.parametrize("test_data", ArangeAdd.test_data)
def test_arange_start_step_tosa_INT(test_data: test_data_t):
    input_data, init_data = test_data
    pipeline = TosaPipelineINT[input_t](
        ArangeAdd(*init_data),
        input_data(),
        ArangeAdd.aten_op,
        ArangeAdd.exir_op,
    )
    pipeline.run()


@common.parametrize("test_data", ArangeAdd.test_data)
@common.XfailIfNoCorstone300
def test_arange_start_step_u55_INT(test_data: test_data_t):
    input_data, init_data = test_data
    pipeline = EthosU55PipelineINT[input_t](
        ArangeAdd(*init_data),
        input_data(),
        ArangeAdd.aten_op,
    )
    pipeline.run()


@common.parametrize("test_data", ArangeAdd.test_data)
@common.XfailIfNoCorstone320
def test_arange_start_step_u85_INT(test_data: test_data_t):
    input_data, init_data = test_data
    pipeline = EthosU85PipelineINT[input_t](
        ArangeAdd(*init_data),
        input_data(),
        ArangeAdd.aten_op,
    )
    pipeline.run()


@common.parametrize("test_data", ArangeAdd.test_data)
@common.SkipIfNoModelConverter
def test_arange_start_step_vgf_no_quant(test_data: test_data_t):
    input_data, init_data = test_data
    module = ArangeAdd(*init_data)
    pipeline = VgfPipeline[input_t](
        module,
        input_data(),
        module.aten_op,
        module.exir_op,
        quantize=False,
    )
    pipeline.run()


@common.parametrize("test_data", ArangeAdd.test_data)
@common.SkipIfNoModelConverter
def test_arange_start_step_vgf_quant(test_data: test_data_t):
    input_data, init_data = test_data
    module = ArangeAdd(*init_data)
    pipeline = VgfPipeline[input_t](
        module,
        input_data(),
        module.aten_op,
        module.exir_op,
        quantize=True,
    )
    pipeline.run()


class LinspaceAdd(torch.nn.Module):
    aten_op: str = "torch.ops.aten.linspace.default"
    exir_op: str = "executorch_exir_dialects_edge__ops_aten_arange_default"

    def __init__(self, start: float, stop: float, step: int, dtype: torch.dtype):
        super().__init__()
        self.args = (start, stop, step)
        self.dtype = dtype

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.linspace(*self.args, dtype=self.dtype) + x

    test_data: dict[str, test_data_t] = {
        "10": (lambda: (torch.randn(10, 1),), (0.0, 10.0, 100, torch.float32)),
        "15": (lambda: (torch.randn(20),), (0.0, 15.0, 20, torch.float32)),
    }


class _LinspaceToFloatAdd(torch.nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        values = torch.linspace(0.0, 1.0, x.shape[0], dtype=torch.float64)
        return values.to(torch.float32) + x


@pytest.mark.parametrize(
    ("dtype", "expected"),
    ((torch.float32, True), (torch.float64, False)),
)
def test_linspace_preservation_depends_on_dtype(
    dtype: torch.dtype, expected: bool
) -> None:
    exported_program = torch.export.export(
        LinspaceAdd(0.0, 1.0, 10, dtype),
        (torch.randn(10),),
    )
    partitioner = TOSAPartitioner(TosaCompileSpec("TOSA-1.0+FP"))
    _, filter_fn = partitioner.ops_to_not_decompose(exported_program)
    linspace = next(
        node
        for node in exported_program.graph.nodes
        if node.target == torch.ops.aten.linspace.default
    )

    assert filter_fn is not None
    assert filter_fn(linspace) is expected


def test_ops_to_not_decompose_are_not_preserved_for_fp64() -> None:
    exported_program = torch.export.export(
        LinspaceAdd(0.0, 1.0, 10, torch.float32),
        (torch.randn(10),),
    )
    partitioner = TOSAPartitioner(TosaCompileSpec("TOSA-1.0+FP"))
    ops_to_not_decompose, filter_fn = partitioner.ops_to_not_decompose(exported_program)
    graph = torch.fx.Graph()
    fake_tensor = FakeTensorMode().from_tensor(torch.empty(1, dtype=torch.float64))

    assert filter_fn is not None
    for op in ops_to_not_decompose:
        node = graph.call_function(op)
        node.meta["val"] = fake_tensor
        assert not filter_fn(node), f"FP64 {op} should be decomposed"


def test_linspace_fp64_decomposes_for_portable_fallback() -> None:
    inputs = (torch.randn(10),)
    exported_program = torch.export.export(_LinspaceToFloatAdd(), inputs)
    partitioner = TOSAPartitioner(TosaCompileSpec("TOSA-1.0+FP"))

    edge_manager = to_edge_transform_and_lower(
        exported_program,
        partitioner=[partitioner],
    )
    targets = {
        node.target
        for node in edge_manager.exported_program().graph.nodes
        if node.op == "call_function"
    }

    assert exir_ops.edge.aten.linspace.default not in targets
    assert exir_ops.edge.aten.arange.start_step in targets

    program = edge_manager.to_executorch().executorch_program
    operators = {(op.name, op.overload) for op in program.execution_plan[0].operators}
    assert not any("linspace" in name for name, _ in operators)


@common.parametrize("test_data", LinspaceAdd.test_data)
def test_linspace_tosa_FP(test_data: test_data_t):
    input_data, init_data = test_data
    pipeline = TosaPipelineFP[input_t](
        LinspaceAdd(*init_data),
        input_data(),
        LinspaceAdd.aten_op,
        LinspaceAdd.exir_op,
    )
    pipeline.run()


@common.parametrize("test_data", LinspaceAdd.test_data)
def test_linspace_tosa_INT(test_data: test_data_t):
    input_data, init_data = test_data
    pipeline = TosaPipelineINT[input_t](
        LinspaceAdd(*init_data),
        input_data(),
        LinspaceAdd.aten_op,
        LinspaceAdd.exir_op,
    )
    pipeline.run()


@common.parametrize("test_data", LinspaceAdd.test_data)
@common.SkipIfNoModelConverter
def test_linspace_vgf_no_quant(test_data: test_data_t):
    input_data, init_data = test_data
    pipeline = VgfPipeline[input_t](
        LinspaceAdd(*init_data),
        input_data(),
        LinspaceAdd.aten_op,
        LinspaceAdd.exir_op,
        quantize=False,
    )
    pipeline.run()


@common.parametrize("test_data", LinspaceAdd.test_data)
@common.SkipIfNoModelConverter
def test_linspace_vgf_quant(test_data: test_data_t):
    input_data, init_data = test_data
    pipeline = VgfPipeline[input_t](
        LinspaceAdd(*init_data),
        input_data(),
        LinspaceAdd.aten_op,
        LinspaceAdd.exir_op,
        quantize=True,
    )
    pipeline.run()


skip_str = "aten.arange.default is decomposed to aten.arange.start_step, so it will never exist in a lowered graph."


@pytest.mark.skip(reason=skip_str)
def test_arange_tosa_FP():
    pass


@pytest.mark.skip(reason=skip_str)
def test_arange_tosa_INT():
    pass


@pytest.mark.skip(reason=skip_str)
def test_arange_u55_INT():
    pass


@pytest.mark.skip(reason=skip_str)
def test_arange_u85_INT():
    pass


@pytest.mark.skip(reason=skip_str)
def test_arange_vgf_no_quant():
    pass


@pytest.mark.skip(reason=skip_str)
def test_arange_vgf_quant():
    pass
