# Copyright 2025-2026 Arm Limited and/or its affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.


from typing import Tuple

import pytest
import torch

from executorch.backends.arm._passes.decompose_index_select_to_gather_pass import (
    DecomposeIndexSelectToGatherPass,
)
from executorch.backends.arm.operator_support.ethos_u55_support import (
    EthosU55IndexSelectCheck,
)
from executorch.backends.arm.test import common
from executorch.backends.arm.test.tester.arm_tester import ArmTester
from executorch.backends.arm.test.tester.test_pipeline import (
    EthosU55PipelineINT,
    EthosU85PipelineINT,
    OpNotSupportedPipeline,
    TosaPipelineFP,
    TosaPipelineINT,
    VgfPipeline,
)

from executorch.exir.backend.utils import WhyNoPartitionReporter
from executorch.exir.dialects._ops import ops as exir_ops


class IndexSelect(torch.nn.Module):
    aten_op = "torch.ops.aten.index_select.default"
    exir_op = "executorch_exir_dialects_edge__ops_aten_index_select_default"

    def forward(self, input_: torch.Tensor, dim: int, index_: torch.Tensor):
        return torch.index_select(input_, dim=dim, index=index_)


class ConstantIndexSelect(torch.nn.Module):
    def __init__(self, dim: int, indices: list[int], dtype: torch.dtype = torch.int32):
        super().__init__()
        self.dim = dim
        self.register_buffer("indices", torch.tensor(indices, dtype=dtype))

    def forward(self, input_: torch.Tensor):
        return torch.index_select(input_, dim=self.dim, index=self.indices)


input_t1 = Tuple[torch.Tensor]
input_params = Tuple[torch.Tensor, int, torch.Tensor]

# ---- FP profile: only float inputs ----
test_data_fp: dict[str, input_params] = {
    # Rank-1: [K] -> index_select dim=0 => [W]
    "test_fp32_rank1_dim0": (
        torch.randn(6, dtype=torch.float32),  # [K=6]
        0,
        torch.tensor([1, 4, 5], dtype=torch.int32),  # [W=3]
    ),
    # Rank-2: [K, C] -> index_select dim=0 => [W, C]
    "test_fp32_rank2_dim0": (
        torch.randn(4, 3, dtype=torch.float32),  # [K=4, C=3]
        0,
        torch.tensor([1, 3], dtype=torch.int32),  # [W=2]
    ),
    # Rank-3: [N, K, C] -> index_select dim=-1 => [N, K, W]
    "test_fp32_rank3_dim_neg1": (
        torch.randn(2, 4, 3, dtype=torch.float32),  # [N=2, K=4, C=3]
        -1,
        torch.tensor([2, 0], dtype=torch.int32),  # [W=2]
    ),
    # Rank-3: [N, K, C] -> index_select dim=1 => [N, W, C]
    "test_fp32_rank3_dim1": (
        torch.randn(2, 4, 3, dtype=torch.float32),  # [N=2, K=4, C=3]
        1,
        torch.tensor([1, 3], dtype=torch.int32),  # [W=2]
    ),
    # Rank-4: [A, B, K, C] -> index_select dim=2 => [A, B, W, C]
    "test_fp32_rank4_dim2": (
        torch.randn(2, 3, 4, 5, dtype=torch.float32),  # [A=2, B=3, K=4, C=5]
        2,
        torch.tensor([3, 1], dtype=torch.int32),  # [W=2]
    ),
}
test_data_fp_bf16: dict[str, input_params] = {
    # Rank-2: [K, C] -> index_select dim=0 => [W, C]
    "test_bf16_rank2_dim0": (
        torch.tensor(
            [[0.5, 1.25, 2.5], [3.5, 4.25, 5.75], [6.5, 7.25, 8.75]],
            dtype=torch.bfloat16,
        ),  # [K=3, C=3]
        0,
        torch.tensor([2, 0], dtype=torch.int32),  # [W=2]
    ),
    # Rank-3: [N, K, C] -> index_select dim=-1 => [N, K, W]
    "test_bf16_rank3_dim_neg1": (
        torch.tensor(
            [[[0.5, 1.5], [2.5, 3.5]], [[4.5, 5.5], [6.5, 7.5]]],
            dtype=torch.bfloat16,
        ),  # [N=2, K=2, C=2]
        -1,
        torch.tensor([1, 0], dtype=torch.int32),  # [W=2]
    ),
}
test_data_fp8: dict[str, input_params] = {
    # Rank-3: [N, K, C] -> index_select dim=1 => [N, W, C]
    "test_fp8e4m3_rank3_dim1": (
        torch.randn(2, 4, 3, dtype=torch.float32).to(
            torch.float8_e4m3fn
        ),  # [N=2, K=4, C=3]
        1,
        torch.tensor([1, 3], dtype=torch.int32),  # [W=2]
        "fp8e4m3",
    ),
    # Rank-4: [A, B, K, C] -> index_select dim=2 => [A, B, W, C]
    "test_fp8e5m2_rank4_dim2": (
        torch.randn(2, 3, 4, 5, dtype=torch.float32).to(
            torch.float8_e5m2
        ),  # [A=2, B=3, K=4, C=5]
        2,
        torch.tensor([3, 1], dtype=torch.int32),  # [W=2]
        "fp8e5m2",
    ),
}

# ---- INT profile: integer inputs + bool ----
test_data_int: dict[str, input_params] = {
    # Rank-1 int8: [K] -> index_select dim=0 => [W]
    "test_int8_rank1_dim0": (
        torch.randint(-6, 6, size=(6,), dtype=torch.int8),  # [K=6]
        0,
        torch.tensor([5, 0, 2], dtype=torch.int32),  # [W=3]
    ),
    # Rank-2 bool: [K, C] -> index_select dim=0 => [W, C]
    "test_bool_rank2_dim0": (
        torch.randint(0, 2, size=(3, 2), dtype=torch.int8).to(torch.bool),  # [K=3, C=2]
        0,
        torch.tensor([2, 0], dtype=torch.int32),  # [W=2]
    ),
    # Rank-3 int8: [N, K, C] -> index_select dim=1 => [N, W, C]
    "test_int8_rank3_dim1": (
        torch.randint(-5, 5, size=(2, 7, 4), dtype=torch.int8),  # [N=2, K=7, C=4]
        1,
        torch.tensor([0, 6, 3], dtype=torch.int32),  # [W=3]
    ),
    # Rank-4 int32: [A, B, K, C] -> index_select dim=2 => [A, B, W, C]
    "test_int32_rank4_dim2": (
        torch.randint(
            -20, 20, size=(2, 3, 5, 4), dtype=torch.int32
        ),  # [A=2, B=3, K=5, C=4]
        2,
        torch.tensor([4, 1], dtype=torch.int32),  # [W=2]
    ),
}


@common.parametrize("test_data", test_data_fp)
def test_index_select_tosa_FP(test_data: input_params):
    pipeline = TosaPipelineFP[input_params](
        IndexSelect(),
        test_data,
        aten_op=IndexSelect.aten_op,
        exir_op=IndexSelect.exir_op,
    )
    pipeline.run()


@common.parametrize("test_data", test_data_fp_bf16)
def test_index_select_tosa_FP_bf16(test_data: input_params):
    pipeline = TosaPipelineFP[input_params](
        IndexSelect(),
        test_data,
        aten_op=IndexSelect.aten_op,
        exir_op=IndexSelect.exir_op,
        tosa_extensions=["bf16"],
    )
    pipeline.run()


@common.parametrize("test_data", test_data_fp8)
def test_index_select_tosa_FP_fp8(test_data):
    input_, dim, index_, tosa_extension = test_data
    pipeline = TosaPipelineFP[input_params](
        IndexSelect(),
        (input_, dim, index_),
        aten_op=IndexSelect.aten_op,
        exir_op=IndexSelect.exir_op,
        compare_tosa_ref_model_outputs=False,
        tosa_extensions=[tosa_extension],
    )
    pipeline.run()


@common.parametrize("test_data", test_data_int | test_data_fp)
def test_index_select_tosa_INT(test_data: input_params):
    # INT profile runs quantized, so we test both int inputs and float inputs here.
    pipeline = TosaPipelineINT[input_params](
        IndexSelect(),
        test_data,
        aten_op=IndexSelect.aten_op,
        exir_op=IndexSelect.exir_op,
    )
    pipeline.run()


@common.parametrize("test_data", test_data_int | test_data_fp)
def test_index_select_u55_INT_not_delegated(test_data: input_params):
    pipeline = OpNotSupportedPipeline[input_params](
        IndexSelect(),
        test_data,
        {IndexSelect.exir_op: 1},
        quantize=True,
        u55_subset=True,
    )
    pipeline.run()


@common.parametrize("test_data", test_data_int | test_data_fp)
def test_index_select_u85_INT(test_data: input_params):
    pipeline = EthosU85PipelineINT[input_params](
        IndexSelect(),
        test_data,
        aten_ops=IndexSelect.aten_op,
        exir_ops=IndexSelect.exir_op,
    )
    pipeline.run()


@common.parametrize("test_data", test_data_fp | test_data_fp_bf16 | test_data_int)
@common.SkipIfNoModelConverter
def test_index_select_vgf_no_quant(test_data: input_params):
    pipeline = VgfPipeline[input_params](
        IndexSelect(),
        test_data,
        aten_op=IndexSelect.aten_op,
        exir_op=IndexSelect.exir_op,
        quantize=False,
    )
    pipeline.run()


@common.parametrize("test_data", test_data_fp | test_data_int)
@common.SkipIfNoModelConverter
def test_index_select_vgf_quant(test_data: input_params):
    pipeline = VgfPipeline[input_params](
        IndexSelect(),
        test_data,
        aten_op=IndexSelect.aten_op,
        exir_op=IndexSelect.exir_op,
        quantize=True,
    )
    pipeline.run()


@common.XfailIfNoCorstone300
def test_index_select_u55_INT_constant_contiguous():
    pipeline = EthosU55PipelineINT[input_t1](
        ConstantIndexSelect(2, [1, 2, 3]),
        (torch.rand(1, 2, 5, 3),),
        aten_ops=[],
        exir_ops=[],
    )
    pipeline.run()


@common.XfailIfNoCorstone300
def test_index_select_u55_INT_constant_contiguous_negative_dim():
    pipeline = EthosU55PipelineINT[input_t1](
        ConstantIndexSelect(-1, [1, 2]),
        (torch.rand(1, 2, 4, 5),),
        aten_ops=[],
        exir_ops=[],
    )
    pipeline.run()


@common.parametrize(
    "indices",
    {
        "noncontiguous": [1, 3],
        "descending": [3, 1],
        "duplicate": [1, 1],
    },
)
@common.XfailIfNoCorstone300
def test_index_select_u55_INT_constant_slices_concat(indices):
    pipeline = EthosU55PipelineINT[input_t1](
        ConstantIndexSelect(2, indices),
        (torch.rand(1, 2, 5, 3),),
        aten_ops=[],
        exir_ops=[],
    )
    pipeline.run()


def test_index_select_u55_INT_constant_contiguous_symbolic_dim_not_delegated():
    selected_dim = torch.export.Dim("selected_dim", min=4, max=8)
    tester = ArmTester(
        ConstantIndexSelect(2, [1, 2, 3]),
        (torch.rand(1, 2, 5, 3),),
        common.get_u55_compile_spec(),
        dynamic_shapes={"input_": {2: selected_dim}},
    )
    tester.quantize().export().to_edge().partition()

    targets = {
        node.target
        for node in tester.stages[tester.cur].artifact.exported_program().graph.nodes
    }
    assert exir_ops.edge.aten.index_select.default in targets
    assert torch.ops.higher_order.executorch_call_delegate not in targets


@common.parametrize(
    "indices",
    {
        "negative_index": [-1],
        "upper_bound_index": [5],
    },
)
def test_index_select_u55_constant_out_of_bounds_raises(indices):
    tester = ArmTester(
        ConstantIndexSelect(2, indices),
        (torch.rand(1, 2, 5, 3),),
        common.get_u55_compile_spec(),
    )
    tester.export().to_edge()
    exported_program = tester.stages[tester.cur].artifact.exported_program()

    with pytest.raises(RuntimeError, match="index_select index out of range"):
        DecomposeIndexSelectToGatherPass(exported_program).call(
            exported_program.graph_module
        )


def test_index_select_u55_scalar_not_supported():
    tester = ArmTester(
        ConstantIndexSelect(0, [0]),
        (torch.tensor(1.0),),
        common.get_u55_compile_spec(),
    )
    tester.export().to_edge()
    exported_program = tester.stages[tester.cur].artifact.exported_program()
    index_select_node = next(
        node
        for node in exported_program.graph.nodes
        if node.target == exir_ops.edge.aten.index_select.default
    )

    assert not EthosU55IndexSelectCheck(
        exported_program, WhyNoPartitionReporter()
    ).is_node_supported({}, index_select_node)


def test_index_select_u55_INT_constant_int64_delegated():
    tester = ArmTester(
        ConstantIndexSelect(2, [1, 2, 3], torch.int64),
        (torch.rand(1, 2, 5, 3),),
        common.get_u55_compile_spec(),
    )
    tester.quantize().export().to_edge().partition()

    targets = {
        node.target
        for node in tester.stages[tester.cur].artifact.exported_program().graph.nodes
    }
    assert torch.ops.higher_order.executorch_call_delegate in targets


@common.XfailIfNoCorstone300
def test_index_select_u55_INT_constant_contiguous_a16w8():
    pipeline = EthosU55PipelineINT[input_t1](
        ConstantIndexSelect(2, [1, 2, 3]),
        (torch.rand(1, 2, 5, 3),),
        aten_ops=[],
        exir_ops=[],
        a16w8_quantization=True,
    )
    pipeline.run()


def test_index_select_u55_INT_constant_empty_not_delegated():
    pipeline = OpNotSupportedPipeline[input_t1](
        ConstantIndexSelect(2, []),
        (torch.rand(1, 2, 5, 3),),
        {IndexSelect.exir_op: 1},
        quantize=True,
        u55_subset=True,
    )
    pipeline.run()
