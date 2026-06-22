# Copyright 2026 Arm Limited and/or its affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from typing import List, Tuple

import executorch.backends.arm.tosa.dialect  # noqa: F401
import torch
from executorch.backends.arm.test import common
from executorch.backends.arm.test.tester.test_pipeline import TosaPipelineFP
from executorch.backends.arm.tosa.backend import TOSABackend
from executorch.backends.arm.tosa.specification import TosaSpecification
from executorch.exir import to_edge
from executorch.exir.dialects._ops import ops as exir_ops

input_t1 = Tuple[torch.Tensor]


class AllDim(torch.nn.Module):
    aten_op = "torch.ops.aten.all.dim"
    exir_op = "executorch_exir_dialects_edge__ops_aten_all_dim"

    def forward(self, x: torch.Tensor, dim: int, keepdim: bool):
        return torch.all(x, dim=dim, keepdim=keepdim)


class AllDims(torch.nn.Module):
    aten_op = "torch.ops.aten.all.dims"
    exir_op = "executorch_exir_dialects_edge__ops_aten_all_dims"

    def forward(self, x: torch.Tensor, dim: List[int], keepdim: bool):
        return torch.all(x, dim=dim, keepdim=keepdim)


class AllReduceAll(torch.nn.Module):
    aten_op = "torch.ops.aten.all.default"
    exir_op = "executorch_exir_dialects_edge__ops_aten_all_default"

    def forward(self, x: torch.Tensor):
        return torch.all(x)


class ReduceAll(torch.nn.Module):
    def __init__(self, axis: int):
        super().__init__()
        self.axis = axis

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return exir_ops.backend.tosa.REDUCE_ALL.default(x, axis=self.axis)


test_input: dict[input_t1] = {
    "rank1": lambda: (torch.tensor([True, False, False], dtype=torch.bool), 0, True),
    "rank1_squeeze": lambda: (
        torch.tensor([True, False, False], dtype=torch.bool),
        -1,
        False,
    ),
    "rank2_dims": lambda: (
        torch.randint(0, 2, (2, 3), dtype=torch.bool),
        [0, 1],
        True,
    ),
    "rank2_dims_squeeze": lambda: (
        torch.randint(0, 2, (2, 3), dtype=torch.bool),
        [-2, 1],
        False,
    ),
    "rank4": lambda: (
        torch.randint(0, 2, (1, 6, 8, 10), dtype=torch.bool),
        1,
        True,
    ),
    "rank4_dims_squeeze": lambda: (
        torch.randint(0, 2, (1, 6, 8, 10), dtype=torch.bool),
        [1, -1],
        False,
    ),
    "rank1_reduce_all": lambda: (torch.tensor([True, False, False], dtype=torch.bool),),
    "rank4_reduce_all": lambda: (torch.randint(0, 2, (1, 6, 8, 10), dtype=torch.bool),),
}


all_test_data = {
    "all_rank1": lambda: (AllDim(), test_input["rank1"]),
    "all_rank1_squeeze": lambda: (AllDim(), test_input["rank1_squeeze"]),
    "all_rank2_dims": lambda: (AllDims(), test_input["rank2_dims"]),
    "all_rank2_dims_squeeze": lambda: (AllDims(), test_input["rank2_dims_squeeze"]),
    "all_rank4": lambda: (AllDim(), test_input["rank4"]),
    "all_rank4_dims_squeeze": lambda: (AllDims(), test_input["rank4_dims_squeeze"]),
    "all_rank1_reduce_all": lambda: (AllReduceAll(), test_input["rank1_reduce_all"]),
    "all_rank4_reduce_all": lambda: (AllReduceAll(), test_input["rank4_reduce_all"]),
}


@common.parametrize("test_data", all_test_data)
def test_all_tosa_FP(test_data: input_t1):
    op, test_input = test_data()
    pipeline = TosaPipelineFP[input_t1](
        op,
        test_input(),
        op.aten_op,
        op.exir_op,
        atol=0,
        rtol=0,
        qtol=0,
    )
    pipeline.run()


test_parameters = {
    "rank1_dim0": lambda: (torch.randint(0, 2, (4,), dtype=torch.bool), 0),
    "rank2_dim1": lambda: (torch.randint(0, 2, (2, 3), dtype=torch.bool), 1),
    "rank4_dim2": lambda: (torch.randint(0, 2, (1, 2, 3, 4), dtype=torch.bool), 2),
}


@common.parametrize("test_data", test_parameters)
def test_reduce_all_tosa_FP(test_data: input_t1) -> None:
    x, axis = test_data()
    exported_program = to_edge(
        torch.export.export(ReduceAll(axis), (x,), strict=True)
    ).exported_program()
    compile_spec = common.get_tosa_compile_spec(
        TosaSpecification.create_from_string("TOSA-1.0+FP")
    )

    result = TOSABackend._preprocess(exported_program, compile_spec)

    assert result.processed_bytes
