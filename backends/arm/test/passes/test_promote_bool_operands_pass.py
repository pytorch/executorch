# Copyright 2025 Arm Limited and/or its affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from typing import ClassVar, Dict, Tuple

import torch
from executorch.backends.arm._passes import PromoteBoolOperandsPass

from executorch.backends.arm.test import common
from executorch.backends.arm.test.tester.test_pipeline import PassPipeline
from executorch.backends.test.harness.stages import StageType
from executorch.exir.dialects._ops import ops as exir_ops

tensor_pair_t = Tuple[torch.Tensor, torch.Tensor]


def _collect_cast_dtypes(pipeline: PassPipeline[tensor_pair_t]) -> list[torch.dtype]:
    exported_program = pipeline.tester.get_artifact(
        StageType.RUN_PASSES
    ).exported_program()
    graph_module = exported_program.graph_module
    cast_dtypes: list[torch.dtype] = []
    for node in graph_module.graph.nodes:
        if (
            node.op == "call_function"
            and node.target == exir_ops.edge.dim_order_ops._to_dim_order_copy.default
            and "dtype" in node.kwargs
        ):
            cast_dtypes.append(node.kwargs["dtype"])
    return cast_dtypes


class BoolBitwiseAndModule(torch.nn.Module):
    test_data: ClassVar[Dict[str, tensor_pair_t]] = {
        "bool_tensors": (
            torch.tensor([[True, False], [False, True]], dtype=torch.bool),
            torch.tensor([[False, True], [True, False]], dtype=torch.bool),
        )
    }

    def forward(self, lhs: torch.Tensor, rhs: torch.Tensor) -> torch.Tensor:
        return torch.bitwise_and(lhs, rhs)


class MixedMulModule(torch.nn.Module):
    test_data: ClassVar[Dict[str, tensor_pair_t]] = {
        "mixed_tensors": (
            torch.tensor([True, False, True, False], dtype=torch.bool),
            torch.tensor([1, 2, 3, 4], dtype=torch.int32),
        )
    }

    def forward(self, lhs: torch.Tensor, rhs: torch.Tensor) -> torch.Tensor:
        return torch.mul(lhs, rhs)


@common.parametrize("test_data", BoolBitwiseAndModule.test_data)
def test_promote_bool_operands_tosa_FP_all_bool(test_data: tensor_pair_t) -> None:
    module = BoolBitwiseAndModule()
    ops_before_pass = {
        "executorch_exir_dialects_edge__ops_aten_bitwise_and_Tensor": 1,
    }
    ops_after_pass = {
        "executorch_exir_dialects_edge__ops_aten_bitwise_and_Tensor": 1,
        "executorch_exir_dialects_edge__ops_dim_order_ops__to_dim_order_copy_default": 3,
    }
    pipeline = PassPipeline[tensor_pair_t](
        module,
        test_data,
        quantize=False,
        ops_before_pass=ops_before_pass,
        ops_after_pass=ops_after_pass,
        pass_list=[PromoteBoolOperandsPass],
    )
    pipeline.run()
    cast_dtypes = _collect_cast_dtypes(pipeline)
    assert cast_dtypes.count(torch.int8) == 2
    assert cast_dtypes.count(torch.bool) == 1


@common.parametrize("test_data", MixedMulModule.test_data)
def test_promote_bool_operands_tosa_FP_mixed_types(test_data: tensor_pair_t) -> None:
    module = MixedMulModule()
    ops_before_pass = {
        "executorch_exir_dialects_edge__ops_aten_mul_Tensor": 1,
    }
    ops_after_pass = {
        "executorch_exir_dialects_edge__ops_aten_mul_Tensor": 1,
        "executorch_exir_dialects_edge__ops_dim_order_ops__to_dim_order_copy_default": 1,
    }
    pipeline = PassPipeline[tensor_pair_t](
        module,
        test_data,
        quantize=False,
        ops_before_pass=ops_before_pass,
        ops_after_pass=ops_after_pass,
        pass_list=[PromoteBoolOperandsPass],
    )
    pipeline.run()
    cast_dtypes = _collect_cast_dtypes(pipeline)
    assert cast_dtypes.count(torch.int32) == 1
