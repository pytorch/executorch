# Copyright 2025 Arm Limited and/or its affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import logging
from typing import Tuple

import torch
from executorch.backends.arm.test import common
from executorch.backends.arm.test.tester.test_pipeline import TosaPipelineFP
from executorch.exir.backend.operator_support import (
    DontPartition,
    DontPartitionModule,
    DontPartitionName,
)
from executorch.exir.dialects._ops import ops as exir_ops

input_t1 = Tuple[torch.Tensor, torch.Tensor]  # Input x, y


class CustomPartitioning(torch.nn.Module):
    inputs = {
        "randn": (torch.randn(10, 4, 5), torch.randn(10, 4, 5)),
    }

    def forward(self, x: torch.Tensor, y: torch.Tensor):
        z = x + y
        s = torch.sigmoid(z)
        return s * z


class NestedModule(torch.nn.Module):
    inputs = {
        "randn": (torch.randn(10, 4, 5), torch.randn(10, 4, 5)),
    }

    def __init__(self):
        super().__init__()
        self.nested = CustomPartitioning()

    def forward(self, x: torch.Tensor, y: torch.Tensor):
        a = x.sigmoid()
        b = a + y
        return self.nested(a, b)


@common.parametrize("test_data", CustomPartitioning.inputs)
def test_single_reject_tosa_FP(caplog, test_data: input_t1):
    caplog.set_level(logging.INFO)

    module = CustomPartitioning()
    pipeline = TosaPipelineFP[input_t1](module, test_data, [], exir_op=[])
    check = DontPartition(exir_ops.edge.aten.sigmoid.default)
    pipeline.change_args("to_edge_transform_and_lower", additional_checks=[check])
    pipeline.change_args(
        "check_count.exir", {"torch.ops.higher_order.executorch_call_delegate": 2}
    )
    pipeline.change_args(
        "check_count.exir",
        {"executorch_exir_dialects_edge__ops_aten_sigmoid_default": 1},
    )
    pipeline.run()
    assert check.has_rejected_node()
    assert "Rejected by DontPartition" in caplog.text


@common.parametrize("test_data", CustomPartitioning.inputs)
def test_multiple_reject_tosa_FP(test_data: input_t1):
    module = CustomPartitioning()
    pipeline = TosaPipelineFP[input_t1](module, test_data, [], exir_op=[])
    check = DontPartition(
        exir_ops.edge.aten.sigmoid.default, exir_ops.edge.aten.mul.Tensor
    )
    pipeline.change_args("to_edge_transform_and_lower", additional_checks=[check])
    pipeline.change_args(
        "check_count.exir", {"torch.ops.higher_order.executorch_call_delegate": 2}
    )
    pipeline.change_args(
        "check_count.exir",
        {"executorch_exir_dialects_edge__ops_aten_sigmoid_default": 1},
    )
    pipeline.run()
    assert check.has_rejected_node()


@common.parametrize("test_data", CustomPartitioning.inputs)
def test_torch_op_reject_tosa_FP(caplog, test_data: input_t1):
    caplog.set_level(logging.INFO)

    module = CustomPartitioning()
    check = DontPartition(torch.ops.aten.sigmoid.default)
    pipeline = TosaPipelineFP[input_t1](module, test_data, [], exir_op=[])
    pipeline.change_args("to_edge_transform_and_lower", additional_checks=[check])
    pipeline.change_args(
        "check_count.exir", {"torch.ops.higher_order.executorch_call_delegate": 2}
    )
    pipeline.change_args(
        "check_count.exir",
        {"executorch_exir_dialects_edge__ops_aten_sigmoid_default": 1},
    )
    pipeline.run()
    assert check.has_rejected_node()
    assert "Rejected by DontPartition" in caplog.text


@common.parametrize("test_data", CustomPartitioning.inputs)
def test_string_op_reject_tosa_FP(test_data: input_t1):
    module = CustomPartitioning()
    check = DontPartition("aten.sigmoid.default")
    pipeline = TosaPipelineFP[input_t1](module, test_data, [], exir_op=[])
    pipeline.change_args("to_edge_transform_and_lower", additional_checks=[check])
    pipeline.change_args(
        "check_count.exir", {"torch.ops.higher_order.executorch_call_delegate": 2}
    )
    pipeline.change_args(
        "check_count.exir",
        {"executorch_exir_dialects_edge__ops_aten_sigmoid_default": 1},
    )
    pipeline.run()
    assert check.has_rejected_node()


@common.parametrize("test_data", CustomPartitioning.inputs)
def test_name_reject_tosa_FP(caplog, test_data: input_t1):
    caplog.set_level(logging.INFO)

    module = CustomPartitioning()
    check = DontPartitionName("mul", "sigmoid", exact=False)
    pipeline = TosaPipelineFP[input_t1](module, test_data, [], exir_op=[])
    pipeline.change_args("to_edge_transform_and_lower", additional_checks=[check])
    pipeline.change_args(
        "check_count.exir",
        {"executorch_exir_dialects_edge__ops_aten_sigmoid_default": 1},
    )
    pipeline.run()
    assert check.has_rejected_node()
    assert "Rejected by DontPartitionName" in caplog.text


@common.parametrize("test_data", CustomPartitioning.inputs)
def test_module_reject_tosa_FP(test_data: input_t1):
    module = NestedModule()
    check = DontPartitionModule(module_name="CustomPartitioning")
    pipeline = TosaPipelineFP[input_t1](module, test_data, [], exir_op=[])
    pipeline.change_args("to_edge_transform_and_lower", additional_checks=[check])
    pipeline.change_args(
        "check_count.exir",
        {"executorch_exir_dialects_edge__ops_aten_sigmoid_default": 1},
    )
    pipeline.run()
    assert check.has_rejected_node()


@common.parametrize("test_data", CustomPartitioning.inputs)
def test_inexact_module_reject_tosa_FP(caplog, test_data: input_t1):
    caplog.set_level(logging.INFO)

    module = NestedModule()
    check = DontPartitionModule(module_name="Custom", exact=False)
    pipeline = TosaPipelineFP[input_t1](module, test_data, [], exir_op=[])
    pipeline.change_args("to_edge_transform_and_lower", additional_checks=[check])
    pipeline.change_args(
        "check_count.exir",
        {"executorch_exir_dialects_edge__ops_aten_sigmoid_default": 1},
    )
    pipeline.run()
    assert check.has_rejected_node()
    assert "Rejected by DontPartitionModule" in caplog.text


@common.parametrize("test_data", CustomPartitioning.inputs)
def test_module_instance_reject_tosa_FP(test_data: input_t1):
    module = NestedModule()
    check = DontPartitionModule(instance_name="nested")
    pipeline = TosaPipelineFP[input_t1](module, test_data, [], exir_op=[])
    pipeline.change_args("to_edge_transform_and_lower", additional_checks=[check])
    pipeline.change_args(
        "check_count.exir",
        {"executorch_exir_dialects_edge__ops_aten_sigmoid_default": 1},
    )
    pipeline.run()
    assert check.has_rejected_node()
