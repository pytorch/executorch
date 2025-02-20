# Copyright 2025 Arm Limited and/or its affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import torch
from executorch.backends.arm.test import common
from executorch.backends.arm.test.tester.arm_tester import ArmTester
from executorch.backends.arm.tosa_partitioner import TOSAPartitioner
from executorch.exir.backend.operator_support import (
    DontPartition,
    DontPartitionModule,
    DontPartitionName,
)
from executorch.exir.dialects._ops import ops as exir_ops


class CustomPartitioning(torch.nn.Module):
    inputs = (torch.randn(10, 4, 5), torch.randn(10, 4, 5))

    def forward(self, x: torch.Tensor, y: torch.Tensor):
        z = x + y
        s = torch.sigmoid(z)
        return s * z


class NestedModule(torch.nn.Module):
    inputs = (torch.randn(10, 4, 5), torch.randn(10, 4, 5))

    def __init__(self):
        super().__init__()
        self.nested = CustomPartitioning()

    def forward(self, x: torch.Tensor, y: torch.Tensor):
        a = x.sigmoid()
        b = a + y
        return self.nested(a, b)


def test_single_reject():
    module = CustomPartitioning()
    inputs = module.inputs
    compile_spec = common.get_tosa_compile_spec("TOSA-0.80+MI")
    check = DontPartition(exir_ops.edge.aten.sigmoid.default)
    partitioner = TOSAPartitioner(compile_spec, additional_checks=[check])
    (
        ArmTester(
            module,
            example_inputs=inputs,
            compile_spec=compile_spec,
        )
        .export()
        .to_edge_transform_and_lower(partitioners=[partitioner])
        .check(["executorch_exir_dialects_edge__ops_aten_sigmoid_default"])
        .check_count({"torch.ops.higher_order.executorch_call_delegate": 2})
        .to_executorch()
        .run_method_and_compare_outputs(inputs=inputs)
    )
    assert check.has_rejected_node()


def test_multiple_reject():
    module = CustomPartitioning()
    inputs = module.inputs
    compile_spec = common.get_tosa_compile_spec("TOSA-0.80+MI")
    check = DontPartition(
        exir_ops.edge.aten.sigmoid.default, exir_ops.edge.aten.mul.Tensor
    )
    partitioner = TOSAPartitioner(compile_spec, additional_checks=[check])
    (
        ArmTester(
            module,
            example_inputs=inputs,
            compile_spec=compile_spec,
        )
        .export()
        .to_edge_transform_and_lower(partitioners=[partitioner])
        .check(["executorch_exir_dialects_edge__ops_aten_sigmoid_default"])
        .check_count({"torch.ops.higher_order.executorch_call_delegate": 1})
        .to_executorch()
        .run_method_and_compare_outputs(inputs=inputs)
    )
    assert check.has_rejected_node()


def test_torch_op_reject():
    module = CustomPartitioning()
    inputs = module.inputs
    compile_spec = common.get_tosa_compile_spec("TOSA-0.80+MI")
    check = DontPartition(torch.ops.aten.sigmoid.default)
    partitioner = TOSAPartitioner(compile_spec, additional_checks=[check])
    (
        ArmTester(
            module,
            example_inputs=inputs,
            compile_spec=compile_spec,
        )
        .export()
        .to_edge_transform_and_lower(partitioners=[partitioner])
        .check(["executorch_exir_dialects_edge__ops_aten_sigmoid_default"])
        .check_count({"torch.ops.higher_order.executorch_call_delegate": 2})
        .to_executorch()
        .run_method_and_compare_outputs(inputs=inputs)
    )
    assert check.has_rejected_node()


def test_string_op_reject():
    module = CustomPartitioning()
    inputs = module.inputs
    compile_spec = common.get_tosa_compile_spec("TOSA-0.80+MI")
    check = DontPartition("aten.sigmoid.default")
    partitioner = TOSAPartitioner(compile_spec, additional_checks=[check])
    (
        ArmTester(
            module,
            example_inputs=inputs,
            compile_spec=compile_spec,
        )
        .export()
        .to_edge_transform_and_lower(partitioners=[partitioner])
        .check(["executorch_exir_dialects_edge__ops_aten_sigmoid_default"])
        .check_count({"torch.ops.higher_order.executorch_call_delegate": 2})
        .to_executorch()
        .run_method_and_compare_outputs(inputs=inputs)
    )

    assert check.has_rejected_node()


def test_name_reject():
    module = CustomPartitioning()
    inputs = module.inputs
    compile_spec = common.get_tosa_compile_spec("TOSA-0.80+MI")
    check = DontPartitionName("mul", "sigmoid", exact=False)
    partitioner = TOSAPartitioner(compile_spec, additional_checks=[check])
    (
        ArmTester(
            module,
            example_inputs=inputs,
            compile_spec=compile_spec,
        )
        .export()
        .to_edge_transform_and_lower(partitioners=[partitioner])
        .check(["executorch_exir_dialects_edge__ops_aten_sigmoid_default"])
        .check_count({"torch.ops.higher_order.executorch_call_delegate": 1})
        .to_executorch()
        .run_method_and_compare_outputs(inputs=inputs)
    )
    assert check.has_rejected_node()


def test_module_reject():
    module = NestedModule()
    inputs = module.inputs
    compile_spec = common.get_tosa_compile_spec("TOSA-0.80+MI")
    check = DontPartitionModule(module_name="CustomPartitioning")
    partitioner = TOSAPartitioner(compile_spec, additional_checks=[check])
    (
        ArmTester(
            module,
            example_inputs=inputs,
            compile_spec=compile_spec,
        )
        .export()
        .to_edge_transform_and_lower(partitioners=[partitioner])
        .check(["executorch_exir_dialects_edge__ops_aten_sigmoid_default"])
        .check_count({"torch.ops.higher_order.executorch_call_delegate": 1})
        .to_executorch()
        .run_method_and_compare_outputs(inputs=inputs)
    )
    assert check.has_rejected_node()


def test_inexact_module_reject():
    module = NestedModule()
    inputs = module.inputs
    compile_spec = common.get_tosa_compile_spec("TOSA-0.80+MI")
    check = DontPartitionModule(module_name="Custom", exact=False)
    partitioner = TOSAPartitioner(compile_spec, additional_checks=[check])
    (
        ArmTester(
            module,
            example_inputs=inputs,
            compile_spec=compile_spec,
        )
        .export()
        .to_edge_transform_and_lower(partitioners=[partitioner])
        .check(["executorch_exir_dialects_edge__ops_aten_sigmoid_default"])
        .check_count({"torch.ops.higher_order.executorch_call_delegate": 1})
        .to_executorch()
        .run_method_and_compare_outputs(inputs=inputs)
    )
    assert check.has_rejected_node()


def test_module_instance_reject():
    module = NestedModule()
    inputs = module.inputs
    compile_spec = common.get_tosa_compile_spec("TOSA-0.80+MI")
    check = DontPartitionModule(instance_name="nested")
    partitioner = TOSAPartitioner(compile_spec, additional_checks=[check])
    (
        ArmTester(
            module,
            example_inputs=inputs,
            compile_spec=compile_spec,
        )
        .export()
        .to_edge_transform_and_lower(partitioners=[partitioner])
        .check(["executorch_exir_dialects_edge__ops_aten_sigmoid_default"])
        .check_count({"torch.ops.higher_order.executorch_call_delegate": 1})
        .to_executorch()
        .run_method_and_compare_outputs(inputs=inputs)
    )
    assert check.has_rejected_node()
