# Copyright 2026 Arm Limited and/or its affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import executorch.backends.arm.operator_support.sym_size_int_support  # noqa: F401
import torch

from executorch.backends.arm.operator_support.tosa_supported_operators import (
    tosa_support_factory,
)
from executorch.backends.arm.tosa.compile_spec import TosaCompileSpec
from executorch.backends.arm.tosa.partitioner import TOSAPartitioner
from executorch.backends.arm.tosa.specification import TosaSpecification
from executorch.exir import EdgeCompileConfig, to_edge
from executorch.exir.backend.utils import WhyNoPartitionReporter
from executorch.exir.dialects._ops import ops as exir_ops
from torch.export import Dim, export


class Add(torch.nn.Module):
    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        return x + y


class Atan2(torch.nn.Module):
    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        return torch.atan2(x, y)


class ReturnSymSize(torch.nn.Module):
    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, int]:
        return x, x.shape[0]


def _exported_program(
    module: torch.nn.Module,
    inputs: tuple[torch.Tensor, ...],
    dynamic_shapes=None,
):
    return to_edge(
        export(module, inputs, dynamic_shapes=dynamic_shapes, strict=True),
        compile_config=EdgeCompileConfig(_check_ir_validity=False),
    ).exported_program()


def _support(tosa_spec: str, exported_program):
    reporter = WhyNoPartitionReporter()
    return (
        tosa_support_factory(
            TosaSpecification.create_from_string(tosa_spec),
            exported_program,
            reporter,
        ),
        reporter,
    )


def _find_node(exported_program, target):
    return exported_program.graph_module.graph.find_nodes(
        op="call_function", target=target
    )[0]


def test_shape_extension_does_not_accept_unsupported_static_op():
    inputs = (torch.randn(2, 3), torch.randn(2, 3))
    exported_program = _exported_program(Atan2(), inputs)
    support, reporter = _support("TOSA-1.1+FP+shape", exported_program)
    atan2_node = _find_node(exported_program, exir_ops.edge.aten.atan2.default)

    assert support.is_node_supported(exported_program.graph_module, atan2_node) is False
    assert "Not included in BaseTOSASupportList" in reporter.get_table_report()


def test_shape_extension_accepts_supported_symbolic_tensor_op():
    inputs = (torch.randn(2, 3), torch.randn(2, 3))
    batch = Dim("batch", min=1, max=4)
    exported_program = _exported_program(
        Add(),
        inputs,
        dynamic_shapes=({0: batch}, {0: batch}),
    )
    support, _ = _support("TOSA-1.1+FP+shape", exported_program)
    add_node = _find_node(exported_program, exir_ops.edge.aten.add.Tensor)

    assert support.is_node_supported(exported_program.graph_module, add_node) is True


def test_without_shape_extension_rejects_supported_symbolic_tensor_op():
    inputs = (torch.randn(2, 3), torch.randn(2, 3))
    batch = Dim("batch", min=1, max=4)
    exported_program = _exported_program(
        Add(),
        inputs,
        dynamic_shapes=({0: batch}, {0: batch}),
    )
    support, reporter = _support("TOSA-1.0+FP", exported_program)
    add_node = _find_node(exported_program, exir_ops.edge.aten.add.Tensor)

    assert support.is_node_supported(exported_program.graph_module, add_node) is False
    assert "Node has symbolic shape" in reporter.get_table_report()


def test_without_shape_extension_rejects_sym_size_int():
    inputs = (torch.randn(2, 3),)
    batch = Dim("batch", min=1, max=4)
    exported_program = _exported_program(
        ReturnSymSize(),
        inputs,
        dynamic_shapes=({0: batch},),
    )
    support, _ = _support("TOSA-1.1+FP", exported_program)
    sym_size_node = _find_node(exported_program, torch.ops.aten.sym_size.int)

    assert (
        support.is_node_supported(exported_program.graph_module, sym_size_node) is False
    )


def test_shape_extension_rejects_sym_size_from_fp32_for_int_spec():
    inputs = (torch.randn(2, 3),)
    batch = Dim("batch", min=1, max=4)
    exported_program = _exported_program(
        ReturnSymSize(),
        inputs,
        dynamic_shapes=({0: batch},),
    )
    support, reporter = _support("TOSA-1.1+INT+shape", exported_program)
    sym_size_node = _find_node(exported_program, torch.ops.aten.sym_size.int)

    assert (
        support.is_node_supported(exported_program.graph_module, sym_size_node) is False
    )
    assert "Node was not marked as quantized" in reporter.get_table_report()


def test_shape_extension_rejects_sym_size_from_int64_without_int64_extension():
    inputs = (torch.ones(2, 3, dtype=torch.int64),)
    batch = Dim("batch", min=1, max=4)
    exported_program = _exported_program(
        ReturnSymSize(),
        inputs,
        dynamic_shapes=({0: batch},),
    )
    support, reporter = _support("TOSA-1.1+FP+shape", exported_program)
    sym_size_node = _find_node(exported_program, torch.ops.aten.sym_size.int)

    assert (
        support.is_node_supported(exported_program.graph_module, sym_size_node) is False
    )
    assert "Non-constant int64 input" in reporter.get_table_report()


def test_shape_extension_partitions_sym_size_int():
    inputs = (torch.randn(2, 3),)
    batch = Dim("batch", min=1, max=4)
    exported_program = _exported_program(
        ReturnSymSize(),
        inputs,
        dynamic_shapes=({0: batch},),
    )

    partition_result = TOSAPartitioner(TosaCompileSpec("TOSA-1.1+FP+shape")).partition(
        exported_program
    )
    sym_size_node = _find_node(
        partition_result.tagged_exported_program, torch.ops.aten.sym_size.int
    )

    assert sym_size_node.meta.get("delegation_tag") in partition_result.partition_tags
