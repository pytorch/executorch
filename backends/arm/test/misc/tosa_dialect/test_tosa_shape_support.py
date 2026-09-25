# Copyright 2026 Arm Limited and/or its affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import operator

import executorch.backends.arm.operator_support.convolution_support  # noqa: F401
import executorch.backends.arm.operator_support.pool_2d_support  # noqa: F401
import executorch.backends.arm.operator_support.reduce_sum_support  # noqa: F401

import executorch.backends.arm.operator_support.sym_size_int_support  # noqa: F401
import pytest
import torch
from executorch.backends.arm.operator_support.slice_copy_support import (
    SliceCopySupported,
)
from executorch.backends.arm.operator_support.symint_arithmetic_support import (
    SymIntArithmeticSupport,
)
from executorch.backends.arm.operator_support.tosa_supported_operators import (
    tosa_support_factory,
)
from executorch.backends.arm.test import common
from executorch.backends.arm.test.tester.arm_tester import ArmTester
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


class Conv2d(torch.nn.Module):
    def __init__(self, stride: int = 1, padding: int = 1) -> None:
        super().__init__()
        self.conv = torch.nn.Conv2d(3, 4, 3, padding=padding, stride=stride)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)


class AvgPool2d(torch.nn.Module):
    def __init__(
        self, kernel_size: int = 2, stride: int | None = None, padding: int = 0
    ) -> None:
        super().__init__()
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.stride is None:
            return torch.nn.functional.avg_pool2d(
                x, kernel_size=self.kernel_size, padding=self.padding
            )
        return torch.nn.functional.avg_pool2d(
            x, self.kernel_size, self.stride, self.padding
        )


class MaxPool2d(torch.nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.nn.functional.max_pool2d(x, kernel_size=2)


class MaxPool2dEmptyStride(torch.nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.nn.functional.max_pool2d(x, kernel_size=2, stride=[])


class MeanDim(torch.nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x.mean(dim=2)


class MeanDefault(torch.nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x.mean()


class Squeeze(torch.nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x.squeeze(2)


class Unsqueeze(torch.nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x.unsqueeze(2)


class Slice(torch.nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x[:, :, 1:, :]


class ScalarTensor(torch.nn.Module):
    def forward(self) -> torch.Tensor:
        return torch.scalar_tensor(1.0)


class ReturnSymSize(torch.nn.Module):
    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, int]:
        return x, x.shape[0]


class ReshapeWithSymSize(torch.nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x.reshape(x.shape[0], 6)


class ReturnSymSizeArithmetic(torch.nn.Module):
    def __init__(self, operation) -> None:
        super().__init__()
        self.operation = operation

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, int]:
        return x, self.operation(x.shape[0], 2)


class Index(torch.nn.Module):
    def forward(self, x: torch.Tensor, indices: torch.Tensor) -> torch.Tensor:
        return x[indices]


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


def test_registered_custom_op_overrides_tosa_support_checks():
    inputs = (torch.randn(2, 3), torch.randn(2, 3))
    exported_program = _exported_program(Atan2(), inputs)
    partitioner = TOSAPartitioner(TosaCompileSpec("TOSA-1.0+FP"))
    partitioner.register_custom_partition_op(torch.ops.aten.atan2.default)

    partition_result = partitioner.partition(exported_program)
    atan2_node = _find_node(
        partition_result.tagged_exported_program, exir_ops.edge.aten.atan2.default
    )

    assert atan2_node.meta.get("delegation_tag") in partition_result.partition_tags


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


def test_without_shape_extension_accepts_supported_symbolic_tensor_op():
    inputs = (torch.randn(2, 3), torch.randn(2, 3))
    batch = Dim("batch", min=1, max=4)
    exported_program = _exported_program(
        Add(),
        inputs,
        dynamic_shapes=({0: batch}, {0: batch}),
    )
    support, _ = _support("TOSA-1.0+FP", exported_program)
    add_node = _find_node(exported_program, exir_ops.edge.aten.add.Tensor)

    assert support.is_node_supported(exported_program.graph_module, add_node) is True


def _assert_rejected_with_reason(exported_program, target, reason: str) -> None:
    support, reporter = _support("TOSA-1.0+FP", exported_program)
    node = _find_node(exported_program, target)

    assert support.is_node_supported(exported_program.graph_module, node) is False
    assert reason in reporter.get_table_report()


@pytest.mark.parametrize(
    "compile_spec",
    (common.get_u55_compile_spec(), common.get_u85_compile_spec()),
)
def test_ethos_rejects_unresolved_tensor_shapes(compile_spec):
    inputs = (torch.randn(2, 3), torch.randn(2, 3))
    batch = Dim("batch", min=1, max=4)
    tester = ArmTester(
        Add(),
        inputs,
        compile_spec,
        dynamic_shapes=({0: batch}, {0: batch}),
    )

    tester.quantize().export().to_edge().partition()
    targets = {
        node.target
        for node in tester.stages[tester.cur].artifact.exported_program().graph.nodes
    }

    assert exir_ops.edge.aten.add.Tensor in targets
    assert torch.ops.higher_order.executorch_call_delegate not in targets


def test_without_shape_extension_accepts_symbolic_spatial_conv2d_without_input_adjustment():
    inputs = (torch.randn(2, 3, 8, 8),)
    height = Dim("height", min=4, max=10)
    exported_program = _exported_program(
        Conv2d(),
        inputs,
        dynamic_shapes=({2: height},),
    )
    partition_result = TOSAPartitioner(TosaCompileSpec("TOSA-1.0+FP")).partition(
        exported_program
    )
    node = _find_node(exported_program, exir_ops.edge.aten.convolution.default)

    assert node.meta.get("delegation_tag") in partition_result.partition_tags


def test_without_shape_extension_rejects_symbolic_spatial_conv2d_needing_input_adjustment():
    inputs = (torch.randn(2, 3, 8, 8),)
    height = Dim("height", min=4, max=10)
    exported_program = _exported_program(
        Conv2d(stride=3),
        inputs,
        dynamic_shapes=({2: height},),
    )

    partition_result = TOSAPartitioner(TosaCompileSpec("TOSA-1.0+FP")).partition(
        exported_program
    )
    node = _find_node(exported_program, exir_ops.edge.aten.convolution.default)

    assert node.meta.get("delegation_tag") not in partition_result.partition_tags


def test_without_shape_extension_rejects_symbolic_spatial_conv2d_needing_dynamic_padding():
    inputs = (torch.randn(2, 3, 8, 8),)
    height = Dim("height", min=4, max=10)
    exported_program = _exported_program(
        Conv2d(stride=3, padding=2),
        inputs,
        dynamic_shapes=({2: height},),
    )

    partition_result = TOSAPartitioner(TosaCompileSpec("TOSA-1.0+FP")).partition(
        exported_program
    )
    node = _find_node(exported_program, exir_ops.edge.aten.convolution.default)

    assert node.meta.get("delegation_tag") not in partition_result.partition_tags


def test_without_shape_extension_accepts_symbolic_spatial_pooling_without_input_adjustment():
    inputs = (torch.randn(2, 3, 8, 8),)
    height = Dim("height", min=2, max=5) * 2
    exported_program = _exported_program(
        AvgPool2d(),
        inputs,
        dynamic_shapes=({2: height},),
    )

    partition_result = TOSAPartitioner(TosaCompileSpec("TOSA-1.0+FP")).partition(
        exported_program
    )
    node = _find_node(exported_program, exir_ops.edge.aten.avg_pool2d.default)

    assert node.meta.get("delegation_tag") in partition_result.partition_tags


def test_without_shape_extension_accepts_value_only_symbolic_max_pooling():
    inputs = (torch.randn(2, 3, 8, 8),)
    height = Dim("height", min=2, max=5) * 2
    exported_program = _exported_program(
        MaxPool2d(),
        inputs,
        dynamic_shapes=({2: height},),
    )

    partition_result = TOSAPartitioner(TosaCompileSpec("TOSA-1.0+FP")).partition(
        exported_program
    )
    node = _find_node(
        exported_program, exir_ops.edge.aten.max_pool2d_with_indices.default
    )

    assert node.meta.get("delegation_tag") in partition_result.partition_tags


def test_without_shape_extension_accepts_symbolic_max_pooling_with_empty_stride():
    inputs = (torch.randn(2, 3, 8, 8),)
    height = Dim("height", min=2, max=5) * 2
    exported_program = _exported_program(
        MaxPool2dEmptyStride(),
        inputs,
        dynamic_shapes=({2: height},),
    )

    partition_result = TOSAPartitioner(TosaCompileSpec("TOSA-1.0+FP")).partition(
        exported_program
    )
    node = _find_node(
        exported_program, exir_ops.edge.aten.max_pool2d_with_indices.default
    )

    assert node.meta.get("delegation_tag") in partition_result.partition_tags


def test_without_shape_extension_rejects_symbolic_spatial_pooling_needing_dynamic_padding():
    inputs = (torch.randn(2, 3, 8, 8),)
    height = Dim("height", min=4, max=10)
    exported_program = _exported_program(
        AvgPool2d(kernel_size=5, stride=3, padding=2),
        inputs,
        dynamic_shapes=({2: height},),
    )

    partition_result = TOSAPartitioner(TosaCompileSpec("TOSA-1.0+FP")).partition(
        exported_program
    )
    node = _find_node(exported_program, exir_ops.edge.aten.avg_pool2d.default)

    assert node.meta.get("delegation_tag") not in partition_result.partition_tags


def test_without_shape_extension_rejects_symbolic_mean_reduction_dim():
    inputs = (torch.randn(2, 3, 8, 8),)
    height = Dim("height", min=4, max=10)
    exported_program = _exported_program(
        MeanDim(),
        inputs,
        dynamic_shapes=({2: height},),
    )

    _assert_rejected_with_reason(
        exported_program,
        exir_ops.edge.aten.mean.dim,
        "Symbolic mean dims unsupported",
    )


def test_without_shape_extension_rejects_symbolic_full_tensor_mean():
    inputs = (torch.randn(2, 3, 8, 8),)
    height = Dim("height", min=4, max=10)
    exported_program = _exported_program(
        MeanDefault(),
        inputs,
        dynamic_shapes=({2: height},),
    )

    _assert_rejected_with_reason(
        exported_program,
        exir_ops.edge.aten.mean.default,
        "Symbolic mean dims unsupported",
    )


def test_without_shape_extension_rejects_symbolic_squeeze():
    inputs = (torch.randn(2, 3, 1, 8),)
    batch = Dim("batch", min=1, max=4)
    exported_program = _exported_program(
        Squeeze(),
        inputs,
        dynamic_shapes=({0: batch},),
    )

    _assert_rejected_with_reason(
        exported_program,
        exir_ops.edge.aten.squeeze_copy.dims,
        "Symbolic view dims unsupported",
    )


def test_without_shape_extension_rejects_symbolic_unsqueeze():
    inputs = (torch.randn(2, 3, 8),)
    batch = Dim("batch", min=1, max=4)
    exported_program = _exported_program(
        Unsqueeze(),
        inputs,
        dynamic_shapes=({0: batch},),
    )

    _assert_rejected_with_reason(
        exported_program,
        exir_ops.edge.aten.unsqueeze_copy.default,
        "Symbolic view dims unsupported",
    )


def test_without_shape_extension_rejects_symbolic_slice():
    inputs = (torch.randn(2, 3, 8, 8),)
    height = Dim("height", min=4, max=10)
    exported_program = _exported_program(
        Slice(),
        inputs,
        dynamic_shapes=({2: height},),
    )

    _assert_rejected_with_reason(
        exported_program,
        exir_ops.edge.aten.slice_copy.Tensor,
        "Symbolic slices unsupported",
    )


def test_without_shape_extension_accepts_zero_input_supported_op():
    exported_program = _exported_program(ScalarTensor(), ())
    support, _ = _support("TOSA-1.0+FP", exported_program)
    scalar_node = _find_node(exported_program, torch.ops.aten.scalar_tensor.default)

    assert support.is_node_supported(exported_program.graph_module, scalar_node) is True


def test_without_shape_extension_rejects_symbolic_shape_argument():
    inputs = (torch.randn(2, 2, 3),)
    batch = Dim("batch", min=1, max=4)
    exported_program = _exported_program(
        ReshapeWithSymSize(),
        inputs,
        dynamic_shapes=({0: batch},),
    )
    support, reporter = _support("TOSA-1.0+FP", exported_program)
    view_node = _find_node(exported_program, exir_ops.edge.aten.view_copy.default)

    assert support.is_node_supported(exported_program.graph_module, view_node) is False
    assert "Node has symbolic shape arguments" in reporter.get_table_report()


@pytest.mark.parametrize(
    "tosa_spec",
    ["TOSA-1.1+FP+INT+shape", "TOSA-1.1+FP+INT"],
)
@pytest.mark.parametrize(
    "dynamic_values, dynamic_index",
    [(False, False), (True, False), (False, True)],
)
def test_index_tensor_shape_support(tosa_spec, dynamic_values, dynamic_index):
    exported_program = _exported_program(
        Index(),
        (torch.randn(8, 4), torch.tensor([0, 2, 3], dtype=torch.int32)),
        dynamic_shapes=(
            {0: Dim("rows", min=4, max=16)} if dynamic_values else {},
            {0: Dim("selected", min=2, max=6)} if dynamic_index else {},
        ),
    )
    support, reporter = _support(tosa_spec, exported_program)
    index_node = _find_node(exported_program, exir_ops.edge.aten.index.Tensor)

    expected_support = not (dynamic_values or dynamic_index)
    assert (
        support.is_node_supported(exported_program.graph_module, index_node)
        is expected_support
    )
    if not expected_support:
        assert "Symbolic value or index shapes" in reporter.get_table_report()


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


@pytest.mark.parametrize(
    "operation",
    [operator.add, operator.sub, operator.mul, operator.mod, operator.floordiv],
)
def test_shape_extension_partitions_symint_arithmetic(operation):
    inputs = (torch.randn(2, 3),)
    batch = Dim("batch", min=1, max=4)
    exported_program = _exported_program(
        ReturnSymSizeArithmetic(operation),
        inputs,
        dynamic_shapes=({0: batch},),
    )

    partition_result = TOSAPartitioner(TosaCompileSpec("TOSA-1.1+FP+shape")).partition(
        exported_program
    )
    tagged_program = partition_result.tagged_exported_program
    sym_size_node = _find_node(tagged_program, torch.ops.aten.sym_size.int)
    arithmetic_node = _find_node(tagged_program, operation)

    assert sym_size_node.meta.get("delegation_tag") in partition_result.partition_tags
    assert arithmetic_node.meta.get("delegation_tag") in partition_result.partition_tags


def test_shape_extension_rejects_non_symint_arithmetic():
    graph = torch.fx.Graph()
    arithmetic_node = graph.call_function(operator.add, (1.0, 2.0))
    arithmetic_node.meta["val"] = 3.0
    graph.output(arithmetic_node)

    support = SymIntArithmeticSupport(
        TosaSpecification.create_from_string("TOSA-1.1+FP+shape"),
        WhyNoPartitionReporter(),
    )

    assert support.is_node_supported({}, arithmetic_node) is False


def test_shape_extension_rejects_slice_with_symbolic_bound():
    class SymbolicBoundSlice(torch.nn.Module):
        def forward(self, x):
            return torch.ops.aten.slice.Tensor(x, 1, 0, x.shape[0], 1)

    inputs = (torch.randn(4, 5),)
    exported_program = _exported_program(
        SymbolicBoundSlice(),
        inputs,
        dynamic_shapes=({0: Dim("batch", min=2, max=5)},),
    )
    support, reporter = _support("TOSA-1.1+FP+shape", exported_program)
    slice_node = _find_node(exported_program, exir_ops.edge.aten.slice_copy.Tensor)

    assert support.is_node_supported(exported_program.graph_module, slice_node) is False
    assert "Symbolic slice bounds" in reporter.get_table_report()


def test_shape_extension_rejects_slice_with_empty_unsliced_dimension():
    class SliceSecondDimension(torch.nn.Module):
        def forward(self, x):
            return torch.ops.aten.slice.Tensor(x, 1, 0, 3, 1)

    exported_program = _exported_program(SliceSecondDimension(), (torch.randn(0, 5),))
    reporter = WhyNoPartitionReporter()
    support = SliceCopySupported(
        TosaSpecification.create_from_string("TOSA-1.1+FP+shape"), reporter
    )
    slice_node = _find_node(exported_program, exir_ops.edge.aten.slice_copy.Tensor)

    assert support.is_node_supported(exported_program.graph_module, slice_node) is False


def test_shape_extension_rejects_empty_slice():
    class EmptySlice(torch.nn.Module):
        def forward(self, x):
            return torch.ops.aten.slice.Tensor(x, 1, 0, -6, 1)

    exported_program = _exported_program(EmptySlice(), (torch.randn(2, 5),))
    reporter = WhyNoPartitionReporter()
    support = SliceCopySupported(
        TosaSpecification.create_from_string("TOSA-1.1+FP+shape"), reporter
    )
    slice_node = _find_node(exported_program, exir_ops.edge.aten.slice_copy.Tensor)

    assert support.is_node_supported(exported_program.graph_module, slice_node) is False
