# Copyright 2025-2026 NXP
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import numpy as np
import pytest
import torch

from executorch.backends.nxp.backend.custom_delegation_options import (
    CustomDelegationOptions,
)
from executorch.backends.nxp.backend.ir.converter.node_converters.ops_converters import (
    ViewCopyConverter,
)
from executorch.backends.nxp.tests.executorch_pipeline import to_quantized_edge_program
from executorch.backends.nxp.tests.executors import graph_contains_any_of_ops
from executorch.exir.dialects._ops import ops as exir_ops

# noinspection PyProtectedMember
ExecutorchDelegateCall = torch.ops.higher_order.executorch_call_delegate


class SingleViewCopyModule(torch.nn.Module):
    def __init__(self, new_shape: list[int]):
        super().__init__()
        self.new_shape = new_shape

    def forward(self, x):
        return torch.reshape(x, self.new_shape)


class ConcatAddNoOpModel(torch.nn.Module):
    def __init__(self, shape: tuple[int, ...]):
        super().__init__()
        self.shape = shape

    def forward(self, x):
        # Concatenate 1 tensor and add 0. Both operations are no-ops.
        x = torch.concat((x,), dim=0)
        zeros = torch.zeros(self.shape)
        return x + zeros


class AddMulSubNoOpModel(torch.nn.Module):
    def __init__(self, shape: tuple[int, ...]):
        super().__init__()
        self.shape = shape

    def forward(self, x):
        zero1 = torch.zeros(self.shape)
        zero2 = torch.zeros(self.shape)
        one = torch.ones(self.shape)

        x = zero1 + x
        x = one * x
        x = x - zero2

        return x


@pytest.fixture(autouse=True)
def reseed_model_per_test_run():
    torch.manual_seed(23)
    np.random.seed(23)


def test_single_view_copy_partition():
    input_shape = (2, 10)
    module = SingleViewCopyModule([1, 20])

    ep = to_quantized_edge_program(module, input_shape).exported_program()

    # Make sure the `view_copy` was not delegated.
    assert graph_contains_any_of_ops(ep.graph, [exir_ops.edge.aten.view_copy.default])
    assert not graph_contains_any_of_ops(ep.graph, [ExecutorchDelegateCall])


def test_single_view_copy_partition__forced_delegation():
    input_shape = (2, 10)
    module = SingleViewCopyModule([1, 20])

    def _supported_partitioning(*_):
        return True

    # Replace the partition support check function, to accept anything.
    original_supports_partitioning_result = (
        ViewCopyConverter.supports_partitioning_result
    )
    ViewCopyConverter.supports_partitioning_result = _supported_partitioning

    # Force the partitioner to delegate the node.
    cdo = CustomDelegationOptions(allow_no_op_partitions=True)

    with pytest.raises(
        RuntimeError,
        match="Model converted with neutron-converter does not contain a NeutronGraph node.",
    ):
        to_quantized_edge_program(
            module, input_shape, custom_delegation_options=cdo
        ).exported_program()

    # Return to the original partition support check function.
    ViewCopyConverter.supports_partitioning_result = (
        original_supports_partitioning_result
    )


def test_noop_partitions__concatenate_one_tensor_and_add_zeros():
    input_shape = (1, 2, 3, 4)
    module = ConcatAddNoOpModel(input_shape)

    ep = to_quantized_edge_program(
        module,
        input_shape,
    ).exported_program()

    # Make sure neither the `cat` nor the `add` was delegated
    assert not any(n.target == ExecutorchDelegateCall for n in ep.graph.nodes)
    assert graph_contains_any_of_ops(
        ep.graph,
        [
            exir_ops.edge.aten.cat.default,
            exir_ops.edge.aten.add.Tensor,
        ],
    )


def test_noop_partitions__concatenate_one_tensor_and_add_zeros__forced_delegation():
    input_shape = (1, 2, 3, 4)
    module = ConcatAddNoOpModel(input_shape)

    # Force the partitioner to delegate the node.
    cdo = CustomDelegationOptions(allow_no_op_partitions=True)

    with pytest.raises(
        RuntimeError,
        match="Model converted with neutron-converter does not contain a NeutronGraph node.",
    ):
        to_quantized_edge_program(
            module, input_shape, custom_delegation_options=cdo
        ).exported_program()


def test_noop_partitions__add_mul_sub_div():
    input_shape = (6, 7)
    module = AddMulSubNoOpModel(input_shape)

    ep = to_quantized_edge_program(
        module,
        input_shape,
    ).exported_program()

    # Make sure nothing was delegated.
    assert not any(n.target == ExecutorchDelegateCall for n in ep.graph.nodes)
    assert graph_contains_any_of_ops(
        ep.graph,
        [
            exir_ops.edge.aten.add.Tensor,
            exir_ops.edge.aten.mul.Tensor,
            exir_ops.edge.aten.sub.Tensor,
        ],
    )


def test_noop_partitions__add_mul_sub_div__forced_delegation():
    input_shape = (6, 7)
    module = AddMulSubNoOpModel(input_shape)

    # Force the partitioner to delegate the node.
    cdo = CustomDelegationOptions(allow_no_op_partitions=True)

    with pytest.raises(
        RuntimeError,
        match="Model converted with neutron-converter does not contain a NeutronGraph node.",
    ):
        to_quantized_edge_program(
            module, input_shape, custom_delegation_options=cdo
        ).exported_program()
