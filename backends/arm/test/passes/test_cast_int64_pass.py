# Copyright 2025-2026 Arm Limited and/or its affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from typing import Tuple

import pytest
import torch
from executorch.backends.arm._passes import CastInt64BuffersToInt32Pass

from executorch.backends.arm.test import common
from executorch.backends.arm.test.tester.test_pipeline import PassPipeline

from executorch.backends.test.harness.stages import StageType

input_t = Tuple[torch.Tensor]  # Input x


class Int64Model(torch.nn.Module):
    test_data = {
        "rand": (torch.rand(4),),
    }

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + 3


class NonPersistentInt64BufferModel(torch.nn.Module):
    test_data = {
        "rand": (torch.rand(4),),
    }

    def __init__(self):
        super().__init__()
        self.register_buffer(
            "position_ids", torch.arange(4, dtype=torch.int64), persistent=False
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.get_buffer("position_ids").to(torch.float32)


@common.parametrize("test_data", Int64Model.test_data)
def test_cast_int64_buffers_to_int32_tosa_FP(test_data: input_t):
    module = Int64Model()
    op_checks = {
        "executorch_exir_dialects_edge__ops_dim_order_ops__to_dim_order_copy_default": 1,
        "executorch_exir_dialects_edge__ops_aten_add_Tensor": 1,
    }
    pipeline = PassPipeline[input_t](
        module,
        test_data,
        quantize=False,
        ops_before_pass=op_checks,
        ops_after_pass=op_checks,
        passes_with_exported_program=[CastInt64BuffersToInt32Pass],
    )
    pipeline.run()

    exported_program = pipeline.tester.get_artifact(
        StageType.RUN_PASSES
    ).exported_program()
    for state in exported_program.state_dict:
        assert exported_program.state_dict[state].dtype == torch.int32


@common.parametrize("test_data", NonPersistentInt64BufferModel.test_data)
def test_cast_non_persistent_int64_buffers_to_int32_tosa_FP(test_data: input_t):
    module = NonPersistentInt64BufferModel()
    pipeline = PassPipeline[input_t](
        module,
        test_data,
        quantize=False,
        passes_with_exported_program=[CastInt64BuffersToInt32Pass],
    )
    pipeline.run()

    exported_program = pipeline.tester.get_artifact(
        StageType.RUN_PASSES
    ).exported_program()
    assert exported_program.constants["position_ids"].dtype == torch.int32


class LiftedInt64IndexModel(torch.nn.Module):
    """A plain tensor attribute is lifted as CONSTANT_TENSOR, not BUFFER."""

    def __init__(self):
        super().__init__()
        self.indices = torch.tensor([0, 2], dtype=torch.int64)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x[:, self.indices]


def test_cast_lifted_int64_index_constant_to_int32():
    """Lifted indices must reach gather as int32."""
    inputs = (torch.arange(32, dtype=torch.float32).reshape(1, 4, 8),)
    pipeline = PassPipeline[input_t](
        LiftedInt64IndexModel(),
        inputs,
        quantize=False,
        passes_with_exported_program=[CastInt64BuffersToInt32Pass],
    )
    # Also compares the output of the transformed Edge graph to the original.
    pipeline.run()

    ep = pipeline.tester.get_artifact(StageType.RUN_PASSES).exported_program()
    lifted = ep.graph_signature.inputs_to_lifted_tensor_constants
    assert len(lifted) == 1
    placeholder_name, constant_name = next(iter(lifted.items()))
    assert ep.constants[constant_name].dtype == torch.int32
    placeholder = next(
        node for node in ep.graph_module.graph.nodes if node.name == placeholder_name
    )
    assert placeholder.meta["val"].dtype == torch.int32


@pytest.mark.parametrize(
    "bad_index,expected_error",
    [
        (2**31, "value >"),
        (-(2**31) - 1, "value <"),
    ],
)
def test_lifted_int64_constant_out_of_int32_range_is_not_truncated(
    bad_index: int, expected_error: str
):
    from executorch.exir import to_edge

    inputs = (torch.randn(1, 4, 8),)
    ep = to_edge(
        torch.export.export(LiftedInt64IndexModel(), inputs)
    ).exported_program()
    lifted = ep.graph_signature.inputs_to_lifted_tensor_constants
    assert len(lifted) == 1
    _, constant_name = next(iter(lifted.items()))
    # Modify after export: an out-of-bounds index need not be run during tracing.
    ep.constants[constant_name] = torch.tensor([0, bad_index], dtype=torch.int64)

    with pytest.raises(RuntimeError, match=expected_error):
        CastInt64BuffersToInt32Pass(ep)(ep.graph_module)
    assert ep.constants[constant_name].dtype == torch.int64


def test_lifted_int64_constant_used_as_data_and_index_is_not_narrowed():
    """The same lifted constant is indexed data and index."""

    class SelfIndexModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.values = torch.tensor([1, 0], dtype=torch.int64)

        def forward(self) -> torch.Tensor:
            return self.values[self.values]

    model = SelfIndexModel().eval()
    expected = model()
    ep = torch.export.export(model, ())
    lifted = ep.graph_signature.inputs_to_lifted_tensor_constants
    assert len(lifted) == 1
    placeholder_name, constant_name = next(iter(lifted.items()))
    node = next(n for n in ep.graph_module.graph.nodes if n.name == placeholder_name)
    users = [user for user in node.users if user.target == torch.ops.aten.index.Tensor]
    assert len(users) == 1
    assert users[0].args[0] is node
    assert any(index is node for index in users[0].args[1])

    cast_pass = CastInt64BuffersToInt32Pass(ep)
    assert not cast_pass._is_index_tensor_input(node)
    assert not cast_pass.should_run_pass(ep.graph_module)
    assert cast_pass._to_int32(ep.graph_module) == 0
    assert ep.constants[constant_name].dtype == torch.int64
    assert node.meta["val"].dtype == torch.int64
    result = ep.module()()
    assert result.dtype == torch.int64
    torch.testing.assert_close(result, expected)


def test_lifted_int64_index_rejects_other_operand_occurrences():
    """The predicate must reject nested positional/keyword operand reuse."""
    for location in ("indexed_data", "positional", "keyword"):
        graph = torch.fx.Graph()
        data = graph.placeholder("data")
        index = graph.placeholder("index")
        args = (data, [index])
        kwargs = {}
        if location == "indexed_data":
            args = (index, [index])
        elif location == "positional":
            args += ({"nested": [index]},)
        else:
            kwargs["extra"] = {"nested": (index,)}
        graph.call_function(torch.ops.aten.index.Tensor, args=args, kwargs=kwargs)
        assert not CastInt64BuffersToInt32Pass._is_index_tensor_input(index)

    graph = torch.fx.Graph()
    data = graph.placeholder("data")
    index = graph.placeholder("index")
    graph.call_function(torch.ops.aten.index.Tensor, args=(data, [index]))
    assert CastInt64BuffersToInt32Pass._is_index_tensor_input(index)


def test_cast_empty_lifted_int64_index_constant_to_int32():
    """Empty lifted indices are valid and must convert without reductions."""
    from executorch.exir import to_edge

    model = LiftedInt64IndexModel()
    model.indices = torch.empty(0, dtype=torch.int64)
    inputs = (torch.arange(32, dtype=torch.float32).reshape(1, 4, 8),)
    ep = to_edge(torch.export.export(model, inputs)).exported_program()

    lifted = ep.graph_signature.inputs_to_lifted_tensor_constants
    assert len(lifted) == 1
    placeholder_name, constant_name = next(iter(lifted.items()))
    assert ep.constants[constant_name].numel() == 0
    assert ep.constants[constant_name].dtype == torch.int64

    cast_pass = CastInt64BuffersToInt32Pass(ep)
    assert cast_pass.should_run_pass(ep.graph_module)
    cast_pass(ep.graph_module)

    assert ep.constants[constant_name].numel() == 0
    assert ep.constants[constant_name].dtype == torch.int32
    placeholder = next(
        node for node in ep.graph_module.graph.nodes if node.name == placeholder_name
    )
    assert placeholder.meta["val"].dtype == torch.int32


class Int64BufferAddModel(torch.nn.Module):
    offset: torch.Tensor
    test_data = {
        "wide": (torch.tensor([2**40, -(2**38), 2**31, -1]),),
    }

    def __init__(self):
        super().__init__()
        self.register_buffer("offset", torch.tensor([2**39, 1, -(2**40), 7]))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.offset


@common.parametrize("test_data", Int64BufferAddModel.test_data)
def test_cast_int64_buffers_keeps_int64_users_tosa_FP_INT_int64(test_data: input_t):
    pipeline = PassPipeline[input_t](
        Int64BufferAddModel(),
        test_data,
        quantize=False,
        passes_with_exported_program=[CastInt64BuffersToInt32Pass],
        tosa_version="1.1",
        tosa_extensions=["INT", "int64"],
    )
    pipeline.run()

    exported_program = pipeline.tester.get_artifact(
        StageType.RUN_PASSES
    ).exported_program()
    assert exported_program.state_dict["offset"].dtype == torch.int64


@common.parametrize("test_data", NonPersistentInt64BufferModel.test_data)
def test_cast_int64_buffers_narrows_other_users_tosa_FP_INT_int64(test_data: input_t):
    pipeline = PassPipeline[input_t](
        NonPersistentInt64BufferModel(),
        test_data,
        quantize=False,
        passes_with_exported_program=[CastInt64BuffersToInt32Pass],
        tosa_version="1.1",
        tosa_extensions=["INT", "int64"],
    )
    pipeline.run()

    exported_program = pipeline.tester.get_artifact(
        StageType.RUN_PASSES
    ).exported_program()
    assert exported_program.constants["position_ids"].dtype == torch.int32


class Int64BufferFloatComparisonModel(torch.nn.Module):
    limit: torch.Tensor
    test_data = {
        "rand": (torch.rand(4),),
    }

    def __init__(self):
        super().__init__()
        self.register_buffer("limit", torch.tensor([-2, -1, 0, 1]))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + (self.limit > -1.5)


@common.parametrize("test_data", Int64BufferFloatComparisonModel.test_data)
def test_cast_int64_buffers_narrows_float_comparison_tosa_FP_INT_int64(
    test_data: input_t,
):
    pipeline = PassPipeline[input_t](
        Int64BufferFloatComparisonModel(),
        test_data,
        quantize=False,
        passes_with_exported_program=[CastInt64BuffersToInt32Pass],
        tosa_version="1.1",
        tosa_extensions=["INT", "int64"],
    )
    pipeline.run()

    exported_program = pipeline.tester.get_artifact(
        StageType.RUN_PASSES
    ).exported_program()
    assert exported_program.state_dict["limit"].dtype == torch.int32
