# Copyright 2025-2026 Arm Limited and/or its affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from typing import Callable, Dict, Tuple

import pytest
import torch
from executorch.backends.arm._passes import ConvertInt64OutputOpsToInt32Pass

from executorch.backends.arm.test import common

from executorch.backends.arm.test.tester.test_pipeline import TosaPipelineFP
from executorch.exir import EdgeCompileConfig, to_edge
from executorch.exir.dialects._ops import ops as exir_ops
from torch.fx import Graph, GraphModule

input_t1 = Tuple[torch.Tensor]  # Input x


#########################################
## Test [int32 | other types] -> int64 ##
#########################################


class CastingToInt64Model(torch.nn.Module):
    def __init__(self, target_dtype: torch.dtype) -> None:
        super().__init__()
        self.target_dtype = target_dtype

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x.to(dtype=self.target_dtype)


test_data_suite_convert: Dict[str, Callable[[], Tuple[torch.Tensor, torch.dtype]]] = {
    "fp32_input": lambda: (torch.rand((1, 2, 3, 4), dtype=torch.float32), torch.int64),
    "fp16_input": lambda: (torch.rand((1, 2, 3, 4), dtype=torch.float16), torch.int64),
}

test_data_suite_remove: Dict[str, Callable[[], Tuple[torch.Tensor, torch.dtype]]] = {
    "int32_input": lambda: (
        torch.randint(-127, 128, (1, 2, 3, 4), dtype=torch.int32),
        torch.int64,
    ),
}


TestDataFactory = Callable[[], Tuple[torch.Tensor, torch.dtype]]


@common.parametrize("test_data", test_data_suite_convert)
def test_convert_int64_output_ops_to_int32_tosa_FP_convert_casting(
    test_data: TestDataFactory,
) -> None:
    test_tensor, target_dtype = test_data()
    module = CastingToInt64Model(target_dtype)

    pipeline = TosaPipelineFP[input_t1](
        module,
        (test_tensor,),
        aten_op="torch.ops.aten.to.dtype",
        exir_op=[],
        transform_passes=[ConvertInt64OutputOpsToInt32Pass()],
    )
    pipeline.pop_stage(
        "run_method_and_compare_outputs"
    )  # As expected: RuntimeError: Int did not match Long
    pipeline.run()


@common.parametrize("test_data", test_data_suite_remove)
def test_convert_int64_output_ops_to_int32_tosa_FP_remove_casting(
    test_data: TestDataFactory,
) -> None:
    test_tensor, target_dtype = test_data()
    module = CastingToInt64Model(target_dtype)

    pipeline = TosaPipelineFP[input_t1](
        module,
        (test_tensor,),
        aten_op=[],
        exir_op=[],
        transform_passes=[ConvertInt64OutputOpsToInt32Pass()],
    )
    pipeline.change_args(
        "check_count.exir", {"torch.ops.higher_order.executorch_call_delegate": 0}
    )  # Empty graph without nodes
    pipeline.pop_stage("run_method_and_compare_outputs")
    pipeline.run()


##########################################################
## Test argmax/argmin int64 output -> int32 cast       ##
##########################################################


@pytest.mark.parametrize(
    "arg_op, aten_op_str",
    [
        (torch.argmax, "torch.ops.aten.argmax.default"),
        (torch.argmin, "torch.ops.aten.argmin.default"),
    ],
    ids=["argmax", "argmin"],
)
def test_convert_int64_output_ops_to_int32_tosa_FP_insert_cast(arg_op, aten_op_str):
    class ArgOpModel(torch.nn.Module):
        def forward(self, x: torch.Tensor) -> torch.Tensor:
            return (10 * arg_op(x, dim=-1) + 10) + 1.5

    pipeline = TosaPipelineFP[input_t1](
        ArgOpModel(),
        (torch.randint(0, 10, (2, 4, 6, 8)),),
        aten_op=[aten_op_str, "torch.ops.aten.mul.Tensor", "torch.ops.aten.add.Tensor"],
        exir_op=[
            "executorch_exir_dialects_edge__ops_aten_mul_Tensor",
            "executorch_exir_dialects_edge__ops_aten_add_Tensor",
        ],
        transform_passes=[ConvertInt64OutputOpsToInt32Pass()],
    )
    pipeline.run()


@pytest.mark.parametrize(
    "arg_op",
    [torch.argmax, torch.argmin],
    ids=["argmax", "argmin"],
)
def test_arg_op_safe_edge_scalar_constant_is_cast_to_int32(arg_op):
    class SafeScalarArithmetic(torch.nn.Module):
        def forward(self, x: torch.Tensor):
            return arg_op(x, dim=1) * 10

    module = SafeScalarArithmetic()
    test_input = torch.randn(2, 8)
    exported_program = to_edge(
        torch.export.export(module, (test_input,)),
        compile_config=EdgeCompileConfig(_check_ir_validity=False),
    ).exported_program()

    result = ConvertInt64OutputOpsToInt32Pass().call(exported_program.graph_module)

    mul = next(
        node
        for node in result.graph_module.graph.nodes
        if node.target == exir_ops.edge.aten.mul.Tensor
    )
    assert mul.args[0].meta["val"].dtype == torch.int32
    assert mul.args[1].meta["val"].dtype == torch.int32

    actual = result.graph_module(torch.tensor(10), test_input)[0]
    expected = module(test_input)
    assert actual.dtype == torch.int64
    torch.testing.assert_close(actual, expected)


@pytest.mark.parametrize(
    "arg_op",
    [torch.argmax, torch.argmin],
    ids=["argmax", "argmin"],
)
@pytest.mark.parametrize("use_edge_ops", [False, True], ids=["aten", "edge"])
def test_arg_op_unsafe_arithmetic_stays_int64(arg_op, use_edge_ops: bool):
    class UnsafeArithmetic(torch.nn.Module):
        def forward(self, x: torch.Tensor):
            indices = arg_op(x, dim=1).unsqueeze(-1)
            return indices * indices, indices

    module = UnsafeArithmetic()
    test_input = torch.zeros(1, 50001)
    test_input[0, -1] = 1 if arg_op is torch.argmax else -1
    exported_program = torch.export.export(module, (test_input,))
    if use_edge_ops:
        exported_program = to_edge(
            exported_program,
            compile_config=EdgeCompileConfig(_check_ir_validity=False),
        ).exported_program()

    result = ConvertInt64OutputOpsToInt32Pass().call(exported_program.graph_module)

    mul_targets = {
        torch.ops.aten.mul.Tensor,
        exir_ops.edge.aten.mul.Tensor,
    }
    relay_targets = {
        torch.ops.aten.unsqueeze.default,
        exir_ops.edge.aten.unsqueeze_copy.default,
    }
    mul = next(
        node for node in result.graph_module.graph.nodes if node.target in mul_targets
    )
    relay = next(
        node for node in result.graph_module.graph.nodes if node.target in relay_targets
    )
    assert mul.args[0].meta["val"].dtype == torch.int64
    assert mul.args[1].meta["val"].dtype == torch.int64
    assert relay.args[0].meta["val"].dtype == torch.int32

    expected = module(test_input)
    actual = result.graph_module(test_input)
    assert actual[0].item() == 2_500_000_000
    assert actual[1].dtype == torch.int64
    for actual_output, expected_output in zip(actual, expected, strict=True):
        torch.testing.assert_close(actual_output, expected_output)


@pytest.mark.parametrize(
    "arg_op",
    [torch.argmax, torch.argmin],
    ids=["argmax", "argmin"],
)
def test_arg_op_direct_output_is_unchanged(arg_op):
    class DirectOutput(torch.nn.Module):
        def forward(self, x: torch.Tensor):
            return arg_op(x, dim=1)

    exported_program = torch.export.export(DirectOutput(), (torch.randn(2, 8),))
    result = ConvertInt64OutputOpsToInt32Pass().call(exported_program.graph_module)

    assert not result.modified
    assert result.graph_module(torch.randn(2, 8))[0].dtype == torch.int64


##############################################################
## Test on_overflow range check for argmax/argmin           ##
##############################################################

_OVERFLOW_DIM = torch.iinfo(torch.int32).max + 1


def _make_argmax_graph_large_dim() -> GraphModule:
    """Construct a minimal graph with an argmax over a dimension > INT32_MAX.

    Uses FakeTensorMode so no memory is allocated for the large dimension.

    """
    from torch._subclasses import FakeTensorMode

    graph = Graph()
    with FakeTensorMode():
        fake_input = torch.empty(_OVERFLOW_DIM, dtype=torch.float32)
        fake_output = torch.empty((), dtype=torch.int64)
    x = graph.placeholder("x")
    x.meta["val"] = fake_input
    out = graph.call_function(torch.ops.aten.argmax.default, (x, 0))
    out.meta["val"] = fake_output
    graph.output(out)
    return GraphModule(torch.nn.Module(), graph)


def test_on_overflow_raise():
    gm = _make_argmax_graph_large_dim()
    with pytest.raises(RuntimeError, match="cannot be safely cast to int32"):
        ConvertInt64OutputOpsToInt32Pass(on_overflow="raise").call(gm)


def test_on_overflow_warn(caplog):
    import logging

    gm = _make_argmax_graph_large_dim()
    with caplog.at_level(logging.WARNING):
        result = ConvertInt64OutputOpsToInt32Pass(on_overflow="warn").call(gm)
    assert not result.modified
    assert "cannot be safely cast to int32" in caplog.text


def test_on_overflow_skip():
    gm = _make_argmax_graph_large_dim()
    result = ConvertInt64OutputOpsToInt32Pass(on_overflow="skip").call(gm)
    assert not result.modified


def test_on_overflow_invalid():
    with pytest.raises(ValueError, match="on_overflow must be"):
        ConvertInt64OutputOpsToInt32Pass(on_overflow="blah")
