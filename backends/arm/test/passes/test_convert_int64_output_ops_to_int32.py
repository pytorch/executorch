# Copyright 2025 Arm Limited and/or its affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from typing import Callable, Dict, Tuple

import pytest
import torch
from executorch.backends.arm._passes import ConvertInt64OutputOpsToInt32Pass

from executorch.backends.arm.test import common

from executorch.backends.arm.test.tester.test_pipeline import TosaPipelineFP
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
