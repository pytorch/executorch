# Copyright 2025 Arm Limited and/or its affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from dataclasses import dataclass
from types import SimpleNamespace

from executorch.backends.arm.common.arm_compile_spec import ArmCompileSpec
from executorch.backends.arm.debug.schema import DebugHook, DebugSchema
from executorch.backends.arm.test import common


@dataclass
class DebugHookTestCase:
    mock_node: SimpleNamespace
    tosa_op: str
    op_id: int
    expected_events: int
    num_nodes_traced: int


def create_mock_node_1():
    def _get_action_str() -> str:
        return "create"

    from_node_2 = SimpleNamespace(
        name="convolution",
        target="aten.convolution.default",
        graph_id=6052414368,
        pass_name="ExportedProgram.module()",
        action="create",
        from_node=[],
        _get_action_string=_get_action_str,
    )

    from_node_1 = SimpleNamespace(
        name="convolution",
        target="aten.convolution.default",
        graph_id=5705954832,
        pass_name="Interpreter_PropagateUnbackedSymInts",
        action="create",
        from_node=[from_node_2],
        _get_action_string=_get_action_str,
    )

    fx_node_mock = SimpleNamespace(
        name="aten_convolution_default",
        target="aten.convolution.default",
        meta={
            "stack_trace": 'File "models/model.py", line 221, in forward\nreturn self.features(x)',
            "nn_module_stack": {"__self__": ["", "model.Model"]},
            "torch_fn": ("conv2d", "builtin_function_or_method.conv2d"),
            "from_node": [from_node_1],
        },
    )

    return fx_node_mock


def create_mock_node_2():
    def _get_action_str() -> str:
        return "create"

    from_node_1 = SimpleNamespace(
        name="convolution",
        target="aten.convolution.default",
        graph_id=5705954832,
        pass_name="Interpreter_PropagateUnbackedSymInts",
        action="create",
        from_node=[],
        _get_action_string=_get_action_str,
    )

    fx_node_mock = SimpleNamespace(
        name="aten_convolution_default",
        target="aten.convolution.default",
        meta={
            "from_node": [from_node_1],
        },
    )

    return fx_node_mock


def create_mock_node_3():
    fx_node_mock = SimpleNamespace(
        name="aten_convolution_default",
        target="aten.convolution.default",
        meta={
            "from_node": [],
        },
    )

    return fx_node_mock


def _compare_tosa_and_schema(debug_event: DebugSchema, tosa_op):
    tosa_info = debug_event.tosa_info

    assert tosa_info.node_name == tosa_op

    # The mapping between op_ids to operator names could change
    # So just check operator_name is a string
    assert isinstance(tosa_info.operator_name, str)


def _compare_node_and_schema(debug_event: DebugSchema, mocked_node):
    # Check aten info
    aten_info = debug_event.aten_info

    assert aten_info.node_name == mocked_node.name
    assert aten_info.operator_name == mocked_node.target

    # Check torch info
    torch_info = debug_event.torch_info

    if "nn_module_stack" in mocked_node.meta:
        assert torch_info.nn_module_stack == mocked_node.meta["nn_module_stack"]
    else:
        assert torch_info.nn_module_stack == "No module stack trace available"

    if "stack_trace" in mocked_node.meta:
        assert torch_info.stack_trace == mocked_node.meta["stack_trace"].split("\n")
    else:
        assert torch_info.stack_trace == ["No stack trace available"]

    if "torch_fn" in mocked_node.meta:
        assert torch_info.torch_fn == mocked_node.meta["torch_fn"]
    else:
        assert torch_info.torch_fn == "No torch_fn available"


TESTCASES = {
    "mocked_node": DebugHookTestCase(
        mock_node=create_mock_node_1(),
        tosa_op="layer-1",
        op_id=3,
        expected_events=1,
        num_nodes_traced=2,
    ),
    "mocked_node_partially_empty": DebugHookTestCase(
        mock_node=create_mock_node_2(),
        tosa_op="layer-1",
        op_id=1,
        expected_events=1,
        num_nodes_traced=1,
    ),
    "mocked_node_all_empty": DebugHookTestCase(
        mock_node=create_mock_node_3(),
        tosa_op="layer-2",
        op_id=1,
        expected_events=1,
        num_nodes_traced=0,
    ),
}


@common.parametrize("test_data", TESTCASES)
def test_debug_hook_add_json(test_data: DebugHookTestCase):
    hook = DebugHook(ArmCompileSpec.DebugMode.JSON)
    hook.add(test_data.mock_node, test_data.tosa_op, test_data.op_id)

    debug_events = hook._debug_events
    assert len(debug_events) == test_data.expected_events
    assert len(debug_events[0].torch_info.node_trace) == test_data.num_nodes_traced

    _compare_tosa_and_schema(debug_events[0], test_data.tosa_op)
    _compare_node_and_schema(debug_events[0], test_data.mock_node)


@common.parametrize("test_data", TESTCASES)
def test_debug_hook_add_tosa(test_data: DebugHookTestCase):
    hook = DebugHook(ArmCompileSpec.DebugMode.TOSA)
    hook.add(test_data.mock_node, test_data.tosa_op, test_data.op_id)

    debug_events = hook._debug_events
    assert len(debug_events) == test_data.expected_events
    assert len(debug_events[0].torch_info.node_trace) == test_data.num_nodes_traced

    assert debug_events[0].tosa_info is None

    _compare_node_and_schema(debug_events[0], test_data.mock_node)
