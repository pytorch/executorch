# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import pytest
import torch
from executorch.backends.espressif import to_espnn_edge
from executorch.backends.espressif.passes import EspPassManager
from executorch.backends.espressif.test.tester import EspRunPasses, EspTester
from executorch.exir import to_edge, to_edge_transform_and_lower
from executorch.exir._serialize._program import deserialize_pte_binary


class Multiply(torch.nn.Module):
    def forward(self, x, y):
        return x * y


class MultiplyAndFloatUsers(torch.nn.Module):
    def forward(self, x, y):
        product = x * y
        return product, torch.sin(x), torch.sin(product)


def _output_scale(program):
    output = next(node for node in program.graph.nodes if node.op == "output")
    return output.args[0][0].args[1]


@pytest.mark.parametrize("entry_point", ["separate", "combined"])
def test_entry_points_preserve_program_behavior(entry_point):
    inputs = (
        torch.linspace(-1, 1, 6).reshape(2, 3),
        torch.linspace(-2, 2, 6).reshape(2, 3),
    )
    tester = EspTester(MultiplyAndFloatUsers(), inputs).quantize().export()
    if entry_point == "combined":
        tester.to_edge_transform_and_lower()
    else:
        tester.to_edge().run_passes()
    tester.run_method_and_compare_outputs(inputs=inputs, atol=0, rtol=0, qtol=1)
    edge = tester.get_artifact()
    assert edge.compile_config._check_ir_validity
    edge.exported_program().validate()
    tester.to_executorch().serialize()
    program = deserialize_pte_binary(tester.get_artifact()).program
    assert program.execution_plan[0].name == "forward"
    assert not program.execution_plan[0].delegates


def test_public_entry_point_keeps_delegate_default():
    inputs = (torch.linspace(-1, 1, 6), torch.linspace(-2, 2, 6))
    program = EspTester(Multiply(), inputs).quantize().export().get_artifact()

    delegated, delegate_report = to_espnn_edge(program, False)
    assert delegate_report["partitions"] == 1
    assert delegate_report["delegated_ops"] == 1
    assert not delegate_report["cpu_ops"]
    assert delegated.to_executorch().executorch_program.execution_plan[0].delegates

    edge, report = to_espnn_edge(program, False, use_operator_library=True)
    assert report == {"esp_nn_ops": 1, "cpu_ops": []}
    assert edge.compile_config._check_ir_validity
    torch.testing.assert_close(
        edge.exported_program().module()(*inputs),
        program.module()(*inputs),
        atol=_output_scale(program),
        rtol=0,
    )
    serialized = deserialize_pte_binary(edge.to_executorch().buffer).program
    assert not serialized.execution_plan[0].delegates


def test_operator_library_strict_mode_rejects_remaining_compute():
    inputs = (torch.linspace(-1, 1, 6), torch.linspace(-2, 2, 6))
    program = (
        EspTester(MultiplyAndFloatUsers(), inputs).quantize().export().get_artifact()
    )
    with pytest.raises(RuntimeError, match=r"forward\.\w+ \(aten::sin\)"):
        to_espnn_edge(program, False, use_operator_library=True)


def test_operator_library_strict_mode_accepts_no_compute():
    program = torch.export.export(torch.nn.Identity(), (torch.ones(3),))
    _, report = to_espnn_edge(program, False, use_operator_library=True)
    assert report == {"esp_nn_ops": 0, "cpu_ops": []}


def test_operator_library_uses_standard_edge_decompositions():
    inputs = (torch.linspace(-1, 1, 6).reshape(2, 3),)
    program = torch.export.export(torch.nn.Linear(3, 2), inputs)
    edge, report = to_espnn_edge(program, use_operator_library=True)
    targets = {op["target"] for op in report["cpu_ops"]}
    assert "aten::linear" not in targets
    assert "aten::addmm" in targets
    torch.testing.assert_close(
        edge.exported_program().module()(*inputs), program.module()(*inputs)
    )
    serialized = deserialize_pte_binary(edge.to_executorch().buffer).program
    assert not serialized.execution_plan[0].delegates


@pytest.mark.parametrize("entry_point", ["separate", "combined"])
def test_passes_preserve_methods_and_constants(entry_point):
    class Scale(torch.nn.Module):
        def __init__(self, scale, offset):
            super().__init__()
            self.register_buffer("scale", scale)
            self.offset = offset

        def forward(self, x):
            return x * self.scale + self.offset

    inputs = (torch.tensor([-1.0, 2.0, 3.0]),)
    models = {
        "forward": Scale(torch.tensor([2.0, 3.0, 4.0]), torch.tensor([0.5, 1.0, 1.5])),
        "other": Scale(torch.tensor([5.0, 6.0, 7.0]), torch.tensor([1.5, 2.0, 2.5])),
    }
    programs = {
        name: torch.export.export(model, inputs) for name, model in models.items()
    }
    if entry_point == "combined":
        transformed = to_edge_transform_and_lower(
            programs, transform_passes=EspPassManager(), constant_methods={"version": 1}
        )
    else:
        stage = EspRunPasses()
        stage.run(to_edge(programs, constant_methods={"version": 1}))
        transformed = stage.artifact
    assert transformed.methods == {"forward", "other"}
    assert transformed.config_methods == {"version"}
    for name, model in models.items():
        program = transformed.exported_program(name)
        program.validate()
        torch.testing.assert_close(
            program.module()(*inputs), model(*inputs), atol=0, rtol=0
        )
    serialized = deserialize_pte_binary(transformed.to_executorch().buffer).program
    assert {plan.name for plan in serialized.execution_plan} == {
        "forward",
        "other",
        "version",
    }
    assert all(not plan.delegates for plan in serialized.execution_plan)


def test_operator_library_reports_all_methods():
    inputs = (torch.linspace(-1, 1, 6), torch.linspace(-2, 2, 6))
    programs = {
        name: EspTester(Multiply(), inputs).quantize().export().get_artifact()
        for name in ("forward", "other")
    }
    edge, report = to_espnn_edge(programs, use_operator_library=True)
    assert report == {"esp_nn_ops": 2, "cpu_ops": []}
    for name, program in programs.items():
        torch.testing.assert_close(
            edge.exported_program(name).module()(*inputs),
            program.module()(*inputs),
            atol=_output_scale(program),
            rtol=0,
        )
    serialized = deserialize_pte_binary(edge.to_executorch().buffer).program
    assert {plan.name for plan in serialized.execution_plan} == {"forward", "other"}
    assert all(not plan.delegates for plan in serialized.execution_plan)
