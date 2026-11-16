# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import pytest
import torch
from executorch.backends.espressif import (
    EspNNQuantizer,
    get_esp_fused_quant_recipe,
    get_operator_library_report,
    to_espnn_edge,
)
from executorch.backends.espressif.passes import EspPassManager
from executorch.backends.espressif.test.tester import EspRunPasses, EspTester
from executorch.backends.test.harness.stages import Quantize
from executorch.backends.transforms.fused_quant.graph_utils import (
    get_qparams_from_node,
    get_scale,
)
from executorch.exir import to_edge, to_edge_transform_and_lower
from executorch.exir._serialize._program import deserialize_pte_binary
from executorch.export import export


class Multiply(torch.nn.Module):
    def forward(self, x, y):
        return x * y


class MultiplyAndFloatUsers(torch.nn.Module):
    def forward(self, x, y):
        product = x * y
        return product, torch.sin(x), torch.sin(product)


class MultiplyReshapeAndSin(torch.nn.Module):
    def forward(self, x, y):
        return torch.sin((x * y).reshape(-1))


def _output_scale(program):
    output = next(node for node in program.graph.nodes if node.op == "output")
    return output.args[0][0].args[1]


@pytest.mark.parametrize("entry_point", ["separate", "combined"])
def test_entry_points_preserve_program_behavior(entry_point):
    inputs = (
        torch.linspace(-1, 1, 6).reshape(2, 3),
        torch.linspace(-2, 2, 6).reshape(2, 3),
    )
    tester = EspTester(MultiplyAndFloatUsers(), inputs).export_fused_quant()
    reference = tester.get_artifact()
    if entry_point == "combined":
        tester.to_edge_transform_and_lower()
    else:
        tester.to_edge().run_passes()
    torch.testing.assert_close(
        tester.get_artifact().exported_program().module()(*inputs),
        reference.module()(*inputs),
        atol=_output_scale(reference),
        rtol=0,
    )
    edge = tester.get_artifact()
    assert edge.compile_config._check_ir_validity
    assert edge.compile_config._skip_dim_order
    edge.exported_program().validate()
    tester.to_executorch().serialize()
    program = deserialize_pte_binary(tester.get_artifact()).program
    assert program.execution_plan[0].name == "forward"
    assert not program.execution_plan[0].delegates


def test_public_entry_points_keep_delegate_and_operator_library_separate():
    inputs = (torch.linspace(-1, 1, 6), torch.linspace(-2, 2, 6))
    program = (
        EspTester(Multiply(), inputs)
        .quantize(Quantize(EspNNQuantizer()))
        .export()
        .get_artifact()
    )
    delegated, delegate_report = to_espnn_edge(program, False)
    assert delegate_report["partitions"] == 1
    assert delegate_report["delegated_ops"] == 1
    assert not delegate_report["cpu_ops"]
    assert delegated.to_executorch().executorch_program.execution_plan[0].delegates

    session = export(
        Multiply(),
        example_inputs=[inputs],
        export_recipe=get_esp_fused_quant_recipe(allow_cpu_fallback=False),
    )
    edge = session.get_edge_program_manager()
    assert get_operator_library_report(edge) == {
        "esp_nn_ops": 1,
        "cpu_ops": [],
        "layout_ops": [],
    }
    assert edge.compile_config._check_ir_validity
    serialized = deserialize_pte_binary(
        session.get_executorch_program_manager().buffer
    ).program
    assert not serialized.execution_plan[0].delegates


@pytest.mark.parametrize(
    "model", [Multiply(), MultiplyAndFloatUsers(), MultiplyReshapeAndSin()]
)
def test_public_recipe_preserves_numerics(model):
    inputs = (
        torch.linspace(-1, 1, 6).reshape(2, 3),
        torch.linspace(-2, 2, 6).reshape(2, 3),
    )
    outputs = []

    def capture_lowered_outputs(edge):
        outputs.append(edge.exported_program().module()(*inputs))
        return []

    recipe = get_esp_fused_quant_recipe()
    recipe.lowering_recipe.edge_manager_transform_passes.append(capture_lowered_outputs)
    session = export(model, example_inputs=[inputs], export_recipe=recipe)
    reference = session.get_exported_program()
    mul = next(
        node
        for node in reference.graph.nodes
        if node.target == torch.ops.fused_quant.mul.default
    )
    scale = get_scale(reference, get_qparams_from_node(mul, "out"))
    assert len(outputs) == 1
    torch.testing.assert_close(
        outputs[0], reference.module()(*inputs), atol=scale, rtol=0
    )
    tester = EspTester(model, inputs).export_fused_quant().to_edge().run_passes()
    torch.testing.assert_close(
        outputs[0],
        tester.get_artifact().exported_program().module()(*inputs),
        atol=0,
        rtol=0,
    )
    assert session.get_executorch_program_manager().buffer


def test_public_report_classifies_memory_views_as_layout():
    inputs = (torch.ones(2, 3), torch.ones(2, 3))
    session = export(
        MultiplyReshapeAndSin(),
        example_inputs=[inputs],
        export_recipe=get_esp_fused_quant_recipe(),
    )
    report = get_operator_library_report(session.get_edge_program_manager())
    assert {op["target"] for op in report["cpu_ops"]} == {"aten::sin.out"}
    assert {op["target"] for op in report["layout_ops"]} == {
        "executorch.exir.memory.view"
    }


def test_operator_library_strict_mode_rejects_remaining_compute():
    inputs = (torch.linspace(-1, 1, 6), torch.linspace(-2, 2, 6))
    with pytest.raises(RuntimeError, match=r"forward\.\w+ \(aten::sin\)"):
        export(
            MultiplyAndFloatUsers(),
            example_inputs=[inputs],
            export_recipe=get_esp_fused_quant_recipe(allow_cpu_fallback=False),
        )


def test_operator_library_strict_mode_accepts_no_compute():
    session = export(
        torch.nn.Identity(),
        example_inputs=[(torch.ones(3),)],
        export_recipe=get_esp_fused_quant_recipe(allow_cpu_fallback=False),
    )
    assert get_operator_library_report(session.get_edge_program_manager()) == {
        "esp_nn_ops": 0,
        "cpu_ops": [],
        "layout_ops": [],
    }


def test_operator_library_uses_standard_edge_decompositions():
    inputs = (torch.linspace(-1, 1, 6).reshape(2, 3),)
    program = torch.export.export(torch.nn.Linear(3, 2), inputs)
    edge = to_edge_transform_and_lower(program, transform_passes=EspPassManager())
    report = get_operator_library_report(edge)
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
    session = export(
        {name: Multiply() for name in ("forward", "other")},
        example_inputs={name: [inputs] for name in ("forward", "other")},
        export_recipe=get_esp_fused_quant_recipe(),
        constant_methods={"version": 1},
    )
    edge = session.get_edge_program_manager()
    assert get_operator_library_report(edge) == {
        "esp_nn_ops": 2,
        "cpu_ops": [],
        "layout_ops": [],
    }
    serialized = deserialize_pte_binary(
        session.get_executorch_program_manager().buffer
    ).program
    assert {plan.name for plan in serialized.execution_plan} == {
        "forward",
        "other",
        "version",
    }
    assert all(not plan.delegates for plan in serialized.execution_plan)


def test_lowering_preserves_fused_quant_source_program():
    inputs = (torch.linspace(-1, 1, 6), torch.linspace(-2, 2, 6))
    tester = EspTester(MultiplyAndFloatUsers(), inputs).export_fused_quant().to_edge()
    reference = tester.get_artifact().exported_program()
    graph_before = str(reference.graph)
    constants_before = reference.constants.copy()
    expected = reference.module()(*inputs)
    tester.run_passes().to_executorch().serialize()
    assert str(reference.graph) == graph_before
    assert reference.constants.keys() == constants_before.keys()
    assert all(
        reference.constants[key] is value for key, value in constants_before.items()
    )
    reference.validate()
    torch.testing.assert_close(reference.module()(*inputs), expected, atol=0, rtol=0)
