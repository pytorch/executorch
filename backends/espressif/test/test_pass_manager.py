# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import copy
import math

import pytest
import torch
from executorch.backends.espressif import (
    EspNNQuantizer,
    get_esp_fused_quant_recipe,
    get_operator_library_report,
    to_espnn_edge,
)
from executorch.backends.espressif.passes import EspPassManager
from executorch.backends.espressif.scratch import get_scratch_queries
from executorch.backends.espressif.target import EspKernelMode, EspTarget
from executorch.backends.espressif.test.tester import EspRunPasses, EspTester
from executorch.backends.test.harness.stages import Quantize
from executorch.backends.transforms.fused_quant.graph_utils import (
    get_qparams_from_node,
    get_scale,
)
from executorch.exir import to_edge, to_edge_transform_and_lower
from executorch.exir._serialize._program import deserialize_pte_binary
from executorch.exir.dialects._ops import ops as exir_ops
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


class MultiplyWithConstants(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.register_buffer("scale", torch.linspace(1, 2, 6))
        self.offset = torch.linspace(0, 1, 6)

    def forward(self, input, y):
        return input * y, torch.sin(input * self.scale), torch.cos(input + self.offset)


class SpatialAndFloatUsers(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(4, 4, 3, groups=2)
        self.pool = torch.nn.MaxPool2d(2)

    def forward(self, x):
        y = self.conv(x)
        return self.pool(y), torch.sin(y)


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
    report = get_operator_library_report(edge)
    assert report["esp_nn_ops"] == 1
    assert report["cpu_ops"] == []
    assert report["layout_ops"] == []
    assert {op["target"] for op in report["quantization_ops"]} == {
        "quantized_decomposed::quantize_per_tensor.out",
        "quantized_decomposed::dequantize_per_tensor.out",
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
        "quantization_ops": [],
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
    report = get_operator_library_report(edge)
    assert report["esp_nn_ops"] == 2
    assert report["cpu_ops"] == []
    assert report["layout_ops"] == []
    assert {op["method"] for op in report["quantization_ops"]} == {"forward", "other"}
    serialized = deserialize_pte_binary(
        session.get_executorch_program_manager().buffer
    ).program
    assert {plan.name for plan in serialized.execution_plan} == {
        "forward",
        "other",
        "version",
    }
    assert all(not plan.delegates for plan in serialized.execution_plan)


@pytest.mark.parametrize(
    "model, inputs",
    [
        (MultiplyAndFloatUsers(), (torch.linspace(-1, 1, 6), torch.linspace(-2, 2, 6))),
        (MultiplyWithConstants(), (torch.linspace(-1, 1, 6), torch.linspace(-2, 2, 6))),
        (torch.nn.MaxPool2d(2), (torch.linspace(-1, 1, 140).reshape(1, 4, 5, 7),)),
        (torch.nn.AvgPool2d(2), (torch.linspace(-1, 1, 100).reshape(1, 4, 5, 5),)),
    ],
    ids=["multiply", "constants", "maxpool", "avgpool"],
)
def test_lowering_is_isolated_and_idempotent(model, inputs):
    tester = EspTester(model, inputs).export_fused_quant().to_edge()
    reference = tester.get_artifact().exported_program()
    reference_before = copy.deepcopy(reference)
    expected = reference.module()(*inputs)
    tester.run_passes()
    lowered = tester.get_artifact().exported_program()
    assert str(lowered.graph) != str(reference_before.graph)
    lowered_before = copy.deepcopy(lowered)
    repeated = EspPassManager()(lowered).exported_program
    for program, snapshot in (
        (reference, reference_before),
        (lowered, lowered_before),
        (repeated, lowered_before),
    ):
        assert str(program.graph) == str(snapshot.graph)
        assert program.graph_signature == snapshot.graph_signature
        torch.testing.assert_close(
            program.state_dict, snapshot.state_dict, atol=0, rtol=0
        )
        torch.testing.assert_close(
            program.constants, snapshot.constants, atol=0, rtol=0
        )
        program.validate()
    torch.testing.assert_close(reference.module()(*inputs), expected, atol=0, rtol=0)
    tester.to_executorch().serialize()


@pytest.mark.parametrize(
    "model, channels, pools, permutes",
    [
        (torch.nn.Conv2d(4, 4, 3, groups=2), 4, 0, 0),
        (torch.nn.AvgPool2d(2, divisor_override=3), 4, 0, 0),
        (torch.nn.MaxPool2d(2), 5, 0, 0),
        (
            torch.nn.Sequential(
                torch.nn.Conv2d(4, 4, 3, groups=2),
                torch.nn.Conv2d(4, 4, 3, groups=2),
            ),
            4,
            0,
            0,
        ),
        (
            torch.nn.Sequential(
                torch.nn.Conv2d(4, 4, 3, groups=2), torch.nn.MaxPool2d(2)
            ),
            4,
            1,
            2,
        ),
        (
            torch.nn.Sequential(
                torch.nn.MaxPool2d(2), torch.nn.Conv2d(4, 4, 3, groups=2)
            ),
            4,
            1,
            2,
        ),
        (SpatialAndFloatUsers(), 4, 1, 2),
        (torch.nn.Sequential(torch.nn.MaxPool2d(2), torch.nn.MaxPool2d(2)), 4, 2, 2),
    ],
    ids=[
        "portable_conv",
        "portable_avgpool",
        "portable_maxpool",
        "portable_conv_chain",
        "portable_conv_to_esp_pool",
        "esp_pool_to_portable_conv",
        "float_users",
        "esp_pool_chain",
    ],
)
def test_spatial_fallback_removes_redundant_layout_copies(
    model, channels, pools, permutes
):
    inputs = (torch.linspace(-1, 1, channels * 9 * 11).reshape(1, channels, 9, 11),)
    tester = EspTester(model, inputs).export_fused_quant().to_edge()
    expected = tester.get_artifact().exported_program().module()(*inputs)
    tester.run_passes().check_node_count(
        {
            exir_ops.edge.aten.permute_copy.default: permutes,
            exir_ops.edge.aten.view_copy.default: 0,
            exir_ops.edge.esp_nn.quantized_max_pool2d_int8.default: pools,
        }
    )
    torch.testing.assert_close(
        tester.get_artifact().exported_program().module()(*inputs),
        expected,
        atol=0,
        rtol=0,
    )
    tester.to_executorch().serialize()


@pytest.mark.parametrize(
    "model, shape",
    [
        (torch.nn.MaxPool2d(2), (1, 4, 5, 7)),
        (torch.nn.MaxPool2d(2), (4, 7, 9)),
        (torch.nn.MaxPool2d((5, 7)), (1, 4, 5, 7)),
        (torch.nn.AvgPool2d(2), (1, 4, 5, 5)),
    ],
    ids=["batched", "unbatched", "global", "avgpool"],
)
def test_strict_export_accepts_required_layout_copies(model, shape):
    inputs = (torch.linspace(-1, 1, math.prod(shape)).reshape(shape),)
    session = export(
        model,
        example_inputs=[inputs],
        export_recipe=get_esp_fused_quant_recipe(allow_cpu_fallback=False),
    )
    report = get_operator_library_report(session.get_edge_program_manager())
    assert report["esp_nn_ops"] == 1
    assert report["cpu_ops"] == []
    assert "aten::permute_copy.out" in {op["target"] for op in report["layout_ops"]}
    assert session.get_executorch_program_manager().buffer


def test_strict_export_accepts_user_permute():
    class PermuteAndPool(torch.nn.Module):
        def forward(self, x):
            return torch.nn.functional.max_pool2d(x.permute(0, 1, 3, 2), 2)

    session = export(
        PermuteAndPool(),
        example_inputs=[(torch.linspace(-1, 1, 140).reshape(1, 4, 5, 7),)],
        export_recipe=get_esp_fused_quant_recipe(allow_cpu_fallback=False),
    )
    report = get_operator_library_report(session.get_edge_program_manager())
    assert report["esp_nn_ops"] == 1
    assert report["cpu_ops"] == []
    assert len(report["layout_ops"]) == 2
    assert session.get_executorch_program_manager().buffer


def test_strict_export_accepts_user_permute_between_esp_operators():
    class PoolPermuteAndPool(torch.nn.Module):
        def forward(self, x):
            pooled = torch.nn.functional.max_pool2d(x, 2)
            return torch.nn.functional.max_pool2d(pooled.permute(0, 2, 3, 1), 2)

    inputs = (
        torch.linspace(-1, 1, 4).reshape(1, 4, 1, 1).expand(1, 4, 16, 16).contiguous(),
    )
    model = PoolPermuteAndPool()
    session = export(
        model,
        example_inputs=[inputs],
        export_recipe=get_esp_fused_quant_recipe(allow_cpu_fallback=False),
    )
    report = get_operator_library_report(session.get_edge_program_manager())
    assert report["esp_nn_ops"] == 2
    assert report["cpu_ops"] == []
    assert "aten::permute_copy.out" in {op["target"] for op in report["layout_ops"]}
    assert session.get_executorch_program_manager().buffer


def test_strict_export_rejects_portable_pool():
    with pytest.raises(RuntimeError, match="aten::max_pool2d_with_indices"):
        export(
            torch.nn.MaxPool2d(2),
            example_inputs=[(torch.ones(1, 5, 5, 7),)],
            export_recipe=get_esp_fused_quant_recipe(allow_cpu_fallback=False),
        )


def test_strict_export_accepts_layout_without_accelerated_compute():
    class PermuteAndReshape(torch.nn.Module):
        def forward(self, x):
            return x.permute(1, 0).reshape(-1)

    inputs = (torch.arange(12, dtype=torch.float32).reshape(3, 4),)
    model = PermuteAndReshape()
    session = export(
        model,
        example_inputs=[inputs],
        export_recipe=get_esp_fused_quant_recipe(allow_cpu_fallback=False),
    )
    edge = session.get_edge_program_manager()
    report = get_operator_library_report(edge)
    assert report["esp_nn_ops"] == 0
    assert report["cpu_ops"] == []
    assert report["quantization_ops"] == []
    assert {op["target"] for op in report["layout_ops"]} == {
        "aten::permute_copy.out",
        "aten::view_copy.out",
    }
    torch.testing.assert_close(
        edge.exported_program().module()(*inputs), model(*inputs)
    )
    assert session.get_executorch_program_manager().buffer


@pytest.mark.parametrize(
    "spatial",
    [
        torch.nn.Conv2d(4, 8, 3),
        torch.nn.Conv2d(4, 4, 3, groups=4),
        torch.nn.MaxPool2d(2),
        torch.nn.AvgPool2d(2),
    ],
    ids=["conv2d", "depthwise", "maxpool", "avgpool"],
)
def test_strict_export_accepts_layout_after_hoisted_activation(spatial):
    if (
        isinstance(spatial, torch.nn.Conv2d)
        and get_scratch_queries(EspTarget.ESP32S3, EspKernelMode.OPTIMIZED) is None
    ):
        pytest.skip("Build with EXECUTORCH_BUILD_ESP_NN_PYBINDS=ON")
    model = torch.nn.Sequential(spatial, torch.nn.Sigmoid())
    inputs = (torch.linspace(-1, 1, 140).reshape(1, 4, 5, 7),)
    session = export(
        model,
        example_inputs=[inputs],
        export_recipe=get_esp_fused_quant_recipe(allow_cpu_fallback=False),
    )
    report = get_operator_library_report(session.get_edge_program_manager())
    assert report["esp_nn_ops"] == 2
    assert report["cpu_ops"] == []
    assert len(report["layout_ops"]) == 2
    assert all(op["target"] == "aten::permute_copy.out" for op in report["layout_ops"])
    assert {op["target"] for op in report["quantization_ops"]} == {
        "quantized_decomposed::quantize_per_tensor.out",
        "quantized_decomposed::dequantize_per_tensor.out",
    }
    assert session.get_executorch_program_manager().buffer


def test_strict_export_rejects_compute_after_layout():
    class PermuteAndSin(torch.nn.Module):
        def forward(self, x):
            return torch.sin(x.permute(1, 0))

    with pytest.raises(RuntimeError, match=r"forward\.\w+ \(aten::sin\)") as error:
        export(
            PermuteAndSin(),
            example_inputs=[(torch.ones(3, 4),)],
            export_recipe=get_esp_fused_quant_recipe(allow_cpu_fallback=False),
        )
    assert "aten::permute_copy" not in str(error.value)


@pytest.mark.parametrize(
    "model, shape",
    [
        (torch.nn.Conv1d(4, 8, 3), (1, 4, 9)),
        (torch.nn.Conv2d(4, 8, 3, groups=2), (1, 4, 7, 9)),
    ],
    ids=["conv1d", "grouped_conv2d"],
)
def test_other_spatial_operators_remain_portable(model, shape):
    inputs = (torch.linspace(-1, 1, math.prod(shape)).reshape(shape),)
    tester = EspTester(model, inputs).export_fused_quant().to_edge()
    expected = tester.get_artifact().exported_program().module()(*inputs)
    tester.run_passes()
    assert get_operator_library_report(tester.get_artifact())["esp_nn_ops"] == 0
    torch.testing.assert_close(
        tester.get_artifact().exported_program().module()(*inputs),
        expected,
        atol=0,
        rtol=0,
    )
    tester.to_executorch().serialize()
