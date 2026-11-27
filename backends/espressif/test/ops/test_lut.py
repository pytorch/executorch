# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import os
from dataclasses import replace
from pathlib import Path

import pytest
import torch
from executorch.backends.espressif import (
    get_esp_fused_quant_recipe,
    get_operator_library_report,
)
from executorch.backends.espressif.passes import EspPassManager
from executorch.backends.espressif.scratch import get_scratch_queries
from executorch.backends.espressif.target import EspKernelMode, EspTarget
from executorch.backends.espressif.test.tester import (
    EspTestCase,
    EspTester,
    ramp_tensor,
)
from executorch.backends.test.suite.operators.test_sigmoid import Model
from executorch.backends.transforms.fused_quant.graph_utils import (
    get_qparams_from_node,
    get_scale,
)
from executorch.backends.transforms.permute_pass_utils import get_arg
from executorch.backends.transforms.utils import get_constant, get_fqn
from executorch.backends.xnnpack.quantizer.xnnpack_quantizer import (
    get_symmetric_quantization_config,
)
from executorch.exir import memory, to_edge
from executorch.exir._serialize._program import deserialize_pte_binary
from executorch.exir.dialects._ops import ops as exir_ops
from executorch.exir.scalar_type import ScalarType
from executorch.exir.schema import KernelCall
from executorch.exir.serde.serialize import deserialize, serialize
from executorch.export import export
from executorch.frontends.fused_quant.quantizer.quantizer import (
    FusedQuantQuantizer,
    OpQuantizer,
)

_FQ_SIGMOID = exir_ops.edge.fused_quant.sigmoid.default
_ESP_LUT = exir_ops.edge.esp_nn.quantized_lut_int8.default
_SIGMOID = exir_ops.edge.aten.sigmoid.default
_PORTABLE_DECOMPOSITIONS = {
    "silu": (torch.ops.aten.silu.default, torch.ops.fused_quant.silu.default),
    "hardswish": (
        torch.ops.aten.hardswish.default,
        torch.ops.fused_quant.hardswish.default,
    ),
    "hardsigmoid": (
        torch.ops.aten.hardsigmoid.default,
        torch.ops.fused_quant.hardsigmoid.default,
    ),
}


class EspLut(torch.nn.Module):
    def __init__(self, activation, fused_op):
        super().__init__()
        self.activation = activation
        self.ops_before_transforms = {fused_op: 1}
        self.ops_after_transforms = {fused_op: 0, _ESP_LUT: 1}

    def forward(self, x):
        return self.activation(x)


class EspSigmoid(Model):
    ops_before_transforms = {_FQ_SIGMOID: 1}
    ops_after_transforms = {_FQ_SIGMOID: 0, _ESP_LUT: 1, _SIGMOID: 0}


class EspSigmoidFloatUsers(EspSigmoid):
    def forward(self, x):
        y = torch.sigmoid(x)
        return y, torch.sin(x), torch.sin(y)


class EspSigmoidChain(EspSigmoid):
    ops_before_transforms = {_FQ_SIGMOID: 2}
    ops_after_transforms = {_FQ_SIGMOID: 0, _ESP_LUT: 2, _SIGMOID: 0}

    def forward(self, x):
        return torch.sigmoid(torch.sigmoid(x))


class EspSigmoidTwoScales(EspSigmoidChain):
    def forward(self, x, y):
        return torch.sigmoid(x), torch.sigmoid(y)


activation_cases = {
    name: EspTestCase(EspLut(activation, fused_op), (ramp_tensor(-16, 16, (4097,)),))
    for name, activation, fused_op in (
        ("sigmoid", torch.nn.Sigmoid(), _FQ_SIGMOID),
        ("tanh", torch.nn.Tanh(), exir_ops.edge.fused_quant.tanh.default),
        ("gelu", torch.nn.GELU(), exir_ops.edge.fused_quant.gelu.default),
        (
            "gelu_tanh",
            torch.nn.GELU(approximate="tanh"),
            exir_ops.edge.fused_quant.gelu.default,
        ),
        ("silu", torch.nn.SiLU(), exir_ops.edge.fused_quant.silu.default),
        (
            "hardswish",
            torch.nn.Hardswish(),
            exir_ops.edge.fused_quant.hardswish.default,
        ),
        (
            "hardsigmoid",
            torch.nn.Hardsigmoid(),
            exir_ops.edge.fused_quant.hardsigmoid.default,
        ),
        ("relu", torch.nn.ReLU(), exir_ops.edge.fused_quant.relu.default),
        ("relu6", torch.nn.ReLU6(), exir_ops.edge.fused_quant.hard_tanh.default),
        (
            "hardtanh",
            torch.nn.Hardtanh(min_val=-2.0, max_val=3.0),
            exir_ops.edge.fused_quant.hard_tanh.default,
        ),
    )
}


test_cases = {
    **activation_cases,
    "scalar": EspTestCase(EspSigmoid(), (torch.tensor(-1.0),)),
    "vector_tail": EspTestCase(EspSigmoid(), (ramp_tensor(-3, 5, (17,)),)),
    "matrix": EspTestCase(EspSigmoid(), (ramp_tensor(-3, 5, (3, 19)),)),
    "rank_four": EspTestCase(EspSigmoid(), (ramp_tensor(-3, 5, (2, 3, 5, 7)),)),
    "positive": EspTestCase(EspSigmoid(), (ramp_tensor(0.5, 6, (17,)),)),
    "negative": EspTestCase(EspSigmoid(), (ramp_tensor(-6, -0.5, (17,)),)),
    "zero": EspTestCase(EspSigmoid(), (torch.zeros(17),)),
    "float_users": EspTestCase(EspSigmoidFloatUsers(), (ramp_tensor(-3, 5, (2, 17)),)),
    "chain": EspTestCase(EspSigmoidChain(), (ramp_tensor(-3, 5, (17,)),)),
    "two_scales": EspTestCase(
        EspSigmoidTwoScales(),
        (ramp_tensor(-0.5, 0.5, (17,)), ramp_tensor(-8, 8, (17,))),
    ),
    "saturation": EspTestCase(
        EspSigmoid(),
        (ramp_tensor(-50, 50, (257,)),),
        [(ramp_tensor(-2, 2, (257,)),)],
    ),
}


@pytest.mark.parametrize("case", test_cases.values(), ids=test_cases.keys())
def test_dialect_lut(case):
    tester = EspTester(case.model, case.example_inputs, case.calibration_samples)
    tester.test_dialect(
        case.model.ops_before_transforms,
        case.model.ops_after_transforms,
        atol=0,
        rtol=0,
    )
    program = tester.get_artifact().exported_program()
    for node in program.graph.find_nodes(op="call_function", target=_ESP_LUT):
        table = get_constant(program, get_arg(node, "table"))
        assert table.dtype == torch.int8
        assert table.shape == (256,)
    tester.to_executorch().serialize()
    plan = deserialize_pte_binary(tester.get_artifact()).program.execution_plan[0]
    assert not plan.delegates
    assert all(not op.name.startswith("fused_quant::") for op in plan.operators)
    lut_calls = 0
    for chain in plan.chains:
        for instruction in chain.instructions:
            call = instruction.instr_args
            if (
                isinstance(call, KernelCall)
                and plan.operators[call.op_index].name == "esp_nn::quantized_lut_int8"
            ):
                lut_calls += 1
                table = plan.values[call.args[1]].val
                assert table.scalar_type == ScalarType.CHAR
                assert table.data_buffer_idx > 0
                assert table.allocation_info is None
    assert lut_calls == case.model.ops_after_transforms[_ESP_LUT]


@pytest.mark.parametrize("case", test_cases.values(), ids=test_cases.keys())
def test_implementation_lut(case, pytestconfig):
    EspTester(
        case.model, case.example_inputs, case.calibration_samples
    ).test_implementation(
        pytestconfig.rootpath, rtol=torch.finfo(torch.float32).eps, num_inferences=2
    )


def test_scalar_lut_output_does_not_alias_table():
    table = torch.arange(-128, 128, dtype=torch.int8)
    expected_table = table.clone()
    output = torch.ops.esp_nn.quantized_lut_int8.default(
        torch.tensor(-1, dtype=torch.int8), table
    )
    torch.testing.assert_close(output, torch.tensor(127, dtype=torch.int8))
    output.zero_()
    torch.testing.assert_close(table, expected_table)


def test_implementation_lut_undersized_table(pytestconfig):
    if not os.environ.get("ESPRESSIF_BUILD_DIR"):
        pytest.skip("Set ESPRESSIF_BUILD_DIR to the ESP32-S3 build directory")
    case = test_cases["zero"]
    tester = (
        EspTester(case.model, case.example_inputs)
        .export_fused_quant()
        .to_edge()
        .run_passes()
    )
    program = tester.get_artifact().exported_program()
    (node,) = program.graph.find_nodes(op="call_function", target=_ESP_LUT)
    table = get_arg(node, "table")
    program.constants[get_fqn(program, table)] = get_constant(program, table)[
        :255
    ].clone()
    table.meta["val"] = table.meta["val"].new_empty((255,))
    with pytest.raises(AssertionError):
        tester.test_implementation(pytestconfig.rootpath)
    worker = os.environ.get("PYTEST_XDIST_WORKER", "main")
    log = (
        Path(os.environ["ESPRESSIF_BUILD_DIR"])
        / f"operator-library-{worker}-{tester.target.value}-{tester.kernel_mode.value}"
        / "emulator.log"
    )
    text = log.read_text()
    assert "op_quantized_lut.cpp" in text
    assert "table.nbytes()" in text


@pytest.mark.parametrize("case", activation_cases.values(), ids=activation_cases.keys())
def test_lut_covers_all_input_codes(case):
    tester = EspTester(case.model, case.example_inputs).test_dialect(
        case.model.ops_before_transforms,
        case.model.ops_after_transforms,
        atol=0,
        rtol=0,
    )
    module = tester.get_artifact().exported_program().module()
    interpreter = torch.fx.Interpreter(module, garbage_collect_values=False)
    interpreter.run(*case.example_inputs)
    (node,) = module.graph.find_nodes(op="call_function", target=_ESP_LUT)
    codes = interpreter.env[get_arg(node, "input")]
    torch.testing.assert_close(
        codes.unique().to(torch.int32),
        torch.arange(-128, 128, dtype=torch.int32),
        atol=0,
        rtol=0,
    )


@pytest.mark.parametrize(
    "float_edge",
    ["input_activation", "output_activation"],
    ids=["float_input", "float_output"],
)
def test_partially_quantized_sigmoid_stays_portable(float_edge):
    config = replace(get_symmetric_quantization_config(), **{float_edge: None})
    quantizer = FusedQuantQuantizer(
        [
            OpQuantizer(
                torch.ops.aten.sigmoid.default,
                torch.ops.fused_quant.sigmoid.default,
                config,
            )
        ]
    )
    case = test_cases["float_users"]
    tester = (
        EspTester(case.model, case.example_inputs)
        .export_fused_quant(quantizer)
        .to_edge()
        .check_node_count({_FQ_SIGMOID: 1})
    )
    expected = tester.get_artifact().exported_program().module()(*case.example_inputs)
    tester.run_passes().check_node_count({_FQ_SIGMOID: 0, _ESP_LUT: 0, _SIGMOID: 1})
    torch.testing.assert_close(
        tester.get_artifact().exported_program().module()(*case.example_inputs),
        expected,
        atol=0,
        rtol=0,
    )
    tester.to_executorch().serialize()


@pytest.mark.parametrize(
    "input_min, output_min", [(-127, -128), (-128, -127)], ids=["input", "output"]
)
def test_sigmoid_narrow_int8_range_stays_portable(input_min, output_min):
    config = get_symmetric_quantization_config()
    config = replace(
        config,
        input_activation=replace(config.input_activation, quant_min=input_min),
        output_activation=replace(config.output_activation, quant_min=output_min),
    )
    quantizer = FusedQuantQuantizer(
        [
            OpQuantizer(
                torch.ops.aten.sigmoid.default,
                torch.ops.fused_quant.sigmoid.default,
                config,
            )
        ]
    )
    case = test_cases["float_users"]
    tester = EspTester(case.model, case.example_inputs).export_fused_quant(quantizer)
    tester.to_edge()
    program = tester.get_artifact().exported_program()
    (node,) = program.graph.find_nodes(op="call_function", target=_FQ_SIGMOID)
    assert get_arg(node, "inp").meta["val"].dtype == torch.int8
    assert node.meta["val"].dtype == torch.int8
    assert get_qparams_from_node(node, "inp").quant_min == input_min
    assert get_qparams_from_node(node, "out").quant_min == output_min
    expected = program.module()(*case.example_inputs)
    tester.run_passes().check_node_count({_FQ_SIGMOID: 0, _ESP_LUT: 0, _SIGMOID: 1})
    torch.testing.assert_close(
        tester.get_artifact().exported_program().module()(*case.example_inputs),
        expected,
        atol=0,
        rtol=0,
    )
    tester.to_executorch().serialize()


def _check_portable_activation(edge, input_cases, expected_outputs):
    restored = deserialize(serialize(edge.exported_program())).module()
    for inputs, expected in zip(input_cases, expected_outputs):
        torch.testing.assert_close(restored(*inputs), expected, atol=1e-6, rtol=1e-6)
    buffer = edge.to_executorch().buffer
    plan = deserialize_pte_binary(buffer).program.execution_plan[0]
    assert not plan.delegates
    assert all(
        op.name
        not in {
            "esp_nn::quantized_lut_int8",
            "aten::silu",
            "aten::hardswish",
            "aten::hardsigmoid",
        }
        and not op.name.startswith("fused_quant::")
        for op in plan.operators
    )
    portable = pytest.importorskip("executorch.extension.pybindings.portable_lib")
    runtime = portable._load_for_executorch_from_buffer(buffer)
    for inputs, expected in zip(input_cases, expected_outputs):
        torch.testing.assert_close(
            runtime.forward(inputs)[0], expected, atol=1e-6, rtol=1e-6
        )


@pytest.mark.parametrize("name", _PORTABLE_DECOMPOSITIONS)
def test_dynamic_activation_falls_back_to_portable(name):
    model = activation_cases[name].model
    inputs = (ramp_tensor(-16, 16, (2, 257)),)
    fused = EspTester(model, inputs).export_fused_quant().get_artifact().module()
    batch = torch.export.Dim("batch", min=1, max=8)
    program = torch.export.export(fused, inputs, dynamic_shapes=({0: batch},))
    input_cases = [inputs, (ramp_tensor(-16, 16, (4, 257)),)]
    expected = [program.module()(*values) for values in input_cases]
    edge = to_edge(
        program,
        compile_config=get_esp_fused_quant_recipe().lowering_recipe.edge_compile_config,
    )
    (fused_op,) = model.ops_before_transforms
    (source,) = edge.exported_program().graph.find_nodes(
        op="call_function", target=fused_op
    )
    edge = edge.transform(EspPassManager())
    first_op = (
        exir_ops.edge.aten.sigmoid.default
        if name == "silu"
        else exir_ops.edge.aten.add.Tensor
    )
    (replacement,) = edge.exported_program().graph.find_nodes(
        op="call_function", target=first_op
    )
    assert replacement.meta["debug_handle"] == source.meta["debug_handle"]
    assert replacement.meta["stack_trace"] == source.meta["stack_trace"]
    _check_portable_activation(edge, input_cases, expected)


@pytest.mark.parametrize("name", _PORTABLE_DECOMPOSITIONS)
@pytest.mark.parametrize("float_edge", ["input_activation", "output_activation"])
def test_partially_quantized_activation_falls_back_to_portable(name, float_edge):
    aten_op, fused_op = _PORTABLE_DECOMPOSITIONS[name]
    config = replace(get_symmetric_quantization_config(), **{float_edge: None})
    quantizer = FusedQuantQuantizer([OpQuantizer(aten_op, fused_op, config)])
    case = activation_cases[name]
    program = (
        EspTester(case.model, case.example_inputs)
        .export_fused_quant(quantizer)
        .get_artifact()
    )
    edge = to_edge(
        program,
        compile_config=get_esp_fused_quant_recipe(
            quantizer=quantizer
        ).lowering_recipe.edge_compile_config,
    )
    (edge_fused_op,) = case.model.ops_before_transforms
    assert list(
        edge.exported_program().graph.find_nodes(
            op="call_function", target=edge_fused_op
        )
    )
    expected = edge.exported_program().module()(*case.example_inputs)
    edge = edge.transform(EspPassManager())
    _check_portable_activation(edge, [case.example_inputs], [expected])


@pytest.mark.parametrize("name", _PORTABLE_DECOMPOSITIONS)
def test_activation_fallback_preserves_convolution_scratch(name):
    if get_scratch_queries(EspTarget.ESP32S3, EspKernelMode.OPTIMIZED) is None:
        pytest.skip("Build with EXECUTORCH_BUILD_ESP_NN_PYBINDS=ON")

    class Mixed(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.activation = activation_cases[name].model.activation
            self.conv = torch.nn.Conv2d(4, 8, 3, padding=1)
            with torch.no_grad():
                self.conv.weight.fill_(0.125)
                self.conv.bias.fill_(0.25)

        def forward(self, dynamic, static):
            return self.activation(dynamic), self.conv(static)

    inputs = (ramp_tensor(-16, 16, (2, 257)), ramp_tensor(-3, 5, (1, 4, 5, 7)))
    recipe = get_esp_fused_quant_recipe()
    recipe.pipeline_stages = recipe.pipeline_stages[:-2]
    session = export(
        Mixed(),
        example_inputs=[inputs],
        dynamic_shapes=({0: torch.export.Dim("batch", min=1, max=8)}, {}),
        export_recipe=recipe,
    )
    edge = session.get_edge_program_manager()
    reference = edge.exported_program()
    expected = reference.module()(*inputs)
    (conv,) = reference.graph.find_nodes(
        op="call_function", target=exir_ops.edge.fused_quant.convolution.default
    )
    conv_scale = get_scale(reference, get_qparams_from_node(conv, "out"))
    edge = edge.transform(EspPassManager())
    program = edge.exported_program()
    actual = program.module()(*inputs)
    torch.testing.assert_close(actual[0], expected[0], atol=0, rtol=0)
    torch.testing.assert_close(actual[1], expected[1], atol=conv_scale, rtol=1e-6)
    esp_conv = exir_ops.edge.esp_nn.quantized_conv2d_int8.default
    (conv,) = program.graph.find_nodes(op="call_function", target=esp_conv)
    scratch = get_arg(conv, "scratch")
    assert scratch.target == memory.alloc
    expected_bytes = scratch.meta["val"].numel()
    assert expected_bytes > 0
    plan = deserialize_pte_binary(edge.to_executorch().buffer).program.execution_plan[0]
    calls = [
        instruction.instr_args
        for chain in plan.chains
        for instruction in chain.instructions
        if isinstance(instruction.instr_args, KernelCall)
        and plan.operators[instruction.instr_args.op_index].name
        == "esp_nn::quantized_conv2d_int8"
    ]
    (call,) = calls
    scratch_index = next(
        i for i, arg in enumerate(esp_conv._schema.arguments) if arg.name == "scratch"
    )
    stored = plan.values[call.args[scratch_index]].val
    assert stored.sizes == [expected_bytes]
    assert stored.scalar_type == ScalarType.BYTE
    assert stored.data_buffer_idx == 0
    assert stored.allocation_info.memory_id > 0
    assert stored.allocation_info.memory_offset % 16 == 0


def test_public_recipe_lut_without_sizing(monkeypatch):
    def unexpected_sizing(*args):
        pytest.fail("A constant lookup table does not require a sizing binding")

    monkeypatch.setattr(
        "executorch.backends.espressif.passes.lower_to_esp.get_scratch_queries",
        unexpected_sizing,
    )
    case = test_cases["two_scales"]
    session = export(
        case.model,
        example_inputs=[case.example_inputs],
        export_recipe=get_esp_fused_quant_recipe(allow_cpu_fallback=False),
    )
    report = get_operator_library_report(session.get_edge_program_manager())
    assert report["esp_nn_ops"] == 2
    assert report["cpu_ops"] == []
    assert report["layout_ops"] == []
    assert report["quantization_ops"]
    assert session.get_executorch_program_manager().buffer
