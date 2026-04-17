# Copyright 2025-2026 Arm Limited and/or its affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.


from typing import Tuple

import pytest
import torch

from executorch.backends.arm._passes.insert_rescales_pass import InsertRescalePass
from executorch.backends.arm.constants import DQ_OPS, Q_OPS
from executorch.backends.arm.quantizer import (
    get_symmetric_quantization_config,
    get_uint8_io_quantization_config,
    TOSAQuantizer,
)

from executorch.backends.arm.test import common
from executorch.backends.arm.test.tester.arm_tester import ArmTester, RunPasses
from executorch.backends.arm.test.tester.quantize import ArmQuantize as Quantize
from executorch.backends.arm.test.tester.test_pipeline import (
    EthosU55PipelineINT,
    TosaPipelineINT,
    VgfPipeline,
)
from executorch.backends.arm.tosa.specification import (
    TosaLoweringContext,
    TosaSpecification,
)
from executorch.backends.cadence.aot.graph_builder import GraphBuilder
from executorch.backends.test.harness.stages import StageType
from executorch.exir.dialects._ops import ops as exir_ops
from executorch.exir.passes.quantize_io_pass import (
    quantize_input,
    quantize_output,
    QuantizeInputs,
    QuantizeOutputs,
)


input_t = Tuple[torch.Tensor, torch.Tensor]


class SimpleModel(torch.nn.Module):
    test_data = {
        "rand_rand": (torch.rand(1, 2, 2, 1), torch.rand(1, 2, 2, 1)),
    }

    def forward(self, x, y):
        return x + y


@common.parametrize("test_data", SimpleModel.test_data)
def test_quantize_io_u55_INT(test_data: input_t):
    """Test the executorch/exir/passes/quantize_io_pass pass works(meaning we
    don't get Q/DQ nodes) on a simple model.
    """
    model = SimpleModel()
    pipeline = EthosU55PipelineINT(
        model,
        test_data,
        aten_ops=[],
        exir_ops=[],
        use_to_edge_transform_and_lower=True,
        run_on_fvp=False,
    )
    pipeline.pop_stage(-1)
    pipeline.run()
    edge = pipeline.tester.get_artifact()
    edge.transform(passes=[QuantizeInputs(edge, [0, 1]), QuantizeOutputs(edge, [0])])
    pipeline.tester.check_not(["edge__ops_quantized_decomposed_quantize_per_tensor"])
    pipeline.tester.check_not(["edge__ops_quantized_decomposed_dequantize_per_tensor"])


class SimpleMLP(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = torch.nn.Linear(4, 8)
        self.relu = torch.nn.ReLU()
        self.fc2 = torch.nn.Linear(8, 4)

    def forward(self, x):
        return self.fc2(self.relu(self.fc1(x)))


def _build_dq_q_graph(
    input_tensor: torch.Tensor,
    dq_dtype: torch.dtype,
    q_dtype: torch.dtype,
    dq_scale: float,
    dq_zp: int,
    q_scale: float,
    q_zp: int,
):
    builder = GraphBuilder()
    x = builder.placeholder("x", input_tensor)
    qd_ops = exir_ops.edge.quantized_decomposed
    dq = builder.call_operator(
        qd_ops.dequantize_per_tensor.default,
        (
            x,
            dq_scale,
            dq_zp,
            torch.iinfo(dq_dtype).min,
            torch.iinfo(dq_dtype).max,
            dq_dtype,
        ),
    )
    q = builder.call_operator(
        qd_ops.quantize_per_tensor.default,
        (
            dq,
            q_scale,
            q_zp,
            torch.iinfo(q_dtype).min,
            torch.iinfo(q_dtype).max,
            q_dtype,
        ),
    )
    builder.output([q])
    return builder.get_graph_module()


def test_insert_rescale_tosa_INT_folds_uint8_input():
    graph_module = _build_dq_q_graph(
        torch.randint(0, 255, (1, 4), dtype=torch.uint8),
        torch.uint8,
        torch.int8,
        dq_scale=0.5,
        dq_zp=0,
        q_scale=0.25,
        q_zp=0,
    )
    with TosaLoweringContext(TosaSpecification.create_from_string("TOSA-1.0+INT")):
        rescale_graph = InsertRescalePass()(graph_module).graph_module.graph
    rescale_nodes = [
        node
        for node in rescale_graph.nodes
        if node.op == "call_function"
        and node.target == exir_ops.backend.tosa.RESCALE.default
    ]
    assert rescale_nodes
    assert rescale_nodes[0].kwargs.get("input_unsigned") is True
    assert rescale_nodes[0].kwargs.get("output_unsigned") is False


def test_insert_rescale_tosa_INT_folds_uint8_output():
    graph_module = _build_dq_q_graph(
        torch.randint(-128, 127, (1, 4), dtype=torch.int8),
        torch.int8,
        torch.uint8,
        dq_scale=0.5,
        dq_zp=0,
        q_scale=0.25,
        q_zp=0,
    )
    with TosaLoweringContext(TosaSpecification.create_from_string("TOSA-1.0+INT")):
        rescale_graph = InsertRescalePass()(graph_module).graph_module.graph
    rescale_nodes = [
        node
        for node in rescale_graph.nodes
        if node.op == "call_function"
        and node.target == exir_ops.backend.tosa.RESCALE.default
    ]
    assert rescale_nodes
    assert rescale_nodes[0].kwargs.get("input_unsigned") is False
    assert rescale_nodes[0].kwargs.get("output_unsigned") is True
    assert rescale_nodes[0].args[1] == torch.int8


def test_quantize_io_tosa_INT_uint8_simple_mlp():
    """Float-input MLP uses uint8 IO quantization and folds to a single
    delegate.
    """
    model = SimpleMLP().eval()
    test_data = (torch.rand(1, 4),)
    compile_spec = common.get_tosa_compile_spec("TOSA-1.0+INT")

    tester = ArmTester(model, test_data, compile_spec)
    quantizer = TOSAQuantizer(compile_spec)
    quantizer.set_global(get_symmetric_quantization_config())
    quantizer.set_io(get_uint8_io_quantization_config())
    quant_stage = Quantize(quantizer, quantization_config=quantizer.global_config)

    tester.quantize(quant_stage)
    tester.export()
    tester.to_edge_transform_and_lower()
    tester.check_count({"torch.ops.higher_order.executorch_call_delegate": 1})
    tester.check(
        [
            "executorch_exir_dialects_edge__ops_quantized_decomposed_quantize_per_tensor_default"
        ]
    )
    lowered_graph = tester.get_artifact().exported_program().graph_module.graph
    delegate_nodes = [
        node
        for node in lowered_graph.nodes
        if node.op == "call_function"
        and node.target == torch.ops.higher_order.executorch_call_delegate
    ]
    assert len(delegate_nodes) == 1
    quant_nodes = [
        node
        for node in lowered_graph.nodes
        if node.op == "call_function"
        and node.target
        == exir_ops.edge.quantized_decomposed.quantize_per_tensor.default
    ]
    assert len(quant_nodes) == 1
    delegate_args = delegate_nodes[0].all_input_nodes
    assert (
        quant_nodes[0] in delegate_args
    ), "Expected input quantize to feed the delegate call."


def test_quantize_io_tosa_INT_uint8():
    """Make sure quantizer doesn't allow uint8 internally."""
    torch_q_ops = (
        torch.ops.quantized_decomposed.quantize_per_tensor.default,
        torch.ops.quantized_decomposed.quantize_per_tensor.tensor,
        torch.ops.quantized_decomposed.quantize_per_channel.default,
    )
    torch_dq_ops = (
        torch.ops.quantized_decomposed.dequantize_per_tensor.default,
        torch.ops.quantized_decomposed.dequantize_per_tensor.tensor,
        torch.ops.quantized_decomposed.dequantize_per_channel.default,
    )

    model = SimpleMLP().eval()
    test_data = (torch.rand(1, 4),)
    compile_spec = common.get_tosa_compile_spec("TOSA-1.0+INT")

    tester = ArmTester(model, test_data, compile_spec)
    quantizer = TOSAQuantizer(compile_spec)
    quantizer.set_global(get_symmetric_quantization_config())
    quantizer.set_io(get_uint8_io_quantization_config())
    quant_stage = Quantize(quantizer, quantization_config=quantizer.global_config)

    tester.quantize(quant_stage)
    tester.export()

    exported_program = tester.get_artifact()
    graph = exported_program.graph_module.graph
    placeholders = [node for node in graph.nodes if node.op == "placeholder"]
    output_nodes = [node for node in graph.nodes if node.op == "output"]

    max_uint8_q_nodes = len(placeholders) + len(output_nodes)
    uint8_q_nodes = []
    bad_nodes = []
    for node in graph.nodes:
        meta_val = node.meta.get("val")
        if not isinstance(meta_val, torch.Tensor):
            continue
        if meta_val.dtype != torch.uint8:
            continue
        if node.op in ("placeholder", "output"):
            continue
        if node.op == "call_function" and node.target in (*Q_OPS, *torch_q_ops):
            uint8_q_nodes.append((node.name, node.target))
            continue
        if node.op == "call_function" and node.target in (*DQ_OPS, *torch_dq_ops):
            bad_nodes.append((node.name, node.target))
            continue
        bad_nodes.append((node.name, node.target))
    assert not bad_nodes, (
        "Found internal uint8 tensors outside IO boundaries: " f"{bad_nodes}"
    )
    assert len(uint8_q_nodes) <= max_uint8_q_nodes, (
        "Expected uint8 quantize nodes only at IO boundaries; "
        f"found {len(uint8_q_nodes)} (max {max_uint8_q_nodes})."
    )


def test_quantize_io_tosa_INT_uint8_pipeline():
    """Use TOSA pipeline to build an end-to-end flow and accept uint8 IO."""
    model = SimpleMLP().eval()
    test_data = (torch.rand(1, 4),)
    pipeline = TosaPipelineINT(
        model,
        test_data,
        [],
        [],
        run_on_tosa_ref_model=True,
        use_to_edge_transform_and_lower=False,
    )

    tester = pipeline.tester
    tester.quantize()
    tester.export()
    tester.to_edge()
    edge = tester.get_artifact()
    edge.transform(
        passes=[
            QuantizeInputs(
                edge,
                {
                    0: {
                        "scale": 1.0,
                        "zp": 0,
                        "dtype": torch.uint8,
                    }
                },
            ),
            QuantizeOutputs(
                edge,
                {
                    0: {
                        "scale": 1.0,
                        "zp": 0,
                        "dtype": torch.uint8,
                    }
                },
            ),
        ]
    )

    exported_program = edge.exported_program()
    graph_module = exported_program.graph_module
    with TosaLoweringContext(TosaSpecification.create_from_string("TOSA-1.0+INT")):
        graph = InsertRescalePass()(graph_module).graph_module.graph

    assert any(
        node.op == "call_function"
        and node.target == exir_ops.backend.tosa.RESCALE.default
        and node.kwargs.get("input_unsigned")
        for node in graph.nodes
    ), "Expected RESCALE with input_unsigned=True in pipeline flow."


def test_quantize_io_tosa_INT_uint8_io_add():
    """Model accepts uint8 inputs/outputs while TOSA uses int8 internally."""
    model = SimpleModel().eval()
    test_data = SimpleModel.test_data["rand_rand"]
    compile_spec = common.get_tosa_compile_spec("TOSA-1.0+INT")

    tester = ArmTester(model, test_data, compile_spec)
    quantizer = TOSAQuantizer(compile_spec)
    quantizer.set_global(get_symmetric_quantization_config())
    quantizer.set_io(get_uint8_io_quantization_config())
    quant_stage = Quantize(quantizer, quantization_config=quantizer.global_config)

    tester.quantize(quant_stage)
    tester.export()
    tester.to_edge()
    edge = tester.get_artifact()
    edge.transform(
        passes=[
            QuantizeInputs(edge, [0, 1]),
            QuantizeOutputs(edge, [0]),
        ]
    )

    exported_program = edge.exported_program()
    graph = exported_program.graph_module.graph
    placeholders = [node for node in graph.nodes if node.op == "placeholder"]
    assert len(placeholders) == 2
    assert placeholders[0].meta["val"].dtype == torch.uint8
    assert placeholders[1].meta["val"].dtype == torch.uint8
    output_node = graph.output_node()
    output_val = output_node.args[0][0]
    assert output_val.meta["val"].dtype == torch.uint8

    graph_module = exported_program.graph_module
    with TosaLoweringContext(TosaSpecification.create_from_string("TOSA-1.0+INT")):
        rescale_graph = InsertRescalePass()(graph_module).graph_module.graph

    rescale_nodes = [
        node
        for node in rescale_graph.nodes
        if node.op == "call_function"
        and node.target == exir_ops.backend.tosa.RESCALE.default
    ]
    assert rescale_nodes, "Expected RESCALE ops after lowering."
    assert any(
        node.kwargs.get("input_unsigned") for node in rescale_nodes
    ), "Expected input_unsigned on IO rescale."
    assert any(
        node.kwargs.get("output_unsigned") for node in rescale_nodes
    ), "Expected output_unsigned on IO rescale."
    assert all(
        node.args[1] == torch.int8
        for node in rescale_nodes
        if node.kwargs.get("input_unsigned") or node.kwargs.get("output_unsigned")
    ), "Unsigned IO rescales must output int8 internally."


def test_quantize_io_tosa_INT_uint8_numeric():
    """Run TOSA flow with uint8 input and verify numerical output."""
    if not TosaPipelineINT.is_tosa_ref_model_available():
        pytest.skip("TOSA reference model not available.")
    model = SimpleModel().eval()
    calib_input = torch.rand(1, 4)
    calib_other = torch.rand(1, 4)

    pipeline = TosaPipelineINT(
        model,
        (calib_input, calib_other),
        aten_op=[],
        exir_op=[],
        use_to_edge_transform_and_lower=True,
    )
    pipeline.quantizer.set_io(get_uint8_io_quantization_config())

    _run_uint8_io_numeric_pipeline(pipeline, model, calib_input, calib_other)


def test_quantize_io_vgf_INT_uint8_numeric():
    """Run VGF flow with uint8 input and verify numerical output."""

    model = SimpleModel().eval()
    calib_input = torch.rand(1, 4)
    calib_other = torch.rand(1, 4)

    pipeline = VgfPipeline(
        model,
        (calib_input, calib_other),
        aten_op=[],
        exir_op=[],
        run_on_vulkan_runtime=True,
        quantize=True,
        use_to_edge_transform_and_lower=True,
        preserve_io_quantization=True,
    )

    pipeline.quantizer.set_io(get_uint8_io_quantization_config())

    if pipeline.has_stage("check_not.exir_quant_nodes"):
        pipeline.pop_stage("check_not.exir_quant_nodes")
    _run_uint8_io_numeric_pipeline(pipeline, model, calib_input, calib_other)


def test_quantize_io_u55_INT_uint8_numeric():
    """Run Ethos-U55 flow with uint8 input and verify numerical output."""
    model = SimpleModel().eval()
    calib_input = torch.rand(1, 4)
    calib_other = torch.rand(1, 4)

    if not (
        common.corstone300_installed()
        and common.arm_executor_runner_exists("corstone-300")
    ):
        pytest.xfail("Did not find Corstone-300 FVP or executor_runner on path")

    pipeline = EthosU55PipelineINT(
        model,
        (calib_input, calib_other),
        aten_ops=[],
        exir_ops=[],
        run_on_fvp=True,
        use_to_edge_transform_and_lower=True,
    )
    pipeline.quantizer.set_io(get_uint8_io_quantization_config())

    _run_uint8_io_numeric_pipeline(pipeline, model, calib_input, calib_other)


def _run_uint8_io_numeric_pipeline(  # noqa: C901
    pipeline, model, calib_input, calib_other
) -> None:
    qparams = {}

    def _apply_uint8_io(ep):
        in0_scale, in0_zp, in0_qmin, in0_qmax, in0_dtype = quantize_input(ep, 0)
        in1_scale, in1_zp, in1_qmin, in1_qmax, in1_dtype = quantize_input(ep, 1)
        out_scale, out_zp, out_qmin, out_qmax, out_dtype = quantize_output(ep, 0)
        qparams.update(
            {
                "in0_scale": in0_scale,
                "in0_zp": in0_zp,
                "in0_qmin": in0_qmin,
                "in0_qmax": in0_qmax,
                "in0_dtype": in0_dtype,
                "in1_scale": in1_scale,
                "in1_zp": in1_zp,
                "in1_qmin": in1_qmin,
                "in1_qmax": in1_qmax,
                "in1_dtype": in1_dtype,
                "out_scale": out_scale,
                "out_zp": out_zp,
                "out_qmin": out_qmin,
                "out_qmax": out_qmax,
                "out_dtype": out_dtype,
            }
        )
        return ep

    class _Uint8ReferenceStage:
        def __init__(self, reference_model):
            self.reference_model = reference_model.eval()

        def stage_type(self):
            return StageType.RUN_PASSES

        @property
        def artifact(self):
            return self.reference_model

        @property
        def graph_module(self):
            return None

        def run_artifact(self, inputs):
            def _quantize(tensor, scale, zp, qmin, qmax, dtype):
                return torch.ops.quantized_decomposed.quantize_per_tensor(
                    tensor, scale, zp, qmin, qmax, dtype
                )

            def _dequantize(tensor, scale, zp, qmin, qmax, dtype):
                return torch.ops.quantized_decomposed.dequantize_per_tensor(
                    tensor, scale, zp, qmin, qmax, dtype
                )

            float_x = _dequantize(
                inputs[0],
                qparams["in0_scale"],
                qparams["in0_zp"],
                qparams["in0_qmin"],
                qparams["in0_qmax"],
                qparams["in0_dtype"],
            )
            float_y = _dequantize(
                inputs[1],
                qparams["in1_scale"],
                qparams["in1_zp"],
                qparams["in1_qmin"],
                qparams["in1_qmax"],
                qparams["in1_dtype"],
            )
            float_out = self.reference_model(float_x, float_y)
            ref_u8 = _quantize(
                float_out,
                qparams["out_scale"],
                qparams["out_zp"],
                qparams["out_qmin"],
                qparams["out_qmax"],
                qparams["out_dtype"],
            )
            # Match TOSA's signless int8 representation of unsigned outputs.
            return ref_u8

    # Insert quantization of inputs/outputs after lowering so we can run uint8 IO.
    pipeline.add_stage_after(
        "to_edge_transform_and_lower",
        pipeline.tester.run_passes,
        RunPasses(pass_functions=[_apply_uint8_io]),
        suffix="uint8_io",
    )
    pipeline.add_stage_after(
        "to_executorch",
        lambda: setattr(
            pipeline.tester,
            "stages",
            {
                **pipeline.tester.stages,
                StageType.RUN_PASSES: _Uint8ReferenceStage(model),
            },
        ),
        suffix="uint8_ref",
    )

    # Run the pipeline to get the quantization parameters without the standard comparison step
    if pipeline.has_stage("run_method_and_compare_outputs"):
        pipeline.pop_stage("run_method_and_compare_outputs")
    pipeline.run()

    assert qparams["in0_dtype"] == torch.uint8
    assert qparams["in1_dtype"] == torch.uint8
    assert qparams["out_dtype"] == torch.uint8

    # Calculate the calib inputs and outputs uint8 values given the
    # calibrated quantization parameters, so we can run the reference with the same quantized inputs.
    input_tensor = torch.ops.quantized_decomposed.quantize_per_tensor(
        calib_input,
        qparams["in0_scale"],
        qparams["in0_zp"],
        qparams["in0_qmin"],
        qparams["in0_qmax"],
        qparams["in0_dtype"],
    )
    other_input = torch.ops.quantized_decomposed.quantize_per_tensor(
        calib_other,
        qparams["in1_scale"],
        qparams["in1_zp"],
        qparams["in1_qmin"],
        qparams["in1_qmax"],
        qparams["in1_dtype"],
    )

    # Compare against a reference that dequantizes uint8 inputs, runs the float model,
    # and requantizes to match TOSA's signless int8 representation.
    def uint8_compare_callback(reference, output, _qparams):
        # Map signless int8 to uint8
        output_u8 = output.to(torch.uint8)
        reference_u8 = reference.to(torch.uint8)
        diff = (output_u8.to(torch.int16) - reference_u8.to(torch.int16)).abs()
        if diff.max().item() > 1:
            raise AssertionError(
                "Output mismatch beyond 1 LSB after uint8 IO flow. "
                f"max abs diff={diff.max().item()}"
            )

    compare_stage = (
        StageType.SERIALIZE
        if pipeline.has_stage("serialize")
        else StageType.TO_EXECUTORCH
    )
    pipeline.tester.run_method_and_compare_outputs(
        stage=compare_stage,
        inputs=(input_tensor, other_input),
        qtol=1,
        reference_stage_type=StageType.RUN_PASSES,
        compare_callback=lambda ref, test, qparmams: uint8_compare_callback(
            ref, test, qparams
        ),
    )
