# Copyright 2026 NXP
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import executorch.backends.nxp.tests.models as models
import numpy as np
import pytest
import torch
from executorch.backends.nxp.aten_passes.fuse_batch_norm_with_linear_pass import (
    FuseBatchNormWithLinearPass,
)
from executorch.backends.nxp.aten_passes.simulated_linear_bn_fusion_passes import (
    AddSimulatedLinearBatchNormFusionQATPass,
    RemoveSimulatedLinearBatchNormFusionQATPass,
)
from executorch.backends.nxp.backend.edge_program_converter import (
    EdgeProgramToIRConverter,
)
from executorch.backends.nxp.backend.graph_utils import (
    batch_norm_target_ops,
    is_batch_norm,
)
from executorch.backends.nxp.quantizer.neutron_quantizer import NeutronQuantizer
from executorch.backends.nxp.tests.executorch_pipeline import (
    get_random_calibration_inputs,
    neutron_target_spec,
    to_model_input_spec,
    to_quantized_edge_program,
)
from executorch.backends.nxp.tests.executors import (
    convert_run_compare,
    graph_contains_any_of_ops,
    ToChannelFirstPreprocess,
    ToChannelLastPreprocess,
)
from executorch.exir.dialects._ops import ops as exir_ops
from torch.export import export, ExportedProgram
from torchao.quantization.pt2e.prepare import _is_activation_post_process_node
from torchao.quantization.pt2e.quantize_pt2e import convert_pt2e, prepare_qat_pt2e


@pytest.mark.parametrize("input_shape", [(2, 3)])
@pytest.mark.parametrize("linear_bias", [True, False])
def test_add_simulated_linear_bn_fusing(input_shape, linear_bias):
    calibration_inputs = get_random_calibration_inputs(to_model_input_spec(input_shape))
    input_sample = calibration_inputs[0]
    model = models.LinearBNModule(
        in_features=input_shape[-1],
        out_features=5,
        linear_bias=linear_bias,
    )
    model.train()
    raw_output = model(input_sample[0])

    exported_model = export(model, input_sample, strict=True)
    prepared_model = prepare_qat_pt2e(
        exported_model.module(), NeutronQuantizer(neutron_target_spec, is_qat=True)
    )
    prepared_model = AddSimulatedLinearBatchNormFusionQATPass()(
        prepared_model
    ).graph_module

    graph_nodes = list(prepared_model.graph.nodes)
    named_modules = dict(prepared_model.named_modules(remove_duplicate=False))
    fake_quantize_output = prepared_model(input_sample[0])

    expected_number_of_nodes = 23 if linear_bias else 18
    linear_node = next(
        (
            n
            for n in graph_nodes
            if hasattr(n, "target") and n.target == torch.ops.aten.linear.default
        ),
        None,
    )

    assert len(graph_nodes) == expected_number_of_nodes

    # Assert Linear weight being quantized and "normalized"
    assert linear_node is not None
    assert all(
        _is_activation_post_process_node(n, named_modules) for n in linear_node.args
    )
    assert linear_node.args[1].args[0].target == torch.ops.aten.mul.Tensor

    # Assert BatchNorm input being "denormalized"
    assert is_batch_norm(graph_nodes[-3])
    if linear_bias:
        assert graph_nodes[-3].args[0].target == torch.ops.aten.add.Tensor
        add_arg_targets = (
            n.target for n in graph_nodes[-3].args[0].args if hasattr(n, "target")
        )
        assert torch.ops.aten.div.Tensor in add_arg_targets
    else:
        assert graph_nodes[-3].args[0].target == torch.ops.aten.div.Tensor

    assert raw_output.shape == fake_quantize_output.shape


@pytest.mark.parametrize("input_shape", [(2, 3)])
@pytest.mark.parametrize("linear_bias", [True, False])
def test_full_linear_bn_fusing(input_shape, linear_bias):
    # TODO: Add pass for quantizing bias node when Linear has bias=False
    if not linear_bias:
        pytest.skip(
            "Linear with bias=False is not yet supported. "
            "The graph currently produces Linear layer without quantized bias which is incorrect."
        )

    calibration_inputs = get_random_calibration_inputs(to_model_input_spec(input_shape))
    input_sample = calibration_inputs[0]
    model = models.LinearBNModule(
        in_features=input_shape[-1],
        out_features=5,
        linear_bias=linear_bias,
    )
    model.train()
    raw_output = model(input_sample[0])

    exported_model = export(model, input_sample, strict=True)
    prepared_model = prepare_qat_pt2e(
        exported_model.module(), NeutronQuantizer(neutron_target_spec, is_qat=True)
    )

    prepared_model = AddSimulatedLinearBatchNormFusionQATPass()(
        prepared_model
    ).graph_module
    prepared_model(input_sample[0])
    prepared_model = RemoveSimulatedLinearBatchNormFusionQATPass()(
        prepared_model
    ).graph_module
    prepared_model = FuseBatchNormWithLinearPass()(prepared_model).graph_module
    converted_model = convert_pt2e(prepared_model)

    quantized_output = converted_model(input_sample[0])
    graph_nodes = list(converted_model.graph.nodes)
    linear_node = graph_nodes[-4]

    assert len(graph_nodes) == 11

    assert not any(is_batch_norm(node) for node in graph_nodes)

    # Assert linear inputs being quantized
    assert linear_node.target == torch.ops.aten.linear.default
    assert (
        linear_node.args[0].target
        == torch.ops.quantized_decomposed.dequantize_per_tensor.default
    )
    assert (
        linear_node.args[1].target
        == torch.ops.quantized_decomposed.dequantize_per_tensor.default
    )

    # Assert linear outputs being quantized
    assert len(linear_node.users) == 1
    assert (
        list(linear_node.users.keys())[0].target
        == torch.ops.quantized_decomposed.quantize_per_tensor.default
    )

    assert raw_output.shape == quantized_output.shape


@pytest.mark.parametrize("input_shape", [(2, 3)])
@pytest.mark.parametrize("linear_bias", [True, False])
@pytest.mark.parametrize("bn_eps", [1e-5, 1e-6])
def test_input_output_graph_equivalence(input_shape, linear_bias, bn_eps):
    # TODO: Add pass for quantizing bias node when Linear has bias=False
    if not linear_bias:
        pytest.skip(
            "Linear with bias=False is not yet supported. "
            "The graph currently produces Linear layer without quantized bias which is incorrect."
        )

    calibration_inputs = get_random_calibration_inputs(to_model_input_spec(input_shape))
    input_sample = calibration_inputs[0]
    model = models.LinearBNModule(
        in_features=input_shape[-1],
        out_features=5,
        linear_bias=linear_bias,
        bn_eps=bn_eps,
    )
    model.eval()

    original_model = export(model, input_sample, strict=True).module()

    processed_model = export(model, input_sample, strict=True).module()
    processed_model = AddSimulatedLinearBatchNormFusionQATPass()(
        processed_model
    ).graph_module

    assert list(processed_model.graph.nodes)[8].args[1] == bn_eps

    processed_model = RemoveSimulatedLinearBatchNormFusionQATPass()(
        processed_model
    ).graph_module

    assert list(processed_model.graph.nodes)[-2].args[7] == bn_eps
    assert torch.equal(
        original_model(input_sample[0]), processed_model(input_sample[0])
    )
    assert len(original_model.graph.nodes) == len(processed_model.graph.nodes)


@pytest.mark.parametrize("input_shape", [(2, 3)])
@pytest.mark.parametrize("linear_bias", [True, False])
@pytest.mark.parametrize("bn_eps", [1e-5, 1e-6])
def test_linear_bn_full_qat_pipeline_conversion(
    mocker, input_shape, linear_bias, bn_eps
):
    # TODO: Add pass for quantizing bias node when Linear has bias=False
    if not linear_bias:
        pytest.skip(
            "Linear with bias=False is not yet supported. "
            "The graph currently produces Linear layer without quantized bias which is incorrect."
        )

    model = models.LinearBNModule(
        in_features=input_shape[-1],
        out_features=5,
        linear_bias=linear_bias,
        bn_eps=bn_eps,
    )
    model.eval()

    converter_spy = mocker.spy(EdgeProgramToIRConverter, "convert_program")

    # Run conversion
    edge_program = to_quantized_edge_program(
        model, input_shape, use_qat=True, use_neutron_for_format_conversion=False
    ).exported_program()

    # Make sure neither `Linear` nor `BatchNorm` is in the graph.
    assert not graph_contains_any_of_ops(
        graph=edge_program.graph,
        ops=[
            exir_ops.edge.aten.addmm.default,
            exir_ops.edge.aten.linear.default,
        ]
        + batch_norm_target_ops,
    )
    assert any("lowered_module" in node.name for node in edge_program.graph.nodes)

    # Capture generated model
    tflite_flatbuffers_model, _ = converter_spy.spy_return

    # Capture converted program
    exported_program: ExportedProgram = converter_spy.call_args.args[1]

    input_data = (np.random.random(input_shape).astype(np.float32) * 50).astype(np.int8)

    convert_run_compare(
        exported_program,
        tflite_input_preprocess=ToChannelLastPreprocess(),
        tfl_model=tflite_flatbuffers_model,
        tflite_output_preprocess=ToChannelFirstPreprocess(),
        input_data=input_data,
        atol=0.0,
    )
