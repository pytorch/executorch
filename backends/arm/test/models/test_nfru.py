# Copyright 2026 Arm Limited and/or its affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import os
from typing import Tuple

os.environ.setdefault("HF_HUB_DISABLE_XET", "1")

import torch
from executorch.backends.arm.quantizer import (
    get_symmetric_quantization_config,
    TOSAQuantizer,
)
from executorch.backends.arm.scripts.neural_graphics_test_data import (
    _NFRU_INPUT_CHANNELS,
    iter_nfru_test_calibration_samples,
    load_nfru_verification_inputs,
)

from executorch.backends.arm.test import common
from executorch.backends.arm.test.models.model_test_utils import (
    download_model_weights,
    PTQ_AND_QAT_DATA,
    REAL_AND_RANDOM_DATA,
    skip_if_frozen_release,
)
from executorch.backends.arm.test.tester.test_pipeline import (
    TosaPipelineFP,
    TosaPipelineINT,
    VgfPipeline,
)
from executorch.backends.arm.tosa import TosaSpecification
from executorch.backends.transforms.duplicate_dynamic_quant_chain import (
    DuplicateDynamicQuantChainPass,
)
from ng_model_gym.usecases.nfru.model.nfru_v1_nn import (  # type: ignore[import-not-found,import-untyped]
    NFRUAutoEncoder,
)
from torch.export import Dim
from torchao.quantization.pt2e import (
    allow_exported_model_train_eval,
    move_exported_model_to_eval,
)
from torchao.quantization.pt2e.quantize_pt2e import convert_pt2e, prepare_qat_pt2e

input_t = Tuple[torch.Tensor]  # Input x

pytestmark = skip_if_frozen_release("NFRU")
_NFRU_HEIGHT = 2 * Dim("_nfru_height", min=16, max=135)
_NFRU_WIDTH = 2 * Dim("_nfru_width", min=16, max=240)
_NFRU_DYNAMIC_SHAPES = ({2: _NFRU_HEIGHT, 3: _NFRU_WIDTH},)


def nfru() -> NFRUAutoEncoder:
    """Get an instance of NFRU with FP32 weights loaded."""
    weights = download_model_weights(
        repo_id="Arm/neural-frame-rate-upscaling",
        filename="nfru_v1_fp32.pt",
        revision="main",
    )
    checkpoint = torch.load(
        weights,
        map_location=torch.device("cpu"),
        weights_only=True,
    )["model_state_dict"]
    prefix = "network.auto_encoder."
    assert all(key.startswith(prefix) for key in checkpoint)

    model = NFRUAutoEncoder()
    model.load_state_dict(
        {key.removeprefix(prefix): value for key, value in checkpoint.items()},
        strict=True,
    )
    return model


def prequantized_nfru(inputs: input_t, dynamic_shapes=None) -> torch.fx.GraphModule:
    weights = download_model_weights(
        repo_id="Arm/neural-frame-rate-upscaling",
        filename="nfru_v1_int8.pt",
        revision="main",
    )
    checkpoint = torch.load(
        weights,
        map_location=torch.device("cpu"),
        weights_only=True,
    )["model_state_dict"]
    prefix = "network.auto_encoder."
    assert all(key.startswith(prefix) for key in checkpoint)
    state_dict = {key.removeprefix(prefix): value for key, value in checkpoint.items()}

    exported = torch.export.export(
        nfru().eval(), inputs, dynamic_shapes=dynamic_shapes, strict=True
    ).module()
    quantizer = TOSAQuantizer(TosaSpecification.create_from_string("TOSA-1.0+INT"))
    quantizer.set_global(
        get_symmetric_quantization_config(is_per_channel=False, is_qat=True)
    )
    prepared = prepare_qat_pt2e(exported, quantizer)

    parameter_keys = list(dict(prepared.named_parameters()))
    lifted_parameter_keys = [
        key for key in state_dict if key.startswith("_param_constant")
    ]
    buffer_keys = [
        key
        for key, _ in prepared.named_buffers()
        if not key.startswith("activation_post_process_")
    ]
    lifted_buffer_keys = [
        key for key in state_dict if key.startswith("_tensor_constant")
    ]
    assert len(parameter_keys) == len(lifted_parameter_keys)
    assert len(buffer_keys) == len(lifted_buffer_keys)
    for lifted_key, parameter_key in zip(
        lifted_parameter_keys, parameter_keys, strict=True
    ):
        state_dict[parameter_key] = state_dict.pop(lifted_key)
    for lifted_key, buffer_key in zip(lifted_buffer_keys, buffer_keys, strict=True):
        state_dict[buffer_key] = state_dict.pop(lifted_key)

    prepared.load_state_dict(state_dict, strict=True)
    move_exported_model_to_eval(prepared)
    converted = convert_pt2e(prepared)
    DuplicateDynamicQuantChainPass()(converted)
    allow_exported_model_train_eval(converted)
    return converted


def example_inputs():
    return load_nfru_verification_inputs()


def random_inputs(memory_format: torch.memory_format = torch.channels_last):
    return (
        torch.randn((1, _NFRU_INPUT_CHANNELS, 270, 480)).to(
            memory_format=memory_format
        ),
    )


input_test_data = REAL_AND_RANDOM_DATA
is_qat_test_data = PTQ_AND_QAT_DATA


def _set_nfru_calibration_samples(pipeline):
    return pipeline.set_quantization_calibration(
        iter_nfru_test_calibration_samples(),
        dynamic_shapes=_NFRU_DYNAMIC_SHAPES,
    )


@common.parametrize("use_real_data", input_test_data)
def test_nfru_tosa_FP(use_real_data):
    pipeline = TosaPipelineFP[input_t](
        nfru().eval(),
        example_inputs() if use_real_data else random_inputs(),
        aten_op=[],
        exir_op=[],
    )
    pipeline.run()


@common.parametrize("is_qat", is_qat_test_data)
@common.parametrize("use_real_data", input_test_data)
def test_nfru_tosa_INT(use_real_data, is_qat):
    pipeline_kwargs = (
        {
            "per_channel_quantization": False,
            "use_to_edge_transform_and_lower": True,
            "frobenius_threshold": None,
            "cosine_threshold": None,
        }
        if is_qat
        else {}
    )
    pipeline = TosaPipelineINT[input_t](
        nfru().eval(),
        example_inputs() if use_real_data else random_inputs(),
        aten_op=[],
        exir_op=[],
        atol=0.2,
        qtol=2 if use_real_data else 1,
        is_qat=is_qat,
        **pipeline_kwargs,
    )
    if use_real_data:
        _set_nfru_calibration_samples(pipeline)
    pipeline.run()


@common.parametrize("use_real_data", input_test_data)
def test_nfru_prequantized_tosa_INT(use_real_data):
    inputs = example_inputs() if use_real_data else random_inputs()
    pipeline = TosaPipelineINT[input_t](
        prequantized_nfru(inputs),
        inputs,
        aten_op=[],
        exir_op=[],
        atol=0.2,
        qtol=2 if use_real_data else 1,
    )
    pipeline.pop_stage("quantize")
    pipeline.pop_stage("check.quant_nodes")
    pipeline.add_stage_after(
        "export",
        pipeline.tester.check,
        ["torch.ops.quantized_decomposed.dequantize_per_tensor.default"],
        suffix="prequant_nodes",
    )
    pipeline.run()


@common.parametrize("use_real_data", input_test_data)
def test_nfru_tosa_INT_a16w8(use_real_data):
    pipeline = TosaPipelineINT[input_t](
        nfru().eval(),
        example_inputs() if use_real_data else random_inputs(),
        aten_op=[],
        exir_op=[],
        tosa_extensions=["int16"],
        atol=0.1,
    )
    if use_real_data:
        _set_nfru_calibration_samples(pipeline)
    pipeline.run()


@common.SkipIfNoModelConverter
@common.parametrize("use_real_data", input_test_data)
def test_nfru_vgf_no_quant(use_real_data):
    pipeline = VgfPipeline[input_t](
        nfru().eval(),
        example_inputs() if use_real_data else random_inputs(),
        aten_op=[],
        exir_op=[],
        tosa_version="TOSA-1.0+FP",
        quantize=False,
    )
    pipeline.run()


@common.SkipIfNoModelConverter
@common.parametrize("is_qat", is_qat_test_data)
@common.parametrize("use_real_data", input_test_data)
def test_nfru_vgf_quant(use_real_data, is_qat):
    pipeline_kwargs = (
        {
            "run_on_vulkan_runtime": True,
            "per_channel_quantization": False,
            "use_to_edge_transform_and_lower": True,
        }
        if is_qat
        else {}
    )
    pipeline = VgfPipeline[input_t](
        nfru().eval(),
        example_inputs() if use_real_data else random_inputs(),
        aten_op=[],
        exir_op=[],
        tosa_version="TOSA-1.0+INT",
        symmetric_io_quantization=True,
        quantize=True,
        is_qat=is_qat,
        atol=0.2,
        qtol=(4 if is_qat else 2) if use_real_data else 1,
        **pipeline_kwargs,
    )
    if use_real_data:
        _set_nfru_calibration_samples(pipeline)
    pipeline.run()


@common.SkipIfNoModelConverter
@common.parametrize("use_real_data", input_test_data)
def test_nfru_prequantized_vgf_INT(use_real_data):
    inputs = example_inputs() if use_real_data else random_inputs()
    pipeline = VgfPipeline[input_t](
        prequantized_nfru(inputs),
        inputs,
        aten_op=[],
        exir_op=[],
        tosa_version="TOSA-1.0+INT",
        quantize=True,
        atol=0.2,
        qtol=2 if use_real_data else 1,
    )
    pipeline.pop_stage("quantize")
    pipeline.pop_stage("check.quant_nodes")
    pipeline.add_stage_after(
        "export",
        pipeline.tester.check,
        ["torch.ops.quantized_decomposed.dequantize_per_tensor.default"],
        suffix="prequant_nodes",
    )
    pipeline.run()


@common.SkipIfNoModelConverter
@common.parametrize("use_real_data", input_test_data)
def test_nfru_vgf_quant_a16w8(use_real_data):
    pipeline = VgfPipeline[input_t](
        nfru().eval(),
        example_inputs() if use_real_data else random_inputs(),
        aten_op=[],
        exir_op=[],
        tosa_version="TOSA-1.0+INT",
        tosa_extensions=["int16"],
        symmetric_io_quantization=True,
        atol=0.2,
    )
    if use_real_data:
        _set_nfru_calibration_samples(pipeline)
    pipeline.run()


@common.parametrize("use_real_data", input_test_data)
def test_nfru_tosa_FP_dynamic_shapes(use_real_data):
    pipeline = TosaPipelineFP[input_t](
        nfru().eval(),
        example_inputs() if use_real_data else random_inputs(),
        aten_op=[],
        exir_op=[],
        dynamic_shapes=_NFRU_DYNAMIC_SHAPES,
        run_on_tosa_ref_model=False,
    )
    pipeline.run()


@common.parametrize("is_qat", is_qat_test_data)
@common.parametrize("use_real_data", input_test_data)
def test_nfru_tosa_INT_dynamic_shapes(use_real_data, is_qat):
    pipeline_kwargs = (
        {
            "per_channel_quantization": False,
            "use_to_edge_transform_and_lower": True,
            "frobenius_threshold": None,
            "cosine_threshold": None,
        }
        if is_qat
        else {}
    )
    pipeline = TosaPipelineINT[input_t](
        nfru().eval(),
        example_inputs() if use_real_data else random_inputs(),
        aten_op=[],
        exir_op=[],
        atol=0.2,
        qtol=2 if use_real_data else 1,
        is_qat=is_qat,
        dynamic_shapes=_NFRU_DYNAMIC_SHAPES,
        run_on_tosa_ref_model=False,
        **pipeline_kwargs,
    )
    if use_real_data:
        _set_nfru_calibration_samples(pipeline)
    pipeline.run()


@common.parametrize("use_real_data", input_test_data)
def test_nfru_prequantized_tosa_INT_dynamic_shapes(use_real_data):
    inputs = example_inputs() if use_real_data else random_inputs()
    pipeline = TosaPipelineINT[input_t](
        prequantized_nfru(inputs, _NFRU_DYNAMIC_SHAPES),
        inputs,
        aten_op=[],
        exir_op=[],
        atol=0.2,
        qtol=2 if use_real_data else 1,
        dynamic_shapes=_NFRU_DYNAMIC_SHAPES,
        run_on_tosa_ref_model=False,
    )
    pipeline.pop_stage("quantize")
    pipeline.pop_stage("check.quant_nodes")
    pipeline.add_stage_after(
        "export",
        pipeline.tester.check,
        ["torch.ops.quantized_decomposed.dequantize_per_tensor.default"],
        suffix="prequant_nodes",
    )
    pipeline.run()


@common.parametrize("use_real_data", input_test_data)
def test_nfru_tosa_INT_a16w8_dynamic_shapes(use_real_data):
    pipeline = TosaPipelineINT[input_t](
        nfru().eval(),
        example_inputs() if use_real_data else random_inputs(),
        aten_op=[],
        exir_op=[],
        tosa_extensions=["int16"],
        atol=0.1,
        dynamic_shapes=_NFRU_DYNAMIC_SHAPES,
        run_on_tosa_ref_model=False,
    )
    if use_real_data:
        _set_nfru_calibration_samples(pipeline)
    pipeline.run()


@common.SkipIfNoModelConverter
@common.parametrize("use_real_data", input_test_data)
def test_nfru_vgf_no_quant_dynamic_shapes(use_real_data):
    pipeline = VgfPipeline[input_t](
        nfru().eval(),
        example_inputs() if use_real_data else random_inputs(),
        aten_op=[],
        exir_op=[],
        tosa_version="TOSA-1.0+FP",
        quantize=False,
        dynamic_shapes=_NFRU_DYNAMIC_SHAPES,
        run_on_vulkan_runtime=False,
    )
    pipeline.run()


@common.SkipIfNoModelConverter
@common.parametrize("is_qat", is_qat_test_data)
@common.parametrize("use_real_data", input_test_data)
def test_nfru_vgf_quant_dynamic_shapes(use_real_data, is_qat):
    pipeline_kwargs = (
        {
            "per_channel_quantization": False,
            "use_to_edge_transform_and_lower": True,
        }
        if is_qat
        else {}
    )
    pipeline = VgfPipeline[input_t](
        nfru().eval(),
        example_inputs() if use_real_data else random_inputs(),
        aten_op=[],
        exir_op=[],
        tosa_version="TOSA-1.0+INT",
        symmetric_io_quantization=True,
        quantize=True,
        is_qat=is_qat,
        atol=0.2,
        qtol=2 if use_real_data else 1,
        dynamic_shapes=_NFRU_DYNAMIC_SHAPES,
        run_on_vulkan_runtime=False,
        **pipeline_kwargs,
    )
    if use_real_data:
        _set_nfru_calibration_samples(pipeline)
    pipeline.run()


@common.SkipIfNoModelConverter
@common.parametrize("use_real_data", input_test_data)
def test_nfru_prequantized_vgf_INT_dynamic_shapes(use_real_data):
    inputs = example_inputs() if use_real_data else random_inputs()
    pipeline = VgfPipeline[input_t](
        prequantized_nfru(inputs, _NFRU_DYNAMIC_SHAPES),
        inputs,
        aten_op=[],
        exir_op=[],
        tosa_version="TOSA-1.0+INT",
        quantize=True,
        atol=0.2,
        qtol=2 if use_real_data else 1,
        dynamic_shapes=_NFRU_DYNAMIC_SHAPES,
        run_on_vulkan_runtime=False,
    )
    pipeline.pop_stage("quantize")
    pipeline.pop_stage("check.quant_nodes")
    pipeline.add_stage_after(
        "export",
        pipeline.tester.check,
        ["torch.ops.quantized_decomposed.dequantize_per_tensor.default"],
        suffix="prequant_nodes",
    )
    pipeline.run()


@common.SkipIfNoModelConverter
@common.parametrize("use_real_data", input_test_data)
def test_nfru_vgf_quant_a16w8_dynamic_shapes(use_real_data):
    pipeline = VgfPipeline[input_t](
        nfru().eval(),
        example_inputs() if use_real_data else random_inputs(),
        aten_op=[],
        exir_op=[],
        tosa_version="TOSA-1.0+INT",
        tosa_extensions=["int16"],
        symmetric_io_quantization=True,
        atol=0.2,
        dynamic_shapes=_NFRU_DYNAMIC_SHAPES,
        run_on_vulkan_runtime=False,
    )
    if use_real_data:
        _set_nfru_calibration_samples(pipeline)
    pipeline.run()
