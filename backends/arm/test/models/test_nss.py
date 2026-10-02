# Copyright 2025-2026 Arm Limited and/or its affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from typing import Tuple

import pytest
import torch
from executorch.backends.arm.quantizer import (
    get_symmetric_quantization_config,
    TOSAQuantizer,
)
from executorch.backends.arm.scripts.neural_graphics_test_data import (
    _NSS_INPUT_CHANNELS,
    iter_nss_test_calibration_samples,
    load_nss_verification_inputs,
)

from executorch.backends.arm.test import common
from executorch.backends.arm.test.models.model_test_utils import (
    download_model_weights,
    PTQ_AND_QAT_DATA,
    REAL_AND_RANDOM_DATA,
    skip_if_frozen_release,
)
from executorch.backends.arm.test.tester.quantize import ArmQuantize
from executorch.backends.arm.test.tester.test_pipeline import (
    EthosU55PipelineINT,
    EthosU85PipelineINT,
    TosaPipelineFP,
    TosaPipelineINT,
    VgfPipeline,
)
from executorch.backends.arm.tosa import TosaSpecification
from executorch.backends.transforms.duplicate_dynamic_quant_chain import (
    DuplicateDynamicQuantChainPass,
)

from ng_model_gym.usecases.nss.model.model_blocks_v1 import (  # type: ignore[import-not-found,import-untyped]
    AutoEncoderV1,
)
from torch.export import Dim
from torchao.quantization.pt2e import (
    allow_exported_model_train_eval,
    FixedQParamsFakeQuantize,
    FixedQParamsObserver,
    move_exported_model_to_eval,
)
from torchao.quantization.pt2e.quantize_pt2e import convert_pt2e, prepare_qat_pt2e

input_t = Tuple[torch.Tensor]  # Input x

pytestmark = skip_if_frozen_release("NSS")

_NSS_HEIGHT = 8 * Dim("_nss_height", min=16, max=68)
_NSS_WIDTH = 8 * Dim("_nss_width", min=16, max=120)
_NSS_DYNAMIC_SHAPES = ({2: _NSS_HEIGHT, 3: _NSS_WIDTH},)


class NSS(torch.nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.auto_encoder = AutoEncoderV1()


class _Fp64ConvReference(torch.fx.Interpreter):
    def call_function(self, target, args, kwargs):
        if target == torch.ops.aten.conv2d.default:
            x, weight, bias, *options = args
            return target(
                x.double(),
                weight.double(),
                bias.double() if bias is not None else None,
                *options,
                **kwargs,
            ).to(x.dtype)
        return super().call_function(target, args, kwargs)


class _NssFp64ReferenceQuantize(ArmQuantize):
    # TODO(MLETORCH-2609): FP32 bias accumulation changes quantization decisions
    # across hosts. Use FP64 only for the quantized reference's convolutions.
    def run_artifact(self, inputs):
        conv_count = sum(
            node.op == "call_function" and node.target == torch.ops.aten.conv2d.default
            for node in self.artifact.graph.nodes
        )
        assert conv_count == 14, f"Expected 14 NSS conv2d nodes, found {conv_count}"
        return _Fp64ConvReference(self.artifact).run(*inputs)


def nss() -> AutoEncoderV1:
    """Get an instance of NSS with weights loaded."""

    weights = download_model_weights(
        repo_id="Arm/neural-super-sampling",
        filename="nss_v1_0_1_high_fp32.pt",
        revision="main",
    )

    checkpoint = torch.load(
        weights, map_location=torch.device("cpu"), weights_only=True
    )
    state_dict = {
        f"auto_encoder.{key.removeprefix('autoencoder.')}": value
        for key, value in checkpoint["model_state_dict"].items()
    }

    nss_model = NSS()
    nss_model.load_state_dict(state_dict, strict=True)
    return nss_model.auto_encoder


def prequantized_nss(inputs: input_t, dynamic_shapes=None) -> torch.fx.GraphModule:
    weights = download_model_weights(
        repo_id="Arm/neural-super-sampling",
        filename="nss_v1_0_1_high_int8.pt",
        revision="main",
    )
    checkpoint = torch.load(
        weights, map_location=torch.device("cpu"), weights_only=True
    )["model_state_dict"]
    prefix = "autoencoder."
    assert all(key.startswith(prefix) for key in checkpoint)
    state_dict = {key.removeprefix(prefix): value for key, value in checkpoint.items()}

    exported = torch.export.export(
        nss().eval(), inputs, dynamic_shapes=dynamic_shapes, strict=True
    ).module()
    quantizer = TOSAQuantizer(TosaSpecification.create_from_string("TOSA-1.0+INT"))
    quantizer.set_global(
        get_symmetric_quantization_config(is_per_channel=True, is_qat=True)
    )
    prepared = prepare_qat_pt2e(exported, quantizer)

    fixed_observers = {
        key.removesuffix(".activation_post_process.scale")
        for key in state_dict
        if key.endswith(".activation_post_process.scale")
    }
    for name in fixed_observers:
        scale = state_dict[f"{name}.activation_post_process.scale"].item()
        zero_point = state_dict[f"{name}.activation_post_process.zero_point"].item()
        observer = FixedQParamsObserver.with_args(
            scale=scale,
            zero_point=zero_point,
            dtype=torch.int8,
            qscheme=torch.per_tensor_affine,
            quant_min=-127,
            quant_max=127,
        )
        prepared.set_submodule(name, FixedQParamsFakeQuantize(observer=observer))

    parameter_keys = list(dict(prepared.named_parameters()))
    lifted_parameter_keys = [
        key for key in state_dict if key.startswith("_param_constant")
    ]
    assert len(parameter_keys) == len(lifted_parameter_keys)
    for lifted_key, parameter_key in zip(
        lifted_parameter_keys, parameter_keys, strict=True
    ):
        state_dict[parameter_key] = state_dict.pop(lifted_key)

    prepared.load_state_dict(state_dict, strict=True)
    move_exported_model_to_eval(prepared)
    converted = convert_pt2e(prepared)
    DuplicateDynamicQuantChainPass()(converted)
    allow_exported_model_train_eval(converted)
    return converted


def example_inputs():
    return load_nss_verification_inputs()


def random_inputs():
    x = torch.rand((1, _NSS_INPUT_CHANNELS, 544, 960))
    return (x.to(memory_format=torch.channels_last),)


input_test_data = REAL_AND_RANDOM_DATA
is_qat_test_data = PTQ_AND_QAT_DATA


def _set_nss_calibration_samples(pipeline):
    return pipeline.set_quantization_calibration(
        iter_nss_test_calibration_samples(),
        dynamic_shapes=_NSS_DYNAMIC_SHAPES,
    )


@common.parametrize("use_real_data", input_test_data)
def test_nss_tosa_FP(use_real_data):
    pipeline = TosaPipelineFP[input_t](
        nss().eval(),
        example_inputs() if use_real_data else random_inputs(),
        aten_op=[],
        exir_op=[],
        use_to_edge_transform_and_lower=True,
    )
    if use_real_data:
        pipeline.add_stage_after("export", pipeline.tester.dump_operator_distribution)
    pipeline.run()


@common.parametrize("is_qat", is_qat_test_data)
@common.parametrize("use_real_data", input_test_data)
def test_nss_tosa_INT(use_real_data, is_qat):
    if is_qat:
        pipeline_kwargs = {
            # Frobenius norm & cosine theshold check disabled for QAT as smoke test has innacurate results and only checks flow functionality.
            "frobenius_threshold": None,
            "cosine_threshold": None,
            "qtol": 12 if use_real_data else 8,
        }
    else:
        pipeline_kwargs = (
            {"frobenius_threshold": 0.32, "qtol": 12} if use_real_data else {"qtol": 7}
        )
    pipeline = TosaPipelineINT[input_t](
        nss().eval(),
        example_inputs() if use_real_data else random_inputs(),
        aten_op=[],
        exir_op=[],
        use_to_edge_transform_and_lower=True,
        is_qat=is_qat,
        **pipeline_kwargs,
    )
    if use_real_data:
        _set_nss_calibration_samples(pipeline)
    elif not is_qat:
        quantize_stage = pipeline._stages[pipeline.find_pos("quantize")].args[0]
        pipeline.change_args(
            "quantize",
            _NssFp64ReferenceQuantize(
                quantizer=quantize_stage.quantizer,
                quantization_config=quantize_stage.quantization_config,
                calibrate=quantize_stage.calibrate,
                calibration_samples=quantize_stage.calibration_samples,
                is_qat=quantize_stage.is_qat,
                set_global=False,
                fold_quantize=quantize_stage.fold_quantize,
                dynamic_shapes=quantize_stage.dynamic_shapes,
            ),
        )
    pipeline.run()


@common.parametrize("use_real_data", input_test_data)
def test_nss_prequantized_tosa_INT(use_real_data):
    inputs = example_inputs() if use_real_data else random_inputs()
    pipeline = TosaPipelineINT[input_t](
        prequantized_nss(inputs),
        inputs,
        aten_op=[],
        exir_op=[],
        use_to_edge_transform_and_lower=True,
        qtol=12,
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


@pytest.mark.skip(reason="No support for aten_upsample_nearest2d_vec on U55")
@common.XfailIfNoCorstone300
def test_nss_u55_INT():
    pipeline = EthosU55PipelineINT[input_t](
        nss().eval(),
        example_inputs(),
        aten_ops=[],
        exir_ops=[],
        run_on_fvp=True,
        use_to_edge_transform_and_lower=True,
    )
    _set_nss_calibration_samples(pipeline)
    pipeline.run()


@pytest.mark.skip(
    reason="Fails at input memory allocation for input shape: [1, 12, 544, 960]"
)
@common.XfailIfNoCorstone320
def test_nss_u85_INT():
    pipeline = EthosU85PipelineINT[input_t](
        nss().eval(),
        example_inputs(),
        aten_ops=[],
        exir_ops=[],
        run_on_fvp=True,
        use_to_edge_transform_and_lower=True,
    )
    _set_nss_calibration_samples(pipeline)
    pipeline.run()


@common.SkipIfNoModelConverter
@common.parametrize("use_real_data", input_test_data)
def test_nss_vgf_FP(use_real_data):
    pipeline = VgfPipeline[input_t](
        nss().eval(),
        example_inputs() if use_real_data else random_inputs(),
        aten_op=[],
        exir_op=[],
        use_to_edge_transform_and_lower=True,
        run_on_vulkan_runtime=True,
        quantize=False,
        # Override tosa version to test FP-only path
        tosa_version="TOSA-1.0+FP",
    )
    pipeline.run()


@common.SkipIfNoModelConverter
@common.parametrize("is_qat", is_qat_test_data)
@common.parametrize("use_real_data", input_test_data)
def test_nss_vgf_INT(use_real_data, is_qat):
    pipeline = VgfPipeline[input_t](
        nss().eval(),
        example_inputs() if use_real_data else random_inputs(),
        aten_op=[],
        exir_op=[],
        symmetric_io_quantization=True,
        use_to_edge_transform_and_lower=True,
        run_on_vulkan_runtime=True,
        quantize=True,
        is_qat=is_qat,
        # Override tosa version to test INT-only path
        tosa_version="TOSA-1.0+INT",
        qtol=12 if use_real_data else (8 if is_qat else 7),
    )
    if use_real_data:
        _set_nss_calibration_samples(pipeline)
    pipeline.run()


@common.SkipIfNoModelConverter
@common.parametrize("use_real_data", input_test_data)
def test_nss_prequantized_vgf_INT(use_real_data):
    inputs = example_inputs() if use_real_data else random_inputs()
    pipeline = VgfPipeline[input_t](
        prequantized_nss(inputs),
        inputs,
        aten_op=[],
        exir_op=[],
        use_to_edge_transform_and_lower=True,
        run_on_vulkan_runtime=True,
        quantize=True,
        tosa_version="TOSA-1.0+INT",
        qtol=12,
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
def test_nss_tosa_FP_dynamic_shapes(use_real_data):
    pipeline = TosaPipelineFP[input_t](
        nss().eval(),
        example_inputs() if use_real_data else random_inputs(),
        aten_op=[],
        exir_op=[],
        use_to_edge_transform_and_lower=True,
        dynamic_shapes=_NSS_DYNAMIC_SHAPES,
        run_on_tosa_ref_model=False,
    )
    if use_real_data:
        pipeline.add_stage_after("export", pipeline.tester.dump_operator_distribution)
    pipeline.run()


@common.parametrize("is_qat", is_qat_test_data)
@common.parametrize("use_real_data", input_test_data)
def test_nss_tosa_INT_dynamic_shapes(use_real_data, is_qat):
    if is_qat:
        pipeline_kwargs = {
            # Frobenius norm & cosine theshold check disabled for QAT as smoke test has innacurate results and only checks flow functionality.
            "frobenius_threshold": None,
            "cosine_threshold": None,
            "qtol": 12 if use_real_data else 8,
        }
    else:
        pipeline_kwargs = (
            {"frobenius_threshold": 0.32, "qtol": 12} if use_real_data else {"qtol": 7}
        )
    pipeline = TosaPipelineINT[input_t](
        nss().eval(),
        example_inputs() if use_real_data else random_inputs(),
        aten_op=[],
        exir_op=[],
        use_to_edge_transform_and_lower=True,
        is_qat=is_qat,
        dynamic_shapes=_NSS_DYNAMIC_SHAPES,
        run_on_tosa_ref_model=False,
        **pipeline_kwargs,
    )
    if use_real_data:
        _set_nss_calibration_samples(pipeline)
    pipeline.run()


@common.parametrize("use_real_data", input_test_data)
def test_nss_prequantized_tosa_INT_dynamic_shapes(use_real_data):
    inputs = example_inputs() if use_real_data else random_inputs()
    pipeline = TosaPipelineINT[input_t](
        prequantized_nss(inputs, _NSS_DYNAMIC_SHAPES),
        inputs,
        aten_op=[],
        exir_op=[],
        use_to_edge_transform_and_lower=True,
        qtol=12,
        dynamic_shapes=_NSS_DYNAMIC_SHAPES,
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


@common.SkipIfNoModelConverter
@common.parametrize("use_real_data", input_test_data)
def test_nss_vgf_FP_dynamic_shapes(use_real_data):
    pipeline = VgfPipeline[input_t](
        nss().eval(),
        example_inputs() if use_real_data else random_inputs(),
        aten_op=[],
        exir_op=[],
        use_to_edge_transform_and_lower=True,
        run_on_vulkan_runtime=False,
        quantize=False,
        dynamic_shapes=_NSS_DYNAMIC_SHAPES,
        tosa_version="TOSA-1.0+FP",
    )
    pipeline.run()


@common.SkipIfNoModelConverter
@common.parametrize("is_qat", is_qat_test_data)
@common.parametrize("use_real_data", input_test_data)
def test_nss_vgf_INT_dynamic_shapes(use_real_data, is_qat):
    pipeline = VgfPipeline[input_t](
        nss().eval(),
        example_inputs() if use_real_data else random_inputs(),
        aten_op=[],
        exir_op=[],
        symmetric_io_quantization=True,
        use_to_edge_transform_and_lower=True,
        run_on_vulkan_runtime=False,
        quantize=True,
        is_qat=is_qat,
        dynamic_shapes=_NSS_DYNAMIC_SHAPES,
        tosa_version="TOSA-1.0+INT",
        qtol=12,
    )
    if use_real_data:
        _set_nss_calibration_samples(pipeline)
    pipeline.run()


@common.SkipIfNoModelConverter
@common.parametrize("use_real_data", input_test_data)
def test_nss_prequantized_vgf_INT_dynamic_shapes(use_real_data):
    inputs = example_inputs() if use_real_data else random_inputs()
    pipeline = VgfPipeline[input_t](
        prequantized_nss(inputs, _NSS_DYNAMIC_SHAPES),
        inputs,
        aten_op=[],
        exir_op=[],
        use_to_edge_transform_and_lower=True,
        run_on_vulkan_runtime=False,
        quantize=True,
        tosa_version="TOSA-1.0+INT",
        qtol=12 if use_real_data else 8,
        dynamic_shapes=_NSS_DYNAMIC_SHAPES,
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
