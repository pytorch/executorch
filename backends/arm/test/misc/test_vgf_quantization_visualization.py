# Copyright 2026 Arm Limited and/or its affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from executorch.backends.arm.vgf.quantization_assessment import (
    VgfGraphNodeMetadata,
    VgfGraphQuantizationAssessment,
    VgfModuleQuantizationAssessment,
)
from executorch.backends.arm.vgf.quantization_quality import VgfQuantizationMetrics
from executorch.backends.arm.vgf.quantization_visualization import (
    _entry_attributes,
    _namespace_from_fqn,
    build_vgf_quantization_visualization_entries,
)


def _assessment() -> VgfGraphQuantizationAssessment:
    metrics = VgfQuantizationMetrics(
        mse=0.021,
        snr_db=17.8,
        cosine_similarity=0.991,
        max_abs_error=0.3,
        relative_error=0.1,
        reference_min=-1.0,
        reference_max=1.0,
        quantized_min=-0.9,
        quantized_max=0.9,
        saturation_ratio=0.032,
        clipping_ratio=0.01,
    )
    node = VgfGraphNodeMetadata(
        fp32_fx_node="linear_7",
        fp32_target="aten.linear.default",
        fp32_capture_node="linear_7",
        fp32_debug_handles=(),
        quantized_fx_node="linear_7",
        quantized_target="aten.linear.default",
        quantized_capture_node="dequantize_per_tensor_default_7",
        quantized_debug_handles=(),
    )
    module = VgfModuleQuantizationAssessment(
        module_fqn="blocks.7.mlp.fc1",
        metrics=metrics,
        occurrences=1,
        samples=4,
        compared_numel=128,
        nodes=(node,),
        precision="INT8",
    )
    return VgfGraphQuantizationAssessment(
        modules={module.module_fqn: module},
        fp32_only_module_fqns=(),
        quantized_only_module_fqns=(),
        skipped_module_fqns={},
    )


def test_builds_model_explorer_entry_from_module_assessment() -> None:
    entries = build_vgf_quantization_visualization_entries(_assessment())

    assert len(entries) == 1
    entry = entries[0]
    assert entry.module_fqn == "blocks.7.mlp.fc1"
    assert entry.fp32_node_ids == ("linear_7",)
    assert entry.precision == "INT8"
    assert entry.mse == 0.021
    assert entry.snr_db == 17.8
    assert entry.cosine == 0.991
    assert entry.saturation_percent == 3.2


def test_formats_expected_node_and_group_attributes() -> None:
    entry = build_vgf_quantization_visualization_entries(_assessment())[0]
    attrs = _entry_attributes(entry)

    assert attrs["Module FQN"] == "blocks.7.mlp.fc1"
    assert attrs["Precision"] == "INT8"
    assert attrs["MSE"] == "0.021"
    assert attrs["SNR"] == "17.8 dB"
    assert attrs["Cosine"] == "0.991"
    assert attrs["Saturation"] == "3.20%"
    assert _namespace_from_fqn(entry.module_fqn) == "blocks/7/mlp/fc1"
