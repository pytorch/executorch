# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from pathlib import Path
from typing import Collection, Mapping

import onnx
import torch
from onnx import numpy_helper
from torch import nn

from ..model.config import TTSConfig
from ..model.duration_predictor import DurationPredictor
from ..model.text_encoder import TextEncoder
from ..model.vector_estimator import VectorEstimator
from ..model.vocoder import Vocoder


def _convnext_targets(prefix: str, num_layers: int) -> tuple[str, ...]:
    fields = (
        "gamma",
        "dwconv.weight",
        "dwconv.bias",
        "norm.norm.weight",
        "norm.norm.bias",
        "pwconv1.weight",
        "pwconv1.bias",
        "pwconv2.weight",
        "pwconv2.bias",
    )
    return tuple(
        f"{prefix}.convnext.{layer}.{field}"
        for layer in range(num_layers)
        for field in fields
    )


def _attention_encoder_targets(prefix: str, num_layers: int) -> tuple[str, ...]:
    attention_fields = (
        "emb_rel_k",
        "emb_rel_v",
        "conv_q.weight",
        "conv_q.bias",
        "conv_k.weight",
        "conv_k.bias",
        "conv_v.weight",
        "conv_v.bias",
        "conv_o.weight",
        "conv_o.bias",
    )
    targets = [
        f"{prefix}.attn_layers.{layer}.{field}"
        for layer in range(num_layers)
        for field in attention_fields
    ]
    for family in ("norm_layers_1",):
        for layer in range(num_layers):
            targets.extend(
                (
                    f"{prefix}.{family}.{layer}.norm.weight",
                    f"{prefix}.{family}.{layer}.norm.bias",
                )
            )
    for layer in range(num_layers):
        targets.extend(
            (
                f"{prefix}.ffn_layers.{layer}.conv_1.weight",
                f"{prefix}.ffn_layers.{layer}.conv_1.bias",
                f"{prefix}.ffn_layers.{layer}.conv_2.weight",
                f"{prefix}.ffn_layers.{layer}.conv_2.bias",
            )
        )
    for layer in range(num_layers):
        targets.extend(
            (
                f"{prefix}.norm_layers_2.{layer}.norm.weight",
                f"{prefix}.norm_layers_2.{layer}.norm.bias",
            )
        )
    return tuple(targets)


_DURATION_TARGETS = (
    (
        "sentence_encoder.sentence_token",
        "sentence_encoder.text_embedder.char_embedder.weight",
    )
    + _convnext_targets("sentence_encoder.convnext", 6)
    + _attention_encoder_targets("sentence_encoder.attn_encoder", 2)
    + (
        "sentence_encoder.proj_out.net.weight",
        "predictor.layers.0.weight",
        "predictor.layers.0.bias",
        "predictor.layers.1.weight",
        "predictor.layers.1.bias",
        "predictor.activation.weight",
    )
)

DURATION_PREDICTOR_INITIALIZER_MAP = {
    target: f"tts.dp.{target}" for target in _DURATION_TARGETS
}

_TEXT_TARGETS = (
    ("text_encoder.text_embedder.char_embedder.weight",)
    + _convnext_targets("text_encoder.convnext", 6)
    + _attention_encoder_targets("text_encoder.attn_encoder", 4)
    + (
        "style_encoder.style_token_layer.style_key",
        "speech_prompted_text_encoder.attention1.W_query.linear.weight",
        "speech_prompted_text_encoder.attention1.W_query.linear.bias",
        "speech_prompted_text_encoder.attention1.W_key.linear.weight",
        "speech_prompted_text_encoder.attention1.W_key.linear.bias",
        "speech_prompted_text_encoder.attention1.W_value.linear.weight",
        "speech_prompted_text_encoder.attention1.W_value.linear.bias",
        "speech_prompted_text_encoder.attention1.out_fc.linear.weight",
        "speech_prompted_text_encoder.attention1.out_fc.linear.bias",
        "speech_prompted_text_encoder.attention2.W_query.linear.weight",
        "speech_prompted_text_encoder.attention2.W_query.linear.bias",
        "speech_prompted_text_encoder.attention2.W_key.linear.weight",
        "speech_prompted_text_encoder.attention2.W_key.linear.bias",
        "speech_prompted_text_encoder.attention2.W_value.linear.weight",
        "speech_prompted_text_encoder.attention2.W_value.linear.bias",
        "speech_prompted_text_encoder.attention2.out_fc.linear.weight",
        "speech_prompted_text_encoder.attention2.out_fc.linear.bias",
        "speech_prompted_text_encoder.norm.norm.weight",
        "speech_prompted_text_encoder.norm.norm.bias",
    )
)

TEXT_ENCODER_INITIALIZER_MAP = {target: f"tts.ttl.{target}" for target in _TEXT_TARGETS}
for index, projection in enumerate(
    (
        "attention1.W_query",
        "attention1.W_key",
        "attention1.W_value",
        "attention1.out_fc",
        "attention2.W_query",
        "attention2.W_key",
        "attention2.W_value",
        "attention2.out_fc",
    ),
    start=3680,
):
    target = f"speech_prompted_text_encoder.{projection}.linear.weight"
    TEXT_ENCODER_INITIALIZER_MAP[target] = f"onnx::MatMul_{index}"


def _linear_targets(prefix: str) -> tuple[str, ...]:
    return tuple(
        f"{prefix}.{projection}.linear.{field}"
        for projection in ("W_query", "W_key", "W_value", "out_fc")
        for field in ("weight", "bias")
    )


_VECTOR_TARGETS = [
    "uncond_masker.text_special_token",
    "uncond_masker.style_key_special_token",
    "uncond_masker.style_value_special_token",
    "style_key",
    "vector_field.proj_in.net.weight",
    "vector_field.time_encoder.mlp.0.linear.weight",
    "vector_field.time_encoder.mlp.0.linear.bias",
    "vector_field.time_encoder.mlp.2.linear.weight",
    "vector_field.time_encoder.mlp.2.linear.bias",
]
for block in range(4):
    offset = block * 6
    _VECTOR_TARGETS.extend(_convnext_targets(f"vector_field.main_blocks.{offset}", 4))
    _VECTOR_TARGETS.extend(
        (
            f"vector_field.main_blocks.{offset + 1}.linear.linear.weight",
            f"vector_field.main_blocks.{offset + 1}.linear.linear.bias",
        )
    )
    _VECTOR_TARGETS.extend(
        _convnext_targets(f"vector_field.main_blocks.{offset + 2}", 1)
    )
    text_prefix = f"vector_field.main_blocks.{offset + 3}"
    _VECTOR_TARGETS.extend(_linear_targets(f"{text_prefix}.attn"))
    _VECTOR_TARGETS.extend(
        (
            f"{text_prefix}.norm.norm.weight",
            f"{text_prefix}.norm.norm.bias",
        )
    )
    if block == 0:
        _VECTOR_TARGETS.extend(
            (f"{text_prefix}.attn.increments", f"{text_prefix}.attn.theta")
        )
    _VECTOR_TARGETS.extend(
        _convnext_targets(f"vector_field.main_blocks.{offset + 4}", 1)
    )
    style_prefix = f"vector_field.main_blocks.{offset + 5}"
    _VECTOR_TARGETS.extend(_linear_targets(f"{style_prefix}.attention"))
    _VECTOR_TARGETS.extend(
        (
            f"{style_prefix}.norm.norm.weight",
            f"{style_prefix}.norm.norm.bias",
        )
    )
_VECTOR_TARGETS.extend(_convnext_targets("vector_field.last_convnext", 4))
_VECTOR_TARGETS.append("vector_field.proj_out.net.weight")

VECTOR_ESTIMATOR_INITIALIZER_MAP = {
    target: f"vector_estimator.tts.ttl.{target}" for target in _VECTOR_TARGETS
}
VECTOR_ESTIMATOR_INITIALIZER_MAP["style_key"] = "/vector_estimator/Expand_output_0"

_VECTOR_MATMUL_WEIGHTS = {
    1: 3384,
    3: 3390,
    5: 3405,
    7: 3429,
    9: 3435,
    11: 3450,
    13: 3474,
    15: 3480,
    17: 3495,
    19: 3519,
    21: 3525,
    23: 3540,
}
for block_index, initializer_index in _VECTOR_MATMUL_WEIGHTS.items():
    if block_index % 6 == 1:
        target = f"vector_field.main_blocks.{block_index}.linear.linear.weight"
        VECTOR_ESTIMATOR_INITIALIZER_MAP[target] = f"onnx::MatMul_{initializer_index}"
        continue
    attention_name = "attn" if block_index % 6 == 3 else "attention"
    for projection, offset in (
        ("W_query", 0),
        ("W_key", 1),
        ("W_value", 2),
        ("out_fc", 9 if attention_name == "attn" else 3),
    ):
        target = (
            f"vector_field.main_blocks.{block_index}.{attention_name}."
            f"{projection}.linear.weight"
        )
        VECTOR_ESTIMATOR_INITIALIZER_MAP[target] = (
            f"onnx::MatMul_{initializer_index + offset}"
        )

_VECTOR_GENERATED_STATIC = {
    "/Constant_3_output_0",
    "/Constant_4_output_0",
    "/Constant_output_0",
    "/vector_estimator/ConstantOfShape_2_output_0",
    "/vector_estimator/Constant_10_output_0",
    "/vector_estimator/Constant_12_output_0",
    "/vector_estimator/Constant_6_output_0",
    "/vector_estimator/Constant_7_output_0",
    "/vector_estimator/Constant_output_0",
    "/vector_estimator/Mul_output_0",
    "/vector_estimator/vector_field/main_blocks.0/convnext.0/act/Constant_1_output_0",
    "/vector_estimator/vector_field/main_blocks.0/convnext.0/act/Constant_2_output_0",
    "/vector_estimator/vector_field/main_blocks.0/convnext.0/act/Constant_output_0",
    "/vector_estimator/vector_field/main_blocks.0/convnext.0/dwconv/Cast_output_0",
    "/vector_estimator/vector_field/main_blocks.0/convnext.1/dwconv/Cast_output_0",
    "/vector_estimator/vector_field/main_blocks.0/convnext.2/dwconv/Cast_output_0",
    "/vector_estimator/vector_field/main_blocks.0/convnext.3/dwconv/Cast_output_0",
    "/vector_estimator/vector_field/main_blocks.3/attn/Constant_27_output_0",
    "/vector_estimator/vector_field/main_blocks.3/attn/Constant_48_output_0",
    "/vector_estimator/vector_field/main_blocks.3/attn/Constant_51_output_0",
    "/vector_estimator/vector_field/main_blocks.3/attn/Constant_54_output_0",
    "/vector_estimator/vector_field/main_blocks.3/attn/Constant_56_output_0",
    "/vector_estimator/vector_field/main_blocks.3/attn/Mul_8_output_0",
    "/vector_estimator/vector_field/main_blocks.3/attn/Mul_9_output_0",
    "/vector_estimator/vector_field/main_blocks.5/attention/Constant_11_output_0",
    "/vector_estimator/vector_field/main_blocks.5/attention/Constant_6_output_0",
    "/vector_estimator/vector_field/time_encoder/sinusoidal/Constant_2_output_0",
    "/vector_estimator/vector_field/time_encoder/sinusoidal/Constant_3_output_0",
    "onnx::ReduceSum_1413",
    "onnx::Tile_1065",
}
_VECTOR_GENERATED_SPLITS = {
    f"/vector_estimator/vector_field/main_blocks.{block}/attn/" f"{split}/{suffix}"
    for block in (3, 9, 15, 21)
    for split, suffixes in (
        (
            "Split",
            (
                "webgpu_axes",
                "webgpu_ends",
                "webgpu_head_shape",
                "webgpu_starts",
            ),
        ),
        (
            "Split_1",
            (
                "webgpu_axes",
                "webgpu_ends",
                "webgpu_head_shape",
                "webgpu_starts",
            ),
        ),
        (
            "Split_2",
            (
                "webgpu_axes",
                "webgpu_ends",
                "webgpu_head_shape",
                "webgpu_starts",
            ),
        ),
        (
            "Split_3",
            (
                "webgpu_axes",
                "webgpu_ends",
                "webgpu_starts",
                "webgpu_width",
            ),
        ),
    )
    for suffix in suffixes
}
VECTOR_ESTIMATOR_GENERATED_INITIALIZERS = frozenset(
    _VECTOR_GENERATED_STATIC | _VECTOR_GENERATED_SPLITS
)

_VOCODER_TARGETS = [
    "normalizer.scale",
    "latent_mean",
    "latent_std",
    "decoder.embed.net.weight",
    "decoder.embed.net.bias",
]
for layer in range(10):
    _VOCODER_TARGETS.extend(
        (
            f"decoder.convnext.{layer}.gamma",
            f"decoder.convnext.{layer}.dwconv.net.weight",
            f"decoder.convnext.{layer}.dwconv.net.bias",
            f"decoder.convnext.{layer}.norm.norm.weight",
            f"decoder.convnext.{layer}.norm.norm.bias",
            f"decoder.convnext.{layer}.pwconv1.weight",
            f"decoder.convnext.{layer}.pwconv1.bias",
            f"decoder.convnext.{layer}.pwconv2.weight",
            f"decoder.convnext.{layer}.pwconv2.bias",
        )
    )
_VOCODER_TARGETS.extend(
    (
        "decoder.final_norm.norm.weight",
        "decoder.final_norm.norm.bias",
        "decoder.final_norm.norm.running_mean",
        "decoder.final_norm.norm.running_var",
        "decoder.head.layer1.net.weight",
        "decoder.head.layer1.net.bias",
        "decoder.head.act.weight",
        "decoder.head.layer2.weight",
    )
)

VOCODER_INITIALIZER_MAP = {target: f"tts.ae.{target}" for target in _VOCODER_TARGETS}
VOCODER_INITIALIZER_MAP.update(
    {
        "normalizer.scale": "tts.ttl.normalizer.scale",
        "latent_mean": "tts.ae.latent_mean",
        "latent_std": "tts.ae.latent_std",
        "decoder.embed.net.weight": "onnx::Conv_1441",
        "decoder.embed.net.bias": "onnx::Conv_1442",
        "decoder.head.act.weight": "onnx::PRelu_1506",
    }
)
VOCODER_GENERATED_INITIALIZERS: frozenset[str] = frozenset()


def _extract_initializers(model: onnx.ModelProto) -> dict[str, torch.Tensor]:
    initializers: dict[str, torch.Tensor] = {}
    for initializer in model.graph.initializer:
        if initializer.name in initializers:
            raise ValueError(f"duplicate initializer name: {initializer.name}")
        initializers[initializer.name] = torch.from_numpy(
            numpy_helper.to_array(initializer).copy()
        )
    return initializers


def extract_initializers(model_path: str | Path) -> dict[str, torch.Tensor]:
    return _extract_initializers(onnx.load(model_path))


def transform_initializer(
    initializer: torch.Tensor, operator: str, *, trans_b: int = 0
) -> torch.Tensor:
    if operator == "Conv":
        return initializer
    if operator == "MatMul" or (operator == "Gemm" and trans_b == 0):
        if initializer.ndim < 2:
            raise ValueError(f"{operator} weight must have at least two dimensions")
        return initializer.transpose(-1, -2).contiguous()
    if operator == "Gemm" and trans_b == 1:
        return initializer
    raise ValueError(f"unsupported initializer operator: {operator}")


def _initializer_layout(
    model: onnx.ModelProto, initializer_name: str
) -> tuple[str, int] | None:
    layouts: set[tuple[str, int]] = set()
    for node in model.graph.node:
        for input_index, input_name in enumerate(node.input):
            if (
                input_name != initializer_name
                or input_index != 1
                or node.op_type not in ("Conv", "Gemm", "MatMul")
            ):
                continue
            trans_b = 0
            if node.op_type == "Gemm":
                trans_b = next(
                    (
                        attribute.i
                        for attribute in node.attribute
                        if attribute.name == "transB"
                    ),
                    0,
                )
            layouts.add((node.op_type, trans_b))
    if len(layouts) > 1:
        raise ValueError(
            f"initializer has ambiguous operator layouts: {initializer_name}"
        )
    return next(iter(layouts), None)


def load_onnx_initializers(
    module: nn.Module,
    model_path: str | Path,
    name_mapping: Mapping[str, str] | None = None,
    *,
    reject_unused: bool = False,
    allowed_unused: Collection[str] = (),
) -> None:
    model = onnx.load(model_path)
    initializers = _extract_initializers(model)
    model_state = module.state_dict()
    mapping = (
        dict(name_mapping)
        if name_mapping is not None
        else {name: name for name in model_state}
    )

    unknown_targets = sorted(set(mapping) - set(model_state))
    if unknown_targets:
        raise ValueError(f"unknown model weights: {', '.join(unknown_targets)}")

    unmapped_targets = sorted(set(model_state) - set(mapping))
    if unmapped_targets:
        raise ValueError(f"unmapped model weights: {', '.join(unmapped_targets)}")

    source_counts: dict[str, int] = {}
    for initializer_name in mapping.values():
        source_counts[initializer_name] = source_counts.get(initializer_name, 0) + 1
    duplicate_sources = sorted(
        name for name, count in source_counts.items() if count > 1
    )
    if duplicate_sources:
        raise ValueError(
            f"duplicate initializer mapping: {', '.join(duplicate_sources)}"
        )

    missing_sources = sorted(set(mapping.values()) - set(initializers))
    if missing_sources:
        raise ValueError(f"missing initializer: {', '.join(missing_sources)}")

    if reject_unused:
        unknown_allowed_unused = sorted(set(allowed_unused) - set(initializers))
        if unknown_allowed_unused:
            raise ValueError(
                "allowed unused initializer not found: "
                + ", ".join(unknown_allowed_unused)
            )
        unused_sources = sorted(set(initializers) - set(mapping.values()))
        unexpected_unused = sorted(set(unused_sources) - set(allowed_unused))
        if unexpected_unused:
            raise ValueError(f"unused initializer: {', '.join(unexpected_unused)}")

    loaded_state: dict[str, torch.Tensor] = {}
    for target_name, initializer_name in mapping.items():
        value = initializers[initializer_name]
        layout = _initializer_layout(model, initializer_name)
        if layout is not None:
            operator, trans_b = layout
            value = transform_initializer(value, operator, trans_b=trans_b)
        expected = model_state[target_name]
        if value.shape != expected.shape:
            raise ValueError(
                f"shape mismatch for {target_name}: expected {tuple(expected.shape)}, "
                f"received {tuple(value.shape)} from {initializer_name}"
            )
        loaded_state[target_name] = value.to(dtype=expected.dtype)

    module.load_state_dict(loaded_state, strict=True)


def load_duration_predictor(
    model_path: str | Path, config: TTSConfig
) -> DurationPredictor:
    model = DurationPredictor(config)
    load_onnx_initializers(
        model,
        model_path,
        DURATION_PREDICTOR_INITIALIZER_MAP,
        reject_unused=True,
    )
    return model


def load_text_encoder(model_path: str | Path, config: TTSConfig) -> TextEncoder:
    model = TextEncoder(config)
    load_onnx_initializers(
        model,
        model_path,
        TEXT_ENCODER_INITIALIZER_MAP,
        reject_unused=True,
    )
    return model


def load_vector_estimator(model_path: str | Path, config: TTSConfig) -> VectorEstimator:
    model = VectorEstimator(config)
    load_onnx_initializers(
        model,
        model_path,
        VECTOR_ESTIMATOR_INITIALIZER_MAP,
        reject_unused=True,
        allowed_unused=VECTOR_ESTIMATOR_GENERATED_INITIALIZERS,
    )
    return model


def load_vocoder(model_path: str | Path, config: TTSConfig) -> Vocoder:
    model = Vocoder(config)
    load_onnx_initializers(
        model,
        model_path,
        VOCODER_INITIALIZER_MAP,
        reject_unused=True,
        allowed_unused=VOCODER_GENERATED_INITIALIZERS,
    )
    return model
