# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from dataclasses import dataclass
from pathlib import Path
from typing import Mapping

import torch
from torch import nn
from torch.export import Dim

from ..loaders import checkpoint_loader
from ..model.config import TTSConfig


METHOD_NAMES = (
    "duration_predictor",
    "text_encoder",
    "vector_estimator",
    "vocoder",
)
MAX_MODEL_POSITIONS = 1000
DEFAULT_FLOW_STEPS = 5


@dataclass(frozen=True)
class ExportBounds:
    text_max: int = 512
    latent_max: int = 512

    def __post_init__(self) -> None:
        for name, value in (
            ("text", self.text_max),
            ("latent", self.latent_max),
        ):
            if value < 2:
                raise ValueError(f"{name} maximum must be at least 2")
            if value > MAX_MODEL_POSITIONS:
                raise ValueError(
                    f"{name} maximum must not exceed {MAX_MODEL_POSITIONS}"
                )


@dataclass(frozen=True)
class MethodContract:
    input_names: tuple[str, ...]
    output_name: str
    output_shape: tuple[str | int, ...]
    output_dtype: torch.dtype


@dataclass(frozen=True)
class SupertonicAssets:
    config: Path
    models: Mapping[str, Path]


def method_contracts(config: TTSConfig) -> dict[str, MethodContract]:
    latent_channels = config.ttl.latent_dim * config.ttl.chunk_compress_factor
    samples_per_latent = config.ttl.chunk_compress_factor * config.ae.base_chunk_size
    return {
        "duration_predictor": MethodContract(
            ("text_ids", "style_dp", "text_mask"),
            "duration",
            ("B",),
            torch.float16,
        ),
        "text_encoder": MethodContract(
            ("text_ids", "style_ttl", "text_mask"),
            "text_emb",
            ("B", 256, "T"),
            torch.float16,
        ),
        "vector_estimator": MethodContract(
            (
                "noisy_latent",
                "text_emb",
                "style_ttl",
                "latent_mask",
                "text_mask",
                "current_step",
                "total_step",
            ),
            "latent",
            ("B", latent_channels, "L"),
            torch.float16,
        ),
        "vocoder": MethodContract(
            ("latent",),
            "waveform",
            ("B", f"L*{samples_per_latent}"),
            torch.float16,
        ),
    }


def dynamic_shapes(bounds: ExportBounds) -> dict[str, tuple[dict | None, ...]]:
    duration_text = Dim("duration_text_length", min=1, max=bounds.text_max)
    encoder_text = Dim("encoder_text_length", min=1, max=bounds.text_max)
    vector_text = Dim("vector_text_length", min=1, max=bounds.text_max)
    vector_latent = Dim("vector_latent_length", min=1, max=bounds.latent_max)
    vocoder_latent = Dim("vocoder_latent_length", min=1, max=bounds.latent_max)
    return {
        "duration_predictor": (
            {1: duration_text},
            None,
            {2: duration_text},
        ),
        "text_encoder": (
            {1: encoder_text},
            None,
            {2: encoder_text},
        ),
        "vector_estimator": (
            {2: vector_latent},
            {2: vector_text},
            None,
            {2: vector_latent},
            {2: vector_text},
            None,
            None,
        ),
        "vocoder": ({2: vocoder_latent},),
    }


def validate_vector_inputs(  # noqa: C901
    inputs: tuple[torch.Tensor, ...],
    config: TTSConfig,
    bounds: ExportBounds,
) -> None:
    """Validate the valid-domain boundary for direct vector PTE execution."""
    if len(inputs) != 7:
        raise ValueError("vector_estimator requires exactly 7 tensors")
    names = (
        "noisy_latent",
        "text_emb",
        "style_ttl",
        "latent_mask",
        "text_mask",
        "current_step",
        "total_step",
    )
    for name, value in zip(names, inputs):
        if not isinstance(value, torch.Tensor):
            raise ValueError(f"{name} must be a tensor")
        if value.dtype != torch.float16:
            raise ValueError(f"{name} must have dtype torch.float16")
        if not value.is_contiguous():
            raise ValueError(f"{name} must be contiguous")

    (
        noisy_latent,
        text_emb,
        style_ttl,
        latent_mask,
        text_mask,
        current_step,
        total_step,
    ) = inputs
    latent_channels = config.ttl.latent_dim * config.ttl.chunk_compress_factor
    if noisy_latent.ndim != 3 or noisy_latent.shape[1] != latent_channels:
        raise ValueError(f"noisy_latent must have shape [1, {latent_channels}, L]")
    if text_emb.ndim != 3 or text_emb.shape[1] != 256:
        raise ValueError("text_emb must have shape [1, 256, T]")
    if style_ttl.ndim != 3 or style_ttl.shape[1:] != (50, 256):
        raise ValueError("style_ttl must have shape [1, 50, 256]")
    if latent_mask.ndim != 3 or latent_mask.shape[1] != 1:
        raise ValueError("latent_mask must have shape [1, 1, L]")
    if text_mask.ndim != 3 or text_mask.shape[1] != 1:
        raise ValueError("text_mask must have shape [1, 1, T]")
    if current_step.ndim != 1 or total_step.ndim != 1:
        raise ValueError("current_step and total_step must have shape [1]")
    if any(value.shape[0] != 1 for value in inputs):
        raise ValueError("vector_estimator batch size must be 1")

    latent_length = noisy_latent.shape[2]
    text_length = text_emb.shape[2]
    if not 1 <= latent_length <= bounds.latent_max:
        raise ValueError(f"latent length must be in [1, {bounds.latent_max}]")
    if not 1 <= text_length <= bounds.text_max:
        raise ValueError(f"text length must be in [1, {bounds.text_max}]")
    if latent_mask.shape[2] != latent_length:
        raise ValueError("noisy_latent and latent_mask latent lengths must match")
    if text_mask.shape[2] != text_length:
        raise ValueError("text_emb and text_mask text lengths must match")

    latent_valid_counts = latent_mask.sum(dim=(1, 2))
    if not torch.all(
        torch.isfinite(latent_valid_counts) & (latent_valid_counts > 0)
    ).item():
        raise ValueError("latent_mask must contain a valid position")
    text_valid_counts = text_mask.sum(dim=(1, 2))
    if not torch.all(
        torch.isfinite(text_valid_counts) & (text_valid_counts > 0)
    ).item():
        raise ValueError("text_mask must contain a valid position")
    if not torch.all(torch.isfinite(current_step)).item():
        raise ValueError("current_step must be finite")
    if not torch.all(torch.isfinite(total_step) & (total_step > 0)).item():
        raise ValueError("total_step must be finite and positive")


def validate_flow_steps(flow_steps: int) -> None:
    if flow_steps != DEFAULT_FLOW_STEPS:
        raise ValueError(
            f"flow steps must be {DEFAULT_FLOW_STEPS} for the native runner"
        )


def text_vocabulary_size(models: Mapping[str, nn.Module]) -> int:
    try:
        duration_size = models[
            "duration_predictor"
        ].sentence_encoder.text_embedder.char_embedder.num_embeddings
        encoder_size = models[
            "text_encoder"
        ].text_encoder.text_embedder.char_embedder.num_embeddings
    except (AttributeError, KeyError) as error:
        raise ValueError("models do not expose the text vocabulary contract") from error
    if duration_size <= 0 or duration_size != encoder_size:
        raise ValueError("duration and text encoders must use the same vocabulary")
    return int(duration_size)


def example_inputs(
    config: TTSConfig,
    bounds: ExportBounds,
    *,
    flow_steps: int = DEFAULT_FLOW_STEPS,
) -> dict[str, tuple[torch.Tensor, ...]]:
    validate_flow_steps(flow_steps)
    generator = torch.Generator().manual_seed(0)
    text_length = bounds.text_max
    latent_length = bounds.latent_max
    latent_channels = config.ttl.latent_dim * config.ttl.chunk_compress_factor

    def random(shape: tuple[int, ...]) -> torch.Tensor:
        return torch.randn(shape, generator=generator, dtype=torch.float16)

    def text_ids() -> torch.Tensor:
        return torch.randint(
            0, 256, (1, text_length), generator=generator, dtype=torch.int64
        )

    def text_mask() -> torch.Tensor:
        return torch.ones((1, 1, text_length), dtype=torch.float16)

    samples = {
        "duration_predictor": (
            text_ids(),
            random((1, 8, 16)),
            text_mask(),
        ),
        "text_encoder": (
            text_ids(),
            random((1, 50, 256)),
            text_mask(),
        ),
        "vector_estimator": (
            random((1, latent_channels, latent_length)),
            random((1, 256, text_length)),
            random((1, 50, 256)),
            torch.ones((1, 1, latent_length), dtype=torch.float16),
            text_mask(),
            torch.tensor([0.0], dtype=torch.float16),
            torch.tensor([float(flow_steps)], dtype=torch.float16),
        ),
        "vocoder": (random((1, latent_channels, latent_length)),),
    }
    validate_vector_inputs(samples["vector_estimator"], config, bounds)
    return samples


def runtime_metadata(
    config: TTSConfig,
    bounds: ExportBounds,
    *,
    text_vocabulary_size: int,
    flow_steps: int = DEFAULT_FLOW_STEPS,
) -> dict[str, object]:
    validate_flow_steps(flow_steps)
    if text_vocabulary_size <= 0:
        raise ValueError("text vocabulary size must be positive")
    return {
        "get_sample_rate": config.ae.sample_rate,
        "get_base_chunk_size": config.ae.base_chunk_size,
        "get_chunk_compress_factor": config.ttl.chunk_compress_factor,
        "get_flow_steps": flow_steps,
        "get_text_vocabulary_size": text_vocabulary_size,
        "get_latent_dim": config.ttl.latent_dim,
        "get_latent_channels": (
            config.ttl.latent_dim * config.ttl.chunk_compress_factor
        ),
        "get_max_text_length": bounds.text_max,
        "get_max_latent_length": bounds.latent_max,
        "get_batch_size": 1,
        "get_activation_dtype": "float16",
        "enable_dynamic_shape": True,
    }


def resolve_assets(asset_dir: str | Path) -> SupertonicAssets:
    asset_dir = Path(asset_dir)
    onnx_dir = asset_dir / "onnx"
    config_path = onnx_dir / "tts.json"
    model_paths = {name: onnx_dir / f"{name}.onnx" for name in METHOD_NAMES}
    required = (config_path, *model_paths.values())
    missing = [path for path in required if not path.is_file()]
    if missing:
        relative = ", ".join(str(path.relative_to(asset_dir)) for path in missing)
        raise FileNotFoundError(f"missing Supertonic assets: {relative}")
    return SupertonicAssets(config=config_path, models=model_paths)


def load_models(
    asset_dir: str | Path,
) -> tuple[TTSConfig, dict[str, nn.Module]]:
    assets = resolve_assets(asset_dir)
    config = TTSConfig.from_json(assets.config)
    models = {
        name: getattr(checkpoint_loader, f"load_{name}")(
            assets.models[name], config
        ).eval()
        for name in METHOD_NAMES
    }
    return config, models


def convert_models_to_fp16(
    models: Mapping[str, nn.Module],
) -> Mapping[str, nn.Module]:
    for model in models.values():
        model.to(dtype=torch.float16)
    return models


def save_pte(et_program, output_path: str | Path) -> Path:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("wb") as output_file:
        et_program.write_to_file(output_file)
    if et_program._tensor_data:
        et_program.write_tensor_data_to_file(str(output_path.parent))
    return output_path
