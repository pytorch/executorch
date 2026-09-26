# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Export the Transformers Nemotron 3 Diarization model to ExecuTorch."""

import argparse
from pathlib import Path

import torch

from executorch.exir import (
    EdgeCompileConfig,
    ExecutorchBackendConfig,
    to_edge_transform_and_lower,
)
from executorch.exir.passes import MemoryPlanningPass
from torch import nn
from torch.export import Dim, export
from transformers import (
    Nemotron3DiarizationForAudioFrameClassification,
    Nemotron3DiarizationProcessor,
)

MIN_ENCODER_FRAMES = 2


class Preprocessor(nn.Module):
    """PCM with one pre-emphasis history sample and complete STFT windows."""

    def __init__(self, feature_extractor):
        super().__init__()
        self.feature_extractor = feature_extractor

    def forward(self, audio, lengths):
        # lengths contains the valid PCM endpoint and number of valid mel frames.
        waveform = audio[1:]
        if self.feature_extractor.preemphasis is not None:
            waveform = waveform - self.feature_extractor.preemphasis * audio[:-1]
        positions = torch.arange(1, audio.shape[0], device=audio.device)
        waveform = waveform.masked_fill(positions >= lengths[0], 0)
        features = self.feature_extractor._torch_extract_fbank_features(
            waveform[None], device=audio.device, center=False
        )
        frames = torch.arange(features.shape[1], device=audio.device)
        return features.masked_fill((frames >= lengths[1])[None, :, None], 0)


class PreEncode(nn.Module):
    """Keep the native frontend interface in float32."""

    def __init__(self, model):
        super().__init__()
        self.embedder = model.model.audio_tower.embedder
        self.dtype = model.dtype

    def forward(self, features):
        return self.embedder(features.to(self.dtype)).float()


class Encode(nn.Module):
    """Adapt the stateless Transformers network to the native streaming ABI."""

    def __init__(self, model):
        super().__init__()
        self.model = model.model
        self.classifier = model.classifier
        self.dtype = model.dtype

    def forward(self, embeddings, lengths):
        positions = torch.arange(embeddings.shape[1], device=embeddings.device)
        mask = (positions[None, :] < lengths[:, None])[:, None, None, :]
        encoded = self.model(
            inputs_embeds=embeddings.to(self.dtype), attention_mask=mask
        )
        return self.classifier(encoded.last_hidden_state).sigmoid().float()


def capture_model(model, feature_extractor):
    """Capture backend-independent programs; streaming state stays in the runner."""
    audio = model.config.audio_config
    head = model.config.head_config
    cache = model.config.streaming_config
    if (
        audio.subsampling_factor != 8
        or feature_extractor.feature_size != audio.num_mel_bins
    ):
        raise ValueError("Unsupported feature dimensions for the native runner")
    model.eval().requires_grad_(False)
    model.model.audio_tower.set_attn_implementation("sdpa")
    metadata = {
        "format_version": 2,
        "sample_rate": feature_extractor.sampling_rate,
        "hop_length": feature_extractor.hop_length,
        "n_fft": feature_extractor.n_fft,
        "num_mels": audio.num_mel_bins,
        "d_model": audio.hidden_size,
        "num_speakers": head.num_speakers,
        "subsampling_factor": audio.subsampling_factor,
        "silence_embedding": model.silence_embeds.detach().float(),
        "spkcache_len": cache.speaker_cache_length,
        "silence_frames": cache.speaker_cache_silence_frames_per_speaker,
        "pred_score_threshold": cache.prediction_score_threshold,
        "scores_boost_latest": cache.latest_frames_score_boost,
        "strong_boost_rate": cache.strong_boost_rate,
        "weak_boost_rate": cache.weak_boost_rate,
        "min_pos_scores_rate": cache.min_positive_scores_rate,
        "max_encoder_frames": 1000,
        "max_feature_frames": 4096,
    }
    with torch.no_grad():
        windows = Dim(
            "window_frames",
            min=1,
            max=metadata["max_feature_frames"] // audio.subsampling_factor,
        )
        frames = Dim(
            "encoder_frames",
            min=MIN_ENCODER_FRAMES,
            max=metadata["max_encoder_frames"],
        )
        pcm_context = feature_extractor.n_fft + 1 - feature_extractor.hop_length
        pcm_samples = 104 * feature_extractor.hop_length + pcm_context
        programs = {
            "preprocessor": export(
                Preprocessor(feature_extractor),
                (torch.zeros(pcm_samples), torch.tensor([pcm_samples, 104])),
                dynamic_shapes=(
                    {
                        0: audio.subsampling_factor
                        * feature_extractor.hop_length
                        * windows
                        + pcm_context
                    },
                    {},
                ),
                strict=False,
            ),
            "pre_encode": export(
                PreEncode(model),
                (torch.zeros(1, 104, audio.num_mel_bins),),
                dynamic_shapes=({1: audio.subsampling_factor * windows},),
                strict=False,
            ),
            "encode": export(
                Encode(model),
                (
                    torch.zeros(1, 128, audio.hidden_size),
                    torch.tensor([128], dtype=torch.int64),
                ),
                dynamic_shapes=({1: frames}, {}),
                strict=False,
            ),
        }
    return programs, metadata


def export_model(
    model_id, output_dir, revision=None, dtype=torch.bfloat16, backend="mlx"
):
    from executorch.backends.xnnpack.partition.xnnpack_partitioner import (
        XnnpackPartitioner,
    )

    passes = []
    if backend == "mlx":
        from executorch.backends.mlx.partitioner import MLXPartitioner
        from executorch.backends.mlx.passes import get_default_passes

        partitioners = [MLXPartitioner()]
        passes = get_default_passes()
    elif backend == "xnnpack":
        partitioners = [XnnpackPartitioner(enable_bf16=dtype == torch.bfloat16)]
    else:
        raise ValueError(f"Unknown backend: {backend}")

    model = Nemotron3DiarizationForAudioFrameClassification.from_pretrained(
        model_id, revision=revision, dtype=dtype
    )
    processor = Nemotron3DiarizationProcessor.from_pretrained(
        model_id, revision=revision
    )
    programs, metadata = capture_model(model, processor.feature_extractor)
    edge = to_edge_transform_and_lower(
        programs,
        partitioner={
            "preprocessor": [XnnpackPartitioner()],
            "pre_encode": partitioners,
            "encode": partitioners,
        },
        transform_passes={
            "preprocessor": [],
            "pre_encode": passes,
            "encode": passes,
        },
        constant_methods=metadata,
        compile_config=EdgeCompileConfig(
            _check_ir_validity=False, _skip_dim_order=True
        ),
    )
    program = edge.to_executorch(
        config=ExecutorchBackendConfig(
            extract_delegate_segments=True,
            memory_planning_pass=MemoryPlanningPass(alloc_graph_input=False),
        )
    )
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / "nemotron3_diarization.pte"
    with path.open("wb") as output:
        program.write_to_file(output)
    print(f"Saved {path} ({path.stat().st_size / 2**20:.1f} MiB)")
    return path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--hf-model",
        default="nvidia/Nemotron-3-Diarization",
        help="Hugging Face model ID or local Transformers checkpoint directory",
    )
    parser.add_argument("--revision", help="Hugging Face model revision")
    parser.add_argument("--backend", choices=("mlx", "xnnpack"), default="mlx")
    parser.add_argument("--output-dir", type=Path, default=Path("nemotron_exports"))
    parser.add_argument(
        "--dtype",
        choices=("bf16", "fp32"),
        default="bf16",
        help="Model weights and compute dtype (default: bf16)",
    )
    args = parser.parse_args()
    dtype = {"bf16": torch.bfloat16, "fp32": torch.float32}[args.dtype]
    export_model(args.hf_model, args.output_dir, args.revision, dtype, args.backend)


if __name__ == "__main__":
    main()
