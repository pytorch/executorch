# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Exports Reverb-TTS to two ExecuTorch programs.

`reverb.pte` holds the backbone as three methods:
  prefill(ids [L] i64, reference codes [8, R] i64) -> cache [2, layers, kv, L + R, D]
  step(tokens [8, T] i64, cache) -> confidence [2, T] f32, prediction [2, T] i64,
                                    held [2, T] f32
  polish(tokens [8, T] i64, reference frames [1] i64, cache) -> codebooks 2-7 [6, T] i64

`codec.pte` holds the Higgs audio codec as two methods:
  decode(codes [8, F] i64) -> waveform [F * 960] f32
  encode(waveform [N] f32) -> codes [8, N / 960] i64

The sampling between passes (Gumbel noise, ranking, remasking) runs in the
native runner, so one program serves any number of steps and any seed.
"""

import argparse
from dataclasses import dataclass
from pathlib import Path

import torch
from torch import nn
from torch.export import Dim, export, ExportedProgram

from ..loaders.checkpoint import load_backbone, load_codec, VARIANTS
from ..model.backbone import ReverbBackbone
from ..model.codec import CodecDecoder, CodecEncoder
from ..model.config import CODEBOOKS, MASK

BACKENDS = ("mlx", "xnnpack", "portable")
DTYPES = {"fp32": torch.float32, "fp16": torch.float16}
DEFAULT_DTYPE = {"mlx": "fp16", "xnnpack": "fp32", "portable": "fp32"}
QLINEAR = {
    "mlx": ("4w", "8w"),
    "xnnpack": ("8da4w", "8da8w"),
    "portable": (),
}
POLISH_BIAS = 1.0  # attention bias towards the reference frames in the polish passes
POLISH_GUIDANCE = 0.5  # weight of (second polish pass - first) on codebooks 4-7
FRAME_RATE = 25
SAMPLE_RATE = 24000
HOP_LENGTH = 960


@dataclass(frozen=True)
class ExportBounds:
    prompt_tokens: int = 512
    reference_frames: int = 512
    target_frames: int = 1024

    def __post_init__(self) -> None:
        for name, value in vars(self).items():
            if value < 2:
                raise ValueError(f"maximum {name} must be at least 2")


class Prefill(nn.Module):
    def __init__(self, backbone: ReverbBackbone) -> None:
        super().__init__()
        self.backbone = backbone

    def forward(self, ids: torch.Tensor, codes: torch.Tensor) -> torch.Tensor:
        return self.backbone.prefill(ids, codes)


class Step(nn.Module):
    def __init__(self, backbone: ReverbBackbone) -> None:
        super().__init__()
        self.backbone = backbone

    def forward(self, tokens: torch.Tensor, cache: torch.Tensor):
        return self.backbone.step(tokens, cache)


class Polish(nn.Module):
    def __init__(self, backbone: ReverbBackbone) -> None:
        super().__init__()
        self.backbone = backbone

    def forward(
        self, tokens: torch.Tensor, reference_frames: torch.Tensor, cache: torch.Tensor
    ) -> torch.Tensor:
        return self.backbone.polish(
            tokens, reference_frames, cache, POLISH_BIAS, POLISH_GUIDANCE
        )


def export_backbone(
    backbone: ReverbBackbone, bounds: ExportBounds
) -> dict[str, ExportedProgram]:
    config = backbone.config
    dtype = backbone.audio_embeddings.weight.dtype
    prompt = min(24, bounds.prompt_tokens)
    reference = min(40, bounds.reference_frames)
    target = min(50, bounds.target_frames)
    prompt_dim = Dim("prompt_tokens", min=1, max=bounds.prompt_tokens)
    reference_dim = Dim("reference_frames", min=1, max=bounds.reference_frames)
    target_dim = Dim("target_frames", min=1, max=bounds.target_frames)
    prefix_dim = Dim(
        "prefix", min=2, max=bounds.prompt_tokens + bounds.reference_frames
    )

    ids = torch.randint(0, config.vocab_size, (prompt,))
    codes = torch.randint(0, MASK, (CODEBOOKS, reference))
    tokens = torch.randint(0, MASK + 1, (CODEBOOKS, target))
    cache = torch.zeros(
        2,
        config.num_layers,
        config.num_kv_heads,
        prompt + reference,
        config.head_dim,
        dtype=dtype,
    )
    samples = {
        "prefill": (
            Prefill(backbone),
            (ids, codes),
            ({0: prompt_dim}, {1: reference_dim}),
        ),
        "step": (Step(backbone), (tokens, cache), ({1: target_dim}, {3: prefix_dim})),
        "polish": (
            Polish(backbone),
            (tokens, torch.tensor([reference]), cache),
            ({1: target_dim}, None, {3: prefix_dim}),
        ),
    }
    with torch.no_grad():
        return {
            name: export(module, args, dynamic_shapes=shapes, strict=False)
            for name, (module, args, shapes) in samples.items()
        }


def export_codec(
    codec, bounds: ExportBounds, dtype: torch.dtype
) -> dict[str, ExportedProgram]:
    decoder = CodecDecoder(codec).to(dtype).eval()
    encoder = CodecEncoder(codec).to(dtype).eval()
    decode_frames = Dim("decode_frames", min=1, max=bounds.target_frames)
    encode_frames = Dim("encode_frames", min=2, max=bounds.reference_frames)
    decode_example = min(50, bounds.target_frames)
    encode_example = min(50, bounds.reference_frames)
    with torch.no_grad():
        return {
            "decode": export(
                decoder,
                (torch.randint(0, 1024, (CODEBOOKS, decode_example)),),
                dynamic_shapes=({1: decode_frames},),
                strict=False,
            ),
            "encode": export(
                encoder,
                (torch.zeros(HOP_LENGTH * encode_example),),
                dynamic_shapes=({0: HOP_LENGTH * encode_frames},),
                strict=False,
            ),
        }


def _partitioners(backend: str, methods):
    if backend == "mlx":
        from executorch.backends.mlx import MLXPartitioner

        return {name: [MLXPartitioner()] for name in methods}
    if backend == "xnnpack":
        from executorch.backends.xnnpack.partition.xnnpack_partitioner import (
            XnnpackDynamicallyQuantizedPartitioner,
            XnnpackPartitioner,
        )

        return {
            name: [XnnpackDynamicallyQuantizedPartitioner(), XnnpackPartitioner()]
            for name in methods
        }
    return {name: [] for name in methods}


def lower(
    programs: dict[str, ExportedProgram],
    backend: str,
    metadata: dict[str, object],
):
    from executorch.exir import (
        EdgeCompileConfig,
        ExecutorchBackendConfig,
        to_edge_transform_and_lower,
    )

    transform_passes = None
    if backend == "mlx":
        from executorch.backends.mlx.passes import get_default_passes

        transform_passes = get_default_passes()
    edge = to_edge_transform_and_lower(
        programs,
        transform_passes=transform_passes,
        partitioner=_partitioners(backend, programs),
        compile_config=EdgeCompileConfig(
            _check_ir_validity=False,
            _skip_dim_order=True,
        ),
        constant_methods=metadata,
    )
    return edge.to_executorch(
        config=ExecutorchBackendConfig(extract_delegate_segments=True)
    )


def save_pte(program, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as file:
        program.write_to_file(file)
    if program._tensor_data:
        program.write_tensor_data_to_file(str(path.parent))
    print(f"Wrote {path} ({path.stat().st_size / 2**20:.1f} MiB)")
    return path


def backbone_metadata(backbone: ReverbBackbone, bounds: ExportBounds, dtype: str):
    return {
        "get_max_prompt_tokens": bounds.prompt_tokens,
        "get_max_reference_frames": bounds.reference_frames,
        "get_max_target_frames": bounds.target_frames,
        "get_num_codebooks": CODEBOOKS,
        "get_mask_code": MASK,
        "get_frame_rate": FRAME_RATE,
        "get_polish_bias": POLISH_BIAS,
        "get_polish_guidance": POLISH_GUIDANCE,
        "get_cache_dtype": dtype,
    }


def codec_metadata(bounds: ExportBounds):
    return {
        "get_sample_rate": SAMPLE_RATE,
        "get_hop_length": HOP_LENGTH,
        "get_num_codebooks": CODEBOOKS,
        "get_max_decode_frames": bounds.target_frames,
        "get_min_encode_frames": 2,
        "get_max_encode_frames": bounds.reference_frames,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--asset-dir", required=True, help="Reverb-TTS download")
    parser.add_argument("--codec-dir", required=True, help="Higgs codec download")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--backend", choices=BACKENDS, default="xnnpack")
    parser.add_argument("--variant", choices=VARIANTS, default="fp32")
    parser.add_argument("--dtype", choices=DTYPES, help="backbone dtype")
    parser.add_argument("--codec-dtype", choices=DTYPES, default="fp32")
    parser.add_argument("--qlinear", help="backbone linear quantization")
    parser.add_argument("--qlinear-group-size", type=int)
    parser.add_argument("--max-prompt-tokens", type=int, default=512)
    parser.add_argument("--max-reference-frames", type=int, default=512)
    parser.add_argument("--max-target-frames", type=int, default=1024)
    parser.add_argument(
        "--skip", choices=("backbone", "codec"), action="append", default=[]
    )
    args = parser.parse_args()

    if args.qlinear and args.qlinear not in QLINEAR[args.backend]:
        parser.error(
            f"--qlinear for {args.backend} must be one of {QLINEAR[args.backend]}"
        )
    dtype = args.dtype or DEFAULT_DTYPE[args.backend]
    bounds = ExportBounds(
        args.max_prompt_tokens, args.max_reference_frames, args.max_target_frames
    )
    output_dir = Path(args.output_dir)

    if "backbone" not in args.skip:
        backbone = load_backbone(
            args.asset_dir, variant=args.variant, dtype=DTYPES[dtype]
        )
        if args.qlinear:
            from executorch.extension.llm.export.quantize import quantize_model_

            # Only the decoder layers: the embeddings and heads stay in `dtype`.
            quantize_model_(
                backbone.llm.layers,
                qlinear_config=args.qlinear,
                qlinear_group_size=args.qlinear_group_size,
            )
        program = lower(
            export_backbone(backbone, bounds),
            args.backend,
            backbone_metadata(backbone, bounds, dtype),
        )
        save_pte(program, output_dir / "reverb.pte")
        del backbone, program

    if "codec" not in args.skip:
        codec = load_codec(args.codec_dir)
        program = lower(
            export_codec(codec, bounds, DTYPES[args.codec_dtype]),
            args.backend,
            codec_metadata(bounds),
        )
        save_pte(program, output_dir / "codec.pte")


if __name__ == "__main__":
    main()
