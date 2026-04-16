#!/usr/bin/env python3

import argparse
import json
from pathlib import Path

import torch
from executorch.examples.models.voxtral_tts.model import load_model
from executorch.extension.pybindings.portable_lib import _load_for_executorch


def load_codes_from_trace(trace_path: Path) -> torch.Tensor:
    payload = json.loads(trace_path.read_text())
    frames = payload.get("frames", [])
    if not frames:
        raise ValueError(f"No frames found in trace: {trace_path}")
    return torch.tensor(
        [[frame["full_codes"] for frame in frames]], dtype=torch.long
    ).transpose(1, 2).contiguous()


def decode_exported_waveform(
    exported,
    codes: torch.Tensor,
    *,
    valid_samples: int,
    max_codec_frames: int | None,
) -> tuple[torch.Tensor, str]:
    try:
        return exported.forward((codes,))[0], "exact"
    except RuntimeError:
        if max_codec_frames is None or codes.shape[2] >= max_codec_frames:
            raise
        padded_codes = torch.zeros(
            (codes.shape[0], codes.shape[1], max_codec_frames),
            dtype=codes.dtype,
        )
        padded_codes[:, :, : codes.shape[2]] = codes
        padded_waveform = exported.forward((padded_codes,))[0]
        return padded_waveform[..., :valid_samples], "padded"


def decode_reference_waveform(
    codec_decoder,
    codes: torch.Tensor,
    *,
    mode: str,
    valid_samples: int,
    max_codec_frames: int | None,
) -> torch.Tensor:
    decode_codes = codes
    if mode == "padded":
        if max_codec_frames is None:
            raise ValueError("max_codec_frames is required for padded codec validation")
        padded_codes = torch.zeros(
            (codes.shape[0], codes.shape[1], max_codec_frames),
            dtype=codes.dtype,
        )
        padded_codes[:, :, : codes.shape[2]] = codes
        decode_codes = padded_codes
    waveform = codec_decoder(decode_codes).detach()
    return waveform[..., :valid_samples]


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Compare eager codec decode against an exported codec_decoder.pte."
    )
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--codec-pte", required=True)
    parser.add_argument("--trace-json", required=True)
    parser.add_argument("--max-seq-len", type=int, default=512)
    parser.add_argument("--max-codec-frames", type=int, default=None)
    parser.add_argument("--atol", type=float, default=1e-5)
    parser.add_argument("--output-json", default=None)
    args = parser.parse_args()

    codes = load_codes_from_trace(Path(args.trace_json))

    model = load_model(
        args.model_path,
        max_seq_len=args.max_seq_len,
        dtype=torch.float32,
        backend="portable",
    )

    exported = _load_for_executorch(args.codec_pte)
    exported_waveform, export_mode = decode_exported_waveform(
        exported,
        codes,
        valid_samples=int(codes.shape[2] * model.config.downsample_factor),
        max_codec_frames=args.max_codec_frames,
    )
    eager_waveform = decode_reference_waveform(
        model.codec_decoder,
        codes,
        mode=export_mode,
        valid_samples=int(exported_waveform.shape[-1]),
        max_codec_frames=args.max_codec_frames,
    )

    diff = (eager_waveform - exported_waveform).abs()
    max_abs = float(diff.max())
    mean_abs = float(diff.mean())

    result = {
        "frames": int(codes.shape[2]),
        "samples": int(eager_waveform.shape[-1]),
        "max_abs_diff": max_abs,
        "mean_abs_diff": mean_abs,
        "atol": args.atol,
        "export_mode": export_mode,
        "ok": max_abs <= args.atol,
    }
    if args.output_json:
        Path(args.output_json).write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result, indent=2))

    return 0 if result["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
