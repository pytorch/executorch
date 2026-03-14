import argparse
import json
import re
from pathlib import Path
from typing import Dict, Optional, Tuple

import torch


def _load_sharded_safetensors(input_dir: Path) -> Dict[str, torch.Tensor]:
    from safetensors.torch import load_file

    index_path = input_dir / "model.safetensors.index.json"
    if index_path.exists():
        with index_path.open("r", encoding="utf-8") as f:
            index = json.load(f)
        weight_map = index["weight_map"]
        shard_to_names = {}
        for name, shard in weight_map.items():
            shard_to_names.setdefault(shard, []).append(name)
        merged = {}
        for shard, names in shard_to_names.items():
            shard_state = load_file(str(input_dir / shard))
            for name in names:
                merged[name] = shard_state[name]
        return merged

    model_path = input_dir / "model.safetensors"
    if model_path.exists():
        return load_file(str(model_path))

    raise FileNotFoundError(f"Could not find safetensors checkpoint under {input_dir}")


def _extract_prefixed_state_dict(
    state_dict: Dict[str, torch.Tensor], prefix: str
) -> Dict[str, torch.Tensor]:
    out = {}
    for key, value in state_dict.items():
        if key.startswith(prefix):
            out[key[len(prefix) :]] = value
    return out


def _sanitize_model_id(model_id: str) -> str:
    cleaned = re.sub(r"[^a-zA-Z0-9._-]+", "_", model_id.strip())
    return cleaned.strip("_") or "qwen3_tts_model"


def _build_decoder_metadata(
    model_id_or_path: str,
    root_cfg: Dict,
    speech_tokenizer_cfg: Dict,
    decoder_checkpoint_name: str,
) -> Dict:
    decoder_cfg = speech_tokenizer_cfg.get("decoder_config", {})
    return {
        "model_id_or_path": model_id_or_path,
        "tokenizer_type": root_cfg.get("tokenizer_type", "qwen3_tts_tokenizer_v2"),
        "tts_model_type": root_cfg.get("tts_model_type", "base"),
        "decoder_checkpoint": decoder_checkpoint_name,
        "output_sample_rate": int(speech_tokenizer_cfg.get("output_sample_rate", 24000)),
        "decode_upsample_rate": int(speech_tokenizer_cfg.get("decode_upsample_rate", 1920)),
        "num_quantizers": int(decoder_cfg.get("num_quantizers", 16)),
        "codebook_size": int(decoder_cfg.get("codebook_size", 2048)),
        "decoder_config": decoder_cfg,
    }


def _resolve_snapshot_dir(
    input_ref: str, cache_dir: Optional[str]
) -> Tuple[Path, Optional[str]]:
    input_path = Path(input_ref)
    if input_path.exists():
        return input_path.resolve(), None

    from huggingface_hub import snapshot_download

    snapshot_path = snapshot_download(
        repo_id=input_ref,
        cache_dir=cache_dir,
        allow_patterns=[
            "config.json",
            "model.safetensors*",
            "model-*.safetensors*",
            "speech_tokenizer/*",
        ],
    )
    return Path(snapshot_path).resolve(), input_ref


def convert_weights(
    input_ref: str,
    output_dir: Path,
    model_id_or_path: Optional[str],
    save_talker: bool,
    cache_dir: Optional[str],
) -> None:
    input_dir, downloaded_model_id = _resolve_snapshot_dir(
        input_ref=input_ref, cache_dir=cache_dir
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    root_cfg_path = input_dir / "config.json"
    if not root_cfg_path.exists():
        raise FileNotFoundError(f"Missing root config: {root_cfg_path}")
    with root_cfg_path.open("r", encoding="utf-8") as f:
        root_cfg = json.load(f)

    speech_tokenizer_dir = input_dir / "speech_tokenizer"
    speech_tokenizer_cfg_path = speech_tokenizer_dir / "config.json"
    if not speech_tokenizer_cfg_path.exists():
        raise FileNotFoundError(f"Missing speech tokenizer config: {speech_tokenizer_cfg_path}")
    with speech_tokenizer_cfg_path.open("r", encoding="utf-8") as f:
        speech_tokenizer_cfg = json.load(f)

    print("Loading speech tokenizer checkpoint...")
    speech_state = _load_sharded_safetensors(speech_tokenizer_dir)
    decoder_state = _extract_prefixed_state_dict(speech_state, "decoder.")
    if not decoder_state:
        raise RuntimeError(
            "Decoder weights were not found in speech tokenizer checkpoint "
            "(expected keys prefixed by 'decoder.')."
        )
    decoder_ckpt = output_dir / "qwen3_tts_decoder.pth"
    torch.save(decoder_state, decoder_ckpt)
    print(f"Saved decoder checkpoint: {decoder_ckpt}")

    talker_ckpt_name = None
    if save_talker:
        print("Loading root model checkpoint for talker extraction...")
        root_state = _load_sharded_safetensors(input_dir)
        talker_state = _extract_prefixed_state_dict(root_state, "talker.")
        if not talker_state:
            raise RuntimeError(
                "Talker weights were not found in root checkpoint "
                "(expected keys prefixed by 'talker.')."
            )
        talker_ckpt = output_dir / "qwen3_tts_talker.pth"
        torch.save(talker_state, talker_ckpt)
        talker_ckpt_name = talker_ckpt.name
        print(f"Saved talker checkpoint: {talker_ckpt}")

    if model_id_or_path is None:
        if downloaded_model_id is not None:
            model_id_or_path = downloaded_model_id
        else:
            model_id_or_path = _sanitize_model_id(input_dir.name)

    metadata = _build_decoder_metadata(
        model_id_or_path=model_id_or_path,
        root_cfg=root_cfg,
        speech_tokenizer_cfg=speech_tokenizer_cfg,
        decoder_checkpoint_name=decoder_ckpt.name,
    )
    if talker_ckpt_name is not None:
        metadata["talker_checkpoint"] = talker_ckpt_name

    metadata_path = output_dir / "decoder_metadata.json"
    with metadata_path.open("w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2, sort_keys=True)
    print(f"Saved decoder metadata: {metadata_path}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Convert Qwen3-TTS HF checkpoints into export-ready decoder artifacts."
    )
    parser.add_argument(
        "input_ref",
        type=str,
        help=(
            "Either a local HF snapshot path or a Hugging Face model id "
            "(e.g., Qwen/Qwen3-TTS-12Hz-0.6B-Base)."
        ),
    )
    parser.add_argument(
        "output_dir",
        type=Path,
        help="Directory where converted artifacts will be written.",
    )
    parser.add_argument(
        "--model-id-or-path",
        type=str,
        default=None,
        help="Original model id or path to record in metadata.",
    )
    parser.add_argument(
        "--save-talker",
        action="store_true",
        help="Also extract and save talker.* weights from root checkpoint.",
    )
    parser.add_argument(
        "--cache-dir",
        type=str,
        default=None,
        help="Optional huggingface_hub cache directory when input_ref is a model id.",
    )
    args = parser.parse_args()
    convert_weights(
        input_ref=args.input_ref,
        output_dir=args.output_dir,
        model_id_or_path=args.model_id_or_path,
        save_talker=args.save_talker,
        cache_dir=args.cache_dir,
    )


if __name__ == "__main__":
    main()
