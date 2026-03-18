"""Convert Qwen3-TTS talker weights from HF format to ExecuTorch/Meta Llama format.

Produces two checkpoint files:
  - talker_main.pth: main talker backbone (Qwen3 format for Llama infra)
  - talker_code_predictor.pth: code predictor backbone (same Qwen3 format)

Also extracts auxiliary weights (text_projection, codec_head, embeddings) into
  - talker_aux.pth: non-transformer weights needed by the C++ runner
"""

import argparse
import json
from pathlib import Path
from typing import Dict

import torch

# Weight key mapping: Meta/Llama format <- HF format
# Same as executorch/examples/models/qwen3/convert_weights.py but adapted
# for the talker checkpoint structure.
_QWEN3_FROM_META = {
    "tok_embeddings.weight": "codec_embedding.weight",
    "norm.weight": "norm.weight",
    "output.weight": "__CODEC_HEAD__",  # handled specially
    "layers.{}.attention.wk.weight": "layers.{}.self_attn.k_proj.weight",
    "layers.{}.attention.k_norm_fn.weight": "layers.{}.self_attn.k_norm.weight",
    "layers.{}.attention.wq.weight": "layers.{}.self_attn.q_proj.weight",
    "layers.{}.attention.q_norm_fn.weight": "layers.{}.self_attn.q_norm.weight",
    "layers.{}.attention.wv.weight": "layers.{}.self_attn.v_proj.weight",
    "layers.{}.attention.wo.weight": "layers.{}.self_attn.o_proj.weight",
    "layers.{}.attention_norm.weight": "layers.{}.input_layernorm.weight",
    "layers.{}.ffn_norm.weight": "layers.{}.post_attention_layernorm.weight",
    # Note: gate_proj and up_proj are swapped (same as Qwen3 text models).
    "layers.{}.feed_forward.w1.weight": "layers.{}.mlp.gate_proj.weight",
    "layers.{}.feed_forward.w2.weight": "layers.{}.mlp.down_proj.weight",
    "layers.{}.feed_forward.w3.weight": "layers.{}.mlp.up_proj.weight",
}


def _convert_backbone(
    hf_state: Dict[str, torch.Tensor],
    prefix: str,
    codec_head_key: str,
) -> Dict[str, torch.Tensor]:
    """Convert a transformer backbone from HF to Meta format."""
    inverted = {v: k for k, v in _QWEN3_FROM_META.items()}
    converted = {}

    for hf_key, tensor in hf_state.items():
        if not hf_key.startswith(prefix):
            continue
        stripped = hf_key[len(prefix):]

        # Try direct match first.
        if stripped in inverted:
            meta_key = inverted[stripped]
            if meta_key == "__CODEC_HEAD__":
                continue  # Handled separately.
            converted[meta_key] = tensor
            continue

        # Try layer-pattern match.
        matched = False
        for meta_pattern, hf_pattern in _QWEN3_FROM_META.items():
            if "{}" not in hf_pattern:
                continue
            hf_parts = hf_pattern.split("{}")
            if stripped.startswith(hf_parts[0]) and stripped.endswith(hf_parts[1]):
                layer_str = stripped[len(hf_parts[0]):-len(hf_parts[1]) if hf_parts[1] else len(stripped)]
                meta_key = meta_pattern.replace("{}", layer_str)
                converted[meta_key] = tensor
                matched = True
                break
        if not matched and stripped not in ("codec_embedding.weight", "text_embedding.weight"):
            print(f"  Skipping unmapped key: {hf_key}")

    # Map codec_head -> output (lm_head equivalent).
    if codec_head_key in hf_state:
        converted["output.weight"] = hf_state[codec_head_key]

    return converted


def _convert_code_predictor(
    hf_state: Dict[str, torch.Tensor],
) -> Dict[str, torch.Tensor]:
    """Convert code predictor backbone to Meta format."""
    return _convert_backbone(hf_state, "code_predictor.model.", "")


def _extract_aux_weights(
    hf_state: Dict[str, torch.Tensor],
) -> Dict[str, torch.Tensor]:
    """Extract non-transformer weights for the C++ runner."""
    aux = {}
    for key in sorted(hf_state.keys()):
        if key.startswith("text_projection."):
            aux[key] = hf_state[key]
        elif key == "codec_head.weight":
            aux[key] = hf_state[key]
        elif key == "model.text_embedding.weight":
            aux[key] = hf_state[key]
        elif key == "model.codec_embedding.weight":
            aux["main_codec_embedding.weight"] = hf_state[key]
        elif key.startswith("code_predictor.model.codec_embedding."):
            # e.g., code_predictor.model.codec_embedding.0.weight -> cp_codec_embedding.0.weight
            suffix = key[len("code_predictor.model."):]
            aux[f"cp_{suffix}"] = hf_state[key]
        elif key.startswith("code_predictor.lm_head."):
            aux[key] = hf_state[key]

    return aux


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Convert Qwen3-TTS talker weights to ExecuTorch Llama format."
    )
    parser.add_argument(
        "--talker-checkpoint",
        type=Path,
        required=True,
        help="Path to qwen3_tts_talker.pth (from convert_weights.py --save-talker).",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="Output directory for converted checkpoints.",
    )
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading talker checkpoint: {args.talker_checkpoint}")
    hf_state = torch.load(
        args.talker_checkpoint, map_location="cpu", weights_only=True
    )

    # Convert main talker backbone.
    print("Converting main talker backbone...")
    main_state = _convert_backbone(hf_state, "model.", "codec_head.weight")
    main_path = args.output_dir / "talker_main.pth"
    torch.save(main_state, main_path)
    print(f"  Saved {len(main_state)} keys -> {main_path}")

    # Convert code predictor backbone.
    print("Converting code predictor backbone...")
    cp_state = _convert_backbone(hf_state, "code_predictor.model.", "")
    cp_path = args.output_dir / "talker_code_predictor.pth"
    torch.save(cp_state, cp_path)
    print(f"  Saved {len(cp_state)} keys -> {cp_path}")

    # Extract auxiliary weights.
    print("Extracting auxiliary weights...")
    aux_state = _extract_aux_weights(hf_state)
    aux_path = args.output_dir / "talker_aux.pth"
    torch.save(aux_state, aux_path)
    print(f"  Saved {len(aux_state)} keys -> {aux_path}")

    # Write config files alongside.
    config_dir = Path(__file__).resolve().parent / "config"
    for name in ("talker_config.json", "code_predictor_config.json"):
        src = config_dir / name
        if src.exists():
            import shutil
            shutil.copy2(src, args.output_dir / name)
            print(f"  Copied {name}")

    print("Done.")


if __name__ == "__main__":
    main()
