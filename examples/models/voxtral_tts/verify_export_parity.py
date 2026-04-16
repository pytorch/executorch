#!/usr/bin/env python3

import argparse
import gc
import json
from pathlib import Path
from typing import Any

import torch
from torch.export import Dim, export

from executorch.examples.models.voxtral_tts.export_voxtral_tts import (
    AudioTokenEmbeddingExport,
    PredictVelocityExport,
    SemanticHeadExport,
    TextDecoderExport,
    TokenEmbeddingExport,
    lower_to_executorch,
    resolve_effective_quantization,
)
from executorch.examples.models.voxtral_tts.model import N_SPECIAL_TOKENS, load_model
from executorch.examples.models.voxtral_tts.parity import (
    build_reference_prompt_ids,
    encode_speech_request_tokens,
    splice_voice_embeddings,
    tensor_summary,
    topk_pairs,
)
from executorch.examples.models.voxtral_tts.voice import load_voice_from_model_dir
from executorch.extension.llm.export.quantize import quantize_model_
from executorch.extension.pybindings.portable_lib import _load_for_executorch


def tokenize_text(tokenizer_path: str, text: str) -> list[int]:
    from mistral_common.tokens.tokenizers.mistral import MistralTokenizer

    tok = MistralTokenizer.from_file(tokenizer_path)
    inner = tok.instruct_tokenizer.tokenizer
    return inner.encode(text, bos=False, eos=False)


def reset_kv_caches(decoder: torch.nn.Module) -> None:
    for layer in decoder.layers:
        layer.attention.kv_cache.k_cache.zero_()
        layer.attention.kv_cache.v_cache.zero_()


def clone_tensor(tensor: torch.Tensor) -> torch.Tensor:
    return tensor.detach().clone().contiguous()


def run_runtime_method(module: Any, method_name: str, *inputs: torch.Tensor) -> torch.Tensor:
    prepared = tuple(clone_tensor(t) for t in inputs)
    try:
        return module.run_method(method_name, prepared)[0]
    except RuntimeError:
        if method_name != "forward":
            return module.forward(prepared)[0]
        raise


def diff_metrics(lhs: torch.Tensor, rhs: torch.Tensor, atol: float) -> dict[str, Any]:
    lhs_f = lhs.detach().float()
    rhs_f = rhs.detach().float()
    diff = (lhs_f - rhs_f).abs()
    same_nonfinite = (~torch.isfinite(lhs_f)) & (~torch.isfinite(rhs_f)) & (lhs_f == rhs_f)
    diff = torch.where(same_nonfinite, torch.zeros_like(diff), diff)
    diff = torch.nan_to_num(diff, nan=float("inf"), posinf=float("inf"), neginf=float("inf"))
    max_abs = float(diff.max().item()) if diff.numel() else 0.0
    mean_abs = float(diff.mean().item()) if diff.numel() else 0.0
    return {
        "max_abs_diff": max_abs,
        "mean_abs_diff": mean_abs,
        "atol": atol,
        "ok": max_abs <= atol,
    }


def summarize_tensor(tensor: torch.Tensor) -> dict[str, Any]:
    if tensor.dtype in (torch.int32, torch.int64) and tensor.numel() <= 64:
        return {
            "shape": list(tensor.shape),
            "values": [int(v) for v in tensor.reshape(-1).tolist()],
        }
    return tensor_summary(tensor)


def stage_report(
    eager: torch.Tensor,
    exported: torch.Tensor,
    runtime: torch.Tensor,
    atol: float,
) -> dict[str, Any]:
    return {
        "eager": summarize_tensor(eager),
        "export": summarize_tensor(exported),
        "runtime": summarize_tensor(runtime),
        "eager_vs_export": diff_metrics(eager, exported, atol),
        "eager_vs_runtime": diff_metrics(eager, runtime, atol),
        "export_vs_runtime": diff_metrics(exported, runtime, atol),
    }


def semantic_triplet_report(
    eager_logits: torch.Tensor,
    export_logits: torch.Tensor,
    runtime_logits: torch.Tensor,
    *,
    atol: float,
) -> tuple[dict[str, Any], dict[str, list[list[float | int]]]]:
    k = min(5, eager_logits.shape[-1], export_logits.shape[-1], runtime_logits.shape[-1])
    return stage_report(
        eager_logits,
        export_logits,
        runtime_logits,
        atol,
    ), {
        "eager": topk_pairs(eager_logits[0], k=k),
        "export": topk_pairs(export_logits[0], k=k),
        "runtime": topk_pairs(runtime_logits[0], k=k),
    }


def quantize_acoustic_codes(x: torch.Tensor, acoustic_levels: int) -> torch.Tensor:
    x_clamped = x.clamp(-1, 1)
    scaled = ((x_clamped + 1) / 2) * (acoustic_levels - 1)
    return scaled.round().long() + N_SPECIAL_TOKENS


def build_canonical_prompt(
    model: torch.nn.Module,
    model_dir: Path,
    text: str,
    voice: str | None,
) -> dict[str, Any]:
    config = model.config
    voice_embed, voice_path = load_voice_from_model_dir(model_dir, voice, dim=config.dim)
    voice_name = voice_path.stem
    tokenizer_path = str(model_dir / "tekken.json")
    text_tokens = tokenize_text(tokenizer_path, text)
    prompt = build_reference_prompt_ids(
        text_tokens=text_tokens,
        voice_len=voice_embed.shape[0],
        begin_audio_token_id=config.begin_audio_token_id,
        audio_token_id=config.audio_token_id,
        text_to_audio_token_id=config.text_to_audio_token_id,
        repeat_audio_text_token_id=config.repeat_audio_text_token_id,
    )
    official_prompt_ids = encode_speech_request_tokens(tokenizer_path, text, voice_name)
    if prompt.token_ids != official_prompt_ids:
        raise RuntimeError(
            "Manual prompt construction diverges from mistral_common "
            f"encode_speech_request for voice={voice_name}"
        )

    prompt_ids_t = torch.tensor([official_prompt_ids], dtype=torch.long)
    prompt_token_embeds = model.decoder.tok_embeddings(prompt_ids_t)
    prompt_embeds = splice_voice_embeddings(
        prompt_token_embeds,
        voice_embed,
        prompt.voice_start,
    )
    seed_ids = torch.tensor([[config.audio_token_id]], dtype=torch.long)
    seed_embed = model.decoder.tok_embeddings(seed_ids)

    return {
        "voice_path": str(voice_path),
        "voice_name": voice_name,
        "voice_len": int(voice_embed.shape[0]),
        "prompt_token_ids": official_prompt_ids,
        "prompt_token_ids_tensor": prompt_ids_t.detach(),
        "prompt_token_embeds": prompt_token_embeds.detach(),
        "voice_start": prompt.voice_start,
        "prompt_embeds": prompt_embeds.detach(),
        "prompt_positions": torch.arange(len(official_prompt_ids), dtype=torch.long),
        "prompt_len": len(official_prompt_ids),
        "seed_token_ids": seed_ids.detach(),
        "seed_embed": seed_embed.detach(),
        "seed_position": torch.tensor([len(official_prompt_ids)], dtype=torch.long),
    }


def resolve_requested_methods(methods_arg: str) -> set[str]:
    requested_methods = {part.strip() for part in methods_arg.split(",") if part.strip()}
    if "all" in requested_methods:
        return {
            "token_embedding",
            "text_decoder",
            "semantic_head",
            "predict_velocity",
            "audio_token_embedding",
        }
    return requested_methods


def apply_quantization(
    model: torch.nn.Module,
    *,
    qlinear: str | None,
    qlinear_group_size: int | None,
    qlinear_packing_format: str | None,
    qembedding: str | None,
    qembedding_group_size: int | None,
    decoder_qlinear_scope: str = "all",
) -> None:
    if qlinear:
        qlinear_kwargs = {
            "qlinear_config": qlinear,
            "qlinear_group_size": qlinear_group_size,
            "qlinear_packing_format": qlinear_packing_format,
        }
        if decoder_qlinear_scope == "all":
            quantize_model_(model.decoder, **qlinear_kwargs)
        elif decoder_qlinear_scope == "attention":
            for layer in model.decoder.layers:
                quantize_model_(layer.attention, **qlinear_kwargs)
        elif decoder_qlinear_scope == "feed_forward":
            for layer in model.decoder.layers:
                quantize_model_(layer.feed_forward, **qlinear_kwargs)
        elif decoder_qlinear_scope != "none":
            raise ValueError(
                f"Unsupported decoder_qlinear_scope: {decoder_qlinear_scope}"
            )
        quantize_model_(
            model.flow_head,
            qlinear_config=qlinear,
            qlinear_group_size=qlinear_group_size,
            qlinear_packing_format=qlinear_packing_format,
            skip_incompatible_shapes=True,
        )

    if qembedding:
        tok_emb_wrapper = TokenEmbeddingExport(model)
        quantize_model_(
            tok_emb_wrapper,
            qembedding_config=qembedding,
            qembedding_group_size=qembedding_group_size,
        )
        audio_tok_emb_wrapper = AudioTokenEmbeddingExport(model)
        quantize_model_(
            audio_tok_emb_wrapper,
            qembedding_config=qembedding,
            qembedding_group_size=qembedding_group_size,
        )


def build_export_and_runtime_modules(
    model: torch.nn.Module,
    requested_methods: set[str],
    max_seq_len: int,
    *,
    backend: str = "portable",
    temp_dir: str | Path | None = None,
    temp_prefix: str = "voxtral_fp32_parity",
) -> tuple[dict[str, Any], dict[str, Any]]:
    config = model.config
    export_modules: dict[str, Any] = {}
    runtime_modules: dict[str, Any] = {}
    temp_root = Path("/tmp") if temp_dir is None else Path(temp_dir)
    temp_root.mkdir(parents=True, exist_ok=True)

    def lower_method(name: str, exported_program: Any) -> None:
        export_modules[name] = exported_program.module()
        et_program = lower_to_executorch(
            {name: exported_program},
            metadata={},
            backend=backend,
        )
        pte_path = temp_root / f"{temp_prefix}_{name}.pte"
        with pte_path.open("wb") as f:
            et_program.write_to_file(f)
        runtime_modules[name] = _load_for_executorch(str(pte_path))
        del et_program
        gc.collect()

    if "token_embedding" in requested_methods:
        tok_seq_dim = Dim("tok_seq_len", min=1, max=max_seq_len)
        sample_ids = torch.tensor([[0, 1, 2, 3]], dtype=torch.long)
        ep = export(
            TokenEmbeddingExport(model),
            (sample_ids,),
            dynamic_shapes={"token_ids": {1: tok_seq_dim}},
            strict=True,
        )
        lower_method("token_embedding", ep)

    if "audio_token_embedding" in requested_methods:
        sample_audio_codes = torch.zeros(1, config.n_codebooks, 1, dtype=torch.long)
        ep = export(
            AudioTokenEmbeddingExport(model),
            (sample_audio_codes,),
            strict=True,
        )
        lower_method("audio_token_embedding", ep)

    if "text_decoder" in requested_methods:
        seq_dim = Dim("seq_len", min=1, max=max_seq_len)
        sample_embeds = torch.randn(1, 4, config.dim, dtype=torch.float32)
        sample_pos = torch.arange(4, dtype=torch.long)
        ep = export(
            TextDecoderExport(model),
            (sample_embeds, sample_pos),
            dynamic_shapes={
                "input_embeds": {1: seq_dim},
                "cache_position": {0: seq_dim},
            },
            strict=True,
        )
        lower_method("text_decoder", ep)

    if "semantic_head" in requested_methods:
        sample_hidden = torch.randn(1, config.dim, dtype=torch.float32)
        ep = export(
            SemanticHeadExport(model),
            (sample_hidden,),
            strict=True,
        )
        lower_method("semantic_head", ep)

    if "predict_velocity" in requested_methods:
        sample_xt = torch.randn(1, config.acoustic_dim, dtype=torch.float32)
        sample_tidx = torch.tensor([0], dtype=torch.long)
        sample_hidden = torch.randn(1, config.dim, dtype=torch.float32)
        ep = export(
            PredictVelocityExport(model),
            (sample_xt, sample_tidx, sample_hidden),
            strict=True,
        )
        lower_method("predict_velocity", ep)

    return export_modules, runtime_modules


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Compare eager FP32, torch.export, and ExecuTorch runtime parity for "
            "Voxtral text_decoder / semantic_head / predict_velocity."
        )
    )
    parser.add_argument("--model-path", required=True)
    parser.add_argument(
        "--backend",
        default="portable",
        choices=["portable", "xnnpack"],
        help="Backend used for lowered export/runtime modules.",
    )
    parser.add_argument("--text", default="Hello, how are you today?")
    parser.add_argument("--voice", default=None)
    parser.add_argument("--max-seq-len", type=int, default=512)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--atol", type=float, default=1e-4)
    parser.add_argument(
        "--qlinear",
        default=None,
        choices=["4w", "8w", "8da4w", "8da8w"],
    )
    parser.add_argument("--qlinear-group-size", type=int, default=None)
    parser.add_argument("--qlinear-packing-format", default=None)
    parser.add_argument(
        "--qembedding",
        default=None,
        choices=["4w", "8w"],
    )
    parser.add_argument("--qembedding-group-size", type=int, default=None)
    parser.add_argument(
        "--decoder-qlinear-scope",
        default="all",
        choices=["all", "attention", "feed_forward", "none"],
        help="Limit decoder linear quantization to a sub-scope for parity isolation.",
    )
    parser.add_argument(
        "--methods",
        default="all",
        help=(
            "Comma-separated subset of methods to compare. "
            "Supported: all,text_decoder,semantic_head,predict_velocity,"
            "audio_token_embedding,token_embedding"
        ),
    )
    parser.add_argument("--output-json", default=None)
    args = parser.parse_args()
    quant_plan = resolve_effective_quantization(
        backend=args.backend,
        qlinear=args.qlinear,
        qembedding=args.qembedding,
    )
    effective_qlinear = quant_plan["qlinear"]
    effective_qembedding = quant_plan["qembedding"]

    model_dir = Path(args.model_path)
    model = load_model(
        args.model_path,
        max_seq_len=args.max_seq_len,
        dtype=torch.float32,
        backend="portable",
    )
    model.eval()

    prompt = build_canonical_prompt(model, model_dir, args.text, args.voice)
    config = model.config

    reset_kv_caches(model.decoder)

    requested_methods = resolve_requested_methods(args.methods)

    prompt_token_ids = clone_tensor(prompt["prompt_token_ids_tensor"])
    eager_prompt_token_embeds = clone_tensor(prompt["prompt_token_embeds"])
    prompt_embeds = clone_tensor(prompt["prompt_embeds"])
    prompt_positions = clone_tensor(prompt["prompt_positions"])
    seed_token_ids = clone_tensor(prompt["seed_token_ids"])
    seed_embed = clone_tensor(prompt["seed_embed"])
    seed_position = clone_tensor(prompt["seed_position"])
    prompt_len = int(prompt["prompt_len"])

    semantic_eager = None
    acoustic_eager = None
    semantic_code_eager = None
    frame0_codes_eager = None
    audio_embed_eager = None
    frame1_hidden_eager = None
    eager_flow_outputs: dict[str, torch.Tensor] = {}
    x0 = None
    zero_hidden = None
    timesteps = None

    with torch.no_grad():
        eager_prefill_all = model.decoder(clone_tensor(prompt_embeds), clone_tensor(prompt_positions))
        eager_prefill_hidden = eager_prefill_all[:, -1, :].detach()
        eager_seed_hidden = model.decoder(
            clone_tensor(seed_embed),
            clone_tensor(seed_position),
        )[:, 0, :].detach()

        if "semantic_head" in requested_methods or "predict_velocity" in requested_methods:
            semantic_eager = model.flow_head.semantic_logits(clone_tensor(eager_seed_hidden)).detach()

        x0 = torch.randn(
            1,
            config.acoustic_dim,
            generator=torch.Generator().manual_seed(args.seed),
        ).float() * config.noise_scale
        zero_hidden = torch.zeros_like(eager_seed_hidden)
        timesteps = torch.linspace(0, 1, config.n_decoding_steps + 1)

        if "predict_velocity" in requested_methods:
            x_eager = clone_tensor(x0)
            for step in range(config.n_decoding_steps):
                t_idx = torch.tensor([step], dtype=torch.long)
                dt = timesteps[step + 1] - timesteps[step]

                eager_v_cond = model.flow_head.predict_velocity(
                    clone_tensor(x_eager),
                    clone_tensor(t_idx),
                    clone_tensor(eager_seed_hidden),
                ).detach()
                eager_v_uncond = model.flow_head.predict_velocity(
                    clone_tensor(x_eager),
                    clone_tensor(t_idx),
                    clone_tensor(zero_hidden),
                ).detach()

                eager_flow_outputs[f"flow_step_{step}_v_cond"] = eager_v_cond
                eager_flow_outputs[f"flow_step_{step}_v_uncond"] = eager_v_uncond

                eager_v = config.cfg_alpha * eager_v_cond + (1 - config.cfg_alpha) * eager_v_uncond
                x_eager = x_eager + eager_v * dt
                eager_flow_outputs[f"flow_step_{step}_x"] = x_eager.detach()

            acoustic_eager = quantize_acoustic_codes(x_eager, config.acoustic_levels)
            if semantic_eager is not None:
                semantic_code_eager = semantic_eager.argmax(dim=-1)
                frame0_codes_eager = torch.cat(
                    [semantic_code_eager.view(1, 1), acoustic_eager],
                    dim=1,
                ).unsqueeze(-1)

        if frame0_codes_eager is not None and "audio_token_embedding" in requested_methods:
            audio_embed_eager = model.audio_token_embedding(clone_tensor(frame0_codes_eager)).detach()
            if "text_decoder" in requested_methods:
                frame1_position = torch.tensor([prompt_len + 1], dtype=torch.long)
                frame1_hidden_eager = model.decoder(
                    clone_tensor(audio_embed_eager),
                    clone_tensor(frame1_position),
                )[:, 0, :].detach()

    if effective_qlinear or effective_qembedding:
        apply_quantization(
            model,
            qlinear=effective_qlinear,
            qlinear_group_size=args.qlinear_group_size,
            qlinear_packing_format=args.qlinear_packing_format,
            qembedding=effective_qembedding,
            qembedding_group_size=args.qembedding_group_size,
            decoder_qlinear_scope=args.decoder_qlinear_scope,
        )

    reset_kv_caches(model.decoder)
    temp_prefix = "voxtral_{}_qlinear_{}_qembedding_{}".format(
        args.backend,
        effective_qlinear or "none",
        effective_qembedding or "none",
    )
    temp_prefix = f"{temp_prefix}_decoder_{args.decoder_qlinear_scope}"
    export_modules, runtime_modules = build_export_and_runtime_modules(
        model,
        requested_methods,
        args.max_seq_len,
        backend=args.backend,
        temp_prefix=temp_prefix,
    )

    export_prefill_hidden = None
    export_seed_hidden = None
    runtime_prefill_hidden = None
    runtime_seed_hidden = None
    token_embed_eager = None
    token_embed_export = None
    token_embed_runtime = None
    seed_token_embed_eager = None
    seed_token_embed_export = None
    seed_token_embed_runtime = None
    semantic_export = None
    semantic_runtime = None
    semantic_export_on_quantized_seed_hidden = None
    semantic_runtime_on_quantized_seed_hidden = None
    flow_stages: dict[str, Any] = {}
    acoustic_export = None
    acoustic_runtime = None
    semantic_code_export = None
    semantic_code_runtime = None
    frame0_codes_export = None
    frame0_codes_runtime = None
    audio_embed_export = None
    audio_embed_runtime = None
    frame1_hidden_export = None
    frame1_hidden_runtime = None

    with torch.no_grad():
        if "token_embedding" in export_modules and "token_embedding" in runtime_modules:
            token_embed_eager = eager_prompt_token_embeds.detach()
            token_embed_export = export_modules["token_embedding"](
                clone_tensor(prompt_token_ids)
            ).detach()
            token_embed_runtime = run_runtime_method(
                runtime_modules["token_embedding"],
                "token_embedding",
                prompt_token_ids,
            ).detach()
            seed_token_embed_eager = seed_embed.detach()
            seed_token_embed_export = export_modules["token_embedding"](
                clone_tensor(seed_token_ids)
            ).detach()
            seed_token_embed_runtime = run_runtime_method(
                runtime_modules["token_embedding"],
                "token_embedding",
                seed_token_ids,
            ).detach()

        export_text_decoder = export_modules.get("text_decoder")
        runtime_text_decoder = runtime_modules.get("text_decoder")
        if export_text_decoder is not None and runtime_text_decoder is not None:
            export_prefill_all = export_text_decoder(
                clone_tensor(prompt_embeds),
                clone_tensor(prompt_positions),
            )
            export_prefill_hidden = export_prefill_all[:, -1, :].detach()
            export_seed_hidden = export_text_decoder(
                clone_tensor(seed_embed),
                clone_tensor(seed_position),
            )[:, 0, :].detach()

            runtime_prefill_all = run_runtime_method(
                runtime_text_decoder,
                "text_decoder",
                prompt_embeds,
                prompt_positions,
            )
            runtime_prefill_hidden = runtime_prefill_all[:, -1, :].detach()
            runtime_seed_hidden = run_runtime_method(
                runtime_text_decoder,
                "text_decoder",
                seed_embed,
                seed_position,
            )[:, 0, :].detach()

        if "semantic_head" in export_modules and "semantic_head" in runtime_modules:
            semantic_export = export_modules["semantic_head"](
                clone_tensor(eager_seed_hidden)
            ).detach()
            semantic_runtime = run_runtime_method(
                runtime_modules["semantic_head"],
                "semantic_head",
                eager_seed_hidden,
            ).detach()
            if export_seed_hidden is not None and runtime_seed_hidden is not None:
                semantic_export_on_quantized_seed_hidden = export_modules["semantic_head"](
                    clone_tensor(export_seed_hidden)
                ).detach()
                semantic_runtime_on_quantized_seed_hidden = run_runtime_method(
                    runtime_modules["semantic_head"],
                    "semantic_head",
                    runtime_seed_hidden,
                ).detach()

        if (
            x0 is not None
            and zero_hidden is not None
            and timesteps is not None
            and "predict_velocity" in export_modules
            and "predict_velocity" in runtime_modules
        ):
            x_export = clone_tensor(x0)
            x_runtime = clone_tensor(x0)

            for step in range(config.n_decoding_steps):
                t_idx = torch.tensor([step], dtype=torch.long)
                dt = timesteps[step + 1] - timesteps[step]

                export_v_cond = export_modules["predict_velocity"](
                    clone_tensor(x_export),
                    clone_tensor(t_idx),
                    clone_tensor(eager_seed_hidden),
                ).detach()
                runtime_v_cond = run_runtime_method(
                    runtime_modules["predict_velocity"],
                    "predict_velocity",
                    x_runtime,
                    t_idx,
                    eager_seed_hidden,
                ).detach()

                export_v_uncond = export_modules["predict_velocity"](
                    clone_tensor(x_export),
                    clone_tensor(t_idx),
                    clone_tensor(zero_hidden),
                ).detach()
                runtime_v_uncond = run_runtime_method(
                    runtime_modules["predict_velocity"],
                    "predict_velocity",
                    x_runtime,
                    t_idx,
                    zero_hidden,
                ).detach()

                flow_stages[f"flow_step_{step}_v_cond"] = stage_report(
                    eager_flow_outputs[f"flow_step_{step}_v_cond"],
                    export_v_cond,
                    runtime_v_cond,
                    args.atol,
                )
                flow_stages[f"flow_step_{step}_v_uncond"] = stage_report(
                    eager_flow_outputs[f"flow_step_{step}_v_uncond"],
                    export_v_uncond,
                    runtime_v_uncond,
                    args.atol,
                )

                export_v = config.cfg_alpha * export_v_cond + (1 - config.cfg_alpha) * export_v_uncond
                runtime_v = config.cfg_alpha * runtime_v_cond + (1 - config.cfg_alpha) * runtime_v_uncond

                x_export = x_export + export_v * dt
                x_runtime = x_runtime + runtime_v * dt

                flow_stages[f"flow_step_{step}_x"] = stage_report(
                    eager_flow_outputs[f"flow_step_{step}_x"],
                    x_export,
                    x_runtime,
                    args.atol,
                )

            acoustic_export = quantize_acoustic_codes(x_export, config.acoustic_levels)
            acoustic_runtime = quantize_acoustic_codes(x_runtime, config.acoustic_levels)

            if semantic_eager is not None and semantic_export is not None and semantic_runtime is not None:
                semantic_code_export = semantic_export.argmax(dim=-1)
                semantic_code_runtime = semantic_runtime.argmax(dim=-1)
                frame0_codes_export = torch.cat(
                    [semantic_code_export.view(1, 1), acoustic_export],
                    dim=1,
                ).unsqueeze(-1)
                frame0_codes_runtime = torch.cat(
                    [semantic_code_runtime.view(1, 1), acoustic_runtime],
                    dim=1,
                ).unsqueeze(-1)

        if (
            frame0_codes_eager is not None
            and "audio_token_embedding" in export_modules
            and "audio_token_embedding" in runtime_modules
        ):
            audio_embed_export = export_modules["audio_token_embedding"](
                clone_tensor(frame0_codes_eager)
            ).detach()
            audio_embed_runtime = run_runtime_method(
                runtime_modules["audio_token_embedding"],
                "audio_token_embedding",
                frame0_codes_eager,
            ).detach()

            if (
                audio_embed_eager is not None
                and export_text_decoder is not None
                and runtime_text_decoder is not None
            ):
                frame1_position = torch.tensor([prompt_len + 1], dtype=torch.long)
                frame1_hidden_export = export_text_decoder(
                    clone_tensor(audio_embed_eager),
                    clone_tensor(frame1_position),
                )[:, 0, :].detach()
                frame1_hidden_runtime = run_runtime_method(
                    runtime_text_decoder,
                    "text_decoder",
                    audio_embed_eager,
                    frame1_position,
                )[:, 0, :].detach()

    stages: dict[str, Any] = {}
    if token_embed_eager is not None and token_embed_export is not None and token_embed_runtime is not None:
        stages["token_embedding_on_prompt_tokens"] = stage_report(
            token_embed_eager,
            token_embed_export,
            token_embed_runtime,
            args.atol,
        )
        stages["token_embedding_on_audio_seed_token"] = stage_report(
            seed_token_embed_eager,
            seed_token_embed_export,
            seed_token_embed_runtime,
            args.atol,
        )
    if export_prefill_hidden is not None and runtime_prefill_hidden is not None:
        stages["prefill_hidden"] = stage_report(
            eager_prefill_hidden,
            export_prefill_hidden,
            runtime_prefill_hidden,
            args.atol,
        )
        stages["seed_hidden"] = stage_report(
            eager_seed_hidden,
            export_seed_hidden,
            runtime_seed_hidden,
            args.atol,
        )
    if semantic_eager is not None and semantic_export is not None and semantic_runtime is not None:
        stages["semantic_logits_on_eager_seed_hidden"] = stage_report(
            semantic_eager,
            semantic_export,
            semantic_runtime,
            args.atol,
        )
    semantic_topk_on_quantized_seed_hidden = None
    if (
        semantic_eager is not None
        and semantic_export_on_quantized_seed_hidden is not None
        and semantic_runtime_on_quantized_seed_hidden is not None
    ):
        (
            stages["semantic_logits_on_quantized_seed_hidden"],
            semantic_topk_on_quantized_seed_hidden,
        ) = semantic_triplet_report(
            semantic_eager,
            semantic_export_on_quantized_seed_hidden,
            semantic_runtime_on_quantized_seed_hidden,
            atol=args.atol,
        )
        stages["semantic_code_on_eager_seed_hidden"] = stage_report(
            semantic_eager.argmax(dim=-1),
            semantic_export.argmax(dim=-1),
            semantic_runtime.argmax(dim=-1),
            0.0,
        )
    if acoustic_eager is not None and acoustic_export is not None and acoustic_runtime is not None:
        stages["frame0_acoustic_codes"] = stage_report(
            acoustic_eager,
            acoustic_export,
            acoustic_runtime,
            0.0,
        )
    if frame0_codes_eager is not None and frame0_codes_export is not None and frame0_codes_runtime is not None:
        stages["frame0_full_codes"] = stage_report(
            frame0_codes_eager,
            frame0_codes_export,
            frame0_codes_runtime,
            0.0,
        )
    if audio_embed_eager is not None and audio_embed_export is not None and audio_embed_runtime is not None:
        stages["audio_token_embedding_on_eager_frame0_codes"] = stage_report(
            audio_embed_eager,
            audio_embed_export,
            audio_embed_runtime,
            args.atol,
        )
    if frame1_hidden_eager is not None and frame1_hidden_export is not None and frame1_hidden_runtime is not None:
        stages["frame1_hidden_from_eager_audio_embed"] = stage_report(
            frame1_hidden_eager,
            frame1_hidden_export,
            frame1_hidden_runtime,
            args.atol,
        )
    stages.update(flow_stages)

    failed = [
        stage_name
        for stage_name, report in stages.items()
        if not all(
            report[pair]["ok"]
            for pair in ("eager_vs_export", "eager_vs_runtime", "export_vs_runtime")
        )
    ]

    likely_root_cause = "unknown"
    if "prefill_hidden" in stages and "seed_hidden" in stages:
        prefill_runtime = stages["prefill_hidden"]["eager_vs_runtime"]
        seed_runtime = stages["seed_hidden"]["eager_vs_runtime"]
        prefill_export = stages["prefill_hidden"]["eager_vs_export"]
        seed_export = stages["seed_hidden"]["eager_vs_export"]
        if prefill_export["ok"] and seed_export["ok"]:
            if (
                prefill_runtime["max_abs_diff"] <= 2 * args.atol
                and seed_runtime["max_abs_diff"] <= 2 * args.atol
            ):
                likely_root_cause = "small_runtime_text_decoder_epsilon"
            elif "semantic_logits_on_eager_seed_hidden" not in stages or stages[
                "semantic_logits_on_eager_seed_hidden"
            ]["eager_vs_runtime"]["ok"]:
                likely_root_cause = "text_decoder_stateful_path"
    elif any(
        not stages[f"flow_step_{step}_v_cond"]["eager_vs_runtime"]["ok"]
        or not stages[f"flow_step_{step}_v_uncond"]["eager_vs_runtime"]["ok"]
        for step in range(config.n_decoding_steps)
        if f"flow_step_{step}_v_cond" in stages and f"flow_step_{step}_v_uncond" in stages
    ):
        likely_root_cause = "predict_velocity_path"
    elif failed:
        likely_root_cause = "later_stage_or_runner_orchestration"
    else:
        likely_root_cause = "no_fp32_export_gap_detected"

    result = {
        "text": args.text,
        "voice_path": prompt["voice_path"],
        "voice_name": prompt["voice_name"],
        "voice_len": prompt["voice_len"],
        "prompt_len": prompt_len,
        "prompt_token_ids": prompt["prompt_token_ids"],
        "backend": args.backend,
        "qlinear": effective_qlinear,
        "qlinear_group_size": args.qlinear_group_size,
        "qlinear_packing_format": args.qlinear_packing_format,
        "qembedding": effective_qembedding,
        "qembedding_group_size": args.qembedding_group_size,
        "requested_qlinear": args.qlinear,
        "requested_qembedding": args.qembedding,
        "decoder_qlinear_scope": args.decoder_qlinear_scope,
        "requested_decoder_qlinear_scope": args.decoder_qlinear_scope,
        "quantization_warning": quant_plan["warning"],
        "requested_methods": sorted(requested_methods),
        "stages": stages,
        "failed_stages": failed,
        "likely_root_cause": likely_root_cause,
        "ok": not failed,
    }
    if semantic_eager is not None and semantic_export is not None and semantic_runtime is not None:
        result["semantic_topk_on_eager_seed_hidden"] = {
            "eager": topk_pairs(semantic_eager[0], k=5),
            "export": topk_pairs(semantic_export[0], k=5),
            "runtime": topk_pairs(semantic_runtime[0], k=5),
        }
    if semantic_topk_on_quantized_seed_hidden is not None:
        result["semantic_topk_on_quantized_seed_hidden"] = (
            semantic_topk_on_quantized_seed_hidden
        )

    if args.output_json:
        Path(args.output_json).write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")

    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if not failed else 1


if __name__ == "__main__":
    raise SystemExit(main())
