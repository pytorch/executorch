#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Generate target-only eager evidence for Gemma 4 speculative decode."""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import math
import socket
import sys

from datetime import datetime, timezone
from pathlib import Path
from typing import Sequence

import torch

from executorch.examples.models.gemma4.target_prefill_contract import (
    canonical_json_bytes,
    file_identity,
    final_chunk_range,
    prompt_plan_sha256,
    prompt_tokens,
    TARGET_PREFILL_ATOL,
    TARGET_PREFILL_AUTHORITY,
    TARGET_PREFILL_CHUNK_SIZE,
    TARGET_PREFILL_CONTEXTS,
    TARGET_PREFILL_ENVELOPE_KIND,
    TARGET_PREFILL_RTOL,
    TARGET_PREFILL_SCHEMA_VERSION,
    validate_target_prefill_receipt,
)
from executorch.examples.models.gemma4.text_decoder.gemma4_attention import (
    Gemma4KVCache,
)


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def chunk_ranges(context: int) -> list[tuple[int, int]]:
    if context <= 0:
        raise ValueError("target-prefill context must be positive")
    ranges: list[tuple[int, int]] = []
    start = 0
    while start < context:
        length = min(TARGET_PREFILL_CHUNK_SIZE, context - start)
        ranges.append((start, length))
        start += length
    return ranges


def _dtype_name(dtype: torch.dtype) -> str:
    names = {
        torch.float32: "float32",
        torch.float16: "float16",
        torch.bfloat16: "bfloat16",
        torch.int64: "int64",
        torch.int32: "int32",
    }
    if dtype not in names:
        raise ValueError(f"unsupported target-prefill tensor dtype: {dtype}")
    return names[dtype]


def _tensor_bytes(tensor: torch.Tensor) -> bytes:
    value = tensor.detach().cpu().contiguous()
    if value.dtype == torch.bfloat16:
        value = value.view(torch.uint16)
    return value.numpy().tobytes()


def _tensor_envelope(tensor: torch.Tensor) -> dict[str, object]:
    if not torch.isfinite(tensor).all().item():
        raise ValueError("target-prefill tensor contains non-finite values")
    encoded = _tensor_bytes(tensor)
    return {
        "byte_order": "little",
        "dtype": _dtype_name(tensor.dtype),
        "layout": "row_major_contiguous",
        "sha256": hashlib.sha256(encoded).hexdigest(),
        "shape": list(tensor.shape),
    }


def _compare_av(fused: torch.Tensor, manual: torch.Tensor) -> dict[str, object]:
    if (
        not torch.isfinite(fused).all().item()
        or not torch.isfinite(manual).all().item()
    ):
        raise ValueError("target-prefill AV contains non-finite values")
    torch.testing.assert_close(
        fused,
        manual,
        rtol=TARGET_PREFILL_RTOL,
        atol=TARGET_PREFILL_ATOL,
    )
    difference = (fused.to(torch.float64) - manual.to(torch.float64)).abs()
    max_abs = float(difference.max().item()) if difference.numel() else 0.0
    reference_rms = float(
        torch.sqrt(torch.mean(fused.to(torch.float64).square())).item()
    )
    error_rms = float(torch.sqrt(torch.mean(difference.square())).item())
    rel_rms = 0.0 if reference_rms == 0.0 else error_rms / reference_rms
    if not math.isfinite(max_abs) or not math.isfinite(rel_rms):
        raise ValueError("target-prefill AV metrics are non-finite")
    return {
        "atol": TARGET_PREFILL_ATOL,
        "max_abs": max_abs,
        "passed": True,
        "rel_rms": rel_rms,
        "rtol": TARGET_PREFILL_RTOL,
    }


def _reset_target_kv_caches(model: torch.nn.Module) -> int:
    count = 0
    with torch.no_grad():
        for module in model.modules():
            if not isinstance(module, Gemma4KVCache):
                continue
            module.k_cache.zero_()
            module.v_cache.zero_()
            count += 1
    if count == 0:
        raise ValueError("target-prefill model has no Gemma4 KV caches")
    return count


def _arm_config(use_custom_sdpa: bool) -> dict[str, object]:
    return {
        "dtype": "float32",
        "enable_dynamic_shape": True,
        "group_size": 128,
        "max_seq_len": 8960,
        "text_quantize": "8da4w+emb4",
        "use_custom_sdpa": use_custom_sdpa,
        "use_kv_cache": True,
        "variant": "e2b",
    }


def _load_target(checkpoint: Path, *, use_custom_sdpa: bool) -> torch.nn.Module:
    from executorch.examples.models.gemma4.quant_utils import (
        apply_embedding_quantization,
        apply_linear_quantization,
        parse_quantize,
    )
    from executorch.examples.models.gemma4.text_decoder.gemma4_config import (
        Gemma4Config,
    )
    from executorch.examples.models.gemma4.text_decoder.gemma4_model import Gemma4Model

    config = Gemma4Config.from_config("e2b")
    config.use_kv_cache = True
    config.max_seq_len = 8960
    config.enable_dynamic_shape = True
    config.use_custom_sdpa = use_custom_sdpa
    model = Gemma4Model(
        config=config,
        checkpoint_path=str(checkpoint.resolve()),
        dtype=torch.float32,
    ).get_eager_model()
    linear_quant, embedding_quant = parse_quantize("8da4w+emb4")
    if embedding_quant:
        model = apply_embedding_quantization(model, embedding_quant).eval()
    if linear_quant:
        model = apply_linear_quantization(model, linear_quant, group_size=128).eval()
    return model.eval()


def _module_attribute(owner: object, name: str, label: str) -> torch.nn.Module:
    value = getattr(owner, name, None)
    if not isinstance(value, torch.nn.Module):
        raise ValueError(f"target-prefill model has no {label}")
    return value


def _resolve_capture_modules(
    model: torch.nn.Module,
) -> tuple[torch.nn.Module, torch.nn.Module]:
    text_model = _module_attribute(model, "model", "text model")
    self_decoder = _module_attribute(text_model, "self_decoder", "self decoder")
    layers = getattr(self_decoder, "layers", None)
    if not isinstance(layers, torch.nn.ModuleList) or len(layers) == 0:
        raise ValueError("target-prefill model has no decoder layers")
    self_attn = _module_attribute(layers[0], "self_attn", "layer-0 attention")
    o_proj = _module_attribute(self_attn, "o_proj", "layer-0 output projection")
    lm_head = _module_attribute(text_model, "lm_head", "LM head")
    return o_proj, lm_head


def _run_arm(
    checkpoint: Path,
    *,
    use_custom_sdpa: bool,
) -> dict[int, dict[str, object]]:
    model = _load_target(checkpoint, use_custom_sdpa=use_custom_sdpa)
    layer0_o_proj, lm_head = _resolve_capture_modules(model)
    captures: dict[str, torch.Tensor] = {}

    def capture_av(_module: torch.nn.Module, inputs: tuple[torch.Tensor, ...]) -> None:
        if len(inputs) != 1:
            raise ValueError("target-prefill o_proj hook expected one input")
        captures["av"] = inputs[0].detach().clone()

    def capture_raw_logits(
        _module: torch.nn.Module,
        _inputs: tuple[torch.Tensor, ...],
        output: torch.Tensor,
    ) -> None:
        captures["raw_logits"] = output.detach().clone()

    av_hook = layer0_o_proj.register_forward_pre_hook(capture_av)
    logits_hook = lm_head.register_forward_hook(capture_raw_logits)
    results: dict[int, dict[str, object]] = {}
    try:
        with torch.inference_mode():
            for context in TARGET_PREFILL_CONTEXTS:
                reset_count = _reset_target_kv_caches(model)
                captures.clear()
                tokens = prompt_tokens(context)
                post_logits: torch.Tensor | None = None
                for start, length in chunk_ranges(context):
                    if length > TARGET_PREFILL_CHUNK_SIZE:
                        raise ValueError("target-prefill model call exceeds 512 tokens")
                    input_ids = torch.tensor(
                        [tokens[start : start + length]], dtype=torch.long
                    )
                    input_pos = torch.arange(start, start + length, dtype=torch.long)
                    post_logits = model(input_ids, input_pos, None)
                if post_logits is None:
                    raise ValueError("target-prefill context produced no logits")
                raw_logits = captures.get("raw_logits")
                av = captures.get("av")
                if raw_logits is None or av is None:
                    raise ValueError(
                        "target-prefill hooks did not capture logits and AV"
                    )
                start, length = final_chunk_range(context)
                if tuple(av.shape) != (1, length, 8 * 256):
                    raise ValueError("target-prefill layer-0 AV has the wrong shape")
                results[context] = {
                    "av": av.reshape(1, length, 8, 256),
                    "cache_reset_count": reset_count,
                    "config": _arm_config(use_custom_sdpa),
                    "final_chunk_length": length,
                    "final_chunk_start": start,
                    "logits_post_softcap": post_logits.detach().clone(),
                    "logits_pre_softcap": raw_logits,
                }
    finally:
        av_hook.remove()
        logits_hook.remove()
        del model
        gc.collect()
    return results


def _load_runtime_source_receipt(
    path: Path,
) -> tuple[dict[str, object], dict[str, object], str]:
    try:
        document = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"invalid runtime-source receipt: {path}") from error
    if not isinstance(document, dict):
        raise ValueError("runtime-source receipt must be an object")
    fbsource_commit = document.get("fbsource_commit")
    if (
        not isinstance(fbsource_commit, str)
        or len(fbsource_commit) != 40
        or any(character not in "0123456789abcdef" for character in fbsource_commit)
    ):
        raise ValueError("runtime-source receipt has an invalid fbsource commit")
    return document, file_identity(path), fbsource_commit


def generate_target_prefill_receipt(
    checkpoint_root: Path,
    runtime_source_receipt: Path,
    *,
    command: Sequence[str],
) -> dict[str, object]:
    from executorch.examples.models.gemma4.webgpu_artifact_manifest import (
        validate_export_identity,
    )

    started_at = _utc_now()
    checkpoint_acquisition = dict(validate_export_identity(checkpoint_root))
    _, runtime_identity, fbsource_commit = _load_runtime_source_receipt(
        runtime_source_receipt
    )
    fused = _run_arm(checkpoint_root, use_custom_sdpa=True)
    manual = _run_arm(checkpoint_root, use_custom_sdpa=False)
    contexts: dict[str, object] = {}
    for context in TARGET_PREFILL_CONTEXTS:
        fused_result = fused[context]
        manual_result = manual[context]
        fused_av = fused_result["av"]
        manual_av = manual_result["av"]
        raw_logits = fused_result["logits_pre_softcap"]
        post_logits = fused_result["logits_post_softcap"]
        assert isinstance(fused_av, torch.Tensor)
        assert isinstance(manual_av, torch.Tensor)
        assert isinstance(raw_logits, torch.Tensor)
        assert isinstance(post_logits, torch.Tensor)
        raw_token = int(torch.argmax(raw_logits[:, -1, :], dim=-1).item())
        post_token = int(torch.argmax(post_logits[:, -1, :], dim=-1).item())
        if raw_token != post_token:
            raise ValueError(
                f"target-prefill softcap changes the token at context {context}"
            )
        contexts[str(context)] = {
            "arm_configs": {
                "custom_sdpa_fused": fused_result["config"],
                "manual_unfused": manual_result["config"],
            },
            "cache_reset_counts": {
                "custom_sdpa_fused": fused_result["cache_reset_count"],
                "manual_unfused": manual_result["cache_reset_count"],
            },
            "chunk_size": TARGET_PREFILL_CHUNK_SIZE,
            "context": context,
            "final_chunk_length": fused_result["final_chunk_length"],
            "final_chunk_start": fused_result["final_chunk_start"],
            "layer0_manual_unfused_vs_custom_sdpa_fused": {
                "agreement": _compare_av(fused_av, manual_av),
                "custom_sdpa_fused": _tensor_envelope(fused_av),
                "manual_unfused": _tensor_envelope(manual_av),
            },
            "logits_post_softcap": _tensor_envelope(post_logits),
            "logits_pre_softcap": _tensor_envelope(raw_logits),
            "prefill_token_post_softcap": post_token,
            "prefill_token_raw": raw_token,
            "prompt_plan_sha256": prompt_plan_sha256(context),
        }
    producer_path = Path(__file__)
    producer_identity = file_identity(producer_path)
    receipt: dict[str, object] = {
        "authority": TARGET_PREFILL_AUTHORITY,
        "checkpoint_acquisition": checkpoint_acquisition,
        "contexts": contexts,
        "envelope_kind": TARGET_PREFILL_ENVELOPE_KIND,
        "producer": {
            "fbsource_commit": fbsource_commit,
            "runtime_source_receipt": runtime_identity,
            "source_path": producer_path.name,
            "source_sha256": producer_identity["sha256"],
        },
        "run": {
            "command": list(command),
            "finished_at_utc": _utc_now(),
            "host": socket.gethostname(),
            "started_at_utc": started_at,
        },
        "schema_version": TARGET_PREFILL_SCHEMA_VERSION,
    }
    validate_target_prefill_receipt(
        receipt,
        expected_checkpoint_acquisition=checkpoint_acquisition,
        expected_producer_path=producer_path,
        expected_producer_sha256=str(producer_identity["sha256"]),
        expected_runtime_source_identity=runtime_identity,
        expected_fbsource_commit=fbsource_commit,
    )
    return receipt


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoints", type=Path, required=True)
    parser.add_argument("--runtime-source-receipt", type=Path, required=True)
    parser.add_argument("--contexts", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    try:
        contexts = tuple(int(value) for value in args.contexts.split(","))
    except ValueError as error:
        raise ValueError("--contexts must be a comma-separated integer list") from error
    if contexts != TARGET_PREFILL_CONTEXTS:
        raise ValueError("target-prefill owner run requires the exact ten contexts")
    command = ["generate_target_prefill_oracle", *(argv or sys.argv[1:])]
    receipt = generate_target_prefill_receipt(
        args.checkpoints,
        args.runtime_source_receipt,
        command=command,
    )
    args.output.write_bytes(canonical_json_bytes(receipt))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
