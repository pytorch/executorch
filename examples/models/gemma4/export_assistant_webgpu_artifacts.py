#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-unsafe

from __future__ import annotations

import argparse
import hashlib
import math
import tempfile
import types
from pathlib import Path
from typing import Any, cast

import executorch.backends.vulkan.custom_ops_lib  # noqa: F401
import torch

from executorch.examples.models.gemma4.eagle_webgpu_round import (
    OFFICIAL_QAT_CENTROID_TOP_K,
    OFFICIAL_QAT_NUM_CENTROIDS,
    OFFICIAL_QAT_SELECTED_TOKEN_COUNT,
    OFFICIAL_QAT_TOKENS_PER_CENTROID,
    validate_qat_token_ordering,
)
from executorch.examples.models.gemma4.webgpu_artifact_manifest import (
    validate_assistant_export_identity,
)
from safetensors import safe_open

QAT_VALIDATION_DONOR_SEQUENCE = (2, 16, 511, 512, 513, 514, 1024, 8960, 2)


def _tensor_sha256(tensor: torch.Tensor) -> str:
    payload = tensor.detach().cpu().contiguous().numpy().tobytes()
    return hashlib.sha256(payload).hexdigest()


def _load_raw_token_ordering(
    checkpoint: Path,
) -> tuple[torch.Tensor, dict[str, Any]]:
    model_path = checkpoint / "model.safetensors"
    with safe_open(str(model_path), framework="pt", device="cpu") as tensors:
        matches = [
            name
            for name in tensors.keys()
            if name.endswith("masked_embedding.token_ordering")
        ]
        if len(matches) != 1:
            raise ValueError(
                f"assistant safetensors token-ordering key mismatch: {matches}"
            )
        ordering = tensors.get_tensor(matches[0])
    if ordering.dtype != torch.int64 or tuple(ordering.shape) != (262144,):
        raise ValueError(
            "assistant raw token ordering must be int64 with shape [262144]"
        )
    evidence = validate_qat_token_ordering(ordering)
    if evidence["rawShape"] != [262144] or evidence["shape"] != [2048, 128]:
        raise ValueError("assistant raw token-ordering shape proof mismatch")
    return ordering, evidence


def validate_qat_centroid_scores(scores: torch.Tensor) -> dict[str, Any]:
    if tuple(scores.shape) != (1, 1, OFFICIAL_QAT_NUM_CENTROIDS):
        raise ValueError("assistant centroid scores must have shape [1, 1, 2048]")
    if scores.dtype != torch.float32 or not bool(torch.isfinite(scores).all()):
        raise ValueError("assistant centroid scores must be finite fp32 values")

    top33_values, top33_indices = torch.topk(
        scores,
        k=OFFICIAL_QAT_CENTROID_TOP_K + 1,
        dim=-1,
        sorted=True,
    )
    top32_values = top33_values[..., :OFFICIAL_QAT_CENTROID_TOP_K]
    top32_indices = top33_indices[..., :OFFICIAL_QAT_CENTROID_TOP_K]
    stable_values, stable_indices = torch.sort(
        scores, dim=-1, descending=True, stable=True
    )
    stable_reference_exact = bool(
        torch.equal(
            top33_values,
            stable_values[..., : OFFICIAL_QAT_CENTROID_TOP_K + 1],
        )
        and torch.equal(
            top33_indices,
            stable_indices[..., : OFFICIAL_QAT_CENTROID_TOP_K + 1],
        )
    )
    pairwise_distinct = bool(torch.all(top32_values[..., :-1] > top32_values[..., 1:]))
    boundary_gap = float(
        top33_values[..., OFFICIAL_QAT_CENTROID_TOP_K - 1]
        - top33_values[..., OFFICIAL_QAT_CENTROID_TOP_K]
    )
    if not pairwise_distinct:
        raise ValueError("assistant top-32 values must be pairwise distinct")
    if not math.isfinite(boundary_gap) or boundary_gap <= 0.0:
        raise ValueError("assistant 32/33 boundary gap must be positive")
    if not stable_reference_exact:
        raise ValueError("assistant top-33 order differs from stable descending order")
    return {
        "allFinite": True,
        "boundaryGap": boundary_gap,
        "indicesSha256": _tensor_sha256(top32_indices),
        "stableReferenceExact": True,
        "top32PairwiseDistinct": True,
        "top33IndicesSha256": _tensor_sha256(top33_indices),
        "top33ValuesSha256": _tensor_sha256(top33_values),
        "valuesSha256": _tensor_sha256(top32_values),
    }


def validate_assistant_checkpoint(
    checkpoint: Path,
) -> dict[str, Any]:
    checkpoint = checkpoint.resolve(strict=True)
    if not checkpoint.is_dir():
        raise ValueError("assistant checkpoint must be a directory")
    return dict(validate_assistant_export_identity(checkpoint))


def _quantized_embedding(weight: torch.Tensor, bits: int) -> torch.nn.Module:
    from executorch.examples.models.llama.source_transformation.quantize import (
        EmbeddingQuantHandler,
    )

    holder: Any = torch.nn.Module()
    holder.embed = torch.nn.Embedding(weight.shape[0], weight.shape[1])
    holder.embed.weight = torch.nn.Parameter(weight)
    quantized: Any = EmbeddingQuantHandler(
        holder, bitwidth=bits, group_size=None, packed=bits == 4
    ).quantized_model()
    return quantized.embed


def _masked_embedding_static_output(
    self: Any,
    hidden_states: torch.Tensor,
    _lm_head_weight: torch.Tensor,
) -> torch.Tensor:
    batch, seq_len = hidden_states.shape[:2]
    centroid_logits = self.centroids(hidden_states)
    _, top_k_indices = torch.topk(
        centroid_logits, k=OFFICIAL_QAT_CENTROID_TOP_K, dim=-1
    )
    selected_canonical = torch.nn.functional.embedding(
        top_k_indices, self._webgpu_token_ordering
    ).to(torch.long)
    selected_flat = selected_canonical.reshape(-1)
    selected_embeddings = self._lm_embed(selected_flat).view(
        batch,
        seq_len,
        OFFICIAL_QAT_SELECTED_TOKEN_COUNT,
        hidden_states.shape[-1],
    )
    selected_logits = (
        hidden_states.unsqueeze(-2) @ selected_embeddings.transpose(-1, -2)
    ).squeeze(-2)
    return self._webgpu_output_template.scatter(
        dim=-1,
        index=selected_canonical.view(batch, seq_len, -1),
        src=selected_logits,
    )


def adapt_masked_embedding_for_webgpu(
    masked_embedding: Any,
) -> Any:
    if not hasattr(masked_embedding, "_lm_embed"):
        raise ValueError("assistant WebGPU export requires a quantized LM head")
    evidence = validate_qat_token_ordering(masked_embedding.token_ordering)
    if evidence["shape"] != [
        OFFICIAL_QAT_NUM_CENTROIDS,
        OFFICIAL_QAT_TOKENS_PER_CENTROID,
    ]:
        raise ValueError("assistant token-ordering logical shape mismatch")
    dtype = masked_embedding.centroids.weight.dtype
    device = masked_embedding.centroids.weight.device
    ordering = masked_embedding.token_ordering.detach().to(torch.long).reshape(-1)
    masked_embedding.register_buffer(
        "_webgpu_token_ordering",
        ordering.to(dtype=torch.float32, device=device).view(
            OFFICIAL_QAT_NUM_CENTROIDS,
            OFFICIAL_QAT_TOKENS_PER_CENTROID,
        ),
        persistent=False,
    )
    masked_embedding.register_buffer(
        "_webgpu_output_template",
        torch.full(
            (1, 1, OFFICIAL_QAT_NUM_CENTROIDS * OFFICIAL_QAT_TOKENS_PER_CENTROID),
            torch.finfo(dtype).min,
            dtype=dtype,
            device=device,
        ),
        persistent=False,
    )
    masked_embedding.forward = types.MethodType(
        _masked_embedding_static_output, masked_embedding
    )
    return masked_embedding


class StaticAssistantMasks(torch.nn.Module):
    def __init__(self, max_seq_len: int, sliding_window: int = 512) -> None:
        super().__init__()
        if max_seq_len < 2 or sliding_window <= 0:
            raise ValueError("invalid assistant attention-mask capacity")
        self.max_seq_len = max_seq_len
        self.register_buffer("full_mask", torch.zeros(max_seq_len), persistent=False)
        sliding_mask = torch.full((max_seq_len,), torch.finfo(torch.float32).min)
        sliding_mask[-(sliding_window + 1) :] = 0
        self.register_buffer("sliding_mask", sliding_mask, persistent=False)

    def forward(
        self, full_k: torch.Tensor, sliding_k: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        full_length = full_k.shape[2]
        sliding_length = sliding_k.shape[2]
        full_mask = self.full_mask.narrow(0, 0, full_length).unsqueeze(0)
        sliding_mask = self.sliding_mask.narrow(
            0, self.max_seq_len - sliding_length, sliding_length
        ).unsqueeze(0)
        return full_mask, sliding_mask


class _StaticAssistantQueryRopeLayer(torch.nn.Module):
    def __init__(self, freqs_cos: torch.Tensor, freqs_sin: torch.Tensor) -> None:
        super().__init__()
        self.register_buffer("freqs_cos", freqs_cos, persistent=False)
        self.register_buffer("freqs_sin", freqs_sin, persistent=False)

    def forward(self, query: torch.Tensor, position_ids: torch.Tensor) -> torch.Tensor:
        start_pos = cast(int, position_ids[0, 0].item())
        torch._check_is_size(start_pos)
        torch._check(start_pos >= 0)
        torch._check(start_pos + query.shape[1] <= self.freqs_cos.shape[0])
        return torch.ops.et_vk.apply_rotary_emb_hf_single.default(
            query, self.freqs_cos, self.freqs_sin, start_pos
        )


class StaticAssistantQueryRope(torch.nn.Module):
    def __init__(self, source: Any, max_seq_len: int) -> None:
        super().__init__()
        source_buffer = next(source.buffers())
        positions = torch.arange(max_seq_len, device=source_buffer.device).unsqueeze(0)
        probe = torch.empty(1, 1, 1, dtype=torch.float32, device=source_buffer.device)
        with torch.no_grad():
            sliding_cos, sliding_sin = source(probe, positions, "sliding_attention")
            full_cos, full_sin = source(probe, positions, "full_attention")
        self.sliding_attention = _StaticAssistantQueryRopeLayer(
            sliding_cos.squeeze(0).contiguous(),
            sliding_sin.squeeze(0).contiguous(),
        )
        self.full_attention = _StaticAssistantQueryRopeLayer(
            full_cos.squeeze(0).contiguous(),
            full_sin.squeeze(0).contiguous(),
        )


class StaticAssistantSharedKVAttention(torch.nn.Module):
    def __init__(
        self,
        source: Any,
        query_rope: _StaticAssistantQueryRopeLayer,
    ) -> None:
        super().__init__()
        if not source.is_kv_shared_layer:
            raise ValueError("assistant attention requires shared target KV")
        self.layer_type = source.layer_type
        self.head_dim = source.head_dim
        self.num_attention_heads = (
            source.num_attention_heads
            if hasattr(source, "num_attention_heads")
            else source.config.num_attention_heads
        )
        self.q_proj: Any = source.q_proj
        self.q_norm: Any = source.q_norm
        self.o_proj: Any = source.o_proj
        self.query_rope = query_rope

    def forward(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: Any,
        attention_mask: torch.Tensor,
        shared_kv_states: dict[str, tuple[torch.Tensor, torch.Tensor]],
        position_ids: torch.Tensor,
        **kwargs: Any,
    ) -> tuple[torch.Tensor, None]:
        del position_embeddings, kwargs
        input_shape = hidden_states.shape[:-1]
        query = self.q_proj(hidden_states).view(
            *input_shape, self.num_attention_heads, self.head_dim
        )
        query = self.q_norm(query)
        query = self.query_rope(query, position_ids)
        key, value = shared_kv_states[self.layer_type]
        output = torch.ops.llama.custom_sdpa.default(
            query,
            key.transpose(1, 2),
            value.transpose(1, 2),
            0,
            attention_mask,
            0.0,
            False,
            1.0,
        )
        return self.o_proj(output.reshape(*input_shape, -1)), None


class _UnusedAssistantRotaryEmbedding(torch.nn.Module):
    def forward(
        self,
        x: torch.Tensor,
        position_ids: torch.Tensor,
        layer_type: str | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        del position_ids, layer_type
        return x, x


def _static_assistant_attention_masks(
    self: Any,
    inputs_embeds: torch.Tensor,
    attention_mask: torch.Tensor | None,
    shared_kv_states: dict[str, tuple[torch.Tensor, torch.Tensor]],
) -> dict[str, torch.Tensor]:
    del inputs_embeds
    if attention_mask is not None:
        raise ValueError("assistant WebGPU export requires attention_mask=None")
    full_mask, sliding_mask = self._webgpu_static_masks(
        shared_kv_states["full_attention"][0],
        shared_kv_states["sliding_attention"][0],
    )
    return {
        "full_attention": full_mask,
        "sliding_attention": sliding_mask,
    }


def adapt_assistant_model_for_webgpu(assistant: Any, max_seq_len: int) -> Any:
    layers = list(assistant.model.layers)
    layer_types = [layer.self_attn.layer_type for layer in layers]
    expected_layer_types = [
        "sliding_attention",
        "sliding_attention",
        "sliding_attention",
        "full_attention",
    ]
    if layer_types != expected_layer_types:
        raise ValueError(f"unexpected assistant layer types: {layer_types}")
    query_rope = StaticAssistantQueryRope(
        assistant.model.rotary_emb, max_seq_len=max_seq_len
    )
    for layer in layers:
        source_rope = getattr(query_rope, layer.self_attn.layer_type)
        layer.self_attn = StaticAssistantSharedKVAttention(
            layer.self_attn,
            _StaticAssistantQueryRopeLayer(
                source_rope.freqs_cos, source_rope.freqs_sin
            ),
        )
    assistant.model.rotary_emb = _UnusedAssistantRotaryEmbedding()
    assistant._webgpu_static_masks = StaticAssistantMasks(max_seq_len)
    assistant.create_attention_masks = types.MethodType(
        _static_assistant_attention_masks, assistant
    )
    adapt_masked_embedding_for_webgpu(assistant.masked_embedding)
    return assistant


class UnfoldedAssistant(torch.nn.Module):
    def __init__(self, assistant: Any) -> None:
        super().__init__()
        self.assistant: Any = assistant
        self._webgpu_qat_selection_evidence: dict[str, Any] = {}

    def forward(
        self,
        inputs_embeds: torch.Tensor,
        position_ids: torch.Tensor,
        full_k: torch.Tensor,
        full_v: torch.Tensor,
        sliding_k: torch.Tensor,
        sliding_v: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        output = self.assistant(
            inputs_embeds=inputs_embeds,
            attention_mask=None,
            position_ids=position_ids,
            shared_kv_states={
                "full_attention": (full_k, full_v),
                "sliding_attention": (sliding_k, sliding_v),
            },
            use_cache=False,
        )
        return output.logits, output.last_hidden_state


def _qat_validation_inputs(config: Any, donor_length: int) -> tuple[torch.Tensor, ...]:
    text_config = config.get_text_config()
    generator = torch.Generator().manual_seed(0xE4A61000 + donor_length)
    return (
        torch.randn(
            1,
            1,
            2 * config.backbone_hidden_size,
            generator=generator,
        ),
        torch.tensor([[donor_length - 1]], dtype=torch.long),
        torch.randn(
            1,
            1,
            donor_length,
            text_config.global_head_dim,
            generator=generator,
        ),
        torch.randn(
            1,
            1,
            donor_length,
            text_config.global_head_dim,
            generator=generator,
        ),
        torch.randn(
            1,
            1,
            donor_length,
            text_config.head_dim,
            generator=generator,
        ),
        torch.randn(
            1,
            1,
            donor_length,
            text_config.head_dim,
            generator=generator,
        ),
    )


def _qat_validation_cases(
    config: Any, max_donor_len: int
) -> tuple[tuple[int, ...], list[tuple[torch.Tensor, ...]]]:
    donor_sequence = tuple(
        donor_length
        for donor_length in QAT_VALIDATION_DONOR_SEQUENCE
        if donor_length <= max_donor_len
    )
    if not donor_sequence or donor_sequence[0] != 2 or donor_sequence[-1] != 2:
        raise ValueError("assistant validation requires replay at donor length 2")
    return donor_sequence, [
        _qat_validation_inputs(config, donor_length) for donor_length in donor_sequence
    ]


def _capture_reference_outputs(
    wrapper: UnfoldedAssistant,
    inputs: list[tuple[torch.Tensor, ...]],
) -> list[tuple[torch.Tensor, torch.Tensor]]:
    with torch.no_grad():
        return [
            tuple(output.detach().clone() for output in wrapper(*case))
            for case in inputs
        ]


def validate_qat_selection_contract(  # noqa: C901
    wrapper: UnfoldedAssistant,
    max_donor_len: int,
    *,
    validation_inputs: list[tuple[torch.Tensor, ...]] | None = None,
    reference_outputs: list[tuple[torch.Tensor, torch.Tensor]] | None = None,
    token_ordering_evidence: dict[str, Any] | None = None,
) -> dict[str, Any]:
    donor_sequence, default_inputs = _qat_validation_cases(
        wrapper.assistant.config, max_donor_len
    )
    inputs = default_inputs if validation_inputs is None else validation_inputs
    if len(inputs) != len(donor_sequence):
        raise ValueError("assistant validation input count mismatch")
    if reference_outputs is not None and len(reference_outputs) != len(inputs):
        raise ValueError("assistant eager-reference count mismatch")

    masked_embedding: Any = wrapper.assistant.masked_embedding
    cases = []
    with torch.no_grad():
        for case_index, (donor_length, case_inputs) in enumerate(
            zip(donor_sequence, inputs)
        ):
            head_inputs: list[torch.Tensor] = []
            hook = masked_embedding.register_forward_pre_hook(
                lambda _module, args, destination=head_inputs: destination.append(
                    args[0].detach()
                )
            )
            try:
                actual = tuple(wrapper(*case_inputs))
            finally:
                hook.remove()
            if len(head_inputs) != 1:
                raise ValueError(
                    "assistant masked head must run exactly once per validation case"
                )
            output_evidence = []
            greedy_token_exact = True
            if reference_outputs is not None:
                expected = reference_outputs[case_index]
                for name, expected_tensor, actual_tensor in zip(
                    ("logits", "last_hidden_state"), expected, actual
                ):
                    difference = (actual_tensor - expected_tensor).abs()
                    close = torch.allclose(
                        actual_tensor, expected_tensor, rtol=1e-3, atol=1e-4
                    )
                    if not close:
                        raise ValueError(
                            f"assistant adapter {name} mismatch at K={donor_length}: "
                            f"max_abs={float(difference.max())}"
                        )
                    output_evidence.append(
                        {
                            "actualSha256": _tensor_sha256(actual_tensor),
                            "bitExact": torch.equal(actual_tensor, expected_tensor),
                            "close": True,
                            "maxAbsError": float(difference.max()),
                            "name": name,
                            "referenceSha256": _tensor_sha256(expected_tensor),
                            "shape": list(actual_tensor.shape),
                        }
                    )
                greedy_token_exact = torch.equal(
                    actual[0].argmax(dim=-1), expected[0].argmax(dim=-1)
                )
                if not greedy_token_exact:
                    raise ValueError(
                        f"assistant adapter greedy token mismatch at K={donor_length}"
                    )
            cases.append(
                {
                    "caseIndex": case_index,
                    "donorLength": donor_length,
                    "greedyTokenExact": greedy_token_exact,
                    "inputSha256": [_tensor_sha256(value) for value in case_inputs],
                    "outputs": output_evidence,
                    "topk": validate_qat_centroid_scores(
                        masked_embedding.centroids(head_inputs[0])
                    ),
                }
            )

    if cases[0]["inputSha256"] != cases[-1]["inputSha256"]:
        raise ValueError("assistant validation replay input mismatch")
    if cases[0]["topk"] != cases[-1]["topk"]:
        raise ValueError("assistant validation replay top-k mismatch")
    if cases[0]["outputs"] != cases[-1]["outputs"]:
        raise ValueError("assistant validation replay output mismatch")
    evidence: dict[str, Any] = {
        "cases": cases,
        "donorSequence": list(donor_sequence),
        "selectionContract": {
            "centroidTopK": OFFICIAL_QAT_CENTROID_TOP_K,
            "numCentroids": OFFICIAL_QAT_NUM_CENTROIDS,
            "selectedTokenCount": OFFICIAL_QAT_SELECTED_TOKEN_COUNT,
            "tokensPerCentroid": OFFICIAL_QAT_TOKENS_PER_CENTROID,
        },
    }
    if reference_outputs is not None:
        evidence["eagerEquivalence"] = {
            "allClose": True,
            "atol": 1e-4,
            "rtol": 1e-3,
        }
    if token_ordering_evidence is not None:
        evidence["tokenOrdering"] = token_ordering_evidence
    return evidence


def load_qat_assistant(
    checkpoint: Path,
    *,
    max_donor_len: int = 8960,
    lm_head_bits: int = 4,
    quantize_backbone: str = "8da4w",
) -> UnfoldedAssistant:
    if max_donor_len < 2:
        raise ValueError("assistant donor capacity must be at least 2")
    if lm_head_bits != 4:
        raise ValueError("official QAT assistant requires a 4-bit LM head")
    if quantize_backbone != "8da4w":
        raise ValueError("official QAT assistant requires 8da4w backbone quantization")
    validate_assistant_checkpoint(checkpoint)
    raw_token_ordering, raw_token_ordering_evidence = _load_raw_token_ordering(
        checkpoint
    )

    import importlib

    transformers_module: Any = importlib.import_module("transformers")
    assistant_type: Any = transformers_module.Gemma4AssistantForCausalLM
    assistant: Any = assistant_type.from_pretrained(
        str(checkpoint.resolve()),
        torch_dtype=torch.float32,
        trust_remote_code=False,
    ).eval()
    masked_embedding = assistant.masked_embedding
    if masked_embedding is None:
        raise ValueError("QAT assistant is missing its masked embedding")
    if (
        masked_embedding.num_centroids != OFFICIAL_QAT_NUM_CENTROIDS
        or masked_embedding.vocab_size_per_centroid != OFFICIAL_QAT_TOKENS_PER_CENTROID
        or masked_embedding.vocab_size
        != OFFICIAL_QAT_NUM_CENTROIDS * OFFICIAL_QAT_TOKENS_PER_CENTROID
    ):
        raise ValueError("QAT assistant selection dimensions mismatch")
    masked_embedding.centroid_intermediate_top_k = OFFICIAL_QAT_CENTROID_TOP_K
    loaded_token_ordering = masked_embedding.token_ordering.detach().cpu().contiguous()
    if loaded_token_ordering.dtype != torch.int64 or tuple(
        loaded_token_ordering.shape
    ) != (262144,):
        raise ValueError(
            "assistant loaded token ordering must be int64 with shape [262144]"
        )
    loaded_token_ordering_evidence = validate_qat_token_ordering(loaded_token_ordering)
    raw_sha256 = _tensor_sha256(raw_token_ordering)
    loaded_sha256 = _tensor_sha256(loaded_token_ordering)
    if (
        raw_sha256 != loaded_sha256
        or raw_sha256 != raw_token_ordering_evidence["sha256"]
        or loaded_sha256 != loaded_token_ordering_evidence["sha256"]
        or not torch.equal(raw_token_ordering, loaded_token_ordering)
    ):
        raise ValueError("assistant raw/loaded token ordering differs")
    token_ordering_evidence = dict(loaded_token_ordering_evidence)
    token_ordering_evidence.update(
        {
            "loaded": loaded_token_ordering_evidence,
            "raw": raw_token_ordering_evidence,
            "rawLoadedByteExact": True,
            "rawSha256": raw_sha256,
        }
    )
    masked_embedding._lm_embed = _quantized_embedding(
        assistant.lm_head.weight.detach(), lm_head_bits
    )

    from executorch.extension.llm.export.quantize import quantize_model_

    quantize_model_(
        assistant.model,
        qlinear_config=quantize_backbone,
        qlinear_group_size=128,
        skip_incompatible_shapes=True,
    )
    _, validation_inputs = _qat_validation_cases(assistant.config, max_donor_len)
    reference_outputs = _capture_reference_outputs(
        UnfoldedAssistant(assistant).eval(), validation_inputs
    )
    adapt_assistant_model_for_webgpu(assistant, max_seq_len=max_donor_len)
    wrapper = UnfoldedAssistant(assistant).eval()
    wrapper._webgpu_qat_selection_evidence = validate_qat_selection_contract(
        wrapper,
        max_donor_len,
        validation_inputs=validation_inputs,
        reference_outputs=reference_outputs,
        token_ordering_evidence=token_ordering_evidence,
    )
    return wrapper


def export_assistant_program(
    checkpoint: Path,
    *,
    max_donor_len: int = 8960,
    lm_head_bits: int = 4,
    quantize_backbone: str = "8da4w",
) -> torch.export.ExportedProgram:
    if max_donor_len < 2:
        raise ValueError("assistant donor capacity must be at least 2")
    wrapper = load_qat_assistant(
        checkpoint,
        max_donor_len=max_donor_len,
        lm_head_bits=lm_head_bits,
        quantize_backbone=quantize_backbone,
    )
    config: Any = wrapper.assistant.config
    text_config: Any = config.get_text_config()
    donor_len = 2
    example = (
        torch.randn(1, 1, 2 * config.backbone_hidden_size),
        torch.tensor([[donor_len - 1]], dtype=torch.long),
        torch.randn(1, 1, donor_len, text_config.global_head_dim),
        torch.randn(1, 1, donor_len, text_config.global_head_dim),
        torch.randn(1, 1, donor_len, text_config.head_dim),
        torch.randn(1, 1, donor_len, text_config.head_dim),
    )
    dynamic_donor = torch.export.Dim("assistant_donor_len", min=2, max=max_donor_len)
    with torch.no_grad():
        return torch.export.export(
            wrapper,
            example,
            dynamic_shapes={
                "inputs_embeds": None,
                "position_ids": None,
                "full_k": {2: dynamic_donor},
                "full_v": {2: dynamic_donor},
                "sliding_k": {2: dynamic_donor},
                "sliding_v": {2: dynamic_donor},
            },
            strict=False,
        )


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Validate and export the Gemma 4 QAT assistant graph"
    )
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--max-donor-len", type=int, default=8960)
    args = parser.parse_args()
    if args.output.suffix != ".pt2":
        raise ValueError("assistant graph output path must end in .pt2")
    if args.output.exists():
        raise ValueError(f"refusing to overwrite existing artifact: {args.output}")
    program = export_assistant_program(
        args.checkpoint, max_donor_len=args.max_donor_len
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(
        prefix=f".{args.output.stem}.", dir=args.output.parent
    ) as staging_directory:
        staged_output = Path(staging_directory) / args.output.name
        torch.export.save(program, staged_output)
        staged_output.replace(args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
