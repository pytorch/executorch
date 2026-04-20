"""Layer-by-layer numerical parity tests for ExecuTorch Gemma4 vs HF reference.

Goal: pinpoint where ET diverges from `transformers.models.gemma4` so the
runtime garbage output (vs HF's correct generation) can be fixed.

Both stacks are loaded as FP32. Same converted weights are used for both
(HF native checkpoint for HF, our `model_et.pth` for ET).

Run:
    pytest examples/models/gemma4/tests/test_parity.py -v -x

Or, no-pytest fallback:
    python examples/models/gemma4/tests/test_parity.py
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import torch

HF_MODEL_DIR = os.environ.get(
    "GEMMA4_HF_DIR", str(Path.home() / "models/gemma-4-E2B-it")
)
ET_CHECKPOINT = os.environ.get(
    "GEMMA4_ET_CHECKPOINT", str(Path(HF_MODEL_DIR) / "model_et.pth")
)
ET_PARAMS = str(Path(__file__).resolve().parents[1] / "config" / "e2b_config.json")

PROMPT = "<|turn>user\nWhat is the capital of France?<turn|>\n<|turn>model\n"


# ---------------------------------------------------------------------------
# Shared loaders (cached per-process)
# ---------------------------------------------------------------------------

_HF_MODEL = None
_HF_CONFIG = None
_HF_TOKENIZER = None
_ET_MODEL = None


def hf_model():
    global _HF_MODEL, _HF_CONFIG
    if _HF_MODEL is None:
        from transformers import AutoModelForCausalLM, AutoConfig

        _HF_CONFIG = AutoConfig.from_pretrained(HF_MODEL_DIR, trust_remote_code=True)
        _HF_MODEL = AutoModelForCausalLM.from_pretrained(
            HF_MODEL_DIR, dtype=torch.float32, trust_remote_code=True
        ).eval()
    return _HF_MODEL


def hf_text_model():
    """The Gemma4TextModel (inside the multimodal wrapper)."""
    return hf_model().model.language_model


def hf_text_config():
    return _HF_CONFIG.text_config if hasattr(_HF_CONFIG, "text_config") else _HF_CONFIG


def hf_tokenizer():
    global _HF_TOKENIZER
    if _HF_TOKENIZER is None:
        from transformers import AutoTokenizer

        _HF_TOKENIZER = AutoTokenizer.from_pretrained(HF_MODEL_DIR)
    return _HF_TOKENIZER


def et_model():
    global _ET_MODEL
    if _ET_MODEL is None:
        from executorch.examples.models.gemma4 import Gemma4Model
        from executorch.extension.llm.export.config.llm_config import LlmConfig

        cfg = LlmConfig()
        cfg.base.params = ET_PARAMS
        cfg.base.checkpoint = ET_CHECKPOINT
        cfg.model.use_kv_cache = False
        cfg.export.max_seq_length = 64
        cfg.export.max_context_length = 64
        m = Gemma4Model(llm_config=cfg).get_eager_model().eval().float()
        _ET_MODEL = m
    return _ET_MODEL


def encode_prompt():
    tok = hf_tokenizer()
    return tok(PROMPT, return_tensors="pt", add_special_tokens=False).input_ids


# ---------------------------------------------------------------------------
# Diff helpers
# ---------------------------------------------------------------------------


def diff_summary(a: torch.Tensor, b: torch.Tensor, name: str) -> dict:
    a, b = a.float(), b.float()
    if a.shape != b.shape:
        return {"name": name, "shape_a": list(a.shape), "shape_b": list(b.shape), "ok": False}
    abs_diff = (a - b).abs()
    cos = torch.nn.functional.cosine_similarity(
        a.flatten().unsqueeze(0), b.flatten().unsqueeze(0)
    ).item()
    info = {
        "name": name,
        "shape": list(a.shape),
        "max_abs_diff": abs_diff.max().item(),
        "mean_abs_diff": abs_diff.mean().item(),
        "cos_sim": cos,
    }
    return info


def report(info: dict, atol: float = 1e-3, cos_min: float = 0.999):
    ok = info["max_abs_diff"] <= atol and info.get("cos_sim", 0) >= cos_min
    flag = "PASS" if ok else "FAIL"
    print(
        f"  [{flag}] {info['name']:30s} max_diff={info['max_abs_diff']:.4e} "
        f"mean_diff={info['mean_abs_diff']:.4e} cos={info['cos_sim']:.6f}"
    )
    return ok


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_token_embedding_parity():
    print("\n=== test_token_embedding_parity ===")
    ids = encode_prompt()
    print(f"  prompt ids: {ids[0].tolist()[:16]}")

    # HF
    hf_text = hf_text_model()
    with torch.no_grad():
        hf_emb = hf_text.embed_tokens(ids)
    print(f"  HF embed shape: {hf_emb.shape}, mean={hf_emb.mean():.4f}")

    # ET — must apply embedding_scale_factor to match HF Gemma4TextScaledWordEmbedding
    em = et_model()
    with torch.no_grad():
        et_emb = em.tok_embeddings(ids) * em.params.embedding_scale_factor
    print(f"  ET embed shape: {et_emb.shape}, mean={et_emb.mean():.4f}")

    info = diff_summary(hf_emb, et_emb, "token_embedding")
    assert report(info, atol=1e-3, cos_min=0.99999), f"Embedding mismatch: {info}"


def test_rmsnorm_parity():
    print("\n=== test_rmsnorm_parity ===")
    from executorch.examples.models.llama.norm import RMSNorm
    from transformers.models.gemma4.modeling_gemma4 import Gemma4RMSNorm

    dim = 1536
    torch.manual_seed(0)
    weights = torch.randn(dim) * 0.1

    hf_norm = Gemma4RMSNorm(dim, eps=1e-6, with_scale=True)
    hf_norm.weight.data.copy_(weights)
    hf_norm.eval()

    et_norm = RMSNorm(dim, eps=1e-6, add_unit_offset=False).float()
    et_norm.weight.data.copy_(weights)
    et_norm.eval()

    x = torch.randn(2, 8, dim) * 5.0
    with torch.no_grad():
        hf_out = hf_norm(x)
        et_out = et_norm(x)

    info = diff_summary(hf_out, et_out, "rmsnorm")
    assert report(info, atol=1e-5, cos_min=0.99999999), f"RMSNorm mismatch: {info}"


def test_pli_inputs_parity():
    print("\n=== test_pli_inputs_parity ===")
    ids = encode_prompt()
    hf_text = hf_text_model()
    em = et_model()

    with torch.no_grad():
        hf_emb = hf_text.embed_tokens(ids)
        hf_pli = hf_text.get_per_layer_inputs(ids, hf_emb)
        hf_pli = hf_text.project_per_layer_inputs(hf_emb, hf_pli)
        # shape: (B, T, n_layers, hidden_per_layer_input)
    print(f"  HF PLI shape: {hf_pli.shape}, mean={hf_pli.mean():.4f}")

    # ET: replicate Transformer.forward PLI computation
    with torch.no_grad():
        et_emb = em.tok_embeddings(ids) * em.params.embedding_scale_factor  # h
        pli_emb = em.pli_embeddings(ids) * em.pli_embed_scale
        pli_proj = em.pli_projection(et_emb) * em.pli_projection_scale
        pli_proj = pli_proj.view(
            *et_emb.shape[:-1], em.params.n_layers, em.hidden_size_per_layer_input
        )
        pli_proj = em.pli_norm(pli_proj)
        pli_emb = pli_emb.view(
            *et_emb.shape[:-1], em.params.n_layers, em.hidden_size_per_layer_input
        )
        et_pli = (pli_proj + pli_emb) * em.pli_combine_scale
    print(f"  ET PLI shape: {et_pli.shape}, mean={et_pli.mean():.4f}")

    info = diff_summary(hf_pli, et_pli, "pli_inputs")
    assert report(info, atol=1e-3, cos_min=0.99999), f"PLI mismatch: {info}"


def test_rope_parity():
    print("\n=== test_rope_parity ===")
    from transformers.models.gemma4.modeling_gemma4 import Gemma4TextRotaryEmbedding

    cfg = hf_text_config()
    hf_rope = Gemma4TextRotaryEmbedding(cfg).eval()
    em = et_model()
    et_rope = em.rope

    pos = torch.arange(16).unsqueeze(0)  # (1, 16)
    overall_pass = True

    for layer_type in ("sliding_attention", "full_attention"):
        # HF expects dummy x (only used for dtype/device)
        # head_dim depends on layer_type for full_attention
        head_dim = (
            cfg.global_head_dim
            if layer_type == "full_attention"
            else cfg.head_dim
        )
        x = torch.zeros(1, 1, head_dim)
        with torch.no_grad():
            hf_cos, hf_sin = hf_rope(x, pos, layer_type=layer_type)
        # hf_cos shape: (1, 16, head_dim)

        # ET: try to retrieve per-layer-type freqs.
        if hasattr(et_rope, "get_freqs_for_layer_type"):
            with torch.no_grad():
                et_cos, et_sin = et_rope.get_freqs_for_layer_type(
                    pos[0], 16, layer_type=layer_type
                )
        else:
            # use_kv_cache=False path doesn't accept input_pos.
            with torch.no_grad():
                et_cos, et_sin = et_rope.get_freqs(None, 16)

        # Both produce (..., head_dim) (concatenated freqs+freqs).
        # Slice/reshape to align.
        if et_cos.dim() == 2:  # (T, head_dim)
            et_cos = et_cos.unsqueeze(0)
            et_sin = et_sin.unsqueeze(0)

        info_cos = diff_summary(hf_cos, et_cos, f"rope[{layer_type}].cos")
        info_sin = diff_summary(hf_sin, et_sin, f"rope[{layer_type}].sin")
        if not report(info_cos, atol=1e-5, cos_min=0.999999):
            overall_pass = False
        if not report(info_sin, atol=1e-5, cos_min=0.999999):
            overall_pass = False

    assert overall_pass, "RoPE parity FAILED — see report above. This blocks decode."


def test_decoder_layer_parity():
    print("\n=== test_decoder_layer_parity ===")
    ids = encode_prompt()

    hf_text = hf_text_model()
    em = et_model()

    with torch.no_grad():
        # Run HF model to layer 1 with output_hidden_states to get inputs.
        hf_out = hf_text(input_ids=ids, output_hidden_states=True, use_cache=False)
        hf_layer0_in = hf_out.hidden_states[0]  # = embedding output
        hf_layer1_out = hf_out.hidden_states[1]  # = after layer 0

    # ET: run only layer 0 from the same input
    with torch.no_grad():
        # Build attn_options like Transformer.forward does
        # (Need PLI inputs for layer 0)
        et_emb = em.tok_embeddings(ids) * em.params.embedding_scale_factor
        pli_emb = em.pli_embeddings(ids) * em.pli_embed_scale
        pli_proj = em.pli_projection(et_emb) * em.pli_projection_scale
        pli_proj = pli_proj.view(
            *et_emb.shape[:-1], em.params.n_layers, em.hidden_size_per_layer_input
        )
        pli_proj = em.pli_norm(pli_proj)
        pli_emb = pli_emb.view(
            *et_emb.shape[:-1], em.params.n_layers, em.hidden_size_per_layer_input
        )
        et_pli = (pli_proj + pli_emb) * em.pli_combine_scale

        seqlen = ids.shape[1]
        freqs_cos, freqs_sin = em.rope.get_freqs(None, seqlen)

        attn_options = {
            "input_pos": None,
            "per_layer_input": et_pli[:, :, 0, :],
        }
        et_layer1_out, _ = em.layers[0](
            hf_layer0_in, freqs_cos, freqs_sin, attn_options
        )

    info = diff_summary(hf_layer1_out, et_layer1_out, "decoder_layer_0")
    assert report(info, atol=1e-2, cos_min=0.999), f"Layer 0 mismatch: {info}"


def test_full_forward_parity():
    print("\n=== test_full_forward_parity ===")
    ids = encode_prompt()

    hf_m = hf_model()
    em = et_model()

    with torch.no_grad():
        hf_logits = hf_m(input_ids=ids, use_cache=False).logits[0, -1, :]
        et_out = em(tokens=ids)
        # ET model returns (B, T, V) or (B, V) depending on generate_full_logits
        if et_out.dim() == 3:
            et_logits = et_out[0, -1, :]
        elif et_out.dim() == 2:
            et_logits = et_out[0]
        else:
            et_logits = et_out

    hf_top = torch.topk(hf_logits, 5)
    et_top = torch.topk(et_logits, 5)
    tok = hf_tokenizer()

    print("  HF top-5:")
    for tid, sc in zip(hf_top.indices.tolist(), hf_top.values.tolist()):
        print(f"    {tid:6d} ({tok.decode([tid])!r:25s}) {sc:.3f}")
    print("  ET top-5:")
    for tid, sc in zip(et_top.indices.tolist(), et_top.values.tolist()):
        print(f"    {tid:6d} ({tok.decode([tid])!r:25s}) {sc:.3f}")

    info = diff_summary(hf_logits, et_logits, "final_logits")
    report(info, atol=0.1, cos_min=0.99)
    # The strong assertion: top-1 token id matches.
    assert hf_top.indices[0].item() == et_top.indices[0].item(), (
        f"Top-1 token mismatch: HF={hf_top.indices[0].item()} ET={et_top.indices[0].item()}"
    )


# ---------------------------------------------------------------------------
# Standalone runner (no pytest)
# ---------------------------------------------------------------------------


def main():
    tests = [
        test_token_embedding_parity,
        test_rmsnorm_parity,
        test_pli_inputs_parity,
        test_rope_parity,
        test_decoder_layer_parity,
        test_full_forward_parity,
    ]
    fails = []
    for t in tests:
        try:
            t()
        except AssertionError as e:
            print(f"  [FAIL] {t.__name__}: {e}")
            fails.append(t.__name__)
        except Exception as e:
            import traceback

            print(f"  [ERROR] {t.__name__}: {e}")
            traceback.print_exc()
            fails.append(t.__name__)
    print()
    print("=" * 60)
    if fails:
        print(f"FAILED: {fails}")
        sys.exit(1)
    print("ALL TESTS PASSED")


if __name__ == "__main__":
    main()
