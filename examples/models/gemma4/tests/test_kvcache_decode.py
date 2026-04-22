"""Probe KV-cache prefill/decode vs non-cached forward on the prepared LLM model.

Goal: determine whether the multimodal generation degeneration originates in the
Python model with KV cache, or only in the export/runtime layer.
"""
from __future__ import annotations
import os, sys, torch
from pathlib import Path

HF_DIR = os.environ.get("GEMMA4_HF_DIR", str(Path.home() / "models/gemma-4-E2B-it"))
ET_CKPT = os.environ.get("GEMMA4_ET_CHECKPOINT", str(Path(HF_DIR) / "model_et.pth"))
ET_PARAMS = str(Path(__file__).resolve().parents[1] / "config" / "e2b_config.json")

PROMPT = "<|turn>user\nWhat is the capital of France?<turn|>\n<|turn>model\n"


def build_kv_model(max_seq=64):
    from executorch.examples.models.llama.export_llama_lib import _prepare_for_llama_export
    from executorch.extension.llm.export.config.llm_config import LlmConfig, ModelType
    cfg = LlmConfig()
    cfg.base.model_class = ModelType.gemma4
    cfg.base.params = ET_PARAMS
    cfg.base.checkpoint = ET_CKPT
    cfg.model.use_kv_cache = True
    cfg.model.use_sdpa_with_kv_cache = True
    cfg.model.enable_dynamic_shape = True
    cfg.export.max_seq_length = max_seq
    cfg.export.max_context_length = max_seq
    builder = _prepare_for_llama_export(cfg)
    return builder.model.eval()


def build_nocache_model(max_seq=64):
    from executorch.examples.models.gemma4 import Gemma4Model
    from executorch.extension.llm.export.config.llm_config import LlmConfig
    cfg = LlmConfig()
    cfg.base.params = ET_PARAMS
    cfg.base.checkpoint = ET_CKPT
    cfg.model.use_kv_cache = False
    cfg.export.max_seq_length = max_seq
    cfg.export.max_context_length = max_seq
    return Gemma4Model(llm_config=cfg).get_eager_model().eval().float()


def main():
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(HF_DIR)
    ids = tok(PROMPT, return_tensors="pt", add_special_tokens=False).input_ids
    S = ids.shape[1]
    print(f"prompt has {S} tokens: {ids[0].tolist()}")

    print("Building no-cache model...")
    nm = build_nocache_model()
    with torch.no_grad():
        out_no = nm(tokens=ids)
        if out_no.dim() == 3:
            ref_logits = out_no[0, -1, :]
        else:
            ref_logits = out_no[0]
    ref_top = torch.topk(ref_logits, 5)
    print("no-cache top-5:")
    for tid, sc in zip(ref_top.indices.tolist(), ref_top.values.tolist()):
        print(f"  {tid:6d} {tok.decode([tid])!r:20s} {sc:.3f}")

    print("\nBuilding KV-cache model...")
    km = build_kv_model()
    # Test 1: batched prefill via tokens=
    print("\n=== Test 1: batched prefill via tokens, input_pos=[0] ===")
    with torch.no_grad():
        attn_opts = {"input_pos": torch.tensor([0], dtype=torch.long)}
        kv_out = km(tokens=ids, attn_options=attn_opts)
    if isinstance(kv_out, tuple):
        kv_out = kv_out[0]
    if kv_out.dim() == 3:
        kv_logits = kv_out[0, -1, :]
    else:
        kv_logits = kv_out[0]
    kv_top = torch.topk(kv_logits, 5)
    print("kv batched top-5:")
    for tid, sc in zip(kv_top.indices.tolist(), kv_top.values.tolist()):
        print(f"  {tid:6d} {tok.decode([tid])!r:20s} {sc:.3f}")
    print(f"top-1 match no-cache: {ref_top.indices[0].item() == kv_top.indices[0].item()}")
    print(f"max_abs_diff={float((ref_logits.float() - kv_logits.float()).abs().max()):.4e}")

    # Test 2: token-by-token prefill (rebuild km to reset KV cache)
    print("\n=== Test 2: token-by-token prefill (sequential) ===")
    km2 = build_kv_model()
    with torch.no_grad():
        last_logits = None
        for i in range(S):
            t = ids[:, i:i+1]
            attn_opts = {"input_pos": torch.tensor([i], dtype=torch.long)}
            o = km2(tokens=t, attn_options=attn_opts)
            if isinstance(o, tuple): o = o[0]
            last_logits = o[0, -1, :] if o.dim() == 3 else o[0]
    seq_top = torch.topk(last_logits, 5)
    print("kv sequential top-5:")
    for tid, sc in zip(seq_top.indices.tolist(), seq_top.values.tolist()):
        print(f"  {tid:6d} {tok.decode([tid])!r:20s} {sc:.3f}")
    print(f"top-1 match no-cache: {ref_top.indices[0].item() == seq_top.indices[0].item()}")
    print(f"max_abs_diff={float((ref_logits.float() - last_logits.float()).abs().max()):.4e}")

    # Test 3: Generate next 10 tokens with KV cache to see if it degenerates
    print("\n=== Test 3: greedy decode 10 tokens after batched prefill ===")
    km3 = build_kv_model(max_seq=128)
    with torch.no_grad():
        attn_opts = {"input_pos": torch.tensor([0], dtype=torch.long)}
        o = km3(tokens=ids, attn_options=attn_opts)
        if isinstance(o, tuple): o = o[0]
        nxt_logits = o[0, -1, :] if o.dim() == 3 else o[0]
        nxt = int(torch.argmax(nxt_logits).item())
        gen = [nxt]
        pos = S
        for _ in range(15):
            t = torch.tensor([[nxt]], dtype=torch.long)
            attn_opts = {"input_pos": torch.tensor([pos], dtype=torch.long)}
            o = km3(tokens=t, attn_options=attn_opts)
            if isinstance(o, tuple): o = o[0]
            nxt_logits = o[0, -1, :] if o.dim() == 3 else o[0]
            nxt = int(torch.argmax(nxt_logits).item())
            gen.append(nxt)
            pos += 1
    print("generated:", repr(tok.decode(gen)))


if __name__ == "__main__":
    main()
